//! Dormant private admission and drain substrate for ADR-088.
//!
//! This slice accounts active preparation/encode guards and their byte leases. Retained-lease
//! transfer into published objects, preparation-only terminal faults, residue debt, and aggregate
//! cleanup errors remain outside this substrate and must land before prepared services use it.

use super::NativeResourceBudget;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResourceAdmissionError {
    RequestExceedsBudget,
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResourceLifecycleError {
    DrainNotStarted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdmissionClass {
    Preparation,
    Encode,
}

impl AdmissionClass {
    fn other(self) -> Self {
        match self {
            Self::Preparation => Self::Encode,
            Self::Encode => Self::Preparation,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdmissionCharge {
    Preparation {
        retained_bytes: u64,
        work_bytes: u64,
    },
    Encode {
        work_bytes: u64,
    },
}

impl AdmissionCharge {
    fn class(self) -> AdmissionClass {
        match self {
            Self::Preparation { .. } => AdmissionClass::Preparation,
            Self::Encode { .. } => AdmissionClass::Encode,
        }
    }
}

struct AdmissionWaiter {
    charge: AdmissionCharge,
    waker: Waker,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdmissionWaiterSlot {
    class: AdmissionClass,
    ticket: u64,
}

struct ReleaseWaiter {
    waker: Waker,
}

fn trim_vacant_tail<T>(queue: &mut Vec<Option<T>>) {
    while queue.last().is_some_and(Option::is_none) {
        queue.pop();
    }
    if queue.is_empty() {
        *queue = Vec::new();
    } else if queue.len() < queue.capacity() / 2 {
        queue.shrink_to_fit();
    }
}

struct ResourceState {
    max_preparations: usize,
    max_encodes: usize,
    max_retained_bytes: u64,
    max_work_bytes: u64,
    active_preparations: usize,
    active_encodes: usize,
    retained_bytes: u64,
    work_bytes: u64,
    closed: bool,
    admission_close_notified: bool,
    next_admission_class: AdmissionClass,
    next_waiter_ticket: u64,
    entitled_waiter: Option<AdmissionWaiterSlot>,
    preparation_waiters: BTreeMap<u64, AdmissionWaiter>,
    encode_waiters: BTreeMap<u64, AdmissionWaiter>,
    release_waiters: Vec<Option<ReleaseWaiter>>,
}

impl ResourceState {
    fn new(budget: NativeResourceBudget) -> Self {
        Self {
            max_preparations: budget.max_concurrent_preparations().get(),
            max_encodes: budget.max_concurrent_encodes().get(),
            max_retained_bytes: budget.max_retained_bytes().get(),
            max_work_bytes: budget.max_transient_work_bytes().get(),
            active_preparations: 0,
            active_encodes: 0,
            retained_bytes: 0,
            work_bytes: 0,
            closed: false,
            admission_close_notified: false,
            next_admission_class: AdmissionClass::Preparation,
            next_waiter_ticket: 0,
            entitled_waiter: None,
            preparation_waiters: BTreeMap::new(),
            encode_waiters: BTreeMap::new(),
            release_waiters: Vec::new(),
        }
    }

    fn request_exceeds_budget(&self, charge: AdmissionCharge) -> bool {
        match charge {
            AdmissionCharge::Preparation {
                retained_bytes,
                work_bytes,
            } => retained_bytes > self.max_retained_bytes || work_bytes > self.max_work_bytes,
            AdmissionCharge::Encode { work_bytes } => work_bytes > self.max_work_bytes,
        }
    }

    fn accounting_valid(&self) -> bool {
        self.active_preparations <= self.max_preparations
            && self.active_encodes <= self.max_encodes
            && self.retained_bytes <= self.max_retained_bytes
            && self.work_bytes <= self.max_work_bytes
    }

    fn can_admit(&self, charge: AdmissionCharge) -> bool {
        if self.closed || !self.accounting_valid() || self.request_exceeds_budget(charge) {
            return false;
        }

        let Some(available_work) = self.max_work_bytes.checked_sub(self.work_bytes) else {
            return false;
        };
        match charge {
            AdmissionCharge::Preparation {
                retained_bytes,
                work_bytes,
            } => self
                .max_retained_bytes
                .checked_sub(self.retained_bytes)
                .is_some_and(|available_retained| {
                    self.active_preparations < self.max_preparations
                        && retained_bytes <= available_retained
                        && work_bytes <= available_work
                }),
            AdmissionCharge::Encode { work_bytes } => {
                self.active_encodes < self.max_encodes && work_bytes <= available_work
            }
        }
    }

    fn reserve(&mut self, charge: AdmissionCharge) -> bool {
        match charge {
            AdmissionCharge::Preparation {
                retained_bytes,
                work_bytes,
            } => {
                let Some(active_preparations) = self.active_preparations.checked_add(1) else {
                    return false;
                };
                let Some(retained_bytes) = self.retained_bytes.checked_add(retained_bytes) else {
                    return false;
                };
                let Some(work_bytes) = self.work_bytes.checked_add(work_bytes) else {
                    return false;
                };
                if active_preparations > self.max_preparations
                    || retained_bytes > self.max_retained_bytes
                    || work_bytes > self.max_work_bytes
                {
                    return false;
                }
                self.active_preparations = active_preparations;
                self.retained_bytes = retained_bytes;
                self.work_bytes = work_bytes;
            }
            AdmissionCharge::Encode { work_bytes } => {
                let Some(active_encodes) = self.active_encodes.checked_add(1) else {
                    return false;
                };
                let Some(work_bytes) = self.work_bytes.checked_add(work_bytes) else {
                    return false;
                };
                if active_encodes > self.max_encodes || work_bytes > self.max_work_bytes {
                    return false;
                }
                self.active_encodes = active_encodes;
                self.work_bytes = work_bytes;
            }
        }
        true
    }

    fn release(&mut self, charge: AdmissionCharge) -> bool {
        match charge {
            AdmissionCharge::Preparation {
                retained_bytes,
                work_bytes,
            } => {
                let Some(active_preparations) = self.active_preparations.checked_sub(1) else {
                    return false;
                };
                let Some(retained_bytes) = self.retained_bytes.checked_sub(retained_bytes) else {
                    return false;
                };
                let Some(work_bytes) = self.work_bytes.checked_sub(work_bytes) else {
                    return false;
                };
                self.active_preparations = active_preparations;
                self.retained_bytes = retained_bytes;
                self.work_bytes = work_bytes;
            }
            AdmissionCharge::Encode { work_bytes } => {
                let Some(active_encodes) = self.active_encodes.checked_sub(1) else {
                    return false;
                };
                let Some(work_bytes) = self.work_bytes.checked_sub(work_bytes) else {
                    return false;
                };
                self.active_encodes = active_encodes;
                self.work_bytes = work_bytes;
            }
        }
        true
    }

    fn resources_released(&self) -> bool {
        self.active_preparations == 0
            && self.active_encodes == 0
            && self.retained_bytes == 0
            && self.work_bytes == 0
    }

    fn admission_waiters(&self, class: AdmissionClass) -> &BTreeMap<u64, AdmissionWaiter> {
        match class {
            AdmissionClass::Preparation => &self.preparation_waiters,
            AdmissionClass::Encode => &self.encode_waiters,
        }
    }

    fn admission_waiters_mut(
        &mut self,
        class: AdmissionClass,
    ) -> &mut BTreeMap<u64, AdmissionWaiter> {
        match class {
            AdmissionClass::Preparation => &mut self.preparation_waiters,
            AdmissionClass::Encode => &mut self.encode_waiters,
        }
    }

    fn admission_waiters_empty(&self) -> bool {
        self.preparation_waiters.is_empty() && self.encode_waiters.is_empty()
    }

    fn register_admission_waiter(
        &mut self,
        slot: &mut Option<AdmissionWaiterSlot>,
        charge: AdmissionCharge,
        waker: &Waker,
    ) -> bool {
        let class = charge.class();
        if let Some(existing) = *slot {
            let queue = self.admission_waiters_mut(existing.class);
            if let Some(waiter) = queue.get_mut(&existing.ticket) {
                waiter.charge = charge;
                if !waiter.waker.will_wake(waker) {
                    waiter.waker = waker.clone();
                }
                return true;
            }
        }

        let ticket = self.next_waiter_ticket;
        let Some(next_ticket) = ticket.checked_add(1) else {
            return false;
        };
        self.next_waiter_ticket = next_ticket;
        self.admission_waiters_mut(class).insert(
            ticket,
            AdmissionWaiter {
                charge,
                waker: waker.clone(),
            },
        );
        *slot = Some(AdmissionWaiterSlot { class, ticket });
        true
    }

    fn remove_admission_waiter(&mut self, slot: Option<AdmissionWaiterSlot>) {
        let Some(slot) = slot else {
            return;
        };
        self.admission_waiters_mut(slot.class).remove(&slot.ticket);
        if self.entitled_waiter == Some(slot) {
            self.entitled_waiter = None;
        }
        if self.admission_waiters_empty() {
            self.next_waiter_ticket = 0;
        }
    }

    fn eligible_class_head(&self, class: AdmissionClass) -> Option<(AdmissionWaiterSlot, Waker)> {
        let (&ticket, waiter) = self.admission_waiters(class).first_key_value()?;
        self.can_admit(waiter.charge)
            .then(|| (AdmissionWaiterSlot { class, ticket }, waiter.waker.clone()))
    }

    fn select_admission(&mut self) -> Option<(AdmissionWaiterSlot, Waker)> {
        if self.closed || self.entitled_waiter.is_some() {
            return None;
        }

        let preferred = self.next_admission_class;
        let selected = self
            .eligible_class_head(preferred)
            .or_else(|| self.eligible_class_head(preferred.other()))?;
        self.entitled_waiter = Some(selected.0);
        self.next_admission_class = selected.0.class.other();
        Some(selected)
    }

    fn close_admission(&mut self) -> Vec<Waker> {
        self.closed = true;
        self.entitled_waiter = None;
        if self.admission_close_notified {
            return Vec::new();
        }
        self.admission_close_notified = true;
        self.preparation_waiters
            .values()
            .chain(self.encode_waiters.values())
            .map(|waiter| waiter.waker.clone())
            .collect()
    }

    fn register_release_waiter(&mut self, slot: &mut Option<usize>, waker: &Waker) {
        if let Some(index) = *slot
            && let Some(Some(waiter)) = self.release_waiters.get_mut(index)
        {
            if !waiter.waker.will_wake(waker) {
                waiter.waker = waker.clone();
            }
            return;
        }

        if let Some((index, vacant)) = self
            .release_waiters
            .iter_mut()
            .enumerate()
            .find(|(_, entry)| entry.is_none())
        {
            *vacant = Some(ReleaseWaiter {
                waker: waker.clone(),
            });
            *slot = Some(index);
            return;
        }

        let index = self.release_waiters.len();
        self.release_waiters.push(Some(ReleaseWaiter {
            waker: waker.clone(),
        }));
        *slot = Some(index);
    }

    fn remove_release_waiter(&mut self, slot: Option<usize>) {
        let Some(index) = slot else {
            return;
        };
        if let Some(entry) = self.release_waiters.get_mut(index) {
            *entry = None;
        }
        trim_vacant_tail(&mut self.release_waiters);
    }

    fn completed_release_wakers(&self) -> Vec<Waker> {
        if !self.closed || !self.resources_released() {
            return Vec::new();
        }
        self.release_waiters
            .iter()
            .filter_map(Option::as_ref)
            .map(|waiter| waiter.waker.clone())
            .collect()
    }
}

struct ResourceSupervisorInner {
    state: Mutex<ResourceState>,
}

impl ResourceSupervisorInner {
    fn lock_state(&self) -> MutexGuard<'_, ResourceState> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn release(&self, charge: AdmissionCharge) {
        let (admission_wakers, release_wakers) = {
            let mut state = self.lock_state();
            let admission_wakers = if state.release(charge) {
                state
                    .select_admission()
                    .map(|(_, waker)| waker)
                    .into_iter()
                    .collect()
            } else {
                state.close_admission()
            };
            (admission_wakers, state.completed_release_wakers())
        };
        wake_all(admission_wakers);
        wake_all(release_wakers);
    }
}

fn wake_all(wakers: Vec<Waker>) {
    for waker in wakers {
        waker.wake();
    }
}

#[derive(Clone)]
pub(super) struct ResourceSupervisor {
    inner: Arc<ResourceSupervisorInner>,
}

impl ResourceSupervisor {
    pub(super) fn from_budget(budget: NativeResourceBudget) -> (Self, ResourceDrain) {
        let inner = Arc::new(ResourceSupervisorInner {
            state: Mutex::new(ResourceState::new(budget)),
        });
        (
            Self {
                inner: Arc::clone(&inner),
            },
            ResourceDrain { inner },
        )
    }

    pub(super) fn admit_preparation(
        &self,
        retained_bytes: u64,
        work_bytes: u64,
    ) -> ResourceAdmissionFuture {
        ResourceAdmissionFuture::new(
            Arc::clone(&self.inner),
            AdmissionCharge::Preparation {
                retained_bytes,
                work_bytes,
            },
        )
    }

    pub(super) fn admit_encode(&self, work_bytes: u64) -> ResourceAdmissionFuture {
        ResourceAdmissionFuture::new(
            Arc::clone(&self.inner),
            AdmissionCharge::Encode { work_bytes },
        )
    }
}

pub(super) struct ResourceAdmissionFuture {
    inner: Arc<ResourceSupervisorInner>,
    charge: AdmissionCharge,
    waiter: Option<AdmissionWaiterSlot>,
    complete: bool,
}

impl ResourceAdmissionFuture {
    fn new(inner: Arc<ResourceSupervisorInner>, charge: AdmissionCharge) -> Self {
        Self {
            inner,
            charge,
            waiter: None,
            complete: false,
        }
    }
}

impl Future for ResourceAdmissionFuture {
    type Output = std::result::Result<ResourceAdmissionGuard, ResourceAdmissionError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        assert!(
            !this.complete,
            "resource admission future polled after completion"
        );
        let inner = Arc::clone(&this.inner);

        loop {
            let mut state = inner.lock_state();

            if state.closed {
                state.remove_admission_waiter(this.waiter.take());
                this.complete = true;
                return Poll::Ready(Err(ResourceAdmissionError::Closed));
            }
            if !state.accounting_valid() {
                state.remove_admission_waiter(this.waiter.take());
                let admission_wakers = state.close_admission();
                let release_wakers = state.completed_release_wakers();
                this.complete = true;
                drop(state);
                wake_all(admission_wakers);
                wake_all(release_wakers);
                return Poll::Ready(Err(ResourceAdmissionError::Closed));
            }
            if state.request_exceeds_budget(this.charge) {
                state.remove_admission_waiter(this.waiter.take());
                let next_waker = state.select_admission().map(|(_, waker)| waker);
                this.complete = true;
                drop(state);
                if let Some(waker) = next_waker {
                    waker.wake();
                }
                return Poll::Ready(Err(ResourceAdmissionError::RequestExceedsBudget));
            }

            let owns_entitlement = this.waiter.is_some() && state.entitled_waiter == this.waiter;
            let uncontended = this.waiter.is_none()
                && state.entitled_waiter.is_none()
                && state.admission_waiters_empty();
            if (owns_entitlement || uncontended) && state.can_admit(this.charge) {
                state.remove_admission_waiter(this.waiter.take());
                if !state.reserve(this.charge) {
                    let admission_wakers = state.close_admission();
                    let release_wakers = state.completed_release_wakers();
                    this.complete = true;
                    drop(state);
                    wake_all(admission_wakers);
                    wake_all(release_wakers);
                    return Poll::Ready(Err(ResourceAdmissionError::Closed));
                }
                let next_waker = state.select_admission().map(|(_, waker)| waker);
                this.complete = true;
                drop(state);
                if let Some(waker) = next_waker {
                    waker.wake();
                }
                return Poll::Ready(Ok(ResourceAdmissionGuard {
                    inner,
                    charge: this.charge,
                }));
            }

            if !state.register_admission_waiter(&mut this.waiter, this.charge, context.waker()) {
                let admission_wakers = state.close_admission();
                let release_wakers = state.completed_release_wakers();
                this.complete = true;
                drop(state);
                wake_all(admission_wakers);
                wake_all(release_wakers);
                return Poll::Ready(Err(ResourceAdmissionError::Closed));
            }

            let selected = state.select_admission();
            if selected
                .as_ref()
                .is_some_and(|(slot, _)| Some(*slot) == this.waiter)
            {
                continue;
            }
            drop(state);
            if let Some((_, waker)) = selected {
                waker.wake();
            }
            return Poll::Pending;
        }
    }
}

impl Drop for ResourceAdmissionFuture {
    fn drop(&mut self) {
        let Some(waiter) = self.waiter.take() else {
            return;
        };
        let next_waker = {
            let mut state = self.inner.lock_state();
            state.remove_admission_waiter(Some(waiter));
            state.select_admission().map(|(_, waker)| waker)
        };
        if let Some(waker) = next_waker {
            waker.wake();
        }
    }
}

pub(super) struct ResourceAdmissionGuard {
    inner: Arc<ResourceSupervisorInner>,
    charge: AdmissionCharge,
}

impl Drop for ResourceAdmissionGuard {
    fn drop(&mut self) {
        self.inner.release(self.charge);
    }
}

#[derive(Clone)]
pub(super) struct ResourceDrain {
    inner: Arc<ResourceSupervisorInner>,
}

impl ResourceDrain {
    pub(super) fn begin_drain(&self) {
        let (admission_wakers, release_wakers) = {
            let mut state = self.inner.lock_state();
            (state.close_admission(), state.completed_release_wakers())
        };
        wake_all(admission_wakers);
        wake_all(release_wakers);
    }

    pub(super) fn wait_released(&self) -> ResourceReleaseFuture {
        ResourceReleaseFuture {
            inner: Arc::clone(&self.inner),
            waiter: None,
            complete: false,
        }
    }
}

pub(super) struct ResourceReleaseFuture {
    inner: Arc<ResourceSupervisorInner>,
    waiter: Option<usize>,
    complete: bool,
}

impl Future for ResourceReleaseFuture {
    type Output = std::result::Result<(), ResourceLifecycleError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        assert!(
            !this.complete,
            "resource release future polled after completion"
        );
        let inner = Arc::clone(&this.inner);
        let mut state = inner.lock_state();

        if !state.closed {
            state.remove_release_waiter(this.waiter.take());
            this.complete = true;
            return Poll::Ready(Err(ResourceLifecycleError::DrainNotStarted));
        }
        if state.resources_released() {
            state.remove_release_waiter(this.waiter.take());
            this.complete = true;
            return Poll::Ready(Ok(()));
        }

        state.register_release_waiter(&mut this.waiter, context.waker());
        Poll::Pending
    }
}

impl Drop for ResourceReleaseFuture {
    fn drop(&mut self) {
        let Some(waiter) = self.waiter.take() else {
            return;
        };
        self.inner.lock_state().remove_release_waiter(Some(waiter));
    }
}

#[cfg(test)]
mod tests;
