//! Dormant private admission and drain substrate for ADR-088.
//!
//! This slice accounts active preparation/encode guards and their byte leases. Retained-lease
//! transfer into published objects, preparation-only terminal faults, residue debt, and aggregate
//! cleanup errors remain outside this substrate and must land before prepared services use it.

use super::NativeResourceBudget;
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

#[derive(Clone, Copy)]
struct AdmissionWaiterSlot {
    class: AdmissionClass,
    index: usize,
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
    preparation_waiters: Vec<Option<AdmissionWaiter>>,
    encode_waiters: Vec<Option<AdmissionWaiter>>,
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
            preparation_waiters: Vec::new(),
            encode_waiters: Vec::new(),
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

    fn admission_waiters_mut(
        &mut self,
        class: AdmissionClass,
    ) -> &mut Vec<Option<AdmissionWaiter>> {
        match class {
            AdmissionClass::Preparation => &mut self.preparation_waiters,
            AdmissionClass::Encode => &mut self.encode_waiters,
        }
    }

    fn register_admission_waiter(
        &mut self,
        slot: &mut Option<AdmissionWaiterSlot>,
        charge: AdmissionCharge,
        waker: &Waker,
    ) {
        let class = charge.class();
        if let Some(existing) = *slot {
            let queue = self.admission_waiters_mut(existing.class);
            if let Some(Some(waiter)) = queue.get_mut(existing.index) {
                waiter.charge = charge;
                if !waiter.waker.will_wake(waker) {
                    waiter.waker = waker.clone();
                }
                return;
            }
        }

        let queue = self.admission_waiters_mut(class);
        if let Some((index, vacant)) = queue
            .iter_mut()
            .enumerate()
            .find(|(_, entry)| entry.is_none())
        {
            *vacant = Some(AdmissionWaiter {
                charge,
                waker: waker.clone(),
            });
            *slot = Some(AdmissionWaiterSlot { class, index });
            return;
        }

        let index = queue.len();
        queue.push(Some(AdmissionWaiter {
            charge,
            waker: waker.clone(),
        }));
        *slot = Some(AdmissionWaiterSlot { class, index });
    }

    fn remove_admission_waiter(&mut self, slot: Option<AdmissionWaiterSlot>) {
        let Some(slot) = slot else {
            return;
        };
        let queue = self.admission_waiters_mut(slot.class);
        if let Some(entry) = queue.get_mut(slot.index) {
            *entry = None;
        }
        trim_vacant_tail(queue);
    }

    fn eligible_admission_wakers(&self) -> Vec<Waker> {
        if self.closed {
            return Vec::new();
        }

        self.preparation_waiters
            .iter()
            .chain(&self.encode_waiters)
            .filter_map(Option::as_ref)
            .filter(|waiter| self.can_admit(waiter.charge))
            .map(|waiter| waiter.waker.clone())
            .collect()
    }

    fn all_admission_wakers(&self) -> Vec<Waker> {
        self.preparation_waiters
            .iter()
            .chain(&self.encode_waiters)
            .filter_map(Option::as_ref)
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
            if !state.release(charge) {
                state.closed = true;
            }
            (
                if state.closed {
                    state.all_admission_wakers()
                } else {
                    state.eligible_admission_wakers()
                },
                state.completed_release_wakers(),
            )
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
        let mut state = inner.lock_state();

        if state.closed {
            state.remove_admission_waiter(this.waiter.take());
            this.complete = true;
            return Poll::Ready(Err(ResourceAdmissionError::Closed));
        }
        if !state.accounting_valid() {
            state.closed = true;
            state.remove_admission_waiter(this.waiter.take());
            let admission_wakers = state.all_admission_wakers();
            let release_wakers = state.completed_release_wakers();
            this.complete = true;
            drop(state);
            wake_all(admission_wakers);
            wake_all(release_wakers);
            return Poll::Ready(Err(ResourceAdmissionError::Closed));
        }
        if state.request_exceeds_budget(this.charge) {
            state.remove_admission_waiter(this.waiter.take());
            this.complete = true;
            return Poll::Ready(Err(ResourceAdmissionError::RequestExceedsBudget));
        }
        if state.can_admit(this.charge) {
            state.remove_admission_waiter(this.waiter.take());
            if !state.reserve(this.charge) {
                state.closed = true;
                let admission_wakers = state.all_admission_wakers();
                let release_wakers = state.completed_release_wakers();
                this.complete = true;
                drop(state);
                wake_all(admission_wakers);
                wake_all(release_wakers);
                return Poll::Ready(Err(ResourceAdmissionError::Closed));
            }
            this.complete = true;
            drop(state);
            return Poll::Ready(Ok(ResourceAdmissionGuard {
                inner,
                charge: this.charge,
            }));
        }

        state.register_admission_waiter(&mut this.waiter, this.charge, context.waker());
        Poll::Pending
    }
}

impl Drop for ResourceAdmissionFuture {
    fn drop(&mut self) {
        let Some(waiter) = self.waiter.take() else {
            return;
        };
        self.inner
            .lock_state()
            .remove_admission_waiter(Some(waiter));
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
            state.closed = true;
            (
                state.all_admission_wakers(),
                state.completed_release_wakers(),
            )
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
mod tests {
    use super::super::NativeResourceBudget;
    use super::{ResourceAdmissionError, ResourceLifecycleError, ResourceSupervisor};
    use std::future::Future;
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Wake, Waker};

    fn budget(
        max_preparations: usize,
        max_encodes: usize,
        max_retained_bytes: u64,
        max_work_bytes: u64,
    ) -> NativeResourceBudget {
        NativeResourceBudget::try_new(
            NonZeroUsize::new(max_preparations).expect("test preparation ceiling is non-zero"),
            NonZeroUsize::new(max_encodes).expect("test encode ceiling is non-zero"),
            NonZeroU64::new(max_retained_bytes).expect("test retained ceiling is non-zero"),
            NonZeroU64::new(max_work_bytes).expect("test work ceiling is non-zero"),
        )
        .expect("test budget is valid")
    }

    fn poll_once<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
        let mut context = Context::from_waker(Waker::noop());
        future.poll(&mut context)
    }

    fn poll_with_waker<F: Future>(future: Pin<&mut F>, waker: &Waker) -> Poll<F::Output> {
        let mut context = Context::from_waker(waker);
        future.poll(&mut context)
    }

    fn expect_immediate<F: Future>(future: F) -> F::Output {
        let mut future = Box::pin(future);
        match poll_once(future.as_mut()) {
            Poll::Ready(output) => output,
            Poll::Pending => panic!("test operation must complete on its first poll"),
        }
    }

    #[derive(Default)]
    struct WakeCounter(AtomicUsize);

    impl Wake for WakeCounter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl WakeCounter {
        fn count(&self) -> usize {
            self.0.load(Ordering::SeqCst)
        }
    }

    #[test]
    fn exact_independent_byte_pool_ceilings_can_be_held_together() {
        let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));

        let _preparation = expect_immediate(supervisor.admit_preparation(10, 1))
            .expect("the exact retained ceiling must be admitted");
        let _encode = expect_immediate(supervisor.admit_encode(9))
            .expect("retained bytes must not consume transient-work capacity");
    }

    #[test]
    fn oversized_single_requests_fail_on_their_first_poll_without_reserving_any_axis() {
        let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));

        let mut retained_oversize = Box::pin(supervisor.admit_preparation(11, 1));
        assert!(matches!(
            poll_once(retained_oversize.as_mut()),
            Poll::Ready(Err(ResourceAdmissionError::RequestExceedsBudget))
        ));

        let mut work_oversize = Box::pin(supervisor.admit_preparation(1, 11));
        assert!(matches!(
            poll_once(work_oversize.as_mut()),
            Poll::Ready(Err(ResourceAdmissionError::RequestExceedsBudget))
        ));

        let mut encode_oversize = Box::pin(supervisor.admit_encode(11));
        assert!(matches!(
            poll_once(encode_oversize.as_mut()),
            Poll::Ready(Err(ResourceAdmissionError::RequestExceedsBudget))
        ));

        let _preparation = expect_immediate(supervisor.admit_preparation(10, 4))
            .expect("rejected requests must not consume preparation, retained, or work capacity");
        let _encode = expect_immediate(supervisor.admit_encode(6))
            .expect("rejected requests must not consume encode or work capacity");
    }

    #[test]
    fn impossible_accounting_corruption_closes_admission_without_false_idle() {
        let (supervisor, drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        supervisor.inner.lock_state().work_bytes = 11;

        let mut admission = Box::pin(supervisor.admit_encode(1));
        assert!(matches!(
            poll_once(admission.as_mut()),
            Poll::Ready(Err(ResourceAdmissionError::Closed))
        ));

        let mut wait = Box::pin(drain.wait_released());
        assert!(matches!(poll_once(wait.as_mut()), Poll::Pending));
    }

    #[test]
    fn queued_preparation_wakes_and_admits_after_capacity_releases() {
        let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let held = expect_immediate(supervisor.admit_preparation(1, 1))
            .expect("first preparation is within budget");
        let mut queued = Box::pin(supervisor.admit_preparation(9, 9));
        let wake_counter = Arc::new(WakeCounter::default());
        let waker = Waker::from(Arc::clone(&wake_counter));

        assert!(matches!(
            poll_with_waker(queued.as_mut(), &waker),
            Poll::Pending
        ));

        drop(held);
        assert!(
            wake_counter.count() > 0,
            "capacity release must wake the waiter"
        );
        assert!(matches!(
            poll_with_waker(queued.as_mut(), &waker),
            Poll::Ready(Ok(_))
        ));
    }

    #[test]
    fn queued_encode_wakes_and_admits_after_capacity_releases() {
        let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let held =
            expect_immediate(supervisor.admit_encode(1)).expect("first encode is within budget");
        let mut queued = Box::pin(supervisor.admit_encode(9));
        let wake_counter = Arc::new(WakeCounter::default());
        let waker = Waker::from(Arc::clone(&wake_counter));

        assert!(matches!(
            poll_with_waker(queued.as_mut(), &waker),
            Poll::Pending
        ));

        drop(held);
        assert!(
            wake_counter.count() > 0,
            "encode capacity release must wake the waiter"
        );
        assert!(matches!(
            poll_with_waker(queued.as_mut(), &waker),
            Poll::Ready(Ok(_))
        ));
    }

    #[test]
    fn repolling_queued_admission_replaces_its_registered_waker() {
        let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let held = expect_immediate(supervisor.admit_preparation(1, 1))
            .expect("first preparation is within budget");
        let mut queued = Box::pin(supervisor.admit_preparation(9, 9));
        let stale_wakes = Arc::new(WakeCounter::default());
        let current_wakes = Arc::new(WakeCounter::default());
        let stale_waker = Waker::from(Arc::clone(&stale_wakes));
        let current_waker = Waker::from(Arc::clone(&current_wakes));

        assert!(matches!(
            poll_with_waker(queued.as_mut(), &stale_waker),
            Poll::Pending
        ));
        assert!(matches!(
            poll_with_waker(queued.as_mut(), &current_waker),
            Poll::Pending
        ));

        drop(held);
        assert_eq!(
            stale_wakes.count(),
            0,
            "capacity release must not wake a replaced waker"
        );
        assert!(
            current_wakes.count() > 0,
            "capacity release must wake the most recently registered waker"
        );
    }

    #[test]
    fn cancelling_slot_blocked_preparation_reserves_neither_byte_pool() {
        let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let held = expect_immediate(supervisor.admit_preparation(1, 1))
            .expect("first preparation is within budget");
        let mut queued = Box::pin(supervisor.admit_preparation(9, 9));

        assert!(matches!(poll_once(queued.as_mut()), Poll::Pending));

        drop(queued);
        drop(held);
        let _exact = expect_immediate(supervisor.admit_preparation(10, 10))
            .expect("cancelled waiter must not leak retained or work capacity");
    }

    #[test]
    fn cancelled_admission_is_absent_from_later_capacity_scans() {
        let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let held = expect_immediate(supervisor.admit_preparation(1, 1))
            .expect("first preparation is within budget");
        let mut queued = Box::pin(supervisor.admit_preparation(9, 9));
        let wake_counter = Arc::new(WakeCounter::default());
        let waker = Waker::from(Arc::clone(&wake_counter));

        assert!(matches!(
            poll_with_waker(queued.as_mut(), &waker),
            Poll::Pending
        ));

        drop(queued);
        drop(held);
        assert_eq!(
            wake_counter.count(),
            0,
            "a later capacity scan must not retain a cancelled admission waker"
        );
    }

    #[test]
    fn cancelled_admission_high_water_releases_storage_and_reuses_slot_zero() {
        let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let held = expect_immediate(supervisor.admit_preparation(1, 1))
            .expect("first preparation is within budget");
        let mut queued = (0..64)
            .map(|_| Box::pin(supervisor.admit_preparation(1, 1)))
            .collect::<Vec<_>>();

        for future in &mut queued {
            assert!(matches!(poll_once(future.as_mut()), Poll::Pending));
        }
        {
            let state = supervisor.inner.lock_state();
            assert_eq!(state.preparation_waiters.len(), 64);
            assert!(state.preparation_waiters.capacity() >= 64);
        }

        drop(queued);
        {
            let state = supervisor.inner.lock_state();
            assert_eq!(state.preparation_waiters.len(), 0);
            assert_eq!(state.preparation_waiters.capacity(), 0);
        }

        let mut reused = Box::pin(supervisor.admit_preparation(1, 1));
        assert!(matches!(poll_once(reused.as_mut()), Poll::Pending));
        {
            let state = supervisor.inner.lock_state();
            assert!(
                state
                    .preparation_waiters
                    .first()
                    .is_some_and(Option::is_some)
            );
            assert_eq!(state.preparation_waiters.len(), 1);
        }
        drop(reused);
        drop(held);
    }

    #[test]
    fn cancelling_slot_blocked_encode_reserves_no_work_bytes() {
        let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let held =
            expect_immediate(supervisor.admit_encode(1)).expect("first encode is within budget");
        let mut queued = Box::pin(supervisor.admit_encode(9));

        assert!(matches!(poll_once(queued.as_mut()), Poll::Pending));

        drop(queued);
        drop(held);
        let _exact = expect_immediate(supervisor.admit_encode(10))
            .expect("cancelled waiter must not leak work capacity");
    }

    #[test]
    fn cancelling_retained_blocked_preparation_reserves_no_slot_or_work_bytes() {
        let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(2, 1, 10, 10));
        let held = expect_immediate(supervisor.admit_preparation(9, 1))
            .expect("first preparation is within budget");
        let mut queued = Box::pin(supervisor.admit_preparation(2, 9));

        assert!(matches!(poll_once(queued.as_mut()), Poll::Pending));

        drop(queued);
        drop(held);
        let _first = expect_immediate(supervisor.admit_preparation(5, 5))
            .expect("failed admission must not leak its first preparation slot");
        let _second = expect_immediate(supervisor.admit_preparation(5, 5))
            .expect("cancelled waiter must not leak retained or work capacity");
    }

    #[test]
    fn cancelling_work_blocked_preparation_reserves_no_slot_or_retained_bytes() {
        let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(2, 1, 10, 10));
        let held = expect_immediate(supervisor.admit_preparation(1, 9))
            .expect("first preparation is within budget");
        let mut queued = Box::pin(supervisor.admit_preparation(9, 2));

        assert!(matches!(poll_once(queued.as_mut()), Poll::Pending));

        drop(queued);
        drop(held);
        let _first = expect_immediate(supervisor.admit_preparation(5, 5))
            .expect("cancelled waiter must not leak its first preparation slot");
        let _second = expect_immediate(supervisor.admit_preparation(5, 5))
            .expect("cancelled waiter must not leak retained or work capacity");
    }

    #[test]
    fn retained_blocked_preparation_does_not_block_work_only_encode() {
        let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(2, 1, 10, 10));
        let _held = expect_immediate(supervisor.admit_preparation(10, 1))
            .expect("first preparation is within budget");
        let mut retained_blocked = Box::pin(supervisor.admit_preparation(1, 1));

        assert!(matches!(
            poll_once(retained_blocked.as_mut()),
            Poll::Pending
        ));
        let _encode = expect_immediate(supervisor.admit_encode(9))
            .expect("a retained-only blocker must not head-of-line block encode work");
    }

    #[test]
    fn wait_before_begin_drain_returns_a_typed_lifecycle_error() {
        let (_supervisor, drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let mut wait = Box::pin(drain.wait_released());

        assert!(matches!(
            poll_once(wait.as_mut()),
            Poll::Ready(Err(ResourceLifecycleError::DrainNotStarted))
        ));
    }

    #[test]
    fn begin_drain_is_idempotent_and_closes_both_admission_classes() {
        let (supervisor, drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));

        drain.begin_drain();
        drain.begin_drain();

        let mut preparation = Box::pin(supervisor.admit_preparation(1, 1));
        assert!(matches!(
            poll_once(preparation.as_mut()),
            Poll::Ready(Err(ResourceAdmissionError::Closed))
        ));

        let mut encode = Box::pin(supervisor.admit_encode(1));
        assert!(matches!(
            poll_once(encode.as_mut()),
            Poll::Ready(Err(ResourceAdmissionError::Closed))
        ));

        let mut wait = Box::pin(drain.wait_released());
        assert!(matches!(poll_once(wait.as_mut()), Poll::Ready(Ok(()))));
    }

    #[test]
    fn begin_drain_wakes_both_queued_admission_classes_with_closed() {
        let (supervisor, drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let preparation = expect_immediate(supervisor.admit_preparation(10, 4))
            .expect("preparation is within budget");
        let encode = expect_immediate(supervisor.admit_encode(6)).expect("encode is within budget");
        let mut queued_preparation = Box::pin(supervisor.admit_preparation(1, 1));
        let mut queued_encode = Box::pin(supervisor.admit_encode(1));
        let preparation_wakes = Arc::new(WakeCounter::default());
        let encode_wakes = Arc::new(WakeCounter::default());
        let preparation_waker = Waker::from(Arc::clone(&preparation_wakes));
        let encode_waker = Waker::from(Arc::clone(&encode_wakes));

        assert!(matches!(
            poll_with_waker(queued_preparation.as_mut(), &preparation_waker),
            Poll::Pending
        ));
        assert!(matches!(
            poll_with_waker(queued_encode.as_mut(), &encode_waker),
            Poll::Pending
        ));

        drain.begin_drain();
        assert!(
            preparation_wakes.count() > 0,
            "drain must wake queued preparations"
        );
        assert!(encode_wakes.count() > 0, "drain must wake queued encodes");
        assert!(matches!(
            poll_with_waker(queued_preparation.as_mut(), &preparation_waker),
            Poll::Ready(Err(ResourceAdmissionError::Closed))
        ));
        assert!(matches!(
            poll_with_waker(queued_encode.as_mut(), &encode_waker),
            Poll::Ready(Err(ResourceAdmissionError::Closed))
        ));

        let mut wait = Box::pin(drain.wait_released());
        assert!(matches!(poll_once(wait.as_mut()), Poll::Pending));
        drop(preparation);
        drop(encode);
        assert!(matches!(poll_once(wait.as_mut()), Poll::Ready(Ok(()))));
    }

    #[test]
    fn cancelled_release_wait_is_absent_from_later_completion_scans() {
        let (supervisor, drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let held = expect_immediate(supervisor.admit_encode(10)).expect("encode is within budget");
        drain.begin_drain();
        let mut wait = Box::pin(drain.wait_released());
        let wake_counter = Arc::new(WakeCounter::default());
        let waker = Waker::from(Arc::clone(&wake_counter));

        assert!(matches!(
            poll_with_waker(wait.as_mut(), &waker),
            Poll::Pending
        ));

        drop(wait);
        drop(held);
        assert_eq!(
            wake_counter.count(),
            0,
            "a later completion scan must not retain a cancelled drain waker"
        );
    }

    #[test]
    fn cancelled_release_wait_high_water_releases_storage_and_reuses_slot_zero() {
        let (supervisor, drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let held = expect_immediate(supervisor.admit_encode(10)).expect("encode is within budget");
        drain.begin_drain();
        let mut waits = (0..64)
            .map(|_| Box::pin(drain.wait_released()))
            .collect::<Vec<_>>();

        for future in &mut waits {
            assert!(matches!(poll_once(future.as_mut()), Poll::Pending));
        }
        {
            let state = supervisor.inner.lock_state();
            assert_eq!(state.release_waiters.len(), 64);
            assert!(state.release_waiters.capacity() >= 64);
        }

        drop(waits);
        {
            let state = supervisor.inner.lock_state();
            assert_eq!(state.release_waiters.len(), 0);
            assert_eq!(state.release_waiters.capacity(), 0);
        }

        let mut reused = Box::pin(drain.wait_released());
        assert!(matches!(poll_once(reused.as_mut()), Poll::Pending));
        {
            let state = supervisor.inner.lock_state();
            assert!(state.release_waiters.first().is_some_and(Option::is_some));
            assert_eq!(state.release_waiters.len(), 1);
        }
        drop(reused);
        drop(held);
    }

    #[test]
    fn final_guard_release_wakes_pending_release_wait() {
        let (supervisor, drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let held = expect_immediate(supervisor.admit_encode(10)).expect("encode is within budget");
        drain.begin_drain();
        let mut wait = Box::pin(drain.wait_released());
        let wake_counter = Arc::new(WakeCounter::default());
        let waker = Waker::from(Arc::clone(&wake_counter));

        assert!(matches!(
            poll_with_waker(wait.as_mut(), &waker),
            Poll::Pending
        ));

        drop(held);
        assert!(
            wake_counter.count() > 0,
            "final resource release must wake the drain waiter"
        );
        assert!(matches!(
            poll_with_waker(wait.as_mut(), &waker),
            Poll::Ready(Ok(()))
        ));
    }

    #[test]
    fn drain_waits_for_every_admitted_guard_to_release() {
        let (supervisor, drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
        let preparation = expect_immediate(supervisor.admit_preparation(6, 4))
            .expect("preparation is within budget");
        let encode = expect_immediate(supervisor.admit_encode(6))
            .expect("encode is within the remaining work budget");

        drain.begin_drain();
        let mut wait = Box::pin(drain.wait_released());
        assert!(matches!(poll_once(wait.as_mut()), Poll::Pending));

        drop(preparation);
        assert!(matches!(poll_once(wait.as_mut()), Poll::Pending));

        drop(encode);
        assert!(matches!(poll_once(wait.as_mut()), Poll::Ready(Ok(()))));
    }
}
