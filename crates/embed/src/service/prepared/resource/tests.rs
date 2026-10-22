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
fn one_capacity_release_wakes_only_the_fifo_entitled_waiter() {
    let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
    let held = expect_immediate(supervisor.admit_preparation(1, 1))
        .expect("first preparation is within budget");
    let mut queued = (0..3)
        .map(|_| Box::pin(supervisor.admit_preparation(1, 1)))
        .collect::<Vec<_>>();
    let wake_counters = (0..3)
        .map(|_| Arc::new(WakeCounter::default()))
        .collect::<Vec<_>>();
    let wakers = wake_counters
        .iter()
        .map(|counter| Waker::from(Arc::clone(counter)))
        .collect::<Vec<_>>();

    for (future, waker) in queued.iter_mut().zip(&wakers) {
        assert!(matches!(
            poll_with_waker(future.as_mut(), waker),
            Poll::Pending
        ));
    }

    drop(held);
    assert_eq!(
        wake_counters
            .iter()
            .map(|counter| counter.count())
            .collect::<Vec<_>>(),
        vec![1, 0, 0],
        "one released preparation slot must grant one FIFO wake entitlement"
    );
}

#[test]
fn an_entitled_waiter_cannot_be_overtaken_by_a_same_class_newcomer() {
    let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
    let held = expect_immediate(supervisor.admit_preparation(1, 1))
        .expect("first preparation is within budget");
    let mut older = Box::pin(supervisor.admit_preparation(1, 1));
    let older_wakes = Arc::new(WakeCounter::default());
    let older_waker = Waker::from(Arc::clone(&older_wakes));

    assert!(matches!(
        poll_with_waker(older.as_mut(), &older_waker),
        Poll::Pending
    ));
    drop(held);
    assert_eq!(older_wakes.count(), 1);

    let mut newcomer = Box::pin(supervisor.admit_preparation(1, 1));
    let newcomer_wakes = Arc::new(WakeCounter::default());
    let newcomer_waker = Waker::from(Arc::clone(&newcomer_wakes));
    assert!(matches!(
        poll_with_waker(newcomer.as_mut(), &newcomer_waker),
        Poll::Pending
    ));

    let Poll::Ready(Ok(older_guard)) = poll_with_waker(older.as_mut(), &older_waker) else {
        panic!("the FIFO-entitled older waiter must make progress");
    };
    drop(older_guard);
    assert_eq!(
        newcomer_wakes.count(),
        1,
        "the newcomer must wake only after the older guard releases"
    );
    assert!(matches!(
        poll_with_waker(newcomer.as_mut(), &newcomer_waker),
        Poll::Ready(Ok(_))
    ));
}

#[test]
fn successful_reservation_hands_off_already_free_capacity() {
    let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(2, 1, 10, 10));
    let first_held = expect_immediate(supervisor.admit_preparation(1, 1))
        .expect("first preparation is within budget");
    let second_held = expect_immediate(supervisor.admit_preparation(1, 1))
        .expect("second preparation is within budget");
    let mut first_waiter = Box::pin(supervisor.admit_preparation(1, 1));
    let mut second_waiter = Box::pin(supervisor.admit_preparation(1, 1));
    let first_wakes = Arc::new(WakeCounter::default());
    let second_wakes = Arc::new(WakeCounter::default());
    let first_waker = Waker::from(Arc::clone(&first_wakes));
    let second_waker = Waker::from(Arc::clone(&second_wakes));

    assert!(matches!(
        poll_with_waker(first_waiter.as_mut(), &first_waker),
        Poll::Pending
    ));
    assert!(matches!(
        poll_with_waker(second_waiter.as_mut(), &second_waker),
        Poll::Pending
    ));

    drop(first_held);
    assert_eq!(first_wakes.count(), 1);
    assert_eq!(second_wakes.count(), 0);

    drop(second_held);
    assert_eq!(second_wakes.count(), 0);

    let Poll::Ready(Ok(first_guard)) = poll_with_waker(first_waiter.as_mut(), &first_waker) else {
        panic!("the first waiter must consume its entitlement");
    };
    assert_eq!(
        second_wakes.count(),
        1,
        "the successful reservation must hand off the second free slot"
    );

    let Poll::Ready(Ok(_second_guard)) = poll_with_waker(second_waiter.as_mut(), &second_waker)
    else {
        panic!("the second waiter must progress before the first guard drops");
    };
    drop(first_guard);
}

#[test]
fn shared_capacity_release_alternates_progress_across_admission_classes() {
    let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 1));
    let held = expect_immediate(supervisor.admit_preparation(1, 1))
        .expect("first preparation is within budget");
    let mut first_preparation = Box::pin(supervisor.admit_preparation(1, 1));
    let mut second_preparation = Box::pin(supervisor.admit_preparation(1, 1));
    let mut encode = Box::pin(supervisor.admit_encode(1));
    let first_preparation_wakes = Arc::new(WakeCounter::default());
    let second_preparation_wakes = Arc::new(WakeCounter::default());
    let encode_wakes = Arc::new(WakeCounter::default());
    let first_preparation_waker = Waker::from(Arc::clone(&first_preparation_wakes));
    let second_preparation_waker = Waker::from(Arc::clone(&second_preparation_wakes));
    let encode_waker = Waker::from(Arc::clone(&encode_wakes));

    assert!(matches!(
        poll_with_waker(first_preparation.as_mut(), &first_preparation_waker),
        Poll::Pending
    ));
    assert!(matches!(
        poll_with_waker(second_preparation.as_mut(), &second_preparation_waker),
        Poll::Pending
    ));
    assert!(matches!(
        poll_with_waker(encode.as_mut(), &encode_waker),
        Poll::Pending
    ));

    drop(held);
    assert_eq!(first_preparation_wakes.count(), 1);
    assert_eq!(second_preparation_wakes.count(), 0);
    assert_eq!(encode_wakes.count(), 0);

    let Poll::Ready(Ok(first_guard)) =
        poll_with_waker(first_preparation.as_mut(), &first_preparation_waker)
    else {
        panic!("the first preparation must own the first entitlement");
    };
    drop(first_guard);
    assert_eq!(second_preparation_wakes.count(), 0);
    assert_eq!(
        encode_wakes.count(),
        1,
        "the next shared-capacity release must advance the other class"
    );

    let Poll::Ready(Ok(encode_guard)) = poll_with_waker(encode.as_mut(), &encode_waker) else {
        panic!("the encode waiter must own the alternating entitlement");
    };
    drop(encode_guard);
    assert_eq!(second_preparation_wakes.count(), 1);
}

#[test]
fn drain_notifies_each_queued_admission_only_once() {
    let (supervisor, drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
    let preparation =
        expect_immediate(supervisor.admit_preparation(1, 4)).expect("preparation is within budget");
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
    assert_eq!(preparation_wakes.count(), 1);
    assert_eq!(encode_wakes.count(), 1);

    drain.begin_drain();
    drop(preparation);
    drop(encode);
    assert_eq!(
        preparation_wakes.count(),
        1,
        "idempotent drain and post-drain releases must not re-wake admissions"
    );
    assert_eq!(
        encode_wakes.count(),
        1,
        "idempotent drain and post-drain releases must not re-wake admissions"
    );
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
    let held = expect_immediate(supervisor.admit_encode(1)).expect("first encode is within budget");
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
fn cancelled_admission_high_water_releases_storage_and_accepts_a_new_waiter() {
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
    }

    drop(queued);
    {
        let state = supervisor.inner.lock_state();
        assert!(state.preparation_waiters.is_empty());
    }

    let mut reused = Box::pin(supervisor.admit_preparation(1, 1));
    assert!(matches!(poll_once(reused.as_mut()), Poll::Pending));
    {
        let state = supervisor.inner.lock_state();
        assert_eq!(state.preparation_waiters.len(), 1);
        assert!(state.preparation_waiters.contains_key(&0));
    }
    drop(reused);
    drop(held);
}

#[test]
fn cancelling_slot_blocked_encode_reserves_no_work_bytes() {
    let (supervisor, _drain) = ResourceSupervisor::from_budget(budget(1, 1, 10, 10));
    let held = expect_immediate(supervisor.admit_encode(1)).expect("first encode is within budget");
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
    let preparation =
        expect_immediate(supervisor.admit_preparation(6, 4)).expect("preparation is within budget");
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
