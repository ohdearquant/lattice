use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Inactive,
    WaitingForConverterBoundary,
    BeforeRejection,
    AfterRejection,
    WaitingForMaterializedWorkingSetBoundary,
    MaterializedWorkingSetBoundary,
    MaterializedWorkingSetBoundaryCompleted,
}

#[derive(Clone, Copy)]
struct State {
    phase: Phase,
    before_rejection_allocation_calls: usize,
    after_rejection_allocation_calls: usize,
    materialized_working_set_allocation_calls: usize,
    rejection_seen: bool,
    reader_boundary_seen: bool,
    materialized_working_set_boundary_seen: bool,
    materialized_working_set_boundary_completed: bool,
}

const INACTIVE: State = State {
    phase: Phase::Inactive,
    before_rejection_allocation_calls: 0,
    after_rejection_allocation_calls: 0,
    materialized_working_set_allocation_calls: 0,
    rejection_seen: false,
    reader_boundary_seen: false,
    materialized_working_set_boundary_seen: false,
    materialized_working_set_boundary_completed: false,
};

std::thread_local! {
    static STATE: Cell<State> = const { Cell::new(INACTIVE) };
}

struct TrackingAllocator;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn record_allocation() {
    let _ = STATE.try_with(|cell| {
        let mut state = cell.get();
        match state.phase {
            Phase::BeforeRejection => {
                state.before_rejection_allocation_calls =
                    state.before_rejection_allocation_calls.saturating_add(1);
            }
            Phase::AfterRejection => {
                state.after_rejection_allocation_calls =
                    state.after_rejection_allocation_calls.saturating_add(1);
            }
            Phase::MaterializedWorkingSetBoundary => {
                state.materialized_working_set_allocation_calls = state
                    .materialized_working_set_allocation_calls
                    .saturating_add(1);
            }
            Phase::Inactive
            | Phase::WaitingForConverterBoundary
            | Phase::WaitingForMaterializedWorkingSetBoundary
            | Phase::MaterializedWorkingSetBoundaryCompleted => return,
        }
        cell.set(state);
    });
}

// SAFETY: allocation requests are observed without modification and then
// forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        System.realloc(ptr, layout, new_size)
    }
}

pub(crate) struct Guard {
    active: bool,
}

pub(crate) struct Observation {
    pub(crate) before_rejection_allocation_calls: usize,
    pub(crate) after_rejection_allocation_calls: usize,
    pub(crate) materialized_working_set_allocation_calls: usize,
    pub(crate) rejection_seen: bool,
    pub(crate) reader_boundary_seen: bool,
    pub(crate) materialized_working_set_boundary_seen: bool,
    pub(crate) materialized_working_set_boundary_completed: bool,
}

fn start_in_phase(phase: Phase) -> Guard {
    STATE.with(|cell| {
        assert_eq!(
            cell.get().phase,
            Phase::Inactive,
            "allocation tracking already active"
        );
        cell.set(State { phase, ..INACTIVE });
    });
    Guard { active: true }
}

pub(super) fn start() -> Guard {
    start_in_phase(Phase::BeforeRejection)
}

pub(crate) fn start_at_converter_boundary() -> Guard {
    start_in_phase(Phase::WaitingForConverterBoundary)
}

pub(crate) fn start_at_materialized_working_set_boundary() -> Guard {
    start_in_phase(Phase::WaitingForMaterializedWorkingSetBoundary)
}

pub(crate) fn mark_converter_boundary() {
    let _ = STATE.try_with(|cell| {
        let mut state = cell.get();
        if state.phase == Phase::WaitingForConverterBoundary {
            state.phase = Phase::BeforeRejection;
            cell.set(state);
        }
    });
}

pub(crate) fn mark_reader_boundary() {
    let _ = STATE.try_with(|cell| {
        let mut state = cell.get();
        if state.phase != Phase::Inactive {
            state.reader_boundary_seen = true;
            cell.set(state);
        }
    });
}

pub(crate) fn mark_materialized_working_set_boundary() {
    let _ = STATE.try_with(|cell| {
        let mut state = cell.get();
        if state.phase != Phase::Inactive {
            state.materialized_working_set_boundary_seen = true;
        }
        if state.phase == Phase::WaitingForMaterializedWorkingSetBoundary {
            state.phase = Phase::MaterializedWorkingSetBoundary;
        }
        cell.set(state);
    });
}

pub(crate) fn mark_materialized_working_set_boundary_completed() {
    let _ = STATE.try_with(|cell| {
        let mut state = cell.get();
        if state.phase != Phase::Inactive {
            state.materialized_working_set_boundary_completed = true;
        }
        if state.phase == Phase::MaterializedWorkingSetBoundary {
            state.phase = Phase::MaterializedWorkingSetBoundaryCompleted;
        }
        cell.set(state);
    });
}

pub(super) fn mark_rejection() {
    let _ = STATE.try_with(|cell| {
        let mut state = cell.get();
        if state.phase == Phase::BeforeRejection {
            state.phase = Phase::AfterRejection;
            state.rejection_seen = true;
            cell.set(state);
        }
    });
}

impl Guard {
    pub(crate) fn finish(mut self) -> Observation {
        self.active = false;
        STATE.with(|cell| {
            let state = cell.replace(INACTIVE);
            Observation {
                before_rejection_allocation_calls: state.before_rejection_allocation_calls,
                after_rejection_allocation_calls: state.after_rejection_allocation_calls,
                materialized_working_set_allocation_calls: state
                    .materialized_working_set_allocation_calls,
                rejection_seen: state.rejection_seen,
                reader_boundary_seen: state.reader_boundary_seen,
                materialized_working_set_boundary_seen: state
                    .materialized_working_set_boundary_seen,
                materialized_working_set_boundary_completed: state
                    .materialized_working_set_boundary_completed,
            }
        })
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.active {
            let _ = STATE.try_with(|cell| cell.set(INACTIVE));
        }
    }
}
