use std::hint::black_box;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::Command;

#[path = "../benches/support/allocation_counts.rs"]
mod allocation_counts;

use allocation_counts::{
    AllocationDelta, AllocationSnapshot, allocation_count_print, allocation_count_report,
    assert_zeroed_allocation_is_counted,
};

const SUITE: &str = "allocation_gate_controls";
const CHILD_ARGUMENT: &str = "--stable-allocation-print-child";

#[derive(Clone, Copy)]
enum Policy {
    Print,
    Report,
}

impl Policy {
    fn invoke(self, label: &str, run: &mut impl FnMut() -> AllocationDelta) {
        match self {
            Self::Print => allocation_count_print(SUITE, label, "decode", 2, run),
            Self::Report => allocation_count_report(SUITE, label, "decode", 2, run),
        }
    }
}

#[derive(Clone, Copy)]
enum Work {
    Zero,
    Allocate,
    AllocateOnCall(usize),
    Reallocate,
}

impl Work {
    fn measure(self, call: usize) -> AllocationDelta {
        if matches!(self, Self::Reallocate) {
            let mut bytes = vec![0x5a_u8; black_box(64)];
            black_box(&mut bytes);
            let before = AllocationSnapshot::capture();
            bytes.reserve_exact(black_box(128));
            black_box(&mut bytes);
            return AllocationSnapshot::delta_since(&before);
        }

        let allocate = match self {
            Self::Allocate => true,
            Self::AllocateOnCall(index) => call == index,
            Self::Zero | Self::Reallocate => false,
        };
        let before = AllocationSnapshot::capture();
        if allocate {
            let bytes = vec![0x5a_u8; black_box(512)];
            black_box(&bytes);
        } else {
            black_box(call);
        }
        AllocationSnapshot::delta_since(&before)
    }
}

#[derive(Default)]
struct Observed {
    calls: usize,
    deltas: [Option<AllocationDelta>; 4],
}

impl Observed {
    fn sample(&mut self, work: Work) -> AllocationDelta {
        let call = self.calls;
        self.calls += 1;
        let delta = work.measure(call);
        self.deltas[call] = Some(delta);
        delta
    }

    fn finish(self) -> [AllocationDelta; 4] {
        assert_eq!(
            self.calls, 4,
            "one warmup and three retained calls required"
        );
        self.deltas
            .map(|delta| delta.expect("callback was observed"))
    }
}

fn assert_zero(delta: AllocationDelta) {
    assert_eq!(delta.alloc_calls, 0);
    assert_eq!(delta.realloc_calls, 0);
    assert_eq!(delta.bytes_allocated, 0);
}

fn assert_allocation(delta: AllocationDelta) {
    assert_eq!(delta.alloc_calls, 1);
    assert_eq!(delta.realloc_calls, 0);
    assert_eq!(delta.bytes_allocated, 512);
}

fn rejected(
    policy: Policy,
    label: &str,
    work: Work,
    expected_reason: &str,
) -> [AllocationDelta; 4] {
    let mut observed = Observed::default();
    let result = catch_unwind(AssertUnwindSafe(|| {
        policy.invoke(label, &mut || observed.sample(work));
    }));
    let deltas = observed.finish();
    let payload = result.expect_err("allocation policy unexpectedly accepted the callback");
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .expect("allocation rejection must carry a string reason");
    assert!(
        message.starts_with(expected_reason),
        "wrong rejection: {message}"
    );
    assert!(
        message.contains(&format!(" for {SUITE}/{label}/decode:")),
        "wrong rejection heading: {message}"
    );
    deltas
}

fn warmup_only_allocation(policy: Policy) {
    let mut observed = Observed::default();
    policy.invoke("warmup_only", &mut || {
        observed.sample(Work::AllocateOnCall(0))
    });
    let deltas = observed.finish();
    assert_allocation(deltas[0]);
    for delta in &deltas[1..] {
        assert_zero(*delta);
    }
}

fn stable_allocation_print_child() {
    let mut observed = Observed::default();
    Policy::Print.invoke("stable_allocation", &mut || observed.sample(Work::Allocate));
    for delta in observed.finish() {
        assert_allocation(delta);
    }
    println!("stable-allocation-print-child: calls=4");
}

fn stable_allocation_print_is_truthful() {
    // A separate sequential process captures the real printer without redirecting
    // descriptors or adding allocations to its measured callbacks.
    let output = Command::new(std::env::current_exe().expect("current control executable"))
        .arg(CHILD_ARGUMENT)
        .output()
        .expect("launch print control");
    assert!(
        output.status.success(),
        "print child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).expect("child stdout is UTF-8"),
        "stable-allocation-print-child: calls=4\n"
    );
    let stderr = String::from_utf8(output.stderr).expect("child stderr is UTF-8");
    assert_eq!(
        stderr,
        "\nallocation_gate_controls/stable_allocation/decode\n\
         tokens=2\n\
         alloc_calls_total=1\n\
         realloc_calls_total=0\n\
         bytes_allocated_total=512\n\
         allocations_per_token=0.50\n\
         reallocations_per_token=0.00\n\
         bytes_allocated_per_token=256\n"
    );
}

fn stable_allocation_report_is_rejected() {
    let deltas = rejected(
        Policy::Report,
        "stable_allocation",
        Work::Allocate,
        "allocation gate failed for ",
    );
    for delta in deltas {
        assert_allocation(delta);
    }
}

fn per_call_reallocation_report_is_rejected() {
    let deltas = rejected(
        Policy::Report,
        "stable_reallocation",
        Work::Reallocate,
        "allocation gate failed for ",
    );
    for delta in deltas {
        assert_eq!(delta.alloc_calls, 0);
        assert_eq!(delta.realloc_calls, 1);
        assert!(delta.bytes_allocated >= 192);
        assert_eq!(delta, deltas[0]);
    }
}

fn retained_sample_drift_is_rejected(policy: Policy, retained_sample: usize) {
    assert!(retained_sample < 3);
    let changed_call = retained_sample + 1;
    let deltas = rejected(
        policy,
        "retained_sample_drift",
        Work::AllocateOnCall(changed_call),
        "allocation count is non-deterministic at run ",
    );
    for (call, delta) in deltas.into_iter().enumerate() {
        if call == changed_call {
            assert_allocation(delta);
        } else {
            assert_zero(delta);
        }
    }
}

fn zero_warm_report_is_accepted() {
    let mut observed = Observed::default();
    Policy::Report.invoke("zero_warm", &mut || observed.sample(Work::Zero));
    for delta in observed.finish() {
        assert_zero(delta);
    }
}

fn zeroed_allocator_is_live() {
    assert_zeroed_allocation_is_counted();
}

fn pass(name: &str, control: impl FnOnce(), completed: &mut usize) {
    control();
    *completed += 1;
    println!("allocation-control {name} ... ok");
}

fn main() {
    let mut arguments = std::env::args_os().skip(1);
    if let Some(argument) = arguments.next() {
        assert_eq!(argument, CHILD_ARGUMENT, "unexpected control argument");
        assert!(arguments.next().is_none(), "unexpected extra argument");
        stable_allocation_print_child();
        return;
    }

    let mut completed = 0;
    pass(
        "warmup_only_allocation_print_accepted",
        || warmup_only_allocation(Policy::Print),
        &mut completed,
    );
    pass(
        "warmup_only_allocation_report_accepted",
        || warmup_only_allocation(Policy::Report),
        &mut completed,
    );
    pass(
        "stable_allocation_print_truthful",
        stable_allocation_print_is_truthful,
        &mut completed,
    );
    pass(
        "stable_allocation_report_rejected",
        stable_allocation_report_is_rejected,
        &mut completed,
    );
    pass(
        "per_call_reallocation_report_rejected",
        per_call_reallocation_report_is_rejected,
        &mut completed,
    );
    for (sample, name) in [
        "retained_sample_0_print_drift_rejected",
        "retained_sample_1_print_drift_rejected",
        "retained_sample_2_print_drift_rejected",
    ]
    .into_iter()
    .enumerate()
    {
        pass(
            name,
            || retained_sample_drift_is_rejected(Policy::Print, sample),
            &mut completed,
        );
    }
    for (sample, name) in [
        "retained_sample_0_report_drift_rejected",
        "retained_sample_1_report_drift_rejected",
        "retained_sample_2_report_drift_rejected",
    ]
    .into_iter()
    .enumerate()
    {
        pass(
            name,
            || retained_sample_drift_is_rejected(Policy::Report, sample),
            &mut completed,
        );
    }
    pass(
        "zero_warm_report_accepted",
        zero_warm_report_is_accepted,
        &mut completed,
    );
    pass(
        "zeroed_allocator_live",
        zeroed_allocator_is_live,
        &mut completed,
    );
    assert_eq!(completed, 13);
    println!("allocation-controls: 13 passed; 0 failed");
}
