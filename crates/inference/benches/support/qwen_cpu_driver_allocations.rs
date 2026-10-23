use super::{
    AllocationDelta, AllocationSnapshot, QWEN_CPU_DRIVER_ALLOCATIONS, allocation_report_heading,
    assert_zeroed_allocation_is_counted,
};
use lattice_inference::GenerateConfig;
use lattice_inference::decoder_bench_support::{
    InterfaceProbe, RouteRun, run_interface_only, run_qwen_cpu_probed, run_qwen_cpu_streaming,
};
use lattice_inference::model::qwen35::test_support::tiny_zero_model;
use lattice_inference::model::qwen35::{Qwen35Model, RawGenEvent};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Measured repeats after one warm-up run. The repeats must agree exactly.
const REPEATS: usize = 3;
/// The region opens at the push of this token. Tokens before it are warm-in.
const WARM_IN_TOKENS: usize = 3;
const SYNTHETIC_TOKENS: usize = 48;
const REAL_TOKENS: usize = 16;
const INTERFACE_ONLY_PROMPT_TOKENS: usize = 8;
const SYNTHETIC_PROMPT: &str = "abc abc abc abc";
const REAL_PROMPT: &str = "The quick brown fox jumps over the lazy dog.";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    Streaming,
    Probed,
    InterfaceOnly,
}

impl Route {
    fn name(self) -> &'static str {
        match self {
            Route::Streaming => "qwen_cpu_streaming",
            Route::Probed => "qwen_cpu_probed",
            Route::InterfaceOnly => "interface_only",
        }
    }
}

/// What a probe does inside the per-token interface. Only the bench ever holds
/// the retaining code; no library build contains it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Injection {
    None,
    AllocOnSelect,
    AllocOnDecode,
    ReallocOnDecode,
}

impl Injection {
    fn name(self) -> &'static str {
        match self {
            Injection::None => "unchanged",
            Injection::AllocOnSelect => "retained_alloc_in_select",
            Injection::AllocOnDecode => "retained_alloc_in_decode",
            Injection::ReallocOnDecode => "retained_realloc_in_decode",
        }
    }
}

struct Profile {
    name: &'static str,
    cfg: GenerateConfig,
}

fn greedy_profile(max_new_tokens: usize, stop_token_ids: Vec<u32>) -> Profile {
    let mut cfg = GenerateConfig::default();
    cfg.max_new_tokens = max_new_tokens;
    cfg.temperature = 0.0;
    cfg.repetition_penalty = 1.0;
    cfg.seed = Some(7);
    cfg.enable_thinking = false;
    cfg.stop_token_ids = stop_token_ids;
    Profile { name: GREEDY, cfg }
}

fn sampled_profile(max_new_tokens: usize, stop_token_ids: Vec<u32>) -> Profile {
    let mut cfg = GenerateConfig::default();
    cfg.max_new_tokens = max_new_tokens;
    cfg.temperature = 0.7;
    cfg.top_k = 50;
    cfg.top_p = 0.9;
    cfg.repetition_penalty = 1.1;
    cfg.seed = Some(7);
    cfg.enable_thinking = false;
    cfg.stop_token_ids = stop_token_ids;
    Profile { name: SAMPLED, cfg }
}

/// Probe that counts its own calls and, per `injection`, retains an allocation or
/// a reallocation inside the interface. Everything it will touch is sized before
/// the request starts, so an unchanged probe allocates nothing.
struct BenchProbe {
    injection: Injection,
    selects: usize,
    decodes: usize,
    // One heap allocation per retained element is the point of the injection.
    #[allow(clippy::vec_box)]
    sink: Vec<Box<[u8; 64]>>,
    grow: Vec<u8>,
}

impl BenchProbe {
    fn new(injection: Injection, capacity: usize) -> Self {
        Self {
            injection,
            selects: 0,
            decodes: 0,
            sink: match injection {
                Injection::AllocOnSelect | Injection::AllocOnDecode => Vec::with_capacity(capacity),
                _ => Vec::new(),
            },
            grow: match injection {
                Injection::ReallocOnDecode => Vec::with_capacity(1),
                _ => Vec::new(),
            },
        }
    }
}

impl InterfaceProbe for BenchProbe {
    fn after_select(&mut self) {
        self.selects += 1;
        if self.injection == Injection::AllocOnSelect {
            self.sink.push(Box::new(black_box([0u8; 64])));
        }
    }

    fn after_decode(&mut self) {
        self.decodes += 1;
        match self.injection {
            Injection::AllocOnDecode => self.sink.push(Box::new(black_box([0u8; 64]))),
            Injection::ReallocOnDecode => {
                // Exact growth by one byte: every call after the first reallocates.
                self.grow.reserve_exact(1);
                self.grow.push(0);
            }
            _ => {}
        }
    }
}

struct ArmSpec<'a> {
    model_label: &'a str,
    route: Route,
    injection: Injection,
    model: Option<&'a Qwen35Model>,
    prompt: &'a str,
    profile: &'a Profile,
}

/// One request's region measurement.
#[derive(Clone, Debug)]
struct Sample {
    delta: AllocationDelta,
    region_tokens: usize,
    completed_tokens: usize,
    prompt_tokens: usize,
    opened: usize,
    consumed: usize,
    selects: usize,
    decodes: usize,
    initial_kv_capacity_floats: Option<usize>,
    token_ids: Vec<u32>,
    /// Per-iteration (alloc_calls, realloc_calls, bytes) inside the region.
    series: Vec<(u64, u64, u64)>,
}

/// Collects one snapshot per pushed token into storage sized before the request.
struct Recorder {
    points: Vec<(usize, AllocationSnapshot)>,
}

impl Recorder {
    fn with_capacity(max_new_tokens: usize) -> Self {
        Self {
            points: Vec::with_capacity(max_new_tokens + 4),
        }
    }

    fn on_event(&mut self, event: RawGenEvent) {
        if let RawGenEvent::RawToken { index } = event {
            self.points.push((index, AllocationSnapshot::capture()));
        }
    }
}

fn sample_from(rec: Recorder, run: RouteRun, probe: &BenchProbe) -> Sample {
    let mut delta = AllocationDelta {
        alloc_calls: 0,
        realloc_calls: 0,
        bytes_allocated: 0,
    };
    let mut region_tokens = 0usize;
    let mut series = Vec::new();
    let start = rec.points.iter().position(|(i, _)| *i == WARM_IN_TOKENS);
    if let (Some(s), Some(last)) = (start, rec.points.last()) {
        let first = &rec.points[s];
        if last.0 > first.0 {
            region_tokens = last.0 - first.0;
            delta = AllocationDelta {
                alloc_calls: last.1.alloc_calls - first.1.alloc_calls,
                realloc_calls: last.1.realloc_calls - first.1.realloc_calls,
                bytes_allocated: last.1.bytes_allocated - first.1.bytes_allocated,
            };
            for w in rec.points[s..].windows(2) {
                series.push((
                    w[1].1.alloc_calls - w[0].1.alloc_calls,
                    w[1].1.realloc_calls - w[0].1.realloc_calls,
                    w[1].1.bytes_allocated - w[0].1.bytes_allocated,
                ));
            }
        }
    }
    Sample {
        delta,
        region_tokens,
        completed_tokens: run.token_ids.len(),
        prompt_tokens: run.prompt_tokens,
        opened: run.opened,
        consumed: run.consumed,
        selects: probe.selects,
        decodes: probe.decodes,
        initial_kv_capacity_floats: run.initial_kv_capacity_floats,
        token_ids: run.token_ids,
        series,
    }
}

fn run_once(spec: &ArmSpec<'_>) -> Sample {
    let cfg = &spec.profile.cfg;
    let mut probe = BenchProbe::new(spec.injection, cfg.max_new_tokens + 4);
    let mut rec = Recorder::with_capacity(cfg.max_new_tokens);
    let run = {
        let mut on_event = |e: RawGenEvent| rec.on_event(e);
        match (spec.route, spec.model) {
            (Route::Streaming, Some(model)) => {
                run_qwen_cpu_streaming(model, spec.prompt, cfg, &mut on_event)
            }
            (Route::Probed, Some(model)) => {
                run_qwen_cpu_probed(model, spec.prompt, cfg, &mut probe, &mut on_event)
            }
            (Route::InterfaceOnly, _) => {
                run_interface_only(INTERFACE_ONLY_PROMPT_TOKENS, cfg, &mut probe, &mut on_event)
            }
            _ => panic!("route {} needs a model", spec.route.name()),
        }
    }
    .unwrap_or_else(|e| panic!("{} run failed: {e}", spec.route.name()));
    sample_from(rec, run, &probe)
}

/// The process must be quiet before an arm starts: no allocation by anything else.
fn assert_process_quiet() {
    let before = AllocationSnapshot::capture();
    std::thread::sleep(Duration::from_millis(25));
    let idle = AllocationSnapshot::delta_since(&before);
    assert_eq!(
        idle,
        AllocationDelta {
            alloc_calls: 0,
            realloc_calls: 0,
            bytes_allocated: 0
        },
        "the process is not quiet: {idle:?} allocated in 25 ms with no request running, so a \
         measurement now would be contaminated"
    );
}

/// The one-thread executor's worker runs the warm-up and every repeat; the caller is
/// parked in `install`. On the ambient pool the matmul splits spread work over several
/// workers, and which repeat a worker's first-use allocation lands in varies between
/// runs, so exact repeat equality refused healthy runs (issue #2008). With one thread
/// the splits are off and every `matmul_bt` takes its serial kernel, so each repeat
/// executes the same way and the counts describe single-threaded execution.
fn measure_unchecked(spec: &ArmSpec<'_>) -> Vec<Sample> {
    super::allocation_counts::counted_sample_executor().install(|| {
        assert_eq!(
            rayon::current_num_threads(),
            1,
            "the measurement must run on the one-thread executor, not on the ambient pool"
        );
        let _warm_up = run_once(spec);
        (0..REPEATS).map(|_| run_once(spec)).collect::<Vec<_>>()
    })
}

fn measure(spec: &ArmSpec<'_>) -> Vec<Sample> {
    assert_process_quiet();
    measure_unchecked(spec)
}

/// Accepts a set of repeats as one certified sample, or names why it refuses.
fn certify(samples: &[Sample], need_hooks: bool) -> Result<Sample, String> {
    let first = samples
        .first()
        .ok_or_else(|| "no samples: the group did not run".to_string())?;
    if samples.len() != REPEATS {
        return Err(format!("expected {REPEATS} repeats, got {}", samples.len()));
    }
    if first.region_tokens == 0 || first.completed_tokens == 0 {
        return Err(format!(
            "zero tokens in the region (completed {}, region {})",
            first.completed_tokens, first.region_tokens
        ));
    }
    if first.consumed + 1 != first.opened {
        return Err(format!(
            "the run did not go through the shared driver (opened {}, consumed {})",
            first.opened, first.consumed
        ));
    }
    if need_hooks && (first.selects < first.region_tokens || first.decodes < first.region_tokens) {
        return Err(format!(
            "interface hooks did not fire for the region (selects {}, decodes {}, region {})",
            first.selects, first.decodes, first.region_tokens
        ));
    }
    for (i, s) in samples.iter().enumerate().skip(1) {
        if s.delta != first.delta
            || s.region_tokens != first.region_tokens
            || s.completed_tokens != first.completed_tokens
            || s.token_ids != first.token_ids
            || s.series != first.series
        {
            return Err(format!(
                "unstable repeats at run {i}: run0=({},{},{}) tokens={} run{i}=({},{},{}) tokens={}",
                first.delta.alloc_calls,
                first.delta.realloc_calls,
                first.delta.bytes_allocated,
                first.region_tokens,
                s.delta.alloc_calls,
                s.delta.realloc_calls,
                s.delta.bytes_allocated,
                s.region_tokens,
            ));
        }
    }
    Ok(first.clone())
}

fn certify_or_panic(label: &str, samples: &[Sample], need_hooks: bool) -> Sample {
    certify(samples, need_hooks).unwrap_or_else(|why| panic!("REFUSE {label}: {why}"))
}

#[derive(Clone, Debug)]
enum Verdict {
    Pass,
    Fail(String),
    Refuse(String),
}

impl Verdict {
    fn kind(&self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail(_) => "FAIL",
            Verdict::Refuse(_) => "REFUSE",
        }
    }

    fn text(&self) -> String {
        match self {
            Verdict::Pass => "PASS".to_string(),
            Verdict::Fail(why) => format!("FAIL({why})"),
            Verdict::Refuse(why) => format!("REFUSE({why})"),
        }
    }
}

/// The incremental comparison: the candidate may not add one allocation call, one
/// reallocation call or one requested byte over the base, each counter on its own.
fn compare_incremental(base: &Sample, cand: &Sample) -> Verdict {
    if base.region_tokens == 0 || cand.region_tokens == 0 {
        return Verdict::Refuse("zero tokens".into());
    }
    if base.region_tokens != cand.region_tokens || base.completed_tokens != cand.completed_tokens {
        return Verdict::Refuse(format!(
            "token counts differ: base {}/{} candidate {}/{}",
            base.region_tokens, base.completed_tokens, cand.region_tokens, cand.completed_tokens
        ));
    }
    let mut added = Vec::new();
    let pairs = [
        (
            "alloc_calls",
            base.delta.alloc_calls,
            cand.delta.alloc_calls,
        ),
        (
            "realloc_calls",
            base.delta.realloc_calls,
            cand.delta.realloc_calls,
        ),
        (
            "bytes_allocated",
            base.delta.bytes_allocated,
            cand.delta.bytes_allocated,
        ),
    ];
    for (name, b, c) in pairs {
        if c > b {
            added.push(format!("{name} +{}", c - b));
        }
    }
    if added.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Fail(added.join(", "))
    }
}

/// The real consumer and the interface-only run must both stay within base. A
/// removal elsewhere in a model can cancel an interface addition in the first;
/// it cannot in the second.
fn combine(real: &Verdict, interface: &Verdict) -> Verdict {
    match (real, interface) {
        (Verdict::Refuse(a), _) | (_, Verdict::Refuse(a)) => Verdict::Refuse(a.clone()),
        (Verdict::Fail(a), Verdict::Fail(b)) => {
            Verdict::Fail(format!("real: {a}; interface-only: {b}"))
        }
        (Verdict::Fail(a), _) => Verdict::Fail(format!("real: {a}")),
        (_, Verdict::Fail(b)) => Verdict::Fail(format!("interface-only: {b}")),
        _ => Verdict::Pass,
    }
}

fn counts(s: &Sample) -> String {
    format!(
        "({},{},{})",
        s.delta.alloc_calls, s.delta.realloc_calls, s.delta.bytes_allocated
    )
}

fn control(name: &str, expected: &str, base: &Sample, cand: &Sample, verdict: &Verdict) {
    eprintln!(
        "CONTROL {name}: expected={expected} observed={} base={} candidate={} tokens={}",
        verdict.text(),
        counts(base),
        counts(cand),
        base.region_tokens
    );
    assert_eq!(
        verdict.kind(),
        expected,
        "control failed: {name} expected {expected} and observed {}",
        verdict.text()
    );
}

fn print_record(model_label: &str, route: Route, injection: Injection, profile: &str, s: &Sample) {
    let n = s.region_tokens as f64;
    let heading = allocation_report_heading(QWEN_CPU_DRIVER_ALLOCATIONS, model_label, route.name());
    let iterations_with_alloc = s.series.iter().filter(|p| p.0 > 0).count();
    let iterations_with_realloc = s.series.iter().filter(|p| p.1 > 0).count();
    // An iteration ending at the push of token k is series[k - WARM_IN_TOKENS - 1].
    // Each entry is (push index, alloc calls, realloc calls, requested bytes).
    let realloc_iterations: Vec<(usize, u64, u64, u64)> = s
        .series
        .iter()
        .enumerate()
        .filter(|(_, p)| p.1 > 0)
        .map(|(i, p)| (WARM_IN_TOKENS + 1 + i, p.0, p.1, p.2))
        .collect();
    let quiet_bytes = s.series.iter().filter(|p| p.1 == 0).map(|p| p.2);
    let (quiet_min, quiet_max) =
        quiet_bytes.fold((u64::MAX, 0u64), |(lo, hi), b| (lo.min(b), hi.max(b)));
    let capacity = s
        .initial_kv_capacity_floats
        .map_or("n/a".to_string(), |c| c.to_string());
    eprintln!(
        "\n{heading}\n\
         variant={} profile={profile}\n\
         prompt_tokens={} completed_tokens={} tokens={} region_from_push={WARM_IN_TOKENS} \
         initial_kv_capacity_floats={capacity} driver_opened={} driver_consumed={}\n\
         alloc_calls_total={}\n\
         realloc_calls_total={}\n\
         bytes_allocated_total={}\n\
         allocations_per_token={:.2}\n\
         reallocations_per_token={:.2}\n\
         bytes_allocated_per_token={:.0}\n\
         iterations_with_alloc={iterations_with_alloc} iterations_with_realloc={iterations_with_realloc} \
         realloc_iterations={realloc_iterations:?}\n\
         bytes_in_iterations_without_realloc_min={quiet_min} max={quiet_max} repeats={REPEATS}",
        injection.name(),
        s.prompt_tokens,
        s.completed_tokens,
        s.region_tokens,
        s.opened,
        s.consumed,
        s.delta.alloc_calls,
        s.delta.realloc_calls,
        s.delta.bytes_allocated,
        s.delta.alloc_calls as f64 / n,
        s.delta.realloc_calls as f64 / n,
        s.delta.bytes_allocated as f64 / n,
    );
    eprintln!(
        "record model={model_label} route={} variant={} profile={profile} prompt_tokens={} \
         completed_tokens={} tokens={} initial_kv_capacity_floats={capacity} alloc_calls={} \
         realloc_calls={} bytes_requested={}",
        route.name(),
        injection.name(),
        s.prompt_tokens,
        s.completed_tokens,
        s.region_tokens,
        s.delta.alloc_calls,
        s.delta.realloc_calls,
        s.delta.bytes_allocated,
    );
}

fn arm<'a>(
    model_label: &'a str,
    route: Route,
    injection: Injection,
    model: Option<&'a Qwen35Model>,
    prompt: &'a str,
    profile: &'a Profile,
) -> ArmSpec<'a> {
    ArmSpec {
        model_label,
        route,
        injection,
        model,
        prompt,
        profile,
    }
}

// Every group the suite is designed to produce is listed by `required_groups`. A group
// that silently stops running leaves the ledger short and the instrument refuses to
// report.
static CERTIFIED_GROUPS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Every group the suite is designed to produce. A group that silently stops running
/// leaves the ledger short and the instrument refuses to report.
const GREEDY: &str = "greedy_t0_rp1.0_seed7";
const SAMPLED: &str = "sampled_t0.7_k50_p0.9_rp1.1_seed7";

fn required_groups(
    model: &str,
    with_interface_only: bool,
    with_offset: bool,
    sampled_full: bool,
) -> Vec<String> {
    let mut keys = Vec::new();
    if with_interface_only {
        for variant in [
            "unchanged",
            "retained_alloc_in_select",
            "retained_realloc_in_decode",
        ] {
            keys.push(format!("interface_only/interface_only/{variant}/{GREEDY}"));
        }
    }
    let mut arms = vec![
        ("qwen_cpu_streaming", "unchanged"),
        ("qwen_cpu_probed", "unchanged"),
        ("qwen_cpu_probed", "retained_alloc_in_select"),
        ("qwen_cpu_probed", "retained_realloc_in_decode"),
    ];
    if with_offset {
        arms.push(("qwen_cpu_probed", "retained_alloc_in_decode"));
    }
    for (route, variant) in arms {
        keys.push(format!("{model}/{route}/{variant}/{GREEDY}"));
    }
    if sampled_full {
        for (route, variant) in [
            ("qwen_cpu_streaming", "unchanged"),
            ("qwen_cpu_probed", "unchanged"),
            ("qwen_cpu_probed", "retained_alloc_in_select"),
            ("qwen_cpu_probed", "retained_realloc_in_decode"),
        ] {
            keys.push(format!("{model}/{route}/{variant}/{SAMPLED}"));
        }
    } else {
        keys.push(format!("{model}/qwen_cpu_streaming/unchanged/{SAMPLED}"));
    }
    keys
}

fn require_groups(required: &[String]) {
    let ledger = CERTIFIED_GROUPS.lock().expect("group ledger poisoned");
    let missing: Vec<&String> = required
        .iter()
        .filter(|key| !ledger.iter().any(|seen| seen == *key))
        .collect();
    assert!(
        missing.is_empty(),
        "REFUSE: groups the suite is designed to run produced no certified sample: {missing:?}"
    );
}

fn measure_certified(spec: &ArmSpec<'_>) -> Sample {
    let need_hooks = spec.route != Route::Streaming;
    let samples = measure(spec);
    let sample = certify_or_panic(
        &format!(
            "{}/{}/{}",
            spec.model_label,
            spec.route.name(),
            spec.injection.name()
        ),
        &samples,
        need_hooks,
    );
    print_record(
        spec.model_label,
        spec.route,
        spec.injection,
        spec.profile.name,
        &sample,
    );
    CERTIFIED_GROUPS
        .lock()
        .expect("group ledger poisoned")
        .push(format!(
            "{}/{}/{}/{}",
            spec.model_label,
            spec.route.name(),
            spec.injection.name(),
            spec.profile.name
        ));
    sample
}

/// Each declared entry point moves its own counter, and an idle region moves none.
fn entry_point_controls() {
    assert_zeroed_allocation_is_counted();

    let before = AllocationSnapshot::capture();
    let boxed = Box::new(black_box(0xA5A5_A5A5_A5A5_A5A5u64));
    let d = AllocationSnapshot::delta_since(&before);
    black_box(&boxed);
    assert!(
        d.alloc_calls >= 1 && d.bytes_allocated >= 8 && d.realloc_calls == 0,
        "control failed: alloc entry point did not move alloc_calls alone: {d:?}"
    );
    eprintln!("CONTROL entry_point_alloc: expected=alloc_calls>=1,realloc_calls=0 observed={d:?}");

    let before = AllocationSnapshot::capture();
    let zeroed = vec![0u8; 65_537];
    let d = AllocationSnapshot::delta_since(&before);
    black_box(&zeroed);
    assert!(
        d.alloc_calls >= 1 && d.bytes_allocated >= 65_537 && d.realloc_calls == 0,
        "control failed: a zeroed allocation was not observed: {d:?}"
    );
    eprintln!(
        "CONTROL entry_point_alloc_zeroed: expected=alloc_calls>=1,bytes>=65537,realloc_calls=0 observed={d:?}"
    );

    let mut grown: Vec<u8> = Vec::with_capacity(16);
    grown.push(1);
    let before = AllocationSnapshot::capture();
    grown.reserve_exact(4096);
    let d = AllocationSnapshot::delta_since(&before);
    black_box(&grown);
    assert!(
        d.realloc_calls == 1 && d.alloc_calls == 0 && d.bytes_allocated >= 4097,
        "control failed: a reallocation was not observed as a realloc: {d:?}"
    );
    eprintln!("CONTROL entry_point_realloc: expected=realloc_calls=1,alloc_calls=0 observed={d:?}");

    let before = AllocationSnapshot::capture();
    let idle = AllocationSnapshot::delta_since(&before);
    assert_eq!(
        idle,
        AllocationDelta {
            alloc_calls: 0,
            realloc_calls: 0,
            bytes_allocated: 0
        },
        "control failed: an empty region moved a counter"
    );
    eprintln!("CONTROL entry_point_idle_region: expected=0,0,0 observed={idle:?}");
}

struct InterfaceEvidence {
    clean: Sample,
    clean_again: Sample,
    alloc_in_select: Sample,
    realloc_in_decode: Sample,
}

/// The bounded interface-only run: the shared driver over a session that does no
/// model work, unchanged and with each retained injection.
fn interface_only_evidence(profile: &Profile) -> InterfaceEvidence {
    eprintln!("\n# interface-only control: the shared driver over a session with no model work");
    let label = "interface_only";
    let clean = measure_certified(&arm(
        label,
        Route::InterfaceOnly,
        Injection::None,
        None,
        "",
        profile,
    ));
    let clean_again = measure_certified(&arm(
        label,
        Route::InterfaceOnly,
        Injection::None,
        None,
        "",
        profile,
    ));
    let alloc_in_select = measure_certified(&arm(
        label,
        Route::InterfaceOnly,
        Injection::AllocOnSelect,
        None,
        "",
        profile,
    ));
    let realloc_in_decode = measure_certified(&arm(
        label,
        Route::InterfaceOnly,
        Injection::ReallocOnDecode,
        None,
        "",
        profile,
    ));

    let v = compare_incremental(&clean, &clean_again);
    control(
        "interface_only_unchanged_passes",
        "PASS",
        &clean,
        &clean_again,
        &v,
    );

    let v = compare_incremental(&clean, &alloc_in_select);
    control(
        "interface_only_retained_alloc_fails",
        "FAIL",
        &clean,
        &alloc_in_select,
        &v,
    );
    let t = clean.region_tokens as u64;
    assert_eq!(
        alloc_in_select.delta.alloc_calls - clean.delta.alloc_calls,
        t,
        "control failed: the injected allocation is not attributed one per token"
    );
    assert_eq!(
        alloc_in_select.delta.bytes_allocated - clean.delta.bytes_allocated,
        64 * t,
        "control failed: the injected bytes are not 64 per token"
    );

    let v = compare_incremental(&clean, &realloc_in_decode);
    control(
        "interface_only_retained_realloc_fails",
        "FAIL",
        &clean,
        &realloc_in_decode,
        &v,
    );
    assert_eq!(
        realloc_in_decode.delta.realloc_calls - clean.delta.realloc_calls,
        t,
        "control failed: the injected reallocation is not attributed one per token"
    );
    assert_eq!(
        realloc_in_decode.delta.alloc_calls, clean.delta.alloc_calls,
        "control failed: a retained reallocation moved alloc_calls"
    );

    InterfaceEvidence {
        clean,
        clean_again,
        alloc_in_select,
        realloc_in_decode,
    }
}

/// The route suite on one model and profile: the clean base, the unchanged probe,
/// and the two retained injections, each judged together with the interface-only run.
fn route_suite(
    model_label: &str,
    model: &Qwen35Model,
    prompt: &str,
    profile: &Profile,
    iface: &InterfaceEvidence,
) -> Sample {
    eprintln!(
        "\n# route suite: model={model_label} profile={}",
        profile.name
    );
    let base = measure_certified(&arm(
        model_label,
        Route::Streaming,
        Injection::None,
        Some(model),
        prompt,
        profile,
    ));

    // The probed route must be the production route plus a decorator, nothing else.
    let unchanged = measure_certified(&arm(
        model_label,
        Route::Probed,
        Injection::None,
        Some(model),
        prompt,
        profile,
    ));
    assert_eq!(
        unchanged.token_ids, base.token_ids,
        "control failed: the probed route generated different tokens from the production route"
    );
    let v = combine(
        &compare_incremental(&base, &unchanged),
        &compare_incremental(&iface.clean, &iface.clean_again),
    );
    control(
        "unchanged_warmed_path_passes",
        "PASS",
        &base,
        &unchanged,
        &v,
    );
    assert_eq!(
        counts(&base),
        counts(&unchanged),
        "control failed: the probed route does not reproduce the production route's counts"
    );

    let t = base.region_tokens as u64;

    let alloc = measure_certified(&arm(
        model_label,
        Route::Probed,
        Injection::AllocOnSelect,
        Some(model),
        prompt,
        profile,
    ));
    let v = combine(
        &compare_incremental(&base, &alloc),
        &compare_incremental(&iface.clean, &iface.alloc_in_select),
    );
    control(
        "retained_alloc_in_interface_fails",
        "FAIL",
        &base,
        &alloc,
        &v,
    );
    assert_eq!(
        alloc.delta.alloc_calls - base.delta.alloc_calls,
        t,
        "control failed: the injected allocation is not attributed one per token"
    );
    assert_eq!(
        alloc.delta.bytes_allocated - base.delta.bytes_allocated,
        64 * t,
        "control failed: the injected bytes are not 64 per token"
    );

    let realloc = measure_certified(&arm(
        model_label,
        Route::Probed,
        Injection::ReallocOnDecode,
        Some(model),
        prompt,
        profile,
    ));
    let v = combine(
        &compare_incremental(&base, &realloc),
        &compare_incremental(&iface.clean, &iface.realloc_in_decode),
    );
    control(
        "retained_realloc_in_interface_fails",
        "FAIL",
        &base,
        &realloc,
        &v,
    );
    assert_eq!(
        realloc.delta.realloc_calls - base.delta.realloc_calls,
        t,
        "control failed: the injected reallocation is not attributed one per token"
    );
    assert_eq!(
        realloc.delta.alloc_calls, base.delta.alloc_calls,
        "control failed: a retained reallocation moved alloc_calls"
    );

    base
}

/// An addition in the interface cancelled by an unrelated removal on the real
/// consumer: the real comparison alone is blind to it, the combined one is not.
fn offset_control(
    model_label: &str,
    model: &Qwen35Model,
    prompt: &str,
    profile: &Profile,
    iface: &InterfaceEvidence,
) {
    let base_with_unrelated = measure_certified(&arm(
        model_label,
        Route::Probed,
        Injection::AllocOnDecode,
        Some(model),
        prompt,
        profile,
    ));
    let head = measure_certified(&arm(
        model_label,
        Route::Probed,
        Injection::AllocOnSelect,
        Some(model),
        prompt,
        profile,
    ));
    let real_only = compare_incremental(&base_with_unrelated, &head);
    control(
        "offset_real_consumer_alone_is_blind",
        "PASS",
        &base_with_unrelated,
        &head,
        &real_only,
    );
    let combined = combine(
        &real_only,
        &compare_incremental(&iface.clean, &iface.alloc_in_select),
    );
    control(
        "offset_caught_by_interface_only_control",
        "FAIL",
        &base_with_unrelated,
        &head,
        &combined,
    );
}

/// Contamination, zero tokens, a missing group and a missing route marker are refused.
fn refusal_controls(model_label: &str, model: &Qwen35Model, prompt: &str, profile: &Profile) {
    let spec = arm(
        model_label,
        Route::Streaming,
        Injection::None,
        Some(model),
        prompt,
        profile,
    );

    let stop = AtomicBool::new(false);
    let noisy = std::thread::scope(|scope| {
        let noise = scope.spawn(|| {
            while !stop.load(Ordering::Relaxed) {
                black_box(Box::new(black_box([0u8; 32])));
            }
        });
        let samples = measure_unchecked(&spec);
        stop.store(true, Ordering::Relaxed);
        noise.join().expect("the noise thread panicked");
        samples
    });
    let refusal = certify(&noisy, false);
    eprintln!(
        "CONTROL concurrent_noise_is_refused: expected=REFUSE(unstable) observed={}",
        match &refusal {
            Ok(_) => "ACCEPTED".to_string(),
            Err(why) => format!("REFUSE({why})"),
        }
    );
    assert!(
        matches!(&refusal, Err(why) if why.contains("unstable")),
        "control failed: contaminated repeats were certified"
    );

    let mut short = Profile {
        name: profile.name,
        cfg: profile.cfg.clone(),
    };
    short.cfg.max_new_tokens = WARM_IN_TOKENS;
    let zero = measure(&arm(
        model_label,
        Route::Streaming,
        Injection::None,
        Some(model),
        prompt,
        &short,
    ));
    let refusal = certify(&zero, false);
    eprintln!(
        "CONTROL zero_tokens_is_refused: expected=REFUSE(zero tokens) observed={}",
        match &refusal {
            Ok(_) => "ACCEPTED".to_string(),
            Err(why) => format!("REFUSE({why})"),
        }
    );
    assert!(
        matches!(&refusal, Err(why) if why.contains("zero tokens")),
        "control failed: a zero-token region was certified"
    );

    let refusal = certify(&[], false);
    eprintln!(
        "CONTROL missing_group_is_refused: expected=REFUSE(no samples) observed={}",
        match &refusal {
            Ok(_) => "ACCEPTED".to_string(),
            Err(why) => format!("REFUSE({why})"),
        }
    );
    assert!(
        matches!(&refusal, Err(why) if why.contains("no samples")),
        "control failed: an empty group was certified"
    );

    let streaming = measure(&spec);
    let refusal = certify(&streaming, true);
    eprintln!(
        "CONTROL missing_interface_hooks_are_refused: expected=REFUSE(hooks) observed={}",
        match &refusal {
            Ok(_) => "ACCEPTED".to_string(),
            Err(why) => format!("REFUSE({why})"),
        }
    );
    assert!(
        matches!(&refusal, Err(why) if why.contains("hooks did not fire")),
        "control failed: a route with no interface hooks was certified as a probed route"
    );
}

fn locate_checkpoint() -> Option<std::path::PathBuf> {
    for var in [
        "LATTICE_CPU_GREEDY_MODEL_DIR",
        "LATTICE_MODEL_DIR",
        "LATTICE_INFERENCE_MODEL_DIR",
    ] {
        if let Ok(dir) = std::env::var(var) {
            let dir = std::path::PathBuf::from(dir);
            assert!(
                dir.is_dir(),
                "{var} names {dir:?}, which is not a directory; refusing to fall back to a skip"
            );
            return Some(dir);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let dir = std::path::PathBuf::from(home).join(".lattice/models/qwen3.5-0.8b");
    dir.is_dir().then_some(dir)
}

pub(super) fn run() {
    eprintln!(
        "\n# {QWEN_CPU_DRIVER_ALLOCATIONS}: counted entry points: alloc, alloc_zeroed (default \
         method, forwards to alloc), realloc; dealloc tallied separately, not gated"
    );
    entry_point_controls();

    let model = tiny_zero_model();
    let synthetic = "synthetic_tiny_zero";
    let greedy = greedy_profile(SYNTHETIC_TOKENS, Vec::new());
    let sampled = sampled_profile(SYNTHETIC_TOKENS, Vec::new());

    let iface = interface_only_evidence(&greedy);

    let greedy_base = route_suite(synthetic, &model, SYNTHETIC_PROMPT, &greedy, &iface);
    route_suite(synthetic, &model, SYNTHETIC_PROMPT, &sampled, &iface);
    offset_control(synthetic, &model, SYNTHETIC_PROMPT, &greedy, &iface);
    refusal_controls(synthetic, &model, SYNTHETIC_PROMPT, &greedy);
    require_groups(&required_groups(synthetic, true, true, true));
    eprintln!(
        "\nR01 SYNTHETIC: INSTRUMENT VALID. warm-region base (greedy) = {} over {} tokens",
        counts(&greedy_base),
        greedy_base.region_tokens
    );

    match locate_checkpoint() {
        None => eprintln!(
            "\nR01 REAL-CHECKPOINT: SKIPPED. No Qwen3.5-0.8B directory found (set \
             LATTICE_CPU_GREEDY_MODEL_DIR, or place it under ~/.lattice/models/qwen3.5-0.8b). \
             A skip is not a pass."
        ),
        Some(dir) => {
            let real = Qwen35Model::from_safetensors(&dir)
                .unwrap_or_else(|e| panic!("loading {dir:?} failed: {e}"));
            let label = "real_qwen3.5-0.8b";
            let stops = vec![151_645];
            let greedy = greedy_profile(REAL_TOKENS, stops.clone());
            let sampled = sampled_profile(REAL_TOKENS, stops);
            let base = route_suite(label, &real, REAL_PROMPT, &greedy, &iface);
            measure_certified(&arm(
                label,
                Route::Streaming,
                Injection::None,
                Some(&real),
                REAL_PROMPT,
                &sampled,
            ));
            require_groups(&required_groups(label, false, false, false));
            eprintln!(
                "\nR01 REAL-CHECKPOINT: INSTRUMENT VALID on {dir:?}. warm-region base (greedy) = {} over {} tokens",
                counts(&base),
                base.region_tokens
            );
        }
    }
}
