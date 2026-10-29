//! Synthetic CPU BatchWorker/FIFO scheduling and chunked-prefill baseline.
//!
//! The callback returns fixed logits and does not run a model or allocate KV pages
//! for worker sequences. The timed interval includes callback/trace collection and
//! worker steps, but excludes worker construction, request submission and validation.

use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::batch::{BatchConfig, BatchWorker, FinishReason, InferenceRequest, SeqId};
use lattice_inference::kv_cache::{EvictionPolicy, PagedKVCacheConfig};
use lattice_inference::sampling::SamplingConfig;

const OUTPUT_TOKENS: usize = 4;
const WINNER: u32 = 7;

#[derive(Clone, Copy)]
struct Case {
    sequences: usize,
    chunk: usize,
    staggered: bool,
}

impl Case {
    fn prompt_len(self, index: usize) -> usize {
        if self.staggered && index == 0 {
            1
        } else {
            self.chunk * 2 + 1
        }
    }
}

#[derive(Debug)]
struct Call {
    id: SeqId,
    iteration: usize,
    start: usize,
    len: usize,
    token: u32,
    uniform: bool,
}

#[derive(Debug)]
struct Event {
    id: SeqId,
    token: u32,
    finished: bool,
    reason: Option<FinishReason>,
}

struct Fixture {
    worker: BatchWorker,
    ids: Vec<SeqId>,
    trace: Trace,
}

struct Trace {
    calls: Vec<Call>,
    events: Vec<Event>,
    steps: usize,
    idle: bool,
}

fn setup(case: Case) -> Fixture {
    let mut worker = BatchWorker::try_new(
        BatchConfig {
            max_batch_size: case.sequences,
            max_seq_len: 64,
            chunk_size: case.chunk,
            prefill_reserve_pages: 2,
        },
        PagedKVCacheConfig {
            page_size: 256,
            max_pages: 32,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 1,
            eviction: EvictionPolicy::None,
        },
        2,
        1,
        None,
    )
    .expect("valid benchmark worker");
    let mut ids = Vec::with_capacity(case.sequences);
    for index in 0..case.sequences {
        let id = worker
            .submit(InferenceRequest {
                prompt_ids: vec![10 + index as u32; case.prompt_len(index)],
                sampling: SamplingConfig::greedy(),
                lora_adapter: None,
                max_new_tokens: OUTPUT_TOKENS,
            })
            .expect("benchmark request admitted");
        ids.push(id);
    }
    Fixture {
        worker,
        ids,
        trace: Trace {
            calls: Vec::with_capacity(case.sequences * (3 + OUTPUT_TOKENS - 1)),
            events: Vec::with_capacity(case.sequences * OUTPUT_TOKENS),
            steps: 0,
            idle: false,
        },
    }
}

fn drive(fixture: &mut Fixture) {
    let worker = &mut fixture.worker;
    let trace = &mut fixture.trace;
    // Three prefill chunks plus the decode budget, with slack for admission.
    for iteration in 0..(3 + OUTPUT_TOKENS + 2) {
        if worker.is_idle() {
            break;
        }
        let result = worker.step(|input, _pool| {
            let token = input.token_ids.first().copied().unwrap_or(u32::MAX);
            trace.calls.push(Call {
                id: input.seq_id,
                iteration,
                start: input.start_pos,
                len: input.token_ids.len(),
                token,
                uniform: input.token_ids.iter().all(|&value| value == token),
            });
            let mut logits = vec![-1.0; 8];
            logits[WINNER as usize] = 1.0;
            logits
        });
        trace.events.extend(result.into_iter().map(|event| Event {
            id: event.seq_id,
            token: event.token_id,
            finished: event.finished,
            reason: event.finish_reason,
        }));
        trace.steps += 1;
    }
    trace.idle = worker.is_idle();
}

fn validate(case: Case, ids: &[SeqId], trace: &Trace) {
    assert!(trace.idle, "worker failed to finish within the bound");
    assert_eq!(ids.len(), case.sequences);
    let mut total_expected_calls = 0;
    for (index, &id) in ids.iter().enumerate() {
        assert!(!ids[..index].contains(&id), "duplicate submitted id");
        let prompt_len = case.prompt_len(index);
        let calls: Vec<_> = trace.calls.iter().filter(|call| call.id == id).collect();
        let mut expected = Vec::new();
        for start in (0..prompt_len).step_by(case.chunk) {
            expected.push((
                start,
                (prompt_len - start).min(case.chunk),
                10 + index as u32,
            ));
        }
        for decode in 0..OUTPUT_TOKENS - 1 {
            expected.push((prompt_len + decode, 1, WINNER));
        }
        total_expected_calls += expected.len();
        assert_eq!(calls.len(), expected.len(), "callback count for {id}");
        for (call, &(start, len, token)) in calls.iter().zip(&expected) {
            assert_eq!((call.start, call.len, call.token), (start, len, token));
            assert!(call.uniform, "unexpected input token within chunk");
        }
        let events: Vec<_> = trace.events.iter().filter(|event| event.id == id).collect();
        assert_eq!(events.len(), OUTPUT_TOKENS, "output count for {id}");
        for (position, event) in events.iter().enumerate() {
            assert_eq!(event.token, WINNER);
            let last = position + 1 == OUTPUT_TOKENS;
            assert_eq!(event.finished, last);
            assert_eq!(event.reason, last.then_some(FinishReason::MaxLength));
        }
    }
    assert_eq!(trace.calls.len(), total_expected_calls);
    assert_eq!(trace.events.len(), case.sequences * OUTPUT_TOKENS);
    let initial_ids: Vec<_> = trace
        .calls
        .iter()
        .filter(|call| call.iteration == 0)
        .map(|call| call.id)
        .collect();
    assert_eq!(initial_ids.as_slice(), ids);
    if case.staggered {
        let mixed = trace.calls.iter().any(|short| {
            short.id == ids[0]
                && short.start >= case.prompt_len(0)
                && trace.calls.iter().any(|long| {
                    long.id == ids[1]
                        && long.iteration == short.iteration
                        && long.start < case.prompt_len(1)
                })
        });
        assert!(mixed, "staggered case did not overlap prefill and decode");
    }
}

fn bench_batch_worker_fifo(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_worker_fifo");
    for chunk in [4, 8] {
        for sequences in [1, 2, 4, 8, 16] {
            for staggered in [false, true] {
                if staggered && sequences == 1 {
                    continue;
                }
                let case = Case {
                    sequences,
                    chunk,
                    staggered,
                };
                let scenario = if staggered { "staggered" } else { "uniform" };
                group.throughput(Throughput::Elements((sequences * OUTPUT_TOKENS) as u64));
                group.bench_with_input(
                    BenchmarkId::new(scenario, format!("n{sequences}_chunk{chunk}")),
                    &case,
                    |b, &case| {
                        // Criterion calls this closure only after matching its filter.
                        let mut witness = setup(case);
                        drive(&mut witness);
                        validate(case, &witness.ids, &witness.trace);
                        eprintln!(
                            "BATCH_WORKER_WITNESS scenario={scenario} n={} chunk={} outputs={} steps={}",
                            case.sequences,
                            case.chunk,
                            witness.trace.events.len(),
                            witness.trace.steps
                        );
                        b.iter_custom(|iters| {
                            let mut elapsed = Duration::ZERO;
                            for _ in 0..iters {
                                let mut fixture = setup(case);
                                let start = Instant::now();
                                drive(&mut fixture);
                                elapsed += start.elapsed();
                                validate(case, &fixture.ids, &fixture.trace);
                                black_box(&fixture.trace);
                            }
                            elapsed
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_batch_worker_fifo);
criterion_main!(benches);
