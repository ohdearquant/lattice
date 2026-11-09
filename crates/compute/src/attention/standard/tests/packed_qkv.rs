use super::super::{
    AttentionBuffers, AttentionProjections, BatchedAttentionScratch, multi_head_attention_batched,
    multi_head_attention_in_place,
};
use crate::lora_hook::{LoraHook, NoopLoraHook};
use lattice_weights::f32_weights::{Tensor1D, Tensor2D};
use std::sync::Mutex;

mod legacy;

const HIDDEN: usize = 12;
const HEADS: usize = 3;
const HEAD_DIM: usize = 4;
const MAX_TOTAL: usize = 16;
const MAX_SEQ: usize = 9;
const MODULES: [&str; 4] = ["query", "key", "value", "attn_output"];
const SCALES: [f32; 4] = [0.5, -0.75, 1.25, -0.5];
const SHIFTS: [usize; 4] = [0, 1, 5, 7];

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn sentinels(length: usize, tag: u32) -> Vec<f32> {
    (0..length)
        .map(|index| match index % 3 {
            0 => -0.0,
            1 => f32::from_bits(0x7fc0_0000 | tag),
            _ => -(700.0 + tag as f32),
        })
        .collect()
}

struct Fixture {
    weights: [Vec<f32>; 4],
    biases: [Vec<f32>; 4],
    fused_weight: Vec<f32>,
    fused_bias: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        let weights: [Vec<f32>; 4] = std::array::from_fn(|projection| {
            let mut matrix = vec![0.0; HIDDEN * HIDDEN];
            for channel in 0..HIDDEN {
                matrix[channel * HIDDEN + (channel + SHIFTS[projection]) % HIDDEN] =
                    SCALES[projection];
            }
            matrix
        });
        let biases: [Vec<f32>; 4] = std::array::from_fn(|projection| {
            (0..HIDDEN)
                .map(|channel| (projection * HIDDEN + channel + 1) as f32 / 64.0)
                .collect()
        });
        let fused_weight = weights[..3].iter().flatten().copied().collect();
        let fused_bias = biases[..3].iter().flatten().copied().collect();
        Self {
            weights,
            biases,
            fused_weight,
            fused_bias,
        }
    }

    fn projections(&self) -> AttentionProjections<'_> {
        let weight = |projection: usize| Tensor2D {
            data: self.weights[projection].as_slice(),
            rows: HIDDEN,
            cols: HIDDEN,
        };
        let bias = |projection: usize| Tensor1D {
            data: self.biases[projection].as_slice(),
            len: HIDDEN,
        };
        AttentionProjections {
            query_weight: weight(0),
            query_bias: bias(0),
            key_weight: weight(1),
            key_bias: bias(1),
            value_weight: weight(2),
            value_bias: bias(2),
            output_weight: weight(3),
            output_bias: bias(3),
        }
    }

    fn projected(&self, projection: usize, input: &[f32], channel: usize) -> f32 {
        input[(channel + SHIFTS[projection]) % HIDDEN] * SCALES[projection]
            + self.biases[projection][channel]
    }
}

fn packed_input(lengths: &[usize], epoch: usize) -> (Vec<usize>, Vec<f32>) {
    let mut offsets = vec![0];
    let mut input = Vec::new();
    let mut total = 0;
    for (segment, &length) in lengths.iter().enumerate() {
        for token in 0..length {
            for channel in 0..HIDDEN {
                let code = (segment * 19 + token * 11 + channel * 7 + epoch * 5) % 67;
                input.push((code as f32 - 31.0) / 32.0);
            }
        }
        total += length;
        offsets.push(total);
    }
    (offsets, input)
}

struct Storage {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    qkv: Vec<f32>,
    concat: Vec<f32>,
    output: Vec<f32>,
    q_head: Vec<f32>,
    k_head: Vec<f32>,
    v_all_t: Vec<f32>,
    scores_head: Vec<f32>,
    context_head: Vec<f32>,
}

impl Storage {
    fn new() -> Self {
        Self {
            q: sentinels(MAX_TOTAL * HIDDEN, 1),
            k: sentinels(MAX_TOTAL * HIDDEN, 2),
            v: sentinels(MAX_TOTAL * HIDDEN, 3),
            qkv: vec![-704.0; MAX_TOTAL * 3 * HIDDEN],
            concat: vec![-705.0; MAX_TOTAL * HIDDEN],
            output: vec![-706.0; MAX_TOTAL * HIDDEN],
            q_head: vec![f32::NAN; MAX_SEQ * HEAD_DIM],
            k_head: vec![f32::NAN; MAX_SEQ * HEAD_DIM],
            v_all_t: vec![f32::NAN; MAX_SEQ * HIDDEN],
            scores_head: vec![f32::NAN; MAX_SEQ * MAX_SEQ],
            context_head: vec![f32::NAN; MAX_SEQ * HEAD_DIM],
        }
    }

    fn arrays(&self) -> [&[f32]; 11] {
        [
            &self.q,
            &self.k,
            &self.v,
            &self.qkv,
            &self.concat,
            &self.output,
            &self.q_head,
            &self.k_head,
            &self.v_all_t,
            &self.scores_head,
            &self.context_head,
        ]
    }

    fn snapshot(&self) -> [Vec<u32>; 11] {
        self.arrays().map(bits)
    }

    fn run(
        &mut self,
        fixture: &Fixture,
        input: &[f32],
        offsets: &[usize],
        borrowed: bool,
        frozen: bool,
        hook: &dyn LoraHook,
        layer: usize,
    ) {
        let scratch = borrowed.then_some(BatchedAttentionScratch {
            q_head: &mut self.q_head,
            k_head: &mut self.k_head,
            v_all_t: &mut self.v_all_t,
            scores_head: &mut self.scores_head,
            context_head: &mut self.context_head,
        });
        let attention = if frozen {
            legacy::legacy_multi_head_attention_batched
        } else {
            multi_head_attention_batched
        };
        attention(
            input,
            &fixture.projections(),
            &fixture.fused_weight,
            &fixture.fused_bias,
            offsets,
            HIDDEN,
            HEADS,
            HEAD_DIM,
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.qkv,
            &mut self.concat,
            &mut self.output,
            scratch,
            hook,
            layer,
        );
    }

    fn assert_tails(&self, before: &[Vec<u32>; 11], total: usize, longest: usize, borrowed: bool) {
        let used = total * HIDDEN;
        let head_used = if borrowed { longest * HEAD_DIM } else { 0 };
        let v_used = if borrowed { longest * HIDDEN } else { 0 };
        let score_used = if borrowed { longest * longest } else { 0 };
        let active = [
            used,
            used,
            used,
            used * 3,
            used,
            used,
            head_used,
            head_used,
            v_used,
            score_used,
            head_used,
        ];
        for (index, values) in self.arrays().iter().enumerate() {
            assert_eq!(
                bits(&values[active[index]..]),
                before[index][active[index]..],
                "inactive scratch tail changed at buffer {index}"
            );
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Event {
    Activity {
        layer: usize,
        module: String,
    },
    Apply {
        layer: usize,
        module: String,
        input: Vec<u32>,
        before: Vec<u32>,
        after: Vec<u32>,
    },
}

struct RecordingHook {
    active: [bool; 4],
    events: Mutex<Vec<Event>>,
}

impl RecordingHook {
    fn new(active: [bool; 4]) -> Self {
        Self {
            active,
            events: Mutex::new(Vec::new()),
        }
    }

    fn projection(module: &str) -> usize {
        MODULES.iter().position(|&name| name == module).unwrap()
    }

    fn adjusted(projection: usize, input: &[f32], channel: usize, value: f32) -> f32 {
        value + (projection + 1) as f32 / 32.0 + input[(channel + 3) % HIDDEN] / 16.0
    }

    fn assert_trace(
        &self,
        fixture: &Fixture,
        input: &[f32],
        concat: &[f32],
        layer: usize,
        activity: bool,
    ) {
        let events = self.events.lock().unwrap();
        let mut cursor = 0;
        for (projection, module) in MODULES.iter().enumerate() {
            if activity {
                assert_eq!(
                    events.get(cursor),
                    Some(&Event::Activity {
                        layer,
                        module: (*module).to_owned(),
                    }),
                    "projection activity must occur exactly once in projection order"
                );
                cursor += 1;
            }
            if !self.active[projection] {
                continue;
            }
            let source = if projection == 3 { concat } else { input };
            for row in source.chunks_exact(HIDDEN) {
                let before: Vec<f32> = (0..HIDDEN)
                    .map(|channel| fixture.projected(projection, row, channel))
                    .collect();
                let after: Vec<f32> = before
                    .iter()
                    .enumerate()
                    .map(|(channel, &value)| Self::adjusted(projection, row, channel, value))
                    .collect();
                assert_eq!(
                    events.get(cursor),
                    Some(&Event::Apply {
                        layer,
                        module: (*module).to_owned(),
                        input: bits(row),
                        before: bits(&before),
                        after: bits(&after),
                    }),
                    "hook row order, input, or projected values changed"
                );
                cursor += 1;
            }
        }
        assert_eq!(events.len(), cursor, "unexpected extra hook events");
    }
}

impl LoraHook for RecordingHook {
    fn is_active(&self, layer: usize, module: &str) -> bool {
        let mut events = self.events.lock().unwrap();
        events.push(Event::Activity {
            layer,
            module: module.to_owned(),
        });
        let calls = events
            .iter()
            .filter(|event| matches!(event, Event::Activity { module: name, .. } if name == module))
            .count();
        self.active[Self::projection(module)] && calls == 1
    }

    fn apply(&self, layer: usize, module: &str, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), HIDDEN, "hook input must be one hidden row");
        assert_eq!(output.len(), HIDDEN, "hook output must be one hidden row");
        let projection = Self::projection(module);
        let before = bits(output);
        for (channel, value) in output.iter_mut().enumerate() {
            *value = Self::adjusted(projection, input, channel, *value);
        }
        self.events.lock().unwrap().push(Event::Apply {
            layer,
            module: module.to_owned(),
            input: bits(input),
            before,
            after: bits(output),
        });
    }
}

struct DefaultActiveHook(RecordingHook);

impl LoraHook for DefaultActiveHook {
    fn apply(&self, layer: usize, module: &str, input: &[f32], output: &mut [f32]) {
        self.0.apply(layer, module, input, output);
    }
}

fn assert_behavior(current: &Storage, frozen: &Storage, used: usize) {
    assert!(
        current.output[..used].iter().all(|value| value.is_finite()),
        "packed output must be finite"
    );
    assert_eq!(
        bits(&current.output[..used]),
        bits(&frozen.output[..used]),
        "packed output must match frozen legacy bits"
    );
    assert_eq!(
        bits(&current.concat[..used]),
        bits(&frozen.concat[..used]),
        "packed context must match frozen legacy bits"
    );
}

fn assert_last_gather(
    storage: &Storage,
    fixture: &Fixture,
    input: &[f32],
    offsets: &[usize],
    active: [bool; 4],
) {
    let last = offsets.windows(2).rfind(|pair| pair[0] < pair[1]).unwrap();
    let rows = input[last[0] * HIDDEN..last[1] * HIDDEN].chunks_exact(HIDDEN);
    let length = last[1] - last[0];
    for (token, row) in rows.enumerate() {
        for channel in 0..HIDDEN {
            let expected = |projection: usize| {
                let value = fixture.projected(projection, row, channel);
                if active[projection] {
                    RecordingHook::adjusted(projection, row, channel, value).to_bits()
                } else {
                    value.to_bits()
                }
            };
            assert_eq!(
                storage.v_all_t[channel * length + token].to_bits(),
                expected(2),
                "V gather must preserve segment, channel, and token coordinates"
            );
            if channel >= HIDDEN - HEAD_DIM {
                let index = token * HEAD_DIM + channel - (HIDDEN - HEAD_DIM);
                assert_eq!(
                    storage.q_head[index].to_bits(),
                    expected(0),
                    "Q gather must preserve final head and segment coordinates"
                );
                assert_eq!(
                    storage.k_head[index].to_bits(),
                    expected(1),
                    "K gather must preserve final head and segment coordinates"
                );
            }
        }
    }
}

#[test]
fn packed_qkv_frozen_reference_bitwise_controls() {
    let fixture = Fixture::new();
    let layouts: &[&[usize]] = &[&[2, 0, 3, 1], &[1, 7, 0, 2], &[0, 1, 0], &[4, 0, 2]];
    for borrowed in [false, true] {
        let mut current = Storage::new();
        let mut frozen = Storage::new();
        let pointers = current.arrays().map(<[f32]>::as_ptr);
        for (layer, lengths) in layouts.iter().enumerate() {
            let (offsets, input) = packed_input(lengths, layer);
            let total = *offsets.last().unwrap();
            let longest = *lengths.iter().max().unwrap();
            let before_current = current.snapshot();
            let before_frozen = frozen.snapshot();
            current.run(
                &fixture,
                &input,
                &offsets,
                borrowed,
                false,
                &NoopLoraHook,
                layer,
            );
            frozen.run(
                &fixture,
                &input,
                &offsets,
                borrowed,
                true,
                &NoopLoraHook,
                layer,
            );
            assert_behavior(&current, &frozen, total * HIDDEN);
            current.assert_tails(&before_current, total, longest, borrowed);
            frozen.assert_tails(&before_frozen, total, longest, borrowed);
            assert_eq!(pointers, current.arrays().map(<[f32]>::as_ptr));
            if borrowed {
                assert_last_gather(&current, &fixture, &input, &offsets, [false; 4]);
            }
        }
    }
}

#[test]
fn packed_qkv_hook_trace_preserves_projection_order_and_values() {
    let fixture = Fixture::new();
    let (offsets, input) = packed_input(&[2, 0, 3, 1], 4);
    let used = input.len();
    let configurations = [
        [true; 4],
        [true, false, true, true],
        [false, true, false, false],
        [false; 4],
    ];
    for borrowed in [false, true] {
        for active in configurations {
            let mut current = Storage::new();
            let mut frozen = Storage::new();
            let current_hook = RecordingHook::new(active);
            let frozen_hook = RecordingHook::new(active);
            current.run(
                &fixture,
                &input,
                &offsets,
                borrowed,
                false,
                &current_hook,
                13,
            );
            frozen.run(&fixture, &input, &offsets, borrowed, true, &frozen_hook, 13);
            assert_behavior(&current, &frozen, used);
            assert_eq!(
                *current_hook.events.lock().unwrap(),
                *frozen_hook.events.lock().unwrap(),
                "hook activity and value trace must match frozen legacy"
            );
            current_hook.assert_trace(&fixture, &input, &current.concat[..used], 13, true);
            frozen_hook.assert_trace(&fixture, &input, &frozen.concat[..used], 13, true);
            if borrowed {
                assert_last_gather(&current, &fixture, &input, &offsets, active);
            }
        }
    }
}

#[test]
fn packed_qkv_default_active_hook_and_output_effect_controls() {
    let fixture = Fixture::new();
    let (offsets, input) = packed_input(&[2, 0, 3, 1], 3);
    let used = input.len();
    for borrowed in [false, true] {
        let mut current = Storage::new();
        let mut frozen = Storage::new();
        let current_hook = DefaultActiveHook(RecordingHook::new([true; 4]));
        let frozen_hook = DefaultActiveHook(RecordingHook::new([true; 4]));
        current.run(
            &fixture,
            &input,
            &offsets,
            borrowed,
            false,
            &current_hook,
            8,
        );
        frozen.run(&fixture, &input, &offsets, borrowed, true, &frozen_hook, 8);
        assert_behavior(&current, &frozen, used);
        assert_eq!(
            *current_hook.0.events.lock().unwrap(),
            *frozen_hook.0.events.lock().unwrap()
        );
        current_hook
            .0
            .assert_trace(&fixture, &input, &current.concat[..used], 8, false);

        let mut inactive = Storage::new();
        let mut output_only = Storage::new();
        let output_hook = RecordingHook::new([false, false, false, true]);
        inactive.run(
            &fixture,
            &input,
            &offsets,
            borrowed,
            false,
            &NoopLoraHook,
            8,
        );
        output_only.run(&fixture, &input, &offsets, borrowed, false, &output_hook, 8);
        assert_eq!(
            bits(&inactive.concat[..used]),
            bits(&output_only.concat[..used])
        );
        assert!(
            bits(&inactive.output[..used]) != bits(&output_only.output[..used]),
            "configured output hook must change the output"
        );
        for (row_index, row) in inactive.concat[..used].chunks_exact(HIDDEN).enumerate() {
            for channel in 0..HIDDEN {
                let index = row_index * HIDDEN + channel;
                assert_eq!(
                    output_only.output[index].to_bits(),
                    RecordingHook::adjusted(3, row, channel, inactive.output[index]).to_bits(),
                    "output hook delta must reach the returned projection"
                );
            }
        }
        output_hook.assert_trace(&fixture, &input, &output_only.concat[..used], 8, true);
    }
}

#[test]
fn packed_qkv_frozen_reference_overwrites_intermediates() {
    let fixture = Fixture::new();
    let (offsets, input) = packed_input(&[2, 0, 3, 1], 0);
    for borrowed in [false, true] {
        let mut frozen = Storage::new();
        let before = frozen.snapshot();
        frozen.run(&fixture, &input, &offsets, borrowed, true, &NoopLoraHook, 0);
        for (projection, values) in [&frozen.q, &frozen.k, &frozen.v].iter().enumerate() {
            assert!(
                bits(&values[..input.len()]) != before[projection][..input.len()],
                "frozen reference must overwrite caller Q/K/V scratch"
            );
            for (row_index, row) in input.chunks_exact(HIDDEN).enumerate() {
                for channel in 0..HIDDEN {
                    assert_eq!(
                        values[row_index * HIDDEN + channel].to_bits(),
                        fixture.projected(projection, row, channel).to_bits(),
                        "frozen projection must match independent sparse-weight oracle"
                    );
                }
            }
        }
    }
}

#[test]
fn packed_qkv_caller_intermediates_remain_untouched() {
    let fixture = Fixture::new();
    let (offsets, input) = packed_input(&[2, 0, 3, 1], 0);
    for borrowed in [false, true] {
        let mut current = Storage::new();
        let before = current.snapshot();
        let hook = RecordingHook::new([true, false, true, true]);
        current.run(&fixture, &input, &offsets, borrowed, false, &hook, 0);
        assert!(
            bits(&current.q) == before[0],
            "packed Q scratch must remain untouched"
        );
        assert!(
            bits(&current.k) == before[1],
            "packed K scratch must remain untouched"
        );
        assert!(
            bits(&current.v) == before[2],
            "packed V scratch must remain untouched"
        );
    }
}

#[test]
fn packed_qkv_invalid_geometry_precedes_hook_activity() {
    let fixture = Fixture::new();
    let (_, input) = packed_input(&[2, 1], 0);
    for active in [[false; 4], [true; 4]] {
        for case in 0..7 {
            let mut storage = Storage::new();
            let hook = RecordingHook::new(active);
            let mut offsets: &[usize] = &[0, 3];
            let mut hidden = HIDDEN;
            let mut heads = HEADS;
            let mut dimension = HEAD_DIM;
            let mut input_len = input.len();
            let mut q_len = input.len();
            let message = match case {
                0 => {
                    offsets = &[0];
                    "cu_seqlens must have at least 2 entries"
                }
                1 => {
                    offsets = &[1, 3];
                    "cu_seqlens must start at 0"
                }
                2 => {
                    heads = 0;
                    "num_heads must be non-zero"
                }
                3 => {
                    dimension = 0;
                    "head_dim must be non-zero"
                }
                4 => {
                    hidden -= 1;
                    "hidden_size must equal num_heads * head_dim"
                }
                5 => {
                    input_len -= 1;
                    "hidden_states length must equal total * hidden_size"
                }
                _ => {
                    q_len -= 1;
                    "q scratch too small"
                }
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                multi_head_attention_batched(
                    &input[..input_len],
                    &fixture.projections(),
                    &fixture.fused_weight,
                    &fixture.fused_bias,
                    offsets,
                    hidden,
                    heads,
                    dimension,
                    &mut storage.q[..q_len],
                    &mut storage.k,
                    &mut storage.v,
                    &mut storage.qkv,
                    &mut storage.concat,
                    &mut storage.output,
                    None,
                    &hook,
                    0,
                );
            }));
            let panic = result.expect_err("invalid packed geometry must panic");
            let text = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(text.contains(message), "packed geometry guard changed");
            assert!(
                hook.events.lock().unwrap().is_empty(),
                "geometry must precede hook activity"
            );
        }
    }
}

#[test]
fn packed_qkv_single_sequence_in_place_keeps_split_scratch_control() {
    let fixture = Fixture::new();
    let (_, input) = packed_input(&[3], 2);
    let mut buffers = AttentionBuffers::new(3, HIDDEN, HEADS, HIDDEN);
    buffers.q.fill(-701.0);
    buffers.k.fill(-702.0);
    buffers.v.fill(-703.0);
    multi_head_attention_in_place(
        &input,
        &fixture.projections(),
        &fixture.fused_weight,
        &fixture.fused_bias,
        &[1, 1, 1],
        3,
        HIDDEN,
        HEADS,
        HEAD_DIM,
        &mut buffers,
        &NoopLoraHook,
        0,
    );
    for (projection, values) in [&buffers.q, &buffers.k, &buffers.v].iter().enumerate() {
        for (row_index, row) in input.chunks_exact(HIDDEN).enumerate() {
            for channel in 0..HIDDEN {
                assert_eq!(
                    values[row_index * HIDDEN + channel].to_bits(),
                    fixture.projected(projection, row, channel).to_bits(),
                    "in-place negative control must retain split Q/K/V scratch"
                );
            }
        }
    }
    assert!(buffers.temp.iter().all(|value| value.is_finite()));
}
