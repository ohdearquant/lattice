//! Numerical and input-validation tests for requested prefill readouts.

use super::*;

fn readout_value_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn readout_ordered_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

fn readout_assert_exact(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    let mut first_mismatch = None;
    let mut first_nonfinite = None;
    let mut max_finite_ulp = None;
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if !got.is_finite() || !want.is_finite() {
            first_nonfinite.get_or_insert(index);
        } else {
            let distance = readout_ordered_bits(got).abs_diff(readout_ordered_bits(want));
            max_finite_ulp = Some(max_finite_ulp.unwrap_or(0).max(distance));
        }
        if got.to_bits() != want.to_bits() {
            first_mismatch.get_or_insert(index);
        }
    }
    if let Some(index) = first_nonfinite.or(first_mismatch) {
        panic!(
            "{context}: first_bad={index}; actual_bits={:08x}; expected_bits={:08x}; \
             first_nonfinite={first_nonfinite:?}; max_finite_ulp={max_finite_ulp:?}",
            actual[index].to_bits(),
            expected[index].to_bits(),
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum ReadoutCorpus {
    DenseMantissas,
    MixedExponents,
    Cancellation,
    RowRouting,
}

fn readout_next_word(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

fn readout_finite_word(word: u32, exponent: u32) -> f32 {
    f32::from_bits((word & 0x8000_0000) | (exponent << 23) | (word & 0x007f_ffff) | 1)
}

fn readout_projection_fixture(
    full_rows: usize,
    corpus: ReadoutCorpus,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let k = 1024;
    let mut state = seed;
    let mut hidden = vec![0.0; k];
    let mut head = vec![0.0; full_rows * k];
    match corpus {
        ReadoutCorpus::DenseMantissas => {
            for value in &mut hidden {
                *value = readout_finite_word(readout_next_word(&mut state), 126);
            }
            for value in &mut head {
                *value = readout_finite_word(readout_next_word(&mut state), 126);
            }
        }
        ReadoutCorpus::MixedExponents => {
            for value in &mut hidden {
                let word = readout_next_word(&mut state);
                *value = readout_finite_word(word, 117 + word % 16);
            }
            for value in &mut head {
                let word = readout_next_word(&mut state);
                *value = readout_finite_word(word, 113 + word % 22);
            }
        }
        ReadoutCorpus::Cancellation => {
            for block in 0..k / 4 {
                let word = readout_next_word(&mut state) & 0x007f_ffff;
                let large_pair_input = f32::from_bits((127 << 23) | word | 1);
                hidden[4 * block] = large_pair_input;
                hidden[4 * block + 2] = large_pair_input;
                hidden[4 * block + 1] = readout_finite_word(readout_next_word(&mut state), 126);
                hidden[4 * block + 3] = readout_finite_word(readout_next_word(&mut state), 127);
            }
            for row in 0..full_rows {
                for block in 0..k / 4 {
                    let offset = row * k + 4 * block;
                    let word = readout_next_word(&mut state) & 0x007f_ffff;
                    let large = f32::from_bits((139 << 23) | word | 1);
                    head[offset] = large;
                    head[offset + 2] = -large;
                    head[offset + 1] = readout_finite_word(readout_next_word(&mut state), 115);
                    head[offset + 3] = readout_finite_word(readout_next_word(&mut state), 123);
                }
            }
        }
        ReadoutCorpus::RowRouting => {
            // A separate routing witness keeps adjacent rows distinguishable;
            // the other corpora exercise rounded, dense accumulation over K.
            hidden[0] = 1.0;
            hidden[1] = f32::from_bits(0x3eaa_aaab);
            for row in 0..full_rows {
                head[row * k] = 2.0 * row as f32 + 0.25;
                head[row * k + 1] = 0.75;
            }
        }
    }
    assert!(hidden.iter().chain(&head).all(|value| value.is_finite()));
    assert!(hidden.iter().any(|value| *value != 0.0));
    (hidden, head)
}

struct ReadoutProjectionObservation {
    context: String,
    actual: Vec<f32>,
    expected: Vec<f32>,
    gather_error: Option<InferenceError>,
    projection_error: Option<InferenceError>,
    first_gather_difference: Option<(usize, usize, u32, u32)>,
    max_finite_ulp: Option<u32>,
    first_full_nonfinite: Option<usize>,
    head_unchanged: bool,
    hidden_unchanged: bool,
    full_hidden_unchanged: bool,
    selected_unchanged: bool,
    routing_distinct: bool,
}

fn readout_projection_t3(full_rows: usize, target_disposition: &str) {
    let k = 1024;
    let restricted_rows = 255;
    let rows: Vec<u32> = (0..restricted_rows)
        .map(|index| ((13 + index * 73) % (full_rows - 1)) as u32)
        .collect();
    let unique: std::collections::BTreeSet<u32> = rows.iter().copied().collect();
    assert_eq!(unique.len(), restricted_rows);
    assert!(rows.windows(2).any(|pair| pair[0] > pair[1]));
    assert!(rows.iter().all(|&row| row as usize + 1 < full_rows));
    assert!(k * full_rows >= 1024 * 1024);
    assert!(k * restricted_rows < 1024 * 1024);

    let cases = [
        (ReadoutCorpus::DenseMantissas, 0x5a71_b609_310c_e207),
        (ReadoutCorpus::MixedExponents, 0x41c8_d276_098f_5b33),
        (ReadoutCorpus::Cancellation, 0x9a20_783d_81fe_60b5),
        (ReadoutCorpus::RowRouting, 0x36ef_1059_4bcd_0281),
    ];
    let mut observations = Vec::with_capacity(cases.len());
    let mut all_case_max_finite_ulp: Option<u32> = None;
    for (corpus, seed) in cases {
        let (hidden, head) = readout_projection_fixture(full_rows, corpus, seed);
        let hidden_for_full = hidden.clone();
        let hidden_before = readout_value_bits(&hidden);
        let head_before = readout_value_bits(&head);
        let hidden_copy_identical = readout_value_bits(&hidden_for_full) == hidden_before;
        let context = format!(
            "target={}-{}; disposition={target_disposition}; full=(1,{k},{full_rows}); \
             restricted=(1,{k},{restricted_rows}); corpus={corpus:?}; seed={seed:016x}; rows={rows:?}",
            std::env::consts::OS,
            std::env::consts::ARCH,
        );

        let mut selected_scratch = vec![f32::NAN; restricted_rows * k];
        let gather_error = super::gather_readout_rows(&head, k, &rows, &mut selected_scratch).err();
        let mut first_gather_difference = None;
        for (index, &row) in rows.iter().enumerate() {
            let got = &selected_scratch[index * k..(index + 1) * k];
            let want = &head[row as usize * k..(row as usize + 1) * k];
            for (element, (&got, &want)) in got.iter().zip(want).enumerate() {
                if first_gather_difference.is_none() && got.to_le_bytes() != want.to_le_bytes() {
                    first_gather_difference = Some((index, element, got.to_bits(), want.to_bits()));
                }
            }
        }

        let mut full = vec![f32::NAN; full_rows];
        crate::forward::cpu::matmul_bt(&hidden_for_full, &head, &mut full, 1, k, full_rows);
        let expected: Vec<f32> = rows.iter().map(|&row| full[row as usize]).collect();
        let mut actual = vec![f32::NAN; restricted_rows];
        let selected_before = readout_value_bits(&selected_scratch);
        let result = super::project_readout_hidden(&hidden, &selected_scratch, &mut actual);
        let max_finite_ulp = actual
            .iter()
            .zip(&expected)
            .filter(|(got, want)| got.is_finite() && want.is_finite())
            .map(|(&got, &want)| readout_ordered_bits(got).abs_diff(readout_ordered_bits(want)))
            .max();
        if let Some(distance) = max_finite_ulp {
            all_case_max_finite_ulp = Some(all_case_max_finite_ulp.unwrap_or(0).max(distance));
        }
        let routing_distinct = !matches!(corpus, ReadoutCorpus::RowRouting)
            || rows
                .iter()
                .all(|&row| full[row as usize].to_bits() != full[row as usize + 1].to_bits());
        observations.push(ReadoutProjectionObservation {
            context,
            actual,
            expected,
            gather_error,
            projection_error: result.err(),
            first_gather_difference,
            max_finite_ulp,
            first_full_nonfinite: full.iter().position(|value| !value.is_finite()),
            head_unchanged: readout_value_bits(&head) == head_before,
            hidden_unchanged: readout_value_bits(&hidden) == hidden_before,
            full_hidden_unchanged: hidden_copy_identical
                && readout_value_bits(&hidden_for_full) == hidden_before,
            selected_unchanged: readout_value_bits(&selected_scratch) == selected_before,
            routing_distinct,
        });
    }
    // Compute every corpus before a numerical assertion so the failure includes
    // both this case's finite ULP maximum and the maximum across all cases.
    for observation in observations {
        let context = format!(
            "{}; case_max_finite_ulp={:?}; all_case_max_finite_ulp={all_case_max_finite_ulp:?}",
            observation.context, observation.max_finite_ulp,
        );
        assert!(
            observation.gather_error.is_none(),
            "{context}: gather refused exact buffers: {:?}",
            observation.gather_error,
        );
        if let Some((index, element, got, want)) = observation.first_gather_difference {
            panic!(
                "{context}: gathered bytes differ at selected index {index}, head row {}, \
                 element {element}; actual_bits={got:08x}; expected_bits={want:08x}",
                rows[index],
            );
        }
        assert!(observation.head_unchanged, "{context}: head changed");
        assert!(observation.hidden_unchanged, "{context}: hidden changed");
        assert!(
            observation.full_hidden_unchanged,
            "{context}: full hidden changed"
        );
        assert!(
            observation.selected_unchanged,
            "{context}: gathered weights changed"
        );
        assert!(
            observation.projection_error.is_none(),
            "{context}: restricted projection refused: {:?}",
            observation.projection_error,
        );
        readout_assert_exact(&observation.actual, &observation.expected, &context);
        assert!(
            observation.first_full_nonfinite.is_none(),
            "{context}: full reference contains nonfinite output at {:?}",
            observation.first_full_nonfinite,
        );
        assert!(
            observation.routing_distinct,
            "{context}: row+1 mutation lacks an adjacent-row numerical witness",
        );
    }
    eprintln!(
        "READOUT_PROJECTION_EXACT_OK target={}-{}; disposition={target_disposition}; \
         full=(1,{k},{full_rows}); restricted=(1,{k},{restricted_rows}); \
         corpus_count={}; all_case_max_finite_ulp={all_case_max_finite_ulp:?}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        cases.len(),
    );
}

/// Accelerate is opaque: this checks exact equality, without a vendor-kernel claim.
#[cfg(target_os = "macos")]
#[test]
fn readout_projection_matches_full_row_macos_bitwise() {
    readout_projection_t3(2048, "Accelerate empirical equality only");
}

/// At revision 1c39a993edbfc112719e6af6270d4df442a8eb58, matmul.rs:150-159,
/// 215-221 and 239-257 show the predicate crossing, but m=1 has zero tiled
/// leading rows. Both calls are direct; this does not claim a kernel crossing.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
#[test]
fn readout_projection_matches_full_row_x86_predicate_crossing_bitwise() {
    readout_projection_t3(2048, "tiling predicate crosses; both calls direct");
}

/// NonmacOS ARM/scalar uses a real tiled/direct crossing at these m=1 shapes.
#[cfg(all(not(target_os = "macos"), not(target_arch = "x86_64")))]
#[test]
fn readout_projection_matches_full_row_tiled_direct_bitwise() {
    readout_projection_t3(1024, "full tiled; restricted direct NEON/scalar");
}

#[test]
fn readout_projection_rejects_nonfinite_outputs() {
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut selected = vec![0.0; 1024];
        selected[0] = nonfinite;
        let hidden = vec![1.0; 1024];
        let mut logits = vec![0.0];
        assert!(matches!(
            super::project_readout_hidden(&hidden, &selected, &mut logits),
            Err(InferenceError::Inference(_)),
        ));
    }
}

#[test]
fn readout_exact_diagnostic_rejects_equal_nan_bits() {
    let same_nan = f32::from_bits(0x7fc0_0041);
    let failure = std::panic::catch_unwind(|| {
        readout_assert_exact(&[same_nan, 1.0], &[same_nan, 1.0], "same NaN bits");
    });
    assert!(failure.is_err());
}

#[test]
fn readout_exact_diagnostic_rejects_one_ulp_and_signed_zero() {
    for (actual, expected) in [(f32::from_bits(0x3f80_0001), 1.0), (-0.0, 0.0)] {
        let failure = std::panic::catch_unwind(|| {
            readout_assert_exact(&[actual], &[expected], "exact bits positive control");
        });
        assert!(failure.is_err());
    }
}

fn readout_test_call(
    model: &Qwen35Model,
    tokens: &[u32],
    slots: &[usize],
    rows: &[u32],
) -> Result<Vec<Vec<f32>>, InferenceError> {
    model.prefill_readout(
        crate::model::qwen35::PrefillInput::TokenIds(tokens),
        slots,
        rows,
    )
}

#[test]
fn readout_last_slot_matches_existing_scorer_fixtures_bitwise() {
    let fixtures: &[(u64, &[u32], &[u32])] = &[
        (0x0A71_0C01, &[11, 42, 7, 7, 63, 5, 20], &[33, 8, 90, 1]),
        (0x0A71_0C02, &[3, 14, 15, 92, 65], &[10, 20, 30]),
        (0x0A71_0C03, &[9, 8, 7], &[4, 5, 6]),
        (0x0A71_0C03, &[1, 8, 7], &[4, 5, 6]),
    ];
    for &(seed, prompt, rows) in fixtures {
        let model = build_model(test_config(), seed);
        let old = model.score_option_letters(prompt, rows).unwrap();
        let actual = readout_test_call(&model, prompt, &[prompt.len() - 1], rows).unwrap();
        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].len(), rows.len());
        let context = format!("last slot seed={seed:08x}; prompt={prompt:?}; rows={rows:?}");
        readout_assert_exact(&actual[0], &old.logits, &context);
        let previous = hand_read_last_logits(&model, &prompt[..prompt.len() - 1]);
        let previous_selected: Vec<f32> = rows.iter().map(|&row| previous[row as usize]).collect();
        assert_ne!(
            readout_value_bits(&old.logits),
            readout_value_bits(&previous_selected),
            "{context}: slot-1 mutation has no fixture witness",
        );
    }
}

#[test]
fn readout_multiple_slots_match_independent_prefix_oracle_bitwise() {
    let model = build_model(test_config(), 0x0A71_0C01);
    let prompt = [11, 42, 7, 7, 63, 5, 20];
    let slots = [1, 3, 6];
    let rows = [33, 8, 90, 1];
    let actual = readout_test_call(&model, &prompt, &slots, &rows).unwrap();
    assert_eq!(actual.len(), slots.len());
    for (index, &slot) in slots.iter().enumerate() {
        assert_eq!(actual[index].len(), rows.len());
        let reference = hand_read_last_logits(&model, &prompt[..=slot]);
        let selected: Vec<f32> = rows.iter().map(|&row| reference[row as usize]).collect();
        let context = format!("prefix oracle slot={slot}; rows={rows:?}");
        readout_assert_exact(&actual[index], &selected, &context);
        let previous = hand_read_last_logits(&model, &prompt[..slot]);
        let previous_selected: Vec<f32> = rows.iter().map(|&row| previous[row as usize]).collect();
        assert_ne!(
            readout_value_bits(&selected),
            readout_value_bits(&previous_selected),
            "{context}: slot-1 mutation has no fixture witness",
        );
    }
    for left in 0..slots.len() {
        for right in left + 1..slots.len() {
            assert_ne!(
                readout_value_bits(&actual[left]),
                readout_value_bits(&actual[right])
            );
        }
    }
}

#[test]
fn readout_repeated_a_b_a_requests_start_with_fresh_state() {
    let model = build_model(test_config(), 0x0A71_0C03);
    let rows = [4, 5, 6];
    let first = readout_test_call(&model, &[9, 8, 7], &[0, 2], &rows).unwrap();
    let other = readout_test_call(&model, &[1, 8, 7], &[0, 2], &rows).unwrap();
    let again = readout_test_call(&model, &[9, 8, 7], &[0, 2], &rows).unwrap();
    assert_eq!(first.len(), 2);
    assert_eq!(again.len(), 2);
    for index in 0..2 {
        readout_assert_exact(&again[index], &first[index], "A-B-A fresh state");
    }
    assert_ne!(readout_value_bits(&first[1]), readout_value_bits(&other[1]));
}

fn readout_refusal_model() -> Qwen35Model {
    let mut cfg = test_config();
    cfg.vocab_size = 300;
    cfg.max_position_embeddings = 8;
    cfg.tie_word_embeddings = false;
    let mut model = build_model(cfg, 0x0A71_0C04);
    // A malformed embedding makes any real token forward observable as a panic.
    // Keep a valid independent head so gathering cannot become the same trap.
    model.weights.lm_head = Some(model.weights.embed_tokens.clone());
    assert_eq!(
        model.weights.lm_head.as_ref().unwrap().len(),
        model.config.vocab_size * model.config.hidden_size,
    );
    assert_eq!(model.max_context(), 8);
    model.weights.embed_tokens.clear();
    model
}

fn readout_assert_refused_before_forward(
    tokens: &[u32],
    slots: &[usize],
    rows: &[u32],
    needle: &str,
) {
    let model = readout_refusal_model();
    let result = readout_test_call(&model, tokens, slots, rows);
    match result {
        Err(InferenceError::InvalidInput(message)) => assert!(
            message.contains(needle),
            "expected {needle:?}; got {message:?}; tokens={tokens:?}; slots={slots:?}; rows={rows:?}",
        ),
        other => panic!("expected InvalidInput containing {needle:?}, got {other:?}"),
    }
}

#[test]
fn readout_refuses_empty_input_before_forward() {
    readout_assert_refused_before_forward(&[], &[0], &[0], "must not be empty");
}

#[test]
fn readout_refuses_excess_context_before_forward() {
    readout_assert_refused_before_forward(&[0; 9], &[0], &[0], "exceeds RoPE capacity");
}

#[test]
fn readout_refuses_later_out_of_vocab_token_before_first_forward() {
    readout_assert_refused_before_forward(&[0, 300], &[0], &[0], "tokens[1]");
}

#[test]
fn readout_refuses_empty_slots_before_forward() {
    readout_assert_refused_before_forward(&[0], &[], &[0], "slots must not be empty");
}

#[test]
fn readout_refuses_slot_at_input_length_before_forward() {
    readout_assert_refused_before_forward(&[0, 1, 2], &[3], &[0], "out of range");
}

#[test]
fn readout_refuses_duplicate_slots_before_forward() {
    readout_assert_refused_before_forward(&[0, 1], &[0, 0], &[0], "strictly increasing");
}

#[test]
fn readout_refuses_descending_slots_before_forward() {
    readout_assert_refused_before_forward(&[0, 1], &[1, 0], &[0], "strictly increasing");
}

#[test]
fn readout_refuses_empty_rows_before_forward() {
    readout_assert_refused_before_forward(&[0], &[0], &[], "must not be empty");
}

#[test]
fn readout_refuses_out_of_vocab_row_before_forward() {
    readout_assert_refused_before_forward(&[0], &[0], &[300], "out of range");
}

#[test]
fn readout_refuses_duplicate_rows_before_forward() {
    readout_assert_refused_before_forward(&[0], &[0], &[4, 5, 4], "duplicate");
}

#[test]
fn readout_refuses_256_distinct_valid_rows_before_forward() {
    let rows: Vec<u32> = (0..256).collect();
    readout_assert_refused_before_forward(&[0], &[0], &rows, "at most 255");
}

#[test]
fn readout_forward_embedding_trap_is_live() {
    let model = readout_refusal_model();
    let mut gdn_states: Vec<GatedDeltaNetState> = (0..model.config.num_linear_attention_layers())
        .map(|_| GatedDeltaNetState::new(&model.config))
        .collect();
    let mut kv_cache = KvCache::new(model.config.num_full_attention_layers());
    let mut scratch = ForwardScratch::new();
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.forward_hidden_step(0, 0, &mut gdn_states, &mut kv_cache, &mut scratch);
    }));
    assert!(failure.is_err());
}

#[test]
fn readout_accepts_single_token_zero_slot() {
    let model = build_model(test_config(), 0x0A71_0C01);
    let rows = [33, 8, 90, 1];
    let result = readout_test_call(&model, &[11], &[0], &rows).unwrap();
    assert_eq!(result.len(), 1);
    let reference = hand_read_last_logits(&model, &[11]);
    let selected: Vec<f32> = rows.iter().map(|&row| reference[row as usize]).collect();
    readout_assert_exact(&result[0], &selected, "single token zero slot");
}

#[test]
fn readout_accepts_exact_context_and_255_rows_in_caller_order() {
    let mut cfg = test_config();
    cfg.vocab_size = 300;
    cfg.max_position_embeddings = 8;
    let model = build_model(cfg, 0x0A71_0C04);
    let prompt = [11, 42, 7, 7, 63, 5, 20, 9];
    let slots = [0, 3, 7];
    let rows: Vec<u32> = (0..255).map(|index| (13 + index * 73) % 299).collect();
    let result = readout_test_call(&model, &prompt, &slots, &rows).unwrap();
    assert_eq!(result.len(), slots.len());
    for (index, &slot) in slots.iter().enumerate() {
        assert_eq!(result[index].len(), 255);
        let reference = hand_read_last_logits(&model, &prompt[..=slot]);
        let selected: Vec<f32> = rows.iter().map(|&row| reference[row as usize]).collect();
        readout_assert_exact(
            &result[index],
            &selected,
            "inclusive boundaries and caller order",
        );
    }
}

#[test]
fn readout_rejects_nonfinite_returned_logits() {
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut model = build_model(test_config(), 0x0A71_0C01);
        model.weights.final_norm[0] = nonfinite;
        let result = readout_test_call(&model, &[11, 42], &[1], &[33, 8]);
        assert!(matches!(result, Err(InferenceError::Inference(_))));
    }
}

#[test]
fn readout_hidden_sibling_matches_original_post_norm_at_every_position() {
    let model = build_model(test_config(), 0x0A71_0C01);
    let prompt = [11, 42, 7, 7, 63, 5, 20];
    let h = model.config.hidden_size;
    let mut original_states: Vec<GatedDeltaNetState> =
        (0..model.config.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::new(&model.config))
            .collect();
    let mut hidden_states: Vec<GatedDeltaNetState> =
        (0..model.config.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::new(&model.config))
            .collect();
    let mut original_kv = KvCache::new(model.config.num_full_attention_layers());
    let mut hidden_kv = KvCache::new(model.config.num_full_attention_layers());
    let mut original_scratch = ForwardScratch::new();
    let mut hidden_scratch = ForwardScratch::new();
    for (position, &token) in prompt.iter().enumerate() {
        model.forward_step(
            token,
            position,
            &mut original_states,
            &mut original_kv,
            &mut original_scratch,
        );
        model.forward_hidden_step(
            token,
            position,
            &mut hidden_states,
            &mut hidden_kv,
            &mut hidden_scratch,
        );
        readout_assert_exact(
            &hidden_scratch.hidden[..h],
            &original_scratch.hidden[..h],
            &format!("post-norm hidden position={position}"),
        );
        if position + 1 < prompt.len() {
            original_kv.seq_len += 1;
            hidden_kv.seq_len += 1;
        }
    }
}

#[test]
fn readout_hidden_sibling_does_not_access_untied_head() {
    let model = build_model(test_config(), 0x0A71_0C01);
    let h = model.config.hidden_size;
    let mut states: Vec<GatedDeltaNetState> = (0..model.config.num_linear_attention_layers())
        .map(|_| GatedDeltaNetState::new(&model.config))
        .collect();
    let mut cache = KvCache::new(model.config.num_full_attention_layers());
    let mut scratch = ForwardScratch::new();
    model.forward_step(11, 0, &mut states, &mut cache, &mut scratch);
    let expected = scratch.hidden[..h].to_vec();

    let mut head_trap_model = build_model(test_config(), 0x0A71_0C01);
    head_trap_model.config.tie_word_embeddings = false;
    head_trap_model.weights.lm_head = Some(Vec::new());
    let mut states: Vec<GatedDeltaNetState> =
        (0..head_trap_model.config.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::new(&head_trap_model.config))
            .collect();
    let mut cache = KvCache::new(head_trap_model.config.num_full_attention_layers());
    let mut scratch = ForwardScratch::new();
    head_trap_model.forward_hidden_step(11, 0, &mut states, &mut cache, &mut scratch);
    readout_assert_exact(&scratch.hidden[..h], &expected, "hidden-only ignores head");

    let mut states: Vec<GatedDeltaNetState> =
        (0..head_trap_model.config.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::new(&head_trap_model.config))
            .collect();
    let mut cache = KvCache::new(head_trap_model.config.num_full_attention_layers());
    let mut scratch = ForwardScratch::new();
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        head_trap_model.forward_step(11, 0, &mut states, &mut cache, &mut scratch);
    }));
    assert!(
        failure.is_err(),
        "malformed-head positive control must trap original projection"
    );
}
