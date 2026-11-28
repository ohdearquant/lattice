use super::*;

#[test]
#[ignore = "requires a real Apple7-or-newer Metal device and LATTICE_QUANT_FORMAT=Q8_0"]
fn q8_prefill_tiled_matches_naive_for_chunk_row_tails() {
    let _gpu_guard = gpu_test_lock();
    for intermediate in [32usize, 64, 96] {
        let (mut cfg, mut weights) = tiny_metal_qwen35_fixture();
        cfg.max_position_embeddings = 512;
        cfg.intermediate_size = intermediate;
        let fill = |values: &mut [f32], salt: usize| {
            for (i, value) in values.iter_mut().enumerate() {
                *value = (((i + salt) % 23) as i32 - 11) as f32 * 0.001;
            }
        };
        for (attention, common) in &mut weights.layers {
            if let AttentionWeights::Full(full) = attention {
                fill(&mut full.q_proj, 1);
                fill(&mut full.k_proj, 2);
                fill(&mut full.v_proj, 3);
                fill(&mut full.o_proj, 4);
            }
            if let FeedForwardWeights::Dense(ffn) = &mut common.ffn {
                ffn.gate_proj.resize(intermediate * cfg.hidden_size, 0.0);
                ffn.up_proj.resize(intermediate * cfg.hidden_size, 0.0);
                ffn.down_proj.resize(cfg.hidden_size * intermediate, 0.0);
                fill(&mut ffn.gate_proj, 5);
                fill(&mut ffn.up_proj, 6);
                fill(&mut ffn.down_proj, 7);
            }
        }
        let mut state =
            MetalQwen35State::new(&weights, &cfg, 512).expect("construct Q8 Metal state");
        assert!(matches!(state.engine.quant_format, QuantFormat::Q8_0));
        let tiled = state
            .engine
            .pipelines
            .gemm_q8_tiled
            .take()
            .expect("Q8 tiled pipeline required");
        for m in [1usize, 25, 31, 32, 33, 63, 64, 65, 361, 512] {
            let tokens = (0..m).map(|i| (i % 63) as u32).collect::<Vec<_>>();
            state.reset_state();
            let naive = state.forward_prefill(&tokens);
            state.engine.pipelines.gemm_q8_tiled = Some(tiled.clone());
            state.reset_state();
            let candidate = state.forward_prefill(&tokens);
            state.engine.pipelines.gemm_q8_tiled = None;
            assert_eq!(naive.len(), candidate.len());
            assert!(naive.iter().chain(&candidate).all(|v| v.is_finite()));
            let max_diff = naive
                .iter()
                .zip(&candidate)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 0.5, "M={m}: max diff {max_diff}");
            assert_eq!(argmax_f32(&naive), argmax_f32(&candidate), "M={m}");
            eprintln!(
                "[Q8_PREFILL_PARITY] m={m} n={intermediate} baseline=naive candidate=tiled max_abs_diff={max_diff}"
            );
        }
    }
}

#[test]
#[ignore = "requires a real Apple7-or-newer Metal device and LATTICE_QUANT_FORMAT=Q8_0"]
fn q8_dispatch_tails_and_nonzero_offsets_match_reference_and_preserve_guards() {
    let _gpu_guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 128).expect("construct Q8 state");
    assert!(matches!(state.engine.quant_format, QuantFormat::Q8_0));
    let tiled = state
        .engine
        .pipelines
        .gemm_q8_tiled
        .take()
        .expect("Q8 tiled pipeline required");
    let device = state.engine.device.clone();
    const K: usize = 64;
    const PAD: usize = 4;
    const WEIGHT_PAD: usize = 68;
    const CANARY: f32 = -31415.0;

    for m in [1usize, 25, 65] {
        for n in [63usize, 65] {
            let x: Vec<f32> = (0..m * K)
                .map(|i| if (i / K + i % K) % 4 == 0 { 0.0 } else { 1.0 })
                .collect();
            let w: Vec<f32> = (0..n * K)
                .map(|i| {
                    if (i % K + i / K) % 4 == 0 {
                        -127.0
                    } else {
                        127.0
                    }
                })
                .collect();
            let mut packed = Vec::new();
            for row in w.chunks_exact(K) {
                packed.extend(quantize_row_q8_0(row).expect("finite Q8 fixture"));
            }
            let mut padded_weights = vec![0x5au8; WEIGHT_PAD];
            padded_weights.extend_from_slice(&packed);
            padded_weights.extend_from_slice(&[0xa5u8; WEIGHT_PAD]);
            let raw = device.new_buffer_with_data(
                packed.as_ptr().cast(),
                packed.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let padded = device.new_buffer_with_data(
                padded_weights.as_ptr().cast(),
                padded_weights.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let qw = Q4WeightBuf::from_buffer(raw);
            let qw_offset = Q4WeightBuf::from_buffer(padded);
            let mut padded_x = vec![CANARY; PAD];
            padded_x.extend_from_slice(&x);
            padded_x.extend_from_slice(&[CANARY; PAD]);
            let x_buf = device.new_buffer_with_data(
                padded_x.as_ptr().cast(),
                (padded_x.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let reference = cpu_matmul_ref(&x, &w, m, n, K);

            let mut naive_results: Vec<Vec<f64>> = Vec::new();
            for use_tiled in [false, true] {
                state.engine.pipelines.gemm_q8_tiled =
                    if use_tiled { Some(tiled.clone()) } else { None };
                let initial = vec![CANARY; PAD + m * n + PAD];
                let ordinary_out = device.new_buffer_with_data(
                    initial.as_ptr().cast(),
                    (initial.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let offset_out = device.new_buffer_with_data(
                    initial.as_ptr().cast(),
                    (initial.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let command = state.engine.queue.new_command_buffer();
                let enc = command.new_compute_command_encoder();
                state.dispatch_gemm(
                    enc,
                    &x_buf,
                    (PAD * 4) as u64,
                    &qw,
                    &ordinary_out,
                    (PAD * 4) as u64,
                    m as u32,
                    n as u32,
                    K as u32,
                );
                state.dispatch_gemm_at(
                    enc,
                    &x_buf,
                    (PAD * 4) as u64,
                    &qw_offset,
                    WEIGHT_PAD as u64,
                    &offset_out,
                    (PAD * 4) as u64,
                    m as u32,
                    n as u32,
                    K as u32,
                );
                enc.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
                for (output_index, output) in [&ordinary_out, &offset_out].into_iter().enumerate() {
                    let values = super::q4_signed_scale::read_f32(output, initial.len());
                    assert!(values[..PAD].iter().all(|&v| v == f64::from(CANARY)));
                    assert!(
                        values[PAD + m * n..]
                            .iter()
                            .all(|&v| v == f64::from(CANARY))
                    );
                    let tolerance = if use_tiled && m > 1 { 0.015 } else { 0.001 };
                    for (actual, expected) in values[PAD..PAD + m * n].iter().zip(&reference) {
                        assert!(
                            (actual - f64::from(*expected)).abs() < tolerance,
                            "M={m} N={n} tiled={use_tiled}: got {actual}, expected {expected}"
                        );
                    }
                    if use_tiled {
                        for (actual, baseline) in values[PAD..PAD + m * n]
                            .iter()
                            .zip(&naive_results[output_index])
                        {
                            assert!(
                                (actual - baseline).abs() < 0.012,
                                "M={m} N={n}: tiled-vs-naive mismatch"
                            );
                        }
                    } else {
                        naive_results.push(values[PAD..PAD + m * n].to_vec());
                    }
                }
                let weight_words = super::q4_signed_scale::read_f32(
                    &qw_offset.buffer,
                    padded_weights.len() / std::mem::size_of::<f32>(),
                );
                let weight_pad_words = WEIGHT_PAD / std::mem::size_of::<f32>();
                let prefix = f64::from(f32::from_ne_bytes([0x5a; 4]));
                let suffix = f64::from(f32::from_ne_bytes([0xa5; 4]));
                assert!(
                    weight_words[..weight_pad_words]
                        .iter()
                        .all(|&v| v == prefix)
                );
                assert!(
                    weight_words[weight_words.len() - weight_pad_words..]
                        .iter()
                        .all(|&v| v == suffix)
                );
                let input = super::q4_signed_scale::read_f32(&x_buf, padded_x.len());
                assert!(
                    input[..PAD]
                        .iter()
                        .chain(&input[PAD + m * K..])
                        .all(|&v| v == f64::from(CANARY))
                );
                eprintln!(
                    "[Q8_OFFSET_PARITY] m={m} n={n} tiled_pipeline_present={use_tiled} status=ok"
                );
            }
            state.engine.pipelines.gemm_q8_tiled = None;
        }
    }
}
