use super::*;

// Only the reference positions are reused; projector output is outside this test.
#[derive(Deserialize)]
struct PositionGolden {
    revision: String,
    image: ImageGeometry,
    input_ids: Vec<u32>,
    position_ids: Vec<Vec<u32>>,
    rope_delta: i64,
}

#[derive(Deserialize)]
struct ImageGeometry {
    grid_thw: [usize; 3],
    num_image_tokens: usize,
}

#[test]
fn metal_ernie45_kv_unequal_axes_match_cpu() {
    let _gpu_guard = gpu_test_lock();
    if metal::Device::system_default().is_none() {
        assert!(
            !super::super::enforce(),
            "Metal device required under enforcement"
        );
        eprintln!("SKIP metal_ernie45 unequal-axis parity: Metal device missing");
        return;
    }
    let Some(dir) = model_dir() else {
        assert!(
            !super::super::enforce(),
            "checkpoint required; set LATTICE_POCR_MODEL_DIR"
        );
        eprintln!("SKIP metal_ernie45 unequal-axis parity: checkpoint missing");
        return;
    };
    const TOKENS: usize = 157;
    const PROMPT: usize = 147;
    const IMAGE_ID: u32 = 100295;
    const COUNTS: (u32, u32, u32, u32, u32, u32, u32) = (127, 37, 36, 18, 36, 0, 18);
    let golden: PositionGolden = serde_json::from_str(include_str!(
        "../../../../../tests/fixtures/paddleocr_vl/e2e/e2e_goldens.json"
    ))
    .expect("committed three-axis position fixture");
    assert_eq!(golden.revision, "c5630abae1d940eafe0697512a0325494b02ab42");
    assert_eq!(golden.image.grid_thw, [1, 18, 32]);
    assert_eq!(golden.image.num_image_tokens, 144);
    assert_eq!(golden.rope_delta, -128);
    assert_eq!(golden.input_ids.len(), TOKENS);
    assert_eq!(golden.position_ids.len(), 3);
    assert!(golden.position_ids.iter().all(|row| row.len() == TOKENS));
    let positions: Vec<[u32; 3]> = (0..TOKENS)
        .map(|i| std::array::from_fn(|axis| golden.position_ids[axis][i]))
        .collect();
    let unequal = |p: &&[u32; 3]| p[0] != p[1] || p[1] != p[2];
    let unequal_tokens = positions.iter().filter(unequal).count();
    assert_eq!(unequal_tokens, 143);
    assert_eq!(positions[..PROMPT].iter().filter(unequal).count(), 141);
    assert_eq!(positions.iter().filter(|p| p[1] != p[2]).count(), 135);
    assert_eq!(positions.iter().filter(|p| p[0] != p[1]).count(), 128);
    assert_eq!(
        positions.iter().position(|p| p[0] != p[1] || p[1] != p[2]),
        Some(6)
    );
    assert_eq!(positions[PROMPT], [5, 13, 19]);
    assert_eq!(positions[PROMPT + 1], [5, 13, 20]);
    let unequal_decode_steps = positions[PROMPT..].iter().filter(unequal).count();
    assert_eq!(unequal_decode_steps, 2);
    for (i, &id) in golden.input_ids.iter().enumerate() {
        assert_eq!(id == IMAGE_ID, (5..149).contains(&i), "image token {i}");
    }

    let text: Golden = serde_json::from_str(include_str!(
        "../../../../../tests/fixtures/paddleocr_vl/decoder/decoder_goldens.json"
    ))
    .expect("committed text embedding fixture");
    assert_eq!(text.revision, golden.revision);
    let text_ids = &text
        .cases
        .iter()
        .find(|case| case.id == "table_row")
        .expect("table-row fixture")
        .ids;
    assert_eq!(text_ids.len(), 11);
    let cfg = Ernie45Config::from_config_json(&dir.join("config.json"))
        .expect("checkpoint configuration");
    assert_eq!(cfg.num_hidden_layers, 18);
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.intermediate_size, 3072);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.vocab_size, 103424);
    assert_eq!(cfg.rope_scaling.mrope_section, [16, 24, 24]);
    let h = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    let mut source =
        SafetensorsFile::open(&dir.join("model.safetensors")).expect("real checkpoint opens");
    let weights = Ernie45Weights::load(&mut source, &cfg).expect("all real layers load");
    drop(source);
    assert_eq!(weights.layers.len(), 18);
    // Vary image-position inputs independently of both grid sides instead of
    // repeating one sentinel embedding. This tests positions, not vision or
    // golden OCR logits.
    let mut embeds = Vec::with_capacity(TOKENS * h);
    for (i, &id) in golden.input_ids.iter().enumerate() {
        let id = if id == IMAGE_ID {
            text_ids[(i - 5) % text_ids.len()]
        } else {
            id
        } as usize;
        assert!(id < vocab);
        embeds.extend_from_slice(&weights.embed_tokens[id * h..][..h]);
    }
    let mut state =
        MetalErnie45State::new(&cfg, &weights, TOKENS).expect("full Metal decoder constructs");
    let model = Ernie45Model::new(cfg, weights).expect("full CPU decoder validates");
    let mut cache = state.new_kv_cache(TOKENS).expect("full Metal cache");
    let mut cpu_cache = model.new_kv_cache(TOKENS).expect("full CPU cache");
    let mut metal_logits = vec![f32::NAN; vocab];
    let mut full_logits = vec![f32::NAN; vocab];
    let metal_positions: Vec<_> = positions
        .iter()
        .copied()
        .map(metal_position_for_axis_control)
        .collect();
    eprintln!(
        "[METAL_ERNIE45_MROPE_FIXTURE] token_count={TOKENS} unequal_tokens={unequal_tokens} grid={:?} prompt_len={PROMPT} decode_steps={} unequal_decode_steps={unequal_decode_steps}",
        golden.image.grid_thw,
        TOKENS - PROMPT
    );
    let cpu_logits = model
        .kv_prefill(&embeds[..PROMPT * h], &positions[..PROMPT], &mut cpu_cache)
        .expect("unequal-axis CPU cached prefill");
    state
        .kv_prefill(
            &embeds[..PROMPT * h],
            &metal_positions[..PROMPT],
            &mut cache,
            &mut metal_logits,
        )
        .expect("unequal-axis Metal cached prefill");
    assert_eq!(state.last_dispatch_counts(), COUNTS);
    assert_eq!(cache.len(), PROMPT);
    assert_eq!(cpu_cache.len(), PROMPT);
    let cpu = assert_cached_parity(
        &metal_logits,
        &cpu_logits,
        "unequal-axis cached prefill CPU",
    );
    let trace = state
        .prefill_embeds_trace_for_test(
            &embeds[..PROMPT * h],
            &positions[..PROMPT],
            &mut full_logits,
        )
        .expect("independent unequal-axis Metal prefill trace");
    assert_eq!(state.last_dispatch_counts(), COUNTS);
    assert_cached_parity(
        &metal_logits,
        &full_logits,
        "unequal-axis cached prefill Metal",
    );
    assert_cache_rows(&cache, &trace, "unequal-axis prefill independent trace");
    eprintln!(
        "[METAL_ERNIE45_MROPE_STEP] step=0 cache_len={PROMPT} bit_exact_all_layers=true cpu={cpu:?}"
    );

    for i in PROMPT..TOKENS {
        let previous = cache_rows(&cache);
        let cpu_logits = model
            .kv_decode_step(&embeds[i * h..(i + 1) * h], positions[i], &mut cpu_cache)
            .expect("unequal-axis CPU cached step");
        state
            .kv_decode_step(
                &embeds[i * h..(i + 1) * h],
                metal_positions[i],
                &mut cache,
                &mut metal_logits,
            )
            .expect("unequal-axis Metal cached step");
        assert_eq!(state.last_dispatch_counts(), COUNTS);
        assert_eq!(cache.len(), i + 1);
        assert_eq!(cpu_cache.len(), cache.len());
        let cpu =
            assert_cached_parity(&metal_logits, &cpu_logits, "unequal-axis cached decode CPU");
        for (layer, previous) in previous.iter().enumerate() {
            let current = cache.layer_rows_for_test(layer);
            assert_same_kv_rows(
                (&current.0[..i * kv_dim], &current.1[..i * kv_dim]),
                (&previous.0, &previous.1),
                &format!("unequal-axis token {i} layer {layer} preserves prefix"),
            );
        }
        let trace = state
            .prefill_embeds_trace_for_test(
                &embeds[..(i + 1) * h],
                &positions[..=i],
                &mut full_logits,
            )
            .expect("independent growing-prefix unequal-axis Metal trace");
        assert_eq!(state.last_dispatch_counts(), COUNTS);
        let full = assert_cached_parity(
            &metal_logits,
            &full_logits,
            "unequal-axis cached decode Metal",
        );
        assert_cache_rows(
            &cache,
            &trace,
            &format!("unequal-axis token {i} independent trace"),
        );
        eprintln!(
            "[METAL_ERNIE45_MROPE_STEP] step={} cache_len={} position={:?} bit_exact_all_layers=true prompt_rows_preserved=true cpu={cpu:?} metal={full:?}",
            i + 1 - PROMPT,
            cache.len(),
            positions[i]
        );
    }
    assert_eq!(cache.len(), TOKENS);
    eprintln!(
        "[METAL_ERNIE45_MROPE_GATE] executed=true layers=18 token_count={TOKENS} unequal_tokens={unequal_tokens} grid={:?} prompt_len={PROMPT} decode_steps={} unequal_decode_steps={unequal_decode_steps} bit_exact_all_layers=true",
        golden.image.grid_thw,
        TOKENS - PROMPT
    );
}
