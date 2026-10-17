use super::*;

fn make_test_adapter() -> LoraAdapter {
    let config = LoraConfig {
        rank: 2,
        alpha: 4.0, // scale = 4.0 / 2 = 2.0
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        dtype: "f32".into(),
    };

    let mut layers = HashMap::new();

    // Layer 0, q_proj: rank=2, d_in=4, d_out=4
    // A = identity-like (first 2 components), B = scale
    layers.insert(
        (0, "q_proj".into()),
        LoraLayer {
            a: vec![
                1.0, 0.0, 0.0, 0.0, // row 0
                0.0, 1.0, 0.0, 0.0, // row 1
            ],
            b: vec![
                1.0, 0.0, // row 0
                0.0, 1.0, // row 1
                0.0, 0.0, // row 2
                0.0, 0.0, // row 3
            ],
            d_in: 4,
            d_out: 4,
            rank: 2,
        },
    );

    LoraAdapter::new(config, layers).expect("valid adapter config")
}

#[test]
fn test_config_scale() {
    let config = LoraConfig {
        rank: 8,
        alpha: 16.0,
        target_modules: vec![],
        dtype: "f32".into(),
    };
    assert!((config.scale() - 2.0).abs() < 1e-6);

    let config_zero = LoraConfig {
        rank: 0,
        alpha: 1.0,
        target_modules: vec![],
        dtype: "f32".into(),
    };
    assert_eq!(config_zero.scale(), 0.0);
}

#[test]
fn test_config_rejects_non_finite_alpha_and_scale_fails_closed() {
    for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let config = LoraConfig {
            rank: 8,
            alpha,
            target_modules: vec![],
            dtype: "f32".into(),
        };

        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("alpha must be finite"));
        assert_eq!(config.scale(), 0.0);
    }
}

#[test]
fn test_adapter_construction_rejects_non_finite_alpha() {
    for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let result = LoraAdapter::new(
            LoraConfig {
                rank: 8,
                alpha,
                target_modules: vec![],
                dtype: "f32".into(),
            },
            HashMap::new(),
        );

        let err = result.expect_err("non-finite alpha must reject adapter construction");
        assert!(err.to_string().contains("alpha must be finite"));
    }
}

#[test]
fn test_adapter_apply() {
    let adapter = make_test_adapter();

    // x = [1, 2, 3, 4], base = [0, 0, 0, 0]
    // A @ x = [1, 2] (picks first 2 components)
    // B @ [1, 2] = [1, 2, 0, 0]
    // scale = 2.0
    // output = [0, 0, 0, 0] + 2.0 * [1, 2, 0, 0] = [2, 4, 0, 0]
    let x = [1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0f32; 4];
    adapter.apply(0, "q_proj", &x, &mut output);

    assert!((output[0] - 2.0).abs() < 1e-6);
    assert!((output[1] - 4.0).abs() < 1e-6);
    assert!((output[2] - 0.0).abs() < 1e-6);
    assert!((output[3] - 0.0).abs() < 1e-6);
}

#[test]
fn test_adapter_noop_for_missing_module() {
    let adapter = make_test_adapter();

    let x = [1.0, 2.0, 3.0, 4.0];
    let mut output = [10.0, 20.0, 30.0, 40.0];
    // v_proj at layer 0 has no adapter -> should be a no-op
    adapter.apply(0, "v_proj", &x, &mut output);

    assert!((output[0] - 10.0).abs() < 1e-6);
    assert!((output[1] - 20.0).abs() < 1e-6);
}

#[test]
fn test_adapter_noop_for_wrong_layer() {
    let adapter = make_test_adapter();

    let x = [1.0, 2.0, 3.0, 4.0];
    let mut output = [10.0, 20.0, 30.0, 40.0];
    // layer 1 has no adapter -> should be a no-op
    adapter.apply(1, "q_proj", &x, &mut output);

    assert!((output[0] - 10.0).abs() < 1e-6);
}

/// Regression for the per-row BERT LoRA dispatch path
/// (`lattice_inference::lora_hook::apply_lora_rows` calls `apply()` once
/// per token row): each row's output must depend only on that row's own
/// input, independent of how many rows came before it. This pins the
/// same per-row output the hash-keyed lookup produced before it was
/// replaced with a borrowed-key scan.
#[test]
fn test_adapter_apply_over_multiple_token_rows_matches_per_row_reference() {
    let adapter = make_test_adapter();

    let rows: [[f32; 4]; 3] = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [0.0; 4]];

    for row in rows {
        let mut output = [0.0f32; 4];
        adapter.apply(0, "q_proj", &row, &mut output);

        // Same reference as `test_adapter_apply`: A picks the first two
        // components of `x`, B is the 4x2 permutation built in
        // `make_test_adapter`, scale = 2.0.
        let expected = [2.0 * row[0], 2.0 * row[1], 0.0, 0.0];
        for (got, want) in output.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "got {output:?}, want {expected:?}"
            );
        }
    }
}

#[test]
fn test_has_adapter() {
    let adapter = make_test_adapter();
    assert!(adapter.has_adapter(0, "q_proj"));
    assert!(!adapter.has_adapter(0, "v_proj"));
    assert!(!adapter.has_adapter(1, "q_proj"));
}

#[test]
fn test_num_parameters() {
    let adapter = make_test_adapter();
    // A: 2*4 = 8, B: 4*2 = 8 => total 16
    assert_eq!(adapter.num_parameters(), 16);
}

#[test]
fn test_num_adapted_layers() {
    let adapter = make_test_adapter();
    assert_eq!(adapter.num_adapted_layers(), 1);
}

#[test]
fn test_validate_modules_all_known() {
    let adapter = make_test_adapter();
    let unknown = adapter.validate_modules(&["q_proj", "v_proj", "k_proj"]);
    assert!(unknown.is_empty());
}

#[test]
fn test_validate_modules_typo() {
    // `config.target_modules` names a real module ("q_proj") so
    // construction itself passes the shared-descriptor allowlist check;
    // the typo lives only in the populated *layer* key ("q_porj"), which
    // is exactly what `validate_modules` (a check over actual layers,
    // not the declared `target_modules` field) exists to catch.
    let config = LoraConfig {
        rank: 2,
        alpha: 4.0,
        target_modules: vec!["q_proj".into()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0, "q_porj".into()),
        LoraLayer {
            a: vec![1.0; 8],
            b: vec![1.0; 8],
            d_in: 4,
            d_out: 4,
            rank: 2,
        },
    );
    let adapter = LoraAdapter::new(config, layers).expect("valid adapter config");
    let unknown = adapter.validate_modules(&["q_proj", "v_proj"]);
    assert_eq!(unknown.len(), 1);
    assert_eq!(unknown[0], (0, "q_porj".to_string()));
}

#[test]
fn test_validate_modules_empty_adapter() {
    let config = LoraConfig {
        rank: 2,
        alpha: 4.0,
        target_modules: vec![],
        dtype: "f32".into(),
    };
    let adapter = LoraAdapter::new(config, HashMap::new()).expect("valid adapter config");
    let unknown = adapter.validate_modules(&["q_proj"]);
    assert!(unknown.is_empty());
}

/// Regression for lattice#972: a layer with *correct* `d_in`/`d_out` (what
/// `validate_against` checked before this fix) but a short `a` buffer
/// must still be rejected at construction, not admitted to later panic
/// (slice-out-of-bounds) inside `apply_lora`.
#[test]
fn test_new_rejects_a_buffer_shorter_than_rank_times_d_in() {
    let config = LoraConfig {
        rank: 4,
        alpha: 4.0,
        target_modules: vec!["q_proj".into()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0, "q_proj".into()),
        LoraLayer {
            a: vec![0.0; 4 * 8 - 1], // one element short of rank(4) * d_in(8)
            b: vec![0.0; 8 * 4],
            d_in: 8,
            d_out: 8,
            rank: 4,
        },
    );
    let err = LoraAdapter::new(config, layers)
        .expect_err("A buffer shorter than rank*d_in must be rejected");
    assert!(err.to_string().contains("A buffer length"));
}

/// Same as above for `b`, the other operand `apply_lora` slices by rank.
#[test]
fn test_new_rejects_b_buffer_longer_than_d_out_times_rank() {
    let config = LoraConfig {
        rank: 4,
        alpha: 4.0,
        target_modules: vec!["q_proj".into()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0, "q_proj".into()),
        LoraLayer {
            a: vec![0.0; 4 * 8],
            b: vec![0.0; 8 * 4 + 3], // padded beyond d_out(8) * rank(4)
            d_in: 8,
            d_out: 8,
            rank: 4,
        },
    );
    let err = LoraAdapter::new(config, layers)
        .expect_err("B buffer longer than d_out*rank must be rejected");
    assert!(err.to_string().contains("B buffer length"));
}

// -------------------------------------------------------------------
// Cross-crate contract: `LoraConfig` must actually carry
// `lattice_fann::lora::LoraDescriptor`, not just a same-shaped copy of
// it. `LoraAdapter::new` (the single construction chokepoint) routes
// `config.validate()` through `LoraConfig::to_descriptor()`, so a
// descriptor built independently and round-tripped through
// `LoraConfig::from_descriptor` / `to_descriptor` must come back
// unchanged. If a future change makes `LoraConfig` stop delegating to
// `LoraDescriptor` (e.g. reimplementing `scale`/`validate` locally),
// this test's round-trip breaks.
// -------------------------------------------------------------------
#[test]
fn lora_config_round_trips_through_shared_descriptor() {
    let descriptor = lattice_fann::lora::LoraDescriptor {
        rank: 8,
        alpha: 16.0,
        target_modules: vec!["q_proj".into(), "up_proj".into()],
        dtype: "bf16".into(),
    };
    let config = LoraConfig::from_descriptor(descriptor.clone());
    assert_eq!(config.to_descriptor(), descriptor);
    assert_eq!(config.scale(), descriptor.scale());
}

// -------------------------------------------------------------------
// `LoraAdapter::new` is the generic construction chokepoint
// (safetensors loading, blending, training all route through it) and is
// deliberately permissive of unrecognized target-module names: the
// adapter representation is model-neutral (see the module docs), and an
// externally produced adapter naming a projection this crate does not
// yet know by name must still load. Name recognition is enforced only
// where the target architecture is actually known:
// `new_with_descriptor` (below) and `validate_against`/
// `validate_against_bert` (see `validate_against_tests` and the BERT
// test module).
// -------------------------------------------------------------------
#[test]
fn lora_adapter_new_accepts_unrecognized_target_module() {
    let config = LoraConfig {
        rank: 2,
        alpha: 2.0,
        target_modules: vec!["not_a_real_module".into()],
        dtype: "f32".into(),
    };
    let adapter = LoraAdapter::new(config, HashMap::new())
        .expect("LoraAdapter::new must not reject an unrecognized target module name");
    assert_eq!(adapter.config().target_modules, ["not_a_real_module"]);
}

#[test]
fn new_with_descriptor_rejects_unrecognized_target_module() {
    // The layer set below matches `target_modules` exactly (same key,
    // same rank), so the pre-existing module-set-agreement check below
    // this one in the function cannot be what rejects the call — only
    // the allowlist check can. Without a matching layer this test would
    // pass for the wrong reason (empty `layers` vs. non-empty
    // `target_modules` already disagrees on its own).
    let descriptor = lattice_fann::lora::LoraDescriptor {
        rank: 2,
        alpha: 2.0,
        target_modules: vec!["not_a_real_module".into()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0, "not_a_real_module".into()),
        LoraLayer {
            a: vec![1.0; 2],
            b: vec![1.0; 2],
            d_in: 1,
            d_out: 1,
            rank: 2,
        },
    );
    let err = LoraAdapter::new_with_descriptor(descriptor, layers)
        .expect_err("new_with_descriptor must reject an unrecognized target module name");
    assert!(
        err.to_string().contains("not_a_real_module"),
        "error must name the unknown module via the shared validator; got: {err}"
    );
}

#[test]
fn new_with_descriptor_accepts_recognized_target_module() {
    // Run alongside the rejection test above so a validator that
    // rejects everything (rather than actually checking names) cannot
    // pass by accident.
    let descriptor = lattice_fann::lora::LoraDescriptor {
        rank: 2,
        alpha: 2.0,
        target_modules: vec!["q_proj".into()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0, "q_proj".into()),
        LoraLayer {
            a: vec![1.0; 2],
            b: vec![1.0; 2],
            d_in: 1,
            d_out: 1,
            rank: 2,
        },
    );
    assert!(LoraAdapter::new_with_descriptor(descriptor, layers).is_ok());
}

// -------------------------------------------------------------------
// Public-boundary tests for the descriptor/layer identity `Self::new`
// does not check: a descriptor declares `rank` and `target_modules`,
// but nothing before this constructor compared either against the
// layers actually supplied. The only in-tree caller that builds both a
// descriptor and its layers (`chat_metal`'s Metal-only LoRA loader)
// derives both from the same parsed tensors, so it can never produce a
// disagreement — these tests construct one directly.
// -------------------------------------------------------------------

#[test]
fn new_bare_constructor_permits_rank_disagreement_as_documented_escape_hatch() {
    // The exact shape from the round's failing input: a config
    // declaring rank=8 alongside a populated layer whose own rank is 4.
    // `LoraAdapter::new` only checks the layer's buffers against its
    // *own* declared rank/d_in/d_out, never against `config.rank`, so
    // this constructs successfully — the documented, caller-owned
    // bypass `new_with_descriptor` (below) closes for descriptor-backed
    // construction.
    let config = LoraConfig {
        rank: 8,
        alpha: 8.0,
        target_modules: vec!["q_proj".into()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0, "q_proj".into()),
        LoraLayer {
            a: vec![1.0; 4],
            b: vec![1.0; 4],
            d_in: 1,
            d_out: 1,
            rank: 4,
        },
    );
    let adapter =
        LoraAdapter::new(config, layers).expect("bare constructor does not cross-check rank");
    assert_eq!(adapter.config().rank, 8);
    assert_eq!(
        adapter
            .layers()
            .get(&(0, "q_proj".to_string()))
            .unwrap()
            .rank,
        4
    );
}

#[test]
fn new_with_descriptor_rejects_rank_disagreement() {
    let descriptor = lattice_fann::lora::LoraDescriptor {
        rank: 8,
        alpha: 8.0,
        target_modules: vec!["q_proj".into()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0, "q_proj".into()),
        LoraLayer {
            a: vec![1.0; 4],
            b: vec![1.0; 4],
            d_in: 1,
            d_out: 1,
            rank: 4,
        },
    );
    let err = LoraAdapter::new_with_descriptor(descriptor, layers)
        .expect_err("layer rank disagreeing with descriptor rank must be rejected");
    assert!(
        err.to_string()
            .contains("layer rank 4 disagrees with descriptor rank 8"),
        "error must name both ranks; got: {err}"
    );
}

#[test]
fn new_with_descriptor_accepts_matching_rank() {
    let descriptor = lattice_fann::lora::LoraDescriptor {
        rank: 4,
        alpha: 4.0,
        target_modules: vec!["q_proj".into()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0, "q_proj".into()),
        LoraLayer {
            a: vec![1.0; 4],
            b: vec![1.0; 4],
            d_in: 1,
            d_out: 1,
            rank: 4,
        },
    );
    assert!(LoraAdapter::new_with_descriptor(descriptor, layers).is_ok());
}

#[test]
fn new_with_descriptor_rejects_module_set_disagreement() {
    // Descriptor declares only "q_proj", but the layers cover "v_proj"
    // instead — the exact-module-set check `load_lora_adapter_with_descriptor`
    // performs on the Metal side, mirrored here for the tune side.
    let descriptor = lattice_fann::lora::LoraDescriptor {
        rank: 4,
        alpha: 4.0,
        target_modules: vec!["q_proj".into()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0, "v_proj".into()),
        LoraLayer {
            a: vec![1.0; 4],
            b: vec![1.0; 4],
            d_in: 1,
            d_out: 1,
            rank: 4,
        },
    );
    let err = LoraAdapter::new_with_descriptor(descriptor, layers)
        .expect_err("module set disagreeing with descriptor target_modules must be rejected");
    assert!(
        err.to_string().contains("disagree with descriptor"),
        "error must report the module-set disagreement; got: {err}"
    );
}

#[test]
fn new_with_descriptor_placeholder_layer_exempt_from_rank_check() {
    // An untrained placeholder (both `a` and `b` empty) still declares
    // its module in the map key, so it must count toward the module-set
    // match, but its `rank` field is not meaningful data to compare
    // against the descriptor.
    let descriptor = lattice_fann::lora::LoraDescriptor {
        rank: 4,
        alpha: 4.0,
        target_modules: vec!["q_proj".into()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0, "q_proj".into()),
        LoraLayer {
            a: vec![],
            b: vec![],
            d_in: 1,
            d_out: 1,
            rank: 999, // deliberately disagreeing, but exempt: no data
        },
    );
    assert!(LoraAdapter::new_with_descriptor(descriptor, layers).is_ok());
}

#[cfg(feature = "inference-hook")]
mod validate_against_tests {
    use super::*;
    use lattice_inference::model::qwen35_config::Qwen35Config;

    fn make_adapter_for_layer(
        layer_idx: usize,
        module: &str,
        d_in: usize,
        d_out: usize,
    ) -> LoraAdapter {
        let rank = 4;
        let mut layers = HashMap::new();
        layers.insert(
            (layer_idx, module.to_string()),
            LoraLayer {
                a: vec![0.0; rank * d_in],
                b: vec![0.0; d_out * rank],
                d_in,
                d_out,
                rank,
            },
        );
        LoraAdapter::new(
            LoraConfig {
                rank,
                alpha: rank as f32,
                target_modules: vec![module.to_string()],
                dtype: "f32".into(),
            },
            layers,
        )
        .expect("valid adapter config")
    }

    #[test]
    fn test_validate_against_layer_out_of_bounds() {
        let cfg = Qwen35Config::qwen35_0_8b();
        // layer 999 does not exist
        let adapter = make_adapter_for_layer(999, "q_proj", 1024, 4096);
        assert!(adapter.validate_against(&cfg).is_err());
    }

    #[test]
    fn test_validate_against_dim_mismatch() {
        let cfg = Qwen35Config::qwen35_0_8b();
        // 0.8b: hidden=1024, full_q_dim=8*256=2048 → q_proj expects (1024, 4096)
        // Supply 2b dims (hidden=2048, full_q_dim=4096 → d_out=8192) — wrong for 0.8b.
        // Layer 3 is full-attention in the 24-layer 0.8b config.
        let adapter = make_adapter_for_layer(3, "q_proj", 2048, 8192);
        assert!(adapter.validate_against(&cfg).is_err());
    }

    #[test]
    fn test_validate_against_correct_dims_passes() {
        let cfg = Qwen35Config::qwen35_0_8b();
        // Layer 3 is full-attention; q_proj: d_in=hidden=1024, d_out=2*full_q_dim=4096.
        let adapter = make_adapter_for_layer(3, "q_proj", 1024, 4096);
        assert!(adapter.validate_against(&cfg).is_ok());
    }

    #[test]
    fn test_validate_against_delegates_to_shared_qwen_projection_shape() {
        use lattice_inference::lora_hook::{LoraProjectionShape, qwen35_projection_shape};

        let cfg = Qwen35Config::qwen35_0_8b();
        let shared = qwen35_projection_shape(&cfg, 3, "q_proj")
            .expect("q_proj is valid on a full-attention layer");
        assert_eq!(
            shared,
            LoraProjectionShape {
                d_in: 1024,
                d_out: 4096,
            }
        );

        let matching = make_adapter_for_layer(3, "q_proj", shared.d_in, shared.d_out);
        assert!(matching.validate_against(&cfg).is_ok());

        let mismatched = make_adapter_for_layer(3, "q_proj", shared.d_in, shared.d_out - 1);
        let err = mismatched
            .validate_against(&cfg)
            .expect_err("Tune must reject geometry that differs from the shared shape");
        assert_eq!(
            err.to_string(),
            "Validation error: LoRA adapter dims mismatch for layer 3 module 'q_proj': \
                 adapter has (d_in=1024, d_out=4095) but model expects (d_in=1024, d_out=4096)"
        );
    }

    #[test]
    fn test_validate_against_mlp_correct() {
        let cfg = Qwen35Config::qwen35_0_8b();
        // gate_proj on any layer: d_in=hidden=1024, d_out=intermediate=3584.
        let adapter = make_adapter_for_layer(0, "gate_proj", 1024, 3584);
        assert!(adapter.validate_against(&cfg).is_ok());
    }

    #[test]
    fn test_validate_against_unknown_module_errors() {
        let cfg = Qwen35Config::qwen35_0_8b();
        // `make_adapter_for_layer` builds through `LoraAdapter::new`,
        // which accepts any target-module name (see
        // `lora_adapter_new_accepts_unrecognized_target_module`), so
        // both a name unknown to `KNOWN_LORA_TARGET_MODULES` and a name
        // like "query" that is known but wrong for Qwen3.5 reach this
        // point unchanged. "query" is used here (a real BERT projection
        // name) specifically to prove `validate_against` rejects it on
        // architecture grounds, not merely because the name is
        // unrecognized anywhere.
        let adapter = make_adapter_for_layer(3, "query", 1024, 4096);
        let err = adapter.validate_against(&cfg).unwrap_err();
        assert!(err.to_string().contains("not a recognised"));
    }

    #[test]
    fn test_validate_against_gdn_all_modules_pass() {
        // Regression: train_grad_full --save emits all five GDN LoRA modules
        // (in_proj_qkv/z/b/a, out_proj). The forward applies every one of them
        // (gdn_fused.rs), so validate_against must accept each with the dims the
        // loader and trainer use. Dims are derived from the config, not hardcoded,
        // so this stays correct if the reference dims change.
        let cfg = Qwen35Config::qwen35_0_8b();
        let gdn_layer = (0..cfg.num_hidden_layers)
            .find(|&i| !cfg.is_full_attention(i))
            .expect("0.8b config has linear-attention layers");
        let h = cfg.hidden_size;
        let cases = [
            ("in_proj_qkv", h, cfg.linear_qkv_dim()),
            ("in_proj_z", h, cfg.linear_output_dim()),
            ("in_proj_b", h, cfg.linear_num_value_heads()),
            ("in_proj_a", h, cfg.linear_num_value_heads()),
            ("out_proj", cfg.linear_output_dim(), h),
        ];
        for (module, d_in, d_out) in cases {
            let adapter = make_adapter_for_layer(gdn_layer, module, d_in, d_out);
            assert!(
                adapter.validate_against(&cfg).is_ok(),
                "GDN LoRA module {module} should validate (d_in={d_in}, d_out={d_out})"
            );
        }
    }

    /// Regression for #792: in_proj_b/in_proj_a
    /// dims must be keyed by `linear_num_value_heads()`, not
    /// `linear_num_key_heads`. Asymmetric-head configs (key_heads !=
    /// value_heads) are the only ones that can tell these apart —
    /// `qwen35_0_8b` has key_heads == value_heads == 16 and would pass
    /// either way.
    #[test]
    fn test_validate_against_gdn_asymmetric_value_heads_35b_a3b() {
        let cfg = Qwen35Config::qwen36_35b_a3b();
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads(), 32);
        let gdn_layer = (0..cfg.num_hidden_layers)
            .find(|&i| !cfg.is_full_attention(i))
            .expect("config has linear-attention layers");
        let h = cfg.hidden_size;

        // Correct dims (value_heads=32) must validate.
        let ok_b = make_adapter_for_layer(gdn_layer, "in_proj_b", h, 32);
        assert!(ok_b.validate_against(&cfg).is_ok());
        let ok_a = make_adapter_for_layer(gdn_layer, "in_proj_a", h, 32);
        assert!(ok_a.validate_against(&cfg).is_ok());

        // The old (wrong) key-head dim (16) must be rejected.
        let bad_b = make_adapter_for_layer(gdn_layer, "in_proj_b", h, 16);
        assert!(
            bad_b.validate_against(&cfg).is_err(),
            "in_proj_b with key-head dim (16) must fail on a 16-key/32-value config"
        );
        let bad_a = make_adapter_for_layer(gdn_layer, "in_proj_a", h, 16);
        assert!(
            bad_a.validate_against(&cfg).is_err(),
            "in_proj_a with key-head dim (16) must fail on a 16-key/32-value config"
        );
    }

    #[test]
    fn test_validate_against_gdn_asymmetric_value_heads_27b() {
        let cfg = Qwen35Config::qwen36_27b();
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads(), 48);
        let gdn_layer = (0..cfg.num_hidden_layers)
            .find(|&i| !cfg.is_full_attention(i))
            .expect("config has linear-attention layers");
        let h = cfg.hidden_size;

        // Correct dims (value_heads=48) must validate.
        let ok_b = make_adapter_for_layer(gdn_layer, "in_proj_b", h, 48);
        assert!(ok_b.validate_against(&cfg).is_ok());
        let ok_a = make_adapter_for_layer(gdn_layer, "in_proj_a", h, 48);
        assert!(ok_a.validate_against(&cfg).is_ok());

        // The old (wrong) key-head dim (16) must be rejected.
        let bad_b = make_adapter_for_layer(gdn_layer, "in_proj_b", h, 16);
        assert!(
            bad_b.validate_against(&cfg).is_err(),
            "in_proj_b with key-head dim (16) must fail on a 16-key/48-value config"
        );
        let bad_a = make_adapter_for_layer(gdn_layer, "in_proj_a", h, 16);
        assert!(
            bad_a.validate_against(&cfg).is_err(),
            "in_proj_a with key-head dim (16) must fail on a 16-key/48-value config"
        );
    }

    /// Regression for #753: `Qwen35Model::set_lora` rejects a mismatched
    /// adapter by calling through `dyn LoraHook::validate_against`, not
    /// the inherent `LoraAdapter::validate_against`. This confirms the
    /// trait-object glue (`impl LoraHook for LoraAdapter`) actually
    /// delegates to the inherent method instead of defaulting to the
    /// trait's no-op `Ok(())`, for both a mismatched and a matching
    /// adapter.
    #[test]
    fn test_lora_hook_trait_validate_against_delegates_to_inherent_method() {
        use lattice_inference::lora_hook::LoraHook;

        let cfg = Qwen35Config::qwen35_0_8b();

        // Same mismatch as `test_validate_against_dim_mismatch`: 2b dims
        // supplied against the 0.8b config's expected q_proj shape.
        // Fully-qualified syntax forces dispatch through the trait method
        // (not `LoraAdapter`'s own inherent `validate_against`, which dot
        // syntax would otherwise prefer).
        let mismatched = make_adapter_for_layer(3, "q_proj", 2048, 8192);
        assert!(
            LoraHook::validate_against(&mismatched, &cfg).is_err(),
            "the LoraHook trait method must surface the same dim mismatch as \
                 the inherent LoraAdapter::validate_against"
        );

        let matching = make_adapter_for_layer(3, "q_proj", 1024, 4096);
        assert!(
            LoraHook::validate_against(&matching, &cfg).is_ok(),
            "the LoraHook trait method must accept an adapter with correct dims"
        );
    }

    /// Regression for lattice#972: a layer whose
    /// projection dims exactly match a real Qwen3.5 model's `q_proj`
    /// (the only thing `validate_against` checked before this fix) but
    /// whose `a` buffer is short must be rejected before it can ever
    /// reach `validate_against` (and hence `Qwen35Model::set_lora`,
    /// which calls it — see `test_lora_hook_trait_validate_against_delegates_to_inherent_method`
    /// above). `LoraAdapter::new` is the sole construction path, so
    /// rejecting it there is equivalent to rejecting it at the
    /// `set_lora` gate: the malformed adapter never becomes an
    /// installable `LoraAdapter` value at all.
    #[test]
    fn test_new_rejects_malformed_buffer_matching_real_model_projection_dims() {
        // Layer 3 is full-attention in `Qwen35Config::qwen35_0_8b()`;
        // q_proj there expects d_in=hidden=1024, d_out=2*full_q_dim=4096
        // (see `test_validate_against_correct_dims_passes` above).
        let (layer_idx, module, d_in, d_out) = (3usize, "q_proj", 1024usize, 4096usize);
        let rank = 4;

        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec![module.to_string()],
            dtype: "f32".into(),
        };
        let mut layers = HashMap::new();
        layers.insert(
            (layer_idx, module.to_string()),
            LoraLayer {
                a: vec![0.0; rank * d_in - 1], // one short of rank*d_in
                b: vec![0.0; d_out * rank],
                d_in,
                d_out,
                rank,
            },
        );

        let err = LoraAdapter::new(config, layers).expect_err(
            "a malformed A buffer must be rejected at construction, even when \
                 d_in/d_out exactly match a real model's projection shape",
        );
        assert!(err.to_string().contains("A buffer length"));

        // No `LoraAdapter` value exists to pass to `validate_against` or
        // `set_lora` — construction itself is the gate here.
    }
}

/// BERT counterpart to `validate_against_tests`: covers
/// `LoraAdapter::validate_against_bert`, the geometry check
/// `CrossEncoderModel::score_with_hook` calls before hooked scoring
/// (lattice#1031 follow-up).
#[cfg(feature = "inference-hook")]
mod validate_against_bert_tests {
    use super::*;

    const NUM_HIDDEN_LAYERS: usize = 12;
    const HIDDEN_SIZE: usize = 384;
    const INTERMEDIATE_SIZE: usize = 1536;

    fn make_bert_adapter(module: &str, d_in: usize, d_out: usize) -> LoraAdapter {
        let rank = 4;
        let mut layers = HashMap::new();
        layers.insert(
            (0, module.to_string()),
            LoraLayer {
                a: vec![0.0; rank * d_in],
                b: vec![0.0; d_out * rank],
                d_in,
                d_out,
                rank,
            },
        );
        LoraAdapter::new(
            LoraConfig {
                rank,
                alpha: rank as f32,
                target_modules: vec![module.to_string()],
                dtype: "f32".into(),
            },
            layers,
        )
        .expect("valid adapter config")
    }

    #[test]
    fn test_validate_against_bert_oversized_d_out_rejected() {
        // The original bug: a self-consistent adapter declaring d_out >
        // hidden_size, which `apply_lora` would slice `output[..d_out]`
        // out of bounds on past a debug_assert release builds compile out.
        let adapter = make_bert_adapter("query", HIDDEN_SIZE, HIDDEN_SIZE + 1);
        let err = adapter
            .validate_against_bert(NUM_HIDDEN_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
            .expect_err("d_out > hidden_size must be rejected");
        assert!(err.to_string().contains("dims mismatch"));
    }

    #[test]
    fn test_validate_against_bert_mismatched_d_in_rejected() {
        // Even when d_out happens to fit, a wrong d_in is a
        // silent-wrong-math bug, not merely a panic risk.
        let adapter = make_bert_adapter("query", HIDDEN_SIZE + 1, HIDDEN_SIZE);
        let err = adapter
            .validate_against_bert(NUM_HIDDEN_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
            .expect_err("mismatched d_in must be rejected");
        assert!(err.to_string().contains("dims mismatch"));
    }

    #[test]
    fn test_validate_against_bert_layer_out_of_bounds_rejected() {
        let adapter = make_bert_adapter("query", HIDDEN_SIZE, HIDDEN_SIZE);
        let err = adapter
            .validate_against_bert(0, HIDDEN_SIZE, INTERMEDIATE_SIZE)
            .expect_err("layer 0 >= num_hidden_layers 0 must be rejected");
        assert!(err.to_string().contains("num_hidden_layers"));
    }

    #[test]
    fn test_validate_against_bert_unknown_module_rejected() {
        // `make_bert_adapter` builds through `LoraAdapter::new`, which
        // accepts any target-module name (see
        // `lora_adapter_new_accepts_unrecognized_target_module`). "q_proj"
        // is used here (a real Qwen3.5 projection name, so it is not
        // merely unrecognized anywhere) specifically to prove
        // `validate_against_bert` rejects it on architecture grounds.
        let adapter = make_bert_adapter("q_proj", HIDDEN_SIZE, HIDDEN_SIZE);
        let err = adapter
            .validate_against_bert(NUM_HIDDEN_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
            .expect_err("unrecognised BERT module must be rejected");
        assert!(err.to_string().contains("not a recognised"));
    }

    #[test]
    fn test_validate_against_bert_all_modules_correct_dims_pass() {
        let cases = [
            ("query", HIDDEN_SIZE, HIDDEN_SIZE),
            ("key", HIDDEN_SIZE, HIDDEN_SIZE),
            ("value", HIDDEN_SIZE, HIDDEN_SIZE),
            ("attn_output", HIDDEN_SIZE, HIDDEN_SIZE),
            ("ffn_intermediate", HIDDEN_SIZE, INTERMEDIATE_SIZE),
            ("ffn_output", INTERMEDIATE_SIZE, HIDDEN_SIZE),
        ];
        for (module, d_in, d_out) in cases {
            let adapter = make_bert_adapter(module, d_in, d_out);
            assert!(
                adapter
                    .validate_against_bert(NUM_HIDDEN_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
                    .is_ok(),
                "module {module} with correct dims (d_in={d_in}, d_out={d_out}) should validate"
            );
        }
    }

    #[test]
    fn test_lora_hook_trait_validate_against_bert_delegates_to_inherent_method() {
        use lattice_inference::lora_hook::LoraHook;

        let mismatched = make_bert_adapter("query", HIDDEN_SIZE, HIDDEN_SIZE + 1);
        assert!(
            LoraHook::validate_against_bert(
                &mismatched,
                NUM_HIDDEN_LAYERS,
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE
            )
            .is_err(),
            "the LoraHook trait method must surface the same dim mismatch as \
                 the inherent LoraAdapter::validate_against_bert"
        );

        let matching = make_bert_adapter("query", HIDDEN_SIZE, HIDDEN_SIZE);
        assert!(
            LoraHook::validate_against_bert(
                &matching,
                NUM_HIDDEN_LAYERS,
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE
            )
            .is_ok(),
            "the LoraHook trait method must accept an adapter with correct dims"
        );
    }

    /// `LoraHook::is_active` must reflect whether this adapter actually
    /// has a layer for `(layer_idx, module)`, so `apply_lora_rows` can
    /// skip its per-row loop entirely for projections this adapter
    /// doesn't touch, instead of paying one no-op virtual call per row.
    #[test]
    fn test_lora_hook_trait_is_active_reflects_has_adapter() {
        use lattice_inference::lora_hook::LoraHook;

        let adapter = make_bert_adapter("query", HIDDEN_SIZE, HIDDEN_SIZE);
        assert!(
            LoraHook::is_active(&adapter, 0, "query"),
            "adapter has a layer for (0, query)"
        );
        assert!(
            !LoraHook::is_active(&adapter, 0, "key"),
            "adapter has no layer for (0, key)"
        );
        assert!(
            !LoraHook::is_active(&adapter, 5, "query"),
            "adapter has no layer for (5, query)"
        );
    }

    /// A layer with an empty `a` factor but a non-empty `b` factor and
    /// declared dims that match a real BERT projection's geometry is not
    /// a valid "untrained placeholder" (that state requires BOTH `a` and
    /// `b` empty) — it must be rejected at construction rather than
    /// reaching `apply_lora`, which indexes into `a` assuming it holds
    /// `rank * d_in` elements.
    #[test]
    fn test_new_rejects_empty_a_factor_with_matching_declared_dims() {
        let rank = 4;
        let mut layers = HashMap::new();
        layers.insert(
            (0, "query".to_string()),
            LoraLayer {
                a: vec![],
                b: vec![0.0; HIDDEN_SIZE * rank],
                d_in: HIDDEN_SIZE,
                d_out: HIDDEN_SIZE,
                rank,
            },
        );
        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["query".to_string()],
            dtype: "f32".into(),
        };
        let err = LoraAdapter::new(config, layers)
            .expect_err("an empty A factor with a populated B factor must be rejected");
        assert!(err.to_string().contains("A buffer length"));
    }

    /// Mirror of the above for an empty `b` factor with a populated `a`.
    #[test]
    fn test_new_rejects_empty_b_factor_with_matching_declared_dims() {
        let rank = 4;
        let mut layers = HashMap::new();
        layers.insert(
            (0, "query".to_string()),
            LoraLayer {
                a: vec![0.0; rank * HIDDEN_SIZE],
                b: vec![],
                d_in: HIDDEN_SIZE,
                d_out: HIDDEN_SIZE,
                rank,
            },
        );
        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["query".to_string()],
            dtype: "f32".into(),
        };
        let err = LoraAdapter::new(config, layers)
            .expect_err("an empty B factor with a populated A factor must be rejected");
        assert!(err.to_string().contains("B buffer length"));
    }

    /// A layer with BOTH `a` and `b` empty is the legitimate
    /// untrained-placeholder state (mirrors `save_peft_safetensors`,
    /// which skips these layers) and must still construct successfully.
    #[test]
    fn test_new_accepts_fully_empty_placeholder_layer() {
        let rank = 4;
        let mut layers = HashMap::new();
        layers.insert(
            (0, "query".to_string()),
            LoraLayer {
                a: vec![],
                b: vec![],
                d_in: HIDDEN_SIZE,
                d_out: HIDDEN_SIZE,
                rank,
            },
        );
        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["query".to_string()],
            dtype: "f32".into(),
        };
        assert!(
            LoraAdapter::new(config, layers).is_ok(),
            "a placeholder layer with both factors empty must still construct"
        );
    }

    /// A rank above `min(d_in, d_out)` is representationally redundant
    /// (`B @ A` cannot exceed that rank no matter how wide the inner
    /// dimension is) but perfectly valid: the factorization computes a
    /// correct update, `apply_lora` derives every buffer offset from
    /// `rank` itself, and `blend_lora_adapters` produces exactly this
    /// shape by summing source ranks. `validate_against_bert` must
    /// accept it. `HIDDEN_SIZE` is 384, so this uses a rank one above
    /// that; the buffers stay consistent with the declared rank, so the
    /// redundancy is the only thing left for a validator to object to.
    #[test]
    fn test_validate_against_bert_accepts_over_complete_rank() {
        let rank = HIDDEN_SIZE + 1;
        let mut layers = HashMap::new();
        layers.insert(
            (0, "query".to_string()),
            LoraLayer {
                a: vec![0.0; rank * HIDDEN_SIZE],
                b: vec![0.0; HIDDEN_SIZE * rank],
                d_in: HIDDEN_SIZE,
                d_out: HIDDEN_SIZE,
                rank,
            },
        );
        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["query".to_string()],
            dtype: "f32".into(),
        };
        let adapter = LoraAdapter::new(config, layers).expect("buffers match the declared rank");
        assert!(
            adapter
                .validate_against_bert(NUM_HIDDEN_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
                .is_ok(),
            "an over-complete but correctly shaped factorization must be accepted"
        );
    }

    /// The buffer-geometry invariant is what actually keeps `apply_lora`
    /// in bounds, and it is independent of the redundancy question:
    /// removing the rank ceiling must not let a layer through whose
    /// buffers disagree with its own declared `rank`/`d_in`/`d_out`.
    /// Asserted at construction, which is where that check lives.
    #[test]
    fn test_over_complete_rank_still_requires_matching_buffer_lengths() {
        let rank = HIDDEN_SIZE + 1;
        let mut layers = HashMap::new();
        layers.insert(
            (0, "query".to_string()),
            LoraLayer {
                // One row short of `rank * d_in`: an over-complete rank
                // does not excuse an under-sized A factor.
                a: vec![0.0; (rank - 1) * HIDDEN_SIZE],
                b: vec![0.0; HIDDEN_SIZE * rank],
                d_in: HIDDEN_SIZE,
                d_out: HIDDEN_SIZE,
                rank,
            },
        );
        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["query".to_string()],
            dtype: "f32".into(),
        };
        let err = LoraAdapter::new(config, layers)
            .expect_err("an A buffer shorter than rank * d_in must be rejected");
        assert!(
            err.to_string().contains("A buffer length"),
            "expected the A-buffer-length rejection, got: {err}"
        );
    }

    /// A geometry mismatch against the model's BERT dimensions must
    /// still be rejected at an over-complete rank — the two conditions
    /// are independent and the removed ceiling was never what caught
    /// this.
    #[test]
    fn test_over_complete_rank_does_not_excuse_bert_dim_mismatch() {
        let rank = HIDDEN_SIZE + 1;
        let mut layers = HashMap::new();
        layers.insert(
            (0, "query".to_string()),
            LoraLayer {
                a: vec![0.0; rank * HIDDEN_SIZE],
                b: vec![0.0; (HIDDEN_SIZE + 1) * rank],
                d_in: HIDDEN_SIZE,
                d_out: HIDDEN_SIZE + 1,
                rank,
            },
        );
        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["query".to_string()],
            dtype: "f32".into(),
        };
        let adapter = LoraAdapter::new(config, layers).expect("buffers match the declared rank");
        let err = adapter
            .validate_against_bert(NUM_HIDDEN_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
            .expect_err("d_out != hidden_size must be rejected whatever the rank");
        assert!(
            err.to_string().contains("dims mismatch"),
            "expected the BERT dim-mismatch rejection, got: {err}"
        );
    }

    /// A rank exactly at the bound (`min(d_in, d_out)`) is the largest
    /// non-redundant value and must still pass.
    #[test]
    fn test_validate_against_bert_accepts_rank_at_min_d_in_d_out_bound() {
        let rank = HIDDEN_SIZE;
        let mut layers = HashMap::new();
        layers.insert(
            (0, "query".to_string()),
            LoraLayer {
                a: vec![0.0; rank * HIDDEN_SIZE],
                b: vec![0.0; HIDDEN_SIZE * rank],
                d_in: HIDDEN_SIZE,
                d_out: HIDDEN_SIZE,
                rank,
            },
        );
        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["query".to_string()],
            dtype: "f32".into(),
        };
        let adapter = LoraAdapter::new(config, layers).expect("valid adapter config");
        assert!(
            adapter
                .validate_against_bert(NUM_HIDDEN_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
                .is_ok(),
            "rank exactly at min(d_in, d_out) must be accepted"
        );
    }

    /// An empty placeholder layer (see
    /// `test_new_accepts_fully_empty_placeholder_layer`) is present in
    /// the map but carries no correction; `has_adapter` (and the
    /// `LoraHook::is_active` delegate `apply_lora_rows` uses to skip
    /// per-row dispatch) must report it as inactive rather than
    /// present-but-inert.
    #[test]
    fn test_has_adapter_false_for_empty_placeholder_layer() {
        let rank = 4;
        let mut layers = HashMap::new();
        layers.insert(
            (0, "query".to_string()),
            LoraLayer {
                a: vec![],
                b: vec![],
                d_in: HIDDEN_SIZE,
                d_out: HIDDEN_SIZE,
                rank,
            },
        );
        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["query".to_string()],
            dtype: "f32".into(),
        };
        let adapter = LoraAdapter::new(config, layers).expect("placeholder layer must construct");
        assert!(
            !adapter.has_adapter(0, "query"),
            "an empty placeholder layer must not report as active"
        );
    }
}
