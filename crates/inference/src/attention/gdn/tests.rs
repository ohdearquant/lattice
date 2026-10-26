use super::*;
use crate::attention::gdn_fused::{GatedDeltaNetFusedScratch, gated_delta_net_step_fused};
use crate::lora_hook::NoopLoraHook;

fn presets() -> [(&'static str, Qwen35Config); 4] {
    [
        ("qwen35_0_8b", Qwen35Config::qwen35_0_8b()),
        ("qwen35_2b", Qwen35Config::qwen35_2b()),
        ("qwen36_27b", Qwen35Config::qwen36_27b()),
        ("qwen36_35b_a3b", Qwen35Config::qwen36_35b_a3b()),
    ]
}

/// A small configuration whose value heads are twice its key heads.
fn small_config() -> Qwen35Config {
    let mut cfg = Qwen35Config::qwen35_2b();
    cfg.hidden_size = 48;
    cfg.linear_num_key_heads = 2;
    cfg.linear_num_value_heads = Some(4);
    cfg.linear_key_head_dim = 8;
    cfg.linear_value_head_dim = 6;
    cfg.linear_conv_kernel_dim = 4;
    cfg
}

fn filled(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            ((x % 2001) as f32 / 1000.0 - 1.0) * scale
        })
        .collect()
}

fn weights_for(cfg: &Qwen35Config) -> GatedDeltaNetWeights {
    let hidden = cfg.hidden_size;
    let value_heads = cfg.linear_num_value_heads();
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let kernel_size = cfg.linear_conv_kernel_dim;
    GatedDeltaNetWeights {
        in_proj_qkv: filled(qkv_dim * hidden, 1, 0.05),
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z: filled(output_dim * hidden, 2, 0.05),
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b: filled(value_heads * hidden, 3, 0.05),
        in_proj_b_rows: value_heads,
        in_proj_b_cols: hidden,
        in_proj_a: filled(value_heads * hidden, 4, 0.05),
        in_proj_a_rows: value_heads,
        in_proj_a_cols: hidden,
        a_log: filled(value_heads, 5, 0.1)
            .iter()
            .map(|a| a - 1.0)
            .collect(),
        dt_bias: filled(value_heads, 6, 0.05),
        conv1d_weight: filled(qkv_dim * kernel_size, 7, 0.05),
        conv_dim: qkv_dim,
        kernel_size,
        norm_weight: filled(cfg.linear_value_head_dim, 8, 0.01)
            .iter()
            .map(|g| g + 1.0)
            .collect(),
        out_proj: filled(hidden * output_dim, 9, 0.05),
        out_proj_rows: hidden,
        out_proj_cols: output_dim,
    }
}

#[test]
fn qwen35_gdn_geometry_reads_every_preset() {
    for (name, cfg) in presets() {
        let geometry = qwen35_gdn_geometry(&cfg).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            geometry.dimensions(),
            GdnDimensions {
                hidden_size: cfg.hidden_size,
                num_key_heads: cfg.linear_num_key_heads,
                num_value_heads: cfg.linear_num_value_heads(),
                key_head_dim: cfg.linear_key_head_dim,
                value_head_dim: cfg.linear_value_head_dim,
                conv_kernel_dim: cfg.linear_conv_kernel_dim,
                rms_norm_eps: cfg.rms_norm_eps,
            },
            "{name}"
        );
        assert_eq!(geometry.qkv_dim(), cfg.linear_qkv_dim(), "{name}");
        assert_eq!(geometry.output_dim(), cfg.linear_output_dim(), "{name}");
    }
}

#[test]
fn qwen35_gdn_geometry_falls_back_to_the_default_value_head_count() {
    let mut cfg = Qwen35Config::qwen35_2b();
    cfg.linear_num_value_heads = None;
    let geometry = qwen35_gdn_geometry(&cfg).expect("default value heads form a valid geometry");
    assert_eq!(geometry.num_value_heads(), cfg.linear_num_value_heads());
}

#[test]
fn from_geometry_matches_the_deprecated_constructor_for_every_preset() {
    for (name, cfg) in presets() {
        let geometry = qwen35_gdn_geometry(&cfg).unwrap_or_else(|e| panic!("{name}: {e}"));
        let neutral = GatedDeltaNetState::from_geometry(&geometry);
        // The deprecated constructor stays through 0.15.0; this pins it to its replacement.
        #[allow(deprecated)]
        let legacy = GatedDeltaNetState::new(&cfg);
        assert_eq!(neutral.s_matrices.len(), legacy.s_matrices.len(), "{name}");
        assert_eq!(
            neutral.conv_buffer.len(),
            legacy.conv_buffer.len(),
            "{name}"
        );
        assert_eq!(format!("{neutral:?}"), format!("{legacy:?}"), "{name}");
    }
}

#[test]
fn qwen35_gdn_geometry_rejects_invalid_configurations() {
    type Mutation = fn(&mut Qwen35Config);
    let cases: [(&str, Mutation); 5] = [
        ("hidden_size must be > 0", |cfg| cfg.hidden_size = 0),
        ("conv_kernel_dim must be > 0", |cfg| {
            cfg.linear_conv_kernel_dim = 0
        }),
        ("value_head_dim must be > 0", |cfg| {
            cfg.linear_value_head_dim = 0
        }),
        ("must be a multiple of num_key_heads", |cfg| {
            cfg.linear_num_value_heads = Some(24)
        }),
        ("rms_norm_eps", |cfg| cfg.rms_norm_eps = 0.0),
    ];
    for (expected, mutate) in cases {
        let mut cfg = Qwen35Config::qwen35_2b();
        mutate(&mut cfg);
        match qwen35_gdn_geometry(&cfg) {
            Err(InferenceError::InvalidInput(message)) => {
                assert!(
                    message.starts_with("invalid GatedDeltaNet geometry: ")
                        && message.contains(expected),
                    "{expected}: {message}"
                );
            }
            other => panic!("{expected}: expected InvalidInput, got {other:?}"),
        }
    }
}

#[test]
fn gated_delta_net_step_matches_the_geometry_kernel_bit_for_bit() {
    let cfg = small_config();
    let geometry = qwen35_gdn_geometry(&cfg).expect("small configuration is valid");
    let weights = weights_for(&cfg);
    let mut wrapped = GatedDeltaNetState::from_geometry(&geometry);
    let mut direct = GatedDeltaNetState::from_geometry(&geometry);
    let mut wrapped_scratch = GatedDeltaNetScratch::default();
    let mut direct_scratch = GatedDeltaNetScratch::default();
    for step in 0..5u32 {
        let input = filled(cfg.hidden_size, 100 + step, 1.0);
        let mut wrapped_out = vec![0.0f32; cfg.hidden_size];
        let mut direct_out = vec![0.0f32; cfg.hidden_size];
        gated_delta_net_step(
            &input,
            &mut wrapped,
            &weights,
            &cfg,
            &mut wrapped_scratch,
            &mut wrapped_out,
        );
        lattice_compute::attention::gdn::gated_delta_net_step(
            &input,
            &mut direct,
            &weights,
            &geometry,
            &mut direct_scratch,
            &mut direct_out,
        );
        assert!(wrapped_out.iter().any(|v| *v != 0.0), "step {step}");
        assert_eq!(
            wrapped_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            direct_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "step {step}"
        );
    }
    assert_eq!(format!("{wrapped:?}"), format!("{direct:?}"));
}

#[test]
fn gated_delta_net_step_fused_matches_the_geometry_kernel_bit_for_bit() {
    let cfg = small_config();
    let geometry = qwen35_gdn_geometry(&cfg).expect("small configuration is valid");
    let weights = weights_for(&cfg);
    let mut wrapped = GatedDeltaNetState::from_geometry(&geometry);
    let mut direct = GatedDeltaNetState::from_geometry(&geometry);
    let mut wrapped_scratch = GatedDeltaNetFusedScratch::default();
    let mut direct_scratch = GatedDeltaNetFusedScratch::default();
    for step in 0..5u32 {
        let input = filled(cfg.hidden_size, 200 + step, 1.0);
        let mut wrapped_out = vec![0.0f32; cfg.hidden_size];
        let mut direct_out = vec![0.0f32; cfg.hidden_size];
        gated_delta_net_step_fused(
            &input,
            &mut wrapped,
            &weights,
            &cfg,
            &mut wrapped_scratch,
            &mut wrapped_out,
            &NoopLoraHook,
            3,
        );
        lattice_compute::attention::gdn_fused::gated_delta_net_step_fused(
            &input,
            &mut direct,
            &weights,
            &geometry,
            &mut direct_scratch,
            &mut direct_out,
            &NoopLoraHook,
            3,
        );
        assert!(wrapped_out.iter().any(|v| *v != 0.0), "step {step}");
        assert_eq!(
            wrapped_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            direct_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "step {step}"
        );
    }
    assert_eq!(format!("{wrapped:?}"), format!("{direct:?}"));
}

#[test]
#[should_panic(expected = "invalid GatedDeltaNet geometry: conv_kernel_dim must be > 0")]
fn gated_delta_net_step_panics_on_an_invalid_geometry() {
    let mut cfg = small_config();
    cfg.linear_conv_kernel_dim = 0;
    let mut state = GatedDeltaNetState::from_geometry(
        &qwen35_gdn_geometry(&small_config()).expect("small configuration is valid"),
    );
    let mut output = vec![0.0f32; cfg.hidden_size];
    gated_delta_net_step(
        &vec![0.0f32; cfg.hidden_size],
        &mut state,
        &weights_for(&small_config()),
        &cfg,
        &mut GatedDeltaNetScratch::default(),
        &mut output,
    );
}

#[test]
#[should_panic(expected = "invalid GatedDeltaNet geometry: conv_kernel_dim must be > 0")]
fn gated_delta_net_step_fused_panics_on_an_invalid_geometry() {
    let mut cfg = small_config();
    cfg.linear_conv_kernel_dim = 0;
    let mut state = GatedDeltaNetState::from_geometry(
        &qwen35_gdn_geometry(&small_config()).expect("small configuration is valid"),
    );
    let mut output = vec![0.0f32; cfg.hidden_size];
    gated_delta_net_step_fused(
        &vec![0.0f32; cfg.hidden_size],
        &mut state,
        &weights_for(&small_config()),
        &cfg,
        &mut GatedDeltaNetFusedScratch::default(),
        &mut output,
        &NoopLoraHook,
        0,
    );
}
