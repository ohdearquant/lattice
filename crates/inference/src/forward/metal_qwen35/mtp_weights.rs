// -----------------------------------------------------------------------------
// MTP Q4/F16 tensor resolution (#630, #636) -- pure, `Device`-free, compiled
// unconditionally in test builds (these previously lived inside
// `mod inner`, gated on `metal-gpu`, so `cargo test -p lattice-inference --lib
// resolve_mtp` with no features reported "running 0 tests" -- the exact
// skip-pass class described below, just one layer up. `mod inner` re-imports
// these via `use super::mtp_weights::{...}` below.)
// -----------------------------------------------------------------------------

/// Build the sanitized MTP tensor file path (dots/slashes -> underscores),
/// mirroring `MetalQwen35State::q4_tensor_path` -- duplicated here (rather than
/// called through `MetalQwen35State`, which only exists under `metal-gpu`) so
/// this module compiles and is unit-testable without the Metal feature at all
/// (#636).
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(super) fn mtp_tensor_path(
    dir: &std::path::Path,
    tensor_name: &str,
    ext: &str,
) -> std::path::PathBuf {
    let sanitized: String = tensor_name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    dir.join(format!("{sanitized}.{ext}"))
}

/// Failure modes for resolving a single MTP tensor from disk (#636).
///
/// `Missing` is the pre-existing graceful case: the caller (the MTP
/// projection-load closure in [`load_mtp_q4_weights`]) turns it into a
/// warn-and-fall-back-to-non-MTP-decode result via
/// [`mtp_missing_weights_warning`]. `Incompatible` is new and
/// deliberately NOT graceful: it means the on-disk artifact exists but
/// cannot be trusted (a stale `.q4` sibling in a QuaRot rotated-space
/// checkpoint, or a shape/transpose mismatch) — silently disabling MTP
/// would be safe, but silently *loading* it would produce garbage
/// drafts, so this propagates as a hard `from_q4_dir` load error
/// instead.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
#[derive(Debug)]
pub(super) enum MtpLoadErr {
    // The path is read by the real `metal-gpu` production caller
    // (`load_mtp_q4_weights`) but only via `matches!` in the pure
    // default-feature-only tests, so it looks unused in that specific
    // build configuration.
    Missing(#[allow(dead_code)] std::path::PathBuf),
    Incompatible(String),
}

#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
impl From<std::path::PathBuf> for MtpLoadErr {
    fn from(path: std::path::PathBuf) -> Self {
        MtpLoadErr::Missing(path)
    }
}

/// One MTP tensor's values as read from disk, before Metal buffer
/// creation — deliberately `Device`-free so flavor selection and shape
/// validation are unit-testable without a GPU (#636).
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
#[derive(Debug)]
pub(super) enum MtpTensorSource {
    F16 {
        // Read by the real `metal-gpu` production caller
        // (`load_mtp_q4_weights`'s `load_proj_as_half`); only the
        // `#[cfg(test)]`-only `shape()`/`is_q4()` accessors read this in a
        // default-feature (no metal-gpu) test-only build.
        #[allow(dead_code)]
        values: Vec<f32>,
        #[allow(dead_code)] // read via `shape()`, only exercised by `#[cfg(test)]` callers
        shape: Vec<usize>,
    },
    Q4(crate::weights::q4_weights::Q4Tensor),
}

#[cfg(test)]
impl MtpTensorSource {
    /// Test-only accessor for the resolved on-disk shape, used to assert
    /// which flavor + shape a `resolve_mtp_projection` call picked
    /// without needing a Metal `Device`.
    fn shape(&self) -> &[usize] {
        match self {
            MtpTensorSource::F16 { shape, .. } => shape,
            MtpTensorSource::Q4(tensor) => &tensor.shape,
        }
    }

    fn is_q4(&self) -> bool {
        matches!(self, MtpTensorSource::Q4(_))
    }
}

/// Resolve one of the 8 MTP projection matrices from `q4_dir`, pure
/// CPU I/O only (no `Device`).
///
/// `quantize_q4` applies the same `should_quantize` name-suffix rule to
/// MTP tensors as to the main model, so the 8 `*_proj*.weight` MTP
/// matrices are written as `.q4` on a plain (non-rotated) Q4 checkpoint.
/// `quantize_quarot` instead keeps every MTP tensor as unrotated `.f16`
/// (ADR-051 Phase 1: MTP counter-rotation needs unquantized weights),
/// and MUST NOT have a `.q4` sibling for any MTP tensor.
///
/// `allow_q4` therefore encodes the checkpoint flavor, determined by
/// the caller BEFORE any projection is loaded (not sniffed per-file):
/// `true` for a plain Q4 checkpoint (no QuaRot rotation seed anywhere),
/// `false` for a QuaRot checkpoint. When `allow_q4` is `false` and a
/// `.q4` sibling exists anyway, that is an incompatible stale artifact
/// — `quantize_q4` was re-run into a QuaRot output directory (the
/// converter reuses a non-empty output dir rather than rejecting it),
/// producing rotated-space `.q4` weights that the runtime's
/// counter-rotation (keyed off the same directory's `quarot_seed`)
/// would silently apply on top of, corrupting every MTP draft without
/// any load failure to reveal it (#636). This is
/// rejected loudly instead of silently preferring `.f16`.
///
/// Also validates the resolved tensor's on-disk shape against
/// `expected_shape`: a same-numel transposed file loads and generates
/// tokens but produces garbage MTP drafts (#636), so
/// shape mismatches are rejected the same way as an incompatible
/// artifact rather than silently accepted.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(super) fn resolve_mtp_projection(
    q4_dir: &std::path::Path,
    name: &str,
    expected_shape: &[usize],
    allow_q4: bool,
) -> Result<MtpTensorSource, MtpLoadErr> {
    use crate::weights::q4_weights::{load_f16_tensor_file, load_q4_file};

    let q4_path = mtp_tensor_path(q4_dir, name, "q4");
    if q4_path.exists() {
        if !allow_q4 {
            return Err(MtpLoadErr::Incompatible(format!(
                "{}: incompatible stale MTP .q4 artifact found alongside a QuaRot \
                 (rotated-space) checkpoint; QuaRot Phase 1 requires MTP projections \
                 to remain unrotated .f16 — a .q4 sibling here means quantize_q4 was \
                 re-run into a QuaRot output directory, and loading it would silently \
                 apply counter-rotation on top of already-rotated-space weights, \
                 corrupting every MTP draft without a load failure to reveal it. \
                 Refusing to load — delete the stale .q4 file or regenerate the \
                 checkpoint.",
                q4_path.display()
            )));
        }
        let tensor = load_q4_file(&q4_path).map_err(|_| MtpLoadErr::Missing(q4_path.clone()))?;
        if tensor.shape != expected_shape {
            return Err(MtpLoadErr::Incompatible(format!(
                "{}: MTP tensor '{name}' has shape {:?}, expected {expected_shape:?} — \
                 refusing to load (a mismatched/transposed weight file loads and \
                 generates tokens but silently produces garbage MTP drafts)",
                q4_path.display(),
                tensor.shape
            )));
        }
        return Ok(MtpTensorSource::Q4(tensor));
    }

    let f16_path = mtp_tensor_path(q4_dir, name, "f16");
    if !f16_path.exists() {
        return Err(MtpLoadErr::Missing(f16_path));
    }
    let (values, shape) =
        load_f16_tensor_file(&f16_path).map_err(|_| MtpLoadErr::Missing(f16_path.clone()))?;
    if shape != expected_shape {
        return Err(MtpLoadErr::Incompatible(format!(
            "{}: MTP tensor '{name}' has shape {shape:?}, expected {expected_shape:?} — \
             refusing to load (a mismatched/transposed weight file loads and generates \
             tokens but silently produces garbage MTP drafts)",
            f16_path.display()
        )));
    }
    Ok(MtpTensorSource::F16 { values, shape })
}

/// Resolve one of the 7 MTP norm tensors from `q4_dir`, pure CPU I/O
/// only (no `Device`). Norms are never quantized by `should_quantize`
/// in either quantizer, so they are always `.f16`-only regardless of
/// checkpoint flavor; a `.q4` sibling under a norm's name is treated
/// as an incompatible artifact rather than silently ignored (defense
/// in depth — should never occur on a checkpoint from either
/// quantizer). Also validates on-disk shape (#636).
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(super) fn resolve_mtp_norm(
    q4_dir: &std::path::Path,
    name: &str,
    expected_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>), MtpLoadErr> {
    use crate::weights::q4_weights::load_f16_tensor_file;

    let q4_path = mtp_tensor_path(q4_dir, name, "q4");
    if q4_path.exists() {
        return Err(MtpLoadErr::Incompatible(format!(
            "{}: unexpected MTP .q4 artifact for norm tensor '{name}' — norms are never \
             quantized by either quantizer; refusing to load an incompatible checkpoint",
            q4_path.display()
        )));
    }
    let f16_path = mtp_tensor_path(q4_dir, name, "f16");
    if !f16_path.exists() {
        return Err(MtpLoadErr::Missing(f16_path));
    }
    let (values, shape) =
        load_f16_tensor_file(&f16_path).map_err(|_| MtpLoadErr::Missing(f16_path.clone()))?;
    if shape != expected_shape {
        return Err(MtpLoadErr::Incompatible(format!(
            "{}: MTP norm tensor '{name}' has shape {shape:?}, expected \
             {expected_shape:?} — refusing to load",
            f16_path.display()
        )));
    }
    Ok((values, shape))
}

#[cfg(test)]
pub(super) mod mtp_resolve_tests {
    use super::*;
    use crate::model::qwen35_config::{LayerType, Qwen35Config};

    /// Minimal `Qwen35Config` for the pure `resolve_mtp_*` tests: no
    /// `ModelWeights`/Metal `Device` construction at all (unlike
    /// `tiny_metal_qwen35_fixture`, which lives inside `mod inner` and is
    /// gated on the `metal-gpu` feature -- these tests must not depend on
    /// it, or they'd silently disappear again on a Metal-less/no-feature
    /// build, exactly the skip-pass class described above). Field values
    /// mirror `tiny_metal_qwen35_fixture`'s config so the derived MTP
    /// projection/norm shapes match the same real-checkpoint-verified
    /// formulas.
    fn tiny_mtp_test_config() -> Qwen35Config {
        Qwen35Config {
            hidden_size: 512,
            num_hidden_layers: 1,
            vocab_size: 64,
            intermediate_size: 64,
            rms_norm_eps: 1e-6,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            linear_num_key_heads: 1,
            linear_num_value_heads: Some(1),
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            mtp_num_hidden_layers: 1,
            mtp_use_dedicated_embeddings: false,
            full_attention_interval: 1,
            layer_types: vec![LayerType::FullAttention],
            layer_mask: vec![true],
            eos_token_id: 63,
            max_position_embeddings: 128,
            quarot_rotation_seed: None,
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        }
    }

    // MTP Q4/F16 tensor resolution (#630, #636): `quantize_q4` applies its
    // `should_quantize` name-suffix rule to MTP tensor names exactly like the
    // main model, so the 8 `*_proj*` matrices (q/k/v/o_proj, gate/up/down_proj,
    // fc) are written as `.q4` while the 7 norm tensors are written as `.f16`.
    // `quantize_quarot` instead keeps ALL 15 MTP tensors as unrotated `.f16`
    // (ADR-051 Phase 1: MTP counter-rotation needs unquantized weights).
    // `resolve_mtp_projection`/`resolve_mtp_norm` must accept both flavors,
    // reject a stale `.q4` MTP sibling under a QuaRot (`is_quarot=true`)
    // directory, and reject a transposed/mismatched tensor
    // shape. These tests live in this portable sibling module,
    // compiled unconditionally in any test build -- not gated on `metal-gpu`
    // -- so this coverage never skip-passes on a Metal-less CI runner
    // (see the module doc
    // comment above `mtp_tensor_path` for why this code moved out of
    // `mod inner` entirely).

    /// Write a tiny `.f16` (KHF1) tensor file with the given `shape`,
    /// filled with copies of `1.0`, mirroring the format
    /// `load_f16_tensor_file` reads (crates/inference/src/weights/q4_weights.rs).
    pub(crate) fn write_tiny_f16_fixture(dir: &std::path::Path, name: &str, shape: &[usize]) {
        use crate::weights::q4_weights::q4_f32_to_f16;
        let numel: usize = shape.iter().product();
        let path = mtp_tensor_path(dir, name, "f16");
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHF1");
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes()); // ndim
        for &dim in shape {
            buf.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        buf.extend_from_slice(&(numel as u64).to_le_bytes()); // numel
        for _ in 0..numel {
            buf.extend_from_slice(&q4_f32_to_f16(1.0).to_le_bytes());
        }
        std::fs::write(&path, buf).expect("write .f16 fixture");
    }

    /// Write a tiny `.q4` tensor file with the given `shape`, filled with
    /// copies of `0.1`.
    pub(crate) fn write_tiny_q4_fixture(dir: &std::path::Path, name: &str, shape: &[usize]) {
        use crate::weights::q4_weights::{quantize_f32_to_q4, save_q4_file};
        let numel: usize = shape.iter().product();
        let data = vec![0.1f32; numel];
        let tensor = quantize_f32_to_q4(&data, shape).expect("quantize tiny tensor");
        let path = mtp_tensor_path(dir, name, "q4");
        save_q4_file(&path, &tensor).expect("save .q4 fixture");
    }

    /// Write a tiny `.q4` 2-D tensor `[rows, cols]` where every element in
    /// row `r` is `row_value(r)` (constant within the row, distinct across
    /// rows). Used to build a MoE router (`mlp.gate.weight`, `[E, H]`) whose
    /// per-expert logit is fully controlled by the caller: since `cols` is a
    /// multiple of the Q4 block size (32) in every fixture this is used
    /// with, each row occupies whole blocks only (never split across two
    /// different rows' values), so `quantize_f32_to_q4` reconstructs each
    /// row's constant exactly (mod one f16 rounding of the input value —
    /// see `q4_const_roundtrip` for the expected-readback helper).
    // Only consumed by the Device-gated MoE integration tests in `mod
    // inner::tests`, which don't exist in a default-feature (no metal-gpu)
    // build -- looks dead in that configuration (same as
    // `write_full_mtp_fixture` above).
    #[allow(dead_code)]
    pub(crate) fn write_tiny_q4_fixture_per_row(
        dir: &std::path::Path,
        name: &str,
        rows: usize,
        cols: usize,
        row_value: impl Fn(usize) -> f32,
    ) {
        use crate::weights::q4_weights::{quantize_f32_to_q4, save_q4_file};
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            data.extend(std::iter::repeat_n(row_value(r), cols));
        }
        let tensor =
            quantize_f32_to_q4(&data, &[rows, cols]).expect("quantize per-row tiny tensor");
        let path = mtp_tensor_path(dir, name, "q4");
        save_q4_file(&path, &tensor).expect("save .q4 fixture");
    }

    /// Write a tiny `.q4` 3-D tensor `[experts, mid, cols]` where every
    /// element at `(e, m, _)` is `value(e, m)` (constant within an `(e, m)`
    /// row, distinct per expert AND per position along `mid` — e.g. to
    /// distinguish the fused gate/up halves of `experts.gate_up_proj`,
    /// `[E, 2I, H]`, or to distinguish experts in `experts.down_proj`,
    /// `[E, H, I]`). Same block-alignment argument as
    /// `write_tiny_q4_fixture_per_row`: `cols` is always a multiple of 32 in
    /// the fixtures this is used with, so every block sits inside exactly
    /// one `(e, m)` row and quantization is exact.
    // Only consumed by the Device-gated MoE integration tests in `mod
    // inner::tests`; dead in a default-feature (no metal-gpu) build.
    #[allow(dead_code)]
    pub(crate) fn write_tiny_q4_fixture_per_expert_row(
        dir: &std::path::Path,
        name: &str,
        experts: usize,
        mid: usize,
        cols: usize,
        value: impl Fn(usize, usize) -> f32,
    ) {
        use crate::weights::q4_weights::{quantize_f32_to_q4, save_q4_file};
        let mut data = Vec::with_capacity(experts * mid * cols);
        for e in 0..experts {
            for m in 0..mid {
                data.extend(std::iter::repeat_n(value(e, m), cols));
            }
        }
        let tensor = quantize_f32_to_q4(&data, &[experts, mid, cols])
            .expect("quantize per-expert-row tiny tensor");
        let path = mtp_tensor_path(dir, name, "q4");
        save_q4_file(&path, &tensor).expect("save .q4 fixture");
    }

    /// The exact f32 value a constant-valued Q4 block round-trips to: Q4
    /// asymmetric quantization stores `scale`/`bias` as f16 and, for a
    /// constant block (`min == max == value`), always packs nibble `0`
    /// (`(value - min) * inv_scale == 0`), so dequant reduces to `0 * scale
    /// + bias == bias == f16(value)` regardless of the block's `scale`
    /// field — exact modulo one f16 rounding of `value` itself.
    ///
    /// Lets tests assert readback against the value the pipeline will
    /// actually produce, not the pre-quantization f32 the fixture requested.
    // Only consumed by the Device-gated MoE integration tests in `mod
    // inner::tests`; dead in a default-feature (no metal-gpu) build.
    #[allow(dead_code)]
    pub(crate) fn q4_const_roundtrip(value: f32) -> f32 {
        use crate::weights::q4_weights::{q4_f16_to_f32, q4_f32_to_f16};
        q4_f16_to_f32(q4_f32_to_f16(value))
    }

    /// The 8 MTP projection tensor names paired with their expected
    /// on-disk shape, derived from the real qwen3.5-0.8b-q4 checkpoint
    /// (`~/.lattice/models/qwen3.5-0.8b-q4/mtp_*.q4` headers, verified by
    /// hand during #636 remediation): on-disk shape is always
    /// `[d_out, d_in]`, matching `expected_lora_shape`'s `(d_in, d_out)`
    /// convention reversed.
    pub(crate) fn mtp_proj_names_and_shapes(cfg: &Qwen35Config) -> [(&'static str, Vec<usize>); 8] {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.full_q_dim();
        let kv_dim = cfg.full_kv_dim();
        let inter = cfg.intermediate_size;
        [
            (
                "mtp.layers.0.self_attn.q_proj.weight",
                vec![2 * q_dim, hidden],
            ),
            ("mtp.layers.0.self_attn.k_proj.weight", vec![kv_dim, hidden]),
            ("mtp.layers.0.self_attn.v_proj.weight", vec![kv_dim, hidden]),
            ("mtp.layers.0.self_attn.o_proj.weight", vec![hidden, q_dim]),
            ("mtp.layers.0.mlp.gate_proj.weight", vec![inter, hidden]),
            ("mtp.layers.0.mlp.up_proj.weight", vec![inter, hidden]),
            ("mtp.layers.0.mlp.down_proj.weight", vec![hidden, inter]),
            ("mtp.fc.weight", vec![hidden, 2 * hidden]),
        ]
    }

    /// The 7 MTP norm tensor names paired with their expected shape —
    /// never quantized (`should_quantize` excludes anything containing
    /// `norm.weight`). `q_norm`/`k_norm` are `[head_dim]`; the rest are
    /// `[hidden]`.
    // Only reachable via `write_full_mtp_fixture` (see its own
    // `#[allow(dead_code)]` note above).
    fn mtp_norm_names_and_shapes(cfg: &Qwen35Config) -> [(&'static str, Vec<usize>); 7] {
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        [
            ("mtp.layers.0.input_layernorm.weight", vec![hidden]),
            ("mtp.layers.0.post_attention_layernorm.weight", vec![hidden]),
            ("mtp.layers.0.self_attn.q_norm.weight", vec![head_dim]),
            ("mtp.layers.0.self_attn.k_norm.weight", vec![head_dim]),
            ("mtp.norm.weight", vec![hidden]),
            ("mtp.pre_fc_norm_embedding.weight", vec![hidden]),
            ("mtp.pre_fc_norm_hidden.weight", vec![hidden]),
        ]
    }

    /// Write a complete, well-formed MTP tensor set to `dir`: the 8
    /// projections in `proj_flavor` (`.q4` or `.f16`) and the 7 norms as
    /// `.f16`.
    // Only consumed by the Device-gated integration tests in `mod
    // inner::tests`, which don't exist in a default-feature (no
    // metal-gpu) build -- looks dead in that configuration.
    #[allow(dead_code)]
    pub(crate) fn write_full_mtp_fixture(
        dir: &std::path::Path,
        cfg: &Qwen35Config,
        proj_as_q4: bool,
    ) {
        for (name, shape) in mtp_proj_names_and_shapes(cfg) {
            if proj_as_q4 {
                write_tiny_q4_fixture(dir, name, &shape);
            } else {
                write_tiny_f16_fixture(dir, name, &shape);
            }
        }
        for (name, shape) in mtp_norm_names_and_shapes(cfg) {
            write_tiny_f16_fixture(dir, name, &shape);
        }
    }

    // --- CPU-only, Device-free coverage of the flavor/shape logic itself ---
    // (these never skip on a Metal-less CI runner.)

    #[test]
    fn resolve_mtp_projection_prefers_q4_when_allowed() {
        let cfg = tiny_mtp_test_config();
        let tmp = tempfile::tempdir().expect("tempdir create");
        let (name, shape) = &mtp_proj_names_and_shapes(&cfg)[0]; // q_proj
        write_tiny_q4_fixture(tmp.path(), name, shape);

        let resolved = resolve_mtp_projection(tmp.path(), name, shape, /* allow_q4 */ true)
            .expect("plain Q4 dir must resolve the .q4 sibling");
        assert!(
            resolved.is_q4(),
            "allow_q4=true with a .q4 file present must prefer the .q4 flavor"
        );
        assert_eq!(resolved.shape(), shape.as_slice());
    }

    #[test]
    fn resolve_mtp_projection_falls_back_to_f16_when_q4_absent() {
        let cfg = tiny_mtp_test_config();
        let tmp = tempfile::tempdir().expect("tempdir create");
        let (name, shape) = &mtp_proj_names_and_shapes(&cfg)[0];
        write_tiny_f16_fixture(tmp.path(), name, shape);

        let resolved = resolve_mtp_projection(tmp.path(), name, shape, /* allow_q4 */ true)
            .expect("QuaRot-shaped all-.f16 dir must resolve the .f16 sibling");
        assert!(!resolved.is_q4(), "no .q4 file exists; must resolve .f16");
    }

    #[test]
    fn resolve_mtp_projection_rejects_stale_q4_sibling_under_quarot() {
        // A QuaRot (rotated-space) checkpoint directory
        // must never load an MTP `.q4` sibling even if one exists —
        // that shape is reachable when `quantize_q4` is re-run into an
        // existing QuaRot output dir (the converter reuses a non-empty
        // output dir rather than rejecting it). Silently preferring
        // `.q4` there would load rotated-space weights while
        // counter-rotation (keyed off the same dir's `quarot_seed`)
        // stays enabled, corrupting every MTP draft with no load
        // failure to reveal it.
        let cfg = tiny_mtp_test_config();
        let tmp = tempfile::tempdir().expect("tempdir create");
        let (name, shape) = &mtp_proj_names_and_shapes(&cfg)[0];
        // Mixed/stale layout: BOTH a .q4 (leftover from a re-run
        // quantize_q4) and the expected QuaRot .f16 exist side by side.
        write_tiny_q4_fixture(tmp.path(), name, shape);
        write_tiny_f16_fixture(tmp.path(), name, shape);

        let err = resolve_mtp_projection(tmp.path(), name, shape, /* allow_q4 */ false).expect_err(
            "allow_q4=false (QuaRot flavor) with a .q4 sibling present must fail closed",
        );
        let MtpLoadErr::Incompatible(message) = err else {
            panic!("expected Incompatible, got {err:?}");
        };
        assert!(
            message.contains("incompatible") && message.contains(".q4"),
            "error must name the incompatible .q4 artifact; got: {message}"
        );
    }

    #[test]
    fn resolve_mtp_projection_rejects_transposed_q4_shape() {
        // A same-numel transposed file must not
        // silently load — it would generate tokens but produce garbage
        // MTP drafts (0% accept class).
        let cfg = tiny_mtp_test_config();
        let tmp = tempfile::tempdir().expect("tempdir create");
        let (name, shape) = &mtp_proj_names_and_shapes(&cfg)[0]; // q_proj: [2*q_dim, hidden]
        let transposed: Vec<usize> = shape.iter().rev().copied().collect();
        assert_ne!(
            shape, &transposed,
            "fixture cfg must have distinguishable q_proj dims for this test to be meaningful"
        );
        write_tiny_q4_fixture(tmp.path(), name, &transposed);

        let err = resolve_mtp_projection(tmp.path(), name, shape, /* allow_q4 */ true)
            .expect_err("a transposed same-numel .q4 tensor must be rejected, not accepted");
        let MtpLoadErr::Incompatible(message) = err else {
            panic!("expected Incompatible, got {err:?}");
        };
        assert!(
            message.contains("shape"),
            "error must name the shape mismatch; got: {message}"
        );
    }

    #[test]
    fn resolve_mtp_projection_rejects_transposed_f16_shape() {
        let cfg = tiny_mtp_test_config();
        let tmp = tempfile::tempdir().expect("tempdir create");
        let (name, shape) = &mtp_proj_names_and_shapes(&cfg)[6]; // down_proj: [hidden, inter]
        let transposed: Vec<usize> = shape.iter().rev().copied().collect();
        assert_ne!(
            shape, &transposed,
            "fixture cfg must have distinguishable down_proj dims for this test to be meaningful"
        );
        write_tiny_f16_fixture(tmp.path(), name, &transposed);

        let err = resolve_mtp_projection(tmp.path(), name, shape, /* allow_q4 */ true)
            .expect_err("a transposed same-numel .f16 tensor must be rejected, not accepted");
        assert!(matches!(err, MtpLoadErr::Incompatible(_)));
    }

    #[test]
    fn resolve_mtp_projection_reports_missing_when_neither_flavor_present() {
        let cfg = tiny_mtp_test_config();
        let tmp = tempfile::tempdir().expect("tempdir create");
        let (name, shape) = &mtp_proj_names_and_shapes(&cfg)[0];

        let err = resolve_mtp_projection(tmp.path(), name, shape, true)
            .expect_err("neither flavor present must be Missing, not Ok");
        assert!(matches!(err, MtpLoadErr::Missing(_)));
    }

    #[test]
    fn resolve_mtp_norm_rejects_transposed_shape() {
        let cfg = tiny_mtp_test_config();
        let tmp = tempfile::tempdir().expect("tempdir create");
        // input_layernorm is [hidden]; use a 2-D reshape as the "wrong
        // shape" stand-in since a 1-D vector has no transpose.
        let name = "mtp.layers.0.input_layernorm.weight";
        let hidden = cfg.hidden_size;
        write_tiny_f16_fixture(tmp.path(), name, &[hidden, 1]);

        let err = resolve_mtp_norm(tmp.path(), name, &[hidden])
            .expect_err("a reshaped norm tensor must be rejected, not accepted");
        assert!(matches!(err, MtpLoadErr::Incompatible(_)));
    }
}
