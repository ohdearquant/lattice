//! MTP Q4 weight loading on a Metal device.

use super::*;
// -------------------------------------------------------------------
// load_mtp_q4_weights: Device-gated integration smoke (#630, #636)
// -------------------------------------------------------------------
//
// Pure, `Device`-free coverage of the flavor/shape-resolution logic
// itself (`resolve_mtp_projection`/`resolve_mtp_norm`) lives in the
// private `mtp_weights::mtp_resolve_tests` module, compiled
// unconditionally in any test build (not gated on the `metal-gpu`
// feature), so it
// never skip-passes on a Metal-less CI runner (#636 -- these
// tests used to live right here, inside this `metal-gpu`-gated
// `mod inner`, and `cargo test --lib resolve_mtp` with no features
// reported "running 0 tests"). The tests below instead exercise
// `load_mtp_q4_weights` end-to-end against a real Metal `Device`.

// --- Device-gated integration smoke on top of the CPU-only coverage above ---

#[test]
fn load_mtp_q4_weights_accepts_q4_projections_with_f16_norms() {
    // Straight (non-QuaRot) Q4 checkpoint shape: 8 projections as
    // `.q4`, 7 norms as `.f16`. Before the #630 fix,
    // `load_mtp_q4_weights` only tried `.f16` for the projections,
    // so it always reported the first projection as missing on
    // this exact directory layout (the real shape `quantize_q4`
    // ships) — reverting the fix must fail this assertion.
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(device) = Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let (mut cfg, _weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;

    let tmp = tempfile::tempdir().expect("tempdir create");
    write_full_mtp_fixture(tmp.path(), &cfg, /* proj_as_q4 */ true);

    let result = MetalQwen35State::load_mtp_q4_weights(
        tmp.path(),
        &cfg,
        &device,
        /* is_quarot */ false,
    )
    .expect("well-formed plain-Q4 MTP directory must not hard-error");

    assert!(
        result.weights.is_some(),
        "load_mtp_q4_weights must load MTP weights when the 8 projections \
                 are .q4 and the 7 norms are .f16 (the real quantize_q4 output \
                 shape); first_missing_file={:?}",
        result.first_missing_file
    );
}

#[test]
fn load_mtp_q4_weights_still_accepts_all_f16_quarot_layout() {
    // QuaRot checkpoints (`quantize_quarot`) keep every MTP tensor,
    // including the 8 projections, as unrotated `.f16` (ADR-051
    // Phase 1). The Q4-preferring loader must fall back to `.f16`
    // when no `.q4` sibling exists, so this checkpoint flavor keeps
    // working exactly as before.
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(device) = Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let (mut cfg, _weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;

    let tmp = tempfile::tempdir().expect("tempdir create");
    write_full_mtp_fixture(tmp.path(), &cfg, /* proj_as_q4 */ false);

    let result =
        MetalQwen35State::load_mtp_q4_weights(tmp.path(), &cfg, &device, /* is_quarot */ true)
            .expect("well-formed all-.f16 QuaRot MTP directory must not hard-error");

    assert!(
        result.weights.is_some(),
        "load_mtp_q4_weights must still accept the all-.f16 QuaRot \
                 layout; first_missing_file={:?}",
        result.first_missing_file
    );
}

#[test]
fn load_mtp_q4_weights_taps_the_post_final_norm_hidden_on_an_unrotated_checkpoint() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(device) = Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let (mut cfg, _weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;

    let tmp = tempfile::tempdir().expect("tempdir create");
    write_full_mtp_fixture(tmp.path(), &cfg, /* proj_as_q4 */ true);

    let result = MetalQwen35State::load_mtp_q4_weights(
        tmp.path(),
        &cfg,
        &device,
        /* is_quarot */ false,
    )
    .expect("well-formed plain-Q4 MTP directory must not hard-error");

    let weights = result.weights.expect("MTP weights load");
    assert_eq!(weights.hidden_tap, MtpHiddenTap::PostFinalNorm);
}

#[test]
fn load_mtp_q4_weights_keeps_the_pre_final_norm_hidden_on_a_rotated_checkpoint() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(device) = Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let (mut cfg, _weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;

    let tmp = tempfile::tempdir().expect("tempdir create");
    write_full_mtp_fixture(tmp.path(), &cfg, /* proj_as_q4 */ false);

    let result =
        MetalQwen35State::load_mtp_q4_weights(tmp.path(), &cfg, &device, /* is_quarot */ true)
            .expect("well-formed all-.f16 QuaRot MTP directory must not hard-error");

    let weights = result.weights.expect("MTP weights load");
    assert_eq!(weights.hidden_tap, MtpHiddenTap::PreFinalNorm);
}

#[test]
fn load_mtp_q4_weights_reports_missing_file_when_neither_flavor_present() {
    // Fail-closed contract: if a projection has neither a `.q4` nor
    // a `.f16` sibling, the loader must report it as missing (and
    // the caller falls back to the non-MTP decode path with a
    // warning) rather than silently loading garbage.
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(device) = Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let (mut cfg, _weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;

    let tmp = tempfile::tempdir().expect("tempdir create");
    // Deliberately write nothing — the directory is empty.

    let result = MetalQwen35State::load_mtp_q4_weights(tmp.path(), &cfg, &device, false)
        .expect("a genuinely-missing (not incompatible) artifact must not hard-error");

    assert!(
        result.weights.is_none(),
        "load_mtp_q4_weights must not synthesize weights when both \
                 .q4 and .f16 are absent"
    );
    assert!(
        result.first_missing_file.is_some(),
        "load_mtp_q4_weights must report a first_missing_file when \
                 nothing is on disk"
    );
}

#[test]
fn load_mtp_q4_weights_hard_errors_on_stale_q4_under_quarot() {
    // Integration coverage: a mixed/stale QuaRot
    // directory (both .q4 and .f16 for a projection, is_quarot=true)
    // must be a hard `Err`, not a gracefully-disabled `None` — this
    // is the exact failure-open scenario this guard closes.
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(device) = Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let (mut cfg, _weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;

    let tmp = tempfile::tempdir().expect("tempdir create");
    write_full_mtp_fixture(tmp.path(), &cfg, /* proj_as_q4 */ false);
    // Stale leftover from a prior `quantize_q4` run into this dir.
    let (name, shape) = &mtp_proj_names_and_shapes(&cfg)[0];
    write_tiny_q4_fixture(tmp.path(), name, shape);

    let result =
        MetalQwen35State::load_mtp_q4_weights(tmp.path(), &cfg, &device, /* is_quarot */ true);
    let Err(err) = result else {
        panic!(
            "a stale MTP .q4 sibling under a QuaRot directory must hard-error, \
                     not silently disable MTP or silently load the .q4"
        )
    };
    assert!(
        err.contains("incompatible") && err.contains(".q4"),
        "error must name the incompatible .q4 artifact; got: {err}"
    );
}

const ORIGINAL_FINAL_NORM_FILE: &str = "quarot_original_final_norm_weight.f16";
const ORIGINAL_FINAL_NORM_Q4_FILE: &str = "quarot_original_final_norm_weight.q4";

fn original_final_norm_fixture() -> (Device, Qwen35Config, tempfile::TempDir) {
    let device = Device::system_default().expect("preserved final-norm controls require Metal");
    let (mut cfg, _) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let tmp = tempfile::tempdir().unwrap();
    write_full_mtp_fixture(tmp.path(), &cfg, false);
    (device, cfg, tmp)
}

pub(super) fn write_original_final_norm_fixture(
    directory: &std::path::Path,
    shape: &[usize],
    values: &[u16],
) {
    let mut bytes = b"KHF1".to_vec();
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &dim in shape {
        bytes.extend_from_slice(&(dim as u64).to_le_bytes());
    }
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(directory.join(ORIGINAL_FINAL_NORM_FILE), bytes).unwrap();
}

fn assert_original_final_norm_refusal(
    directory: &std::path::Path,
    cfg: &Qwen35Config,
    device: &Device,
    rotated: bool,
    filename: &str,
) {
    let result = MetalQwen35State::load_mtp_q4_weights(directory, cfg, device, rotated);
    let Err(error) = result else {
        panic!("present incompatible {filename} must refuse the MTP load");
    };
    assert!(
        error.contains(filename),
        "refusal must name {filename}: {error}"
    );
}

#[test]
fn original_final_norm_present_selects_post_final_norm() {
    let _gpu_guard = gpu_test_lock();
    let (device, cfg, tmp) = original_final_norm_fixture();
    let values: Vec<u16> = (0..cfg.hidden_size)
        .map(|i| [0x0000, 0x3c00, 0xb800, 0xbc00][i % 4])
        .collect();
    write_original_final_norm_fixture(tmp.path(), &[cfg.hidden_size], &values);
    let result = MetalQwen35State::load_mtp_q4_weights(tmp.path(), &cfg, &device, true).unwrap();
    assert_eq!(
        result.weights.unwrap().hidden_tap,
        MtpHiddenTap::PostFinalNorm
    );
}

#[test]
fn original_final_norm_absent_keeps_legacy_tap() {
    let _gpu_guard = gpu_test_lock();
    let (device, cfg, tmp) = original_final_norm_fixture();
    assert!(!tmp.path().join(ORIGINAL_FINAL_NORM_FILE).exists());
    let result = MetalQwen35State::load_mtp_q4_weights(tmp.path(), &cfg, &device, true).unwrap();
    assert_eq!(
        result.weights.unwrap().hidden_tap,
        MtpHiddenTap::PreFinalNorm
    );
}

#[test]
fn original_final_norm_rejects_same_numel_wrong_shape() {
    let _gpu_guard = gpu_test_lock();
    let (device, cfg, tmp) = original_final_norm_fixture();
    write_original_final_norm_fixture(tmp.path(), &[1, cfg.hidden_size], &vec![0; cfg.hidden_size]);
    assert_original_final_norm_refusal(tmp.path(), &cfg, &device, true, ORIGINAL_FINAL_NORM_FILE);
}

#[test]
fn original_final_norm_rejects_wrong_length() {
    let _gpu_guard = gpu_test_lock();
    let (device, cfg, tmp) = original_final_norm_fixture();
    write_original_final_norm_fixture(
        tmp.path(),
        &[cfg.hidden_size + 1],
        &vec![0; cfg.hidden_size + 1],
    );
    assert_original_final_norm_refusal(tmp.path(), &cfg, &device, true, ORIGINAL_FINAL_NORM_FILE);
}

#[test]
fn original_final_norm_rejects_truncated_payload() {
    let _gpu_guard = gpu_test_lock();
    let (device, cfg, tmp) = original_final_norm_fixture();
    write_original_final_norm_fixture(tmp.path(), &[cfg.hidden_size], &vec![0; cfg.hidden_size]);
    let path = tmp.path().join(ORIGINAL_FINAL_NORM_FILE);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.pop();
    std::fs::write(path, bytes).unwrap();
    assert_original_final_norm_refusal(tmp.path(), &cfg, &device, true, ORIGINAL_FINAL_NORM_FILE);
}

#[test]
fn original_final_norm_rejects_nonfinite_payload() {
    let _gpu_guard = gpu_test_lock();
    let (device, cfg, tmp) = original_final_norm_fixture();
    let mut values = vec![0; cfg.hidden_size];
    *values.last_mut().unwrap() = 0x7c00;
    write_original_final_norm_fixture(tmp.path(), &[cfg.hidden_size], &values);
    assert_original_final_norm_refusal(tmp.path(), &cfg, &device, true, ORIGINAL_FINAL_NORM_FILE);
}

#[test]
fn original_final_norm_rejects_unreadable_entry() {
    let _gpu_guard = gpu_test_lock();
    let (device, cfg, tmp) = original_final_norm_fixture();
    std::fs::create_dir(tmp.path().join(ORIGINAL_FINAL_NORM_FILE)).unwrap();
    assert_original_final_norm_refusal(tmp.path(), &cfg, &device, true, ORIGINAL_FINAL_NORM_FILE);
}

#[cfg(unix)]
#[test]
fn original_final_norm_rejects_dangling_link() {
    let _gpu_guard = gpu_test_lock();
    let (device, cfg, tmp) = original_final_norm_fixture();
    std::os::unix::fs::symlink("missing-scale", tmp.path().join(ORIGINAL_FINAL_NORM_FILE)).unwrap();
    assert_original_final_norm_refusal(tmp.path(), &cfg, &device, true, ORIGINAL_FINAL_NORM_FILE);
}

#[test]
fn original_final_norm_rejects_quantized_sibling() {
    let _gpu_guard = gpu_test_lock();
    let (device, cfg, tmp) = original_final_norm_fixture();
    std::fs::write(
        tmp.path().join(ORIGINAL_FINAL_NORM_Q4_FILE),
        b"stale quantized sibling",
    )
    .unwrap();
    assert_original_final_norm_refusal(
        tmp.path(),
        &cfg,
        &device,
        true,
        ORIGINAL_FINAL_NORM_Q4_FILE,
    );
}

#[test]
fn original_final_norm_rejects_unrotated_claim() {
    let _gpu_guard = gpu_test_lock();
    let (device, cfg, tmp) = original_final_norm_fixture();
    write_original_final_norm_fixture(tmp.path(), &[cfg.hidden_size], &vec![0; cfg.hidden_size]);
    assert_original_final_norm_refusal(tmp.path(), &cfg, &device, false, ORIGINAL_FINAL_NORM_FILE);
}

#[test]
fn original_final_norm_disabled_mtp_ignores_optional_claims() {
    let _gpu_guard = gpu_test_lock();
    let (device, mut cfg, tmp) = original_final_norm_fixture();
    cfg.mtp_num_hidden_layers = 0;
    std::fs::create_dir(tmp.path().join(ORIGINAL_FINAL_NORM_FILE)).unwrap();
    std::fs::write(
        tmp.path().join(ORIGINAL_FINAL_NORM_Q4_FILE),
        b"not a tensor",
    )
    .unwrap();
    for rotated in [false, true] {
        let result =
            MetalQwen35State::load_mtp_q4_weights(tmp.path(), &cfg, &device, rotated).unwrap();
        assert!(result.weights.is_none());
        assert!(result.first_missing_file.is_none());
    }
}

#[test]
fn original_final_norm_missing_mtp_ignores_rotated_q4_claim() {
    let _gpu_guard = gpu_test_lock();
    let (device, cfg, tmp) = original_final_norm_fixture();
    std::fs::remove_file(tmp.path().join("mtp_fc_weight.f16")).unwrap();
    std::fs::create_dir(tmp.path().join(ORIGINAL_FINAL_NORM_FILE)).unwrap();
    std::fs::write(
        tmp.path().join(ORIGINAL_FINAL_NORM_Q4_FILE),
        b"not a tensor",
    )
    .unwrap();
    let result = MetalQwen35State::load_mtp_q4_weights(tmp.path(), &cfg, &device, true).unwrap();
    assert!(result.weights.is_none());
    assert_eq!(
        result.first_missing_file.as_deref(),
        Some(tmp.path().join("mtp_fc_weight.f16").as_path())
    );
}

#[test]
fn original_final_norm_missing_mtp_ignores_unrotated_f16_claim() {
    let _gpu_guard = gpu_test_lock();
    let (device, cfg, tmp) = original_final_norm_fixture();
    std::fs::remove_file(tmp.path().join("mtp_fc_weight.f16")).unwrap();
    std::fs::create_dir(tmp.path().join(ORIGINAL_FINAL_NORM_FILE)).unwrap();
    let result = MetalQwen35State::load_mtp_q4_weights(tmp.path(), &cfg, &device, false).unwrap();
    assert!(result.weights.is_none());
    assert_eq!(
        result.first_missing_file.as_deref(),
        Some(tmp.path().join("mtp_fc_weight.f16").as_path())
    );
}

#[test]
#[ignore = "requires the real converter output at LATTICE_QUAROT_MTP_FIXTURE_ROOT and Metal"]
fn original_final_norm_loads_converted_fixture() {
    let _gpu_guard = gpu_test_lock();
    let device = Device::system_default().expect("converted final-norm roundtrip requires Metal");
    let root = std::path::PathBuf::from(
        std::env::var_os("LATTICE_QUAROT_MTP_FIXTURE_ROOT")
            .expect("run the explicit converter fixture producer first"),
    );
    assert!(root.is_absolute());
    let input = root.join("input");
    let output = root.join("output");
    let receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("fixture.json")).expect("converter fixture receipt is required"),
    )
    .unwrap();
    assert_eq!(receipt["schema"], "quarot-mtp-final-norm-fixture-v1");
    let cfg = Qwen35Config::from_config_json_str(
        &std::fs::read_to_string(output.join("config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(cfg.hidden_size, 8);
    assert_eq!(cfg.mtp_num_hidden_layers, 1);
    let reader = lattice_quant::quarot::io::QuarotTensorReader::open(&input).unwrap();
    let (source, shape) = reader
        .read_tensor_f64("model.language_model.norm.weight")
        .unwrap();
    assert_eq!(shape, [8]);
    assert_eq!(source.len(), 8);
    for (i, &value) in source.iter().enumerate() {
        assert_eq!(value, f64::from([0.0f32, 1.0001, -0.5001, -1.0001][i % 4]));
    }
    let bytes = std::fs::read(output.join(ORIGINAL_FINAL_NORM_FILE)).unwrap();
    let mut expected = b"KHF1".to_vec();
    expected.extend_from_slice(&1u32.to_le_bytes());
    expected.extend_from_slice(&1u32.to_le_bytes());
    expected.extend_from_slice(&8u64.to_le_bytes());
    expected.extend_from_slice(&8u64.to_le_bytes());
    for i in 0..8 {
        expected.extend_from_slice(&[0x0000u16, 0x3c00, 0xb800, 0xbc00][i % 4].to_le_bytes());
    }
    assert_eq!(
        bytes, expected,
        "converter output must equal independently rounded source values"
    );
    let loaded = MetalQwen35State::load_mtp_q4_weights(&output, &cfg, &device, true)
        .expect("the actual converter output must be accepted by the actual MTP loader");
    let weights = loaded
        .weights
        .expect("complete converted MTP set must load");
    assert_eq!(weights.hidden_tap, MtpHiddenTap::PostFinalNorm);
}

fn load_original_final_norm_full_constructor_fixture() -> MetalQwen35State {
    let _device = Device::system_default().expect("full-constructor compatibility requires Metal");
    let directory = std::path::PathBuf::from(
        std::env::var_os("LATTICE_QUAROT_MTP_FULL_ARTIFACT")
            .expect("LATTICE_QUAROT_MTP_FULL_ARTIFACT must name the bound converted artifact"),
    );
    assert!(directory.is_absolute());
    let cfg = Qwen35Config::from_config_json_str(
        &std::fs::read_to_string(directory.join("config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.num_attention_heads, 1);
    assert_eq!(cfg.num_key_value_heads, 1);
    assert_eq!(cfg.num_hidden_layers, 1);
    assert_eq!(cfg.mtp_num_hidden_layers, 1);
    MetalQwen35State::from_q4_dir(
        &directory,
        &directory.join("unused-tokenizer.json"),
        &cfg,
        16,
    )
    .expect("the actual converted artifact must load through the full Q4 constructor")
}

#[test]
#[ignore = "requires the bound full-constructor converted artifact and Metal"]
fn original_final_norm_full_constructor_uses_stored_scale() {
    let _gpu_guard = gpu_test_lock();
    let state = load_original_final_norm_full_constructor_fixture();
    assert!(state.engine.quarot_rotation.is_some());
    assert!(state.engine.mtp_weights.is_some());
    assert_eq!(state.mtp_hidden_tap(), Some(MtpHiddenTap::PostFinalNorm));
}

#[test]
#[ignore = "requires a rotated artifact copy missing mtp_fc_weight.f16, with optional f16 and q4 claims"]
fn original_final_norm_full_constructor_no_mtp_rotated() {
    let _gpu_guard = gpu_test_lock();
    let state = load_original_final_norm_full_constructor_fixture();
    assert!(state.engine.quarot_rotation.is_some());
    assert!(state.engine.mtp_weights.is_none());
    assert_eq!(state.mtp_hidden_tap(), None);
}

#[test]
#[ignore = "requires an unrotated artifact copy missing mtp_fc_weight.f16, with an optional f16 claim"]
fn original_final_norm_full_constructor_no_mtp_unrotated() {
    let _gpu_guard = gpu_test_lock();
    let state = load_original_final_norm_full_constructor_fixture();
    assert!(state.engine.quarot_rotation.is_none());
    assert!(state.engine.mtp_weights.is_none());
    assert_eq!(state.mtp_hidden_tap(), None);
}

#[test]
#[ignore = "requires real converter output and the candidate's preserved-scale field"]
fn original_final_norm_loaded_scale_matches_converted_source() {
    let _gpu_guard = gpu_test_lock();
    let device = Device::system_default().expect("loaded preserved-scale equality requires Metal");
    let root = std::path::PathBuf::from(
        std::env::var_os("LATTICE_QUAROT_MTP_FIXTURE_ROOT")
            .expect("run the explicit converter fixture producer first"),
    );
    assert!(root.is_absolute());
    let output = root.join("output");
    let cfg = Qwen35Config::from_config_json_str(
        &std::fs::read_to_string(output.join("config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(cfg.hidden_size, 8);
    let reader = lattice_quant::quarot::io::QuarotTensorReader::open(&root.join("input")).unwrap();
    let (source, shape) = reader
        .read_tensor_f64("model.language_model.norm.weight")
        .unwrap();
    assert_eq!(shape, [8]);
    assert_eq!(source.len(), 8);
    for (i, &value) in source.iter().enumerate() {
        assert_eq!(value, f64::from([0.0f32, 1.0001, -0.5001, -1.0001][i % 4]));
    }
    let loaded = MetalQwen35State::load_mtp_q4_weights(&output, &cfg, &device, true).unwrap();
    let weights = loaded
        .weights
        .expect("complete converted MTP set must load");
    assert_eq!(
        weights.original_final_norm_weight.as_deref(),
        Some(&[0.0f32, 1.0, -0.5, -1.0, 0.0, 1.0, -0.5, -1.0][..]),
        "actual loaded field must equal the source values after f16 rounding"
    );
}
