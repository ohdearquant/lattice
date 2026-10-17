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
