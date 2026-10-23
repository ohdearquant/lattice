use super::mtp_missing_weights_warning;
use std::path::Path;

#[test]
fn requested_missing_mtp_weights_warns_and_disables() {
    let warning = mtp_missing_weights_warning(
        true,
        false,
        Some(Path::new("mtp_layers_0_self_attn_q_proj_weight.f16")),
    );

    assert_eq!(
        warning.as_deref(),
        Some(
            "LATTICE_MTP=1 requested MTP, but first required weight file is missing: mtp_layers_0_self_attn_q_proj_weight.f16; MTP is DISABLED, decode falls back to the plain path."
        )
    );
}

#[test]
fn loaded_or_not_requested_mtp_weights_do_not_warn() {
    let path = Some(Path::new("mtp_layers_0_self_attn_q_proj_weight.f16"));

    assert_eq!(mtp_missing_weights_warning(true, true, path), None);
    assert_eq!(mtp_missing_weights_warning(false, false, path), None);
}
