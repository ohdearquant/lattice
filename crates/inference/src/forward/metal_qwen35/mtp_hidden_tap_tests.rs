use super::MtpHiddenTap;

#[test]
fn rotated_checkpoints_keep_the_pre_final_norm_hidden() {
    assert_eq!(
        MtpHiddenTap::for_checkpoint(true),
        MtpHiddenTap::PreFinalNorm
    );
    assert_eq!(
        MtpHiddenTap::for_checkpoint(false),
        MtpHiddenTap::PostFinalNorm
    );
}

#[test]
fn labels_are_the_stable_log_names() {
    assert_eq!(MtpHiddenTap::PostFinalNorm.label(), "post-final-norm");
    assert_eq!(MtpHiddenTap::PreFinalNorm.label(), "pre-final-norm");
}
