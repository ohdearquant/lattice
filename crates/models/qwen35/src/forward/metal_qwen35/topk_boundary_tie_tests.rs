use super::*;

fn test_token_set(tokens: &[u32]) -> std::collections::HashSet<u32> {
    tokens.iter().copied().collect()
}

#[test]
fn topk_boundary_tie_tight_margin_mismatch_is_accepted() {
    let logits = [3.0, 2.0, 1.0, 0.9995, 0.0];
    let baseline_set = test_token_set(&[0, 1, 2]);
    let candidate_set = test_token_set(&[0, 1, 3]);

    match topk_set_agreement_or_boundary_tie(
        &logits,
        &baseline_set,
        &candidate_set,
        3,
        LM_HEAD_TOPK_TIE_EPSILON,
    ) {
        Ok(TopkSetAgreement::AcceptedBoundaryTie {
            margin,
            differing_tokens,
            ..
        }) => {
            assert!(margin < LM_HEAD_TOPK_TIE_EPSILON);
            assert_eq!(
                differing_tokens,
                vec![
                    TopkSetDiffToken {
                        token_id: 2,
                        baseline_logit: 1.0,
                    },
                    TopkSetDiffToken {
                        token_id: 3,
                        baseline_logit: 0.9995,
                    },
                ]
            );
        }
        other => panic!("tight boundary mismatch must be accepted, got {other:?}"),
    }
}

#[test]
fn topk_boundary_tie_wide_margin_mismatch_is_rejected() {
    let logits = [3.0, 2.0, 1.0, 0.98, 0.0];
    let baseline_set = test_token_set(&[0, 1, 2]);
    let candidate_set = test_token_set(&[0, 1, 3]);

    let result = topk_set_agreement_or_boundary_tie(
        &logits,
        &baseline_set,
        &candidate_set,
        3,
        LM_HEAD_TOPK_TIE_EPSILON,
    );
    match result {
        Err(err) => {
            assert_eq!(err.reason, TopkSetMismatchReason::BoundaryMarginTooWide);
            assert!(err.margin > LM_HEAD_TOPK_TIE_EPSILON);
            assert_eq!(err.k, 3);
            assert_eq!(err.epsilon, LM_HEAD_TOPK_TIE_EPSILON);
        }
        Ok(ok) => panic!("wide boundary margin must be rejected, got {ok:?}"),
    }
}

#[test]
fn topk_boundary_tie_far_differing_token_is_rejected() {
    let logits = [10.0, 2.0, 1.0, 0.9995, 0.0];
    let baseline_set = test_token_set(&[0, 1, 2]);
    let candidate_set = test_token_set(&[1, 2, 3]);

    let result = topk_set_agreement_or_boundary_tie(
        &logits,
        &baseline_set,
        &candidate_set,
        3,
        LM_HEAD_TOPK_TIE_EPSILON,
    );
    match result {
        Err(err) => {
            assert_eq!(
                err.reason,
                TopkSetMismatchReason::DifferingTokenOutsideBoundary
            );
            assert!(err.margin < LM_HEAD_TOPK_TIE_EPSILON);
            assert!(
                err.differing_tokens
                    .iter()
                    .any(|diff| diff.token_id == 0 && diff.baseline_logit == 10.0)
            );
        }
        Ok(ok) => panic!("far differing token must be rejected, got {ok:?}"),
    }
}

#[test]
fn topk_boundary_tie_identical_sets_pass_exactly() {
    let logits = [3.0, 2.0, 1.0, 0.0];
    let baseline_set = test_token_set(&[0, 1, 2]);
    let candidate_set = test_token_set(&[0, 1, 2]);

    assert_eq!(
        topk_set_agreement_or_boundary_tie(
            &logits,
            &baseline_set,
            &candidate_set,
            3,
            LM_HEAD_TOPK_TIE_EPSILON,
        ),
        Ok(TopkSetAgreement::ExactSet)
    );
}
