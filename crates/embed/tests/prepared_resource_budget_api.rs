#![cfg(feature = "native")]

use lattice_embed::{EmbedError, NativeResourceBudget};
use std::num::{NonZeroU64, NonZeroUsize};

fn nonzero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

fn nonzero_u64(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

#[test]
fn resource_budget_accepts_exact_checked_sum_ceiling() {
    let budget = NativeResourceBudget::try_new(
        nonzero_usize(1),
        nonzero_usize(2),
        nonzero_u64(u64::MAX - 1),
        nonzero_u64(1),
    )
    .unwrap();
    assert_eq!(budget.max_concurrent_preparations().get(), 1);
    assert_eq!(budget.max_concurrent_encodes().get(), 2);
    assert_eq!(budget.max_retained_bytes().get(), u64::MAX - 1);
    assert_eq!(budget.max_transient_work_bytes().get(), 1);
    assert_eq!(budget.total_accounted_bytes(), u64::MAX);
}

#[test]
fn resource_budget_rejects_sum_overflow() {
    let error = NativeResourceBudget::try_new(
        nonzero_usize(1),
        nonzero_usize(1),
        nonzero_u64(u64::MAX),
        nonzero_u64(1),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        EmbedError::ResourceBudgetOverflow {
            retained_bytes: u64::MAX,
            transient_work_bytes: 1,
        }
    ));
}
