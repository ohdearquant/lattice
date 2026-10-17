use super::{
    TensorInventoryEntry, TensorInventoryError, checked_name_length_prefix, checked_rank_prefix,
    digest_tensor_inventory,
};
use crate::service::prepared::limits::{
    ChargeExpression, LimitAxis, PreparationLimitError, test_tensor_inventory_ceilings,
};

const GOLDEN: &str = "ea8c24dd5cc09ac5bc3ec55e2fab1df29bb34334988ee0c1a299f6489b694e39";

fn entry<'a>(
    name: &'a [u8],
    source_dtype: &'a str,
    dimensions: &'a [u64],
) -> TensorInventoryEntry<'a> {
    TensorInventoryEntry::new(name, source_dtype, dimensions, 0)
}

fn ceilings(
    max_tensors: usize,
    max_name_bytes: usize,
    max_rank: usize,
    max_dimension: usize,
    max_elements: u64,
    max_metadata: u64,
) -> crate::service::prepared::limits::PreparationCeilings {
    test_tensor_inventory_ceilings(
        max_tensors,
        max_name_bytes,
        max_rank,
        max_dimension,
        max_elements,
        max_metadata,
    )
}

fn permissive_ceilings() -> crate::service::prepared::limits::PreparationCeilings {
    ceilings(32, 128, 16, 4096, 1_000_000, 1_000_000)
}

fn digest_hex(entries: &mut [TensorInventoryEntry<'_>]) -> String {
    let digest = digest_tensor_inventory(entries, &permissive_ceilings()).unwrap();
    digest
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn golden_entries<'a>() -> [TensorInventoryEntry<'a>; 3] {
    [
        entry("é".as_bytes(), "BF16", &[4, 5, 6]),
        entry(b"b", "F16", &[2, 3]),
        entry(b"a", "F32", &[1]),
    ]
}

#[test]
fn exact_framing_matches_independent_golden() {
    let mut entries = golden_entries();
    assert_eq!(digest_hex(&mut entries), GOLDEN);
    assert_eq!(entries[0].name(), b"a");
    assert_eq!(entries[1].name(), b"b");
    assert_eq!(entries[2].name(), "é".as_bytes());
}

#[test]
fn all_insertion_permutations_share_one_digest() {
    let source = golden_entries();
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mut entries = [source[order[0]], source[order[1]], source[order[2]]];
        assert_eq!(digest_hex(&mut entries), GOLDEN, "order {order:?}");
    }
}

#[test]
fn missing_inventory_fails_closed() {
    let mut entries = [];
    assert_eq!(
        digest_tensor_inventory(&mut entries, &permissive_ceilings()).unwrap_err(),
        TensorInventoryError::MissingInventory
    );
}

#[test]
fn invalid_utf8_name_fails_without_echoing_bytes() {
    let mut entries = [entry(&[0xff], "F32", &[1])];
    assert_eq!(
        digest_tensor_inventory(&mut entries, &permissive_ceilings()).unwrap_err(),
        TensorInventoryError::InvalidUtf8Name { entry: 0 }
    );
}

#[test]
fn unsupported_source_dtype_labels_fail_exactly() {
    for label in ["I64", "F64", "F8_E4M3", "f32", "unknown"] {
        let mut entries = [entry(b"a", label, &[1])];
        assert_eq!(
            digest_tensor_inventory(&mut entries, &permissive_ceilings()).unwrap_err(),
            TensorInventoryError::UnsupportedSourceDtype { entry: 0 },
            "label {label}"
        );
    }
}

#[test]
fn non_adjacent_duplicate_names_fail_after_sorting() {
    let mut entries = [
        entry(b"same", "F32", &[1]),
        entry(b"middle", "F16", &[2, 3]),
        entry(b"same", "BF16", &[4, 5, 6]),
    ];
    assert_eq!(
        digest_tensor_inventory(&mut entries, &permissive_ceilings()).unwrap_err(),
        TensorInventoryError::DuplicateName
    );
}

#[test]
fn every_identity_field_mutation_changes_the_digest() {
    let mut baseline = golden_entries();
    let baseline = digest_hex(&mut baseline);

    let mut removed = [entry(b"a", "F32", &[1]), entry(b"b", "F16", &[2, 3])];
    assert_ne!(digest_hex(&mut removed), baseline, "tensor count removal");

    let mut added = [
        entry(b"a", "F32", &[1]),
        entry(b"b", "F16", &[2, 3]),
        entry(b"c", "BF16", &[4, 5, 6]),
        entry("é".as_bytes(), "BF16", &[4, 5, 6]),
    ];
    assert_ne!(digest_hex(&mut added), baseline, "tensor count addition");

    let mut name = golden_entries();
    name[2] = entry(b"A", "F32", &[1]);
    assert_ne!(digest_hex(&mut name), baseline, "name byte");

    let mut name_length = golden_entries();
    name_length[2] = entry(b"aa", "F32", &[1]);
    assert_ne!(digest_hex(&mut name_length), baseline, "name byte length");

    let mut dtype = golden_entries();
    dtype[2] = entry(b"a", "F16", &[1]);
    assert_ne!(digest_hex(&mut dtype), baseline, "dtype");

    let mut rank = golden_entries();
    rank[1] = entry(b"b", "F16", &[2, 3, 1]);
    assert_ne!(digest_hex(&mut rank), baseline, "rank");

    for (index, dimensions) in [&[9, 3][..], &[2, 9][..]].into_iter().enumerate() {
        let mut dimension = golden_entries();
        dimension[1] = entry(b"b", "F16", dimensions);
        assert_ne!(
            digest_hex(&mut dimension),
            baseline,
            "dimension position {index}"
        );
    }

    let mut dimension_order = golden_entries();
    dimension_order[1] = entry(b"b", "F16", &[3, 2]);
    assert_ne!(
        digest_hex(&mut dimension_order),
        baseline,
        "dimension order"
    );
}

#[test]
fn utf8_is_framed_by_raw_bytes_without_normalization() {
    let mut composed = [entry("é".as_bytes(), "F32", &[1])];
    let mut decomposed = [entry("e\u{301}".as_bytes(), "F32", &[1])];
    assert_ne!(digest_hex(&mut composed), digest_hex(&mut decomposed));

    let mut one_byte = [entry(b"e", "F32", &[1])];
    assert_ne!(digest_hex(&mut composed), digest_hex(&mut one_byte));
}

#[test]
fn zero_rank_and_rank_one_are_distinct() {
    let mut scalar = [entry(b"a", "F32", &[])];
    let mut vector = [entry(b"a", "F32", &[1])];
    assert_ne!(digest_hex(&mut scalar), digest_hex(&mut vector));
}

#[test]
fn accounted_metadata_bytes_are_not_identity_bytes() {
    let mut low = [TensorInventoryEntry::new(b"a", "F32", &[1], 1)];
    let mut high = [TensorInventoryEntry::new(b"a", "F32", &[1], 17)];
    let limits = ceilings(1, 1, 1, 1, 1, 18);
    assert_eq!(
        digest_tensor_inventory(&mut low, &limits).unwrap(),
        digest_tensor_inventory(&mut high, &limits).unwrap()
    );
}

#[test]
fn each_tensor_limit_accepts_exact_and_rejects_plus_one() {
    let cases = [
        (
            vec![entry(b"a", "F32", &[1]), entry(b"b", "F32", &[1])],
            ceilings(1, 8, 4, 8, 64, 64),
            PreparationLimitError::Exceeded {
                axis: LimitAxis::Tensors,
                actual: 2,
                limit: 1,
            },
        ),
        (
            vec![entry("é".as_bytes(), "F32", &[1])],
            ceilings(1, 1, 4, 8, 64, 64),
            PreparationLimitError::Exceeded {
                axis: LimitAxis::TensorNameBytes,
                actual: 2,
                limit: 1,
            },
        ),
        (
            vec![entry(b"a", "F32", &[1, 2])],
            ceilings(1, 8, 1, 8, 64, 64),
            PreparationLimitError::Exceeded {
                axis: LimitAxis::Rank,
                actual: 2,
                limit: 1,
            },
        ),
        (
            vec![entry(b"a", "F32", &[9])],
            ceilings(1, 8, 4, 8, 64, 64),
            PreparationLimitError::Exceeded {
                axis: LimitAxis::Dimension,
                actual: 9,
                limit: 8,
            },
        ),
        (
            vec![entry(b"a", "F32", &[2, 3])],
            ceilings(1, 8, 4, 8, 5, 64),
            PreparationLimitError::Exceeded {
                axis: LimitAxis::AggregateElements,
                actual: 6,
                limit: 5,
            },
        ),
        (
            vec![TensorInventoryEntry::new(b"a", "F32", &[1], 5)],
            ceilings(1, 8, 4, 8, 64, 5),
            PreparationLimitError::Exceeded {
                axis: LimitAxis::TensorMetadataBytes,
                actual: 6,
                limit: 5,
            },
        ),
    ];

    for (mut entries, limits, expected) in cases {
        assert_eq!(
            digest_tensor_inventory(&mut entries, &limits).unwrap_err(),
            TensorInventoryError::Limit(expected)
        );
    }

    let mut exact = [TensorInventoryEntry::new(b"a", "F32", &[2, 3], 4)];
    assert!(digest_tensor_inventory(&mut exact, &ceilings(1, 1, 2, 3, 6, 5)).is_ok());
}

#[test]
fn be32_framing_rejects_synthetic_unrepresentable_lengths() {
    assert_eq!(
        checked_name_length_prefix(u64::from(u32::MAX)).unwrap(),
        u32::MAX.to_be_bytes()
    );
    assert_eq!(
        checked_name_length_prefix(u64::from(u32::MAX) + 1).unwrap_err(),
        TensorInventoryError::NameLengthUnrepresentable {
            length: u64::from(u32::MAX) + 1,
        }
    );
    assert_eq!(
        checked_rank_prefix(u64::from(u32::MAX)).unwrap(),
        u32::MAX.to_be_bytes()
    );
    assert_eq!(
        checked_rank_prefix(u64::from(u32::MAX) + 1).unwrap_err(),
        TensorInventoryError::RankUnrepresentable {
            rank: u64::from(u32::MAX) + 1,
        }
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn per_tensor_and_aggregate_payload_overflow_fail_closed() {
    let limits = ceilings(2, 8, 2, usize::MAX, u64::MAX, 64);
    let mut product = [entry(b"a", "F32", &[u64::MAX, 2])];
    assert_eq!(
        digest_tensor_inventory(&mut product, &limits).unwrap_err(),
        TensorInventoryError::Limit(PreparationLimitError::ArithmeticOverflow {
            expression: ChargeExpression::TensorElements,
        })
    );

    let mut aggregate = [
        entry(b"a", "F16", &[u64::MAX / 2]),
        entry(b"b", "F16", &[1]),
    ];
    assert_eq!(
        digest_tensor_inventory(&mut aggregate, &limits).unwrap_err(),
        TensorInventoryError::Limit(PreparationLimitError::ArithmeticOverflow {
            expression: ChargeExpression::TensorBytes,
        })
    );
}
