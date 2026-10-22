use std::num::NonZeroU64;

use super::*;
use crate::weights::prepared_safetensors::{
    PreparedSafetensorsFramingLimits, PreparedSafetensorsHeaderPlan,
    plan_prepared_safetensors_header,
};

fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

fn plan(header_len: usize, data_len: usize) -> PreparedSafetensorsHeaderPlan {
    let declared = 8_u64
        .checked_add(u64::try_from(header_len).unwrap())
        .and_then(|value| value.checked_add(u64::try_from(data_len).unwrap()))
        .unwrap();
    let limits = PreparedSafetensorsFramingLimits::try_new(
        nz(declared),
        nz(u64::try_from(header_len).unwrap()),
    )
    .unwrap();
    plan_prepared_safetensors_header(
        u64::try_from(header_len).unwrap().to_le_bytes(),
        declared,
        &limits,
    )
    .unwrap()
}

fn wide_limits() -> PreparedSafetensorsInventoryLimits {
    PreparedSafetensorsInventoryLimits::try_new(
        nz(128),
        nz(256),
        nz(32),
        nz(1_000_000),
        nz(10_000_000),
        nz(1_000_000),
        nz(2_000_000),
    )
    .unwrap()
}

fn parse(
    header: &[u8],
    data_len: usize,
) -> Result<PreparedSafetensorsInventoryFacts, PreparedSafetensorsInventoryError> {
    parse_prepared_safetensors_header_inventory(
        header,
        &plan(header.len(), data_len),
        &wide_limits(),
    )
}

fn tensor<'a>(
    facts: &'a PreparedSafetensorsInventoryFacts,
    name: &[u8],
) -> PreparedSafetensorsTensorView<'a> {
    facts
        .tensors()
        .find(|tensor| tensor.name_bytes() == name)
        .unwrap()
}

fn assert_malformed(header: &[u8], data_len: usize, expected: HeaderSyntaxFault) {
    let error = parse(header, data_len).unwrap_err();
    assert!(
        matches!(
            error,
            PreparedSafetensorsInventoryError::MalformedHeader { fault, .. }
                if fault == expected
        ),
        "expected {expected:?}, got {error:?}"
    );
}

const CANONICAL: &str = r#"{"zz":{"shape":[1,2],"data_offsets":[4,8],"dtype":"F16"},"__metadata__":{"format":"pt","rocket":"\ud83d\ude80"},"aa":{"data_offsets":[0,4],"dtype":"F32","shape":[1]},"mm":{"dtype":"BF16","shape":[],"data_offsets":[8,10]}}"#;

#[test]
fn canonical_inventory_is_sorted_deterministic_and_exactly_accounted() {
    let first = parse(CANONICAL.as_bytes(), 10).unwrap();
    let permuted = r#"{"mm":{"shape":[],"data_offsets":[8,10],"dtype":"BF16"},"aa":{"shape":[1],"dtype":"F32","data_offsets":[0,4]},"__metadata__":{"rocket":"\ud83d\ude80","format":"pt"},"zz":{"dtype":"F16","data_offsets":[4,8],"shape":[1,2]}}"#;
    let second = parse(permuted.as_bytes(), 10).unwrap();

    assert_ne!(first.name_bytes, second.name_bytes);
    assert_ne!(first.dimensions, second.dimensions);
    assert_eq!(first, second);
    let renamed = parse(CANONICAL.replacen("\"aa\"", "\"ab\"", 1).as_bytes(), 10).unwrap();
    assert_ne!(first, renamed);
    let reshaped = parse(CANONICAL.replacen("[1,2]", "[2,1]", 1).as_bytes(), 10).unwrap();
    assert_ne!(first, reshaped);
    let names: Vec<_> = first
        .tensors()
        .map(|tensor| tensor.name_bytes().to_vec())
        .collect();
    assert_eq!(names, [b"aa".to_vec(), b"mm".to_vec(), b"zz".to_vec()]);

    let a = tensor(&first, b"aa");
    assert_eq!(a.dtype(), PreparedSafetensorsDtype::F32);
    assert_eq!(a.dimensions(), &[1]);
    assert_eq!(a.data_range(), 0..4);
    assert_eq!(a.elements(), 1);
    assert_eq!(a.source_bytes(), 4);

    let m = tensor(&first, b"mm");
    assert_eq!(m.dtype(), PreparedSafetensorsDtype::Bf16);
    assert!(m.dimensions().is_empty());
    assert_eq!(m.elements(), 1);
    assert_eq!(m.source_bytes(), 2);

    let z = tensor(&first, b"zz");
    assert_eq!(z.dtype(), PreparedSafetensorsDtype::F16);
    assert_eq!(z.dimensions(), &[1, 2]);
    assert_eq!(z.elements(), 2);
    assert_eq!(z.source_bytes(), 4);

    let retained = 3 * std::mem::size_of::<PreparedSafetensorsTensorRecord>() + 6 + 3 * 8;
    let parse_work = CANONICAL.len() + 2 * std::mem::size_of::<DecodedSpan>() + 12;
    assert_eq!(first.tensor_count(), 3);
    assert_eq!(first.aggregate_elements(), 4);
    assert_eq!(first.source_payload_bytes(), 10);
    assert_eq!(first.logical_retained_inventory_metadata_bytes(), retained);
    assert_eq!(first.logical_safetensors_parse_work_bytes(), parse_work);
    assert_eq!(first.decoded_metadata_value_bytes(), 6);
}

#[test]
fn exact_header_envelope_and_closed_outer_grammar_are_pinned() {
    let one = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    assert!(parse(one, 4).is_ok());
    assert!(parse(
            b"{ \n \t\"a\" : { \"dtype\" : \"F32\" , \"shape\" : [ 1 ] , \"data_offsets\" : [ 0 , 4 ] } }",
            4,
        )
        .is_ok());
    assert_eq!(
        parse_prepared_safetensors_header_inventory(
            &one[..one.len() - 1],
            &plan(one.len(), 4),
            &wide_limits(),
        )
        .unwrap_err(),
        PreparedSafetensorsInventoryError::HeaderLengthMismatch {
            expected: one.len(),
            actual: one.len() - 1,
        }
    );
    let mut longer = one.to_vec();
    longer.push(b' ');
    assert_eq!(
        parse_prepared_safetensors_header_inventory(&longer, &plan(one.len(), 4), &wide_limits(),)
            .unwrap_err(),
        PreparedSafetensorsInventoryError::HeaderLengthMismatch {
            expected: one.len(),
            actual: one.len() + 1,
        }
    );

    let invalid_utf8 = b"{\"\xff\":{}}";
    assert!(matches!(
        parse(invalid_utf8, 0),
        Err(PreparedSafetensorsInventoryError::InvalidUtf8 { .. })
    ));
    assert_malformed(
        br#" {"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        4,
        HeaderSyntaxFault::ExpectedObjectStart,
    );
    assert_malformed(
        br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},}"#,
        4,
        HeaderSyntaxFault::TrailingComma,
    );
    // JSON whitespace (RFC 8259) is space, tab, CR, and LF — exactly the
    // set `serde_json` (what the reference `safetensors` crate parses
    // the header with) skips before requiring EOF, and exactly the set
    // `skip_ws` already accepts everywhere else in this header. Each is
    // valid trailing padding; only a genuinely non-whitespace byte after
    // the closing brace is malformed.
    for trailing in [b" ".as_slice(), b"\t", b"\n", b"\r", b" \t\r\n"] {
        let mut padded = one.to_vec();
        padded.extend_from_slice(trailing);
        assert!(parse(&padded, 4).is_ok(), "trailing {trailing:?}");
    }
    assert_malformed(
        b"{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}x",
        4,
        HeaderSyntaxFault::TrailingNonWhitespace,
    );
    for empty in [
        br#"{}"#.as_slice(),
        br#"{"__metadata__":{"format":"pt"}}"#.as_slice(),
    ] {
        assert_eq!(
            parse(empty, 0).unwrap_err(),
            PreparedSafetensorsInventoryError::MissingInventory
        );
    }
}

#[test]
fn every_limit_accepts_exact_and_rejects_plus_one_before_reserve() {
    let facts = parse(CANONICAL.as_bytes(), 10).unwrap();
    let exact = PreparedSafetensorsInventoryLimits::try_new(
        nz(3),
        nz(2),
        nz(2),
        nz(2),
        nz(4),
        nz(u64::try_from(facts.logical_retained_inventory_metadata_bytes()).unwrap()),
        nz(u64::try_from(facts.logical_safetensors_parse_work_bytes()).unwrap()),
    )
    .unwrap();
    assert!(
        parse_prepared_safetensors_header_inventory(
            CANONICAL.as_bytes(),
            &plan(CANONICAL.len(), 10),
            &exact,
        )
        .is_ok()
    );

    let cases = [
        (
            InventoryAxis::Tensors,
            3,
            2,
            PreparedSafetensorsInventoryLimits::try_new(
                nz(2),
                nz(256),
                nz(32),
                nz(1_000_000),
                nz(10_000_000),
                nz(1_000_000),
                nz(2_000_000),
            )
            .unwrap(),
        ),
        (
            InventoryAxis::TensorNameBytes,
            2,
            1,
            PreparedSafetensorsInventoryLimits::try_new(
                nz(128),
                nz(1),
                nz(32),
                nz(1_000_000),
                nz(10_000_000),
                nz(1_000_000),
                nz(2_000_000),
            )
            .unwrap(),
        ),
        (
            InventoryAxis::Rank,
            2,
            1,
            PreparedSafetensorsInventoryLimits::try_new(
                nz(128),
                nz(256),
                nz(1),
                nz(1_000_000),
                nz(10_000_000),
                nz(1_000_000),
                nz(2_000_000),
            )
            .unwrap(),
        ),
        (
            InventoryAxis::Dimension,
            2,
            1,
            PreparedSafetensorsInventoryLimits::try_new(
                nz(128),
                nz(256),
                nz(32),
                nz(1),
                nz(10_000_000),
                nz(1_000_000),
                nz(2_000_000),
            )
            .unwrap(),
        ),
        (
            InventoryAxis::AggregateElements,
            4,
            3,
            PreparedSafetensorsInventoryLimits::try_new(
                nz(128),
                nz(256),
                nz(32),
                nz(1_000_000),
                nz(3),
                nz(1_000_000),
                nz(2_000_000),
            )
            .unwrap(),
        ),
        (
            InventoryAxis::MetadataBytes,
            u64::try_from(facts.logical_retained_inventory_metadata_bytes()).unwrap(),
            u64::try_from(facts.logical_retained_inventory_metadata_bytes() - 1).unwrap(),
            PreparedSafetensorsInventoryLimits::try_new(
                nz(128),
                nz(256),
                nz(32),
                nz(1_000_000),
                nz(10_000_000),
                nz(u64::try_from(facts.logical_retained_inventory_metadata_bytes() - 1).unwrap()),
                nz(2_000_000),
            )
            .unwrap(),
        ),
        (
            InventoryAxis::SafetensorsParseWorkBytes,
            u64::try_from(facts.logical_safetensors_parse_work_bytes()).unwrap(),
            u64::try_from(facts.logical_safetensors_parse_work_bytes() - 1).unwrap(),
            PreparedSafetensorsInventoryLimits::try_new(
                nz(128),
                nz(256),
                nz(32),
                nz(1_000_000),
                nz(10_000_000),
                nz(1_000_000),
                nz(u64::try_from(facts.logical_safetensors_parse_work_bytes() - 1).unwrap()),
            )
            .unwrap(),
        ),
    ];

    for (axis, actual, limit, limits) in cases {
        let mut attempts = Vec::new();
        let error = parse_prepared_safetensors_header_inventory_with_reserve_probe(
            CANONICAL.as_bytes(),
            &plan(CANONICAL.len(), 10),
            &limits,
            &mut |arena, _| {
                attempts.push(arena);
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            PreparedSafetensorsInventoryError::Exceeded {
                axis,
                actual,
                limit,
            }
        );
        assert!(attempts.is_empty(), "{axis:?} failed after reserve began");
    }
}

#[test]
fn limit_policy_pins_platform_and_canonical_u32_framing() {
    assert!(
        PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
            nz(8),
            nz(8),
            nz(8),
            nz(8),
            nz(1),
            nz(8),
            nz(1),
            8,
            8,
        )
        .is_ok()
    );
    let platform_cases = [
        (
            InventoryAxis::Tensors,
            PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
                nz(9),
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                8,
                8,
            ),
        ),
        (
            InventoryAxis::TensorNameBytes,
            PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
                nz(1),
                nz(9),
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                8,
                8,
            ),
        ),
        (
            InventoryAxis::Rank,
            PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
                nz(1),
                nz(1),
                nz(9),
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                8,
                8,
            ),
        ),
        (
            InventoryAxis::Dimension,
            PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
                nz(1),
                nz(1),
                nz(1),
                nz(9),
                nz(1),
                nz(1),
                nz(1),
                8,
                8,
            ),
        ),
        (
            InventoryAxis::MetadataBytes,
            PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                nz(9),
                nz(1),
                8,
                8,
            ),
        ),
    ];
    for (axis, result) in platform_cases {
        assert_eq!(
            result.unwrap_err(),
            PreparedSafetensorsInventoryError::PlatformUnrepresentable { axis, value: 9 }
        );
    }

    for (axis, name, rank) in [
        (
            InventoryAxis::TensorNameBytes,
            CANONICAL_INVENTORY_U32_FIELD_MAX + 1,
            1,
        ),
        (
            InventoryAxis::Rank,
            1,
            CANONICAL_INVENTORY_U32_FIELD_MAX + 1,
        ),
    ] {
        let result = PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
            nz(1),
            nz(name),
            nz(rank),
            nz(1),
            nz(1),
            nz(1),
            nz(1),
            u64::MAX,
            u64::MAX,
        );
        assert_eq!(
            result.unwrap_err(),
            PreparedSafetensorsInventoryError::Exceeded {
                axis,
                actual: CANONICAL_INVENTORY_U32_FIELD_MAX + 1,
                limit: CANONICAL_INVENTORY_U32_FIELD_MAX,
            }
        );
    }

    let isize_max = u64::try_from(isize::MAX).unwrap();
    assert!(
        PreparedSafetensorsInventoryLimits::try_new(
            nz(1),
            nz(1),
            nz(1),
            nz(1),
            nz(1),
            nz(isize_max),
            nz(1),
        )
        .is_ok()
    );
    assert_eq!(
        PreparedSafetensorsInventoryLimits::try_new(
            nz(1),
            nz(1),
            nz(1),
            nz(1),
            nz(1),
            nz(isize_max + 1),
            nz(1),
        )
        .unwrap_err(),
        PreparedSafetensorsInventoryError::PlatformUnrepresentable {
            axis: InventoryAxis::MetadataBytes,
            value: isize_max + 1,
        }
    );
}

#[test]
fn semantic_equality_compares_each_independent_materialized_fact() {
    let f16 = parse(
        br#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}} "#,
        2,
    )
    .unwrap();
    let bf16 = parse(
        br#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
        2,
    )
    .unwrap();
    assert_eq!(
        f16.logical_safetensors_parse_work_bytes(),
        bf16.logical_safetensors_parse_work_bytes()
    );
    assert_ne!(f16, bf16);

    let original_ranges = parse(
            br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
            8,
        )
        .unwrap();
    let swapped_ranges = parse(
            br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            8,
        )
        .unwrap();
    assert_ne!(original_ranges, swapped_ranges);

    let compact = parse(
        br#"{"a":{"dtype":"F16","shape":[0],"data_offsets":[0,0]}}"#,
        0,
    )
    .unwrap();
    let padded = parse(
        br#"{"a":{"dtype":"F16","shape":[0],"data_offsets":[0,0]}} "#,
        0,
    )
    .unwrap();
    assert_ne!(compact, padded);

    let one_decoded_value_byte = parse(
        br#"{"__metadata__":{"k":"\n"},"a":{"dtype":"F16","shape":[0],"data_offsets":[0,0]}}"#,
        0,
    )
    .unwrap();
    let two_decoded_value_bytes = parse(
        br#"{"__metadata__":{"k":"ab"},"a":{"dtype":"F16","shape":[0],"data_offsets":[0,0]}}"#,
        0,
    )
    .unwrap();
    assert_eq!(
        one_decoded_value_byte.logical_safetensors_parse_work_bytes(),
        two_decoded_value_bytes.logical_safetensors_parse_work_bytes()
    );
    assert_ne!(one_decoded_value_byte, two_decoded_value_bytes);
}

#[test]
fn decoded_names_and_member_aliases_use_json_semantics_without_normalization() {
    let duplicate = r#"{"é":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"middle":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"\u00e9":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
    assert_eq!(
        parse(duplicate.as_bytes(), 0).unwrap_err(),
        PreparedSafetensorsInventoryError::DuplicateTensorName
    );

    let distinct = r#"{"é":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"e\u0301":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"\u0000":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"a/b":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"../x":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
    let facts = parse(distinct.as_bytes(), 0).unwrap();
    let names: Vec<_> = facts
        .tensors()
        .map(|tensor| tensor.name_bytes().to_vec())
        .collect();
    assert!(names.contains(&Vec::new()));
    assert!(names.contains(&vec![0]));
    assert!(names.contains(&b"a/b".to_vec()));
    assert!(names.contains(&b"../x".to_vec()));
    assert!(names.contains(&"é".as_bytes().to_vec()));
    assert!(names.contains(&"e\u{301}".as_bytes().to_vec()));

    let surrogate = parse(
        br#"{"\ud83d\ude80":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#,
        0,
    )
    .unwrap();
    assert_eq!(
        tensor(&surrogate, "🚀".as_bytes()).name_bytes(),
        "🚀".as_bytes()
    );

    let duplicate_member =
        br#"{"a":{"dtype":"F32","d\u0074ype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
    assert!(matches!(
        parse(duplicate_member, 0),
        Err(PreparedSafetensorsInventoryError::DuplicateTensorMember {
            member: TensorMember::Dtype,
            ..
        })
    ));
}

#[test]
fn metadata_is_bounded_string_map_and_values_never_enter_an_arena() {
    let valid = r#"{"__metadata__":{"raw":"é","escaped":"\n","bmp":"\u03bb","pair":"\ud83d\ude80"},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
    let facts = parse(valid.as_bytes(), 0).unwrap();
    assert_eq!(facts.decoded_metadata_value_bytes(), 2 + 1 + 2 + 4);
    assert!(facts.decoded_metadata_value_bytes() <= valid.len());

    let repeated = br#"{"__metadata__":{},"__metadata__":{},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
    assert_eq!(
        parse(repeated, 0).unwrap_err(),
        PreparedSafetensorsInventoryError::DuplicateMetadataMember
    );
    let duplicate_key = r#"{"__metadata__":{"é":"x","middle":"z","\u00e9":"y"},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
    assert_eq!(
        parse(duplicate_key.as_bytes(), 0).unwrap_err(),
        PreparedSafetensorsInventoryError::DuplicateMetadataKey
    );
    for bad in [
        br#"{"__metadata__":[],"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#.as_slice(),
        br#"{"__metadata__":{"x":1},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#
            .as_slice(),
    ] {
        assert_eq!(
            parse(bad, 0).unwrap_err(),
            PreparedSafetensorsInventoryError::InvalidMetadata
        );
    }
    let missing_colon =
        br#"{"__metadata__":{"x" "y"},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
    let colon_at = missing_colon
        .windows(b" \"y\"".len())
        .position(|window| window == b" \"y\"")
        .unwrap()
        + 1;
    assert_eq!(
        parse(missing_colon, 0).unwrap_err(),
        PreparedSafetensorsInventoryError::MalformedHeader {
            at: colon_at,
            fault: HeaderSyntaxFault::ExpectedColon,
        }
    );
    let unknown =
        br#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0],"bomb":{"deep":[[[[]]]]}}}"#;
    assert!(matches!(
        parse(unknown, 0),
        Err(PreparedSafetensorsInventoryError::UnknownTensorMember { .. })
    ));
}

#[test]
fn dtype_members_and_unsigned_number_grammar_are_closed() {
    for dtype in ["I64", "F64", "f32", "F8_E4M3", "FUTURE"] {
        let header = format!(r#"{{"a":{{"dtype":"{dtype}","shape":[0],"data_offsets":[0,0]}}}}"#);
        assert!(matches!(
            parse(header.as_bytes(), 0),
            Err(PreparedSafetensorsInventoryError::UnsupportedDtype { .. })
        ));
    }

    let scalar = parse(
            br#"{"scalar":{"dtype":"F16","shape":[],"data_offsets":[0,2]},"zero":{"dtype":"BF16","shape":[7,0,9],"data_offsets":[2,2]}}"#,
            2,
        )
        .unwrap();
    assert_eq!(tensor(&scalar, b"scalar").elements(), 1);
    assert_eq!(tensor(&scalar, b"zero").elements(), 0);

    assert_eq!(
        parse(br#"{"a":{"shape":[0],"data_offsets":[0,0]}}"#, 0).unwrap_err(),
        PreparedSafetensorsInventoryError::MissingTensorMember {
            entry: 0,
            member: TensorMember::Dtype,
        }
    );
    assert_eq!(
        parse(
            br#"{"a":{"dtype":"F32","dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#,
            0,
        )
        .unwrap_err(),
        PreparedSafetensorsInventoryError::DuplicateTensorMember {
            entry: 0,
            member: TensorMember::Dtype,
        }
    );
    assert_eq!(
        parse(br#"{"a":{"dtype":1,"shape":[0],"data_offsets":[0,0]}}"#, 0,).unwrap_err(),
        PreparedSafetensorsInventoryError::InvalidTensorMember {
            entry: 0,
            member: TensorMember::Dtype,
        }
    );
    assert_malformed(
        br#"{"a":{"dtype":"\q","shape":[0],"data_offsets":[0,0]}}"#,
        0,
        HeaderSyntaxFault::InvalidStringEscape,
    );

    for header in [
        br#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}}"#.as_slice(),
        br#"{"a":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}}"#.as_slice(),
        br#"{"a":{"dtype":"F32","shape":[1e0],"data_offsets":[0,4]}}"#.as_slice(),
        br#"{"a":{"dtype":"F32","shape":[01],"data_offsets":[0,4]}}"#.as_slice(),
        br#"{"a":{"dtype":"F32","shape":[18446744073709551616],"data_offsets":[0,0]}}"#.as_slice(),
    ] {
        assert_malformed(header, 4, HeaderSyntaxFault::InvalidUnsignedInteger);
    }
    for (header, actual) in [
        (
            br#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0]}}"#.as_slice(),
            1,
        ),
        (
            br#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0,0]}}"#.as_slice(),
            3,
        ),
    ] {
        assert_eq!(
            parse(header, 0).unwrap_err(),
            PreparedSafetensorsInventoryError::InvalidDataOffsetsArity { entry: 0, actual }
        );
    }
}

#[test]
fn local_ranges_require_order_bounds_and_exact_dtype_extent() {
    assert_eq!(
        parse(
            br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
            4,
        )
        .unwrap_err(),
        PreparedSafetensorsInventoryError::ReversedDataRange {
            entry: 0,
            start: 4,
            end: 0,
        }
    );
    assert_eq!(
        parse(
            br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,5]}}"#,
            4,
        )
        .unwrap_err(),
        PreparedSafetensorsInventoryError::DataRangePastEnd {
            entry: 0,
            end: 5,
            data_len: 4,
        }
    );
    for (header, data_len, expected, actual) in [
        (
            br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,3]}}"#.as_slice(),
            3,
            4,
            3,
        ),
        (
            br#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,3]}}"#.as_slice(),
            3,
            2,
            3,
        ),
    ] {
        assert_eq!(
            parse(header, data_len).unwrap_err(),
            PreparedSafetensorsInventoryError::TensorByteLengthMismatch {
                entry: 0,
                expected,
                actual,
            }
        );
    }
}

#[test]
fn global_layout_is_exact_and_zero_ranges_only_live_at_the_frontier() {
    let valid = br#"{"z0":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"zm":{"dtype":"F16","shape":[0],"data_offsets":[4,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"ze":{"dtype":"BF16","shape":[0],"data_offsets":[8,8]}}"#;
    assert!(parse(valid, 8).is_ok());

    for (header, expected_start, actual_start) in [
            (
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#.as_slice(),
                0,
                1,
            ),
            (
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[5,9]}}"#.as_slice(),
                4,
                5,
            ),
            (
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[3,7]}}"#.as_slice(),
                4,
                3,
            ),
            (
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.as_slice(),
                4,
                0,
            ),
            (
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"z":{"dtype":"F32","shape":[0],"data_offsets":[2,2]}}"#.as_slice(),
                4,
                2,
            ),
        ] {
            assert_eq!(
                parse(header, 9).unwrap_err(),
                PreparedSafetensorsInventoryError::NonContiguousDataRange {
                    expected_start,
                    actual_start,
                }
            );
        }
    assert_eq!(
        parse(
            br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            5,
        )
        .unwrap_err(),
        PreparedSafetensorsInventoryError::TrailingPayload {
            covered: 4,
            data_len: 5,
        }
    );

    let empty = br#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"b":{"dtype":"F16","shape":[0],"data_offsets":[0,0]}}"#;
    assert!(parse(empty, 0).is_ok());
}

#[test]
fn f32_alignment_is_only_an_absolute_file_offset_fact() {
    let base = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    for spaces in 0..4 {
        let mut header = base.as_bytes().to_vec();
        header.extend(std::iter::repeat_n(b' ', spaces));
        let facts = parse(&header, 4).unwrap();
        let actual = tensor(&facts, b"a").f32_file_offset_alignment();
        let expected = if (8 + header.len()).is_multiple_of(4) {
            Some(F32FileOffsetAlignment::FourByteAligned)
        } else {
            Some(F32FileOffsetAlignment::Unaligned)
        };
        assert_eq!(actual, expected);
    }

    let half = parse(
            br#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#,
            4,
        )
        .unwrap();
    assert_eq!(tensor(&half, b"a").f32_file_offset_alignment(), None);
    assert_eq!(tensor(&half, b"b").f32_file_offset_alignment(), None);

    let mut shifted = br#"{"half":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"float":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#.to_vec();
    while !(8 + shifted.len()).is_multiple_of(4) {
        shifted.push(b' ');
    }
    let shifted = parse(&shifted, 6).unwrap();
    assert_eq!(
        tensor(&shifted, b"float").f32_file_offset_alignment(),
        Some(F32FileOffsetAlignment::Unaligned)
    );
}

#[test]
fn every_arena_reservation_is_fallible_and_ordered() {
    let requested = [
        3 * std::mem::size_of::<PreparedSafetensorsTensorRecord>(),
        6,
        3 * std::mem::size_of::<u64>(),
        2 * std::mem::size_of::<DecodedSpan>(),
        12,
    ];
    for (index, arena) in AllocationArena::ALL.into_iter().enumerate() {
        let mut attempts = Vec::new();
        let error = parse_prepared_safetensors_header_inventory_with_reserve_probe(
            CANONICAL.as_bytes(),
            &plan(CANONICAL.len(), 10),
            &wide_limits(),
            &mut |actual, requested_bytes| {
                attempts.push(actual);
                if actual == arena {
                    Err(PreparedSafetensorsInventoryError::AllocationFailed {
                        arena: actual,
                        requested_bytes,
                    })
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            PreparedSafetensorsInventoryError::AllocationFailed {
                arena,
                requested_bytes: requested[index],
            }
        );
        assert_eq!(attempts, AllocationArena::ALL[..=index]);
    }
}

#[test]
fn arithmetic_and_second_pass_invariants_fail_closed_with_typed_errors() {
    let unlimited = PreparedSafetensorsInventoryLimits {
        max_tensors: u64::MAX,
        max_tensor_name_bytes: u64::MAX,
        max_rank: u64::MAX,
        max_dimension: u64::MAX,
        max_aggregate_elements: u64::MAX,
        max_metadata_bytes: u64::MAX,
        max_parse_work_bytes: u64::MAX,
    };
    let mut record_overflow = InventoryCensus {
        tensor_count: u64::MAX,
        ..InventoryCensus::default()
    };
    assert_eq!(
        finish_census(0, &unlimited, &mut record_overflow).unwrap_err(),
        PreparedSafetensorsInventoryError::ArithmeticOverflow {
            expression: InventoryExpression::MetadataBytes,
        }
    );
    let mut parse_work_overflow = InventoryCensus {
        metadata_key_count: u64::MAX,
        ..InventoryCensus::default()
    };
    assert_eq!(
        finish_census(0, &unlimited, &mut parse_work_overflow).unwrap_err(),
        PreparedSafetensorsInventoryError::ArithmeticOverflow {
            expression: InventoryExpression::SafetensorsParseWorkBytes,
        }
    );
    assert_eq!(
        checked_product(u64::MAX, 1, InventoryExpression::TensorElements).unwrap(),
        u64::MAX
    );
    assert_eq!(
        checked_product(u64::MAX, 2, InventoryExpression::TensorElements).unwrap_err(),
        PreparedSafetensorsInventoryError::ArithmeticOverflow {
            expression: InventoryExpression::TensorElements,
        }
    );
    assert_eq!(
        checked_product(
            u64::MAX,
            PreparedSafetensorsDtype::F32.width(),
            InventoryExpression::TensorSourceBytes,
        )
        .unwrap_err(),
        PreparedSafetensorsInventoryError::ArithmeticOverflow {
            expression: InventoryExpression::TensorSourceBytes,
        }
    );

    #[cfg(target_pointer_width = "64")]
    {
        let shape_overflow =
            br#"{"a":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#;
        assert_eq!(
            parse_prepared_safetensors_header_inventory(
                shape_overflow,
                &plan(shape_overflow.len(), 0),
                &unlimited,
            )
            .unwrap_err(),
            PreparedSafetensorsInventoryError::ArithmeticOverflow {
                expression: InventoryExpression::TensorElements,
            }
        );
        let source_overflow =
            br#"{"a":{"dtype":"F32","shape":[18446744073709551615],"data_offsets":[0,0]}}"#;
        assert_eq!(
            parse_prepared_safetensors_header_inventory(
                source_overflow,
                &plan(source_overflow.len(), 0),
                &unlimited,
            )
            .unwrap_err(),
            PreparedSafetensorsInventoryError::ArithmeticOverflow {
                expression: InventoryExpression::TensorSourceBytes,
            }
        );
    }

    let expected = InventoryCensus {
        tensor_count: 1,
        ..InventoryCensus::default()
    };
    assert_eq!(
        validate_second_pass(expected, InventoryCensus::default(), true).unwrap_err(),
        PreparedSafetensorsInventoryError::SecondPassCensusMismatch
    );
    assert_eq!(
        validate_second_pass(expected, expected, false).unwrap_err(),
        PreparedSafetensorsInventoryError::SecondPassCensusMismatch
    );
}
