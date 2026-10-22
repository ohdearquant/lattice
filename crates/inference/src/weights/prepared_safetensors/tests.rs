use super::*;

fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

fn limits(file: u64, header: u64) -> PreparedSafetensorsFramingLimits {
    PreparedSafetensorsFramingLimits::try_new(nz(file), nz(header)).unwrap()
}

fn prefix(header_len: u64) -> [u8; 8] {
    header_len.to_le_bytes()
}

#[test]
fn exact_bounds_produce_exact_scalar_ranges() {
    let plan = plan_prepared_safetensors_header(prefix(8), 24, &limits(24, 8)).unwrap();

    assert_eq!(plan.declared_file_len(), 24);
    assert_eq!(plan.header_len(), 8);
    assert_eq!(plan.header_end(), 16);
    assert_eq!(plan.data_len(), 8);
    assert_eq!(plan.header_range(), 8..16);
    assert_eq!(plan.data_range(), 16..24);
}

#[test]
fn limit_constructor_pins_hard_cap_platform_and_frame_relation() {
    let hard = PREPARED_SAFETENSORS_HARD_MAX_HEADER_BYTES;
    let span_max = u64::try_from(isize::MAX).unwrap();
    assert_eq!(hard, 4_194_304, "prepared ADR-003 cap is an exact golden");
    assert!(PreparedSafetensorsFramingLimits::try_new(nz(hard + 8), nz(hard)).is_ok());
    assert!(PreparedSafetensorsFramingLimits::try_new(nz(span_max), nz(1)).is_ok());
    assert_eq!(
        PreparedSafetensorsFramingLimits::try_new(nz(span_max + 1), nz(1)).unwrap_err(),
        PreparedSafetensorsHeaderError::PlatformUnrepresentable {
            axis: PreparedSafetensorsSizeAxis::WeightFileBytes,
            value: span_max + 1,
        }
    );
    assert_eq!(
        PreparedSafetensorsFramingLimits::try_new(nz(hard + 9), nz(hard + 1)).unwrap_err(),
        PreparedSafetensorsHeaderError::Exceeded {
            axis: PreparedSafetensorsSizeAxis::HeaderBytes,
            actual: hard + 1,
            limit: hard,
        }
    );
    assert_eq!(
        PreparedSafetensorsFramingLimits::try_new_with_platform_max(
            nz(hard + 9),
            nz(hard + 1),
            hard,
        )
        .unwrap_err(),
        PreparedSafetensorsHeaderError::Exceeded {
            axis: PreparedSafetensorsSizeAxis::HeaderBytes,
            actual: hard + 1,
            limit: hard,
        },
        "absolute ADR-003 cap precedes synthetic platform and relation failures",
    );
    assert_eq!(
        PreparedSafetensorsFramingLimits::try_new(nz(15), nz(8)).unwrap_err(),
        PreparedSafetensorsHeaderError::HeaderFrameExceedsFileLimit {
            header_frame: 16,
            file_limit: 15,
        }
    );

    assert!(
        PreparedSafetensorsFramingLimits::try_new_with_platform_max(nz(32), nz(24), 32,).is_ok()
    );
    assert_eq!(
        PreparedSafetensorsFramingLimits::try_new_with_platform_max(nz(33), nz(24), 32)
            .unwrap_err(),
        PreparedSafetensorsHeaderError::PlatformUnrepresentable {
            axis: PreparedSafetensorsSizeAxis::WeightFileBytes,
            value: 33,
        }
    );
    assert_eq!(
        PreparedSafetensorsFramingLimits::try_new_with_platform_max(nz(34), nz(25), 24)
            .unwrap_err(),
        PreparedSafetensorsHeaderError::PlatformUnrepresentable {
            axis: PreparedSafetensorsSizeAxis::WeightFileBytes,
            value: 34,
        }
    );
    assert_eq!(
        PreparedSafetensorsFramingLimits::try_new_with_platform_max(nz(24), nz(17), 16)
            .unwrap_err(),
        PreparedSafetensorsHeaderError::PlatformUnrepresentable {
            axis: PreparedSafetensorsSizeAxis::WeightFileBytes,
            value: 24,
        }
    );
    assert_eq!(
        PreparedSafetensorsFramingLimits::try_new_with_platform_max(nz(24), nz(25), 24)
            .unwrap_err(),
        PreparedSafetensorsHeaderError::PlatformUnrepresentable {
            axis: PreparedSafetensorsSizeAxis::HeaderBytes,
            value: 25,
        },
        "header span platform failure precedes the relation failure",
    );
}

#[test]
fn checked_header_frame_arithmetic_has_exact_overflow_boundary() {
    assert_eq!(checked_header_end(u64::MAX - 8), Ok(u64::MAX));
    assert_eq!(
        checked_header_end(u64::MAX - 7),
        Err(PreparedSafetensorsHeaderError::HeaderFrameOverflow {
            header_len: u64::MAX - 7,
        })
    );
    assert_eq!(
        checked_header_end(u64::MAX),
        Err(PreparedSafetensorsHeaderError::HeaderFrameOverflow {
            header_len: u64::MAX,
        })
    );
}

#[test]
fn little_endian_prefix_is_the_only_header_length_authority() {
    let expected = 0x0000_0000_0001_0203_u64;
    let plan = plan_prepared_safetensors_header(
        [0x03, 0x02, 0x01, 0, 0, 0, 0, 0],
        8 + expected,
        &limits(8 + expected, expected),
    )
    .unwrap();
    assert_eq!(plan.header_len(), usize::try_from(expected).unwrap());
}

#[test]
fn actual_file_and_header_caps_accept_exact_and_reject_plus_one() {
    let bounded = limits(24, 8);
    assert!(plan_prepared_safetensors_header(prefix(8), 24, &bounded).is_ok());
    assert_eq!(
        plan_prepared_safetensors_header(prefix(8), 25, &bounded).unwrap_err(),
        PreparedSafetensorsHeaderError::Exceeded {
            axis: PreparedSafetensorsSizeAxis::WeightFileBytes,
            actual: 25,
            limit: 24,
        }
    );
    assert_eq!(
        plan_prepared_safetensors_header(prefix(9), 24, &bounded).unwrap_err(),
        PreparedSafetensorsHeaderError::Exceeded {
            axis: PreparedSafetensorsSizeAxis::HeaderBytes,
            actual: 9,
            limit: 8,
        }
    );
}

#[test]
fn framing_rejects_short_file_empty_header_and_header_past_end() {
    let bounded = limits(32, 16);
    assert_eq!(
        plan_prepared_safetensors_header(prefix(1), 7, &bounded).unwrap_err(),
        PreparedSafetensorsHeaderError::FileTooShort {
            declared_file_len: 7,
        }
    );
    assert_eq!(
        plan_prepared_safetensors_header(prefix(u64::MAX), 7, &bounded).unwrap_err(),
        PreparedSafetensorsHeaderError::FileTooShort {
            declared_file_len: 7,
        },
        "an impossible-to-read prefix cannot outrank physical framing",
    );
    assert_eq!(
        plan_prepared_safetensors_header(prefix(0), 8, &bounded).unwrap_err(),
        PreparedSafetensorsHeaderError::EmptyHeader
    );
    assert_eq!(
        plan_prepared_safetensors_header(prefix(16), 23, &bounded).unwrap_err(),
        PreparedSafetensorsHeaderError::HeaderPastEnd {
            header_end: 24,
            declared_file_len: 23,
        }
    );
}

#[test]
fn cap_errors_win_before_later_prefix_or_extent_checks() {
    let bounded = limits(24, 8);
    assert_eq!(
        plan_prepared_safetensors_header(prefix(u64::MAX), 25, &bounded).unwrap_err(),
        PreparedSafetensorsHeaderError::Exceeded {
            axis: PreparedSafetensorsSizeAxis::WeightFileBytes,
            actual: 25,
            limit: 24,
        }
    );
    assert_eq!(
        plan_prepared_safetensors_header(prefix(9), 8, &bounded).unwrap_err(),
        PreparedSafetensorsHeaderError::Exceeded {
            axis: PreparedSafetensorsSizeAxis::HeaderBytes,
            actual: 9,
            limit: 8,
        }
    );
}

#[test]
fn header_may_consume_the_file_and_leave_empty_data() {
    let plan = plan_prepared_safetensors_header(prefix(2), 10, &limits(10, 2)).unwrap();
    assert_eq!(plan.header_range(), 8..10);
    assert_eq!(plan.data_range(), 10..10);
    assert_eq!(plan.data_len(), 0);
}
