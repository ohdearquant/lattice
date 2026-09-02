//! Dormant SafeTensors framing facts for sealed native preparation.
//!
//! This module is a pure scalar planner. It performs no I/O, allocation,
//! mapping, JSON parsing, or tensor/layout validation and has no live caller.
//! A successful plan is not an inspected or trusted checkpoint, does not bind
//! bytes to an opened handle, and provides no TOCTOU protection or admission
//! lease. A later reader must obtain the prefix and declared length from the
//! same bounded handle, read exactly the planned header range, and separately
//! validate the complete inventory before mapping or loading.

use std::num::NonZeroU64;
use std::ops::Range;

/// Prepared mode cannot admit a header the existing ADR-003 loader rejects.
pub(crate) const PREPARED_SAFETENSORS_HARD_MAX_HEADER_BYTES: u64 =
    super::f32_weights::max_safetensors_header_bytes() as u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreparedSafetensorsSizeAxis {
    WeightFileBytes,
    HeaderBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreparedSafetensorsHeaderError {
    Exceeded {
        axis: PreparedSafetensorsSizeAxis,
        actual: u64,
        limit: u64,
    },
    PlatformUnrepresentable {
        axis: PreparedSafetensorsSizeAxis,
        value: u64,
    },
    HeaderFrameExceedsFileLimit {
        header_frame: u64,
        file_limit: u64,
    },
    FileTooShort {
        declared_file_len: u64,
    },
    EmptyHeader,
    HeaderFrameOverflow {
        header_len: u64,
    },
    HeaderPastEnd {
        header_end: u64,
        declared_file_len: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PreparedSafetensorsFramingLimits {
    max_weight_file_bytes: NonZeroU64,
    max_header_bytes: NonZeroU64,
    platform_span_max: u64,
}

impl PreparedSafetensorsFramingLimits {
    pub(crate) fn try_new(
        max_weight_file_bytes: NonZeroU64,
        max_header_bytes: NonZeroU64,
    ) -> Result<Self, PreparedSafetensorsHeaderError> {
        let usize_max = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
        let isize_max = u64::try_from(isize::MAX).unwrap_or(u64::MAX);
        let platform_span_max = usize_max.min(isize_max);
        Self::try_new_with_platform_max(max_weight_file_bytes, max_header_bytes, platform_span_max)
    }

    fn try_new_with_platform_max(
        max_weight_file_bytes: NonZeroU64,
        max_header_bytes: NonZeroU64,
        platform_span_max: u64,
    ) -> Result<Self, PreparedSafetensorsHeaderError> {
        let file_limit = max_weight_file_bytes.get();
        let header_limit = max_header_bytes.get();

        if header_limit > PREPARED_SAFETENSORS_HARD_MAX_HEADER_BYTES {
            return Err(PreparedSafetensorsHeaderError::Exceeded {
                axis: PreparedSafetensorsSizeAxis::HeaderBytes,
                actual: header_limit,
                limit: PREPARED_SAFETENSORS_HARD_MAX_HEADER_BYTES,
            });
        }
        validate_platform(
            PreparedSafetensorsSizeAxis::WeightFileBytes,
            file_limit,
            platform_span_max,
        )?;
        validate_platform(
            PreparedSafetensorsSizeAxis::HeaderBytes,
            header_limit,
            platform_span_max,
        )?;
        let header_frame = checked_header_end(header_limit)?;
        if header_frame > file_limit {
            return Err(
                PreparedSafetensorsHeaderError::HeaderFrameExceedsFileLimit {
                    header_frame,
                    file_limit,
                },
            );
        }

        Ok(Self {
            max_weight_file_bytes,
            max_header_bytes,
            platform_span_max,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedSafetensorsHeaderPlan {
    declared_file_len: usize,
    header_len: usize,
    header_end: usize,
    data_len: usize,
}

impl PreparedSafetensorsHeaderPlan {
    pub(crate) fn declared_file_len(&self) -> usize {
        self.declared_file_len
    }

    pub(crate) fn header_len(&self) -> usize {
        self.header_len
    }

    pub(crate) fn header_end(&self) -> usize {
        self.header_end
    }

    pub(crate) fn data_len(&self) -> usize {
        self.data_len
    }

    pub(crate) fn header_range(&self) -> Range<usize> {
        8..self.header_end
    }

    pub(crate) fn data_range(&self) -> Range<usize> {
        self.header_end..self.declared_file_len
    }
}

pub(crate) fn plan_prepared_safetensors_header(
    prefix: [u8; 8],
    declared_file_len: u64,
    limits: &PreparedSafetensorsFramingLimits,
) -> Result<PreparedSafetensorsHeaderPlan, PreparedSafetensorsHeaderError> {
    if declared_file_len > limits.max_weight_file_bytes.get() {
        return Err(PreparedSafetensorsHeaderError::Exceeded {
            axis: PreparedSafetensorsSizeAxis::WeightFileBytes,
            actual: declared_file_len,
            limit: limits.max_weight_file_bytes.get(),
        });
    }
    validate_platform(
        PreparedSafetensorsSizeAxis::WeightFileBytes,
        declared_file_len,
        limits.platform_span_max,
    )?;
    if declared_file_len < 8 {
        return Err(PreparedSafetensorsHeaderError::FileTooShort { declared_file_len });
    }

    let header_len = u64::from_le_bytes(prefix);
    if header_len > limits.max_header_bytes.get() {
        return Err(PreparedSafetensorsHeaderError::Exceeded {
            axis: PreparedSafetensorsSizeAxis::HeaderBytes,
            actual: header_len,
            limit: limits.max_header_bytes.get(),
        });
    }
    validate_platform(
        PreparedSafetensorsSizeAxis::HeaderBytes,
        header_len,
        limits.platform_span_max,
    )?;
    if header_len == 0 {
        return Err(PreparedSafetensorsHeaderError::EmptyHeader);
    }
    let header_end = checked_header_end(header_len)?;
    if header_end > declared_file_len {
        return Err(PreparedSafetensorsHeaderError::HeaderPastEnd {
            header_end,
            declared_file_len,
        });
    }
    let data_len = declared_file_len.checked_sub(header_end).ok_or(
        PreparedSafetensorsHeaderError::HeaderPastEnd {
            header_end,
            declared_file_len,
        },
    )?;

    Ok(PreparedSafetensorsHeaderPlan {
        declared_file_len: to_usize(
            PreparedSafetensorsSizeAxis::WeightFileBytes,
            declared_file_len,
        )?,
        header_len: to_usize(PreparedSafetensorsSizeAxis::HeaderBytes, header_len)?,
        header_end: to_usize(PreparedSafetensorsSizeAxis::HeaderBytes, header_end)?,
        data_len: to_usize(PreparedSafetensorsSizeAxis::WeightFileBytes, data_len)?,
    })
}

fn validate_platform(
    axis: PreparedSafetensorsSizeAxis,
    value: u64,
    platform_max: u64,
) -> Result<(), PreparedSafetensorsHeaderError> {
    if value > platform_max {
        return Err(PreparedSafetensorsHeaderError::PlatformUnrepresentable { axis, value });
    }
    Ok(())
}

fn checked_header_end(header_len: u64) -> Result<u64, PreparedSafetensorsHeaderError> {
    8_u64
        .checked_add(header_len)
        .ok_or(PreparedSafetensorsHeaderError::HeaderFrameOverflow { header_len })
}

fn to_usize(
    axis: PreparedSafetensorsSizeAxis,
    value: u64,
) -> Result<usize, PreparedSafetensorsHeaderError> {
    usize::try_from(value)
        .map_err(|_| PreparedSafetensorsHeaderError::PlatformUnrepresentable { axis, value })
}

#[cfg(test)]
mod tests {
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
            PreparedSafetensorsFramingLimits::try_new_with_platform_max(nz(32), nz(24), 32,)
                .is_ok()
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
}
