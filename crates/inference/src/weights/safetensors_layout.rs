use crate::error::InferenceError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SafetensorsDType {
    pub(crate) name: &'static str,
    pub(crate) bits_per_element: usize,
}

pub(crate) fn safetensors_dtype(name: &str) -> Option<SafetensorsDType> {
    let (name, bits_per_element) = match name {
        "F4" => ("F4", 4),
        "F6_E2M3" => ("F6_E2M3", 6),
        "F6_E3M2" => ("F6_E3M2", 6),
        "BOOL" => ("BOOL", 8),
        "U8" => ("U8", 8),
        "I8" => ("I8", 8),
        "F8_E4M3" => ("F8_E4M3", 8),
        "F8_E5M2" => ("F8_E5M2", 8),
        "F8_E8M0" => ("F8_E8M0", 8),
        "F8_E4M3FNUZ" => ("F8_E4M3FNUZ", 8),
        "F8_E5M2FNUZ" => ("F8_E5M2FNUZ", 8),
        "I16" => ("I16", 16),
        "U16" => ("U16", 16),
        "F16" => ("F16", 16),
        "BF16" => ("BF16", 16),
        "I32" => ("I32", 32),
        "U32" => ("U32", 32),
        "F32" => ("F32", 32),
        "I64" => ("I64", 64),
        "U64" => ("U64", 64),
        "F64" => ("F64", 64),
        "C64" => ("C64", 64),
        _ => return None,
    };
    Some(SafetensorsDType {
        name,
        bits_per_element,
    })
}

pub(crate) struct SafetensorsLayoutEntry<'a> {
    pub(crate) name: &'a str,
    pub(crate) dtype: &'a str,
    pub(crate) shape: &'a [usize],
    pub(crate) start: usize,
    pub(crate) end: usize,
}

pub(crate) fn validate_safetensors_layout(
    source: &str,
    data_len: usize,
    entries: &[SafetensorsLayoutEntry<'_>],
) -> Result<(), InferenceError> {
    for entry in entries {
        if entry.start > entry.end {
            return Err(InferenceError::InvalidSafetensors(format!(
                "{source}: tensor {} has invalid data_offsets [{}, {})",
                entry.name, entry.start, entry.end
            )));
        }
        if entry.end > data_len {
            return Err(InferenceError::InvalidSafetensors(format!(
                "{source}: tensor {} data_offsets end={} past data_len={data_len}",
                entry.name, entry.end
            )));
        }

        let numel = entry.shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "{source}: tensor {} shape {:?} overflows usize",
                    entry.name, entry.shape
                ))
            })
        })?;
        let dtype = safetensors_dtype(entry.dtype).ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!(
                "{source}: tensor {} has unrecognized SafeTensors dtype {:?}",
                entry.name, entry.dtype
            ))
        })?;
        let total_bits = numel.checked_mul(dtype.bits_per_element).ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!(
                "{source}: tensor {} bit length overflows usize",
                entry.name
            ))
        })?;
        if total_bits % 8 != 0 {
            return Err(InferenceError::InvalidSafetensors(format!(
                "{source}: tensor {} sub-byte dtype {} with shape {:?} produces {total_bits} bits, \
                 which is not byte-aligned",
                entry.name, dtype.name, entry.shape
            )));
        }
        let expected = total_bits / 8;
        let actual = entry.end - entry.start;
        if actual != expected {
            return Err(InferenceError::InvalidSafetensors(format!(
                "{source}: tensor {} byte length mismatch for {} {:?}: expected {expected}, \
                 got {actual}",
                entry.name, dtype.name, entry.shape
            )));
        }
    }

    let mut ranges: Vec<_> = entries
        .iter()
        .map(|entry| (entry.start, entry.end, entry.name))
        .collect();
    ranges.sort_unstable();
    let mut previous_end = 0usize;
    for (start, end, name) in ranges {
        if start != previous_end {
            return Err(InferenceError::InvalidSafetensors(format!(
                "{source}: data_offsets non-contiguous at tensor {name}: expected \
                 start={previous_end}, got [{start}, {end})"
            )));
        }
        previous_end = end;
    }
    if previous_end != data_len {
        return Err(InferenceError::InvalidSafetensors(format!(
            "{source}: data section is {data_len} bytes but tensors cover {previous_end} bytes \
             (trailing or missing payload)"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_dtype_table_has_exact_bit_widths() {
        let cases = [
            ("F4", 4),
            ("F6_E2M3", 6),
            ("F6_E3M2", 6),
            ("BOOL", 8),
            ("U8", 8),
            ("I8", 8),
            ("F8_E4M3", 8),
            ("F8_E5M2", 8),
            ("F8_E8M0", 8),
            ("F8_E4M3FNUZ", 8),
            ("F8_E5M2FNUZ", 8),
            ("I16", 16),
            ("U16", 16),
            ("F16", 16),
            ("BF16", 16),
            ("I32", 32),
            ("U32", 32),
            ("F32", 32),
            ("I64", 64),
            ("U64", 64),
            ("F64", 64),
            ("C64", 64),
        ];

        for (name, bits) in cases {
            assert_eq!(
                safetensors_dtype(name),
                Some(SafetensorsDType {
                    name,
                    bits_per_element: bits,
                })
            );
        }
        assert_eq!(safetensors_dtype("FUTURE_DTYPE"), None);
    }
}
