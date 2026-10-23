//! Canonical tensor-inventory identity framing for sealed native preparation.

use super::limits::{
    PreparationCeilings, PreparationLimitError, TensorCensus, TensorDtype, TensorFact,
};
use sha2::{Digest, Sha256};

const DOMAIN: &[u8] = b"lattice.embedding-tensor-inventory.v1\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TensorInventoryDigest([u8; 32]);

impl TensorInventoryDigest {
    pub(super) fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TensorInventoryEntry<'a> {
    name: &'a [u8],
    source_dtype: &'a str,
    dimensions: &'a [u64],
    accounted_metadata_bytes: u64,
}

impl<'a> TensorInventoryEntry<'a> {
    pub(super) fn new(
        name: &'a [u8],
        source_dtype: &'a str,
        dimensions: &'a [u64],
        accounted_metadata_bytes: u64,
    ) -> Self {
        Self {
            name,
            source_dtype,
            dimensions,
            accounted_metadata_bytes,
        }
    }

    pub(super) fn name(&self) -> &[u8] {
        self.name
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TensorInventoryError {
    MissingInventory,
    InvalidUtf8Name { entry: u64 },
    UnsupportedSourceDtype { entry: u64 },
    DuplicateName,
    NameLengthUnrepresentable { length: u64 },
    RankUnrepresentable { rank: u64 },
    Limit(PreparationLimitError),
}

pub(super) fn digest_tensor_inventory(
    entries: &mut [TensorInventoryEntry<'_>],
    ceilings: &PreparationCeilings,
) -> Result<TensorInventoryDigest, TensorInventoryError> {
    if entries.is_empty() {
        return Err(TensorInventoryError::MissingInventory);
    }

    let mut census = TensorCensus::new();
    for (index, entry) in entries.iter().enumerate() {
        let error_index = u64::try_from(index).unwrap_or(u64::MAX);
        std::str::from_utf8(entry.name)
            .map_err(|_| TensorInventoryError::InvalidUtf8Name { entry: error_index })?;
        let source_dtype = source_dtype(entry.source_dtype)
            .ok_or(TensorInventoryError::UnsupportedSourceDtype { entry: error_index })?;
        let name_length = u64::try_from(entry.name.len()).unwrap_or(u64::MAX);
        checked_name_length_prefix(name_length)?;
        let rank = u64::try_from(entry.dimensions.len()).unwrap_or(u64::MAX);
        checked_rank_prefix(rank)?;
        census
            .push(
                TensorFact::new(
                    name_length,
                    entry.accounted_metadata_bytes,
                    entry.dimensions,
                    source_dtype,
                ),
                ceilings,
            )
            .map_err(TensorInventoryError::Limit)?;
    }

    entries.sort_unstable_by(|left, right| left.name.cmp(right.name));
    if entries.windows(2).any(|pair| pair[0].name == pair[1].name) {
        return Err(TensorInventoryError::DuplicateName);
    }

    let tensor_count = u64::try_from(entries.len()).unwrap_or(u64::MAX);
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN);
    hasher.update(tensor_count.to_be_bytes());
    for (index, entry) in entries.iter().enumerate() {
        let name_length = u64::try_from(entry.name.len()).unwrap_or(u64::MAX);
        hasher.update(checked_name_length_prefix(name_length)?);
        hasher.update(entry.name);
        let error_index = u64::try_from(index).unwrap_or(u64::MAX);
        let dtype = source_dtype(entry.source_dtype)
            .ok_or(TensorInventoryError::UnsupportedSourceDtype { entry: error_index })?;
        hasher.update([dtype.digest_tag()]);
        let rank = u64::try_from(entry.dimensions.len()).unwrap_or(u64::MAX);
        hasher.update(checked_rank_prefix(rank)?);
        for dimension in entry.dimensions {
            hasher.update(dimension.to_be_bytes());
        }
    }
    Ok(TensorInventoryDigest(hasher.finalize().into()))
}

fn source_dtype(label: &str) -> Option<TensorDtype> {
    match label {
        "F32" => Some(TensorDtype::F32),
        "F16" => Some(TensorDtype::F16),
        "BF16" => Some(TensorDtype::Bf16),
        _ => None,
    }
}

fn checked_name_length_prefix(length: u64) -> Result<[u8; 4], TensorInventoryError> {
    u32::try_from(length)
        .map(u32::to_be_bytes)
        .map_err(|_| TensorInventoryError::NameLengthUnrepresentable { length })
}

fn checked_rank_prefix(rank: u64) -> Result<[u8; 4], TensorInventoryError> {
    u32::try_from(rank)
        .map(u32::to_be_bytes)
        .map_err(|_| TensorInventoryError::RankUnrepresentable { rank })
}

#[cfg(test)]
mod tests;
