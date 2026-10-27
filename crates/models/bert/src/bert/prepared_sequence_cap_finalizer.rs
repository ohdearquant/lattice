//! Dormant final validation of prepared-BERT sequence-cap facts against matched tensors.

use super::prepared_sequence_cap::{
    PreparedBertSequenceCap, PreparedBertSequenceCapKey, PreparedBertSequenceCapSource,
};
use super::prepared_tensor_inventory::MatchedBertInventoryFacts;
use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertSequenceCapFinalizationError {
    ExceedsValidatedPositionRows {
        capped_value: NonZeroU64,
        validated_position_rows: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RealizedPreparedBertSequenceCapFacts {
    source: PreparedBertSequenceCapSource,
    key: PreparedBertSequenceCapKey,
    raw_value: NonZeroU64,
    capped_value: NonZeroU64,
    validated_position_rows: usize,
}

impl RealizedPreparedBertSequenceCapFacts {
    pub(super) fn source(self) -> PreparedBertSequenceCapSource {
        self.source
    }

    pub(super) fn key(self) -> PreparedBertSequenceCapKey {
        self.key
    }

    pub(super) fn raw_value(self) -> NonZeroU64 {
        self.raw_value
    }

    pub(super) fn capped_value(self) -> NonZeroU64 {
        self.capped_value
    }

    pub(super) fn validated_position_rows(self) -> usize {
        self.validated_position_rows
    }
}

pub(super) fn finalize_prepared_bert_sequence_cap(
    sequence_cap: PreparedBertSequenceCap,
    matched_inventory: MatchedBertInventoryFacts,
) -> Result<RealizedPreparedBertSequenceCapFacts, PreparedBertSequenceCapFinalizationError> {
    finalize_prepared_bert_sequence_cap_with_position_rows(
        sequence_cap,
        matched_inventory.geometry().position_embeddings(),
    )
}

fn finalize_prepared_bert_sequence_cap_with_position_rows(
    sequence_cap: PreparedBertSequenceCap,
    validated_position_rows: usize,
) -> Result<RealizedPreparedBertSequenceCapFacts, PreparedBertSequenceCapFinalizationError> {
    let capped_value = sequence_cap.capped_value();
    if u64::try_from(validated_position_rows)
        .is_ok_and(|validated_position_rows| capped_value.get() > validated_position_rows)
    {
        return Err(
            PreparedBertSequenceCapFinalizationError::ExceedsValidatedPositionRows {
                capped_value,
                validated_position_rows,
            },
        );
    }
    Ok(RealizedPreparedBertSequenceCapFacts {
        source: sequence_cap.source(),
        key: sequence_cap.key(),
        raw_value: sequence_cap.raw_value(),
        capped_value,
        validated_position_rows,
    })
}

#[cfg(test)]
mod tests;
