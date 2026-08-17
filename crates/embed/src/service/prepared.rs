//! Sealed native embedding attestation contracts.

use crate::error::{EmbedError, Result};
use std::num::{NonZeroU64, NonZeroUsize};

/// **Unstable**: smallest accepted attestation report in bytes.
pub const MIN_ATTESTATION_REPORT_BYTES: usize = 1;

/// **Unstable**: largest accepted attestation report in bytes.
pub const MAX_ATTESTATION_REPORT_BYTES: usize = 4096;

/// **Unstable**: finite shared admission ceilings for one prepared native resource domain.
#[derive(Debug)]
pub struct NativeResourceBudget {
    max_concurrent_preparations: NonZeroUsize,
    max_concurrent_encodes: NonZeroUsize,
    max_retained_bytes: NonZeroU64,
    max_transient_work_bytes: NonZeroU64,
    total_accounted_bytes: u64,
}

impl NativeResourceBudget {
    /// Constructs independent retained and transient-work pools after checked accounting.
    pub fn try_new(
        max_concurrent_preparations: NonZeroUsize,
        max_concurrent_encodes: NonZeroUsize,
        max_retained_bytes: NonZeroU64,
        max_transient_work_bytes: NonZeroU64,
    ) -> Result<Self> {
        let retained_bytes = max_retained_bytes.get();
        let transient_work_bytes = max_transient_work_bytes.get();
        let total_accounted_bytes = retained_bytes.checked_add(transient_work_bytes).ok_or(
            EmbedError::ResourceBudgetOverflow {
                retained_bytes,
                transient_work_bytes,
            },
        )?;

        Ok(Self {
            max_concurrent_preparations,
            max_concurrent_encodes,
            max_retained_bytes,
            max_transient_work_bytes,
            total_accounted_bytes,
        })
    }

    /// Returns the maximum number of concurrently admitted preparations.
    pub fn max_concurrent_preparations(&self) -> NonZeroUsize {
        self.max_concurrent_preparations
    }

    /// Returns the maximum number of concurrently admitted encode jobs.
    pub fn max_concurrent_encodes(&self) -> NonZeroUsize {
        self.max_concurrent_encodes
    }

    /// Returns the retained-pool byte ceiling.
    pub fn max_retained_bytes(&self) -> NonZeroU64 {
        self.max_retained_bytes
    }

    /// Returns the transient-work-pool byte ceiling.
    pub fn max_transient_work_bytes(&self) -> NonZeroU64 {
        self.max_transient_work_bytes
    }

    /// Returns the checked sum of both independent byte-pool ceilings.
    pub fn total_accounted_bytes(&self) -> u64 {
        self.total_accounted_bytes
    }
}

/// **Unstable**: Lattice-owned immutable evidence produced by a caller attestor.
///
/// A successful attestor transfers these bytes into Lattice. The private representation exposes
/// no mutable alias, and a future prepared service retains the value with its sealed model.
#[derive(Debug, PartialEq, Eq)]
pub struct OpaqueAttestationReport(Box<[u8]>);

impl OpaqueAttestationReport {
    /// Constructs a report after enforcing the closed public byte bound.
    pub fn try_from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let length = bytes.len();
        if !(MIN_ATTESTATION_REPORT_BYTES..=MAX_ATTESTATION_REPORT_BYTES).contains(&length) {
            return Err(EmbedError::AttestationReportSize {
                length,
                min: MIN_ATTESTATION_REPORT_BYTES,
                max: MAX_ATTESTATION_REPORT_BYTES,
            });
        }
        Ok(Self(bytes.into_boxed_slice()))
    }

    /// Borrows the exact immutable report bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// **Unstable**: caller implementation of bounded streaming checkpoint attestation.
///
/// Lattice calls [`CheckpointAttestor::begin`] once with the exact file count, then emits files in
/// strict lexicographic order by normalized logical relative-path bytes. Each file has one
/// [`CheckpointAttestor::begin_file`] call, zero or more contiguous chunk calls, and one
/// [`CheckpointAttestor::end_file`] call before the next file begins. Every non-final chunk is
/// exactly 1 MiB; the final chunk is 1..=1 MiB; an empty file has no chunk calls. Chunks cover
/// exactly the declared length. Any callback error aborts the attestation pass and prevents
/// prepared-service publication. [`CheckpointAttestor::finish`] consumes the fresh attestor and
/// transfers its bounded immutable report to Lattice.
pub trait CheckpointAttestor: Send + 'static {
    /// Starts one inventory pass with its exact file count.
    fn begin(&mut self, file_count: u64) -> Result<()>;

    /// Starts one normalized logical file and declares its exact byte length.
    fn begin_file(&mut self, logical_path: &[u8], declared_len: u64) -> Result<()>;

    /// Receives the next contiguous canonical chunk for the current file.
    fn chunk(&mut self, bytes: &[u8]) -> Result<()>;

    /// Completes the current file after exactly its declared bytes were supplied.
    fn end_file(&mut self) -> Result<()>;

    /// Finalizes this fresh pass and transfers its immutable report bytes.
    fn finish(self) -> Result<OpaqueAttestationReport>;
}

#[cfg_attr(not(test), allow(dead_code))]
mod resource;
