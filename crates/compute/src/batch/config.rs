//! Configuration for the continuous batching engine.
//!
//! [`BatchConfig`] controls resource limits and scheduling parameters
//! for the iteration-level batch scheduler defined in ADR-048.

/// **Unstable**: configuration for the continuous batching engine; fields may
/// be extended as Phase 2 (disaggregated prefill/decode) is implemented.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of sequences executing concurrently (prefill + decode).
    ///
    /// Determines the size of the [`GdnStatePool`] and the upper bound on
    /// per-iteration batch size. Each sequence occupies one GDN state slot
    /// (9.4 MB for Qwen3.5-2B on M2 Max) and a variable number of KV pages.
    ///
    /// [`GdnStatePool`]: super::worker::GdnStatePool
    pub max_batch_size: usize,

    /// Hard cap on the total sequence length (prompt + generated tokens) for
    /// any single sequence. Sequences that would exceed this limit at admission
    /// time are rejected.
    pub max_seq_len: usize,

    /// Number of prompt tokens processed per prefill chunk.
    ///
    /// Long prompts are split into chunks of this size and interleaved with
    /// decode steps, bounding prefill latency spikes. Must be ≤ 512 to stay
    /// within Metal device timeout (see ADR-048 R3).
    pub chunk_size: usize,

    /// KV pages reserved exclusively for prefill allocations.
    ///
    /// When free pages fall below this threshold the scheduler stops admitting
    /// new sequences rather than evicting running ones (Phase 1 policy).
    pub prefill_reserve_pages: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_seq_len: 4096,
            chunk_size: 512,
            prefill_reserve_pages: 8,
        }
    }
}

impl BatchConfig {
    /// Validate that config values are internally consistent.
    ///
    /// Returns `Err` with a human-readable message on the first violation found.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_batch_size == 0 {
            return Err("max_batch_size must be > 0".into());
        }
        if self.max_seq_len == 0 {
            return Err("max_seq_len must be > 0".into());
        }
        if self.chunk_size == 0 {
            return Err("chunk_size must be > 0".into());
        }
        if self.chunk_size > 512 {
            return Err(format!(
                "chunk_size {} exceeds 512-token safety limit (ADR-048 R3: Metal device timeout)",
                self.chunk_size
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        assert!(BatchConfig::default().validate().is_ok());
    }

    #[test]
    fn zero_batch_size_is_invalid() {
        let cfg = BatchConfig {
            max_batch_size: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_chunk_size_is_invalid() {
        let cfg = BatchConfig {
            chunk_size: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn chunk_size_exceeds_512_is_invalid() {
        let cfg = BatchConfig {
            chunk_size: 513,
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("512"), "error message should cite the limit");
    }

    #[test]
    fn chunk_size_512_is_valid() {
        let cfg = BatchConfig {
            chunk_size: 512,
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn zero_max_seq_len_is_invalid() {
        let cfg = BatchConfig {
            max_seq_len: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }
}
