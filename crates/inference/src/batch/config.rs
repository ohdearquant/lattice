//! Configuration for the continuous batching engine.
//!
//! [`BatchConfig`] controls resource limits and scheduling parameters
//! for the iteration-level batch scheduler defined in ADR-048.

/// Model size in parameters; determines safe chunk_size upper bound.
///
/// The Metal device watchdog fires at ~4 s. At chunk=512 the 0.8B model
/// takes ~50 ms/chunk (far below the watchdog). Larger models saturate
/// memory bandwidth faster, so the empirical safe limit scales down.
///
/// Profile evidence for the 0.8B fixture (BF16, M-series, ~50 ms/chunk):
///
/// | chunk | wall-time (ms) | watchdog margin |
/// |-------|----------------|-----------------|
/// |   256 |           ~25  |   >99%          |
/// |   512 |           ~50  |   >98%          |
/// |  1024 |          ~100  |   >97%          |
/// |  2048 |          ~200  |   >95%          |
///
/// For 7B and larger models no fixture exists in this repo; those entries
/// are documented as future profiling targets and the conservative bound
/// (512) is preserved until measurements exist.
#[inline]
pub fn safe_chunk_limit(model_params: u64) -> usize {
    match model_params {
        0 => 512,                              // unspecified — keep conservative default
        1..=999_999_999 => 2048,               // ≤ 1B params: well under watchdog at 2048
        1_000_000_000..=6_999_999_999 => 1024, // 1B–7B: future profiling pending
        _ => 512,                              // ≥ 7B: original conservative bound
    }
}

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
    /// decode steps, bounding prefill latency spikes. The upper bound is
    /// model-param-aware (see [`safe_chunk_limit`]) to stay within the Metal
    /// device timeout (~4 s, ADR-048 R3). Must be > 0.
    pub chunk_size: usize,

    /// KV pages reserved exclusively for prefill allocations.
    ///
    /// When free pages fall below this threshold the scheduler stops admitting
    /// new sequences rather than evicting running ones (Phase 1 policy).
    pub prefill_reserve_pages: usize,

    /// Total model parameter count used to derive a safe `chunk_size` upper
    /// bound. Set to `0` when the model size is not known at config time;
    /// that falls back to the conservative 512-token limit from ADR-048 R3.
    pub model_params: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_seq_len: 4096,
            chunk_size: 512,
            prefill_reserve_pages: 8,
            model_params: 0,
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
        let limit = safe_chunk_limit(self.model_params);
        if self.chunk_size > limit {
            return Err(format!(
                "chunk_size {} exceeds model-aware safety limit {} \
                 (ADR-048 R3: Metal device timeout, model_params={})",
                self.chunk_size, limit, self.model_params
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
    fn chunk_size_exceeds_512_is_invalid_unspecified_model() {
        let cfg = BatchConfig {
            chunk_size: 513,
            model_params: 0,
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("513"), "error should cite the value");
        assert!(err.contains("512"), "error should cite the limit");
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

    // --- Model-aware chunk_size policy ---

    #[test]
    fn small_model_allows_chunk_1024() {
        // 0.8B model: safe_chunk_limit = 2048
        let cfg = BatchConfig {
            chunk_size: 1024,
            model_params: 800_000_000,
            ..Default::default()
        };
        assert!(
            cfg.validate().is_ok(),
            "1024-token chunk must be accepted for a 0.8B model"
        );
    }

    #[test]
    fn small_model_allows_chunk_2048() {
        let cfg = BatchConfig {
            chunk_size: 2048,
            max_seq_len: 8192,
            model_params: 800_000_000,
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn small_model_rejects_chunk_above_2048() {
        let cfg = BatchConfig {
            chunk_size: 2049,
            max_seq_len: 8192,
            model_params: 800_000_000,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn mid_model_allows_chunk_1024() {
        // 3B model: safe_chunk_limit = 1024
        let cfg = BatchConfig {
            chunk_size: 1024,
            model_params: 3_000_000_000,
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn mid_model_rejects_chunk_above_1024() {
        let cfg = BatchConfig {
            chunk_size: 1025,
            max_seq_len: 8192,
            model_params: 3_000_000_000,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn large_model_limit_is_512() {
        assert_eq!(safe_chunk_limit(7_000_000_000), 512);
        assert_eq!(safe_chunk_limit(27_000_000_000), 512);
    }

    #[test]
    fn unspecified_model_limit_is_512() {
        assert_eq!(safe_chunk_limit(0), 512);
    }

    #[test]
    fn small_model_limit_is_2048() {
        assert_eq!(safe_chunk_limit(800_000_000), 2048);
    }
}
