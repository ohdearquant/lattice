//! Qwen3.5 family compatibility exports.

#[doc(inline)]
pub use lattice_model_qwen35::speculative::MAX_NGRAM_LIMIT;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::NgramSpeculator;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::argmax;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::MtpConfig;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::MtpMetrics;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::MtpVerifyResult;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::MtpForwardOutput;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::MtpDraft;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::MtpWeights;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::MtpLayerWeights;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::MtpAttentionWeights;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::MtpMoeWeights;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::MtpVerifier;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::MtpTargetVerifier;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::mtp_verify_draft_with_seed;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::mtp_verify_draft;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::RejectionSampleResult;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::rejection_sample_draft;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::verify_draft;

#[doc(inline)]
pub use lattice_model_qwen35::speculative::generate_with_speculation;
