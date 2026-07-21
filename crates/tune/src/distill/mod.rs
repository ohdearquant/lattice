//! Knowledge-distillation interfaces for producing soft intent labels.
//!
//! It configures a teacher, sanitizes raw conversational input, records label
//! outcomes, and converts successful results after embeddings are supplied.
//! See `docs/distill.md` for the full data flow, security policy, and current
//! placeholder boundary.

mod pipeline;
mod teacher;

pub use pipeline::{
    DistillationConfig, DistillationPipeline, DistillationStats, LabelSource, LabelingResult,
    RawExample, TeacherAuth,
};
pub use teacher::{EndpointSecurity, TeacherConfig, TeacherConfigBuilder, TeacherProvider};
