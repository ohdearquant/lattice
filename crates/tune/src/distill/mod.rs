//! Knowledge distillation module
//!
//! This module provides infrastructure for distilling knowledge from
//! large teacher models (Claude, GPT, Gemini) into smaller student models.
//!
//! # Architecture
//!
//! ```text
//! Raw Data → Teacher (LLM) → Soft Labels → Dataset → Student Training
//! ```
//!
//! # Example
//!
//! ```ignore
//! use lattice_tune::distill::{TeacherConfig, DistillationPipeline, DistillationConfig};
//!
//! // Configure the teacher model
//! let teacher = TeacherConfig::claude_sonnet()
//!     .temperature(0.3)
//!     .build();
//!
//! // Create distillation pipeline
//! let pipeline = DistillationPipeline::new(teacher, DistillationConfig::default());
//!
//! // Label raw examples
//! let labeled = pipeline.label_batch(&raw_examples).await?;
//! ```

mod pipeline;
mod teacher;

pub use pipeline::{
    DistillationConfig, DistillationPipeline, DistillationStats, LabelingResult, RawExample,
};
pub use teacher::{EndpointSecurity, TeacherConfig, TeacherConfigBuilder, TeacherProvider};
