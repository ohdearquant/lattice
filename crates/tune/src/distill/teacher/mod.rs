//! Teacher model configuration

mod config;
mod security;

pub use config::{TeacherConfig, TeacherConfigBuilder};
pub use security::EndpointSecurity;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Supported teacher model providers
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum TeacherProvider {
    /// Anthropic Claude models
    Claude,
    /// OpenAI GPT models
    OpenAI,
    /// Google Gemini models
    Gemini,
    /// Local model (e.g., via Ollama)
    Local,
    /// Custom provider
    Custom(String),
}

impl std::fmt::Display for TeacherProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TeacherProvider::Claude => write!(f, "claude"),
            TeacherProvider::OpenAI => write!(f, "openai"),
            TeacherProvider::Gemini => write!(f, "gemini"),
            TeacherProvider::Local => write!(f, "local"),
            TeacherProvider::Custom(name) => write!(f, "custom:{name}"),
        }
    }
}

#[cfg(test)]
mod tests;
