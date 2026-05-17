use std::fmt::{Display, Formatter};

/// **Stable**: primary error type consumed by `lattice-embed`; adding new variants
/// is backward-compatible, but removing or renaming them requires a SemVer bump.
#[derive(Debug)]
pub enum InferenceError {
    ModelNotFound(String),
    UnsupportedModel(String),
    InvalidSafetensors(String),
    MissingTensor(String),
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    Tokenizer(String),
    Download(String),
    ChecksumMismatch {
        file: String,
        expected: String,
        actual: String,
    },
    Io(std::io::Error),
    Inference(String),
    /// Caller-supplied input was invalid (empty prompt, token ID out of
    /// range, etc.) — distinct from a missing model or runtime failure.
    InvalidInput(String),
}

impl Display for InferenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelNotFound(msg) => write!(f, "Model not found: {msg}"),
            Self::UnsupportedModel(msg) => write!(f, "Unsupported model: {msg}"),
            Self::InvalidSafetensors(msg) => write!(f, "Invalid safetensors file: {msg}"),
            Self::MissingTensor(name) => write!(f, "Missing tensor: {name}"),
            Self::ShapeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "Shape mismatch for tensor {name}: expected {expected:?}, got {actual:?}"
            ),
            Self::Tokenizer(msg) => write!(f, "Tokenizer error: {msg}"),
            Self::Download(msg) => write!(f, "Download failed: {msg}"),
            Self::ChecksumMismatch {
                file,
                expected,
                actual,
            } => write!(
                f,
                "Checksum mismatch for {file}: expected {expected}, got {actual}"
            ),
            Self::Io(err) => write!(f, "IO error: {err}"),
            Self::Inference(msg) => write!(f, "Inference error: {msg}"),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl std::error::Error for InferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for InferenceError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_input_display() {
        let e = InferenceError::InvalidInput("Empty prompt".into());
        assert_eq!(e.to_string(), "Invalid input: Empty prompt");
    }
}
