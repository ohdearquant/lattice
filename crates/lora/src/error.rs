//! The error type returned by `lattice-lora`'s fallible APIs.
//!
//! Entry points in other crates keep their own published error types and map
//! these variants at their boundary, so a caller's error category does not
//! change when its implementation moves here.

use std::fmt;

/// An error from a `lattice-lora` operation.
#[derive(Debug)]
#[non_exhaustive]
pub enum LoraError {
    /// Reading or writing adapter bytes failed.
    Io(std::io::Error),
    /// The input is not a well-formed adapter: unreadable container, missing or
    /// malformed tensors, unsupported dtype, or inconsistent factor shapes.
    Format(String),
    /// A value is outside its allowed range, or two inputs disagree.
    Validation(String),
    /// The input exceeds a resource limit or asks for an unsupported mode.
    Unsupported(String),
}

impl fmt::Display for LoraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Format(msg) => write!(f, "format error: {msg}"),
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for LoraError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Format(_) | Self::Validation(_) | Self::Unsupported(_) => None,
        }
    }
}

impl From<std::io::Error> for LoraError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl LoraError {
    /// The message without the category prefix that [`fmt::Display`] adds.
    ///
    /// Wrappers that map into another crate's error type use this so the
    /// message their callers see is unchanged.
    pub fn message(&self) -> String {
        match self {
            Self::Io(e) => e.to_string(),
            Self::Format(msg) | Self::Validation(msg) | Self::Unsupported(msg) => msg.clone(),
        }
    }
}

#[cfg(test)]
mod tests;
