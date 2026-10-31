//! Generic digest reports captured from bytes consumed by successful loaders.

use sha2::{Digest, Sha256};

/// Amount of weight content requested by a load report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightDigestMode {
    /// Omits tensor data: header-preserving weight-value edits are undetected.
    Header,
    /// Includes the complete consumed weight file.
    #[default]
    Full,
}

/// An identity digest request, independent of ordinary parsing and tensor reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DigestRequest {
    /// Stable name supplied by the loader, never an ambient pathname.
    pub logical_identifier: String,
    /// Input requested by the SHA-256 path.
    pub kind: DigestKind,
    /// Number of bytes requested from the consumed input.
    pub bytes: u64,
}

/// Generic digest input category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DigestKind {
    /// Exact consumed auxiliary-file bytes.
    ExactBytes,
    /// Framed header recipe, including a content digest in Full mode.
    SafetensorsRecipe,
    /// Complete consumed safetensors file, including its length prefix and header.
    SafetensorsContent,
}

fn hash(
    bytes: &[u8],
    name: &str,
    kind: DigestKind,
    observe: &mut dyn FnMut(DigestRequest),
) -> [u8; 32] {
    observe(DigestRequest {
        logical_identifier: name.into(),
        kind,
        bytes: bytes.len() as u64,
    });
    Sha256::digest(bytes).into()
}

/// SHA-256 and size of exact consumed auxiliary bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumedBytes {
    size: u64,
    digest: [u8; 32],
}

impl ConsumedBytes {
    /// Hash bytes already read by the parser; this does not reopen any file.
    pub fn capture(
        bytes: &[u8],
        logical_identifier: &str,
        observe: &mut dyn FnMut(DigestRequest),
    ) -> Self {
        Self {
            size: bytes.len() as u64,
            digest: hash(bytes, logical_identifier, DigestKind::ExactBytes, observe),
        }
    }
    /// Complete consumed input length.
    pub fn size(&self) -> u64 {
        self.size
    }
    /// SHA-256 of the complete consumed input.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Evidence retained from the same backing that passed safetensors parsing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumedSafetensors {
    mode: WeightDigestMode,
    header: Vec<u8>,
    file_size: u64,
    content_digest: Option<[u8; 32]>,
    recipe_digest: [u8; 32],
}

impl ConsumedSafetensors {
    pub(crate) fn capture(
        bytes: &[u8],
        data_offset: usize,
        mode: WeightDigestMode,
        logical_identifier: &str,
        observe: &mut dyn FnMut(DigestRequest),
    ) -> Self {
        let header = bytes[8..data_offset].to_vec();
        let content_digest = match mode {
            WeightDigestMode::Header => None,
            WeightDigestMode::Full => Some(hash(
                bytes,
                logical_identifier,
                DigestKind::SafetensorsContent,
                observe,
            )),
        };
        let mut recipe = ((data_offset - 8) as u64).to_le_bytes().to_vec();
        recipe.extend_from_slice(&header);
        if let Some(digest) = content_digest {
            recipe.extend_from_slice(&digest);
        }
        let recipe_digest = hash(
            &recipe,
            logical_identifier,
            DigestKind::SafetensorsRecipe,
            observe,
        );
        Self {
            mode,
            header,
            file_size: bytes.len() as u64,
            content_digest,
            recipe_digest,
        }
    }
    /// Selected content-digest mode.
    pub fn mode(&self) -> WeightDigestMode {
        self.mode
    }
    /// Exact length decoded from the consumed safetensors prefix.
    pub fn header_len(&self) -> u64 {
        self.header.len() as u64
    }
    /// Complete raw consumed header span, including whitespace.
    pub fn header_bytes(&self) -> &[u8] {
        &self.header
    }
    /// Complete consumed backing length.
    pub fn file_size(&self) -> u64 {
        self.file_size
    }
    /// Complete-file SHA-256, requested only in Full mode.
    pub fn content_digest(&self) -> Option<[u8; 32]> {
        self.content_digest
    }
    /// SHA-256 of framed raw header and, in Full mode, complete-file digest.
    pub fn recipe_digest(&self) -> [u8; 32] {
        self.recipe_digest
    }
}

#[cfg(test)]
mod tests;
