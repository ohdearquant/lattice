//! Canonical declared embedding recipes. Known equality does not prove numerical equivalence.

use std::fmt;
use std::str::FromStr;

/// Weight evidence selected for an embedding recipe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IdentityMode {
    /// Header-preserving weight-value edits are undetected in this explicit opt-in mode.
    Header,
    /// Includes complete consumed weight-file content digests.
    #[default]
    Full,
}

impl IdentityMode {
    /// Canonical v1 mode tag; the lower tag does not select the default.
    pub const fn tag(self) -> u8 {
        match self {
            Self::Header => 0,
            Self::Full => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Header => "header",
            Self::Full => "full",
        }
    }
}

/// Artifact role in the canonical stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ArtifactRole {
    /// Exact consumed configuration bytes.
    Config,
    /// Exact consumed tokenizer input bytes.
    Tokenizer,
    /// A consumed safetensors file.
    Safetensors,
    /// Exact parsed sharded safetensors index bytes.
    SafetensorsIndex,
}

impl ArtifactRole {
    fn tag(self) -> u8 {
        match self {
            Self::Config => 0,
            Self::Tokenizer => 1,
            Self::Safetensors => 2,
            Self::SafetensorsIndex => 3,
        }
    }
}

/// One artifact's load-bound evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Artifact {
    /// Stable logical role, independent of ambient filesystem paths.
    pub logical_identifier: String,
    /// Artifact role.
    pub role: ArtifactRole,
    /// Size of the complete consumed file.
    pub file_size: u64,
    /// SHA-256 of the artifact recipe defined by the selected mode.
    pub recipe_digest: [u8; 32],
}

/// A typed effective policy value. Finite float bit distinctions are preserved.
#[derive(Clone, Debug, PartialEq)]
pub enum PolicyValue {
    /// Boolean.
    Boolean(bool),
    /// Unsigned integer.
    Unsigned(u64),
    /// Signed integer.
    Signed(i64),
    /// UTF-8 string.
    String(String),
    /// Opaque bytes.
    Bytes(Vec<u8>),
    /// Finite IEEE binary32.
    F32(f32),
    /// Finite IEEE binary64.
    F64(f64),
}

/// One named effective output setting.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyEntry {
    /// Case-sensitive UTF-8 key.
    pub key: String,
    /// Resolved value, after defaults and caps.
    pub value: PolicyValue,
}

/// A v1 declared recipe. Hash [`Self::canonical_bytes`] with SHA-256 in the consumer.
#[derive(Clone, Debug)]
pub struct EmbeddingSpaceDescriptor {
    /// Evidence mode; Full is the default selection.
    pub mode: IdentityMode,
    /// Explicit implementation output revision, independent of package versions.
    pub output_revision: String,
    /// Producer family and effective-policy schema.
    pub producer_family: String,
    /// Consumed artifacts; canonicalization sorts and rejects duplicate role/name pairs.
    pub artifacts: Vec<Artifact>,
    /// Effective policy; canonicalization sorts and rejects duplicate keys.
    pub policy: Vec<PolicyEntry>,
}

/// Invalid canonical recipe or unsupported known-ID text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityError {
    /// A framed length or sequence count exceeds u32.
    LengthOverflow,
    /// Duplicate artifact role/name pair.
    DuplicateArtifact,
    /// Duplicate policy key.
    DuplicatePolicy,
    /// Nonfinite floating-point policy value.
    NonfinitePolicy,
    /// Unsupported or malformed known-ID representation.
    InvalidKnownId,
}

impl fmt::Display for IdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::LengthOverflow => "embedding identity length exceeds u32",
            Self::DuplicateArtifact => "duplicate embedding artifact role and identifier",
            Self::DuplicatePolicy => "duplicate embedding policy key",
            Self::NonfinitePolicy => "nonfinite embedding policy value",
            Self::InvalidKnownId => "unsupported or malformed embedding identity",
        })
    }
}
impl std::error::Error for IdentityError {}

fn length(value: usize) -> Result<[u8; 4], IdentityError> {
    u32::try_from(value)
        .map(u32::to_le_bytes)
        .map_err(|_| IdentityError::LengthOverflow)
}

fn frame(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), IdentityError> {
    out.extend_from_slice(&length(bytes.len())?);
    out.extend_from_slice(bytes);
    Ok(())
}

impl EmbeddingSpaceDescriptor {
    /// Encode the single tagged, length-framed v1 stream, with no hashing dependency.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, IdentityError> {
        let mut artifacts: Vec<_> = self.artifacts.iter().collect();
        artifacts.sort_by(|a, b| {
            (a.role.tag(), a.logical_identifier.as_bytes())
                .cmp(&(b.role.tag(), b.logical_identifier.as_bytes()))
        });
        if artifacts.windows(2).any(|pair| {
            pair[0].role == pair[1].role && pair[0].logical_identifier == pair[1].logical_identifier
        }) {
            return Err(IdentityError::DuplicateArtifact);
        }
        let mut policy: Vec<_> = self.policy.iter().collect();
        policy.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));
        if policy.windows(2).any(|pair| pair[0].key == pair[1].key) {
            return Err(IdentityError::DuplicatePolicy);
        }
        let mut out = b"LATTICE_EMBED_SPACE\0".to_vec();
        out.extend_from_slice(&1u16.to_le_bytes());
        out.push(self.mode.tag());
        frame(&mut out, self.output_revision.as_bytes())?;
        frame(&mut out, self.producer_family.as_bytes())?;
        out.extend_from_slice(&length(artifacts.len())?);
        for artifact in artifacts {
            out.push(artifact.role.tag());
            frame(&mut out, artifact.logical_identifier.as_bytes())?;
            out.extend_from_slice(&artifact.file_size.to_le_bytes());
            out.extend_from_slice(&artifact.recipe_digest);
        }
        out.extend_from_slice(&length(policy.len())?);
        for entry in policy {
            frame(&mut out, entry.key.as_bytes())?;
            match &entry.value {
                PolicyValue::Boolean(value) => {
                    out.extend_from_slice(&[0, u8::from(*value)]);
                }
                PolicyValue::Unsigned(value) => {
                    out.push(1);
                    out.extend_from_slice(&value.to_le_bytes());
                }
                PolicyValue::Signed(value) => {
                    out.push(2);
                    out.extend_from_slice(&value.to_le_bytes());
                }
                PolicyValue::String(value) => {
                    out.push(3);
                    frame(&mut out, value.as_bytes())?;
                }
                PolicyValue::Bytes(value) => {
                    out.push(4);
                    frame(&mut out, value)?;
                }
                PolicyValue::F32(value) => {
                    if !value.is_finite() {
                        return Err(IdentityError::NonfinitePolicy);
                    }
                    out.push(5);
                    out.extend_from_slice(&value.to_bits().to_le_bytes());
                }
                PolicyValue::F64(value) => {
                    if !value.is_finite() {
                        return Err(IdentityError::NonfinitePolicy);
                    }
                    out.push(6);
                    out.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
        }
        Ok(out)
    }
}

/// A supported known v1 identity. Mode namespaces remain distinct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KnownEmbeddingSpaceId {
    mode: IdentityMode,
    digest: [u8; 32],
}

impl KnownEmbeddingSpaceId {
    /// Construct from SHA-256 of the canonical stream in the declared mode.
    pub const fn from_digest(mode: IdentityMode, digest: [u8; 32]) -> Self {
        Self { mode, digest }
    }
    /// Evidence mode.
    pub const fn mode(self) -> IdentityMode {
        self.mode
    }
    /// SHA-256 of the canonical stream.
    pub const fn digest(self) -> [u8; 32] {
        self.digest
    }
}

impl fmt::Display for KnownEmbeddingSpaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lattice-embedding-space:v1:{}:sha256:", self.mode.name())?;
        for byte in self.digest {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for KnownEmbeddingSpaceId {
    type Err = IdentityError;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let rest = text
            .strip_prefix("lattice-embedding-space:v1:")
            .ok_or(IdentityError::InvalidKnownId)?;
        let (mode, hex) = if let Some(hex) = rest.strip_prefix("full:sha256:") {
            (IdentityMode::Full, hex)
        } else if let Some(hex) = rest.strip_prefix("header:sha256:") {
            (IdentityMode::Header, hex)
        } else {
            return Err(IdentityError::InvalidKnownId);
        };
        if hex.len() != 64 {
            return Err(IdentityError::InvalidKnownId);
        }
        let mut digest = [0; 32];
        for (index, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
            let nibble = |byte| match byte {
                b'0'..=b'9' => Ok(byte - b'0'),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                _ => Err(IdentityError::InvalidKnownId),
            };
            digest[index] = (nibble(pair[0])? << 4) | nibble(pair[1])?;
        }
        Ok(Self { mode, digest })
    }
}

/// Availability is separate from the known-ID namespace.
#[derive(Clone, Debug)]
pub enum EmbeddingSpaceIdentity {
    /// Authoritative consumed inputs and effective policy were declared.
    Known(KnownEmbeddingSpaceId),
    /// No authoritative recipe is available; two such values never establish a match.
    Unavailable,
}

impl EmbeddingSpaceIdentity {
    /// True only for equal known recipes and policies; this is no numerical proof.
    pub fn known_match(&self, other: &Self) -> bool {
        matches!((self, other), (Self::Known(a), Self::Known(b)) if a == b)
    }
}

#[cfg(test)]
mod tests;
