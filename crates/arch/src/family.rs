//! Model families (ADR-103 Amendment 21 item 3): static descriptors that name a family, the
//! `config.json` `model_type` values it claims and the loaders it offers, and a consumer-owned
//! [`Registry`] that validates a table of them and resolves a family by exact name or
//! `model_type`.
//!
//! **Unstable**: this module is the supported family contract. A family crate (today the
//! `lattice-inference` facade) defines its descriptors as statics; a consumer builds a registry
//! from the descriptors it was compiled with. This crate names no family and keeps no list of
//! them, so "unavailable" means absent from the consumer's table, never unknown to Lattice.
//!
//! Resolution does no I/O and never falls back to another family: a miss, a family that does not
//! offer the requested kind, and a backend the family does not offer are distinct refusals.
//! Loading reads the checkpoint the [`LoadRequest`] names, on the backend it fixes.

use crate::decoder::DecoderModel;
use crate::encoder::Encoder;
use crate::error::InferenceError;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The execution backend a load fixes for the loaded model's life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// CPU execution.
    Cpu,
    /// Metal execution on Apple silicon.
    Metal,
}

impl Backend {
    /// The backend's lowercase name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What a loader loads: a checkpoint directory and the backend to run it on.
///
/// Owned, so a consumer can move it with a loader to the worker that loads and keeps the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadRequest {
    model_dir: PathBuf,
    backend: Backend,
}

impl LoadRequest {
    /// Loads the checkpoint in `model_dir` on the CPU.
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            backend: Backend::Cpu,
        }
    }

    /// Sets the backend.
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// The checkpoint directory.
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// The backend the loaded model runs on.
    pub const fn backend(&self) -> Backend {
        self.backend
    }
}

/// Loads a decoder. The loaded owner stays on the thread that called the loader.
pub type DecoderLoader = fn(&LoadRequest) -> Result<Box<dyn DecoderModel>, InferenceError>;

/// Loads an encoder, shareable across threads.
pub type EncoderLoader = fn(&LoadRequest) -> Result<Arc<dyn Encoder>, InferenceError>;

/// The kind of model a consumer asks a family for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelKind {
    /// A text decoder: [`DecoderModel`].
    Decoder,
    /// A text encoder: [`Encoder`].
    Encoder,
}

impl ModelKind {
    /// The kind's lowercase name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Decoder => "decoder",
            Self::Encoder => "encoder",
        }
    }
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How a consumer names the family it wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResolveKey<'a> {
    /// A family name, compared exactly.
    Family(&'a str),
    /// A `config.json` `model_type` value the consumer read, compared exactly.
    ModelType(&'a str),
}

const CPU_ONLY: &[Backend] = &[Backend::Cpu];

/// A model family: its name, the `model_type` values it claims, the backends it offers and a
/// decoder loader, an encoder loader or both.
///
/// Built in a `static` with the const constructors, which each set at least one loader. The CPU
/// is the only backend until [`FamilyDescriptor::with_backends`] says otherwise.
#[derive(Clone, Copy)]
pub struct FamilyDescriptor {
    name: &'static str,
    model_types: &'static [&'static str],
    backends: &'static [Backend],
    decoder: Option<DecoderLoader>,
    encoder: Option<EncoderLoader>,
}

impl FamilyDescriptor {
    /// A family that offers a decoder.
    pub const fn decoder(
        name: &'static str,
        model_types: &'static [&'static str],
        loader: DecoderLoader,
    ) -> Self {
        Self {
            name,
            model_types,
            backends: CPU_ONLY,
            decoder: Some(loader),
            encoder: None,
        }
    }

    /// A family that offers an encoder.
    pub const fn encoder(
        name: &'static str,
        model_types: &'static [&'static str],
        loader: EncoderLoader,
    ) -> Self {
        Self {
            name,
            model_types,
            backends: CPU_ONLY,
            decoder: None,
            encoder: Some(loader),
        }
    }

    /// Adds or replaces the decoder loader.
    pub const fn with_decoder(mut self, loader: DecoderLoader) -> Self {
        self.decoder = Some(loader);
        self
    }

    /// Adds or replaces the encoder loader.
    pub const fn with_encoder(mut self, loader: EncoderLoader) -> Self {
        self.encoder = Some(loader);
        self
    }

    /// Sets the backends the loaders accept.
    pub const fn with_backends(mut self, backends: &'static [Backend]) -> Self {
        self.backends = backends;
        self
    }

    /// The family name.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// The `model_type` values the family claims.
    pub const fn model_types(&self) -> &'static [&'static str] {
        self.model_types
    }

    /// The backends the loaders accept.
    pub const fn backends(&self) -> &'static [Backend] {
        self.backends
    }

    /// Whether the family offers a model of `kind`.
    pub const fn offers(&self, kind: ModelKind) -> bool {
        match kind {
            ModelKind::Decoder => self.decoder.is_some(),
            ModelKind::Encoder => self.encoder.is_some(),
        }
    }

    fn claims(&self, model_type: &str) -> bool {
        self.model_types.contains(&model_type)
    }

    fn refuse(&self, what: String) -> InferenceError {
        InferenceError::UnsupportedModel(format!("family {} {what}", self.name))
    }

    fn check_backend(&self, request: &LoadRequest) -> Result<(), InferenceError> {
        if self.backends.contains(&request.backend()) {
            Ok(())
        } else {
            Err(self.refuse(format!("does not offer the {} backend", request.backend())))
        }
    }

    /// Loads the family's decoder.
    ///
    /// # Errors
    ///
    /// [`InferenceError::UnsupportedModel`] when the family offers no decoder or not the
    /// requested backend, before the loader runs; otherwise the loader's error.
    pub fn load_decoder(
        &self,
        request: &LoadRequest,
    ) -> Result<Box<dyn DecoderModel>, InferenceError> {
        let loader = self
            .decoder
            .ok_or_else(|| self.refuse(format!("offers no {}", ModelKind::Decoder)))?;
        self.check_backend(request)?;
        loader(request)
    }

    /// Loads the family's encoder.
    ///
    /// # Errors
    ///
    /// [`InferenceError::UnsupportedModel`] when the family offers no encoder or not the
    /// requested backend, before the loader runs; otherwise the loader's error.
    pub fn load_encoder(&self, request: &LoadRequest) -> Result<Arc<dyn Encoder>, InferenceError> {
        let loader = self
            .encoder
            .ok_or_else(|| self.refuse(format!("offers no {}", ModelKind::Encoder)))?;
        self.check_backend(request)?;
        loader(request)
    }
}

impl fmt::Debug for FamilyDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FamilyDescriptor")
            .field("name", &self.name)
            .field("model_types", &self.model_types)
            .field("backends", &self.backends)
            .field("decoder", &self.decoder.is_some())
            .field("encoder", &self.encoder.is_some())
            .finish()
    }
}

/// Why a registry could not be built or could not resolve a key.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegistryError {
    /// Two entries carry the same family name, or one descriptor was listed twice.
    DuplicateFamily {
        /// The repeated name.
        name: &'static str,
    },
    /// More than one entry claims the same `model_type` for the same kind.
    DuplicateClaim {
        /// The kind both entries offer.
        kind: ModelKind,
        /// The contested `model_type`.
        model_type: &'static str,
        /// The claiming families, sorted.
        families: Vec<&'static str>,
    },
    /// No entry has this family name.
    UnavailableFamily {
        /// The requested name.
        name: String,
    },
    /// No entry claims this `model_type` for any kind.
    UnregisteredModelType {
        /// The requested `model_type`.
        model_type: String,
    },
    /// The key names families that do not offer the requested kind.
    WrongKind {
        /// The requested kind.
        requested: ModelKind,
        /// The families the key resolved to, sorted.
        families: Vec<&'static str>,
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateFamily { name } => write!(f, "family {name} is registered twice"),
            Self::DuplicateClaim {
                kind,
                model_type,
                families,
            } => write!(
                f,
                "model_type {model_type} is claimed for {kind} by more than one family: {}",
                families.join(", ")
            ),
            Self::UnavailableFamily { name } => {
                write!(f, "family {name} is not available in this registry")
            }
            Self::UnregisteredModelType { model_type } => {
                write!(
                    f,
                    "no family in this registry claims model_type {model_type}"
                )
            }
            Self::WrongKind {
                requested,
                families,
            } => write!(f, "{} offers no {requested}", families.join(", ")),
        }
    }
}

impl std::error::Error for RegistryError {}

impl From<RegistryError> for InferenceError {
    fn from(value: RegistryError) -> Self {
        match value {
            RegistryError::DuplicateFamily { .. } | RegistryError::DuplicateClaim { .. } => {
                InferenceError::InvalidInput(value.to_string())
            }
            _ => InferenceError::UnsupportedModel(value.to_string()),
        }
    }
}

/// A validated table of family descriptors, owned by the consumer that built it.
#[derive(Debug, Clone)]
pub struct Registry {
    entries: Vec<&'static FamilyDescriptor>,
}

impl Registry {
    /// Validates `entries` and keeps them in the order given.
    ///
    /// # Errors
    ///
    /// [`RegistryError::DuplicateFamily`] when two entries share a name, including one
    /// descriptor listed twice; [`RegistryError::DuplicateClaim`] when two entries, or one entry
    /// twice, claim a `model_type` for the same kind. The first conflict in sorted order is
    /// reported. Different kinds may claim the same `model_type`.
    pub fn new(entries: &[&'static FamilyDescriptor]) -> Result<Self, RegistryError> {
        let mut names: Vec<&'static str> = entries.iter().map(|d| d.name).collect();
        names.sort_unstable();
        if let Some(pair) = names.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(RegistryError::DuplicateFamily { name: pair[0] });
        }
        for kind in [ModelKind::Decoder, ModelKind::Encoder] {
            let mut claims: Vec<(&'static str, &'static str)> = entries
                .iter()
                .filter(|d| d.offers(kind))
                .flat_map(|d| {
                    d.model_types
                        .iter()
                        .map(move |model_type| (*model_type, d.name))
                })
                .collect();
            claims.sort_unstable();
            let mut start = 0;
            while start < claims.len() {
                let model_type = claims[start].0;
                let end = start
                    + claims[start..]
                        .iter()
                        .take_while(|c| c.0 == model_type)
                        .count();
                if end - start > 1 {
                    return Err(RegistryError::DuplicateClaim {
                        kind,
                        model_type,
                        families: claims[start..end].iter().map(|c| c.1).collect(),
                    });
                }
                start = end;
            }
        }
        Ok(Self {
            entries: entries.to_vec(),
        })
    }

    /// The descriptors, in the order given to [`Registry::new`].
    pub fn entries(&self) -> &[&'static FamilyDescriptor] {
        &self.entries
    }

    /// The family `key` names, if it offers `kind`.
    ///
    /// # Errors
    ///
    /// [`RegistryError::UnavailableFamily`] or [`RegistryError::UnregisteredModelType`] when no
    /// entry matches the key; [`RegistryError::WrongKind`] when the matching entries offer
    /// another kind only.
    pub fn resolve(
        &self,
        key: ResolveKey<'_>,
        kind: ModelKind,
    ) -> Result<&'static FamilyDescriptor, RegistryError> {
        let matches: Vec<&'static FamilyDescriptor> = match key {
            ResolveKey::Family(name) => self
                .entries
                .iter()
                .copied()
                .filter(|d| d.name == name)
                .collect(),
            ResolveKey::ModelType(model_type) => self
                .entries
                .iter()
                .copied()
                .filter(|d| d.claims(model_type))
                .collect(),
        };
        if let Some(found) = matches.iter().copied().find(|d| d.offers(kind)) {
            return Ok(found);
        }
        if matches.is_empty() {
            return Err(match key {
                ResolveKey::Family(name) => RegistryError::UnavailableFamily {
                    name: name.to_owned(),
                },
                ResolveKey::ModelType(model_type) => RegistryError::UnregisteredModelType {
                    model_type: model_type.to_owned(),
                },
            });
        }
        let mut families: Vec<&'static str> = matches.iter().map(|d| d.name).collect();
        families.sort_unstable();
        Err(RegistryError::WrongKind {
            requested: kind,
            families,
        })
    }
}

#[cfg(test)]
mod tests;
