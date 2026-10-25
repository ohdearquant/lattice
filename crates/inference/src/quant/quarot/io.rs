//! Reading source tensors for QuaRot conversion and the online-rotation artifact descriptor, re-exported from
//! `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.

#[doc(inline)]
pub use lattice_quant::quarot::io::{
    ArtifactVersion, OnlineArtifactDescriptor, QuarotTensorReader, SourceDType,
};
