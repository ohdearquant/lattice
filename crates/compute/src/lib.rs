//! Host-side compute for Lattice inference.
//!
//! This crate holds the computation that runs on the host for a decoder forward pass. It currently
//! contains rotary position embedding tables ([`rope::RopeTable`]) and flat KV storage/prefix planning; the
//! CPU kernels, attention, pooling and sampling move here in later releases.
//!
//! `lattice-inference` re-exports the supported items at their original paths, so
//! `lattice_inference::rope::RopeTable` and `lattice_compute::rope::RopeTable` name the same type.

#![allow(clippy::needless_range_loop)]

pub mod kv_cache;
pub mod rope;
