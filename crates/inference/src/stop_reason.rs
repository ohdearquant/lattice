//! Stop-reason facade over `lattice-arch`.
//!
//! [`StopReason`] keeps the path it had before it moved into `lattice-arch` (ADR-103 Amendment 21).
//!
//! **Compatibility path.** The items here are re-exported from `lattice-arch` so that code written
//! against the pre-split `lattice-inference` keeps compiling. These low-level re-exports are not a
//! stable API and will be removed at 1.0; import from `lattice-arch` directly.

#[doc(inline)]
pub use lattice_arch::stop_reason::StopReason;
