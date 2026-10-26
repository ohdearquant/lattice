//! Stop-reason facade over `lattice-arch`.
//!
//! [`StopReason`] keeps the path it had before it moved into `lattice-arch` (ADR-103 Amendment 21).

#[doc(inline)]
pub use lattice_arch::stop_reason::StopReason;
