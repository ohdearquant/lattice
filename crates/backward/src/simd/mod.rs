//! Test-only forced-dispatch controls for the staged backward SIMD workstream.

use std::cell::Cell;

/// Backends reserved by ADR-083 for backward CPU dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Portable scalar reference path.
    Scalar,
    /// AArch64 NEON path.
    Neon,
    /// x86-64 AVX2 plus FMA path.
    Avx2Fma,
}

/// Fail-closed errors emitted by a forced dispatch probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedDispatchError {
    /// The requested accelerated backend has not landed yet.
    BackendUnavailable { requested: Backend },
    /// The exercised path returned without recording a backend.
    MarkerMissing { requested: Backend },
    /// The exercised path recorded a backend other than the forced selection.
    MarkerMismatch {
        requested: Backend,
        executed: Backend,
    },
    /// One dispatch invocation attempted to record more than one backend.
    MarkerAlreadySet { first: Backend, second: Backend },
}

/// Per-invocation forced selection and execution marker.
///
/// The probe is deliberately local rather than process-global so parallel tests
/// cannot overwrite each other's evidence.
#[derive(Debug)]
pub struct ForcedDispatch {
    requested: Backend,
    executed: Cell<Option<Backend>>,
}

/// Force the scalar reference path.
pub fn force_scalar() -> ForcedDispatch {
    ForcedDispatch {
        requested: Backend::Scalar,
        executed: Cell::new(None),
    }
}

/// Force one ADR-083 backend or fail when that backend is unavailable.
///
/// Stage 0 has no accelerated kernels, so NEON and AVX2+FMA fail closed. Each
/// later stage can enable its backend here only after its real kernel exists.
pub fn force_backend(backend: Backend) -> Result<ForcedDispatch, ForcedDispatchError> {
    match backend {
        Backend::Scalar => Ok(force_scalar()),
        Backend::Neon | Backend::Avx2Fma => {
            Err(ForcedDispatchError::BackendUnavailable { requested: backend })
        }
    }
}

impl ForcedDispatch {
    /// Record the backend that actually executed.
    pub fn mark_executed(&self, backend: Backend) -> Result<(), ForcedDispatchError> {
        if backend != self.requested {
            return Err(ForcedDispatchError::MarkerMismatch {
                requested: self.requested,
                executed: backend,
            });
        }
        if let Some(first) = self.executed.get() {
            return Err(ForcedDispatchError::MarkerAlreadySet {
                first,
                second: backend,
            });
        }
        self.executed.set(Some(backend));
        Ok(())
    }

    /// Verify that exactly the forced backend recorded execution.
    pub fn verify(&self) -> Result<Backend, ForcedDispatchError> {
        self.executed
            .get()
            .ok_or(ForcedDispatchError::MarkerMissing {
                requested: self.requested,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_sum(values: &[f32], dispatch: &ForcedDispatch) -> Result<f32, ForcedDispatchError> {
        let sum = values.iter().sum();
        dispatch.mark_executed(Backend::Scalar)?;
        Ok(sum)
    }

    #[test]
    fn forced_scalar_records_the_executed_backend() {
        let dispatch = force_scalar();
        let sum = scalar_sum(&[1.0, 2.0, 3.0], &dispatch).unwrap();

        assert_eq!(sum, 6.0);
        assert_eq!(dispatch.verify(), Ok(Backend::Scalar));
    }

    #[test]
    fn unavailable_accelerated_backend_fails_closed() {
        for backend in [Backend::Neon, Backend::Avx2Fma] {
            assert_eq!(
                force_backend(backend).unwrap_err(),
                ForcedDispatchError::BackendUnavailable { requested: backend }
            );
        }
    }

    #[test]
    fn mismatched_marker_is_rejected_without_recording_success() {
        let dispatch = force_scalar();

        assert_eq!(
            dispatch.mark_executed(Backend::Neon),
            Err(ForcedDispatchError::MarkerMismatch {
                requested: Backend::Scalar,
                executed: Backend::Neon,
            })
        );
        assert_eq!(
            dispatch.verify(),
            Err(ForcedDispatchError::MarkerMissing {
                requested: Backend::Scalar,
            })
        );
    }

    #[test]
    fn missing_and_duplicate_markers_fail_closed() {
        let missing = force_backend(Backend::Scalar).unwrap();
        assert_eq!(
            missing.verify(),
            Err(ForcedDispatchError::MarkerMissing {
                requested: Backend::Scalar,
            })
        );

        let duplicate = force_scalar();
        assert_eq!(duplicate.mark_executed(Backend::Scalar), Ok(()));
        assert_eq!(
            duplicate.mark_executed(Backend::Scalar),
            Err(ForcedDispatchError::MarkerAlreadySet {
                first: Backend::Scalar,
                second: Backend::Scalar,
            })
        );
    }
}
