//! Test-only forced-dispatch controls for the staged backward SIMD workstream.

use std::cell::Cell;

mod composite;
mod scalar;

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

/// Adapter B gradients, adapter A gradients, and the input-gradient contribution.
pub type LoraGradients = (Vec<f32>, Vec<f32>, Vec<f32>);

/// Backward operation whose complete execution a probe attests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// Completion-masked cross entropy gradient.
    CrossEntropyBackward,
    /// Frozen linear projection input gradient.
    LinearVjp,
    /// Adapter weight and input gradients.
    LoraVjp,
    /// RMS normalization input gradient.
    RmsnormBackward,
    /// Rotary position input gradient.
    RopeBackward,
    /// SwiGLU input and down-projection gradients.
    SwigluBackward,
    /// Grouped-query attention input and adapter gradients.
    GqaBackward,
    /// Gated delta recurrence input and adapter gradients.
    GdnBackward,
}

impl Operation {
    /// Stable operation name used by benchmark execution receipts.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CrossEntropyBackward => "cross_entropy_backward",
            Self::LinearVjp => "linear_vjp",
            Self::LoraVjp => "lora_vjp",
            Self::RmsnormBackward => "rmsnorm_backward",
            Self::RopeBackward => "rope_backward",
            Self::SwigluBackward => "swiglu_backward",
            Self::GqaBackward => "gqa_backward",
            Self::GdnBackward => "gdn_backward",
        }
    }
}

/// Validation or completion failure from a forced arithmetic entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedExecutionError {
    /// The probe could not admit or attest the requested dispatch.
    Dispatch(ForcedDispatchError),
    /// A shape or range did not satisfy the operation's complete contract.
    Shape {
        /// Operation whose inputs were rejected.
        operation: Operation,
        /// Static description of the invalid shape or range.
        problem: &'static str,
    },
    /// Completion evidence belongs to a different operation.
    OperationMismatch {
        /// Operation required by the caller or admitted invocation.
        expected: Operation,
        /// Operation actually recorded by the implementation.
        executed: Operation,
    },
}

impl From<ForcedDispatchError> for ForcedExecutionError {
    fn from(error: ForcedDispatchError) -> Self {
        Self::Dispatch(error)
    }
}

#[derive(Debug, Clone, Copy)]
enum State {
    Fresh,
    Running,
    Completed(Operation),
    Failed,
}

/// Single-use, invocation-local selection and implementation-owned evidence.
///
/// Call an arithmetic method, then verify its operation. Scalar means the named
/// portable reference loops, bypassing explicit SIMD and BLAS dispatch; it does
/// not disable compiler auto-vectorization. This does not force a training tape
/// or any forward computation used to construct cached activations.
#[derive(Debug)]
pub struct ForcedDispatch {
    requested: Backend,
    state: Cell<State>,
}

struct Completion<'a> {
    dispatch: &'a ForcedDispatch,
    expected: Operation,
    finished: bool,
}

impl Completion<'_> {
    fn finish(mut self, actual: Operation) -> Result<(), ForcedExecutionError> {
        if actual != self.expected {
            return Err(ForcedExecutionError::OperationMismatch {
                expected: self.expected,
                executed: actual,
            });
        }
        if !matches!(self.dispatch.state.get(), State::Running) {
            return Err(self.dispatch.missing().into());
        }
        self.dispatch.state.set(State::Completed(actual));
        self.finished = true;
        Ok(())
    }
}

impl Drop for Completion<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.dispatch.state.set(State::Failed);
        }
    }
}

/// Force the portable scalar reference path for one backward invocation.
pub fn force_scalar() -> ForcedDispatch {
    ForcedDispatch {
        requested: Backend::Scalar,
        state: Cell::new(State::Fresh),
    }
}

/// Force one ADR-083 backend or fail if its implementation is unavailable.
///
/// Hardware capability alone does not admit a backend: no accelerated backward
/// implementation has landed, so NEON and AVX2+FMA currently refuse execution.
pub fn force_backend(backend: Backend) -> Result<ForcedDispatch, ForcedDispatchError> {
    match backend {
        Backend::Scalar => Ok(force_scalar()),
        Backend::Neon | Backend::Avx2Fma => {
            Err(ForcedDispatchError::BackendUnavailable { requested: backend })
        }
    }
}

impl ForcedDispatch {
    fn missing(&self) -> ForcedDispatchError {
        ForcedDispatchError::MarkerMissing {
            requested: self.requested,
        }
    }

    fn begin(&self, expected: Operation) -> Result<Completion<'_>, ForcedExecutionError> {
        match self.state.replace(State::Failed) {
            State::Fresh => {
                self.state.set(State::Running);
                Ok(Completion {
                    dispatch: self,
                    expected,
                    finished: false,
                })
            }
            State::Completed(_) => Err(ForcedDispatchError::MarkerAlreadySet {
                first: self.requested,
                second: self.requested,
            }
            .into()),
            State::Running | State::Failed => Err(self.missing().into()),
        }
    }

    /// Refuse caller-written evidence and invalidate the probe.
    ///
    /// Retained for source compatibility. Only successful arithmetic methods can
    /// record completion; a caller-supplied backend label is never evidence.
    #[deprecated(note = "call a forced arithmetic method and verify_for its operation")]
    pub fn mark_executed(&self, backend: Backend) -> Result<(), ForcedDispatchError> {
        self.state.set(State::Failed);
        if backend != self.requested {
            Err(ForcedDispatchError::MarkerMismatch {
                requested: self.requested,
                executed: backend,
            })
        } else {
            Err(self.missing())
        }
    }

    /// Verify that the selected backend completed one complete operation.
    pub fn verify(&self) -> Result<Backend, ForcedDispatchError> {
        match self.state.get() {
            State::Completed(_) => Ok(self.requested),
            _ => Err(self.missing()),
        }
    }

    /// Verify completion of this exact operation; a mismatch poisons the probe.
    pub fn verify_for(&self, expected: Operation) -> Result<Backend, ForcedExecutionError> {
        match self.state.get() {
            State::Completed(executed) if executed == expected => Ok(self.requested),
            State::Completed(executed) => {
                self.state.set(State::Failed);
                Err(ForcedExecutionError::OperationMismatch { expected, executed })
            }
            _ => Err(self.missing().into()),
        }
    }

    /// Execute `cross_entropy_backward` through the portable scalar implementation.
    pub fn cross_entropy_backward(
        &self,
        logits: &[f32],
        targets: &[u32],
        seq_len: usize,
        vocab_size: usize,
        completion_start: usize,
    ) -> Result<Vec<f32>, ForcedExecutionError> {
        scalar::cross_entropy_backward(
            self.begin(Operation::CrossEntropyBackward)?,
            logits,
            targets,
            seq_len,
            vocab_size,
            completion_start,
        )
    }

    /// Execute `cross_entropy_backward` into caller buffers after validating every shape.
    pub fn cross_entropy_backward_into(
        &self,
        logits: &[f32],
        targets: &[u32],
        seq_len: usize,
        vocab_size: usize,
        completion_start: usize,
        dlogits: &mut [f32],
    ) -> Result<(), ForcedExecutionError> {
        scalar::cross_entropy_backward_into(
            self.begin(Operation::CrossEntropyBackward)?,
            logits,
            targets,
            seq_len,
            vocab_size,
            completion_start,
            dlogits,
        )
    }

    /// Execute `linear_vjp` through the portable scalar implementation.
    pub fn linear_vjp(
        &self,
        w: &[f32],
        g: &[f32],
        d_in: usize,
        d_out: usize,
    ) -> Result<Vec<f32>, ForcedExecutionError> {
        scalar::linear_vjp(self.begin(Operation::LinearVjp)?, w, g, d_in, d_out)
    }

    /// Execute `linear_vjp` into caller buffers after validating every shape.
    pub fn linear_vjp_into(
        &self,
        w: &[f32],
        g: &[f32],
        d_in: usize,
        d_out: usize,
        dx: &mut [f32],
    ) -> Result<(), ForcedExecutionError> {
        scalar::linear_vjp_into(self.begin(Operation::LinearVjp)?, w, g, d_in, d_out, dx)
    }

    /// Execute `lora_vjp` through the portable scalar implementation.
    pub fn lora_vjp(
        &self,
        g: &[f32],
        x: &[f32],
        h: &[f32],
        a: &[f32],
        b: &[f32],
        rank: usize,
        d_in: usize,
        d_out: usize,
        scale: f32,
    ) -> Result<LoraGradients, ForcedExecutionError> {
        scalar::lora_vjp(
            self.begin(Operation::LoraVjp)?,
            g,
            x,
            h,
            a,
            b,
            rank,
            d_in,
            d_out,
            scale,
        )
    }

    /// Execute `lora_vjp` into caller buffers after validating every shape.
    pub fn lora_vjp_into(
        &self,
        g: &[f32],
        x: &[f32],
        h: &[f32],
        a: &[f32],
        b: &[f32],
        rank: usize,
        d_in: usize,
        d_out: usize,
        scale: f32,
        grad_b: &mut [f32],
        grad_a: &mut [f32],
        dx: &mut [f32],
        bt_g: &mut [f32],
    ) -> Result<(), ForcedExecutionError> {
        scalar::lora_vjp_into(
            self.begin(Operation::LoraVjp)?,
            g,
            x,
            h,
            a,
            b,
            rank,
            d_in,
            d_out,
            scale,
            grad_b,
            grad_a,
            dx,
            bt_g,
        )
    }

    /// Execute `rmsnorm_backward` through the portable scalar implementation.
    pub fn rmsnorm_backward(
        &self,
        x: &[f32],
        w: &[f32],
        inv_rms: f32,
        g: &[f32],
    ) -> Result<Vec<f32>, ForcedExecutionError> {
        scalar::rmsnorm_backward(self.begin(Operation::RmsnormBackward)?, x, w, inv_rms, g)
    }

    /// Execute `rmsnorm_backward` into caller buffers after validating every shape.
    pub fn rmsnorm_backward_into(
        &self,
        x: &[f32],
        w: &[f32],
        inv_rms: f32,
        g: &[f32],
        dx: &mut [f32],
    ) -> Result<(), ForcedExecutionError> {
        scalar::rmsnorm_backward_into(
            self.begin(Operation::RmsnormBackward)?,
            x,
            w,
            inv_rms,
            g,
            dx,
        )
    }

    /// Execute `rope_backward` through the portable scalar implementation.
    pub fn rope_backward(
        &self,
        g: &[f32],
        cos_vals: &[f32],
        sin_vals: &[f32],
        rope_dim: usize,
    ) -> Result<Vec<f32>, ForcedExecutionError> {
        scalar::rope_backward(
            self.begin(Operation::RopeBackward)?,
            g,
            cos_vals,
            sin_vals,
            rope_dim,
        )
    }

    /// Execute `rope_backward` into caller buffers after validating every shape.
    pub fn rope_backward_into(
        &self,
        g: &[f32],
        cos_vals: &[f32],
        sin_vals: &[f32],
        rope_dim: usize,
        dx: &mut [f32],
    ) -> Result<(), ForcedExecutionError> {
        scalar::rope_backward_into(
            self.begin(Operation::RopeBackward)?,
            g,
            cos_vals,
            sin_vals,
            rope_dim,
            dx,
        )
    }

    /// Execute `swiglu_backward` through the portable scalar implementation.
    pub fn swiglu_backward(
        &self,
        dy: &[f32],
        gate_pre: &[f32],
        up_pre: &[f32],
        w_down: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        hidden: usize,
        inter: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), ForcedExecutionError> {
        scalar::swiglu_backward(
            self.begin(Operation::SwigluBackward)?,
            dy,
            gate_pre,
            up_pre,
            w_down,
            w_gate,
            w_up,
            hidden,
            inter,
        )
    }

    /// Execute `swiglu_backward` into caller buffers after validating every shape.
    pub fn swiglu_backward_into(
        &self,
        dy: &[f32],
        gate_pre: &[f32],
        up_pre: &[f32],
        w_down: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        hidden: usize,
        inter: usize,
        dx: &mut [f32],
        dm: &mut [f32],
        d_gate: &mut [f32],
    ) -> Result<(), ForcedExecutionError> {
        scalar::swiglu_backward_into(
            self.begin(Operation::SwigluBackward)?,
            dy,
            gate_pre,
            up_pre,
            w_down,
            w_gate,
            w_up,
            hidden,
            inter,
            dx,
            dm,
            d_gate,
        )
    }

    /// Execute `gqa_backward` through the portable scalar implementation.
    pub fn gqa_backward(
        &self,
        dy_out: &[f32],
        cache: &crate::backward::attention_gqa::AttnCache,
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        q_norm_w: &[f32],
        k_norm_w: &[f32],
        lora_a_q: Option<&[f32]>,
        lora_b_q: Option<&[f32]>,
        lora_a_v: Option<&[f32]>,
        lora_b_v: Option<&[f32]>,
        lora_rank: usize,
        lora_scale: f32,
    ) -> Result<crate::backward::attention_gqa::AttnGrads, ForcedExecutionError> {
        composite::gqa_backward(
            self.begin(Operation::GqaBackward)?,
            dy_out,
            cache,
            w_q,
            w_k,
            w_v,
            w_o,
            q_norm_w,
            k_norm_w,
            lora_a_q,
            lora_b_q,
            lora_a_v,
            lora_b_v,
            lora_rank,
            lora_scale,
        )
    }

    /// Execute `gdn_backward` through the portable scalar implementation.
    pub fn gdn_backward(
        &self,
        grad_outputs: &[f32],
        saved: &crate::attention::gdn_backward::GdnSaved,
        weights: &crate::attention::gdn::GatedDeltaNetWeights,
    ) -> Result<crate::attention::gdn_backward::GdnGrads, ForcedExecutionError> {
        composite::gdn_backward(
            self.begin(Operation::GdnBackward)?,
            grad_outputs,
            saved,
            weights,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::cpu::observe_matmul_calls;

    #[test]
    fn scalar_execution_records_operation_and_bypasses_real_gemm() {
        let w = [0.5, -0.25, 1.0, 0.75, -0.5, 0.125];
        let g = [0.3, -0.8];
        let (ordinary, ordinary_calls) =
            observe_matmul_calls(|| crate::backward::ops::linear_vjp(&w, &g, 3, 2));
        assert_eq!(ordinary_calls, 1);
        let dispatch = force_scalar();
        let (forced, forced_calls) =
            observe_matmul_calls(|| dispatch.linear_vjp(&w, &g, 3, 2).unwrap());
        assert_eq!(forced_calls, 0);
        assert_eq!(forced.len(), ordinary.len());
        assert!(forced.iter().any(|v| v.abs() > 0.1));
        for (&actual, &expected) in forced.iter().zip(&ordinary) {
            assert!(actual.is_finite() && expected.is_finite());
            assert!((actual - expected).abs() <= 1e-6 * (1.0 + expected.abs()));
        }
        assert_eq!(
            dispatch.verify_for(Operation::LinearVjp),
            Ok(Backend::Scalar)
        );
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
    #[allow(deprecated)]
    fn manual_labels_never_attest_arithmetic_and_poison_reuse() {
        for backend in [Backend::Scalar, Backend::Neon, Backend::Avx2Fma] {
            let dispatch = force_scalar();
            assert!(dispatch.mark_executed(backend).is_err());
            assert!(dispatch.verify().is_err());
            assert!(dispatch.linear_vjp(&[2.0], &[3.0], 1, 1).is_err());
            assert!(dispatch.verify_for(Operation::LinearVjp).is_err());
        }
    }

    #[test]
    fn completion_for_another_operation_invalidates_evidence() {
        let dispatch = force_scalar();
        assert_eq!(dispatch.linear_vjp(&[2.0], &[3.0], 1, 1).unwrap(), [6.0]);
        assert_eq!(
            dispatch.verify_for(Operation::LoraVjp),
            Err(ForcedExecutionError::OperationMismatch {
                expected: Operation::LoraVjp,
                executed: Operation::LinearVjp,
            })
        );
        assert!(dispatch.verify().is_err());
        assert!(dispatch.verify_for(Operation::LinearVjp).is_err());
    }

    #[test]
    fn duplicate_execution_invalidates_prior_success_without_writing() {
        let dispatch = force_scalar();
        assert_eq!(dispatch.linear_vjp(&[2.0], &[3.0], 1, 1).unwrap(), [6.0]);
        let mut out = [91.0];
        assert_eq!(
            dispatch.linear_vjp_into(&[4.0], &[5.0], 1, 1, &mut out),
            Err(ForcedExecutionError::Dispatch(
                ForcedDispatchError::MarkerAlreadySet {
                    first: Backend::Scalar,
                    second: Backend::Scalar,
                }
            ))
        );
        assert_eq!(out, [91.0]);
        assert!(dispatch.verify().is_err());
    }

    #[test]
    fn missing_completion_and_unwind_never_produce_success() {
        let missing = force_scalar();
        assert!(missing.verify().is_err());
        {
            let _unfinished = missing.begin(Operation::LinearVjp).unwrap();
        }
        assert!(missing.verify().is_err());
        assert!(missing.linear_vjp(&[2.0], &[3.0], 1, 1).is_err());

        let unwound = force_scalar();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _unfinished = unwound.begin(Operation::LinearVjp).unwrap();
                panic!("unfinished scalar invocation");
            }))
            .is_err()
        );
        assert!(unwound.verify().is_err());
        assert!(unwound.linear_vjp(&[2.0], &[3.0], 1, 1).is_err());
    }

    #[test]
    fn wrong_implementation_completion_and_reentrant_use_fail_closed() {
        let wrong = force_scalar();
        assert_eq!(
            wrong
                .begin(Operation::LinearVjp)
                .unwrap()
                .finish(Operation::LoraVjp),
            Err(ForcedExecutionError::OperationMismatch {
                expected: Operation::LinearVjp,
                executed: Operation::LoraVjp,
            })
        );
        assert!(wrong.verify().is_err());

        let reentrant = force_scalar();
        let completion = reentrant.begin(Operation::LinearVjp).unwrap();
        assert!(reentrant.linear_vjp(&[2.0], &[3.0], 1, 1).is_err());
        assert!(completion.finish(Operation::LinearVjp).is_err());
        assert!(reentrant.verify().is_err());
    }

    #[test]
    fn invalid_shape_and_overflow_poison_probe_and_preserve_output() {
        for (w, d_in, d_out) in [(&[2.0][..], 2, 1), (&[][..], usize::MAX, 2)] {
            let dispatch = force_scalar();
            let mut out = [91.0];
            assert!(matches!(
                dispatch.linear_vjp_into(w, &[3.0], d_in, d_out, &mut out),
                Err(ForcedExecutionError::Shape {
                    operation: Operation::LinearVjp,
                    ..
                })
            ));
            assert_eq!(out, [91.0]);
            assert!(dispatch.verify().is_err());
            assert!(dispatch.linear_vjp(&[2.0], &[3.0], 1, 1).is_err());
        }
    }

    #[test]
    fn probes_and_matmul_observation_are_isolated_between_threads() {
        let threads: Vec<_> = (1..=4)
            .map(|value| {
                std::thread::spawn(move || {
                    let dispatch = force_scalar();
                    let (out, calls) = observe_matmul_calls(|| {
                        dispatch.linear_vjp(&[value as f32], &[3.0], 1, 1).unwrap()
                    });
                    assert_eq!(out, [value as f32 * 3.0]);
                    assert_eq!(calls, 0);
                    assert_eq!(
                        dispatch.verify_for(Operation::LinearVjp),
                        Ok(Backend::Scalar)
                    );
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    }

    #[test]
    fn nested_observation_preserves_real_gemm_counts_including_unwind() {
        let ((_, inner), outer) = observe_matmul_calls(|| {
            observe_matmul_calls(|| crate::backward::ops::linear_vjp(&[2.0], &[3.0], 1, 1))
        });
        assert_eq!((inner, outer), (1, 1));
        let (_, outer) = observe_matmul_calls(|| {
            assert!(
                std::panic::catch_unwind(|| {
                    observe_matmul_calls(|| {
                        crate::backward::ops::linear_vjp(&[2.0], &[3.0], 1, 1);
                        panic!("nested observer unwinds");
                    });
                })
                .is_err()
            );
        });
        assert_eq!(outer, 1);
    }

    #[test]
    fn observation_restores_after_panicking_work() {
        assert!(
            std::panic::catch_unwind(|| observe_matmul_calls(|| {
                panic!("observer unwinds with operation");
            }))
            .is_err()
        );
        let (_, calls) =
            observe_matmul_calls(|| crate::backward::ops::linear_vjp(&[2.0], &[3.0], 1, 1));
        assert_eq!(calls, 1);
    }
}
