//! Experimental f32 CPU GDN state-health observations.
//!
//! A scoped observer runs synchronously on the installing thread after each
//! f32 CPU `Qwen35Model::forward_step` recurrence and each `prefill_prompt`
//! recurrence. Batched prefill reports in layer-major order; sequential
//! decoding reports in token-major order. Events describe executed updates,
//! including work later rolled back or abandoned. The scope does not follow
//! work to other threads and does not cover debug-only, f16, Q8, NEON or Metal
//! paths. These observations are not a non-finite execution guard.

use crate::attention::gdn::GatedDeltaNetState;
use lattice_arch::qwen35_config::Qwen35Config;
use std::cell::RefCell;
use std::rc::Rc;

/// Summary of one recurrent-state matrix or all of a layer's matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct StateHealth {
    /// Number of finite values.
    pub finite_count: usize,
    /// Number of NaN values.
    pub nan_count: usize,
    /// Number of positive or negative infinities.
    pub infinite_count: usize,
    /// Frobenius norm, or `None` if any value is non-finite.
    pub frobenius_norm: Option<f64>,
}

/// One layer and token position after its recurrent-state update.
#[derive(Debug, Clone, PartialEq)]
pub enum GdnStateObservation {
    /// State matches the expected geometry.
    Health {
        /// Zero-based model layer index.
        layer: usize,
        /// Zero-based absolute token position.
        position: usize,
        /// Summary of the full recurrent state.
        total: StateHealth,
        /// Per-value-head summaries in head order.
        heads: Vec<StateHealth>,
    },
    /// A state with incompatible or overflowing geometry was observed.
    InvalidLayout {
        /// Zero-based model layer index.
        layer: usize,
        /// Zero-based absolute token position.
        position: usize,
        /// Expected number of values, or `None` for zero/overflowing geometry.
        expected: Option<usize>,
        /// Actual number of recurrent-state values.
        actual: usize,
    },
}

type Sink = Rc<dyn Fn(GdnStateObservation)>;

thread_local! {
    static OBSERVER: RefCell<Option<Sink>> = const { RefCell::new(None) };
}

/// Run `work` with an observer installed on the current thread.
///
/// Nested scopes restore the preceding observer, including during unwinding.
/// The callback is synchronous and its panic propagates. No state is copied to
/// another thread; callers must explicitly correlate concurrent requests. The
/// returned observations describe only the f32 CPU paths named in this module.
pub fn with_observer<R>(
    sink: impl Fn(GdnStateObservation) + 'static,
    work: impl FnOnce() -> R,
) -> R {
    struct Restore(Option<Sink>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OBSERVER.with(|slot| {
                slot.replace(self.0.take());
            });
        }
    }
    let previous = OBSERVER.with(|slot| slot.replace(Some(Rc::new(sink))));
    let _restore = Restore(previous);
    work()
}

fn health(values: &[f32]) -> StateHealth {
    let mut finite_count = 0;
    let mut nan_count = 0;
    let mut infinite_count = 0;
    let mut squared = 0.0f64;
    for &value in values {
        if value.is_nan() {
            nan_count += 1;
        } else if value.is_infinite() {
            infinite_count += 1;
        } else {
            finite_count += 1;
            let value = f64::from(value);
            squared += value * value;
        }
    }
    StateHealth {
        finite_count,
        nan_count,
        infinite_count,
        frobenius_norm: (nan_count == 0 && infinite_count == 0).then(|| squared.sqrt()),
    }
}

fn summarize(
    layer: usize,
    position: usize,
    values: &[f32],
    heads: usize,
    key: usize,
    value: usize,
) -> GdnStateObservation {
    let head_size = key.checked_mul(value).filter(|&n| n != 0);
    let expected = head_size
        .and_then(|size| heads.checked_mul(size))
        .filter(|_| heads != 0);
    let Some(size) = head_size.filter(|_| expected == Some(values.len())) else {
        return GdnStateObservation::InvalidLayout {
            layer,
            position,
            expected,
            actual: values.len(),
        };
    };
    GdnStateObservation::Health {
        layer,
        position,
        total: health(values),
        heads: values.chunks_exact(size).map(health).collect(),
    }
}

pub(crate) fn observe(
    layer: usize,
    position: usize,
    state: &GatedDeltaNetState,
    cfg: &Qwen35Config,
) {
    let sink = OBSERVER.with(|slot| slot.borrow().clone());
    if let Some(sink) = sink {
        sink(summarize(
            layer,
            position,
            &state.s_matrices,
            cfg.linear_num_value_heads(),
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
        ));
    }
}

#[cfg(test)]
mod tests;
