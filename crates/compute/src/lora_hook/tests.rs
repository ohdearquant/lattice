use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct RowSensitiveHook {
    calls: AtomicUsize,
}

impl LoraHook for RowSensitiveHook {
    fn apply(&self, _layer_idx: usize, _module: &str, x: &[f32], output: &mut [f32]) {
        assert_eq!(x.len(), 2);
        assert_eq!(output.len(), 3);
        self.calls.fetch_add(1, Ordering::Relaxed);
        output.fill(x[0]);
    }
}

#[test]
fn applies_lora_to_each_flattened_token_row() {
    let hook = RowSensitiveHook {
        calls: AtomicUsize::new(0),
    };
    let input = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
    let mut output = [0.0; 9];

    apply_lora_rows(&hook, 0, "projection", &input, &mut output, 2, 3);

    assert_eq!(hook.calls.load(Ordering::Relaxed), 3);
    assert_eq!(output, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
}

struct InactiveHook {
    calls: AtomicUsize,
}

impl LoraHook for InactiveHook {
    fn apply(&self, _layer_idx: usize, _module: &str, _x: &[f32], _output: &mut [f32]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    fn is_active(&self, _layer_idx: usize, _module: &str) -> bool {
        false
    }
}

/// A hook that reports `is_active == false` for a projection must never
/// have `apply` dispatched for any row of that projection: `apply_lora_rows`
/// checks activity once, before the per-row loop, not per row.
#[test]
fn skips_the_per_row_loop_entirely_when_the_hook_is_inactive() {
    let hook = InactiveHook {
        calls: AtomicUsize::new(0),
    };
    let input = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
    let mut output = [7.0; 9];

    apply_lora_rows(&hook, 0, "projection", &input, &mut output, 2, 3);

    assert_eq!(
        hook.calls.load(Ordering::Relaxed),
        0,
        "apply must not be called for any row when is_active is false"
    );
    assert_eq!(output, [7.0; 9], "output must be untouched");
}

/// `NoopLoraHook` (the default when no adapter is loaded) must report
/// itself inactive for every projection, so the default (no-adapter)
/// path stays off the per-row virtual-dispatch cost.
#[test]
fn noop_hook_reports_inactive_for_any_projection() {
    let hook = NoopLoraHook;
    assert!(!hook.is_active(0, "query"));
    assert!(!hook.is_active(41, "ffn_output"));
}

struct LocalGeometry;

impl LoraTargetGeometry for LocalGeometry {
    fn num_layers(&self) -> usize {
        1
    }

    fn projection_shape(
        &self,
        _layer_idx: usize,
        _module: &str,
    ) -> Result<LoraProjectionShape, String> {
        Err("no projections declared".to_string())
    }
}

#[test]
fn default_geometry_validation_accepts_neutral_target() {
    let hook: &dyn LoraHook = &NoopLoraHook;
    assert_eq!(hook.validate_geometry(&LocalGeometry), Ok(()));
}
