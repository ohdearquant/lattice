//! Internal completed-command-buffer GPU timing.

use lattice_inference_core::InferenceError;
use metal::{CommandBufferRef, MTLCommandBufferStatus};

fn gpu_span_ms(start: f64, end: f64) -> Result<f64, InferenceError> {
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end <= start {
        return Err(InferenceError::Inference(
            "Metal GPU timestamps are unavailable or invalid".into(),
        ));
    }
    let milliseconds = (end - start) * 1_000.0;
    if !milliseconds.is_finite() || milliseconds <= 0.0 {
        return Err(InferenceError::Inference(
            "Metal GPU timestamp span is unavailable or invalid".into(),
        ));
    }
    Ok(milliseconds)
}

/// Return a completed command buffer's GPU span in milliseconds.
///
/// Returns an error for an incomplete or failed command buffer, unavailable
/// properties, or timestamps that do not form a finite, positive span.
#[doc(hidden)]
pub fn completed_command_buffer_gpu_ms(command: &CommandBufferRef) -> Result<f64, InferenceError> {
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(InferenceError::Inference(format!(
            "Metal command buffer is not completed: {:?}",
            command.status()
        )));
    }
    let start_selector = objc::runtime::Sel::register("GPUStartTime");
    let end_selector = objc::runtime::Sel::register("GPUEndTime");
    // SAFETY: both selectors are the documented no-argument CFTimeInterval command-buffer
    // properties. The command buffer completed before this read, so the timestamps are final.
    let start: f64 = unsafe {
        objc::Message::send_message(command, start_selector, ()).map_err(|error| {
            InferenceError::Inference(format!("Metal GPUStartTime unavailable: {error}"))
        })?
    };
    // SAFETY: the completed command buffer responds to the documented no-argument
    // GPUEndTime property with a CFTimeInterval value.
    let end: f64 = unsafe {
        objc::Message::send_message(command, end_selector, ()).map_err(|error| {
            InferenceError::Inference(format!("Metal GPUEndTime unavailable: {error}"))
        })?
    };
    gpu_span_ms(start, end)
}

#[cfg(test)]
mod tests {
    use super::gpu_span_ms;

    #[test]
    fn finite_ordered_timestamps_convert_to_milliseconds() {
        let milliseconds = gpu_span_ms(2.0, 2.25).unwrap();
        assert_eq!(milliseconds, 250.0);
    }

    #[test]
    fn invalid_or_unavailable_timestamps_are_rejected() {
        for (start, end) in [
            (0.0, 0.0),
            (0.0, 1.0),
            (-1.0, 1.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (1.0, f64::INFINITY),
            (1.0, f64::MAX),
        ] {
            assert!(gpu_span_ms(start, end).is_err(), "start={start}, end={end}");
        }
    }
}
