//! Which family and backend serve a checkpoint, and the evidence a request
//! leaves of the route it took.
//!
//! [`select_route`] is the `lattice serve` startup decision: the checkpoint
//! format picks the backend (a safetensors directory runs on the CPU, a native
//! Q4 directory on Metal) and `config.json` picks the family. Gemma 4 has a CPU
//! route only, so a Gemma 4 checkpoint in the Q4 format is refused rather than
//! handed to the Metal loader or to the CPU one.
//!
//! The markers are lines on the server's stderr. A selection line is written
//! once at startup and a request line once per CPU-served request. The request
//! line reports the shared decoder driver's own ledger counters, so
//! `driver=shared` is derived from a run that opened predictions, not asserted
//! by the code that prints it.
//!
//! Not a stable API: `#[doc(hidden)]` is a convention, not a semver guarantee.

use crate::model_format::{ModelFamily, ModelFormat, unrecognized_format_message};
use std::path::Path;

/// Stable code carried by the refusal of a Gemma 4 checkpoint on Metal.
pub const GEMMA_METAL_UNSUPPORTED_CODE: &str = "gemma_metal_unsupported";

/// The execution backend a checkpoint is served on.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServedBackend {
    /// Safetensors weights executed on the CPU.
    Cpu,
    /// Native Q4 weights executed by the Metal worker.
    Metal,
}

impl ServedBackend {
    /// The spelling used in route markers.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

/// A family served on a backend.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServedRoute {
    pub family: ModelFamily,
    pub backend: ServedBackend,
}

impl ServedRoute {
    /// Qwen3.5 on the CPU safetensors backend.
    pub const QWEN35_CPU: Self = Self {
        family: ModelFamily::Qwen35,
        backend: ServedBackend::Cpu,
    };

    /// Gemma 4 on the CPU safetensors backend.
    pub const GEMMA4_CPU: Self = Self {
        family: ModelFamily::Gemma4,
        backend: ServedBackend::Cpu,
    };

    /// The startup line recording which route was selected for the model
    /// directory's on-disk `format`.
    pub fn selection_marker(&self, format: ModelFormat) -> String {
        let format = match format {
            ModelFormat::Safetensors => "safetensors",
            ModelFormat::Q4 => "q4",
            _ => "unknown",
        };
        format!(
            "[route] selected family={} backend={} format={format}",
            self.family.name(),
            self.backend.name()
        )
    }

    /// The line one served request leaves behind. `driver=shared` only when
    /// the shared decoder driver opened at least one prediction for it.
    pub fn request_marker(&self, stream: bool, evidence: DriverEvidence) -> String {
        let driver = if evidence.went_through_driver() {
            "shared"
        } else {
            "bypassed"
        };
        format!(
            "[route] served family={} backend={} mode={} driver={driver} opened={} consumed={}",
            self.family.name(),
            self.backend.name(),
            if stream { "stream" } else { "nonstream" },
            evidence.opened,
            evidence.consumed
        )
    }
}

/// Why no route serves a checkpoint directory.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteRefusal {
    /// The directory holds neither safetensors weights nor Q4 tensors.
    UnrecognizedFormat,
    /// A Gemma 4 checkpoint in the Q4 format, which only the Metal backend
    /// reads. Gemma 4 has no Metal route.
    GemmaMetalUnsupported,
}

impl RouteRefusal {
    /// The startup error text for `dir`. The unrecognized-format text is the
    /// one `lattice serve` printed before routing existed.
    pub fn message(&self, dir: &Path) -> String {
        match self {
            Self::UnrecognizedFormat => unrecognized_format_message(dir),
            Self::GemmaMetalUnsupported => format!(
                "{GEMMA_METAL_UNSUPPORTED_CODE}: '{}' is a Gemma 4 checkpoint in the native Q4 \
                 format, which only the Metal backend reads, and Gemma 4 is served on the CPU \
                 backend only. Point --model at the Gemma 4 safetensors directory instead.",
                dir.display()
            ),
        }
    }
}

/// Pick the route for a checkpoint of `format` and `family`.
///
/// The format decides the backend: safetensors runs on the CPU, Q4 on Metal.
/// There is no backend switch, so asking for Gemma 4 on Metal can only mean a
/// Gemma 4 directory in the Q4 format, and that is refused here, before any
/// loader runs and whatever features the binary was built with.
pub fn select_route(format: ModelFormat, family: ModelFamily) -> Result<ServedRoute, RouteRefusal> {
    match (format, family) {
        (ModelFormat::Safetensors, family) => Ok(ServedRoute {
            family,
            backend: ServedBackend::Cpu,
        }),
        (ModelFormat::Q4, ModelFamily::Qwen35) => Ok(ServedRoute {
            family: ModelFamily::Qwen35,
            backend: ServedBackend::Metal,
        }),
        (ModelFormat::Q4, ModelFamily::Gemma4) => Err(RouteRefusal::GemmaMetalUnsupported),
        _ => Err(RouteRefusal::UnrecognizedFormat),
    }
}

/// The shared decoder driver's prediction-ledger counters for one request: a
/// prediction is opened per sampling step and consumed per decode step.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DriverEvidence {
    pub opened: usize,
    pub consumed: usize,
}

impl DriverEvidence {
    /// Whether the driver ran any sampling step for the request.
    pub fn went_through_driver(&self) -> bool {
        self.opened > 0
    }
}

impl DriverEvidence {
    pub(crate) fn from_trace(trace: crate::decoder::driver::DriverTrace) -> Self {
        Self {
            opened: trace.opened,
            consumed: trace.consumed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_picks_the_backend_and_config_picks_the_family() {
        assert_eq!(
            select_route(ModelFormat::Safetensors, ModelFamily::Qwen35),
            Ok(ServedRoute::QWEN35_CPU)
        );
        assert_eq!(
            select_route(ModelFormat::Safetensors, ModelFamily::Gemma4),
            Ok(ServedRoute::GEMMA4_CPU)
        );
        assert_eq!(
            select_route(ModelFormat::Q4, ModelFamily::Qwen35),
            Ok(ServedRoute {
                family: ModelFamily::Qwen35,
                backend: ServedBackend::Metal
            })
        );
    }

    #[test]
    fn gemma_in_the_metal_format_is_refused_not_rerouted() {
        let refusal = select_route(ModelFormat::Q4, ModelFamily::Gemma4)
            .expect_err("Gemma 4 has no Metal route");
        assert_eq!(refusal, RouteRefusal::GemmaMetalUnsupported);
        let text = refusal.message(Path::new("/models/gemma-q4"));
        assert!(
            text.starts_with(GEMMA_METAL_UNSUPPORTED_CODE),
            "the refusal leads with its stable code: {text}"
        );
        assert!(text.contains("CPU"), "{text}");
        assert!(text.contains("/models/gemma-q4"), "{text}");
    }

    #[test]
    fn an_unrecognized_directory_keeps_the_existing_error_text() {
        let dir = Path::new("/models/empty");
        let refusal = select_route(ModelFormat::Unknown, ModelFamily::Qwen35)
            .expect_err("an unrecognized directory has no route");
        assert_eq!(refusal, RouteRefusal::UnrecognizedFormat);
        assert_eq!(refusal.message(dir), unrecognized_format_message(dir));
        assert_eq!(
            select_route(ModelFormat::Unknown, ModelFamily::Gemma4),
            Err(RouteRefusal::UnrecognizedFormat)
        );
    }

    #[test]
    fn selection_marker_names_family_backend_and_format() {
        assert_eq!(
            ServedRoute::GEMMA4_CPU.selection_marker(ModelFormat::Safetensors),
            "[route] selected family=gemma4 backend=cpu format=safetensors"
        );
        assert_eq!(
            ServedRoute::QWEN35_CPU.selection_marker(ModelFormat::Safetensors),
            "[route] selected family=qwen35 backend=cpu format=safetensors"
        );
    }

    #[test]
    fn request_marker_derives_the_driver_field_from_the_counters() {
        let ran = DriverEvidence {
            opened: 5,
            consumed: 4,
        };
        assert_eq!(
            ServedRoute::GEMMA4_CPU.request_marker(true, ran),
            "[route] served family=gemma4 backend=cpu mode=stream driver=shared opened=5 \
             consumed=4"
        );
        assert_eq!(
            ServedRoute::QWEN35_CPU.request_marker(false, ran),
            "[route] served family=qwen35 backend=cpu mode=nonstream driver=shared opened=5 \
             consumed=4"
        );
        // A request the driver never sampled for cannot claim the driver.
        let untouched = ServedRoute::GEMMA4_CPU.request_marker(false, DriverEvidence::default());
        assert!(untouched.contains("driver=bypassed"), "{untouched}");
        assert!(!untouched.contains("driver=shared"), "{untouched}");
    }
}
