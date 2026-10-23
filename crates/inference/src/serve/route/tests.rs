use super::*;
use crate::model_format::unrecognized_format_message;

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
fn the_standalone_table_runs_qwen_on_metal_in_both_formats() {
    for format in [ModelFormat::Safetensors, ModelFormat::Q4] {
        assert_eq!(
            select_standalone_route(format, ModelFamily::Qwen35),
            Ok(ServedRoute {
                family: ModelFamily::Qwen35,
                backend: ServedBackend::Metal
            }),
            "{format:?}"
        );
    }
    assert_eq!(
        select_standalone_route(ModelFormat::Safetensors, ModelFamily::Gemma4),
        Ok(ServedRoute::GEMMA4_CPU)
    );
}

#[test]
fn the_two_tables_differ_only_in_the_qwen_safetensors_backend() {
    let formats = [
        ModelFormat::Safetensors,
        ModelFormat::Q4,
        ModelFormat::Unknown,
    ];
    let families = [ModelFamily::Qwen35, ModelFamily::Gemma4];
    for format in formats {
        for family in families {
            let cli = select_route(format, family);
            let standalone = select_standalone_route(format, family);
            if (format, family) == (ModelFormat::Safetensors, ModelFamily::Qwen35) {
                assert_ne!(cli, standalone);
            } else {
                assert_eq!(cli, standalone, "{format:?} {family:?}");
            }
        }
    }
}

#[test]
fn the_standalone_table_refuses_gemma_in_the_metal_format() {
    assert_eq!(
        select_standalone_route(ModelFormat::Q4, ModelFamily::Gemma4),
        Err(RouteRefusal::GemmaMetalUnsupported)
    );
    assert_eq!(
        select_standalone_route(ModelFormat::Unknown, ModelFamily::Qwen35),
        Err(RouteRefusal::UnrecognizedFormat)
    );
}

#[test]
fn gemma_in_the_metal_format_is_refused_not_rerouted() {
    let refusal =
        select_route(ModelFormat::Q4, ModelFamily::Gemma4).expect_err("Gemma 4 has no Metal route");
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
