//! Fresh Chronos-2 request allocations with warmed CPU backend infrastructure.
//!
//! The tiny shapes keep Rust matmul work on the caller thread. This checks one
//! fixture against deterministic accounting, not RSS or excluded backend storage.

use std::hint::black_box;
use std::mem::size_of;
use std::panic::{UnwindSafe, catch_unwind, resume_unwind};

use allocation_counter::{AllocationInfo, measure};
use lattice_forecast::{Chronos2, Chronos2Config, Chronos2Series, QuantileForecast};
use lattice_inference_core::InferenceError;
use lattice_weights::SafetensorsFile;
use serde_json::{Map, Value, json};

fn tensor(
    header: &mut Map<String, Value>,
    data: &mut Vec<u8>,
    name: &str,
    shape: &[usize],
    value: f32,
) {
    let begin = data.len();
    for _ in 0..shape.iter().product::<usize>() {
        data.extend_from_slice(&value.to_le_bytes());
    }
    assert!(
        header
            .insert(
                name.into(),
                json!({
                    "dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]
                })
            )
            .is_none()
    );
}

fn linear(
    header: &mut Map<String, Value>,
    data: &mut Vec<u8>,
    name: &str,
    output: usize,
    input: usize,
    bias: bool,
) {
    tensor(
        header,
        data,
        &format!("{name}.weight"),
        &[output, input],
        0.0,
    );
    if bias {
        tensor(header, data, &format!("{name}.bias"), &[output], 0.0);
    }
}

fn residual(
    header: &mut Map<String, Value>,
    data: &mut Vec<u8>,
    name: &str,
    input: usize,
    hidden: usize,
    output: usize,
) {
    linear(
        header,
        data,
        &format!("{name}.hidden_layer"),
        hidden,
        input,
        true,
    );
    linear(
        header,
        data,
        &format!("{name}.output_layer"),
        output,
        hidden,
        true,
    );
    linear(
        header,
        data,
        &format!("{name}.residual_layer"),
        output,
        input,
        true,
    );
}

fn tiny_model() -> Chronos2 {
    let config = Chronos2Config {
        d_model: 8,
        d_kv: 4,
        num_heads: 2,
        d_ff: 16,
        num_layers: 1,
        layer_norm_epsilon: 1e-6,
        rope_theta: 10_000.0,
        context_length: 64,
        patch_size: 4,
        max_output_patches: 3,
        quantiles: vec![0.5],
        time_encoding_scale: 64.0,
        use_arcsinh: true,
        use_reg_token: false,
    };
    let (d, ff, inner) = (config.d_model, config.d_ff, config.num_heads * config.d_kv);
    let mut header = Map::new();
    let mut data = Vec::new();
    tensor(&mut header, &mut data, "shared.weight", &[1, d], 0.0);
    residual(
        &mut header,
        &mut data,
        "input_patch_embedding",
        3 * config.patch_size,
        ff,
        d,
    );
    for block in 0..config.num_layers {
        let prefix = format!("encoder.block.{block}.layer");
        for layer in 0..3 {
            tensor(
                &mut header,
                &mut data,
                &format!("{prefix}.{layer}.layer_norm.weight"),
                &[d],
                1.0,
            );
        }
        for layer in 0..2 {
            let attention = format!("{prefix}.{layer}.self_attention");
            for projection in ["q", "k", "v"] {
                linear(
                    &mut header,
                    &mut data,
                    &format!("{attention}.{projection}"),
                    inner,
                    d,
                    false,
                );
            }
            linear(
                &mut header,
                &mut data,
                &format!("{attention}.o"),
                d,
                inner,
                false,
            );
        }
        linear(
            &mut header,
            &mut data,
            &format!("{prefix}.2.mlp.wi"),
            ff,
            d,
            false,
        );
        linear(
            &mut header,
            &mut data,
            &format!("{prefix}.2.mlp.wo"),
            d,
            ff,
            false,
        );
    }
    tensor(
        &mut header,
        &mut data,
        "encoder.final_layer_norm.weight",
        &[d],
        1.0,
    );
    residual(
        &mut header,
        &mut data,
        "output_patch_embedding",
        d,
        ff,
        config.quantiles.len() * config.patch_size,
    );
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    let file = SafetensorsFile::from_bytes(bytes).unwrap();
    Chronos2::from_safetensors(config, &file).unwrap()
}

fn accounted_bound<T: std::fmt::Debug>(result: Result<T, InferenceError>) -> usize {
    let Err(InferenceError::InvalidInput(message)) = result else {
        panic!("zero budget did not produce a request-memory refusal: {result:?}");
    };
    let decimal = message
        .strip_prefix("Chronos-2: request memory requires ")
        .and_then(|value| value.strip_suffix(" bytes, exceeding budget 0 bytes"))
        .expect("exact request-memory refusal format");
    assert!(!decimal.is_empty() && decimal.bytes().all(|byte| byte.is_ascii_digit()));
    let bound = decimal.parse::<usize>().unwrap();
    assert!(bound > 0);
    assert_eq!(
        message,
        format!("Chronos-2: request memory requires {bound} bytes, exceeding budget 0 bytes")
    );
    bound
}

fn assert_observation(
    label: &str,
    observed: AllocationInfo,
    bound: usize,
    retained_bytes: usize,
    retained_allocations: i64,
) {
    assert!(
        observed.count_total > 0,
        "{label}: no allocation was observed"
    );
    assert!(
        observed.bytes_max > 0,
        "{label}: no live-byte peak was observed"
    );
    assert_eq!(
        observed.count_current, retained_allocations,
        "{label}: retained allocations"
    );
    assert_eq!(
        observed.bytes_current,
        i64::try_from(retained_bytes).unwrap(),
        "{label}: live bytes must match the returned buffers"
    );
    assert!(
        observed.bytes_max >= retained_bytes as u64,
        "{label}: peak below retained bytes"
    );
    assert!(
        observed.bytes_total >= observed.bytes_max,
        "{label}: peak above cumulative bytes"
    );
    let peak = usize::try_from(observed.bytes_max).unwrap();
    assert!(
        peak <= bound,
        "{label}: observed request peak {peak} exceeds accounted request peak {bound}"
    );
    println!(
        "{label}: accounted={bound} observed={peak} slack={} live={} allocations={}",
        bound - peak,
        observed.bytes_current,
        observed.count_total
    );
}

fn measure_call<T>(operation: impl FnOnce() -> T + UnwindSafe) -> (T, AllocationInfo) {
    let mut outcome = None;
    // Let measure restore its thread-local stack before propagating a request panic.
    let observed = measure(|| outcome = Some(catch_unwind(operation)));
    let value = match outcome.unwrap() {
        Ok(value) => value,
        Err(payload) => resume_unwind(payload),
    };
    (value, observed)
}

#[test]
fn predict_peak_fits_request_memory_account() {
    let mut model = tiny_model();
    let context = [2.0];
    model.set_request_memory_budget(Some(0));
    let bound = accounted_bound(model.predict(&context, 3));
    model.set_request_memory_budget(None);
    drop(model.predict(&context, 3).unwrap());

    // The model and input are borrowed; their existing allocations cannot be freed
    // by this call. The result slot starts empty and keeps returned storage live.
    let (result, observed) =
        measure_call(|| black_box(model.predict(black_box(&context), black_box(3))));
    let forecast = result.unwrap();
    assert_eq!(forecast.horizon, 3);
    assert_eq!(forecast.values, [2.0; 3]);
    assert_observation(
        "predict",
        observed,
        bound,
        forecast.values.capacity() * size_of::<f32>(),
        1,
    );
}

#[test]
fn trace_peak_fits_request_memory_account() {
    let mut model = tiny_model();
    let batch = [Chronos2Series {
        context: vec![2.0],
        ..Default::default()
    }];
    model.set_request_memory_budget(Some(0));
    let bound = accounted_bound(model.forward_with_trace(&batch, 1));
    model.set_request_memory_budget(None);
    drop(model.forward_with_trace(&batch, 1).unwrap());

    let (result, observed) =
        measure_call(|| black_box(model.forward_with_trace(black_box(&batch), black_box(1))));
    let (forecasts, trace) = result.unwrap();
    assert_eq!(forecasts.len(), 1);
    assert_eq!(forecasts[0].horizon, 4);
    assert_eq!(forecasts[0].values, [2.0; 4]);
    assert_eq!(trace.blocks.len(), 1);
    for state in std::iter::once(&trace.embeddings)
        .chain(&trace.blocks)
        .chain(std::iter::once(&trace.final_state))
    {
        assert_eq!(state.len(), 16);
        assert!(state.iter().all(|value| *value == 0.0));
    }
    let retained_bytes = forecasts.capacity() * size_of::<QuantileForecast>()
        + forecasts
            .iter()
            .map(|forecast| forecast.values.capacity() * size_of::<f32>())
            .sum::<usize>()
        + trace.embeddings.capacity() * size_of::<f32>()
        + trace.blocks.capacity() * size_of::<Vec<f32>>()
        + trace
            .blocks
            .iter()
            .map(|state| state.capacity() * size_of::<f32>())
            .sum::<usize>()
        + trace.final_state.capacity() * size_of::<f32>();
    // The growing descriptor vector's spare capacity is included in observation.
    // This geometry's earlier attention peak exceeds that later trace/output peak.
    assert_observation("trace", observed, bound, retained_bytes, 6);
}

#[test]
fn allocation_instrument_observes_zeroed_storage_and_growth() {
    let (values, zeroed) = measure_call(|| black_box(vec![0_u8; 64]));
    assert_eq!(zeroed.count_total, 1);
    assert_eq!(zeroed.count_current, 1);
    assert_eq!(zeroed.bytes_current, values.capacity() as i64);
    assert_eq!(zeroed.bytes_max, values.capacity() as u64);
    assert!(zeroed.bytes_max >= 64);
    drop(values);

    let (values, growth) = measure_call(|| {
        let mut values = black_box(Vec::<u8>::with_capacity(16));
        values.extend_from_slice(&[1; 16]);
        values.reserve_exact(64);
        black_box(values)
    });
    assert!(
        growth.count_total >= 2,
        "growth did not reach the allocator"
    );
    assert_eq!(growth.count_current, 1);
    assert_eq!(growth.bytes_current, values.capacity() as i64);
    assert!(growth.bytes_max >= values.capacity() as u64);
    assert!(values.capacity() >= 80);
}
