//! Opt-in grouped GPU timing for ordinary direct decode.

use std::cell::RefCell;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use lattice_inference_core::InferenceError;
use lattice_metal_runtime::measurement::{
    ResolvedTimestamps, completed_command_buffer_gpu_ms, read_resolved_timestamps,
};
use metal::{
    CommandBufferRef, ComputeCommandEncoderRef, ComputePassDescriptor,
    ComputePassSampleBufferAttachmentDescriptor, DeviceRef, MTLDispatchType,
};

const LAYERS: usize = 24;
const GROUPS: usize = LAYERS * 2 + 1;
pub(super) const SAMPLES: usize = GROUPS * 2;

struct Step {
    position: usize,
    raw: Vec<u64>,
    command_ms: Result<f64, String>,
}

#[derive(Default)]
struct Trace {
    request_id: u64,
    prompt_tokens: usize,
    before: Option<(u64, u64)>,
    steps: Vec<Step>,
    error: Option<String>,
}

thread_local! {
    static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
}

pub(super) fn begin(prompt_tokens: usize) -> Result<(), InferenceError> {
    TRACE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(InferenceError::Inference(
                "GPU timing request already active".into(),
            ));
        }
        static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
        *slot = Some(Trace {
            request_id: NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed),
            prompt_tokens,
            ..Trace::default()
        });
        Ok(())
    })
}

pub(super) fn active() -> bool {
    TRACE.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|trace| trace.error.is_none())
    })
}

pub(super) fn invalidate(message: impl Into<String>) {
    TRACE.with(|slot| {
        if let Some(trace) = slot.borrow_mut().as_mut() {
            if trace.error.is_none() {
                trace.error = Some(message.into());
            }
        }
    });
}

pub(super) fn start_first_decode(device: &DeviceRef) {
    TRACE.with(|slot| {
        if let Some(trace) = slot.borrow_mut().as_mut() {
            if trace.before.is_none() && trace.error.is_none() {
                let (mut cpu, mut gpu) = (0, 0);
                device.sample_timestamps(&mut cpu, &mut gpu);
                if cpu == 0 || gpu == 0 {
                    trace.error = Some("invalid initial clock calibration".into());
                } else {
                    trace.before = Some((cpu, gpu));
                }
            }
        }
    });
}

fn sample_indices(group: usize) -> Result<(u64, u64), InferenceError> {
    if group >= GROUPS {
        return Err(InferenceError::Inference(
            "GPU timing group out of bounds".into(),
        ));
    }
    Ok(((group * 2) as u64, (group * 2 + 1) as u64))
}

pub(super) fn encoder<'a>(
    command: &'a CommandBufferRef,
    samples: &ResolvedTimestamps,
    group: usize,
) -> Result<&'a ComputeCommandEncoderRef, InferenceError> {
    let (start_index, end_index) = sample_indices(group)?;
    let pass = ComputePassDescriptor::new();
    pass.set_dispatch_type(MTLDispatchType::Serial);
    let attachment = ComputePassSampleBufferAttachmentDescriptor::new();
    attachment.set_sample_buffer(samples.sample_buffer());
    attachment.set_start_of_encoder_sample_index(start_index);
    attachment.set_end_of_encoder_sample_index(end_index);
    pass.sample_buffer_attachments()
        .set_object_at(0, Some(&attachment));
    Ok(command.compute_command_encoder_with_descriptor(pass))
}

pub(super) fn group_encoder<'a>(
    command: &'a CommandBufferRef,
    samples: &ResolvedTimestamps,
    group: usize,
) -> &'a ComputeCommandEncoderRef {
    match encoder(command, samples, group) {
        Ok(encoder) => encoder,
        Err(error) => {
            invalidate(error.to_string());
            command.new_compute_command_encoder()
        }
    }
}

fn retain_step(trace: &mut Trace, position: usize, raw: Vec<u64>, command_ms: Result<f64, String>) {
    let error = command_ms.as_ref().err().cloned();
    trace.steps.push(Step {
        position,
        raw,
        command_ms,
    });
    if trace.error.is_none() {
        trace.error = error;
    }
}

pub(super) fn collect(command: &CommandBufferRef, owner: &mut ResolvedTimestamps, position: usize) {
    match read_resolved_timestamps(command, owner, 0, SAMPLES) {
        Ok(raw) => {
            let command_ms =
                completed_command_buffer_gpu_ms(command).map_err(|error| error.to_string());
            TRACE.with(|slot| {
                if let Some(trace) = slot.borrow_mut().as_mut() {
                    retain_step(trace, position, raw, command_ms);
                }
            });
        }
        Err(error) => invalidate(error.to_string()),
    }
}

fn invalid_step_line(request_id: u64, index: usize, step: &Step) -> String {
    let raw = step
        .raw
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let control = match &step.command_ms {
        Ok(value) => format!("c_ms={value}"),
        Err(error) => format!("c_ms=unavailable control_reason={}", reason_token(error)),
    };
    format!(
        "[gpu_timing] status=invalid request={request_id} step={index} position={} {control} raw={raw}",
        step.position
    )
}

fn group_name(group: usize) -> &'static str {
    if group == GROUPS - 1 {
        "head"
    } else if group % 2 == 1 {
        "mlp"
    } else if (group / 2) % 4 == 3 {
        "gqa"
    } else {
        "gdn"
    }
}

struct FormattedStep {
    line: String,
    command_ms: f64,
    sum_ms: f64,
    residual_ms: f64,
}

fn reason_token(message: &str) -> String {
    message
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn control_residual(command_ms: f64, sum: f64) -> Result<f64, String> {
    if !command_ms.is_finite() || command_ms <= 0.0 || !sum.is_finite() || sum <= 0.0 {
        return Err("invalid command or group duration".into());
    }
    let residual = command_ms - sum;
    let lower = 0.020_f64.max(0.005 * command_ms);
    let upper = 0.100_f64.max(0.05 * command_ms);
    if residual < -lower || residual > upper {
        return Err(format!(
            "command control failed C={command_ms} S={sum} R={residual}"
        ));
    }
    Ok(residual)
}

fn format_step(
    request_id: u64,
    index: usize,
    step: &Step,
    cpu_span: u64,
    gpu_start: u64,
    gpu_end: u64,
) -> Result<FormattedStep, String> {
    let command_ms = *step
        .command_ms
        .as_ref()
        .map_err(|error| format!("command control unavailable: {error}"))?;
    if step.raw.len() != SAMPLES || !command_ms.is_finite() || command_ms <= 0.0 {
        return Err("invalid sample population or command duration".into());
    }
    let gpu_span = gpu_end
        .checked_sub(gpu_start)
        .filter(|&n| n > 0)
        .ok_or("invalid GPU calibration")?;
    let mut entries = String::new();
    let mut sum = 0.0;
    let mut gaps = 0_u64;
    let mut prior_end = 0;
    for (group, pair) in step.raw.chunks_exact(2).enumerate() {
        let (start, end) = (pair[0], pair[1]);
        if start == 0
            || end == u64::MAX
            || start == u64::MAX
            || end <= start
            || start < gpu_start
            || end > gpu_end
            || (group > 0 && start < prior_end)
        {
            return Err(format!("invalid or overlapping samples in group {group}"));
        }
        if group > 0 {
            gaps = gaps
                .checked_add(start - prior_end)
                .ok_or("sample gap overflow")?;
        }
        prior_end = end;
        let ms = (end - start) as f64 / gpu_span as f64 * cpu_span as f64 / 1_000_000.0;
        if !ms.is_finite() || ms <= 0.0 {
            return Err(format!("invalid converted interval in group {group}"));
        }
        sum += ms;
        if group != 0 {
            entries.push(',');
        }
        write!(
            &mut entries,
            "{group}:{}:{start}:{end}:{ms:.9}",
            group_name(group)
        )
        .map_err(|error| error.to_string())?;
    }
    let envelope_ms =
        (prior_end - step.raw[0]) as f64 / gpu_span as f64 * cpu_span as f64 / 1_000_000.0;
    let gaps_ms = gaps as f64 / gpu_span as f64 * cpu_span as f64 / 1_000_000.0;
    if !envelope_ms.is_finite() || !gaps_ms.is_finite() {
        return Err("invalid envelope or gap duration".into());
    }
    let residual = control_residual(command_ms, sum)?;
    Ok(FormattedStep {
        line: format!(
            "[gpu_timing] status=ok request={request_id} step={index} position={} groups={GROUPS} samples={SAMPLES} c_ms={:.9} s_ms={sum:.9} r_ms={residual:.9} s_over_c={:.9} r_over_c={:.9} envelope_ms={envelope_ms:.9} gaps_ms={gaps_ms:.9} entries={entries}",
            step.position,
            command_ms,
            sum / command_ms,
            residual / command_ms
        ),
        command_ms,
        sum_ms: sum,
        residual_ms: residual,
    })
}

pub(super) fn finish(
    device: &DeviceRef,
    generated: usize,
    requested: usize,
) -> Result<(), InferenceError> {
    let trace = TRACE
        .with(|slot| slot.borrow_mut().take())
        .ok_or_else(|| InferenceError::Inference("GPU timing request missing".into()))?;
    let (mut cpu_end, mut gpu_end) = (0, 0);
    device.sample_timestamps(&mut cpu_end, &mut gpu_end);
    let result = (|| -> Result<(Vec<FormattedStep>, u64, u64), String> {
        if let Some(error) = trace.error.as_ref() {
            return Err(error.clone());
        }
        if generated != requested || trace.steps.len() != requested.saturating_sub(1) {
            return Err(format!(
                "unexpected population outputs={generated} requested={requested} steps={}",
                trace.steps.len()
            ));
        }
        if trace.steps.first().map(|step| step.position) != Some(trace.prompt_tokens) {
            return Err("first decode position differs from prompt length".into());
        }
        if trace
            .steps
            .windows(2)
            .any(|pair| pair[0].position.checked_add(1) != Some(pair[1].position))
        {
            return Err("noncontiguous decode positions".into());
        }
        let (cpu_start, gpu_start) = trace.before.ok_or("missing initial calibration")?;
        let cpu_span = cpu_end
            .checked_sub(cpu_start)
            .filter(|&n| n > 0)
            .ok_or("invalid CPU clock calibration")?;
        gpu_end
            .checked_sub(gpu_start)
            .filter(|&n| n > 0)
            .ok_or("invalid GPU clock calibration")?;
        let mut lines = Vec::with_capacity(trace.steps.len());
        for (index, step) in trace.steps.iter().enumerate() {
            lines.push(format_step(
                trace.request_id,
                index,
                step,
                cpu_span,
                gpu_start,
                gpu_end,
            )?);
        }
        Ok((lines, cpu_start, gpu_start))
    })();
    match result {
        Ok((lines, cpu_start, gpu_start)) => {
            let (mut c_sum, mut s_sum, mut r_sum) = (0.0, 0.0, 0.0);
            for step in &lines {
                c_sum += step.command_ms;
                s_sum += step.sum_ms;
                r_sum += step.residual_ms;
                eprintln!("{}", step.line);
            }
            let first = trace.steps.first().map_or(0, |step| step.position);
            let last = trace.steps.last().map_or(0, |step| step.position);
            eprintln!(
                "[gpu_timing_request] status=ok request={} prompt_tokens={} outputs={generated} steps={} first_position={first} last_position={last} cpu_start_ns={cpu_start} cpu_end_ns={cpu_end} gpu_start={gpu_start} gpu_end={gpu_end} c_sum_ms={c_sum:.9} s_sum_ms={s_sum:.9} r_sum_ms={r_sum:.9}",
                trace.request_id,
                trace.prompt_tokens,
                lines.len()
            );
            Ok(())
        }
        Err(error) => {
            for (index, step) in trace.steps.iter().enumerate() {
                eprintln!("{}", invalid_step_line(trace.request_id, index, step));
            }
            eprintln!(
                "[gpu_timing_request] status=invalid request={} prompt_tokens={} outputs={generated} requested={requested} steps={} cpu_start_ns={} cpu_end_ns={cpu_end} gpu_start={} gpu_end={gpu_end} reason={}",
                trace.request_id,
                trace.prompt_tokens,
                trace.steps.len(),
                trace.before.map_or(0, |pair| pair.0),
                trace.before.map_or(0, |pair| pair.1),
                reason_token(&error)
            );
            Err(InferenceError::Inference(format!(
                "GPU timing invalid: {error}"
            )))
        }
    }
}

pub(super) fn abort(device: &DeviceRef, requested: usize, error: &InferenceError) {
    if let Some(trace) = TRACE.with(|slot| slot.borrow_mut().take()) {
        let (mut cpu_end, mut gpu_end) = (0, 0);
        device.sample_timestamps(&mut cpu_end, &mut gpu_end);
        for (index, step) in trace.steps.iter().enumerate() {
            eprintln!("{}", invalid_step_line(trace.request_id, index, step));
        }
        eprintln!(
            "[gpu_timing_request] status=invalid request={} prompt_tokens={} outputs=unknown requested={requested} steps={} cpu_start_ns={} cpu_end_ns={cpu_end} gpu_start={} gpu_end={gpu_end} reason={}",
            trace.request_id,
            trace.prompt_tokens,
            trace.steps.len(),
            trace.before.map_or(0, |pair| pair.0),
            trace.before.map_or(0, |pair| pair.1),
            reason_token(&error.to_string())
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GROUPS, SAMPLES, Step, Trace, control_residual, format_step, invalid_step_line,
        reason_token, retain_step, sample_indices,
    };

    fn parse_entries(line: &str) -> Result<Vec<(usize, u64, u64)>, &'static str> {
        let entries = line
            .split_whitespace()
            .find_map(|field| field.strip_prefix("entries="))
            .ok_or("entries missing")?;
        let mut parsed = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for entry in entries.split(',') {
            let parts = entry.split(':').collect::<Vec<_>>();
            if parts.len() != 5 {
                return Err("invalid group record");
            }
            let group: usize = parts[0].parse().map_err(|_| "invalid index")?;
            let start: u64 = parts[2].parse().map_err(|_| "invalid start")?;
            let end: u64 = parts[3].parse().map_err(|_| "invalid end")?;
            let _: f64 = parts[4].parse().map_err(|_| "invalid duration")?;
            if !seen.insert(group) || start >= end {
                return Err("duplicate or invalid group");
            }
            parsed.push((group, start, end));
        }
        if parsed.len() != GROUPS {
            return Err("missing group");
        }
        Ok(parsed)
    }

    #[test]
    fn failed_control_preserves_the_successfully_read_raw_samples() {
        let mut trace = Trace::default();
        let samples = vec![17, 19, u64::MAX];
        retain_step(
            &mut trace,
            2073,
            samples.clone(),
            Err("C unavailable".into()),
        );
        assert_eq!(trace.steps[0].raw, samples);
        assert_eq!(trace.error.as_deref(), Some("C unavailable"));
        let line = invalid_step_line(1, 0, &trace.steps[0]);
        assert!(line.contains("c_ms=unavailable control_reason=C_unavailable"));
        assert!(line.contains("raw=17,19,18446744073709551615"));
    }

    #[test]
    fn error_reason_remains_one_parseable_token() {
        assert_eq!(
            reason_token("bad status=error\nnext"),
            "bad_status_error_next"
        );
    }

    #[test]
    fn sample_indices_cover_exactly_the_frozen_population() {
        let mut all = std::collections::BTreeSet::new();
        for group in 0..GROUPS {
            let (start, end) = sample_indices(group).unwrap();
            assert!(all.insert(start));
            assert!(all.insert(end));
        }
        assert_eq!(all.len(), SAMPLES);
        assert!(sample_indices(GROUPS).is_err());
    }

    #[test]
    fn step_formatter_is_parseable_and_rejects_invalid_samples() {
        let raw = (0..SAMPLES).map(|i| 100 + i as u64).collect();
        let mut step = Step {
            position: 7,
            raw,
            command_ms: Ok(GROUPS as f64 / 1_000_000.0),
        };
        let formatted = format_step(1, 0, &step, 1000, 1, 1001).unwrap();
        let fields = formatted.line.split_whitespace().collect::<Vec<_>>();
        assert!(fields.contains(&"status=ok"));
        let entries = fields
            .iter()
            .find_map(|s| s.strip_prefix("entries="))
            .unwrap();
        assert_eq!(entries.split(',').count(), GROUPS);
        let parsed = parse_entries(&formatted.line).unwrap();
        assert_eq!(parsed.len(), GROUPS);
        assert_eq!(parsed[0], (0, 100, 101));
        let duplicate = formatted.line.replace("1:mlp:102:103", "0:mlp:102:103");
        assert!(parse_entries(&duplicate).is_err());
        for (index, value) in [(0, 0), (1, u64::MAX), (1, 100)] {
            let original = step.raw[index];
            step.raw[index] = value;
            assert!(format_step(1, 0, &step, 1000, 1, 1001).is_err());
            step.raw[index] = original;
        }
        step.raw.pop();
        assert!(format_step(1, 0, &step, 1000, 1, 1001).is_err());
    }

    #[test]
    fn control_tolerances_fail_outside_both_relative_and_absolute_windows() {
        for (command, residual, accepted) in [
            (10.0, -0.049999, true),
            (10.0, -0.050001, false),
            (10.0, 0.499999, true),
            (10.0, 0.500001, false),
            (1.0, -0.019999, true),
            (1.0, -0.020001, false),
            (1.0, 0.099999, true),
            (1.0, 0.100001, false),
        ] {
            assert_eq!(
                control_residual(command, command - residual).is_ok(),
                accepted
            );
        }
    }

    #[test]
    fn integer_delta_preserves_high_timestamp_bits_and_touching_boundaries() {
        let base = 1_u64 << 60;
        let mut raw = Vec::new();
        let mut start = base + 3;
        for _ in 0..GROUPS {
            raw.push(start);
            raw.push(start + 7);
            start += 7;
        }
        let step = Step {
            position: 8,
            raw,
            command_ms: Ok(GROUPS as f64 * 0.014),
        };
        let formatted = format_step(2, 0, &step, 2_000_000, base, base + 1000).unwrap();
        assert!(formatted.line.contains(":0.014000000"));
        let mut overlapping = step;
        overlapping.raw[2] -= 1;
        assert!(format_step(2, 0, &overlapping, 2_000_000, base, base + 1000).is_err());
    }
}
