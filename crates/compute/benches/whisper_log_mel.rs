//! Reused-plan CPU log-mel timings for deterministic one- and thirty-second PCM.
//!
//! Run from the repository root with `scripts/bench-command.sh --label
//! whisper-log-mel --durable -- cargo bench --locked -p lattice-compute --bench
//! whisper_log_mel`. The three samples are descriptive, not a statistical gate.

use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use lattice_compute::audio::WhisperLogMelPlan;

const SAMPLE_RATE: usize = 16_000;
const MEASURED_SAMPLES: usize = 3;

fn measure(
    plan: &WhisperLogMelPlan,
    samples: &[f32],
    expected_frames: usize,
) -> Result<Duration, Box<dyn Error>> {
    let start = Instant::now();
    let output = black_box(plan).extract(black_box(samples), SAMPLE_RATE)?;
    black_box(&output);
    let elapsed = start.elapsed();

    if output.frames != expected_frames
        || output.values.len() != 128 * expected_frames
        || !output.values.iter().all(|value| value.is_finite())
    {
        return Err(format!(
            "invalid log-mel output: frames={}, values={}, expected frames={expected_frames}",
            output.frames,
            output.values.len()
        )
        .into());
    }
    Ok(elapsed)
}

fn main() -> Result<(), Box<dyn Error>> {
    let plan = WhisperLogMelPlan::new();
    for seconds in [1, 30] {
        let samples: Vec<f32> = (0..seconds * SAMPLE_RATE)
            .map(|i| ((i % 257) as f32 - 128.0) / 128.0)
            .collect();
        let frames = seconds * 100;

        let _ = measure(&plan, &samples, frames)?;
        let mut elapsed = [Duration::ZERO; MEASURED_SAMPLES];
        for (index, duration) in elapsed.iter_mut().enumerate() {
            *duration = measure(&plan, &samples, frames)?;
            println!(
                "whisper_log_mel seconds={seconds} input_samples={} frames={frames} warmups=1 sample={} elapsed_seconds={:.9}",
                samples.len(),
                index + 1,
                duration.as_secs_f64()
            );
        }
        elapsed.sort_unstable();
        println!(
            "whisper_log_mel seconds={seconds} measured_samples={MEASURED_SAMPLES} median_seconds={:.9}",
            elapsed[MEASURED_SAMPLES / 2].as_secs_f64()
        );
    }
    Ok(())
}
