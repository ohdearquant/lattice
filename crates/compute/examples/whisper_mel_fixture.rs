//! Binary fixture bridge for `scripts/compare_whisper_log_mel.py`.

use std::io::{Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input)?;
    if input.len() % 4 != 0 {
        return Err("PCM input must be little-endian f32 bytes".into());
    }
    let samples: Vec<f32> = input
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect();
    let output = lattice_compute::audio::WhisperLogMelPlan::new().extract(&samples, 16_000)?;
    let frames = u32::try_from(output.frames)?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&frames.to_le_bytes())?;
    for value in output.values {
        stdout.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
