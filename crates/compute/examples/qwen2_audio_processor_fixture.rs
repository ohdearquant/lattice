//! Binary fixture for the Qwen2-Audio processor differential.
use std::io::{Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    std::io::stdin()
        .take(480_000 * 4 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() % 4 != 0 || bytes.len() > 480_000 * 4 {
        return Err("expected at most 480000 little-endian f32 samples".into());
    }
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let plan = lattice_compute::audio::WhisperLogMelPlan::new();
    let result = lattice_compute::audio::qwen2_audio::process(&plan, &samples, 16_000)?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&u32::try_from(result.feature_attention_mask.len())?.to_le_bytes())?;
    stdout.write_all(&u32::try_from(result.audio_tokens)?.to_le_bytes())?;
    stdout.write_all(&result.feature_attention_mask)?;
    for value in result.features {
        stdout.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
