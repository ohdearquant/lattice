// Compatibility entry point for the historical layer-23 trainer.

use std::path::PathBuf;

use lattice_tune::train_support as train_common;
use train_common::ArgView;
use train_common::full_driver::{self, FullDriverConfig};

const LEGACY_LAYER: usize = 23;
const LEGACY_A_INIT_AMP: f32 = 0.02;

fn usage() {
    eprintln!(
        "usage: train_grad_layer23 [OPTIONS]

Options:
  --model-dir  <PATH>   Model directory (default: $HOME/.lattice/models/qwen3.5-0.8b)
  --data-dir   <PATH>   Dataset directory with train.jsonl (default: data/lora-train)
  --steps      <N>      Adam steps (default: 25)
  --lr         <F>      Learning rate (default: 1e-3)
  --rank       <N>      LoRA rank (default: 8)
  --alpha      <F>      LoRA alpha (default: 16.0)
  --seq-len    <N>      Max tokens per sample (default: 64)
  --max-train  <N>      Training samples cap (default: 3)
  --log-every  <N>      Print NLL every N steps (default: 5)
  --save       <PATH>   Save trained adapter as PEFT safetensors
  -h, --help            Print this help"
    );
}

fn parse_alpha(argv: &ArgView<'_>) -> Result<f32, String> {
    if !argv.flag("--alpha") {
        return Ok(16.0);
    }
    let value = argv
        .arg("--alpha")
        .ok_or_else(|| "invalid --alpha: missing value".to_string())?;
    let alpha = value
        .parse::<f32>()
        .map_err(|_| format!("invalid --alpha '{value}': expected a finite number"))?;
    if !alpha.is_finite() {
        return Err(format!("invalid --alpha '{value}': value must be finite"));
    }
    Ok(alpha)
}

fn parse_config(argv: &ArgView<'_>) -> Result<FullDriverConfig, String> {
    let log_every = argv
        .arg("--log-every")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    if log_every == 0 {
        return Err("--log-every must be >= 1".to_string());
    }
    Ok(FullDriverConfig {
        model_dir: argv
            .arg("--model-dir")
            .map(PathBuf::from)
            .unwrap_or_else(train_common::default_model_dir),
        data_dir: argv
            .arg("--data-dir")
            .map(PathBuf::from)
            .unwrap_or_else(train_common::default_data_dir),
        first_layer: LEGACY_LAYER,
        steps: argv
            .arg("--steps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(25),
        lr: argv
            .arg("--lr")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-3),
        rank: argv.arg("--rank").and_then(|s| s.parse().ok()).unwrap_or(8),
        alpha: parse_alpha(argv)?,
        seq_len_cap: argv
            .arg("--seq-len")
            .and_then(|s| s.parse().ok())
            .unwrap_or(64),
        max_train: argv
            .arg("--max-train")
            .and_then(|s| s.parse().ok())
            .unwrap_or(3),
        max_valid: 0,
        log_every,
        gradcheck: false,
        gradcheck_strided_probes: false,
        probe: 6,
        fd_eps: 4e-3,
        save_path: argv.arg("--save"),
        a_init_amp: Some(LEGACY_A_INIT_AMP),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let argv = ArgView::new(&args);
    if argv.flag("-h") || argv.flag("--help") {
        usage();
        return Ok(());
    }
    eprintln!("train_grad_layer23 is deprecated; use train_grad_full --first-layer 23");
    full_driver::run(parse_config(&argv)?).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(extra: &[&str]) -> Vec<String> {
        let mut values = vec!["train_grad_layer23".to_string()];
        values.extend(extra.iter().map(ToString::to_string));
        values
    }

    #[test]
    fn defaults_map_exactly_to_legacy_driver_contract() {
        let values = args(&[]);
        let cfg = parse_config(&ArgView::new(&values)).unwrap();
        assert_eq!(cfg.model_dir, train_common::default_model_dir());
        assert_eq!(cfg.data_dir, PathBuf::from("data/lora-train"));
        assert_eq!(cfg.first_layer, 23);
        assert_eq!(cfg.steps, 25);
        assert_eq!(cfg.lr, 1e-3);
        assert_eq!(cfg.rank, 8);
        assert_eq!(cfg.alpha, 16.0);
        assert_eq!(cfg.seq_len_cap, 64);
        assert_eq!(cfg.max_train, 3);
        assert_eq!(cfg.max_valid, 0);
        assert_eq!(cfg.log_every, 5);
        assert!(!cfg.gradcheck);
        assert!(!cfg.gradcheck_strided_probes);
        assert_eq!(cfg.probe, 6);
        assert_eq!(cfg.fd_eps, 4e-3);
        assert_eq!(cfg.save_path, None);
        assert_eq!(cfg.a_init_amp, Some(0.02));
    }

    #[test]
    fn full_only_flags_cannot_change_compatibility_semantics() {
        let values = args(&[
            "--first-layer",
            "0",
            "--max-valid",
            "99",
            "--gradcheck",
            "--probe",
            "1",
            "--fd-eps",
            "1",
        ]);
        let cfg = parse_config(&ArgView::new(&values)).unwrap();
        assert_eq!(cfg.first_layer, 23);
        assert_eq!(cfg.max_valid, 0);
        assert!(!cfg.gradcheck);
        assert_eq!(cfg.probe, 6);
        assert_eq!(cfg.fd_eps, 4e-3);
    }

    #[test]
    fn explicit_legacy_flags_override_defaults() {
        let values = args(&[
            "--model-dir",
            "/tmp/m",
            "--data-dir",
            "/tmp/d",
            "--steps",
            "7",
            "--lr",
            "3e-4",
            "--rank",
            "2",
            "--alpha",
            "4",
            "--seq-len",
            "16",
            "--max-train",
            "1",
            "--log-every",
            "2",
            "--save",
            "/tmp/out.safetensors",
        ]);
        let cfg = parse_config(&ArgView::new(&values)).unwrap();
        assert_eq!(cfg.model_dir, PathBuf::from("/tmp/m"));
        assert_eq!(cfg.data_dir, PathBuf::from("/tmp/d"));
        assert_eq!(cfg.steps, 7);
        assert_eq!(cfg.lr, 3e-4);
        assert_eq!(cfg.rank, 2);
        assert_eq!(cfg.alpha, 4.0);
        assert_eq!(cfg.seq_len_cap, 16);
        assert_eq!(cfg.max_train, 1);
        assert_eq!(cfg.log_every, 2);
        assert_eq!(cfg.save_path, Some("/tmp/out.safetensors".to_string()));
    }

    #[test]
    fn log_every_zero_is_rejected_before_driver() {
        let values = args(&["--log-every", "0"]);
        let err = parse_config(&ArgView::new(&values)).unwrap_err();
        assert!(err.contains("--log-every must be >= 1"));
    }
}
