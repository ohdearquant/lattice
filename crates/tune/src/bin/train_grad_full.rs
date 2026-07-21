// Exact multi-layer LoRA trainer — see docs/design.md (§train_grad_full details).

use std::path::PathBuf;

use lattice_tune::train_support as train_common;
use train_common::ArgView;
use train_common::full_driver::{self, FullDriverConfig};

fn usage() {
    eprintln!(
        "usage: train_grad_full [OPTIONS]

Options:
  --model-dir   <PATH>   Model directory (default: $HOME/.lattice/models/qwen3.5-0.8b)
  --data-dir    <PATH>   Dataset directory with train.jsonl + valid.jsonl (default: data/lora-train)
  --first-layer <N>      First materialised (trained) layer (default: 19)
  --steps       <N>      Adam steps (default: 25)
  --lr          <F>      Learning rate (default: 1e-3)
  --rank        <N>      LoRA rank (default: 8)
  --alpha       <F>      LoRA alpha (default: 16.0)
  --seq-len     <N>      Max tokens per sample (default: 64)
  --max-train   <N>      Training samples cap (default: 3)
  --max-valid   <N>      Held-out valid.jsonl samples for eval, 0=off (default: 16)
  --log-every   <N>      Print NLL every N steps (default: 5)
  --save        <PATH>   Save trained adapter as a PEFT safetensors file (requires --features safetensors)
  --gradcheck            Run finite-difference gradcheck instead of training
  --probe       <N>      Gradcheck entries probed per array per layer (default: 6)
  --fd-eps      <F>      Gradcheck central-difference step (default: 4e-3)
  -h, --help             Print this help"
    );
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
        first_layer: argv
            .arg("--first-layer")
            .and_then(|s| s.parse().ok())
            .unwrap_or(19),
        steps: argv
            .arg("--steps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(25),
        lr: argv
            .arg("--lr")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-3),
        rank: argv.arg("--rank").and_then(|s| s.parse().ok()).unwrap_or(8),
        alpha: argv
            .arg("--alpha")
            .and_then(|s| s.parse().ok())
            .unwrap_or(16.0),
        seq_len_cap: argv
            .arg("--seq-len")
            .and_then(|s| s.parse().ok())
            .unwrap_or(64),
        max_train: argv
            .arg("--max-train")
            .and_then(|s| s.parse().ok())
            .unwrap_or(3),
        max_valid: argv
            .arg("--max-valid")
            .and_then(|s| s.parse().ok())
            .unwrap_or(16),
        log_every,
        gradcheck: argv.flag("--gradcheck"),
        gradcheck_strided_probes: true,
        probe: argv
            .arg("--probe")
            .and_then(|s| s.parse().ok())
            .unwrap_or(6),
        fd_eps: argv
            .arg("--fd-eps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(4e-3),
        save_path: argv.arg("--save"),
        a_init_amp: None,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let argv = ArgView::new(&args);
    if argv.flag("-h") || argv.flag("--help") {
        usage();
        return Ok(());
    }
    full_driver::run(parse_config(&argv)?).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(extra: &[&str]) -> Vec<String> {
        let mut values = vec!["train_grad_full".to_string()];
        values.extend(extra.iter().map(ToString::to_string));
        values
    }

    #[test]
    fn defaults_match_documented_table() {
        let values = args(&[]);
        let cfg = parse_config(&ArgView::new(&values)).unwrap();
        assert_eq!(cfg.data_dir, PathBuf::from("data/lora-train"));
        assert_eq!(cfg.first_layer, 19);
        assert_eq!(cfg.steps, 25);
        assert_eq!(cfg.lr, 1e-3);
        assert_eq!(cfg.rank, 8);
        assert_eq!(cfg.alpha, 16.0);
        assert_eq!(cfg.seq_len_cap, 64);
        assert_eq!(cfg.max_train, 3);
        assert_eq!(cfg.max_valid, 16);
        assert_eq!(cfg.log_every, 5);
        assert!(!cfg.gradcheck);
        assert!(cfg.gradcheck_strided_probes);
        assert_eq!(cfg.probe, 6);
        assert_eq!(cfg.fd_eps, 4e-3);
        assert_eq!(cfg.save_path, None);
        assert_eq!(cfg.a_init_amp, None);
        assert_eq!(cfg.model_dir, train_common::default_model_dir());
    }

    #[test]
    fn explicit_flags_override_defaults() {
        let values = args(&[
            "--model-dir",
            "/tmp/m",
            "--data-dir",
            "/tmp/d",
            "--first-layer",
            "20",
            "--steps",
            "9",
            "--lr",
            "5e-4",
            "--rank",
            "16",
            "--alpha",
            "32",
            "--seq-len",
            "48",
            "--max-train",
            "1",
            "--max-valid",
            "0",
            "--log-every",
            "3",
            "--gradcheck",
            "--probe",
            "2",
            "--fd-eps",
            "1e-3",
            "--save",
            "/tmp/out.safetensors",
        ]);
        let cfg = parse_config(&ArgView::new(&values)).unwrap();
        assert_eq!(cfg.model_dir, PathBuf::from("/tmp/m"));
        assert_eq!(cfg.data_dir, PathBuf::from("/tmp/d"));
        assert_eq!(cfg.first_layer, 20);
        assert_eq!(cfg.steps, 9);
        assert_eq!(cfg.lr, 5e-4);
        assert_eq!(cfg.rank, 16);
        assert_eq!(cfg.alpha, 32.0);
        assert_eq!(cfg.seq_len_cap, 48);
        assert_eq!(cfg.max_train, 1);
        assert_eq!(cfg.max_valid, 0);
        assert_eq!(cfg.log_every, 3);
        assert!(cfg.gradcheck);
        assert_eq!(cfg.probe, 2);
        assert_eq!(cfg.fd_eps, 1e-3);
        assert_eq!(cfg.save_path, Some("/tmp/out.safetensors".to_string()));
    }

    #[test]
    fn log_every_zero_is_rejected() {
        let values = args(&["--log-every", "0"]);
        let err = parse_config(&ArgView::new(&values)).unwrap_err();
        assert!(err.contains("--log-every must be >= 1"));
    }
}
