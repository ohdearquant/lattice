use std::path::PathBuf;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use serde::Serialize;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use std::{collections::BTreeMap, path::Path};

#[derive(clap::Args, Debug)]
pub(crate) struct Args {
    /// Q4 model directory produced by `quantize_q4` or `quantize_quarot`.
    #[arg(long)]
    pub(crate) q4_dir: PathBuf,
    /// Directory containing tokenizer.json for the source model.
    #[arg(long)]
    pub(crate) tokenizer_dir: PathBuf,
    /// UTF-8 corpus used to score layer importance.
    #[arg(long)]
    pub(crate) calibration_corpus: PathBuf,
    /// Separate UTF-8 corpus used for the held-out perplexity gate.
    #[arg(long)]
    pub(crate) validation_corpus: PathBuf,
    /// Number of layers to remove in the candidate plan.
    #[arg(
        long,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pub(crate) prune_layers: usize,
    /// Path for the JSON pruning-plan artifact.
    #[arg(long, default_value = "lattice_pruning.json")]
    pub(crate) output: PathBuf,
    /// Tokens per calibration sequence passed to the layer scorer.
    #[arg(
        long,
        default_value = "128",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pub(crate) calibration_sequence_length: usize,
    /// Maximum calibration tokens; zero uses the full corpus.
    #[arg(long, default_value = "8192")]
    pub(crate) max_calibration_tokens: usize,
    /// Maximum validation tokens; zero uses the full corpus.
    #[arg(long, default_value = "0")]
    pub(crate) max_validation_tokens: usize,
    /// Perplexity window size.
    #[arg(
        long,
        default_value = "512",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(2..)
    )]
    pub(crate) window: usize,
    /// Perplexity stride.
    #[arg(
        long,
        default_value = "256",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pub(crate) stride: usize,
    /// Largest accepted increase in held-out perplexity.
    #[arg(long, default_value = "0.3")]
    pub(crate) max_delta_ppl: f64,
    /// Metal KV-cache capacity.
    #[arg(
        long,
        default_value = "4096",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pub(crate) max_cache_len: usize,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Serialize)]
struct PrunePlanArtifact {
    schema_version: u32,
    method: &'static str,
    source_model: SourceModelArtifact,
    calibration: CorpusArtifact,
    removed_layers: Vec<usize>,
    ffn_keep_indices: BTreeMap<usize, Vec<usize>>,
    attention_group_keep_indices: BTreeMap<usize, Vec<usize>>,
    residual_width: Option<usize>,
    pca_rotations: Option<Vec<serde_json::Value>>,
    constraints: PruneConstraintsArtifact,
    metrics: PplResultArtifact,
    scores: Vec<LayerScoreArtifact>,
    recommended_mask: Vec<bool>,
    warning: Option<String>,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Serialize)]
struct SourceModelArtifact {
    path: String,
    sha256: String,
    files_hashed: usize,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Serialize)]
struct CorpusArtifact {
    path: String,
    sha256: String,
    tokens: usize,
    sequence_length: usize,
    sequences: usize,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Serialize)]
struct PruneConstraintsArtifact {
    preserve_rope_pairs: bool,
    preserve_gqa_grouping: bool,
    min_full_attention_layers_per_group: usize,
    protect_first_n_layers: usize,
    protect_last_n_layers: usize,
    hidden_dim_multiple: usize,
    ffn_dim_multiple: usize,
    max_delta_ppl: Option<f64>,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Serialize)]
struct PplResultArtifact {
    validation_corpus: CorpusArtifact,
    dense_ppl: f64,
    pruned_ppl: f64,
    delta_ppl: f64,
    tokens_scored: usize,
    passed: bool,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Serialize)]
struct LayerScoreArtifact {
    layer_idx: usize,
    layer_type: &'static str,
    mean_cosine: f32,
    importance: f32,
}

pub(crate) fn run(args: &Args) -> Result<bool, String> {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        run_metal(args)
    }
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        let _ = args;
        Err("prune-score requires macOS + metal-gpu feature".to_string())
    }
}

#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
fn calibration_sequences(tokens: &[u32], sequence_length: usize) -> Result<Vec<Vec<u32>>, String> {
    if sequence_length == 0 {
        return Err("--calibration-sequence-length must be greater than zero".to_string());
    }
    if tokens.is_empty() {
        return Err("calibration corpus produced no tokens".to_string());
    }
    Ok(tokens
        .chunks(sequence_length)
        .map(<[u32]>::to_vec)
        .collect())
}

#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
fn ppl_gate_passed(dense_ppl: f64, pruned_ppl: f64, max_delta_ppl: f64) -> bool {
    dense_ppl.is_finite()
        && pruned_ppl.is_finite()
        && max_delta_ppl.is_finite()
        && pruned_ppl <= dense_ppl + max_delta_ppl
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_metal(args: &Args) -> Result<bool, String> {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::PerplexityConfig;
    use lattice_inference::model::qwen35_config::{LayerType, Qwen35Config};
    use lattice_inference::tokenizer::bpe::BpeTokenizer;

    validate_args(args)?;

    let tokenizer_path = args.tokenizer_dir.join("tokenizer.json");
    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)
        .map_err(|e| format!("failed to load tokenizer {}: {e}", tokenizer_path.display()))?;
    let calibration = tokenize_corpus(
        &tokenizer,
        &args.calibration_corpus,
        args.max_calibration_tokens,
        1,
    )?;
    let validation = tokenize_corpus(
        &tokenizer,
        &args.validation_corpus,
        args.max_validation_tokens,
        2,
    )?;
    let calibration_prompts =
        calibration_sequences(&calibration.tokens, args.calibration_sequence_length)?;

    let cfg = Qwen35Config::from_model_dir(&args.q4_dir).map_err(|e| e.to_string())?;
    validate_model_args(args, &cfg)?;
    eprintln!(
        "Calibration: {} tokens in {} sequences; validation: {} tokens",
        calibration.tokens.len(),
        calibration_prompts.len(),
        validation.tokens.len()
    );

    let ppl_cfg = PerplexityConfig {
        window: args.window,
        stride: args.stride,
    };
    let mut dense_state =
        MetalQwen35State::from_q4_dir(&args.q4_dir, &tokenizer_path, &cfg, args.max_cache_len)
            .map_err(|e| format!("failed to load dense model: {e}"))?;
    let dense_report = dense_state
        .compute_perplexity(&validation.tokens, &ppl_cfg)
        .map_err(|e| format!("dense perplexity evaluation failed: {e}"))?;
    let plan = dense_state
        .score_layer_importance(&calibration_prompts, args.prune_layers)
        .map_err(|e| format!("layer-importance scoring failed: {e}"))?;
    drop(dense_state);

    let removed_layers: Vec<usize> = plan
        .recommended_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, &keep)| (!keep).then_some(idx))
        .collect();
    if removed_layers.len() != args.prune_layers {
        return Err(format!(
            "scorer selected {} layers, but --prune-layers requested {}",
            removed_layers.len(),
            args.prune_layers
        ));
    }

    let mut pruned_cfg = cfg.clone();
    pruned_cfg.apply_layer_mask(plan.recommended_mask.clone());
    let mut pruned_state = MetalQwen35State::from_q4_dir(
        &args.q4_dir,
        &tokenizer_path,
        &pruned_cfg,
        args.max_cache_len,
    )
    .map_err(|e| format!("failed to load pruned model: {e}"))?;
    let pruned_report = pruned_state
        .compute_perplexity(&validation.tokens, &ppl_cfg)
        .map_err(|e| format!("pruned perplexity evaluation failed: {e}"))?;

    let delta_ppl = pruned_report.ppl - dense_report.ppl;
    let passed = ppl_gate_passed(dense_report.ppl, pruned_report.ppl, args.max_delta_ppl);
    let (model_sha256, files_hashed) = model_checksum(&args.q4_dir)?;
    let scores = plan
        .scores
        .iter()
        .map(|score| LayerScoreArtifact {
            layer_idx: score.layer_idx,
            layer_type: match score.layer_type {
                LayerType::LinearAttention => "linear_attention",
                LayerType::FullAttention => "full_attention",
            },
            mean_cosine: score.mean_cosine,
            importance: score.importance,
        })
        .collect();
    let artifact = PrunePlanArtifact {
        schema_version: 1,
        method: "block_influence",
        source_model: SourceModelArtifact {
            path: args.q4_dir.display().to_string(),
            sha256: model_sha256,
            files_hashed,
        },
        calibration: CorpusArtifact {
            path: args.calibration_corpus.display().to_string(),
            sha256: calibration.sha256,
            tokens: calibration.tokens.len(),
            sequence_length: args.calibration_sequence_length,
            sequences: calibration_prompts.len(),
        },
        removed_layers,
        ffn_keep_indices: BTreeMap::new(),
        attention_group_keep_indices: BTreeMap::new(),
        residual_width: None,
        pca_rotations: None,
        constraints: PruneConstraintsArtifact {
            preserve_rope_pairs: true,
            preserve_gqa_grouping: true,
            min_full_attention_layers_per_group: 0,
            protect_first_n_layers: 0,
            protect_last_n_layers: 0,
            hidden_dim_multiple: 1,
            ffn_dim_multiple: 1,
            max_delta_ppl: Some(args.max_delta_ppl),
        },
        metrics: PplResultArtifact {
            validation_corpus: CorpusArtifact {
                path: args.validation_corpus.display().to_string(),
                sha256: validation.sha256,
                tokens: validation.tokens.len(),
                sequence_length: args.window,
                sequences: pruned_report.num_windows,
            },
            dense_ppl: dense_report.ppl,
            pruned_ppl: pruned_report.ppl,
            delta_ppl,
            tokens_scored: pruned_report.num_tokens_scored,
            passed,
        },
        scores,
        recommended_mask: plan.recommended_mask,
        warning: plan.warning,
    };
    write_artifact(&args.output, &artifact)?;
    print_result(&artifact, &args.output);
    Ok(passed)
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
struct TokenizedCorpus {
    tokens: Vec<u32>,
    sha256: String,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn tokenize_corpus(
    tokenizer: &lattice_inference::tokenizer::bpe::BpeTokenizer,
    path: &Path,
    max_tokens: usize,
    min_tokens: usize,
) -> Result<TokenizedCorpus, String> {
    use lattice_inference::tokenizer::common::Tokenizer;
    use sha2::{Digest, Sha256};

    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read corpus {}: {e}", path.display()))?;
    let sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
    let uncapped = tokenizer.with_max_seq_len(text.len().saturating_add(64));
    let tokenized = uncapped.tokenize(&text);
    let mut tokens = tokenized.input_ids[..tokenized.real_length].to_vec();
    if max_tokens > 0 && tokens.len() > max_tokens {
        tokens.truncate(max_tokens);
    }
    if tokens.len() < min_tokens {
        return Err(format!(
            "corpus {} produced {} tokens; at least {min_tokens} required",
            path.display(),
            tokens.len()
        ));
    }
    Ok(TokenizedCorpus { tokens, sha256 })
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn validate_args(args: &Args) -> Result<(), String> {
    if !args.max_delta_ppl.is_finite() || args.max_delta_ppl < 0.0 {
        return Err("--max-delta-ppl must be a finite, non-negative number".to_string());
    }
    if args.stride >= args.window {
        return Err("--stride must be smaller than --window".to_string());
    }
    if args.calibration_sequence_length > args.max_cache_len || args.window > args.max_cache_len {
        return Err(
            "--max-cache-len must cover both --calibration-sequence-length and --window"
                .to_string(),
        );
    }
    let calibration = std::fs::canonicalize(&args.calibration_corpus).map_err(|e| {
        format!(
            "failed to resolve calibration corpus {}: {e}",
            args.calibration_corpus.display()
        )
    })?;
    let validation = std::fs::canonicalize(&args.validation_corpus).map_err(|e| {
        format!(
            "failed to resolve validation corpus {}: {e}",
            args.validation_corpus.display()
        )
    })?;
    if calibration == validation {
        return Err("calibration and validation corpora must be separate files".to_string());
    }
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn validate_model_args(
    args: &Args,
    cfg: &lattice_inference::model::qwen35_config::Qwen35Config,
) -> Result<(), String> {
    if args.prune_layers >= cfg.num_active_layers() {
        return Err(format!(
            "--prune-layers ({}) must be smaller than the active layer count ({})",
            args.prune_layers,
            cfg.num_active_layers()
        ));
    }
    if args.max_cache_len > cfg.max_position_embeddings {
        return Err(format!(
            "--max-cache-len ({}) exceeds the model context limit ({})",
            args.max_cache_len, cfg.max_position_embeddings
        ));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn model_checksum(dir: &Path) -> Result<(String, usize), String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read model directory {}: {e}", dir.display()))?
    {
        let entry = entry.map_err(|e| format!("failed to read model directory entry: {e}"))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let include = matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("q4" | "q3" | "f16")
        ) || matches!(
            path.file_name().and_then(|name| name.to_str()),
            Some("config.json" | "quantize_index.json")
        );
        if include {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(format!(
            "model directory {} has no Q4 model artifacts to checksum",
            dir.display()
        ));
    }

    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    for path in &files {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("model artifact path is not valid UTF-8: {}", path.display()))?;
        hasher.update(name.len().to_le_bytes());
        hasher.update(name.as_bytes());
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("failed to open model artifact {}: {e}", path.display()))?;
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|e| format!("failed to hash model artifact {}: {e}", path.display()))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }
    Ok((format!("{:x}", hasher.finalize()), files.len()))
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn write_artifact(path: &Path, artifact: &PrunePlanArtifact) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(artifact)
        .map_err(|e| format!("failed to serialize pruning plan: {e}"))?;
    std::fs::write(path, json)
        .map_err(|e| format!("failed to write pruning plan {}: {e}", path.display()))
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn print_result(artifact: &PrunePlanArtifact, output: &Path) {
    println!("Layer importance scores (lowest importance = best prune candidate):");
    for score in &artifact.scores {
        println!(
            "layer {:>3}  {:<16} cosine={:.6} importance={:.6}",
            score.layer_idx, score.layer_type, score.mean_cosine, score.importance
        );
    }
    println!("Removed layers: {:?}", artifact.removed_layers);
    println!("Dense PPL:      {:.6}", artifact.metrics.dense_ppl);
    println!("Pruned PPL:     {:.6}", artifact.metrics.pruned_ppl);
    println!("PPL delta:      {:+.6}", artifact.metrics.delta_ppl);
    println!(
        "Gate:           {}",
        if artifact.metrics.passed {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!("Pruning plan:   {}", output.display());
    if let Some(warning) = &artifact.warning {
        eprintln!("Warning: {warning}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_parses_required_corpora_and_conservative_defaults() {
        let cli = crate::Cli::try_parse_from([
            "lattice",
            "prune-score",
            "--q4-dir",
            "model-q4",
            "--tokenizer-dir",
            "model",
            "--calibration-corpus",
            "calibration.txt",
            "--validation-corpus",
            "validation.txt",
            "--prune-layers",
            "4",
        ])
        .expect("valid prune-score command");

        let crate::Command::PruneScore { args } = cli.command else {
            panic!("expected prune-score command");
        };
        assert_eq!(args.output, PathBuf::from("lattice_pruning.json"));
        assert_eq!(args.calibration_sequence_length, 128);
        assert_eq!(args.max_calibration_tokens, 8192);
        assert_eq!(args.max_delta_ppl, 0.3);
    }

    #[test]
    fn calibration_sequences_use_all_tokens_without_empty_tail() {
        let tokens: Vec<u32> = (0..10).collect();
        assert_eq!(
            calibration_sequences(&tokens, 4).expect("valid sequence length"),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]]
        );
    }

    #[test]
    fn calibration_sequences_reject_zero_length() {
        assert!(calibration_sequences(&[1, 2], 0).is_err());
    }

    #[test]
    fn ppl_gate_accepts_delta_at_threshold() {
        assert!(ppl_gate_passed(12.0, 12.3, 0.3));
        assert!(!ppl_gate_passed(12.0, 12.300_001, 0.3));
        assert!(!ppl_gate_passed(f64::INFINITY, f64::INFINITY, 0.3));
    }
}
