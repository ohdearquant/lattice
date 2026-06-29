//! # Router Online-Learning Convergence Bench
//!
//! 4-algorithm × 3-regime head-to-head comparison for the selector-gate
//! policy-gradient trainer.  Plain `fn main()` binary (`harness = false`).
//! Prints per-regime convergence tables and a summary matrix.
//!
//! ## Algorithms
//!
//! - **A — oracle ceiling**: trainer receives the true-best adapter index as
//!   target each round (upper bound on achievable accuracy).
//! - **B — M=1 reward-on-served**: on-policy categorical sample; `+1` if
//!   served adapter matches true best, `−1` otherwise.
//! - **C — phase-2 positive-only**: on-policy sample via `rloo_step`; only
//!   confirmed-positive events are consumed (negative events are dropped).
//! - **D — hybrid**: routes positive events through the multi-sample update,
//!   negative events through the single-sample update (no events dropped).
//!
//! ## Regimes
//!
//! - **dense**: every round produces a feedback event (800 rounds).
//! - **noisy**: every round, 20 % of events are corrupted (800 rounds).
//! - **sparse**: 15 % feedback probability, 1 600 rounds.
//!
//! ## DEFERRED — not in this file
//!
//! - **Bench 3 — adapter routing latency** (`bench_router_latency.rs`):
//!   depends on the adapter-blend API (LoRA mixture path) which lives on a
//!   different branch.  Deferred until that branch lands on main.
//!
//! - **Bench 4 — EWC++ Fisher stability under distribution shift**: requires
//!   composing `DiagonalFisher` penalty gradients between gradient-compute and
//!   weight-apply — a seam that `RlooTrainer::step` does not currently expose
//!   (it applies weights internally and returns only the scalar loss).
//!   Deferred pending a public `GradientDelta` return value from `step`.

use lattice_fann::training::{RlooConfig, RlooTrainer};
use lattice_fann::{Activation, Network, NetworkBuilder};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Task constants
// ---------------------------------------------------------------------------

/// Number of adapters (selector output dimension).
const K: usize = 4;
/// Context dimension (selector input dimension).
const D: usize = 16;
/// Held-out evaluation set size — generated ONCE, never trained on.
const N_EVAL: usize = 200;
/// Gaussian noise scale for class-conditioned context generation.
const NOISE: f32 = 0.5;

// Checkpoint schedules.
const DENSE_NOISY_CPS: &[usize] = &[0, 100, 400, 800];
const SPARSE_CPS: &[usize] = &[0, 200, 800, 1600];

// ---------------------------------------------------------------------------
// Seeds — all printed at runtime
// ---------------------------------------------------------------------------

/// Seed for the K ground-truth unit weight vectors.
const WEIGHT_SEED: u64 = 42;
/// Seed for selector-gate initialisation (shared across all arms).
const GATE_SEED: u64 = 7;
/// Seed for eval-set class draws.
const EVAL_CLASS_SEED: u64 = 200;
/// Seed for eval-set Gaussian noise draws.
const EVAL_NOISE_SEED: u64 = 201;
/// Seed for training class draws (per arm, reseeded).
const TRAIN_CLASS_SEED: u64 = 100;
/// Seed for training Gaussian context noise (per arm, reseeded).
const CTX_NOISE_SEED: u64 = 101;
/// Seed for categorical action sampling — B and C only (per arm, reseeded).
const ACTION_SEED: u64 = 400;
/// Seed for reward/label noise draws — noisy regime only (per arm, reseeded).
const REWARD_NOISE_SEED: u64 = 500;
/// Seed for sparse feedback gate draws — sparse regime only (per arm, reseeded).
const SPARSE_SEED: u64 = 600;
/// Seed for `RlooTrainer`'s internal RNG (Phase-2 Gumbel path).
const TRAINER_SEED: u64 = 300;

// ---------------------------------------------------------------------------
// Algorithm / regime tags
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Algorithm {
    OracleCeiling,
    M1RewardOnServed,
    Phase2PositiveOnly,
    /// Routes positive events through the multi-sample update, negative events
    /// through the single-sample update.
    Hybrid,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Regime {
    Dense,
    Noisy,
    Sparse,
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct CheckpointRow {
    t: usize,
    accuracy: f32,
    entropy_bits: f32,
}

struct ArmResult {
    rows: Vec<CheckpointRow>,
    effective_updates: usize,
    /// Positive-only events that were discarded (Phase-2 only; 0 for A and B).
    dropped: usize,
}

// ---------------------------------------------------------------------------
// Pure helpers (no unwrap / panic)
// ---------------------------------------------------------------------------

/// Dot product of two equal-length slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Numerically stable softmax (local copy — private in rloo.rs).
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&s| (s - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Index of the largest value; returns 0 on empty slice.
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Inverse-CDF categorical sample; falls back to last index on edge cases.
fn sample_categorical(probs: &[f32], rng: &mut SmallRng) -> usize {
    let u: f32 = rng.gen_range(0.0_f32..1.0_f32);
    let mut cumsum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u <= cumsum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Ground-truth adapter index: `argmax_k dot(ctx, weights[k])`.
fn true_best(ctx: &[f32], weights: &[Vec<f32>]) -> usize {
    (0..weights.len())
        .max_by(|&a, &b| {
            dot(ctx, &weights[a])
                .partial_cmp(&dot(ctx, &weights[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0)
}

/// Box-Muller transform: one standard-normal variate from two uniform draws.
fn gaussian_f32(rng: &mut SmallRng) -> f32 {
    let u1: f32 = rng.r#gen::<f32>().clamp(1e-38_f32, 1.0 - f32::EPSILON);
    let u2: f32 = rng.r#gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Class-conditioned context: `w[class] + NOISE * g`, `g ~ N(0, I_D)`.
///
/// With unit weight vectors and NOISE=0.5, `argmax_k dot(ctx, w_k)` reliably
/// recovers `class` — the task is crisply learnable.
fn gen_context(class: usize, weights: &[Vec<f32>], noise_rng: &mut SmallRng) -> Vec<f32> {
    weights[class]
        .iter()
        .map(|&w| w + NOISE * gaussian_f32(noise_rng))
        .collect()
}

/// Greedy evaluation on the held-out set.
///
/// Returns `(accuracy, entropy_bits)`:
/// - **accuracy**: fraction where `argmax(logits) == true_best`.
/// - **entropy_bits**: Shannon entropy H = −Σ f_k log₂(f_k) of the empirical
///   adapter-selection distribution across `N_EVAL` contexts.
fn eval_metrics(
    gate: &mut Network,
    eval_contexts: &[Vec<f32>],
    eval_true_best: &[usize],
) -> Result<(f32, f32), Box<dyn std::error::Error>> {
    let mut correct = 0_usize;
    let mut counts = vec![0_usize; K];

    for (ctx, &best) in eval_contexts.iter().zip(eval_true_best.iter()) {
        let logits = gate.forward(ctx.as_slice())?.to_vec();
        let selected = argmax(&logits);
        if selected == best {
            correct += 1;
        }
        counts[selected] += 1;
    }

    let accuracy = correct as f32 / N_EVAL as f32;
    let n = N_EVAL as f32;
    let mut entropy = 0.0_f32;
    for &c in &counts {
        if c > 0 {
            let p = c as f32 / n;
            entropy -= p * p.log2();
        }
    }
    Ok((accuracy, entropy))
}

// ---------------------------------------------------------------------------
// Training arm
// ---------------------------------------------------------------------------

/// Run one `(algorithm, regime)` arm end-to-end.
///
/// Every arm receives freshly reseeded independent RNG streams so the
/// sparse/noise masks are reproducible and comparable across algorithms within
/// the same regime.
#[allow(clippy::too_many_arguments)]
fn run_arm(
    weights: &[Vec<f32>],
    eval_contexts: &[Vec<f32>],
    eval_true_best: &[usize],
    algo: Algorithm,
    regime: Regime,
    config: RlooConfig,
    rounds: usize,
    checkpoints: &[usize],
) -> Result<ArmResult, Box<dyn std::error::Error>> {
    let mut gate = NetworkBuilder::new()
        .input(D)
        .hidden(16, Activation::ReLU)
        .output(K, Activation::Linear)
        .build_with_seed(GATE_SEED)?;

    let mut trainer = RlooTrainer::with_seed(config, TRAINER_SEED);

    // Per-arm independent streams reseeded from constants.
    let mut class_rng = SmallRng::seed_from_u64(TRAIN_CLASS_SEED);
    let mut ctx_noise_rng = SmallRng::seed_from_u64(CTX_NOISE_SEED);
    let mut action_rng = SmallRng::seed_from_u64(ACTION_SEED);
    let mut reward_noise_rng = SmallRng::seed_from_u64(REWARD_NOISE_SEED);
    let mut sparse_rng = SmallRng::seed_from_u64(SPARSE_SEED);

    let mut effective_updates = 0_usize;
    let mut dropped = 0_usize;
    let mut rows: Vec<CheckpointRow> = Vec::new();

    // T=0: evaluate before any training (sanity / leakage check).
    if checkpoints.contains(&0) {
        let (accuracy, entropy_bits) = eval_metrics(&mut gate, eval_contexts, eval_true_best)?;
        rows.push(CheckpointRow {
            t: 0,
            accuracy,
            entropy_bits,
        });
    }

    for round in 1..=rounds {
        // Step 1: draw class, build class-conditioned context.
        let c = class_rng.gen_range(0..K);
        let ctx = gen_context(c, weights, &mut ctx_noise_rng);
        let best = true_best(&ctx, weights);

        // Step 2: feedback gate (sparse draws happen every round in sparse regime).
        let give_feedback = if regime == Regime::Sparse {
            sparse_rng.r#gen::<f32>() < 0.15
        } else {
            true
        };

        // Step 3: algorithm branch.
        //
        // RNG draw order within each algorithm is kept consistent so that
        // the sparse/noise masks align across A/B/C within the same regime.
        // action_rng is NOT drawn for A (oracle has no exploration step).
        match algo {
            Algorithm::OracleCeiling => {
                if give_feedback {
                    let mut target = best;
                    if regime == Regime::Noisy && reward_noise_rng.r#gen::<f32>() < 0.20 {
                        // Corrupt: pick a uniformly-random index != best.
                        let mut wrong = reward_noise_rng.gen_range(0..(K - 1));
                        if wrong >= best {
                            wrong += 1;
                        }
                        target = wrong;
                    }
                    trainer.step(&mut gate, ctx.as_slice(), target, 1.0)?;
                    effective_updates += 1;
                }
            }

            Algorithm::M1RewardOnServed => {
                // On-policy sample (always — not gated by give_feedback).
                let logits = gate.forward(ctx.as_slice())?.to_vec();
                let served = sample_categorical(&softmax(&logits), &mut action_rng);
                if give_feedback {
                    let mut reward = if served == best { 1.0_f32 } else { -1.0_f32 };
                    if regime == Regime::Noisy && reward_noise_rng.r#gen::<f32>() < 0.20 {
                        reward = -reward;
                    }
                    trainer.step(&mut gate, ctx.as_slice(), served, reward)?;
                    effective_updates += 1;
                }
            }

            Algorithm::Phase2PositiveOnly => {
                // On-policy sample (always — not gated by give_feedback).
                let logits = gate.forward(ctx.as_slice())?.to_vec();
                let served = sample_categorical(&softmax(&logits), &mut action_rng);
                if give_feedback {
                    let mut positive = served == best;
                    if regime == Regime::Noisy && reward_noise_rng.r#gen::<f32>() < 0.20 {
                        positive = !positive;
                    }
                    if positive {
                        // k=1 (top-1 subset), M=6 Gumbel samples with LOO baseline.
                        trainer.rloo_step(&mut gate, ctx.as_slice(), served, 1, 6)?;
                        effective_updates += 1;
                    } else {
                        // No negative-event interface in Phase-2 — drop.
                        dropped += 1;
                    }
                }
            }

            Algorithm::Hybrid => {
                // On-policy sample (always — not gated by give_feedback).
                let logits = gate.forward(ctx.as_slice())?.to_vec();
                let served = sample_categorical(&softmax(&logits), &mut action_rng);
                if give_feedback {
                    let mut positive = served == best;
                    if regime == Regime::Noisy && reward_noise_rng.r#gen::<f32>() < 0.20 {
                        positive = !positive;
                    }
                    if positive {
                        // Multi-sample update: k=1 top-1 subset, M=6 Gumbel samples.
                        trainer.rloo_step(&mut gate, ctx.as_slice(), served, 1, 6)?;
                    } else {
                        // Single-sample update with negative reward.
                        trainer.step(&mut gate, ctx.as_slice(), served, -1.0)?;
                    }
                    effective_updates += 1;
                }
            }
        }

        // Step 4: checkpoint eval (greedy argmax on held-out set).
        if checkpoints.contains(&round) {
            let (accuracy, entropy_bits) = eval_metrics(&mut gate, eval_contexts, eval_true_best)?;
            rows.push(CheckpointRow {
                t: round,
                accuracy,
                entropy_bits,
            });
        }
    }

    Ok(ArmResult {
        rows,
        effective_updates,
        dropped,
    })
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

fn print_arm_table(title: &str, result: &ArmResult) {
    println!(
        "\n--- {title}  (eff_updates={}) ---",
        result.effective_updates
    );
    println!("{:>6} | {:>10} | {:>14}", "T", "accuracy", "entropy (bits)");
    println!("{}", "-".repeat(38));
    for row in &result.rows {
        println!(
            "{:>6} | {:>10.4} | {:>14.4}",
            row.t, row.accuracy, row.entropy_bits
        );
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Router Online-Learning 4×3 Head-to-Head Bench ===");
    println!(
        "seeds  weight:{WEIGHT_SEED}  gate:{GATE_SEED}  \
         eval_class:{EVAL_CLASS_SEED}  eval_noise:{EVAL_NOISE_SEED}  \
         train_class:{TRAIN_CLASS_SEED}  ctx_noise:{CTX_NOISE_SEED}  \
         action:{ACTION_SEED}  reward_noise:{REWARD_NOISE_SEED}  \
         sparse:{SPARSE_SEED}  trainer:{TRAINER_SEED}"
    );
    println!(
        "K={K}  D={D}  N_eval={N_EVAL}  ctx_noise={NOISE}  \
         lr=0.05  aux_loss_coeff=0.01  z_loss_coeff=0.001"
    );

    // -------------------------------------------------------------------------
    // Ground-truth unit weight vectors (K × D).
    // Each drawn from Uniform(−1,1) then L2-normalised → near-orthogonal in R^16.
    // Routing rule: true_best(ctx) = argmax_k dot(ctx, w[k]).
    // -------------------------------------------------------------------------
    let mut weight_rng = SmallRng::seed_from_u64(WEIGHT_SEED);
    let weights: Vec<Vec<f32>> = (0..K)
        .map(|_| {
            let raw: Vec<f32> = (0..D)
                .map(|_| weight_rng.gen_range(-1.0_f32..1.0_f32))
                .collect();
            let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9_f32);
            raw.iter().map(|x| x / norm).collect()
        })
        .collect();

    // -------------------------------------------------------------------------
    // Held-out eval set: N_EVAL class-conditioned contexts.
    // Seeds (200, 201) are disjoint from training seeds (100, 101).
    // -------------------------------------------------------------------------
    let mut eval_class_rng = SmallRng::seed_from_u64(EVAL_CLASS_SEED);
    let mut eval_noise_rng = SmallRng::seed_from_u64(EVAL_NOISE_SEED);
    let eval_contexts: Vec<Vec<f32>> = (0..N_EVAL)
        .map(|_| {
            let c = eval_class_rng.gen_range(0..K);
            gen_context(c, &weights, &mut eval_noise_rng)
        })
        .collect();
    let eval_true_best: Vec<usize> = eval_contexts
        .iter()
        .map(|c| true_best(c, &weights))
        .collect();

    let config = RlooConfig {
        learning_rate: 0.05,
        aux_loss_coeff: 0.01,
        z_loss_coeff: 0.001,
    };

    let algos = [
        Algorithm::OracleCeiling,
        Algorithm::M1RewardOnServed,
        Algorithm::Phase2PositiveOnly,
        Algorithm::Hybrid,
    ];
    let algo_names = [
        "A oracle-ceiling",
        "B m1-reward-on-served",
        "C phase2-positive-only",
        "D hybrid",
    ];

    // (regime, label, rounds, checkpoints)
    let regime_configs: &[(Regime, &str, usize, &[usize])] = &[
        (Regime::Dense, "dense", 800, DENSE_NOISY_CPS),
        (Regime::Noisy, "noisy", 800, DENSE_NOISY_CPS),
        (Regime::Sparse, "sparse", 1600, SPARSE_CPS),
    ];

    // Run all 12 cells: [regime][algo].
    let mut all_results: Vec<Vec<ArmResult>> = Vec::new();

    for &(regime, regime_name, rounds, checkpoints) in regime_configs {
        println!("\n========== REGIME: {regime_name} ==========");
        let mut regime_results: Vec<ArmResult> = Vec::new();

        for (ai, &algo) in algos.iter().enumerate() {
            let result = run_arm(
                &weights,
                &eval_contexts,
                &eval_true_best,
                algo,
                regime,
                config.clone(),
                rounds,
                checkpoints,
            )?;

            let title = if algo == Algorithm::Phase2PositiveOnly {
                format!("{} (dropped={})", algo_names[ai], result.dropped)
            } else {
                algo_names[ai].to_string()
            };
            print_arm_table(&title, &result);
            regime_results.push(result);
        }

        all_results.push(regime_results);
    }

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("\n========== SUMMARY MATRIX ==========");

    // T=0 sanity from dense/A (all arms share the same initialisation).
    let t0_opt = all_results
        .first()
        .and_then(|r| r.first())
        .and_then(|a| a.rows.first());

    if let Some(t0) = t0_opt {
        println!(
            "\nT=0 SANITY: accuracy={:.4}  (expected ≈ {:.4} = 1/K)  \
             entropy={:.4} bits  (expected ≈ {:.4} = log2(K))",
            t0.accuracy,
            1.0_f32 / K as f32,
            t0.entropy_bits,
            (K as f32).log2()
        );
        if t0.accuracy > 0.40 {
            println!(
                "  *** LEAKAGE WARNING: T=0 accuracy {:.4} >> chance (0.25). \
                 Verify eval seeds ({EVAL_CLASS_SEED},{EVAL_NOISE_SEED}) != \
                 train seeds ({TRAIN_CLASS_SEED},{CTX_NOISE_SEED}). ***",
                t0.accuracy
            );
        } else {
            println!("  (T=0 looks clean — no leakage detected)");
        }
    }

    println!();
    println!(
        "{:<8} | {:>12} | {:>12} | {:>12} | {:>12} | {:>8} | {:>8} | {:>8} | {:>8}",
        "regime",
        "A_final_acc",
        "B_final_acc",
        "C_final_acc",
        "D_final_acc",
        "A_ent",
        "B_ent",
        "C_ent",
        "D_ent"
    );
    println!("{}", "-".repeat(112));

    let regime_labels = ["dense", "noisy", "sparse"];
    for (ri, regime_results) in all_results.iter().enumerate() {
        let final_acc: Vec<f32> = regime_results
            .iter()
            .map(|r| r.rows.last().map(|row| row.accuracy).unwrap_or(0.0))
            .collect();
        let final_ent: Vec<f32> = regime_results
            .iter()
            .map(|r| r.rows.last().map(|row| row.entropy_bits).unwrap_or(0.0))
            .collect();

        println!(
            "{:<8} | {:>12.4} | {:>12.4} | {:>12.4} | {:>12.4} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8.4}",
            regime_labels[ri],
            final_acc.first().copied().unwrap_or(0.0),
            final_acc.get(1).copied().unwrap_or(0.0),
            final_acc.get(2).copied().unwrap_or(0.0),
            final_acc.get(3).copied().unwrap_or(0.0),
            final_ent.first().copied().unwrap_or(0.0),
            final_ent.get(1).copied().unwrap_or(0.0),
            final_ent.get(2).copied().unwrap_or(0.0),
            final_ent.get(3).copied().unwrap_or(0.0),
        );
    }

    // Per-regime headline.
    println!();
    for (ri, regime_results) in all_results.iter().enumerate() {
        let a_acc = regime_results
            .first()
            .and_then(|r| r.rows.last())
            .map(|row| row.accuracy)
            .unwrap_or(0.0);
        let b_acc = regime_results
            .get(1)
            .and_then(|r| r.rows.last())
            .map(|row| row.accuracy)
            .unwrap_or(0.0);
        let c_acc = regime_results
            .get(2)
            .and_then(|r| r.rows.last())
            .map(|row| row.accuracy)
            .unwrap_or(0.0);
        let d_acc = regime_results
            .get(3)
            .and_then(|r| r.rows.last())
            .map(|row| row.accuracy)
            .unwrap_or(0.0);

        let b_pct_of_a = if a_acc > 0.0 {
            b_acc / a_acc * 100.0
        } else {
            0.0
        };
        let c_vs_b_pts = (c_acc - b_acc) * 100.0;
        let c_label = if c_vs_b_pts >= 0.0 { "beats" } else { "trails" };
        let d_vs_b_pts = (d_acc - b_acc) * 100.0;
        let d_label = if d_vs_b_pts >= 0.0 { "beats" } else { "trails" };

        println!(
            "{}: B reaches {:.1}% of A accuracy; C {} B by {:.1} pts; D {} B by {:.1} pts",
            regime_labels[ri],
            b_pct_of_a,
            c_label,
            c_vs_b_pts.abs(),
            d_label,
            d_vs_b_pts.abs()
        );
    }

    Ok(())
}
