//! Pooled hidden-state embeddings, CPU against Metal.

use super::*;
// -------------------------------------------------------------
// Pooled hidden-state embeddings: CPU versus Metal equivalence.
//
// WHAT THIS GATES. `Qwen35Model::embed_tokens` and
// `MetalQwen35State::embed_tokens` are documented to return the same
// quantity: the last position's hidden state AFTER the final RMSNorm
// and before the language-model head. On the Metal side that vector is
// copied out of `activations.hidden` by `encode_capture_final_hidden`,
// encoded immediately after the norm dispatch, at three separate
// forward paths. Nothing about "immediately after" is enforced by the
// type system, and the buffer sitting beside it — `pre_final_hidden`,
// which MTP owns — holds the SAME TOKEN AT A DIFFERENT POINT IN THE
// NETWORK. A capture placed one dispatch too early returns a
// well-formed, correctly-sized vector differing from the reference only
// by the norm's per-channel rescale. No smoke test catches that. This
// does.
//
// WHY THE SAME WEIGHTS ON BOTH SIDES. Both engines are built from one
// `ModelWeights` loaded from one safetensors checkpoint, so a
// disagreement is a difference in what the two paths compute rather
// than quantization error. Comparing the f16 CPU model against a Q4
// Metal artifact would measure quantization and could not resolve a
// layer-offset bug underneath it.
//
// FAIL-CLOSED. Without a checkpoint these print a skip line and return.
// With LATTICE_HIDDEN_PARITY_ENFORCE=1 a missing model panics instead,
// so a provisioning failure cannot masquerade as a passing gate.
// Mirrors LATTICE_Q4_COMPOSED_GATE_ENFORCE in
// tests/quarot_q4_composed_golden.rs.
//
// MUTATION EVIDENCE. A test that passes with the fix reverted is
// decoration. Moving any of the three `encode_capture_final_hidden`
// calls to before its `dispatch_rms_norm` must make this fail. That
// check is run by hand, since it means editing the source under test.
//
//   LATTICE_MODEL_DIR=~/.lattice/models/qwen3.5-0.8b \
//   cargo test --release -p lattice-inference --features "f16,metal-gpu" \
//       hidden_state_ -- --nocapture --test-threads=1
// -------------------------------------------------------------

/// Cosine floor. The two paths run different kernels over the same
/// weights, so f32 accumulation order differs and bit equality is not
/// the contract. It is far tighter than the gap a mislocated capture
/// opens: the final norm rescales per channel by (1 + gamma), so a
/// pre-norm vector sits nowhere near cosine 0.999 of a post-norm one on
/// a checkpoint whose gamma is not zero — which
/// `hidden_parity_model()` establishes for the checkpoint in hand
/// rather than assuming.
const HIDDEN_PARITY_MIN_COSINE: f32 = 0.999;

/// Token ids kept small so they are in vocabulary for any Qwen3.5
/// checkpoint, and varied so the sequence is not a repeated token
/// (which would make the attention pattern degenerate).
const HIDDEN_PARITY_MULTI: &[u32] = &[1, 25, 9, 314, 77, 2, 1041, 58];
const HIDDEN_PARITY_SINGLE: &[u32] = &[314];

fn hidden_parity_enforced() -> bool {
    matches!(
        std::env::var("LATTICE_HIDDEN_PARITY_ENFORCE").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// Load the CPU model, or report why there is nothing to compare.
/// Returns `None` only in the un-enforced, no-checkpoint case.
fn hidden_parity_model() -> Option<crate::model::qwen35::Qwen35Model> {
    let Some(dir) = hidden_parity_model_dir() else {
        assert!(
            !hidden_parity_enforced(),
            "LATTICE_HIDDEN_PARITY_ENFORCE=1 but no checkpoint found: set \
                     LATTICE_MODEL_DIR to a Qwen3.5 safetensors checkout. Refusing to \
                     report a pass for a comparison that never ran."
        );
        println!(
            "SKIP hidden-state parity: no Qwen3.5 checkpoint (set \
                     LATTICE_MODEL_DIR, or LATTICE_HIDDEN_PARITY_ENFORCE=1 to make \
                     this a failure)"
        );
        return None;
    };
    println!("hidden-state parity model: {}", dir.display());
    let model = crate::model::qwen35::Qwen35Model::from_safetensors(&dir)
        .expect("load CPU model from LATTICE_MODEL_DIR");

    // The control that keeps the comparison from being vacuous. If this
    // checkpoint's final norm were an identity, a capture taken before
    // it would equal one taken after, and every assertion below would
    // pass while proving nothing about where the capture sits. Qwen3.5
    // uses the (1 + gamma) convention, so identity is gamma == 0.
    let gamma = &model.weights().final_norm;
    let max_gamma = gamma.iter().fold(0.0f32, |m, g| m.max(g.abs()));
    println!(
        "control: final_norm has {} channels, max |gamma| = {max_gamma:.4}",
        gamma.len()
    );
    assert!(
        max_gamma > 1e-3,
        "final_norm is indistinguishable from an identity on this checkpoint \
                 (max |gamma| = {max_gamma}), so a pre-norm capture would score \
                 identically to a post-norm one and this test cannot discriminate. \
                 Use a different checkpoint rather than trusting the pass."
    );
    Some(model)
}

fn hidden_parity_cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        na > 0.0 && nb > 0.0,
        "a zero vector has no direction, so cosine is undefined rather than \
                 low; one side produced all zeros (|a|={na}, |b|={nb})"
    );
    dot / (na * nb)
}

/// Max minus min. A capture returning a zeroed or constant buffer would
/// otherwise satisfy every equality assertion in the test.
fn hidden_parity_spread(v: &[f32]) -> f32 {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &x in v {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    hi - lo
}

#[test]
fn hidden_state_cpu_and_metal_embed_tokens_agree_on_real_weights() {
    let _gpu = gpu_test_lock();
    let Some(model) = hidden_parity_model() else {
        return;
    };
    let cfg = model.config().clone();
    let mut metal = MetalQwen35State::new(model.weights(), &cfg, 4096)
        .expect("build Metal state from the same weights the CPU model holds");

    assert_eq!(
        model.hidden_size(),
        metal.hidden_size(),
        "the two paths disagree about hidden size before any vector is compared"
    );

    // Two sequences, because they take DIFFERENT forward paths on
    // Metal: a multi-token prompt goes through the batched prefill
    // chunk (capture site 2) and a single token goes through the decode
    // head (capture site 3). Testing only the multi-token case would
    // leave the single-token path uncovered while reading as coverage.
    for (label, tokens) in [
        ("multi-token (prefill path)", HIDDEN_PARITY_MULTI),
        ("single token (decode path)", HIDDEN_PARITY_SINGLE),
    ] {
        let cpu = model
            .embed_tokens(tokens, crate::model::qwen35::HiddenPooling::LastToken)
            .expect("CPU embed");
        let gpu = metal
            .embed_tokens(tokens, crate::model::qwen35::HiddenPooling::LastToken)
            .expect("Metal embed");

        assert_eq!(cpu.len(), model.hidden_size(), "{label}: CPU length");
        assert_eq!(gpu.len(), model.hidden_size(), "{label}: Metal length");

        let cpu_spread = hidden_parity_spread(&cpu);
        let gpu_spread = hidden_parity_spread(&gpu);
        assert!(
            cpu_spread > 1e-4 && gpu_spread > 1e-4,
            "{label}: a vector is constant or near-constant (cpu spread \
                     {cpu_spread}, metal spread {gpu_spread}); agreement between two \
                     flat vectors is not evidence"
        );

        let cos = hidden_parity_cosine(&cpu, &gpu);
        let mad = cpu
            .iter()
            .zip(&gpu)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!("{label}: cosine {cos:.6}, max abs diff {mad:.3e}, spread {cpu_spread:.3}");
        assert!(
            cos >= HIDDEN_PARITY_MIN_COSINE,
            "{label}: CPU and Metal disagree, cosine {cos} < \
                     {HIDDEN_PARITY_MIN_COSINE}. If the gap is a uniform per-channel \
                     rescale, the Metal capture is on the wrong side of the final RMSNorm."
        );
    }
}

/// Every final-RMSNorm dispatch is followed by a capture, enforced at
/// the source rather than by inspection.
///
/// The runtime parity test above covers the two forward paths the
/// public `embed_tokens` can reach: the batched prefill chunk and the
/// single-token decode head. It does NOT cover the MTP verify path,
/// which is reachable only through speculative decoding and has no CPU
/// counterpart to compare against — moving that site's capture to the
/// wrong side of its norm leaves the parity test green, which was
/// measured, not assumed. No runtime test can cover a *fourth* site
/// somebody adds next year either.
///
/// So the invariant is checked where it actually lives: in the text of
/// this file. Each dispatch against `engine.final_norm` must be
/// followed, within a short window, by an `encode_capture_final_hidden`
/// call. A new forward path that norms and forgets to capture fails
/// here with a line number instead of returning a stale vector at
/// runtime.
#[test]
fn hidden_state_every_final_norm_dispatch_is_followed_by_a_capture() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/forward/metal_qwen35.rs");
    // Fail closed: an unreadable source file is an instrument failure,
    // not an absence of violations.
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read {} to check the capture invariant: {e}",
            path.display()
        )
    });
    let all_lines: Vec<&str> = src.lines().collect();

    // Scan production code only. This test's own body quotes the match
    // patterns as string literals, so scanning the whole file makes the
    // test match itself — it did, on the first run, and the failure was
    // its own source line. Fail closed if the boundary is not found
    // rather than silently scanning everything again.
    let tests_start = all_lines
        .iter()
        .position(|l| l.trim() == "mod tests {")
        .unwrap_or_else(|| {
            panic!(
                "cannot find the `mod tests {{` boundary in {}; refusing to scan \
                         the test module's own pattern literals as if they were dispatch sites",
                path.display()
            )
        });
    let lines = &all_lines[..tests_start];

    // A dispatch site binds the weight buffer, so require the line to
    // be a buffer argument rather than prose mentioning the name.
    let sites: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| {
            let t = l.trim();
            !t.starts_with("//")
                && (t == "&self.engine.final_norm," || t.contains("Some(&self.engine.final_norm)"))
        })
        .map(|(i, _)| i)
        .collect();

    // Positive control. A pattern that matches nothing would otherwise
    // report a clean pass over an empty set — the failure mode this
    // whole test exists to prevent, one level up.
    assert!(
        sites.len() >= 3,
        "expected at least 3 final-norm dispatch sites, found {}. The match \
                 pattern has drifted from the code; fix the pattern rather than \
                 trusting the pass.",
        sites.len()
    );

    // Window: the norm dispatch's own argument list, then the capture.
    // Wide enough for a multi-line dispatch, narrow enough that a
    // capture belonging to some later block cannot satisfy it.
    const WINDOW: usize = 14;
    let mut uncovered = Vec::new();
    for &site in &sites {
        let end = (site + WINDOW).min(lines.len());
        let covered = lines[site..end]
            .iter()
            .any(|l| l.contains("encode_capture_final_hidden("));
        if !covered {
            uncovered.push(site + 1);
        }
    }
    assert!(
        uncovered.is_empty(),
        "final-RMSNorm dispatch without a following encode_capture_final_hidden \
                 at line(s) {uncovered:?} of {}. A forward path that norms without \
                 capturing leaves `activations.final_hidden` holding whatever the \
                 previous call left there, and embed_tokens would return a well-formed \
                 vector for somebody else's input.",
        path.display()
    );
    println!(
        "capture invariant: {} final-norm dispatch sites, all covered",
        sites.len()
    );
}

#[test]
fn hidden_state_metal_refuses_mean_pooling_rather_than_substituting_last_token() {
    let _gpu = gpu_test_lock();
    let Some(model) = hidden_parity_model() else {
        return;
    };
    let cfg = model.config().clone();
    let mut metal = MetalQwen35State::new(model.weights(), &cfg, 4096).expect("build Metal state");

    let err = metal
        .embed_tokens(
            HIDDEN_PARITY_MULTI,
            crate::model::qwen35::HiddenPooling::Mean,
        )
        .expect_err("Metal must refuse mean pooling, not quietly return last-token pooling");
    let msg = err.to_string();
    assert!(
        msg.contains("Mean"),
        "the refusal should name the pooling mode it cannot serve: {msg}"
    );
}

#[test]
fn hidden_state_metal_rejects_out_of_vocab_and_empty_input_without_panicking() {
    let _gpu = gpu_test_lock();
    let Some(model) = hidden_parity_model() else {
        return;
    };
    let cfg = model.config().clone();
    let mut metal = MetalQwen35State::new(model.weights(), &cfg, 4096).expect("build Metal state");

    let err = metal
        .embed_tokens(&[], crate::model::qwen35::HiddenPooling::LastToken)
        .expect_err("empty input must be rejected");
    assert!(err.to_string().contains("at least 1 token"), "{err}");

    // `forward_prefill` asserts on an out-of-range id, so without a
    // check ahead of it a library consumer gets a panic where the CPU
    // sibling returns an error.
    let bad = cfg.vocab_size as u32;
    let err = metal
        .embed_tokens(&[bad], crate::model::qwen35::HiddenPooling::LastToken)
        .expect_err("out-of-vocab id must be rejected");
    assert!(err.to_string().contains("vocab_size"), "{err}");
}
