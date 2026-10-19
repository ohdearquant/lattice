//! MoE Q4 checkpoints: the FFN branch, routed-expert LRU, prefetch and mmap dequant validation.

use super::*;
// -------------------------------------------------------------------
// from_q4_dir MoE FFN branch (#876): a MoE Q4 checkpoint has no
// `mlp.{gate,up,down}_proj.weight` files (`should_quantize` in
// quantize_q4.rs routes MoE tensors under `mlp.experts.*` /
// `mlp.shared_expert*` / `mlp.gate.weight` instead — see #874), so
// `from_q4_dir` must detect `cfg.is_moe()` and load the MoE branch
// instead of unconditionally building `MetalFfnWeights::Dense` and
// crashing at layer 0 with a missing-file error (the bug this issue
// reports). This builds a tiny synthetic 1-layer full-attention MoE
// checkpoint entirely on disk (no downloaded model required) and
// proves `from_q4_dir` loads it and produces a finite-logit forward
// pass end to end — the same code path `chat_metal --model <dir>`
// and `MetalQwen35State::from_q4_dir` callers exercise. A real
// Qwen3.6-35B-A3B Q4 run is tracked separately (issue body: "35B
// verification dir ... in-flight") — this is the fast, CI-safe leg.
//
// The original fixture
// filled every Q4 tensor with the same `0.1` constant, so an expert
// swap, wrong gate/up boundary, or an equal-length transpose left
// the test numerically finite and effectively unchanged — it could
// not catch the layout-corruption class the from_q4_dir MoE-loading fix above
// guards against. Rewritten to (a) give every expert/projection a
// DISTINCT block-constant value, (b) directly read the loaded
// `routed_gate_up`/`routed_down` Metal buffers at expert 0/1's
// known offsets and assert they equal the constants that expert
// was written with (proves the expert-major layout end to end,
// independent of generation), and (c) force the router to
// deterministically select a single nonzero expert (#2 of 4) and
// assert decode output changes when that expert's on-disk weights
// are corrupted to look like expert 0's (mutation-sensitive to
// exactly the "expert swap" bug class the from_q4_dir MoE-loading fix rejects at
// load time — here proven at the numerical-output level too, for a
// malformed-but-shape-valid corruption the shape gate cannot catch
// by itself, e.g. two experts' payload bytes swapped without
// touching either header).
// -------------------------------------------------------------------

/// Per-expert/per-projection block constants for the synthetic MoE
/// fixture below. Every routed tensor is filled row/expert-constant
/// (see `write_tiny_q4_fixture_per_expert_row`'s block-alignment
/// argument), so these functions double as both "what to write" and
/// "what to expect back" (via `q4_const_roundtrip`).
fn moe_fixture_gate_const(e: usize) -> f32 {
    0.02 * (e as f32 + 1.0)
}
fn moe_fixture_up_const(e: usize) -> f32 {
    0.05 * (e as f32 + 1.0)
}
fn moe_fixture_down_const(e: usize) -> f32 {
    0.03 * (e as f32 + 1.0)
}

/// Router row constants: expert 2 (of 4) carries the largest value,
/// so it deterministically wins top-1 routing. This relies on the
/// rest of the synthetic checkpoint (embed/attn/norm weights) being
/// entirely non-negative (`write_tiny_q4_fixture`'s default `0.1`,
/// `write_tiny_f16_fixture`'s default `1.0`): a softmax-weighted
/// average of non-negative `V` rows is always non-negative
/// regardless of the attention pattern, and RMSNorm with a
/// positive, uniform weight maps a uniform non-negative input to a
/// uniform non-negative output — so the hidden state reaching the
/// router is a uniform *non-negative* vector, and per-expert router
/// logits (a per-row-constant weight dotted with that vector) are
/// each `row_const[e] * (non-negative scalar)`, preserving the
/// `row_const` ordering across experts regardless of the exact
/// hidden-state magnitude.
// First 4 entries are load-bearing for every `num_experts=4` fixture
// (expert 2 must deterministically win top-1 routing — see the doc
// comment above). Entries beyond index 3 exist only so a bigger
// `num_experts` fixture (e.g. the Stage 2 throughput benchmark,
// which always overrides routing via `FORCED_MOE_EXPERTS_FOR_TEST`
// and so never actually reads these logits) doesn't index out of
// bounds; their exact values are unused/don't-care.
const MOE_FIXTURE_ROUTER_ROW_VALUES: [f32; 16] = [
    1.0, 2.0, 6.0, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
];
const MOE_FIXTURE_TARGET_EXPERT: usize = 2;

/// Write the full synthetic 1-layer MoE Q4 checkpoint used by both
/// the "correct" and "sabotaged" (mutation) variants below.
/// `gate_const`/`up_const`/`down_const` are injected so the
/// sabotaged variant can override exactly `MOE_FIXTURE_TARGET_EXPERT`'s
/// values to expert 0's, while everything else (router included)
/// stays identical between the two builds.
fn write_synthetic_moe_checkpoint(
    dir: &std::path::Path,
    cfg: &Qwen35Config,
    gate_const: impl Fn(usize) -> f32,
    up_const: impl Fn(usize) -> f32,
    down_const: impl Fn(usize) -> f32,
) {
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let num_experts = cfg.num_experts.expect("fixture cfg must set num_experts");
    let moe_inter = cfg.moe_intermediate_size();
    let shared_inter = cfg.shared_expert_intermediate_size();
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();

    write_tiny_q4_fixture(
        dir,
        "model.language_model.embed_tokens.weight",
        &[vocab, hidden],
    );
    write_tiny_f16_fixture(dir, "model.language_model.norm.weight", &[hidden]);

    let prefix = "model.language_model.layers.0";
    write_tiny_f16_fixture(dir, &format!("{prefix}.input_layernorm.weight"), &[hidden]);
    write_tiny_f16_fixture(
        dir,
        &format!("{prefix}.post_attention_layernorm.weight"),
        &[hidden],
    );
    write_tiny_q4_fixture(
        dir,
        &format!("{prefix}.self_attn.q_proj.weight"),
        &[2 * q_dim, hidden],
    );
    write_tiny_q4_fixture(
        dir,
        &format!("{prefix}.self_attn.k_proj.weight"),
        &[kv_dim, hidden],
    );
    write_tiny_q4_fixture(
        dir,
        &format!("{prefix}.self_attn.v_proj.weight"),
        &[kv_dim, hidden],
    );
    write_tiny_q4_fixture(
        dir,
        &format!("{prefix}.self_attn.o_proj.weight"),
        &[hidden, q_dim],
    );
    write_tiny_f16_fixture(
        dir,
        &format!("{prefix}.self_attn.q_norm.weight"),
        &[cfg.head_dim],
    );
    write_tiny_f16_fixture(
        dir,
        &format!("{prefix}.self_attn.k_norm.weight"),
        &[cfg.head_dim],
    );

    // MoE FFN tensors — the exact set `should_quantize` (quantize_q4.rs)
    // routes to `.q4` for a MoE layer (#874), matching what
    // `Self::load_moe_ffn_q4` (this PR) reads back.
    write_tiny_q4_fixture_per_row(
        dir,
        &format!("{prefix}.mlp.gate.weight"),
        num_experts,
        hidden,
        |e| MOE_FIXTURE_ROUTER_ROW_VALUES[e],
    );
    write_tiny_q4_fixture_per_expert_row(
        dir,
        &format!("{prefix}.mlp.experts.gate_up_proj"),
        num_experts,
        2 * moe_inter,
        hidden,
        |e, m| {
            if m < moe_inter {
                gate_const(e)
            } else {
                up_const(e)
            }
        },
    );
    // `h_row` (the down-proj output row, one of `hidden`) is folded
    // into the value on purpose: an expert whose gate/up/down blocks
    // are all constant across every dimension produces an FFN
    // contribution that is *uniform* across the output hidden
    // vector (every output row gets the same scalar). A single
    // uniform perturbation added to an already-uniform residual
    // stream is exactly what the model's final RMSNorm (this is a
    // 1-layer fixture — the FFN output feeds straight into the
    // final norm before lm_head) is invariant to: RMSNorm divides
    // by the input's own magnitude, so `x/rms(x)` collapses to
    // `sign(x)` for any uniform `x`, erasing the perturbation's
    // magnitude entirely and making an all-uniform mutation
    // invisible at the logits regardless of which expert ran. This
    // per-row multiplier breaks that uniformity so a corrupted
    // expert's contribution is distinguishable post-norm too, not
    // just via the direct buffer read in test (b) above.
    write_tiny_q4_fixture_per_expert_row(
        dir,
        &format!("{prefix}.mlp.experts.down_proj"),
        num_experts,
        hidden,
        moe_inter,
        |e, h_row| down_const(e) * (1.0 + 0.003 * h_row as f32),
    );
    write_tiny_q4_fixture_per_row(
        dir,
        &format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
        shared_inter,
        hidden,
        |_| 0.07,
    );
    write_tiny_q4_fixture_per_row(
        dir,
        &format!("{prefix}.mlp.shared_expert.up_proj.weight"),
        shared_inter,
        hidden,
        |_| 0.09,
    );
    write_tiny_q4_fixture_per_row(
        dir,
        &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
        hidden,
        shared_inter,
        |_| 0.11,
    );
    write_tiny_q4_fixture_per_row(
        dir,
        &format!("{prefix}.mlp.shared_expert_gate.weight"),
        1,
        hidden,
        |_| 0.5,
    );
}

fn synthetic_moe_test_config() -> Qwen35Config {
    let hidden = 512usize;
    let vocab = 64usize;
    let num_experts = 4usize;
    let top_k = 1usize;
    let moe_inter = 32usize;
    let shared_inter = 16usize;
    Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 1,
        vocab_size: vocab,
        intermediate_size: 64, // unused under MoE; kept for struct parity
        rms_norm_eps: 1e-6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 256,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        rope_parameters: None,
        linear_num_key_heads: 1,
        linear_num_value_heads: Some(1),
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        num_experts: Some(num_experts),
        num_experts_per_tok: Some(top_k),
        moe_intermediate_size: Some(moe_inter),
        shared_expert_intermediate_size: Some(shared_inter),
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        full_attention_interval: 1,
        layer_types: vec![LayerType::FullAttention],
        layer_mask: vec![true],
        eos_token_id: (vocab - 1) as u32,
        max_position_embeddings: 128,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    }
}

/// Same fixture shape as [`synthetic_moe_test_config`] (4 experts,
/// same hidden/vocab/inter sizes — `write_synthetic_moe_checkpoint`
/// and the `moe_fixture_*`/`MOE_FIXTURE_TARGET_EXPERT` constants all
/// key off `num_experts=4` only, not `top_k`) but with `top_k=2`,
/// so `encode_moe_ffn`'s real CPU routing resolves two distinct
/// experts per token instead of one — needed to exercise a genuine
/// within-token hit-then-miss `ExpertSlotCache::resolve` sequence
/// through the real encoded command buffer (see
/// `FORCED_MOE_EXPERTS_FOR_TEST`).
fn synthetic_moe_test_config_top_k2() -> Qwen35Config {
    let mut cfg = synthetic_moe_test_config();
    cfg.num_experts_per_tok = Some(2);
    cfg
}

#[test]
fn from_q4_dir_loads_and_generates_on_synthetic_moe_checkpoint() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config();
    assert!(
        cfg.is_moe(),
        "fixture config must exercise from_q4_dir's MoE branch"
    );
    let vocab = cfg.vocab_size;
    let moe_inter = cfg.moe_intermediate_size();
    let hidden = cfg.hidden_size;

    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    write_synthetic_moe_checkpoint(
        dir,
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );

    let tokenizer_path = std::path::Path::new("/dev/null");
    let mut state = MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16)
        .expect("from_q4_dir must load a synthetic MoE Q4 checkpoint (#876 fix)");

    // --- (b) direct buffer read: expert-major layout at expert 0/1 offsets ---
    // Read the loaded Metal buffers straight from the built state, before
    // any decode runs, to isolate loader correctness from generation
    // numerics — this is the layout proof for the expert-major layout
    // invariant. `from_q4_dir` always builds `RoutedExpertStorage::Cached`
    // (#682 Stage 1) — this exercises `ExpertSlotCache::resolve`'s
    // cache-miss dequant path directly, reading each resolved slot at
    // LOCAL offsets (`0` gate, `moe_inter * hidden` up, `0` down)
    // rather than the pre-#682 global `e * ...` offsets into one giant
    // resident buffer.
    {
        use crate::weights::q4_weights::q4_f16_to_f32;
        let (_, common) = &state.engine.layer_weights[0];
        let MetalFfnWeights::Moe(moe) = &common.ffn else {
            panic!("layer 0 must build MetalFfnWeights::Moe for an is_moe() config");
        };
        let RoutedExpertStorage::Cached { gate_up, down } = &moe.routed else {
            panic!("from_q4_dir must build RoutedExpertStorage::Cached (#682 Stage 1)");
        };
        // SAFETY: StorageModeShared buffers, no GPU work has been
        // encoded yet (state was just constructed) — safe CPU read.
        unsafe {
            for e in [0usize, 1usize] {
                let gate_up_buf = gate_up.borrow_mut().resolve(e).clone();
                let down_buf = down.borrow_mut().resolve(e).clone();
                let gate_up_ptr = gate_up_buf.contents() as *const u16;
                let down_ptr = down_buf.contents() as *const u16;
                let up_off = moe_inter * hidden;
                let gate_read = q4_f16_to_f32(*gate_up_ptr.add(0));
                let up_read = q4_f16_to_f32(*gate_up_ptr.add(up_off));
                let down_read = q4_f16_to_f32(*down_ptr.add(0));
                assert_eq!(
                    gate_read,
                    q4_const_roundtrip(moe_fixture_gate_const(e)),
                    "expert {e} gate slice mismatch (expert-major gate/up layout bug, \
                             or #682 cache resolve() addressing bug)"
                );
                assert_eq!(
                    up_read,
                    q4_const_roundtrip(moe_fixture_up_const(e)),
                    "expert {e} up slice at element offset {up_off} mismatch \
                             (expert-major gate/up boundary bug, or #682 cache resolve() \
                             addressing bug)"
                );
                assert_eq!(
                    down_read,
                    q4_const_roundtrip(moe_fixture_down_const(e)),
                    "expert {e} down slice mismatch (expert-major down-proj layout bug, \
                             or #682 cache resolve() addressing bug)"
                );
            }
        }
    }

    // ADR-053 v1's MoE dispatch is decode-mode only (batched multi-token
    // prefill GEMM panics on `MetalFfnWeights::Moe` — see
    // `forward_prefill`'s "MoE layers are not supported in batch
    // prefill" guard); drive this checkpoint through `forward_step`
    // (the M=1 GEMV decode path `encode_moe_ffn` implements) token by
    // token instead, exactly like a real `chat_metal` decode loop.
    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    let mut logits = Vec::new();
    for (position, &token) in tokens.iter().enumerate() {
        logits = state.forward_step(token, position);
        assert_eq!(logits.len(), vocab);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "MoE decode step at position {position} produced non-finite logits: {logits:?}"
        );
    }

    let next = cpu_topk_token_ids(&logits, 1)[0];
    assert!((next as usize) < vocab);

    let logits2 = state.forward_step(next, tokens.len());
    assert_eq!(logits2.len(), vocab);
    assert!(
        logits2.iter().all(|v| v.is_finite()),
        "MoE decode step produced non-finite logits: {logits2:?}"
    );
}

#[test]
fn from_q4_dir_builds_the_cpu_embedding_lookup_from_the_q4_payload() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config();
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let dir = tempfile::tempdir().expect("tempdir create");
    // Distinct values per row, so a lookup that read the wrong row or the
    // wrong stride cannot match. `hidden` is a multiple of 32, so each row
    // occupies whole blocks and the quantization reconstructs its constant
    // exactly (mod one f16 rounding -- `q4_const_roundtrip`).
    let embed_row_value = |row: usize| 0.001 * (row as f32 + 1.0);
    // Every other tensor is irrelevant to this test; fill the rest of the
    // checkpoint so `from_q4_dir`'s existence preflight and loaders see a
    // complete directory. The synthetic MoE writer rewrites embed_tokens
    // with its own constant, so the row-specific table is written after.
    write_synthetic_moe_checkpoint(
        dir.path(),
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );
    write_tiny_q4_fixture_per_row(
        dir.path(),
        "model.language_model.embed_tokens.weight",
        vocab,
        hidden,
        embed_row_value,
    );

    let state =
        MetalQwen35State::from_q4_dir(dir.path(), std::path::Path::new("/dev/null"), &cfg, 16)
            .expect("from_q4_dir must load the synthetic Q4 checkpoint");
    assert!(
        matches!(&state.engine.embed_tokens, EmbeddingTable::Q4(_)),
        "a Q4 checkpoint must keep the CPU embedding lookup on the Q4 payload, \
         not build an f16 copy"
    );

    for id in [0usize, 1, 41, 42, vocab - 1] {
        let mut actual = vec![0.0f32; hidden];
        // SAFETY: `actual` holds `hidden` f32 values and is disjoint from the
        // embedding table; `id` is below `vocab_size`.
        unsafe {
            state
                .engine
                .embed_tokens
                .write_row(id, actual.as_mut_ptr(), hidden)
        };
        let value = q4_const_roundtrip(embed_row_value(id));
        assert!(
            actual.iter().all(|&v| v == value),
            "token {id} must read the checkpoint's own row #{id}"
        );
    }
}

// --- (c) mutation sensitivity: an "expert swap" corruption changes decode output ---
//
// Builds a second checkpoint identical to the one above except that
// MOE_FIXTURE_TARGET_EXPERT's (expert 2's) gate/up/down constants are
// overridden to expert 0's — i.e. exactly what a wrong-offset or
// transposed-index bug in `load_moe_ffn_q4` would produce (expert 2's
// *slot* ends up holding expert 0's weights). The router is
// unchanged, so routing still deterministically selects expert 2
// (routing must deterministically select a nonzero expert). If the
// loader's expert-major offsets were wrong in a way
// the shape gate above cannot catch (e.g. two experts' bytes
// transposed without changing either header's shape), this
// corruption would silently have NO effect on which weights actually
// get used at expert 2's dispatch offset — so decode output
// differing here is direct evidence the loaded weights, not just the
// headers, are wired to the offsets this test expects.
#[test]
fn from_q4_dir_moe_decode_is_sensitive_to_expert_weight_corruption() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config();
    let vocab = cfg.vocab_size;
    let tokens: Vec<u32> = vec![1, 2, 3, 4];

    let run = |gate_const: fn(usize) -> f32,
               up_const: fn(usize) -> f32,
               down_const: fn(usize) -> f32|
     -> Vec<f32> {
        let tmp = tempfile::tempdir().expect("tempdir create");
        let dir = tmp.path();
        write_synthetic_moe_checkpoint(dir, &cfg, gate_const, up_const, down_const);
        let tokenizer_path = std::path::Path::new("/dev/null");
        let mut state = MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16)
            .expect("from_q4_dir must load a synthetic MoE Q4 checkpoint (#876 fix)");
        let mut logits = Vec::new();
        for (position, &token) in tokens.iter().enumerate() {
            logits = state.forward_step(token, position);
        }
        assert_eq!(logits.len(), vocab);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "MoE decode produced non-finite logits: {logits:?}"
        );
        logits
    };

    let correct = run(
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );
    let sabotaged = run(
        |e| {
            if e == MOE_FIXTURE_TARGET_EXPERT {
                moe_fixture_gate_const(0)
            } else {
                moe_fixture_gate_const(e)
            }
        },
        |e| {
            if e == MOE_FIXTURE_TARGET_EXPERT {
                moe_fixture_up_const(0)
            } else {
                moe_fixture_up_const(e)
            }
        },
        |e| {
            if e == MOE_FIXTURE_TARGET_EXPERT {
                moe_fixture_down_const(0)
            } else {
                moe_fixture_down_const(e)
            }
        },
    );

    assert_ne!(
        correct, sabotaged,
        "decode logits must change when the routed/deterministically-selected expert's \
                 weights are corrupted to another expert's values — identical output here would \
                 mean the selected expert's dispatch offset isn't actually wired to its own \
                 loaded weights (exactly the expert-swap/offset bug class this PR's shape gate \
                 and mmap offsets must prevent)"
    );
}

// -------------------------------------------------------------------
// #682 Stage 1: dequant-on-demand routed experts with a bounded LRU
// cache. `from_q4_dir` always builds `RoutedExpertStorage::Cached`
// now (there is no more eager/lazy fork at the `.q4` loader level —
// that fork moved to `RoutedExpertStorage`'s two variants, and only
// `MetalQwen35Engine::new()`'s safetensors-upload path still builds
// `Eager`). So "eager vs. lazy" for the `.q4` path becomes
// "N=num_experts (zero-eviction fast path) vs. N<num_experts
// (forced eviction)" — both use the SAME dequant-on-demand
// machinery, just with different eviction pressure. The three tests
// below are the Stage 1 test-strategy items from PLAN.md: (a) an
// eviction-forcing N vs. a zero-eviction N must produce identical
// decode logits; (b) is folded into (a) — the forced-eviction run
// deliberately interleaves resolves of every OTHER expert between
// decode steps so the deterministically-selected expert's slot is
// repeatedly evicted and reloaded; (c) corrupting an expert's
// on-disk bytes after it has been evicted-and-reloaded at least
// once must still change decode output, proving the reload path
// re-reads rather than serving something stale.
// -------------------------------------------------------------------

/// A drop guard that restores (or clears) an env var on scope exit,
/// so a panicking assertion inside a test can't leak
/// `LATTICE_MOE_EXPERT_CACHE_SLOTS` into whichever GPU test the
/// `gpu_test_lock()` mutex hands the machine to next.
struct EnvVarGuard {
    key: &'static str,
    prior: Option<String>,
}
impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prior = std::env::var(key).ok();
        // SAFETY: callers only mutate `LATTICE_MOE_EXPERT_CACHE_SLOTS`
        // from inside a `gpu_test_lock()`-guarded test, which
        // serializes every GPU test in this binary (same pattern as
        // `with_self_spec_env` above) — no concurrent reader/writer.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, prior }
    }
}
impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: see `EnvVarGuard::set`.
        unsafe {
            match &self.prior {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

/// Directly evict/reload `expert_id` through a layer's routed-expert
/// caches, bypassing `encode_moe_ffn`/routing entirely. Used to
/// simulate "many decode steps' worth" of eviction pressure against
/// experts routing never actually selects in the deterministic
/// synthetic fixture (`MOE_FIXTURE_TARGET_EXPERT` always wins), so
/// tests can force real LRU eviction/reload cycles on a target
/// expert without needing routing itself to vary per token.
fn force_resolve_layer0(state: &MetalQwen35State, expert_id: usize) {
    let (_, common) = &state.engine.layer_weights[0];
    let MetalFfnWeights::Moe(moe) = &common.ffn else {
        panic!("layer 0 must build MetalFfnWeights::Moe for an is_moe() config");
    };
    let RoutedExpertStorage::Cached { gate_up, down } = &moe.routed else {
        panic!("from_q4_dir must build RoutedExpertStorage::Cached (#682 Stage 1)");
    };
    gate_up.borrow_mut().begin_token();
    down.borrow_mut().begin_token();
    let _ = gate_up.borrow_mut().resolve(expert_id);
    let _ = down.borrow_mut().resolve(expert_id);
}

#[test]
fn from_q4_dir_moe_forced_eviction_matches_zero_eviction_baseline() {
    let Some(device) = Device::system_default() else {
        eprintln!("[moe-eviction-route-proof] SKIPPED: no Metal device present on this machine");
        return;
    };
    // #899: the paravirtual CI GPU diverges deterministically here, but
    // this fixture cannot exercise the Apple7-gated Q4 GEMM selection:
    // MoE batch prefill is rejected and every projection below runs at
    // M=1, which dispatches GEMV before either GEMM branch. Keep the
    // device exception explicit while the zero-dispatch assertion below
    // prevents the paravirtual discrepancy from being misattributed to
    // the fallback GEMM again.
    if !device.supports_family(MTLGPUFamily::Apple7) {
        eprintln!(
            "[moe-eviction-route-proof] SKIPPED: device does not support Apple7 (fixture \
                     cannot discriminate the Q4 GEMM fallback route on this GPU)"
        );
        return;
    }
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config();
    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    let tokenizer_path = std::path::Path::new("/dev/null");

    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    write_synthetic_moe_checkpoint(
        dir,
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );

    // --- Baseline: N = num_experts (zero-eviction fast path). ---
    // SAFETY: see `EnvVarGuard::set` — serialized by `gpu_test_lock()`.
    unsafe {
        std::env::remove_var(crate::forward::moe_expert_cache::MOE_EXPERT_CACHE_SLOTS_ENV);
    }
    let mut baseline = MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16)
        .expect("baseline (N=num_experts) load must succeed");
    assert!(
        baseline.force_q4_gemm_fallback_for_test(),
        "Apple7+ discrimination requires the tiled Q4 pipeline to exist before \
                 forcing the non-Apple7 fallback selection"
    );
    MetalQwen35State::reset_q4_gemm_fallback_dispatches_for_test();
    let mut baseline_logits = Vec::new();
    for (position, &token) in tokens.iter().enumerate() {
        baseline_logits = baseline.forward_step(token, position);
    }

    // --- Forced eviction: N=2 slots for this 4-expert layer. ---
    let mut forced = {
        let _env = EnvVarGuard::set(
            crate::forward::moe_expert_cache::MOE_EXPERT_CACHE_SLOTS_ENV,
            "2",
        );
        MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16)
            .expect("forced-eviction (N=2) load must succeed")
    };
    assert!(
        forced.force_q4_gemm_fallback_for_test(),
        "Apple7+ discrimination requires the tiled Q4 pipeline to exist before \
                 forcing the non-Apple7 fallback selection"
    );

    let mut forced_logits = Vec::new();
    for (position, &token) in tokens.iter().enumerate() {
        // Between decode steps, directly evict/reload the three
        // experts routing never selects (0, 1, 3 — routing always
        // deterministically picks expert 2). With only 2 slots,
        // resolving 3 distinct experts guarantees at least one
        // eviction each round, and necessarily pushes expert 2's
        // slot out at least once per round too (pigeonhole: 2 slots
        // can't hold {0,1,2,3} simultaneously). The next
        // `forward_step` below must then re-dequant expert 2 from
        // disk before dispatching its GEMVs.
        for &other in &[0usize, 1usize, 3usize] {
            force_resolve_layer0(&forced, other);
        }
        forced_logits = forced.forward_step(token, position);
    }

    let (_, common) = &forced.engine.layer_weights[0];
    let MetalFfnWeights::Moe(moe) = &common.ffn else {
        panic!("layer 0 must build MetalFfnWeights::Moe for an is_moe() config");
    };
    let RoutedExpertStorage::Cached { gate_up, down } = &moe.routed else {
        panic!("from_q4_dir must build RoutedExpertStorage::Cached");
    };
    for (label, cache) in [("gate_up", gate_up), ("down", down)] {
        let (_, _, evictions) = cache.borrow().hit_miss_eviction_counts();
        eprintln!("[moe-eviction-proof] cache={label} evictions={evictions}");
        assert!(
            evictions > 0,
            "{label} cache must evict at least one resident expert before logit parity \
                     can establish that eviction is numerically transparent"
        );
    }

    assert_eq!(baseline_logits.len(), forced_logits.len());
    for (i, (b, f)) in baseline_logits.iter().zip(forced_logits.iter()).enumerate() {
        assert!(
            (b - f).abs() < 1e-4,
            "logit {i} diverged between the N=num_experts baseline ({b}) and the \
                     forced-eviction N=2 run ({f}) after repeated evict/reload cycles — a \
                     correct dequant-on-demand cache must be numerically transparent to \
                     decode output regardless of eviction pressure (baseline={baseline_logits:?}, \
                     forced={forced_logits:?})"
        );
    }
    let fallback_dispatches = MetalQwen35State::q4_gemm_fallback_dispatches_for_test();
    eprintln!(
        "[moe-eviction-route-proof] q4_gemm_fallback_dispatches={fallback_dispatches} \
                 route=m1-gemv"
    );
    assert_eq!(
        fallback_dispatches, 0,
        "MoE eviction parity must remain an M=1 GEMV test; a Q4 fallback GEMM dispatch \
                 would invalidate #899's route discrimination"
    );
}

/// Restores `FORCED_MOE_EXPERTS_FOR_TEST` to `None` on scope exit
/// (including on panic), so a forced-routing test can't leak its
/// override into whichever GPU test the `gpu_test_lock()` mutex
/// hands this OS thread's cargo-test worker to next (thread-locals
/// persist across tests reusing the same worker thread, unlike a
/// per-call stack variable).
struct ForcedMoeExpertsGuard;
impl ForcedMoeExpertsGuard {
    fn set(experts: Vec<usize>) -> Self {
        set_forced_moe_experts_for_test(Some(experts));
        Self
    }
}
impl Drop for ForcedMoeExpertsGuard {
    fn drop(&mut self) {
        set_forced_moe_experts_for_test(None);
    }
}

/// The Medium finding this test closes: the forced-eviction test
/// above and the corruption test below never actually observe a
/// hit *followed by* a miss inside one real, still-open command
/// buffer (the fixture's `top_k=1` only ever resolves one expert
/// per `encode_moe_ffn` call, and `force_resolve_layer0` bypasses
/// encoding entirely). This test drives `top_k=2` real CPU routing
/// through [`set_forced_moe_experts_for_test`] so two distinct
/// experts are resolved per token via the SAME `encode_moe_ffn`
/// call `forward_step` uses in production, with a 2-slot cache:
///
/// - Token 0 forces experts `{2, 3}` — cache starts empty, so both
///   resolves are misses that fill both slots.
/// - Token 1 forces experts `{2, 1}` — expert 2 is still
///   LRU-resident (a HIT), resolved first; expert 1 is not (a MISS)
///   and is resolved second, in the SAME token's single command
///   buffer as the hit that preceded it. With only 2 slots and
///   expert 2 protected by this token's `slot_touched` flag (set by
///   its own hit just now), the only evictable slot is expert 3's —
///   `pick_eviction_slot` is exercised for real, mid-encoder-pass.
///
/// Asserts the *effective* slot count (so an ignored/garbage env
/// override can't silently make this "test" run with more slots
/// than intended and never actually force the hit-then-miss path)
/// and the exact cumulative hit/miss/eviction counts, not just
/// finite output.
///
/// Mutation check, manually verified (not compiled in): reverting
/// `ExpertSlotCache::pick_eviction_slot` to ignore `slot_touched`
/// entirely (evict whatever is at the LRU front, unconditionally)
/// does NOT make this test fail — this was checked by hand and is
/// expected, not a gap in this test. `touch()` moves a slot to the
/// LRU back every time `resolve()` touches it (hit or miss), so a
/// slot touched earlier in the *same* token is always more
/// recently used than any slot not yet touched this token; combined
/// with `num_slots >= top_k` (enforced at construction — see the
/// module doc comment), plain recency-order LRU can never reach a
/// same-token-touched slot before it runs out of untouched slots to
/// pick from first, for any single real `encode_moe_ffn` call. The
/// `slot_touched` guard's actual job is turning a hypothetical
/// `num_slots < top_k` misconfiguration (which construction-time
/// validation should already prevent) into a loud
/// `pick_eviction_slot` panic instead of a silent same-token
/// overwrite race — not changing the eviction target in the
/// well-configured case this test exercises. This test still earns
/// its keep as the one exercising a real hit-then-miss sequence
/// inside one real, still-open command buffer, with counters
/// pinning the exact hit/miss/eviction shape.
#[test]
fn from_q4_dir_moe_within_token_hit_then_miss_evicts_only_the_untouched_slot() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config_top_k2();
    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    write_synthetic_moe_checkpoint(
        dir,
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );

    let tokenizer_path = std::path::Path::new("/dev/null");
    let mut state = {
        let _env = EnvVarGuard::set(
            crate::forward::moe_expert_cache::MOE_EXPERT_CACHE_SLOTS_ENV,
            "2",
        );
        MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16)
            .expect("from_q4_dir (top_k=2, forced-eviction N=2) must load")
    };

    // Token 0: {2, 3} — both miss, both slots filled from empty.
    let _forced0 = ForcedMoeExpertsGuard::set(vec![2, 3]);
    let logits0 = state.forward_step(1, 0);
    assert!(
        logits0.iter().all(|v| v.is_finite()),
        "token 0 decode produced non-finite logits: {logits0:?}"
    );

    // Token 1: {2, 1} — expert 2 hits (still resident), expert 1
    // misses and forces eviction of expert 3's (untouched) slot —
    // both resolves happen inside this ONE forward_step's single
    // encoded command buffer.
    let _forced1 = ForcedMoeExpertsGuard::set(vec![2, 1]);
    let logits1 = state.forward_step(2, 1);
    assert!(
        logits1.iter().all(|v| v.is_finite()),
        "token 1 decode produced non-finite logits: {logits1:?}"
    );

    let (_, common) = &state.engine.layer_weights[0];
    let MetalFfnWeights::Moe(moe) = &common.ffn else {
        panic!("layer 0 must build MetalFfnWeights::Moe for an is_moe() config");
    };
    let RoutedExpertStorage::Cached { gate_up, down } = &moe.routed else {
        panic!("from_q4_dir must build RoutedExpertStorage::Cached (#682 Stage 1)");
    };

    for (label, cache) in [("gate_up", gate_up), ("down", down)] {
        let cache = cache.borrow();
        assert_eq!(
            cache.num_slots(),
            2,
            "{label} cache: effective slot count must be exactly the forced N=2 — a \
                     silently-ignored or reclamped env override would let this test run with \
                     more slots and never actually exercise the hit-then-miss eviction path"
        );
        assert_eq!(
            cache.hit_miss_eviction_counts(),
            (1, 3, 1),
            "{label} cache: expected exactly 1 hit (expert 2 on token 1), 3 misses \
                     (experts 2 and 3 on token 0, expert 1 on token 1), and 1 eviction (expert \
                     3's slot reclaimed for expert 1 on token 1) — a wrong count means either \
                     the within-token touched-slot protection isn't working (expert 2's slot \
                     evicted instead) or eviction pressure isn't reaching this cache at all"
        );
    }
}

/// #682 Stage 4: the routing-trace collector's hot path is a single
/// branch on `MOE_ROUTING_TRACE`'s thread-local `None` state when
/// never armed — `encode_moe_ffn` must record nothing while
/// disarmed, and the collector must stay disarmed rather than
/// silently self-arming on first use.
///
/// Mutation check (documented as a manual check, not
/// compiled in): replacing `encode_moe_ffn`'s `if let Some(trace) =
/// c.borrow_mut().as_mut() { trace.push(..) }` guard with an
/// unconditional `c.borrow_mut().get_or_insert_with(Vec::new).push(..)`
/// (the collector self-arming instead of staying `None`) makes this test
/// fail: `take_moe_routing_trace()` returns 2 records instead of the
/// expected empty trace.
#[test]
fn moe_routing_trace_disarmed_records_nothing() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config();
    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    write_synthetic_moe_checkpoint(
        dir,
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );
    let tokenizer_path = std::path::Path::new("/dev/null");
    let mut state = MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16)
        .expect("from_q4_dir must load");

    // Defensive: drain+disarm in case a prior test on this worker
    // thread left the collector armed — thread-locals persist across
    // tests reusing the same `gpu_test_lock()`-serialized OS thread
    // (same reasoning as `ForcedMoeExpertsGuard`'s doc comment).
    let _ = take_moe_routing_trace();

    let logits0 = state.forward_step(1, 0);
    let logits1 = state.forward_step(2, 1);
    assert!(logits0.iter().all(|v| v.is_finite()));
    assert!(logits1.iter().all(|v| v.is_finite()));

    let trace = take_moe_routing_trace();
    assert!(
        trace.is_empty(),
        "encode_moe_ffn must record nothing while MOE_ROUTING_TRACE is disarmed \
                 (None) — got {} record(s): {trace:?}",
        trace.len()
    );
}

/// #682 Stage 4: armed, `encode_moe_ffn` records exactly the forced
/// routing selection (post-renormalization gate weights) for each
/// decoded token, keyed by the real `(layer_idx, token_idx)` the
/// token was decoded at. For the ordinary contiguous `0, 1` decode
/// sequence used here, a per-layer invocation counter is numerically
/// identical to the decode position, so the first two records cannot
/// discriminate the two — the discriminating probe is the third
/// decode, which runs after `reset_state()` at position `0`: the
/// threaded `position` reads back `0`, while any invocation counter
/// (since arm or since state creation) would stand at `2`. That third
/// record is what proves `position` is the value actually threaded
/// through `encode_mlp_block` into `encode_moe_ffn`. Also exercises
/// `dump_moe_routing_trace_jsonl`'s round-trip.
///
/// Mutation check (documented as a manual check, not
/// compiled in): commenting out the `trace.push(MoeRoutingTraceRecord
/// { .. })` call inside `encode_moe_ffn`'s `MOE_ROUTING_TRACE.with(..)`
/// block makes this test fail — `take_moe_routing_trace()` returns an
/// empty `Vec` instead of the three expected records.
#[test]
fn moe_routing_trace_armed_records_forced_selection() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config_top_k2();
    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    write_synthetic_moe_checkpoint(
        dir,
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );
    let tokenizer_path = std::path::Path::new("/dev/null");
    let mut state = MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16)
        .expect("from_q4_dir (top_k=2) must load");

    let _ = take_moe_routing_trace(); // start from a known-disarmed state
    arm_moe_routing_trace();

    let _forced0 = ForcedMoeExpertsGuard::set(vec![2, 3]);
    let logits0 = state.forward_step(1, 0);
    assert!(logits0.iter().all(|v| v.is_finite()));
    drop(_forced0);

    let _forced1 = ForcedMoeExpertsGuard::set(vec![2, 1]);
    let logits1 = state.forward_step(2, 1);
    assert!(logits1.iter().all(|v| v.is_finite()));
    drop(_forced1);

    // Third decode restarts the session and decodes at position 0.
    // The original discriminator here repeated position 1 while the
    // live cursor had already advanced to 2, and documented that as
    // mechanically safe because raw `forward_step` had no
    // position-monotonicity check. That was true when this test was
    // written and stopped being true when #1339 added
    // `check_live_cursor`, which rejects a supplied position that does
    // not equal `kv_cache.seq_len` before dispatch: KV placement and
    // attention length derive from the cursor while RoPE uses the
    // supplied position, so the two disagreeing is exactly the state
    // that guard exists to refuse.
    //
    // `reset_state()` returns the cursor to 0 and does not touch the
    // thread-local trace, so the same counter-vs-position question is
    // asked in a form the guard admits, and asked more sharply: the
    // threaded `position` reads 0 here, while a per-layer invocation
    // counter — since arm or since state creation — stands at 2 by
    // this third call. 0 and 2 cannot be confused, and 0 also differs
    // from the previous record's 1, so a stale carried-over value
    // cannot pass either.
    state.reset_state();
    let _forced2 = ForcedMoeExpertsGuard::set(vec![0, 3]);
    let logits2 = state.forward_step(3, 0);
    assert!(logits2.iter().all(|v| v.is_finite()));
    drop(_forced2);

    let trace = take_moe_routing_trace();
    assert_eq!(
        trace.len(),
        3,
        "expected exactly one record per decode call (1 MoE layer x 3 calls): {trace:?}"
    );
    assert_eq!(trace[0].layer_idx, 0);
    assert_eq!(trace[0].token_idx, 0);
    assert_eq!(trace[0].selected_ids, vec![2, 3]);
    assert_eq!(trace[1].layer_idx, 0);
    assert_eq!(trace[1].token_idx, 1);
    assert_eq!(trace[1].selected_ids, vec![2, 1]);
    assert_eq!(trace[2].layer_idx, 0);
    assert_eq!(
        trace[2].token_idx, 0,
        "post-reset decode must record the threaded position (0), \
                 not an invocation count (which would be 2 by this call): {trace:?}"
    );
    assert_eq!(trace[2].selected_ids, vec![0, 3]);
    for record in &trace {
        assert_eq!(record.gate_weights.len(), 2);
        for w in &record.gate_weights {
            assert!(
                (w - 0.5).abs() < 1e-6,
                "forced top_k=2 equal-weight gate renormalizes to 0.5 each: {trace:?}"
            );
        }
    }

    let dump_path = dir.join("routing_trace.jsonl");
    dump_moe_routing_trace_jsonl(&trace, &dump_path).expect("jsonl dump must succeed");
    let dumped = std::fs::read_to_string(&dump_path).expect("read back jsonl");
    let lines: Vec<&str> = dumped.lines().collect();
    assert_eq!(lines.len(), 3, "one JSONL line per record: {dumped:?}");
    let parsed: MoeRoutingTraceRecord =
        serde_json::from_str(lines[0]).expect("line 0 must parse as MoeRoutingTraceRecord");
    assert_eq!(parsed.token_idx, 0);
    assert_eq!(parsed.selected_ids, vec![2, 3]);

    // Collector auto-disarms on take; a follow-up decode must not append
    // further. Position 1 because the reset above restarted the session
    // and only one token has been decoded since, so 1 is the live cursor
    // this call has to match.
    let _more = state.forward_step(3, 1);
    assert!(take_moe_routing_trace().is_empty());
}

/// Mutation check for the test above: deliberately break
/// `ExpertSlotCache`'s within-token eviction guard (make
/// `pick_eviction_slot` always evict LRU-front regardless of
/// `slot_touched`) and confirm this test fails. This is exercised
/// manually (not compiled in), documented here so the invariant it
/// guards is traceable: reverting `pick_eviction_slot` to ignore
/// `slot_touched` makes `from_q4_dir_moe_forced_eviction_matches_zero_eviction_baseline`
/// either panic (a resolve() overwrites a slot an already-encoded-
/// but-not-yet-executed GEMV needs) or, on this synthetic fixture
/// (whose `top_k=1` never triggers the within-token case, since
/// only one expert is ever selected per `encode_moe_ffn` call), pass
/// unchanged — which is exactly why the module doc comment's
/// invariant proof relies on `num_slots >= top_k`, not on this test
/// catching every possible violation.
///
/// The corruption test below is this file's mutation-sensitive proof
/// that reload *addressing* is correct: it builds two DIFFERENT
/// checkpoints (identical except `MOE_FIXTURE_TARGET_EXPERT`'s
/// weights) and shows repeated evict/reload cycles still land on
/// each checkpoint's own bytes rather than some other expert's slot.
/// It does NOT prove that a reload re-reads bytes mutated on disk
/// *after* the engine already loaded and decoded once — that is a
/// separate claim, proven separately by
/// `from_q4_dir_moe_reload_rereads_expert_bytes_mutated_after_initial_load`
/// below (mmap semantics mean "loaded once" and "never reads disk
/// again" are not the same thing, and only mutating an
/// already-mapped file distinguishes them).
#[test]
fn from_q4_dir_moe_reload_after_eviction_addresses_the_correct_expert() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config();
    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    let tokenizer_path = std::path::Path::new("/dev/null");

    let run = |gate_const: fn(usize) -> f32,
               up_const: fn(usize) -> f32,
               down_const: fn(usize) -> f32|
     -> Vec<f32> {
        let tmp = tempfile::tempdir().expect("tempdir create");
        let dir = tmp.path();
        write_synthetic_moe_checkpoint(dir, &cfg, gate_const, up_const, down_const);

        let mut state = {
            let _env = EnvVarGuard::set(
                crate::forward::moe_expert_cache::MOE_EXPERT_CACHE_SLOTS_ENV,
                "2",
            );
            MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16)
                .expect("from_q4_dir (forced-eviction N=2) must load")
        };

        let mut logits = Vec::new();
        for (position, &token) in tokens.iter().enumerate() {
            // Force expert 2 (the deterministically-routed target)
            // out of the cache and back in at least once before
            // this token's real decode step, so the assertion below
            // is about the RELOAD path specifically, not just the
            // initial load.
            for &other in &[0usize, 1usize, 3usize] {
                force_resolve_layer0(&state, other);
            }
            force_resolve_layer0(&state, MOE_FIXTURE_TARGET_EXPERT);
            logits = state.forward_step(token, position);
        }
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "MoE decode produced non-finite logits: {logits:?}"
        );
        logits
    };

    let correct = run(
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );
    let sabotaged = run(
        |e| {
            if e == MOE_FIXTURE_TARGET_EXPERT {
                moe_fixture_gate_const(0)
            } else {
                moe_fixture_gate_const(e)
            }
        },
        |e| {
            if e == MOE_FIXTURE_TARGET_EXPERT {
                moe_fixture_up_const(0)
            } else {
                moe_fixture_up_const(e)
            }
        },
        |e| {
            if e == MOE_FIXTURE_TARGET_EXPERT {
                moe_fixture_down_const(0)
            } else {
                moe_fixture_down_const(e)
            }
        },
    );

    assert_ne!(
        correct, sabotaged,
        "decode logits must change when the target expert's checkpoint-provided \
                 weights differ, even though every decode step in this test forces that \
                 expert's cache slot to be evicted and reloaded before use — identical output \
                 here would mean reload addressing is wrong (some other expert's bytes are \
                 landing in the target expert's slot)"
    );
}

/// Companion to the addressing test above, proving the claim that
/// one does NOT prove: that `ExpertSlotCache::resolve`'s reload path
/// actually re-reads the on-disk `.q4` file rather than serving a
/// value memoized at (or cached from) the *first* load, once an
/// expert has already been loaded, decoded against, evicted, and
/// reloaded within the SAME `MetalQwen35State`/mmap.
///
/// Sequence: load a checkpoint, run one real decode step (so
/// `MOE_FIXTURE_TARGET_EXPERT`'s slot is resident from a genuine
/// `forward_step`, not a bypass helper) — THEN overwrite the routed
/// expert tensor files on disk in place (`File::create` truncates
/// the existing inode rather than replacing it, so the state's
/// already-open mmap observes the new bytes; see `save_q4_file`) —
/// THEN force that expert's slot out of the 2-slot cache and decode
/// again. Comparing against an otherwise-identical run that skips
/// the on-disk mutation isolates the effect of the mutation itself
/// (both runs still force the same eviction/reload cycle).
#[test]
fn from_q4_dir_moe_reload_rereads_expert_bytes_mutated_after_initial_load() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config();
    let moe_inter = cfg.moe_intermediate_size();
    let hidden = cfg.hidden_size;
    let num_experts = cfg.num_experts.expect("fixture cfg must set num_experts");
    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    let tokenizer_path = std::path::Path::new("/dev/null");
    let prefix = "model.language_model.layers.0";

    // Overwrite ONLY the two routed-expert tensor files already on
    // disk in `dir`, in place, with `MOE_FIXTURE_TARGET_EXPERT`'s
    // constants swapped to expert 0's (same "expert swap" shape as
    // the sibling addressing test).
    let mutate_target_expert_on_disk = |dir: &std::path::Path| {
        write_tiny_q4_fixture_per_expert_row(
            dir,
            &format!("{prefix}.mlp.experts.gate_up_proj"),
            num_experts,
            2 * moe_inter,
            hidden,
            |e, m| {
                let e = if e == MOE_FIXTURE_TARGET_EXPERT { 0 } else { e };
                if m < moe_inter {
                    moe_fixture_gate_const(e)
                } else {
                    moe_fixture_up_const(e)
                }
            },
        );
        write_tiny_q4_fixture_per_expert_row(
            dir,
            &format!("{prefix}.mlp.experts.down_proj"),
            num_experts,
            hidden,
            moe_inter,
            |e, h_row| {
                let e = if e == MOE_FIXTURE_TARGET_EXPERT { 0 } else { e };
                moe_fixture_down_const(e) * (1.0 + 0.003 * h_row as f32)
            },
        );
    };

    let run = |mutate_on_disk_after_first_decode: bool| -> Vec<f32> {
        let tmp = tempfile::tempdir().expect("tempdir create");
        let dir = tmp.path();
        write_synthetic_moe_checkpoint(
            dir,
            &cfg,
            moe_fixture_gate_const,
            moe_fixture_up_const,
            moe_fixture_down_const,
        );

        let mut state = {
            let _env = EnvVarGuard::set(
                crate::forward::moe_expert_cache::MOE_EXPERT_CACHE_SLOTS_ENV,
                "2",
            );
            MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16)
                .expect("from_q4_dir (forced-eviction N=2) must load")
        };

        // Decode once against the correct, as-loaded weights — this
        // is the "already loaded and decoded once" precondition;
        // MOE_FIXTURE_TARGET_EXPERT's slot is resident from a real
        // `forward_step`, and `state`'s `ExpertByteTable` mmaps are
        // now open over `dir`'s files.
        let _ = state.forward_step(tokens[0], 0);

        if mutate_on_disk_after_first_decode {
            mutate_target_expert_on_disk(dir);
        }

        // Force every routed expert but the target out of the
        // 2-slot cache; with only 2 slots and 4 experts this
        // necessarily evicts the target's slot too (pigeonhole),
        // so the next real decode step must re-dequantize it.
        for &other in &[0usize, 1usize, 3usize] {
            force_resolve_layer0(&state, other);
        }

        state.forward_step(tokens[1], 1)
    };

    let unmutated = run(false);
    let mutated = run(true);

    assert!(
        unmutated.iter().all(|v| v.is_finite()),
        "unmutated-continuation decode produced non-finite logits: {unmutated:?}"
    );
    assert!(
        mutated.iter().all(|v| v.is_finite()),
        "mutated-continuation decode produced non-finite logits: {mutated:?}"
    );
    assert_ne!(
        unmutated, mutated,
        "decode logits after a forced evict+reload must change once the target \
                 expert's on-disk bytes were mutated post-load — identical output here would \
                 mean the reload path serves memoized/stale data instead of actually re-reading \
                 the (changed) file, exactly the mmap-staleness risk this test isolates by \
                 mutating an already-loaded, already-decoded-against checkpoint in place rather \
                 than starting from two independently built checkpoints (unmutated={unmutated:?}, \
                 mutated={mutated:?})"
    );
}

// -------------------------------------------------------------------
// #682 Stage 2: prefetch overlap (hide I/O behind compute). Every
// routed expert's cache load for a token is now planned and
// dequantized up front (`ExpertSlotCache::prefetch_experts`), right
// after CPU routing and before Step 2's shared-expert GEMVs are
// encoded, instead of one at a time inside the Step 3 dispatch loop
// Stage 1 used — see `moe_expert_cache`'s module doc comment for why
// the plan/dequant split keeps the within-token touched-slot
// invariant intact even when the dequant phase fans out across
// rayon threads.
// -------------------------------------------------------------------

/// Pure cache-bookkeeping test: constructs two real `ExpertSlotCache`s
/// (Metal buffer allocation needs a real `Device`, hence
/// `gpu_test_lock()` — this file's blanket "every Metal-touching
/// test serializes" convention) but never encodes or dispatches a
/// single GPU command, so there is no numerics/timing surface to
/// race here — this is `plan_prefetch`'s bookkeeping logic in
/// isolation, not a decode-path integration test.
///
/// Builds two independently constructed `ExpertSlotCache`s over the
/// identical tiny synthetic tensor (same shape, same slot count),
/// replays the SAME sequence of `plan_prefetch` calls (simulating
/// three tokens' worth of routing decisions, engineered to force
/// both hits and an evicting miss) against both, and asserts:
/// (1) determinism — two independently built caches with identical
/// starting state produce the byte-identical sequence of
/// `(slot, expert_id)` prefetch tasks for the same inputs; (2)
/// non-overlap — within any single `plan_prefetch` call, every
/// returned task's slot is distinct, i.e. nothing is ever planned
/// to write into a slot another task from the same call also owns
/// (the invariant `prefetch_experts`'s parallel dequant phase
/// relies on to be race-free).
#[test]
fn moe_prefetch_plan_is_deterministic_and_non_overlapping() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    // 6 experts, a tiny per-expert-row tensor — shape
    // [experts, mid, cols], `cols` a 32-multiple (block-aligned)
    // like every other synthetic MoE fixture in this file.
    let experts = 6usize;
    let mid = 1usize;
    let cols = 32usize;
    let name = "prefetch_plan_test.experts.gate_up_proj";
    write_tiny_q4_fixture_per_expert_row(dir, name, experts, mid, cols, |e, _m| {
        0.01 * (e as f32 + 1.0)
    });
    let path = mtp_tensor_path(dir, name, "q4");

    let build_cache = || {
        crate::forward::moe_expert_cache::ExpertSlotCache::new(
            &device,
            &path,
            &[experts, mid, cols],
            3, // num_slots < num_experts: forces eviction below
            "prefetch_plan_test.cache",
        )
        .expect("ExpertSlotCache::new over the tiny fixture")
    };
    let mut cache_a = build_cache();
    let mut cache_b = build_cache();

    // Three "tokens" worth of top_k=3 selections: token 0 is
    // all-miss (cold cache, fills all 3 slots); token 1 mixes hits
    // (0, 1 still resident) with a miss (5) that must evict the one
    // untouched slot (2's, since 0 and 1 are protected by this same
    // call's hit-touch); token 2 is all-miss again after full
    // turnover.
    let token_selections: [&[usize]; 3] = [&[0, 1, 2], &[0, 1, 5], &[3, 4, 5]];

    for selection in token_selections {
        cache_a.begin_token();
        cache_b.begin_token();
        let tasks_a = cache_a.plan_prefetch(selection);
        let tasks_b = cache_b.plan_prefetch(selection);

        let as_pairs = |tasks: &[crate::forward::moe_expert_cache::PrefetchTask]| {
            tasks
                .iter()
                .map(|t| (t.slot, t.expert_id))
                .collect::<Vec<_>>()
        };
        let pairs_a = as_pairs(&tasks_a);
        let pairs_b = as_pairs(&tasks_b);
        assert_eq!(
            pairs_a, pairs_b,
            "plan_prefetch({selection:?}) must be deterministic: two independently \
                     built caches with identical starting state produced different \
                     (slot, expert_id) task lists"
        );

        let mut slots: Vec<usize> = pairs_a.iter().map(|&(slot, _)| slot).collect();
        let before = slots.len();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(
            slots.len(),
            before,
            "plan_prefetch({selection:?}) assigned the same slot to more than one miss \
                     task within a single call — this is exactly the race the pre-assignment \
                     step exists to prevent"
        );
    }
}

/// Integration parity test: the real `encode_moe_ffn` decode path,
/// under enough eviction pressure to force real hits/misses/
/// evictions across several tokens, must produce byte-identical
/// logits whether `ExpertSlotCache::prefetch_experts`'s dequant
/// phase fans out across rayon (the production default) or runs
/// the identical pre-assigned plan through a plain serial iterator
/// (`set_moe_prefetch_parallel_for_test(Some(false))`) — prefetch
/// must only change WHEN bytes are dequantized, never WHICH bytes
/// end up in which slot.
///
/// Mutation check (manually verified, not compiled in): forcing
/// `plan_prefetch` to assign every miss task `slot: 0` (breaking
/// the pre-assignment step, simulating a lost race between
/// concurrent dequant writes into the same slot) was checked by
/// hand. `prefetch_experts` still runs the collect-then-apply
/// sequence single-threaded either way (the current design never
/// actually races a write even when the plan is broken — it only
/// ever applies one result per slot at a time, "last result wins"
/// under the broken plan), so this particular mutation does NOT
/// reliably fail this decode-parity test — it was observed to pass
/// unchanged. What DOES catch it, immediately and deterministically,
/// is the plan-determinism/non-overlap test above
/// (`moe_prefetch_plan_is_deterministic_and_non_overlapping`),
/// which asserts every task in one `plan_prefetch` call has a
/// distinct slot — it failed loudly against the broken
/// `plan_prefetch` (`assigned the same slot to more than one miss
/// task`). That test is therefore the one this module actually
/// relies on to guard the pre-assignment invariant; this
/// decode-parity test's job is proving prefetch changes nothing
/// about correct output, not proving the invariant itself. The
/// change was reverted after confirming this.
#[test]
fn moe_prefetch_parallel_and_serial_produce_identical_decode_output() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config_top_k2();
    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    write_synthetic_moe_checkpoint(
        dir,
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );
    let tokenizer_path = std::path::Path::new("/dev/null");

    // ~50%-style miss pressure on a 4-expert, top_k=2 fixture: 2
    // slots means every token's 2 selections can never both be
    // resident from the previous token unless they're the exact
    // same pair, so this selection cycle forces a genuine mix of
    // hits and evicting misses across 5 decode steps.
    let selections: [Vec<usize>; 5] = [vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0], vec![0, 1]];
    let tokens: [u32; 5] = [1, 2, 3, 4, 1];

    let run = |parallel: bool| -> Vec<Vec<f32>> {
        let mut state = {
            let _env = EnvVarGuard::set(
                crate::forward::moe_expert_cache::MOE_EXPERT_CACHE_SLOTS_ENV,
                "2",
            );
            MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16)
                .expect("from_q4_dir (top_k=2, forced-eviction N=2) must load")
        };
        set_moe_prefetch_parallel_for_test(Some(parallel));
        let mut all_logits = Vec::new();
        for (position, (&token, selection)) in tokens.iter().zip(selections.iter()).enumerate() {
            let _forced = ForcedMoeExpertsGuard::set(selection.clone());
            let logits = state.forward_step(token, position);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "parallel={parallel} step {position} produced non-finite logits: \
                         {logits:?}"
            );
            all_logits.push(logits);
        }
        set_moe_prefetch_parallel_for_test(None);
        all_logits
    };

    let serial = run(false);
    let parallel = run(true);

    assert_eq!(
        serial, parallel,
        "prefetch must not change decode output, only timing: serial and rayon-\
                 parallel dequant fan-out over the identical pre-assigned plan produced \
                 different logits across a forced hit/miss/eviction decode sequence"
    );
}

/// A bigger MoE fixture than [`synthetic_moe_test_config`]'s
/// (`moe_inter=32`, negligible per-expert bytes) purely for the
/// throughput measurement below — the tiny correctness fixture's
/// dequant cost is too small relative to GEMV dispatch/encode
/// overhead to show any prefetch-overlap effect at all. Still far
/// smaller than a real checkpoint's per-expert size (§Stage 2's
/// PLAN.md caveat applies): `num_experts=16`, `moe_inter=256`,
/// `hidden=256`, `top_k=8` — per-expert gate_up+down f16 payload is
/// `2*256*256*2 + 256*256*2 ≈ 384 KiB`, dequantized from a Q4 file
/// roughly `120 KiB`/expert.
fn synthetic_moe_test_config_throughput() -> Qwen35Config {
    let mut cfg = synthetic_moe_test_config();
    cfg.hidden_size = 256;
    cfg.moe_intermediate_size = Some(256);
    cfg.shared_expert_intermediate_size = Some(64);
    cfg.num_experts = Some(16);
    cfg.num_experts_per_tok = Some(8);
    cfg
}

/// Latency-focused measurement from PLAN.md's Stage 2 test
/// strategy: decode tok/s with `ExpertSlotCache::prefetch_experts`
/// fanning its dequant phase out across rayon vs. running the
/// identical pre-assigned plan through a plain serial iterator, on
/// a synthetic checkpoint sized to force a controlled ~50% miss
/// rate per token.
///
/// `#[ignore]`d: this is a manual/reproducibility measurement, not
/// a correctness gate — wall-clock timing on a synthetic fixture
/// this small is noisy and not something CI should assert bounds
/// on. Run explicitly with `cargo test --features metal-gpu --lib
/// -- --ignored moe_prefetch_throughput`.
///
/// Miss-rate construction: `N=8` slots (`LATTICE_MOE_EXPERT_CACHE_SLOTS`),
/// exactly `top_k`, and the forced selection alternates between two
/// 8-expert sets overlapping by 4 (`{0..=7}` / `{4..=11}`) each
/// decode step. After the first step, every alternation is exactly
/// 4 hits (the overlapping 4, protected by this token's touch) and
/// 4 misses (forcing eviction of the other 4's untouched slots) —
/// a steady-state 50% per-token miss rate, matching the plan's
/// "N set so ~50% of each token's top-8 misses" spec.
#[test]
#[ignore = "manual throughput measurement, not a correctness gate"]
fn moe_prefetch_throughput_manual_benchmark() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config_throughput();
    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    write_synthetic_moe_checkpoint(
        dir,
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );
    let tokenizer_path = std::path::Path::new("/dev/null");

    let set_a: Vec<usize> = (0..8).collect();
    let set_b: Vec<usize> = (4..12).collect();

    const WARMUP_STEPS: usize = 6;
    const TIMED_STEPS: usize = 60;
    const N_REPEATS: usize = 3;

    let run_once = |parallel: bool| -> std::time::Duration {
        let mut state = {
            let _env = EnvVarGuard::set(
                crate::forward::moe_expert_cache::MOE_EXPERT_CACHE_SLOTS_ENV,
                "8",
            );
            MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 96)
                .expect("from_q4_dir (throughput fixture, N=8) must load")
        };
        set_moe_prefetch_parallel_for_test(Some(parallel));

        let mut position = 0usize;
        let mut next_token = |position: &mut usize, selection: Vec<usize>| -> Vec<f32> {
            let _forced = ForcedMoeExpertsGuard::set(selection);
            let logits = state.forward_step(1, *position);
            *position += 1;
            logits
        };

        for step in 0..WARMUP_STEPS {
            let selection = if step.is_multiple_of(2) {
                set_a.clone()
            } else {
                set_b.clone()
            };
            let _ = next_token(&mut position, selection);
        }

        let t0 = std::time::Instant::now();
        for step in 0..TIMED_STEPS {
            let selection = if step.is_multiple_of(2) {
                set_a.clone()
            } else {
                set_b.clone()
            };
            let logits = next_token(&mut position, selection);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "parallel={parallel} step {step} produced non-finite logits"
            );
        }
        let elapsed = t0.elapsed();

        set_moe_prefetch_parallel_for_test(None);
        elapsed
    };

    let mut serial_times = Vec::with_capacity(N_REPEATS);
    let mut parallel_times = Vec::with_capacity(N_REPEATS);
    for _ in 0..N_REPEATS {
        serial_times.push(run_once(false));
        parallel_times.push(run_once(true));
    }

    let median = |mut v: Vec<std::time::Duration>| -> std::time::Duration {
        v.sort_unstable();
        v[v.len() / 2]
    };
    let serial_median = median(serial_times.clone());
    let parallel_median = median(parallel_times.clone());
    let serial_tok_s = TIMED_STEPS as f64 / serial_median.as_secs_f64();
    let parallel_tok_s = TIMED_STEPS as f64 / parallel_median.as_secs_f64();

    eprintln!(
        "#682 Stage 2 prefetch throughput (synthetic fixture, N_REPEATS={N_REPEATS}, \
                 WARMUP_STEPS={WARMUP_STEPS}, TIMED_STEPS={TIMED_STEPS}, ~50% forced miss rate):\n\
                 \x20 serial   median={serial_median:?}  ({serial_tok_s:.1} tok/s)  all={serial_times:?}\n\
                 \x20 parallel median={parallel_median:?}  ({parallel_tok_s:.1} tok/s)  all={parallel_times:?}\n\
                 \x20 speedup = {:.3}x",
        serial_median.as_secs_f64() / parallel_median.as_secs_f64()
    );
}

/// Ordering proof: `encode_moe_ffn`'s
/// routed-expert dequant work must genuinely overlap Step 2's
/// shared-expert GEMV encoding, not just fan out across rayon
/// before Step 2 starts and then block in front of it. Proven
/// deterministically — no sleeps — via a channel handshake
/// installed on gate_up's dequant spawn only (proving it for one
/// cache is sufficient evidence of the mechanism; down's spawn is
/// unaffected):
///
/// `spawn_dequant`'s closure sends `started` and then blocks on
/// `release` before doing any real dequant work; `encode_moe_ffn`
/// sends `step2_done` immediately after Step 2 is fully encoded,
/// strictly before joining the dequant spawn(s). A controller
/// thread (owning only the channel endpoints — `MetalQwen35State`
/// cannot be moved across threads, Metal objects are not `Send`,
/// so `forward_step` itself runs on this test's own thread as
/// usual) waits for `started`, then waits for `step2_done`, and
/// only THEN sends `release`. If `step2_done` ever fires it can
/// only be because Step 2 encoding completed while the dequant
/// spawn was still blocked open (it cannot have returned before
/// `release` is sent) — i.e. genuine overlap, not
/// spawn-then-immediately-block-then-encode.
#[test]
fn moe_prefetch_step2_overlaps_dequant_deterministically() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config_top_k2();
    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    write_synthetic_moe_checkpoint(
        dir,
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );
    let tokenizer_path = std::path::Path::new("/dev/null");

    let mut state = {
        let _env = EnvVarGuard::set(
            crate::forward::moe_expert_cache::MOE_EXPERT_CACHE_SLOTS_ENV,
            "2",
        );
        MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16).expect("from_q4_dir must load")
    };

    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let (step2_tx, step2_rx) = std::sync::mpsc::channel();

    set_moe_prefetch_ordering_gate_for_test(Some(
        crate::forward::moe_expert_cache::PrefetchOrderingGate {
            started_tx,
            release_rx,
        },
    ));
    set_moe_prefetch_step2_done_tx_for_test(Some(step2_tx));

    let _forced = ForcedMoeExpertsGuard::set(vec![0, 1]);

    let controller = std::thread::spawn(move || {
        started_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect(
                "gate_up's dequant spawn never signaled `started` — spawn_dequant not \
                         wired up, or the test-only ordering gate was not installed/consumed",
            );
        step2_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect(
                "Step 2 never signaled `done` while the dequant spawn was still gated \
                         open — Step 2 encoding is NOT overlapping the dequant phase (exactly \
                         the regression this test guards against: a blocking join before Step \
                         2 is encoded)",
            );
        release_tx
            .send(())
            .expect("release_rx should still be listening — forward_step has not returned yet");
    });

    let logits = state.forward_step(1, 0);
    controller.join().expect("controller thread must not panic");

    assert!(
        logits.iter().all(|v| v.is_finite()),
        "overlapped decode produced non-finite logits: {logits:?}"
    );
}

/// Failure-isolation proof: a dequant
/// task panicking after `plan_prefetch` already committed its
/// slot's ownership bookkeeping must not let a later token read
/// stale/never-written bytes out of that slot.
///
/// Deliberately does NOT drive this through `encode_moe_ffn`/
/// `forward_step`: a panic that unwinds through a live, not-yet-
/// `endEncoding`'d `ComputeCommandEncoderRef` is an unrecoverable
/// Metal-level process abort (`-[_MTLCommandEncoder dealloc]:
/// failed assertion 'Command encoder released without
/// endEncoding'`), verified by hand — Metal's own safety net, not
/// something any amount of Rust-level `catch_unwind` can paper
/// over. Production's `encode_moe_ffn` still applies whichever
/// cache succeeded and calls `std::panic::resume_unwind` for a
/// real panic (see its comment), which is the right thing to do
/// for the CPU-side bookkeeping even though a real dequant panic
/// during actual GPU encoding is fatal to the process either way.
/// This test instead exercises the exact same
/// `plan_prefetch`/`spawn_dequant`/`apply_prefetch_results`
/// primitives `encode_moe_ffn` calls, directly, with no
/// `ComputeCommandEncoderRef` anywhere in the loop — the panic is
/// caught at `ScopedJoinHandle::join()`, on a background thread,
/// same as inside `encode_moe_ffn`, but nothing here ever unwinds
/// past a live encoder, so it's safe to assert on.
///
/// Sequence: token 0 loads experts `{0, 1}` cleanly (fills both of
/// a 2-slot cache). Token 1 plans `{2, 3}` (both misses, evicting
/// `{0, 1}`) and injects a dequant panic on expert 2; the join
/// must return `Err`, and — since rayon's `collect()` loses the
/// WHOLE batch's results on any one task's panic (see
/// `spawn_dequant`'s doc comment) — BOTH experts' slots (2's own
/// panic and 3's collateral loss) must stay `slot_ready == false`,
/// `get_prefetched` must refuse both. Token 2 re-plans `{2, 3}`
/// with no panic: `plan_prefetch` must treat both as reload-in-
/// place misses (same slots, `miss_count` up, no extra eviction),
/// and the recovered slot bytes must be BIT-IDENTICAL to an
/// independently built, never-panicked cache resolving the same
/// experts — proving the reload genuinely re-dequantized rather
/// than serving whatever was left behind.
#[test]
fn moe_prefetch_dequant_panic_recovers_via_readiness_reload() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    let experts = 4usize;
    let mid = 1usize;
    let cols = 32usize;
    let name = "prefetch_panic_recovery_test.experts.gate_up_proj";
    write_tiny_q4_fixture_per_expert_row(dir, name, experts, mid, cols, |e, _m| {
        0.01 * (e as f32 + 1.0)
    });
    let path = mtp_tensor_path(dir, name, "q4");

    let build_cache = || {
        crate::forward::moe_expert_cache::ExpertSlotCache::new(
            &device,
            &path,
            &[experts, mid, cols],
            2, // num_slots < num_experts: {2,3} necessarily evicts {0,1}
            "prefetch_panic_recovery_test.cache",
        )
        .expect("ExpertSlotCache::new over the tiny fixture")
    };

    let mut cache = build_cache();

    // Token 0: clean warm-up, fills both slots with {0, 1}.
    cache.begin_token();
    cache.prefetch_experts(&[0, 1], true);
    assert!(
        cache.debug_snapshot().slot_ready.iter().all(|&r| r),
        "warm-up token must leave every touched slot ready"
    );

    // Token 1: plan {2, 3} (evicts {0, 1}), spawn with expert 2's
    // dequant deliberately panicking.
    cache.begin_token();
    let tasks = cache.plan_prefetch(&[2, 3]);
    assert_eq!(tasks.len(), 2, "both 2 and 3 must be misses this token");
    let panicked_slots: Vec<usize> = tasks.iter().map(|t| t.slot).collect();

    let join_result = std::thread::scope(|scope| {
        let handle = cache.spawn_dequant(tasks, true, Some(2), None, scope);
        handle.expect("tasks is non-empty").join()
    });
    assert!(
        join_result.is_err(),
        "test-injected dequant panic for expert 2 must propagate out of spawn_dequant's \
                 join, not be silently swallowed"
    );

    let after_panic = cache.debug_snapshot();
    for &slot in &panicked_slots {
        assert!(
            !after_panic.slot_ready[slot],
            "slot {slot} must stay unready after its dequant task panicked (or was lost \
                     to a sibling task's panic in the same rayon collect())"
        );
    }
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cache.get_prefetched(2))).is_err(),
        "get_prefetched must refuse an unready slot rather than hand back \
                 uninitialized/stale bytes"
    );

    // Token 2: re-plan {2, 3} with no panic injected — must be
    // treated as reload-in-place misses (same slots, no further
    // eviction), not hits.
    cache.begin_token();
    let (_, miss_before, evict_before) = cache.hit_miss_eviction_counts();
    let tasks2 = cache.plan_prefetch(&[2, 3]);
    assert_eq!(
        tasks2.len(),
        2,
        "an unready mapping must be replanned as a miss even though expert_to_slot \
                 already contains it"
    );
    let reload_slots: Vec<usize> = tasks2.iter().map(|t| t.slot).collect();
    assert_eq!(
        {
            let mut a = reload_slots.clone();
            a.sort_unstable();
            let mut b = panicked_slots.clone();
            b.sort_unstable();
            a
        },
        {
            let mut b = panicked_slots.clone();
            b.sort_unstable();
            b
        },
        "reload must reuse the SAME slots the panicked experts already own, not evict \
                 anything new"
    );
    let (_, miss_after, evict_after) = cache.hit_miss_eviction_counts();
    assert_eq!(
        miss_after - miss_before,
        2,
        "recovery replan must count as 2 misses"
    );
    assert_eq!(
        evict_after, evict_before,
        "recovery replan must not evict anything — both experts already own their slots"
    );

    let join_result2 = std::thread::scope(|scope| {
        let handle = cache.spawn_dequant(tasks2, true, None, None, scope);
        handle.expect("tasks2 is non-empty").join()
    });
    let results2 = join_result2.expect("recovery dequant must not panic this time");
    cache.apply_prefetch_results(&results2);

    assert!(
        cache.debug_snapshot().slot_ready.iter().all(|&r| r),
        "every slot must be ready again after the recovery apply"
    );

    // Independently built, never-panicked cache: resolve the same
    // two experts directly and compare raw slot bytes.
    let mut clean_cache = build_cache();
    clean_cache.begin_token();
    let _ = clean_cache.resolve(2);
    let _ = clean_cache.resolve(3);

    for (&recovered_slot, expert_id) in reload_slots.iter().zip([2usize, 3usize]) {
        let clean_slot = *clean_cache
            .debug_snapshot()
            .expert_to_slot
            .iter()
            .find(|&&(e, _)| e == expert_id)
            .map(|(_, s)| s)
            .expect("clean_cache must have resolved this expert");
        assert_eq!(
            cache.slot_bits(recovered_slot),
            clean_cache.slot_bits(clean_slot),
            "expert {expert_id}'s recovered slot bytes must be bit-identical to a \
                     never-panicked cache's resolve() of the same expert — a mismatch would \
                     mean the reload served stale/uninitialized data instead of genuinely \
                     re-dequantizing"
        );
    }
}

/// Direct Stage-1-equivalence regression: the `plan_prefetch` + dequant + `apply_prefetch_results`
/// path must leave a cache in EXACTLY the same bookkeeping state
/// (`slot_owner`, `expert_to_slot`, `slot_touched`, `slot_ready`,
/// LRU order, hit/miss/eviction counters) as calling `resolve()`
/// once per expert, in the same order, would have — the whole
/// premise `plan_prefetch`'s doc comment relies on ("identical
/// side effects to what calling `resolve` once per id ... would
/// produce"), checked here directly rather than only inferred
/// from decode-output parity.
///
/// Drives two independently constructed caches over the identical
/// tiny fixture through the same multi-token sequence — including
/// a repeated id within one token (`5, 5, 4`: the second `5` is a
/// same-token hit against the first), a hit-then-miss token, and
/// eviction pressure (4 experts' worth of misses through 3
/// slots) — comparing `debug_snapshot()` after every token.
#[test]
fn moe_prefetch_plan_matches_resolve_bookkeeping_exactly() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    let experts = 6usize;
    let mid = 1usize;
    let cols = 32usize;
    let name = "prefetch_equivalence_test.experts.gate_up_proj";
    write_tiny_q4_fixture_per_expert_row(dir, name, experts, mid, cols, |e, _m| {
        0.01 * (e as f32 + 1.0)
    });
    let path = mtp_tensor_path(dir, name, "q4");

    let build_cache = || {
        crate::forward::moe_expert_cache::ExpertSlotCache::new(
            &device,
            &path,
            &[experts, mid, cols],
            3, // num_slots < num_experts: forces eviction below
            "prefetch_equivalence_test.cache",
        )
        .expect("ExpertSlotCache::new over the tiny fixture")
    };
    let mut cache_resolve = build_cache();
    let mut cache_prefetch = build_cache();

    let token_selections: [&[usize]; 4] = [&[0, 1, 2], &[0, 1, 5], &[3, 4, 5], &[5, 5, 4]];

    for selection in token_selections {
        cache_resolve.begin_token();
        cache_prefetch.begin_token();

        for &expert_id in selection {
            let _ = cache_resolve.resolve(expert_id);
        }
        cache_prefetch.prefetch_experts(selection, true);

        assert_eq!(
            cache_resolve.debug_snapshot(),
            cache_prefetch.debug_snapshot(),
            "after selection {selection:?}: plan_prefetch+dequant+apply diverged from \
                     an equivalent sequence of resolve() calls"
        );
    }
}

/// Regression: a duplicate COLD id within ONE `plan_prefetch` call
/// used to push a second `PrefetchTask` for
/// the same expert — the first occurrence assigns a slot but
/// leaves it `slot_ready == false` until `apply_prefetch_results`
/// runs (deferred, unlike `resolve()`'s synchronous `load_into`),
/// so the second occurrence's `expert_to_slot` lookup found an
/// UNREADY mapping and took the reload branch, pushing a second
/// task for the SAME slot — violating the distinct-slot contract
/// [`PrefetchTask`]'s doc comment claims ("every task in one
/// `plan_prefetch` call owns a distinct slot") and silently
/// double-mutating that slot's bookkeeping. The equivalence test
/// above never caught this: its one repeated id (`5, 5, 4`) was
/// already `slot_ready` from a PRIOR token, so it took the
/// ready-hit branch, not the buggy path. Production is unaffected
/// (`encode_moe_ffn`'s top-k selection never repeats an id within
/// one call), but the defensive contract and its claimed coverage
/// were wrong.
///
/// Plans a single call with a fresh (never-before-seen, i.e. cold)
/// duplicate id `[7, 7]` and asserts: exactly one `PrefetchTask`
/// (not two), the distinct-slot invariant holds trivially, and —
/// after dequanting and applying that one task — the cache's
/// bookkeeping matches an independently built cache driven by two
/// `resolve(7)` calls (the second of which is a hit against the
/// first's now-ready slot, exactly what the dedup guard makes the
/// second occurrence here equivalent to).
#[test]
fn moe_prefetch_plan_dedups_within_call_cold_duplicate() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    let experts = 8usize;
    let mid = 1usize;
    let cols = 32usize;
    let name = "prefetch_cold_dup_test.experts.gate_up_proj";
    write_tiny_q4_fixture_per_expert_row(dir, name, experts, mid, cols, |e, _m| {
        0.01 * (e as f32 + 1.0)
    });
    let path = mtp_tensor_path(dir, name, "q4");

    let build_cache = || {
        crate::forward::moe_expert_cache::ExpertSlotCache::new(
            &device,
            &path,
            &[experts, mid, cols],
            4,
            "prefetch_cold_dup_test.cache",
        )
        .expect("ExpertSlotCache::new over the tiny fixture")
    };

    let mut cache_prefetch = build_cache();
    cache_prefetch.begin_token();
    let tasks = cache_prefetch.plan_prefetch(&[7, 7]);
    assert_eq!(
        tasks.len(),
        1,
        "a cold id repeated within one plan_prefetch call must produce exactly one \
                 task, not one per occurrence"
    );
    let mut slots: Vec<usize> = tasks.iter().map(|t| t.slot).collect();
    let before = slots.len();
    slots.sort_unstable();
    slots.dedup();
    assert_eq!(
        slots.len(),
        before,
        "distinct-slot invariant: every task in one plan_prefetch call must own a \
                 distinct slot"
    );

    let results = std::thread::scope(|scope| {
        let handle = cache_prefetch.spawn_dequant(tasks, true, None, None, scope);
        handle.expect("tasks is non-empty").join()
    })
    .expect("dequant must not panic");
    cache_prefetch.apply_prefetch_results(&results);

    let mut cache_resolve = build_cache();
    cache_resolve.begin_token();
    let _ = cache_resolve.resolve(7);
    let _ = cache_resolve.resolve(7);

    assert_eq!(
        cache_resolve.debug_snapshot(),
        cache_prefetch.debug_snapshot(),
        "plan_prefetch([7, 7]) (deduped) must match two resolve(7) calls' bookkeeping \
                 exactly — the second resolve(7) is a hit against the first's now-ready slot, \
                 which is exactly what the within-call dedup guard makes the second [7, 7] \
                 occurrence equivalent to"
    );
}

// -------------------------------------------------------------------
// `load_q4_mmap_dequant_f16`/
// `load_q4_mmap_dequant_f32` used to validate a MoE tensor's Q4
// header only *self*-consistently (`validate_q4_header_payload_bounds`:
// `product(shape) == original_len`, payload physically present) —
// never against the shape `encode_moe_ffn` actually requires. A
// structurally valid but short or same-numel-transposed file would
// silently load, then violate `encode_moe_ffn`'s unsafe
// `num_experts * ...`-sized slice invariant. These two tests are
// regression coverage: a too-short router
// header, and a same-numel transposed routed-expert header, must
// both make `from_q4_dir` return an `Err` before `forward_step`.
// -------------------------------------------------------------------

#[test]
fn from_q4_dir_rejects_short_moe_router_header() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config();
    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    write_synthetic_moe_checkpoint(
        dir,
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );

    // Overwrite the router file with a self-consistent but SHORT
    // header: shape `[1, hidden]` instead of the required
    // `[num_experts, hidden]` — `product(shape) == original_len`
    // still holds (so `validate_q4_header_payload_bounds` alone
    // would accept it), but it is not the shape `load_moe_ffn_q4`
    // requires.
    write_tiny_q4_fixture_per_row(
        dir,
        "model.language_model.layers.0.mlp.gate.weight",
        1,
        cfg.hidden_size,
        |_| 0.42,
    );

    let tokenizer_path = std::path::Path::new("/dev/null");
    let Err(err) = MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16) else {
        panic!(
            "from_q4_dir must reject a short MoE router header before forward_step, \
                     not silently under-allocate encode_moe_ffn's router buffer"
        );
    };
    assert!(
        err.contains("shape"),
        "error must name the shape mismatch so it's diagnosable, got: {err}"
    );
}

#[test]
fn from_q4_dir_rejects_transposed_embedding_header() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config();
    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    write_synthetic_moe_checkpoint(
        dir,
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );
    write_tiny_q4_fixture(
        dir,
        "model.language_model.embed_tokens.weight",
        &[cfg.hidden_size, cfg.vocab_size],
    );

    let Err(err) = MetalQwen35State::from_q4_dir(dir, std::path::Path::new("/dev/null"), &cfg, 16)
    else {
        panic!("from_q4_dir must reject a same-numel transposed embedding tensor");
    };
    assert!(
        err.contains("expected") && err.contains("shape"),
        "error must report configured embedding geometry, got: {err}"
    );
}

#[test]
fn from_q4_dir_rejects_rank_mismatched_f16_norm_header() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config();
    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    write_synthetic_moe_checkpoint(
        dir,
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );
    write_tiny_f16_fixture(
        dir,
        "model.language_model.layers.0.input_layernorm.weight",
        &[1, cfg.hidden_size],
    );

    let Err(err) = MetalQwen35State::from_q4_dir(dir, std::path::Path::new("/dev/null"), &cfg, 16)
    else {
        panic!("from_q4_dir must reject an F16 norm with mismatched rank");
    };
    assert!(
        err.contains("expected") && err.contains("shape"),
        "error must report configured F16 geometry, got: {err}"
    );
}

#[test]
fn from_q4_dir_rejects_transposed_moe_routed_expert_header() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = synthetic_moe_test_config();
    let moe_inter = cfg.moe_intermediate_size();
    let hidden = cfg.hidden_size;
    let num_experts = cfg.num_experts.expect("fixture cfg must set num_experts");
    let tmp = tempfile::tempdir().expect("tempdir create");
    let dir = tmp.path();
    write_synthetic_moe_checkpoint(
        dir,
        &cfg,
        moe_fixture_gate_const,
        moe_fixture_up_const,
        moe_fixture_down_const,
    );

    // Overwrite `experts.down_proj` with a same-numel TRANSPOSED
    // header: `[E, I, H]` instead of the required `[E, H, I]` (both
    // have the same element count `E*H*I`, so a bounds-only check
    // cannot distinguish them — only an exact-shape check can).
    write_tiny_q4_fixture_per_expert_row(
        dir,
        "model.language_model.layers.0.mlp.experts.down_proj",
        num_experts,
        moe_inter,
        hidden,
        |_e, _m| 0.42,
    );

    let tokenizer_path = std::path::Path::new("/dev/null");
    let Err(err) = MetalQwen35State::from_q4_dir(dir, tokenizer_path, &cfg, 16) else {
        panic!(
            "from_q4_dir must reject a same-numel transposed routed-expert header \
                     before forward_step, not silently feed encode_moe_ffn a wrong-layout buffer"
        );
    };
    assert!(
        err.contains("shape"),
        "error must name the shape mismatch so it's diagnosable, got: {err}"
    );
}
