//! R3 orientation reference — issue #703 PR1's named open question.
//!
//! The design doc (§B "Insertion-point analysis", §D "R3 mapping with GQA,
//! cache and gated attention is underspecified") flags that QuaRot's R3
//! ("Hadamard heads" on the attention output) has more than one plausible
//! insertion point once Qwen3.5's output-gated attention is in the picture:
//! rotate the cached value stream, rotate the gated context right before
//! `o_proj`, or some mix. Only one of those is an exact numerical
//! reparameterization of the dense (unrotated) forward — the others silently
//! break because a Hadamard rotation does not commute with the per-dimension
//! sigmoid gate (`ctx *= sigmoid(gate_z)`) unless the rotation is inserted
//! **after** the gate is applied.
//!
//! ## The contract this module proves and records
//!
//! For one full-attention Qwen3.5-0.8B layer's attention-axis shape (GQA:
//! `num_attention_heads=8`, `num_key_value_heads=2`, `head_dim=256`, so
//! `q_dim=2048`, `kv_dim=512`, groups=4 — read from
//! [`crate::model::qwen35_config::Qwen35Config::qwen35_0_8b`]; the test
//! fixture's `HIDDEN` constant is intentionally smaller than the real
//! `hidden_size=1024` — see that constant's doc for why), the winning
//! orientation is:
//!
//! **R3 rotates the gated attention output (`context * sigmoid(gate_z)`),
//! immediately before `o_proj`. `o_proj` absorbs the rotation's transpose on
//! its INPUT side** — i.e. exactly [`crate::quant::quarot::plan::AbsorptionSide::InputSide`]
//! applied to `o_proj`'s `q_dim` axis, matching `rotation.rs`'s documented
//! `y = W · R^T · (R · x)` identity with `x` = the gated context. This is
//! recorded in [`crate::quant::quarot::plan::OnlineRotationSpec::r3_full_attention`]
//! as `side: AbsorptionSide::InputSide`.
//!
//! ## The rotation is cross-head, not intra-head
//!
//! The Kronecker **axis** matters and is easy to get backwards: QuaRot's
//! online "Hadamard heads" operation (paper Eq. 9, arXiv:2404.00456 Stage
//! 1c) is `Z ← Z · (H_num_heads ⊗ I_head_dim)` — it mixes values **across
//! heads at each fixed within-head channel index**, not within a single
//! head's own channels. Verbatim from the paper: "it remains to apply
//! `(H_nh ⊗ I)` to `W_out`, which results in a complete transformation of
//! `W_out ← H W_out`, and to insert a block into the forward pass that
//! computes `Z ← Z(H_nh ⊗ I)` where `Z` is the attention activation. This
//! block is denoted Hadamard heads in Figure 6." The per-head block-diagonal
//! `(I_num_heads ⊗ H_head_dim)` factor from the same section (Eq. 7-8) is a
//! *different*, purely offline operation fused into `W_v`/`W_out` with zero
//! online cost — it is not what "Hadamard heads" refers to, and a rotation
//! built from independently-seeded per-head `head_dim` blocks (the
//! `I ⊗ H_head_dim` axis) is a different transform than QuaRot's R3, even
//! though — as this module's own enumeration test demonstrates — either
//! orthogonal transform reparameterizes exactly through `o_proj`'s
//! counter-rotation and so cannot be told apart by a cancellation test
//! alone. `candidate_post_gate_context` below therefore builds the rotation
//! from a genuine cross-head [`RandomizedHadamard`] over the
//! `num_attention_heads` axis (applied identically to every `head_dim`
//! channel), not a [`BlockHadamard`] over contiguous `head_dim` blocks.
//!
//! A cache-side (value-path) rotation is *compatible* with this orientation
//! **only if it is inverted before the gate is applied** — rotating V at
//! cache-write time and then undoing that exact rotation on `context` before
//! gating is a mathematical no-op on the forward pass (see the
//! `value_round_trip_then_post_gate` candidate below), so a future PR can use
//! it to reduce KV-cache outliers without changing this contract. Rotating V
//! and leaving that rotation in place through the gate does NOT work — the
//! gate's per-dimension sigmoid does not commute with a cross-dimension
//! Hadamard mix (see the `pre_gate_value_only` candidate). The value-path
//! round trip in `value_round_trip_then_post_gate` uses a per-KV-head
//! `head_dim`-block rotation (QuaRot's *offline* Eq. 7-8 factor) purely as a
//! KV-cache-outlier companion; it is independent of, and composes cleanly
//! with, the cross-head R3 rotation applied afterward.
//!
//! ## Scope note
//!
//! RoPE is intentionally omitted from this synthetic forward: RoPE only
//! touches Q/K (the score computation), which is orthogonal to the question
//! this module answers (whether a rotation of the value/context/gate/O path
//! commutes with the gate). Q/K are still projected and used to produce a
//! non-trivial 3-position softmax, so the attention-weighted-sum step is
//! exercised for real, not stubbed to an identity.

#[cfg(test)]
mod tests;
