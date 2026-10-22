use super::*;
use crate::model::qwen35_config::Qwen35Config;

/// Mutation sensitivity: this test pins the exact byte formula for the
/// Qwen3.5-0.8B preset. Changing `conv_bytes_per_layer`, `s_bytes_per_layer`,
/// or the layer-count formulas in `try_from_config` fails this assertion.
#[test]
fn gdn_state_traffic_shape_qwen35_08b_matches_formula() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let shape = GdnStateTrafficShape::try_from_config(&cfg).expect("shape from 0.8B config");

    assert_eq!(shape.conv_bytes_per_layer, 73_728);
    assert_eq!(shape.s_bytes_per_layer, 1_048_576);
    assert_eq!(shape.active_state_bytes, 20_201_472);
}

/// `try_from_config` defaults `allocated_gdn_layers` to the architectural
/// `num_linear_attention_layers()` (the f16 constructor's allocation), so
/// pruning an active layer shrinks `active_gdn_layers` while leaving the
/// config-level `allocated_gdn_layers` unchanged. The Q4 constructor's
/// smaller allocation is covered by
/// `gdn_state_traffic_with_allocated_layers_matches_real_buffer_count`.
#[test]
fn gdn_state_traffic_shape_pruned_active_layers_keeps_allocated_copy_size() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    let pruned_idx = (0..cfg.num_hidden_layers)
        .find(|&i| !cfg.is_full_attention(i))
        .expect("0.8B preset has at least one linear-attention layer");
    cfg.layer_mask[pruned_idx] = false;

    let shape = GdnStateTrafficShape::try_from_config(&cfg).expect("shape from pruned config");

    assert_eq!(shape.active_gdn_layers, 17);
    assert_eq!(shape.allocated_gdn_layers, 18);
    assert_ne!(shape.active_state_bytes, shape.allocated_state_bytes);
}

/// The Q4 constructor allocates only `num_active_linear_attention_layers()`
/// GDN buffers, so its checkpoint/restore blit copies that many. Mutation
/// sensitivity: `with_allocated_layers` must re-point `allocated_state_bytes`
/// at the real buffer count so `record_copy` does not overcount a pruned Q4
/// session (dropping the recompute leaves the architectural 18-layer total).
#[test]
fn gdn_state_traffic_with_allocated_layers_matches_real_buffer_count() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    let pruned_idx = (0..cfg.num_hidden_layers)
        .find(|&i| !cfg.is_full_attention(i))
        .expect("0.8B preset has at least one linear-attention layer");
    cfg.layer_mask[pruned_idx] = false;

    let active = cfg.num_active_linear_attention_layers();
    let shape = GdnStateTrafficShape::try_from_config(&cfg)
        .expect("shape from pruned config")
        .with_allocated_layers(active);

    assert_eq!(active, 17);
    assert_eq!(shape.allocated_gdn_layers, 17);
    assert_eq!(
        shape.allocated_state_bytes,
        shape.per_layer_state_bytes * 17
    );
    assert_eq!(shape.allocated_state_bytes, shape.active_state_bytes);
}

/// Mutation sensitivity: removing any of the decode layer-access, MTP
/// layer-access, snapshot, or restore increments changes an asserted
/// value. Uses `record_layer_access` (the production accounting fn
/// called from all decode/MTP-verify dispatch sites) with
/// `active_gdn_layers` so byte totals equal one full active-state pass.
#[test]
fn gdn_state_traffic_counts_decode_and_rejected_mtp_verify() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let shape = GdnStateTrafficShape::try_from_config(&cfg).expect("shape from 0.8B config");
    let active_layers = shape.active_gdn_layers;
    let mut counters = GdnStateTrafficCounters::new(shape);

    counters.record_layer_access(GdnStateTrafficScope::Decode, active_layers);

    counters.record_copy(GdnStateTrafficScope::MtpVerify, GdnStateCopyKind::Snapshot);
    for _ in 0..2 {
        counters.record_layer_access(GdnStateTrafficScope::MtpVerify, active_layers);
        counters.record_copy(GdnStateTrafficScope::MtpVerify, GdnStateCopyKind::Snapshot);
    }
    counters.record_copy(GdnStateTrafficScope::MtpVerify, GdnStateCopyKind::Restore);

    let report = counters.report();
    assert_eq!(report.decode.read_bytes, 20_201_472);
    assert_eq!(report.decode.write_bytes, 20_201_472);
    assert_eq!(report.decode.copy_count(), 0);
    assert_eq!(report.mtp_verify.snapshot_count, 3);
    assert_eq!(report.mtp_verify.restore_count, 1);
    assert_eq!(report.mtp_verify.read_bytes, 121_208_832);
    assert_eq!(report.mtp_verify.write_bytes, 121_208_832);

    // reset() must zero every bucket field while preserving shape — this is
    // the accumulation-window boundary take_gdn_state_traffic_report relies
    // on after handing back a report.
    counters.reset();
    let cleared = counters.report();
    assert_eq!(cleared.decode.read_bytes, 0);
    assert_eq!(cleared.decode.write_bytes, 0);
    assert_eq!(cleared.mtp_verify.read_bytes, 0);
    assert_eq!(cleared.mtp_verify.write_bytes, 0);
    assert_eq!(cleared.mtp_verify.snapshot_count, 0);
    assert_eq!(cleared.mtp_verify.restore_count, 0);
    assert_eq!(cleared.shape, report.shape);
}
