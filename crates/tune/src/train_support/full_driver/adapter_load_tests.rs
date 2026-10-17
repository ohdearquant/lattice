use super::*;
use crate::lora::{LoraAdapter, LoraConfig, LoraLayer};
use std::collections::HashMap;

const RANK: usize = 2;

fn dims() -> Dims {
    Dims {
        hidden: 4,
        vocab: 32,
        num_q_heads: 3,
        num_kv_heads: 1,
        head_dim: 2,
        rope_dim: 2,
        inter: 8,
        q_dim: 3,
        kv_dim: 2,
        eps: 1e-6,
    }
}

fn gdn_dims() -> GdnDims {
    GdnDims {
        num_kh: 1,
        value_heads: 2,
        key_dim: 2,
        value_dim: 2,
        qkv_dim: 5,
        output_dim: 3,
        kernel_size: 2,
        scale: 1.0,
    }
}

/// Distinct, position-dependent values: a loader that swapped two arrays, or
/// wrote A where B belongs, is caught by the VALUES rather than by the lengths.
fn ramp(tag: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| tag + i as f32 / 128.0).collect()
}

fn layer(tag: f32, d_in: usize, d_out: usize) -> LoraLayer {
    LoraLayer {
        a: ramp(tag, RANK * d_in),
        b: ramp(tag + 0.5, d_out * RANK),
        d_in,
        d_out,
        rank: RANK,
    }
}

/// Exactly the set the save block writes for this selection, built from the same
/// geometry `run` would pass.
fn saved(selection: GdnModuleSelection) -> HashMap<(usize, String), LoraLayer> {
    let (d, g) = (dims(), gdn_dims());
    let mut m = HashMap::new();
    m.insert(
        (23, "q_proj".to_string()),
        layer(1.0, d.hidden, 2 * d.q_dim),
    );
    m.insert((23, "v_proj".to_string()), layer(2.0, d.hidden, d.kv_dim));
    m.insert(
        (20, "in_proj_qkv".to_string()),
        layer(3.0, d.hidden, g.qkv_dim),
    );
    m.insert(
        (20, "in_proj_z".to_string()),
        layer(4.0, d.hidden, g.output_dim),
    );
    m.insert(
        (20, "out_proj".to_string()),
        layer(5.0, g.output_dim, d.hidden),
    );
    if selection.trains_fused_kernel_projections() {
        m.insert(
            (20, "in_proj_b".to_string()),
            layer(6.0, d.hidden, g.value_heads),
        );
        m.insert(
            (20, "in_proj_a".to_string()),
            layer(7.0, d.hidden, g.value_heads),
        );
    }
    m
}

fn adapter(layers: HashMap<(usize, String), LoraLayer>) -> LoraAdapter {
    let config = LoraConfig {
        rank: RANK,
        alpha: 4.0,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        dtype: "f32".into(),
    };
    LoraAdapter::new(config, layers).expect("fixture adapter")
}

fn fresh() -> (Vec<LoraParams>, Vec<GdnLoraParams>) {
    let (d, g) = (dims(), gdn_dims());
    let (loras, rng) = initial_training_loras(7, 1, &d, RANK, 0.25);
    let gdn = zero_b_gdn_loras(1, RANK, d.hidden, &g, rng, 0.25);
    (loras, gdn)
}

fn overwrite(
    adapter: &LoraAdapter,
    selection: GdnModuleSelection,
) -> Result<(usize, Vec<LoraParams>, Vec<GdnLoraParams>), String> {
    let (d, g) = (dims(), gdn_dims());
    let (mut loras, mut gdn) = fresh();
    let consumed = overwrite_from_adapter(
        adapter,
        &mut loras,
        &[23],
        &mut gdn,
        &[20],
        &d,
        &g,
        RANK,
        selection,
    )?;
    Ok((consumed, loras, gdn))
}

/// Neither weight struct implements `Debug`, and deriving it so that `unwrap`
/// compiles would dump every array on a failure. These two unwrap by hand.
fn expect_ok(
    r: Result<(usize, Vec<LoraParams>, Vec<GdnLoraParams>), String>,
) -> (usize, Vec<LoraParams>, Vec<GdnLoraParams>) {
    match r {
        Ok(v) => v,
        Err(e) => panic!("expected the load to succeed, got: {e}"),
    }
}

fn expect_err(r: Result<(usize, Vec<LoraParams>, Vec<GdnLoraParams>), String>) -> String {
    match r {
        Ok((consumed, _, _)) => {
            panic!("expected a refusal, but the load consumed {consumed} entries")
        }
        Err(e) => e,
    }
}

#[test]
fn every_served_array_lands_with_the_saved_values() {
    let selection = GdnModuleSelection::Served;
    let (before_gqa, before_gdn) = fresh();
    let (consumed, loras, gdn) = expect_ok(overwrite(&adapter(saved(selection)), selection));
    assert_eq!(consumed, 5, "2 GQA + 3 served GDN arrays");
    let (d, g) = (dims(), gdn_dims());

    assert_eq!(loras[0].a_q, ramp(1.0, RANK * d.hidden));
    assert_eq!(loras[0].b_q, ramp(1.5, 2 * d.q_dim * RANK));
    assert_eq!(loras[0].a_v, ramp(2.0, RANK * d.hidden));
    assert_eq!(loras[0].b_v, ramp(2.5, d.kv_dim * RANK));
    assert_eq!(gdn[0].a_qkv, ramp(3.0, RANK * d.hidden));
    assert_eq!(gdn[0].b_qkv, ramp(3.5, g.qkv_dim * RANK));
    assert_eq!(gdn[0].a_z, ramp(4.0, RANK * d.hidden));
    assert_eq!(gdn[0].b_z, ramp(4.5, g.output_dim * RANK));
    assert_eq!(gdn[0].a_out, ramp(5.0, RANK * g.output_dim));
    assert_eq!(gdn[0].b_out, ramp(5.5, d.hidden * RANK));

    // The control that makes the ten assertions above mean something: the fresh
    // state they replaced was not already equal to the saved values.
    assert_ne!(before_gqa[0].a_q, loras[0].a_q);
    assert_ne!(before_gdn[0].a_qkv, gdn[0].a_qkv);
    // Untrained under `served`, and untouched: left at their fresh values rather
    // than zeroed, which is what the step loop and the save block both assume.
    assert_eq!(gdn[0].a_b, before_gdn[0].a_b);
    assert_eq!(gdn[0].a_a, before_gdn[0].a_a);
}

#[test]
fn the_fused_projections_are_required_exactly_when_this_run_trains_them() {
    let all = GdnModuleSelection::All;
    let (consumed, _, gdn) = expect_ok(overwrite(&adapter(saved(all)), all));
    assert_eq!(consumed, 7);
    let (d, g) = (dims(), gdn_dims());
    assert_eq!(gdn[0].a_b, ramp(6.0, RANK * d.hidden));
    assert_eq!(gdn[0].b_a, ramp(7.5, g.value_heads * RANK));

    // An `all` adapter into a `served` run: the two fused entries are real
    // trained weights this run would never read. Dropping them silently is the
    // exact failure the consumed-key accounting exists to refuse.
    let err = expect_err(overwrite(&adapter(saved(all)), GdnModuleSelection::Served));
    assert!(err.contains("unread: in_proj_a@20, in_proj_b@20"), "{err}");

    // And the other direction: a `served` adapter cannot satisfy an `all` run.
    let err = expect_err(overwrite(&adapter(saved(GdnModuleSelection::Served)), all));
    assert!(err.contains("carries no in_proj_b at layer 20"), "{err}");
}

#[test]
fn a_missing_entry_or_a_mismatched_shape_refuses_rather_than_partially_loading() {
    let selection = GdnModuleSelection::Served;

    let mut missing = saved(selection);
    missing.remove(&(23, "v_proj".to_string()));
    let err = expect_err(overwrite(&adapter(missing), selection));
    assert!(err.contains("carries no v_proj at layer 23"), "{err}");

    let (d, g) = (dims(), gdn_dims());
    let mut wrong_rank = saved(selection);
    wrong_rank.insert(
        (23, "q_proj".to_string()),
        LoraLayer {
            // rank 1, so this is rank * d_in with rank == 1.
            a: vec![0.0; d.hidden],
            b: vec![0.0; 2 * d.q_dim],
            d_in: d.hidden,
            d_out: 2 * d.q_dim,
            rank: 1,
        },
    );
    let err = expect_err(overwrite(&adapter(wrong_rank), selection));
    assert!(err.contains("this run is rank 2"), "{err}");

    // Same rank, wrong width: a rank check alone would pass this.
    let mut wrong_width = saved(selection);
    wrong_width.insert(
        (20, "in_proj_z".to_string()),
        layer(4.0, d.hidden, g.output_dim + 1),
    );
    let err = expect_err(overwrite(&adapter(wrong_width), selection));
    assert!(err.contains("in_proj_z at layer 20"), "{err}");

    // Right module, wrong LAYER: the adapter was trained at a different
    // --first-layer, so every array is present and none of them belongs here.
    let mut wrong_layer = saved(selection);
    let moved = wrong_layer.remove(&(23, "q_proj".to_string())).unwrap();
    wrong_layer.insert((22, "q_proj".to_string()), moved);
    let err = expect_err(overwrite(&adapter(wrong_layer), selection));
    assert!(err.contains("carries no q_proj at layer 23"), "{err}");
}

/// The score-level acceptance this lane ran (train, save, load, score the same
/// data with `--steps 0`) reproduced the post-training NLL to every printed
/// digit, but the log prints four decimals and cannot resolve the 2.9e-6
/// tolerance it was measured against. This arm closes that gap from the other
/// side and with a stronger claim: through the REAL serializer and the REAL
/// loader, every array comes back bit-identical, so the forward cannot differ
/// at any tolerance. Together the two cover the whole chain, because this test
/// does not exercise the save block's own (slot, module) mapping and the
/// measured run does.
#[test]
fn a_real_file_round_trip_returns_every_array_bit_identical() {
    let selection = GdnModuleSelection::Served;
    let layers = saved(selection);
    let dir = std::env::temp_dir().join(format!("lattice-rt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tempdir");
    let path = dir.join("roundtrip.safetensors");

    adapter(layers.clone())
        .save_safetensors(&path, None)
        .expect("save");
    let reloaded = crate::lora::load_peft_safetensors(&path).expect("load");

    let (d, g) = (dims(), gdn_dims());
    let (mut loras, mut gdn) = fresh();
    let consumed = overwrite_from_adapter(
        &reloaded,
        &mut loras,
        &[23],
        &mut gdn,
        &[20],
        &d,
        &g,
        RANK,
        selection,
    )
    .expect("overwrite from the reloaded file");
    assert_eq!(consumed, 5);

    // Bit-identical, not approximately equal: f32 written and read back with no
    // dtype conversion has no rounding step to excuse a difference.
    for (module, got, want) in [
        (
            "q_proj.a",
            &loras[0].a_q,
            &layers[&(23, "q_proj".to_string())].a,
        ),
        (
            "q_proj.b",
            &loras[0].b_q,
            &layers[&(23, "q_proj".to_string())].b,
        ),
        (
            "v_proj.a",
            &loras[0].a_v,
            &layers[&(23, "v_proj".to_string())].a,
        ),
        (
            "v_proj.b",
            &loras[0].b_v,
            &layers[&(23, "v_proj".to_string())].b,
        ),
        (
            "in_proj_qkv.a",
            &gdn[0].a_qkv,
            &layers[&(20, "in_proj_qkv".to_string())].a,
        ),
        (
            "in_proj_qkv.b",
            &gdn[0].b_qkv,
            &layers[&(20, "in_proj_qkv".to_string())].b,
        ),
        (
            "in_proj_z.a",
            &gdn[0].a_z,
            &layers[&(20, "in_proj_z".to_string())].a,
        ),
        (
            "in_proj_z.b",
            &gdn[0].b_z,
            &layers[&(20, "in_proj_z".to_string())].b,
        ),
        (
            "out_proj.a",
            &gdn[0].a_out,
            &layers[&(20, "out_proj".to_string())].a,
        ),
        (
            "out_proj.b",
            &gdn[0].b_out,
            &layers[&(20, "out_proj".to_string())].b,
        ),
    ] {
        let g: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
        let w: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
        assert_eq!(g, w, "{module} did not survive the file round trip");
    }

    std::fs::remove_file(&path).ok();
    std::fs::remove_dir(&dir).ok();
}

#[test]
fn a_module_the_save_block_writes_and_this_loader_ignores_is_refused() {
    // Stand-in for the real drift: someone adds a module to the save block and
    // not here. Without the accounting the run would start from a half-loaded
    // adapter and the number would be blamed on the data.
    let selection = GdnModuleSelection::Served;
    let (d, _) = (dims(), gdn_dims());
    let mut extra = saved(selection);
    extra.insert((23, "o_proj".to_string()), layer(9.0, d.hidden, d.hidden));
    let err = expect_err(overwrite(&adapter(extra), selection));
    assert!(err.contains("6 layer/module entries"), "{err}");
    assert!(err.contains("unread: o_proj@23"), "{err}");
}
