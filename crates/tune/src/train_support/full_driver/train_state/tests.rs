use super::*;

fn stepped_adam() -> AdamState {
    let mut adam = AdamState::new();
    let mut a = vec![0.25f32, -0.5, 0.125];
    let mut b = vec![1.0f32; 5];
    for k in 0..3 {
        let ga: Vec<f32> = (0..3).map(|i| 0.01 * (i + k) as f32 - 0.02).collect();
        let gb: Vec<f32> = (0..5).map(|i| 0.003 * (i * k) as f32 + 0.001).collect();
        adam.step("l3_a_q", &mut a, &ga, 1e-3, 0.9, 0.999, 1e-8, 0.0, false);
        adam.step("l3_b_q", &mut b, &gb, 1e-3, 0.9, 0.999, 1e-8, 0.0, false);
    }
    adam
}

fn state() -> TrainState {
    TrainState {
        next_step: 77,
        train_count: 376,
        data: [1; 32],
        config: [2; 32],
        adapter: [3; 32],
        adam: stepped_adam(),
    }
}

fn sample(tokens: &[u32], completion_start: usize) -> Sample {
    Sample {
        tokens: tokens.to_vec(),
        completion_start,
    }
}

#[test]
fn state_round_trips_every_field() {
    let original = state();
    let bytes = original.to_bytes();
    let back = TrainState::from_bytes(&bytes).unwrap();
    assert_eq!(back.next_step, 77);
    assert_eq!(back.train_count, 376);
    assert_eq!(back.data, [1; 32]);
    assert_eq!(back.config, [2; 32]);
    assert_eq!(back.adapter, [3; 32]);
    assert_eq!(back.adam.to_bytes(), original.adam.to_bytes());
    assert_eq!(back.to_bytes(), bytes);
}

#[test]
fn state_refuses_damage_anywhere() {
    let bytes = state().to_bytes();
    for at in [
        0,
        9,
        13,
        21,
        40,
        120,
        HEADER_LEN + 3,
        bytes.len() - 40,
        bytes.len() - 1,
    ] {
        let mut damaged = bytes.clone();
        damaged[at] ^= 0x10;
        let err = TrainState::from_bytes(&damaged).err().unwrap();
        assert!(err.contains("checksum"), "byte {at}: {err}");
    }
    for len in [0, 10, HEADER_LEN, bytes.len() - 1] {
        assert!(
            TrainState::from_bytes(&bytes[..len]).is_err(),
            "length {len}"
        );
    }
}

/// Re-seal a modified body with a valid checksum, so a test reaches the field
/// check behind the checksum instead of stopping at it.
fn resealed(mut body: Vec<u8>) -> Vec<u8> {
    let digest = Sha256::digest(&body);
    body.extend_from_slice(&digest);
    body
}

#[test]
fn state_refuses_foreign_or_inconsistent_contents_behind_a_valid_checksum() {
    let bytes = state().to_bytes();
    let body = bytes[..bytes.len() - DIGEST_LEN].to_vec();

    let mut magic = body.clone();
    magic[0] = b'X';
    assert!(
        TrainState::from_bytes(&resealed(magic))
            .err()
            .unwrap()
            .contains("not a training state")
    );

    let mut version = body.clone();
    version[8] = 2;
    assert!(
        TrainState::from_bytes(&resealed(version))
            .err()
            .unwrap()
            .contains("version 2")
    );

    let mut zero_step = body.clone();
    zero_step[12..20].copy_from_slice(&0u64.to_le_bytes());
    assert!(
        TrainState::from_bytes(&resealed(zero_step))
            .err()
            .unwrap()
            .contains("step 0")
    );

    let mut short_adam = body.clone();
    short_adam.pop();
    assert!(
        TrainState::from_bytes(&resealed(short_adam))
            .err()
            .unwrap()
            .contains("optimizer bytes")
    );
}

#[test]
fn write_atomic_then_read_returns_the_same_state() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.state");
    let original = state();
    original.write_atomic(&path).unwrap();
    assert!(!dir.path().join("run.state.partial").exists());
    let back = TrainState::read(&path).unwrap();
    assert_eq!(back.to_bytes(), original.to_bytes());
}

#[test]
fn write_atomic_refuses_a_leftover_partial_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.state");
    let partial = dir.path().join("run.state.partial");
    std::fs::write(&path, b"previous state").unwrap();
    std::fs::write(&partial, b"leftover partial").unwrap();
    let err = state().write_atomic(&path).unwrap_err();
    assert!(err.contains(&partial.display().to_string()), "{err}");
    assert_eq!(std::fs::read(&partial).unwrap(), b"leftover partial");
    assert_eq!(std::fs::read(&path).unwrap(), b"previous state");
}

#[cfg(unix)]
#[test]
fn write_atomic_refuses_a_partial_symlink_without_touching_its_target() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.state");
    let partial = dir.path().join("run.state.partial");
    let target = dir.path().join("owned-target");
    std::fs::write(&target, b"unchanged target").unwrap();
    std::os::unix::fs::symlink(&target, &partial).unwrap();
    let err = state().write_atomic(&path).unwrap_err();
    assert!(err.contains(&partial.display().to_string()), "{err}");
    assert_eq!(std::fs::read(&target).unwrap(), b"unchanged target");
    assert_eq!(std::fs::read_link(&partial).unwrap(), target);
    assert!(!path.exists());
}

#[cfg(unix)]
#[test]
fn write_atomic_creates_a_private_state_file() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.state");
    state().write_atomic(&path).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
}

#[test]
fn step_window_lists_the_samples_the_steps_visit() {
    assert_eq!(step_window(1, 3, 5), vec![0, 1, 2]);
    assert_eq!(step_window(4, 3, 5), vec![0, 3, 4]);
    assert_eq!(step_window(377, 2, 376), vec![0, 1]);
    assert_eq!(step_window(2, 9, 4), vec![0, 1, 2, 3]);
    assert_eq!(step_window(6, 0, 4), vec![1]);
    assert_eq!(step_window(1, 3, 0), Vec::<usize>::new());
}

#[test]
fn step_window_has_exactly_one_entry_per_visited_sample() {
    for count in 1..=9 {
        for start in 1..=3 * count {
            for steps in 1..=3 * count {
                let window = step_window(start, steps, count);
                assert_eq!(window.len(), steps.min(count));
                assert!(window.iter().all(|&i| i < count));
                assert!(window.windows(2).all(|pair| pair[0] < pair[1]));
                let distinct: std::collections::BTreeSet<_> = window.into_iter().collect();
                assert_eq!(distinct.len(), steps.min(count));
            }
        }
    }
}

#[test]
fn data_fingerprint_sees_tokens_boundaries_and_order() {
    let base = data_fingerprint(&[sample(&[1, 2, 3], 1), sample(&[4, 5], 1)]);
    assert_eq!(
        base,
        data_fingerprint(&[sample(&[1, 2, 3], 1), sample(&[4, 5], 1)])
    );
    assert_ne!(
        base,
        data_fingerprint(&[sample(&[1, 2, 4], 1), sample(&[4, 5], 1)])
    );
    assert_ne!(
        base,
        data_fingerprint(&[sample(&[1, 2, 3], 2), sample(&[4, 5], 1)])
    );
    assert_ne!(
        base,
        data_fingerprint(&[sample(&[4, 5], 1), sample(&[1, 2, 3], 1)])
    );
    assert_ne!(base, data_fingerprint(&[sample(&[1, 2, 3], 1)]));
}

fn dims() -> Dims {
    Dims {
        hidden: 8,
        vocab: 16,
        num_q_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        rope_dim: 4,
        inter: 12,
        q_dim: 8,
        kv_dim: 4,
        eps: 1e-6,
    }
}

#[test]
fn config_fingerprint_sees_every_setting() {
    let d = dims();
    let id = |lr: f32, seed: u64, sel: GdnModuleSelection, gdn: &'static [usize]| {
        config_fingerprint(&RunIdentity {
            first_layer: 19,
            rank: 8,
            alpha: 16.0,
            lr,
            seed,
            gdn_modules: sel,
            dims: &d,
            slot_layers: &[19, 23],
            gdn_slot_layers: gdn,
        })
    };
    let base = id(1e-3, 7, GdnModuleSelection::Served, &[20, 21, 22]);
    assert_eq!(base, id(1e-3, 7, GdnModuleSelection::Served, &[20, 21, 22]));
    assert_ne!(base, id(5e-4, 7, GdnModuleSelection::Served, &[20, 21, 22]));
    assert_ne!(base, id(1e-3, 8, GdnModuleSelection::Served, &[20, 21, 22]));
    assert_ne!(base, id(1e-3, 7, GdnModuleSelection::All, &[20, 21, 22]));
    assert_ne!(base, id(1e-3, 7, GdnModuleSelection::Served, &[20, 21]));
}

fn gdn(fill: f32) -> GdnLoraParams {
    let v = |n: usize| vec![fill; n];
    GdnLoraParams {
        a_qkv: v(4),
        b_qkv: v(4),
        a_z: v(4),
        b_z: v(4),
        a_b: v(2),
        b_b: v(2),
        a_a: v(2),
        b_a: v(2),
        a_out: v(4),
        b_out: v(4),
    }
}

fn lora(fill: f32) -> LoraParams {
    LoraParams {
        a_q: vec![fill; 4],
        b_q: vec![fill; 4],
        a_v: vec![fill; 4],
        b_v: vec![fill; 4],
    }
}

#[test]
fn adapter_fingerprint_covers_exactly_the_saved_projections() {
    let served = GdnModuleSelection::Served;
    let all = GdnModuleSelection::All;
    let base = adapter_fingerprint(&[lora(0.5)], &[gdn(0.25)], served);

    let mut q = lora(0.5);
    q.b_v[3] = f32::from_bits(q.b_v[3].to_bits() + 1);
    assert_ne!(base, adapter_fingerprint(&[q], &[gdn(0.25)], served));

    let mut out = gdn(0.25);
    out.b_out[0] = -0.25;
    assert_ne!(base, adapter_fingerprint(&[lora(0.5)], &[out], served));

    // The fused projections are not saved under `served`, so they are not bound.
    let mut fused = gdn(0.25);
    fused.b_a[1] = 9.0;
    assert_eq!(
        base,
        adapter_fingerprint(&[lora(0.5)], std::slice::from_ref(&fused), served)
    );
    assert_ne!(
        adapter_fingerprint(&[lora(0.5)], &[gdn(0.25)], all),
        adapter_fingerprint(&[lora(0.5)], &[fused], all)
    );
}
