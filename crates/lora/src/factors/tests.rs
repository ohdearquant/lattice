use super::*;

#[test]
fn test_apply_lora_identity_like() {
    // rank=1, d_in=3, d_out=2
    // A = [[1, 0, 0]]  (1x3)
    // B = [[1], [0]]    (2x1)
    // x = [5, 3, 1]
    // A @ x = [5]
    // B @ [5] = [5, 0]
    // scale = 2.0 => output += [10, 0]
    let lora = LoraLayer {
        a: vec![1.0, 0.0, 0.0],
        b: vec![1.0, 0.0],
        d_in: 3,
        d_out: 2,
        rank: 1,
    };

    let x = [5.0, 3.0, 1.0];
    let mut output = [100.0, 200.0]; // pre-existing base output
    apply_lora(&lora, 2.0, &x, &mut output);

    assert!((output[0] - 110.0).abs() < 1e-6);
    assert!((output[1] - 200.0).abs() < 1e-6);
}

#[test]
fn test_apply_lora_rank2() {
    // rank=2, d_in=2, d_out=2
    // A = [[1, 0], [0, 1]]  (2x2, identity)
    // B = [[1, 2], [3, 4]]  (2x2)
    // x = [1, 1]
    // A @ x = [1, 1]
    // B @ [1, 1] = [3, 7]
    // scale = 0.5 => output += [1.5, 3.5]
    let lora = LoraLayer {
        a: vec![1.0, 0.0, 0.0, 1.0],
        b: vec![1.0, 2.0, 3.0, 4.0],
        d_in: 2,
        d_out: 2,
        rank: 2,
    };

    let x = [1.0, 1.0];
    let mut output = [0.0, 0.0];
    apply_lora(&lora, 0.5, &x, &mut output);

    assert!((output[0] - 1.5).abs() < 1e-6);
    assert!((output[1] - 3.5).abs() < 1e-6);
}

#[test]
fn test_apply_lora_zero_scale() {
    let lora = LoraLayer {
        a: vec![1.0, 2.0],
        b: vec![3.0, 4.0],
        d_in: 2,
        d_out: 2,
        rank: 1,
    };

    let x = [1.0, 1.0];
    let mut output = [10.0, 20.0];
    apply_lora(&lora, 0.0, &x, &mut output);

    // Output should be unchanged
    assert!((output[0] - 10.0).abs() < 1e-6);
    assert!((output[1] - 20.0).abs() < 1e-6);
}

/// Ranks above `STACK_RANK_CAPACITY` take the heap-fallback branch for
/// the `A @ x` intermediate. `A` is the `rank x rank` identity, so
/// `intermediate == x`; `B` is a single all-ones row, so the result is
/// `scale * sum(x)` — checkable without hand-writing 200 products.
#[test]
fn test_apply_lora_rank_above_stack_capacity_uses_heap_fallback_correctly() {
    let rank = STACK_RANK_CAPACITY + 72;
    let mut a = vec![0.0f32; rank * rank];
    for i in 0..rank {
        a[i * rank + i] = 1.0;
    }
    let b = vec![1.0f32; rank];
    let lora = LoraLayer {
        a,
        b,
        d_in: rank,
        d_out: 1,
        rank,
    };

    let x: Vec<f32> = (0..rank).map(|i| i as f32 * 0.01).collect();
    let expected: f32 = x.iter().sum();

    let mut output = [0.0f32];
    apply_lora(&lora, 2.0, &x, &mut output);

    assert!(
        (output[0] - 2.0 * expected).abs() < 1e-3,
        "got {}, want {}",
        output[0],
        2.0 * expected
    );
}

/// `apply_lora` must never index into a layer's `a`/`b` buffers past
/// their actual length: a layer whose declared `rank`/`d_in` implies a
/// buffer larger than what `a` actually holds (malformed/untrusted
/// adapter data that reached this function through some path other than
/// `LoraAdapter::new`) must be a no-op, not an out-of-bounds slice.
#[test]
fn test_apply_lora_mismatched_a_buffer_is_noop_not_panic() {
    let lora = LoraLayer {
        a: vec![], // declares rank=1, d_in=3 but holds zero elements
        b: vec![1.0, 0.0],
        d_in: 3,
        d_out: 2,
        rank: 1,
    };

    let x = [5.0, 3.0, 1.0];
    let mut output = [100.0, 200.0];
    apply_lora(&lora, 2.0, &x, &mut output);

    assert_eq!(
        output,
        [100.0, 200.0],
        "mismatched buffer must leave output untouched"
    );
}

/// Simulates `apply_lora_rows` calling `apply_lora` once per token row:
/// each call's stack scratch buffer must be reinitialized fresh, not
/// carry state from the previous row.
#[test]
fn test_apply_lora_repeated_calls_are_independent_per_row() {
    let lora = LoraLayer {
        a: vec![1.0, 0.0, 0.0, 1.0],
        b: vec![1.0, 2.0, 3.0, 4.0],
        d_in: 2,
        d_out: 2,
        rank: 2,
    };

    let rows = [[1.0, 1.0], [2.0, 0.0], [0.0, 3.0], [5.0, -1.0]];
    for row in rows {
        let mut output = [0.0, 0.0];
        apply_lora(&lora, 1.0, &row, &mut output);

        // A = identity => intermediate = row.
        let expected0 = row[0] + 2.0 * row[1];
        let expected1 = 3.0 * row[0] + 4.0 * row[1];
        assert!((output[0] - expected0).abs() < 1e-6);
        assert!((output[1] - expected1).abs() < 1e-6);
    }
}
