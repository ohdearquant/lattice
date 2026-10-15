//! A Q4 block's scale may be negative (a block repacked from another format keeps
//! its source scale). Every Metal Q4 consumer, and every CPU site in this module
//! that dequantizes Q4 into a Metal buffer, runs the shared negative-scale fixture
//! here: its output must match the f64 reference, and the same consumer on the
//! `|scale|` and zero-scale copies of the fixture (exactly what a consumer that
//! dropped or clamped the sign would compute) must miss it by more than ten times
//! the tolerance.

use super::*;
use crate::weights::q4_weights::q4_f16_to_f32;
use crate::weights::q4_weights::signed_scale_fixture::{
    abs_scales, assert_sign_honoured, block_bytes, matmul_magnitude, matmul_reference,
    negative_scale_tensor, positive_activations, reference, zero_scales,
};

/// Same fail-closed rule as `dispatch_matmul_q4_writes_all_rows`: under
/// `LATTICE_METAL_TEST_ENFORCE=1` a missing device is a failure, not a skip.
fn metal_device() -> Option<Device> {
    let enforce = std::env::var("LATTICE_METAL_TEST_ENFORCE").is_ok();
    let device = Device::system_default();
    assert!(
        device.is_some() || !enforce,
        "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
    );
    device
}

fn shared_buffer<T>(device: &Device, data: &[T], label: &str) -> Buffer {
    let buf = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    buf.set_label(label);
    buf
}

fn read_f32(buf: &Buffer, len: usize) -> Vec<f64> {
    // SAFETY: StorageModeShared, GPU work completed by the caller, `len` f32s allocated.
    let y: &[f32] = unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, len) };
    y.iter().map(|&v| f64::from(v)).collect()
}

fn read_f16(buf: &Buffer, len: usize) -> Vec<f64> {
    // SAFETY: StorageModeShared, written on the CPU before return, `len` u16s allocated.
    let y: &[u16] = unsafe { std::slice::from_raw_parts(buf.contents() as *const u16, len) };
    y.iter().map(|&b| f64::from(q4_f16_to_f32(b))).collect()
}

const ROWS: usize = 64;
const K: usize = 512;

#[test]
fn gemv_q4_decode_honours_negative_scales() {
    let _gpu_guard = gpu_test_lock();
    let Some(device) = metal_device() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    let t = negative_scale_tensor(ROWS, K, 0x51);
    let x = positive_activations(K, 0x52);
    let x_buf = shared_buffer(&device, &x, "x");
    let run = |t: &crate::weights::q4_weights::Q4Tensor| {
        let qw = Q4WeightBuf::from_buffer(shared_buffer(&device, &block_bytes(t), "qw"));
        let y_buf = shared_buffer(&device, &vec![0.0f32; ROWS], "y");
        let cmd = state.engine.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        state.dispatch_matmul_q4(enc, &x_buf, &qw, &y_buf, 1, ROWS as u32, K as u32);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        read_f32(&y_buf, ROWS)
    };
    let w = reference(&t);
    let tol = 1e-4 * matmul_magnitude(&w, &x, 1, ROWS, K);
    assert_sign_honoured(
        "gemv_q4_decode",
        &matmul_reference(&w, &x, 1, ROWS, K),
        &run(&t),
        &run(&abs_scales(&t)),
        &run(&zero_scales(&t)),
        tol,
    );
}

fn gemm_case(tiled: bool) {
    let _gpu_guard = gpu_test_lock();
    let Some(device) = metal_device() else {
        return;
    };
    if tiled && !device.supports_family(MTLGPUFamily::Apple7) {
        // The tiled kernel needs Apple7 simdgroup matrices; a paravirtual CI GPU lacks
        // them. The marker below is what a host gate counts to prove the arm ran.
        return;
    }
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    let (pipeline, name, tol_rel) = if tiled {
        (
            state
                .engine
                .pipelines
                .gemm_q4_tiled
                .as_ref()
                .expect("gemm_q4_tiled must exist on Apple7+"),
            "gemm_q4_tiled",
            4e-3,
        )
    } else {
        (&state.engine.pipelines.gemm_q4, "gemm_q4", 1e-4)
    };
    let m = 8usize;
    let t = negative_scale_tensor(ROWS, K, 0x61);
    let x = positive_activations(m * K, 0x62);
    let x_buf = shared_buffer(&device, &x, "x");
    let run = |t: &crate::weights::q4_weights::Q4Tensor| {
        let qw = shared_buffer(&device, &block_bytes(t), "qw");
        let y_buf = shared_buffer(&device, &vec![0.0f32; m * ROWS], "y");
        let (tg_x, tg_y) = if tiled {
            (ROWS.div_ceil(32) as u64, m.div_ceil(64) as u64)
        } else {
            (ROWS.div_ceil(2) as u64, m.div_ceil(4) as u64)
        };
        run_gemm_q4_kernel(
            &state.engine.queue,
            pipeline,
            &qw,
            &x_buf,
            &y_buf,
            m as u32,
            ROWS as u32,
            K as u32,
            tg_x,
            tg_y,
        );
        read_f32(&y_buf, m * ROWS)
    };
    let w = reference(&t);
    let tol = tol_rel * matmul_magnitude(&w, &x, m, ROWS, K);
    assert_sign_honoured(
        name,
        &matmul_reference(&w, &x, m, ROWS, K),
        &run(&t),
        &run(&abs_scales(&t)),
        &run(&zero_scales(&t)),
        tol,
    );
    eprintln!("Q4_SIGNED_SCALE_RAN {name}");
}

#[test]
fn gemm_q4_honours_negative_scales() {
    gemm_case(false);
}

#[test]
fn gemm_q4_tiled_honours_negative_scales() {
    gemm_case(true);
}

/// Device layout of `TopKCandidate { float logit; uint token_id; }`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Candidate {
    logit: f32,
    token_id: u32,
}

#[test]
fn lm_head_block_topk_q4_honours_negative_scales() {
    let _gpu_guard = gpu_test_lock();
    let Some(device) = metal_device() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    // One threadgroup covers the whole vocabulary, and the widest LOCAL_K returns
    // every row's logit, so the comparison sees all of them.
    let slot = LM_HEAD_LOCAL_KS
        .iter()
        .position(|&k| k as usize >= vocab)
        .expect("a LOCAL_K variant covering the fixture vocabulary");
    assert!(vocab <= LM_HEAD_ROWS_PER_TG as usize);
    let local_k = LM_HEAD_LOCAL_KS[slot] as usize;
    let pipeline = &state.engine.pipelines.lm_head_block_topk_q4[slot];

    let t = negative_scale_tensor(vocab, hidden, 0x71);
    let x = positive_activations(hidden, 0x72);
    let x_buf = shared_buffer(&device, &x, "hidden");
    let vocab_u32 = vocab as u32;
    let run = |t: &crate::weights::q4_weights::Q4Tensor| {
        let qw = shared_buffer(&device, &block_bytes(t), "qw");
        let sentinel = Candidate {
            logit: f32::NAN,
            token_id: u32::MAX,
        };
        let out = shared_buffer(&device, &vec![sentinel; local_k], "candidates");
        let cmd = state.engine.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&qw), 0);
        enc.set_buffer(2, Some(&out), 0);
        enc.set_bytes(3, 4, &vocab_u32 as *const u32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        // SAFETY: StorageModeShared, GPU work completed, `local_k` candidates allocated.
        let cands: &[Candidate] =
            unsafe { std::slice::from_raw_parts(out.contents() as *const Candidate, local_k) };
        let mut logits = vec![f64::NAN; vocab];
        for c in cands.iter().filter(|c| (c.token_id as usize) < vocab) {
            logits[c.token_id as usize] = f64::from(c.logit);
        }
        logits
    };
    let w = reference(&t);
    let tol = 1e-4 * matmul_magnitude(&w, &x, 1, vocab, hidden);
    assert_sign_honoured(
        "lm_head_block_topk_q4",
        &matmul_reference(&w, &x, 1, vocab, hidden),
        &run(&t),
        &run(&abs_scales(&t)),
        &run(&zero_scales(&t)),
        tol,
    );
}

/// The quantized `embed_tokens` lookup buffer, dequantized on the CPU to f16.
#[test]
fn make_buffer_f16_from_q4_honours_negative_scales() {
    let _gpu_guard = gpu_test_lock();
    let Some(device) = metal_device() else {
        return;
    };
    let t = negative_scale_tensor(4, 96, 0x81);
    let run = |t: &crate::weights::q4_weights::Q4Tensor| {
        let buf = MetalQwen35State::make_buffer_f16_from_q4(&device, t, "embed");
        read_f16(&buf, t.original_len)
    };
    assert_sign_honoured(
        "make_buffer_f16_from_q4",
        &reference(&t),
        &run(&t),
        &run(&abs_scales(&t)),
        &run(&zero_scales(&t)),
        1e-3,
    );
}

/// The mmap loader that validates and dequantizes a `.q4` file to f16. Its ingress
/// check runs on the fixture, so before the scale rule was widened this refused the
/// file outright.
#[test]
fn load_q4_mmap_dequant_f16_honours_negative_scales() {
    let _gpu_guard = gpu_test_lock();
    let Some(device) = metal_device() else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let t = negative_scale_tensor(4, 96, 0x91);
    let shape = t.shape.clone();
    let load = |t: &crate::weights::q4_weights::Q4Tensor, name: &str| {
        let path = dir.path().join(name);
        crate::weights::q4_weights::save_q4_file(&path, t).expect("save");
        MetalQwen35State::load_q4_mmap_dequant_f16(&device, &path, name, &shape)
    };
    let run = |t: &crate::weights::q4_weights::Q4Tensor, name: &str| {
        let buf = load(t, name).expect("a nonzero scale of either sign loads");
        read_f16(&buf, t.original_len)
    };
    // The zero-scale copy is refused at ingress, so its control output is what a
    // clamping consumer would compute: every weight equal to its bias.
    let zero = zero_scales(&t);
    let err = load(&zero, "zero.q4").expect_err("a zero scale is refused at ingress");
    assert!(
        err.contains("finite and nonzero"),
        "unexpected error: {err}"
    );
    let zero_control: Vec<f64> = reference(&zero);
    assert_sign_honoured(
        "load_q4_mmap_dequant_f16",
        &reference(&t),
        &run(&t, "neg.q4"),
        &run(&abs_scales(&t), "abs.q4"),
        &zero_control,
        1e-3,
    );
}
