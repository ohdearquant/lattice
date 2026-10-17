use super::*;
use crate::quant::quarot::plan::{AbsorptionSide, RotationId};
use std::fs::File;
use std::io::Write;

/// On-disk dtype for synthetic test fixtures.
#[derive(Copy, Clone)]
enum FixtureDType {
    F32,
    F16,
    BF16,
}

fn dtype_name(d: FixtureDType) -> &'static str {
    match d {
        FixtureDType::F32 => "F32",
        FixtureDType::F16 => "F16",
        FixtureDType::BF16 => "BF16",
    }
}

fn bytes_per_elem(d: FixtureDType) -> usize {
    match d {
        FixtureDType::F32 => 4,
        FixtureDType::F16 | FixtureDType::BF16 => 2,
    }
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    // Round-to-nearest-even via the canonical f32→bf16 algorithm.
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        // NaN: preserve quiet payload top bit, ensure non-zero payload.
        ((bits >> 16) as u16) | 0x0040
    } else {
        let round_bit = (bits >> 16) & 1;
        let half = 0x7fff + round_bit;
        ((bits.wrapping_add(half)) >> 16) as u16
    }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) as u16) << 15;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    // ±Inf / NaN
    if exp == 0xff {
        return if frac == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7c00 | (((frac >> 13) as u16) | 0x0200)
        };
    }
    // ±0 / very small (subnormal-in-f32 → 0 in f16)
    if exp == 0 {
        return sign;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal in f16. Shift mantissa right by (1 - e), add implicit 1.
        let shift = 14 - exp;
        let mant = (frac | 0x0080_0000) >> shift;
        return sign | (mant as u16);
    }
    sign | ((e as u16) << 10) | ((frac >> 13) as u16)
}

fn encode_value(v: f32, dtype: FixtureDType) -> Vec<u8> {
    match dtype {
        FixtureDType::F32 => v.to_le_bytes().to_vec(),
        FixtureDType::BF16 => f32_to_bf16_bits(v).to_le_bytes().to_vec(),
        FixtureDType::F16 => f32_to_f16_bits(v).to_le_bytes().to_vec(),
    }
}

fn write_safetensors(path: &Path, tensors: &[(&str, FixtureDType, Vec<usize>, &[f32])]) {
    let mut header = serde_json::Map::new();
    let mut payload: Vec<u8> = Vec::new();
    for (name, dtype, shape, values) in tensors {
        assert_eq!(values.len(), shape.iter().product::<usize>());
        let start = payload.len();
        for &v in *values {
            payload.extend_from_slice(&encode_value(v, *dtype));
        }
        let end = payload.len();
        assert_eq!(end - start, values.len() * bytes_per_elem(*dtype));

        let mut entry = serde_json::Map::new();
        entry.insert("dtype".into(), Value::String(dtype_name(*dtype).into()));
        entry.insert(
            "shape".into(),
            Value::Array(shape.iter().map(|d| Value::from(*d as u64)).collect()),
        );
        entry.insert(
            "data_offsets".into(),
            Value::Array(vec![Value::from(start as u64), Value::from(end as u64)]),
        );
        header.insert((*name).to_string(), Value::Object(entry));
    }
    let header_str = serde_json::to_string(&Value::Object(header)).unwrap();
    let header_bytes = header_str.as_bytes();
    let mut file = File::create(path).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header_bytes).unwrap();
    file.write_all(&payload).unwrap();
}

fn write_raw_safetensor(
    path: &Path,
    tensor_name: &str,
    dtype: FixtureDType,
    shape: &[usize],
    payload: &[u8],
) {
    assert_eq!(
        payload.len(),
        shape.iter().product::<usize>() * bytes_per_elem(dtype)
    );
    let header = serde_json::json!({
        (tensor_name): {
            "dtype": dtype_name(dtype),
            "shape": shape,
            "data_offsets": [0, payload.len()],
        }
    });
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(payload).unwrap();
}

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn single_file_f32_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<f32> = vec![0.0, 1.0, -2.5, 7.125, -0.0001, 1e6];
    write_safetensors(
        &dir.path().join("model.safetensors"),
        &[("w", FixtureDType::F32, vec![2, 3], &values)],
    );

    let reader = QuarotTensorReader::open(dir.path()).unwrap();
    assert!(reader.has_tensor("w"));
    assert_eq!(reader.tensor_shape("w").unwrap(), vec![2, 3]);
    assert_eq!(reader.source_dtype("w").unwrap(), SourceDType::F32);

    let (got, shape) = reader.read_tensor_f64("w").unwrap();
    assert_eq!(shape, vec![2, 3]);
    for (g, &v) in got.iter().zip(values.iter()) {
        assert_eq!(*g, v as f64);
    }
}

// #1368: `Shard::open` (reached via `QuarotTensorReader::open`) is a
// raw-open mmap site that must route through the mmap trust boundary the
// same as every other checkpoint mmap site in this crate. This proves
// the wiring at the public entry point, not just the underlying
// `reject_if_open_mmap_file_untrusted` predicate (already covered
// directly in `mmap_trust`'s own tests).
#[cfg(unix)]
#[test]
fn open_rejects_a_group_or_other_writable_checkpoint_file() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    write_safetensors(&path, &[("w", FixtureDType::F32, vec![1], &[1.0])]);
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).expect("chmod 0o666");

    let err = QuarotTensorReader::open(dir.path())
        .expect_err("a group/other-writable checkpoint file must be refused");
    assert!(
        matches!(&err, InferenceError::InvalidSafetensors(msg) if msg.contains("refusing to load")),
        "expected a trust-boundary refusal, got: {err:?}"
    );

    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod 0o600");
    QuarotTensorReader::open(dir.path())
        .expect("an owner-only checkpoint file must still be accepted");
}

// #1380: `Shard::open` (reached via `QuarotTensorReader::open`) is one
// of the three entry points the issue named. It used to call plain
// `File::open` before any file-type check ran, so a FIFO planted at the
// checkpoint path blocked the call indefinitely. Proves the fix the same
// way as the other loader entry points: the call must return (any
// return) inside a deadline, and the return must specifically reject the
// FIFO as non-regular. `mkfifo` (POSIX, not macOS-only) keeps this test
// portable to the Linux CI legs that run this suite.
#[test]
fn open_rejects_a_fifo_without_blocking() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let status = std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .expect("run mkfifo");
    assert!(status.success(), "mkfifo must succeed to set up this test");

    let (tx, rx) = std::sync::mpsc::channel();
    let probe_dir = dir.path().to_path_buf();
    // A pre-fix blocking `File::open` on this FIFO (no writer ever
    // connects) would hang this thread forever; running the call on a
    // detached thread and racing it against a deadline on
    // `rx.recv_timeout` turns "hangs forever" into an observable test
    // failure instead of an actually-hung test process.
    std::thread::spawn(move || {
        let result = QuarotTensorReader::open(&probe_dir);
        let _ = tx.send(result);
    });
    let result = rx.recv_timeout(std::time::Duration::from_secs(5)).expect(
        "QuarotTensorReader::open did not return within 5s -- it blocked on the \
             planted FIFO, meaning the open-time regular-file guard regressed",
    );
    let err = result.expect_err("a FIFO checkpoint must be rejected, not opened");
    assert!(
        matches!(&err, InferenceError::InvalidSafetensors(msg) if msg.contains("not a regular file")),
        "expected a not-a-regular-file refusal, got: {err:?}"
    );
}

#[test]
fn single_file_bf16_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<f32> = vec![0.0, 1.0, -1.0, 2.5, -3.75, 100.0];
    write_safetensors(
        &dir.path().join("model.safetensors"),
        &[("w", FixtureDType::BF16, vec![3, 2], &values)],
    );

    let reader = QuarotTensorReader::open(dir.path()).unwrap();
    assert_eq!(reader.source_dtype("w").unwrap(), SourceDType::BF16);
    let (got, shape) = reader.read_tensor_f64("w").unwrap();
    assert_eq!(shape, vec![3, 2]);
    // bf16 has 7-bit mantissa: relative precision ~2^-8 ≈ 4e-3 for these
    // values; absolute tolerance scaled to magnitude.
    for (g, &v) in got.iter().zip(values.iter()) {
        let tol = (v.abs() as f64) * (1.0 / 128.0) + 1e-6;
        assert!(approx_eq(*g, v as f64, tol), "got {g} expected ~{v}");
    }
}

#[test]
fn single_file_f16_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    // Values within f16 range; f16 has 10-bit mantissa.
    let values: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -2.0, 100.0];
    write_safetensors(
        &dir.path().join("model.safetensors"),
        &[("w", FixtureDType::F16, vec![6], &values)],
    );

    let reader = QuarotTensorReader::open(dir.path()).unwrap();
    assert_eq!(reader.source_dtype("w").unwrap(), SourceDType::F16);
    let (got, _) = reader.read_tensor_f64("w").unwrap();
    for (g, &v) in got.iter().zip(values.iter()) {
        let tol = (v.abs() as f64) * (1.0 / 1024.0) + 1e-6;
        assert!(approx_eq(*g, v as f64, tol), "got {g} expected ~{v}");
    }
}

/// Mutation-sensitive lattice#801 gate: removing a raw exponent-marker
/// reduction makes that dtype's NaN/Inf cases return `Ok`.
#[test]
fn read_tensor_f64_rejects_non_finite_values_for_every_supported_dtype() {
    for dtype in [FixtureDType::F32, FixtureDType::F16, FixtureDType::BF16] {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for bad_index in [5, 17] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("model.safetensors");
                let mut values = vec![1.0; 18];
                values[bad_index] = bad;
                write_safetensors(&path, &[("bad.weight", dtype, vec![18], &values)]);

                let reader = QuarotTensorReader::open(dir.path()).unwrap();
                let err = reader.read_tensor_f64("bad.weight").unwrap_err();
                match err {
                    InferenceError::InvalidSafetensors(message) => {
                        assert!(message.contains(&path.display().to_string()), "{message}");
                        assert!(message.contains("bad.weight"), "{message}");
                        assert!(message.contains(dtype_name(dtype)), "{message}");
                        assert!(message.contains("non-finite value"), "{message}");
                        assert!(
                            message.contains(&format!("element index {bad_index}")),
                            "{message}"
                        );
                    }
                    other => panic!("expected InvalidSafetensors, got {other:?}"),
                }
            }
        }
    }
}

#[test]
fn read_tensor_f64_accepts_signed_zero_and_smallest_subnormal_for_every_dtype() {
    let mut f32_bits = vec![1.0_f32.to_bits(); 18];
    f32_bits[0] = 0;
    f32_bits[1] = 0x8000_0000;
    f32_bits[2] = 1;
    f32_bits[16] = 0x8000_0000;
    f32_bits[17] = 1;
    let f32_payload = f32_bits
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();

    let mut f16_bits = vec![0x3c00_u16; 18];
    f16_bits[0] = 0;
    f16_bits[1] = 0x8000;
    f16_bits[2] = 1;
    f16_bits[16] = 0x8000;
    f16_bits[17] = 1;
    let f16_payload = f16_bits
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();

    let mut bf16_bits = vec![0x3f80_u16; 18];
    bf16_bits[0] = 0;
    bf16_bits[1] = 0x8000;
    bf16_bits[2] = 1;
    bf16_bits[16] = 0x8000;
    bf16_bits[17] = 1;
    let bf16_payload = bf16_bits
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();

    for (dtype, payload, smallest_subnormal) in [
        (FixtureDType::F32, f32_payload, 2.0_f64.powi(-149)),
        (FixtureDType::F16, f16_payload, 2.0_f64.powi(-24)),
        (FixtureDType::BF16, bf16_payload, 2.0_f64.powi(-133)),
    ] {
        let dir = tempfile::tempdir().unwrap();
        write_raw_safetensor(
            &dir.path().join("model.safetensors"),
            "finite.weight",
            dtype,
            &[18],
            &payload,
        );

        let reader = QuarotTensorReader::open(dir.path()).unwrap();
        let (decoded, shape) = reader.read_tensor_f64("finite.weight").unwrap();
        assert_eq!(shape, vec![18]);
        assert_eq!(decoded[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(decoded[1].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(decoded[2], smallest_subnormal, "{}", dtype_name(dtype));
        assert_eq!(decoded[16].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(decoded[17], smallest_subnormal, "{}", dtype_name(dtype));
    }
}

#[test]
fn sharded_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let shard_a = "model-00001-of-00002.safetensors";
    let shard_b = "model-00002-of-00002.safetensors";

    let a_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_vals: Vec<f32> = vec![-1.0, -2.0];

    write_safetensors(
        &dir.path().join(shard_a),
        &[("a.weight", FixtureDType::F32, vec![2, 2], &a_vals)],
    );
    write_safetensors(
        &dir.path().join(shard_b),
        &[("b.weight", FixtureDType::BF16, vec![2], &b_vals)],
    );

    let index = serde_json::json!({
        "metadata": {"total_size": 24usize},
        "weight_map": {
            "a.weight": shard_a,
            "b.weight": shard_b,
        },
    });
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        serde_json::to_string(&index).unwrap(),
    )
    .unwrap();

    let reader = QuarotTensorReader::open(dir.path()).unwrap();
    assert!(reader.has_tensor("a.weight"));
    assert!(reader.has_tensor("b.weight"));
    assert!(!reader.has_tensor("missing"));

    let mut names = reader.tensor_names();
    names.sort();
    assert_eq!(names, vec!["a.weight".to_string(), "b.weight".to_string()]);

    let (a, a_shape) = reader.read_tensor_f64("a.weight").unwrap();
    assert_eq!(a_shape, vec![2, 2]);
    assert_eq!(a, vec![1.0, 2.0, 3.0, 4.0]);

    let (b, b_shape) = reader.read_tensor_f64("b.weight").unwrap();
    assert_eq!(b_shape, vec![2]);
    // BF16 of -1.0 and -2.0 are exact.
    assert_eq!(b, vec![-1.0, -2.0]);

    // Re-reading the same tensor exercises the cached-shard path.
    let (a_again, _) = reader.read_tensor_f64("a.weight").unwrap();
    assert_eq!(a_again, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn missing_tensor_errors() {
    let dir = tempfile::tempdir().unwrap();
    write_safetensors(
        &dir.path().join("model.safetensors"),
        &[("present", FixtureDType::F32, vec![1], &[1.0])],
    );
    let reader = QuarotTensorReader::open(dir.path()).unwrap();
    let err = reader.read_tensor_f64("absent").unwrap_err();
    match err {
        InferenceError::MissingTensor(name) => assert_eq!(name, "absent"),
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}

#[test]
fn no_safetensors_in_dir_errors() {
    let dir = tempfile::tempdir().unwrap();
    let err = QuarotTensorReader::open(dir.path()).unwrap_err();
    assert!(matches!(err, InferenceError::InvalidSafetensors(_)));
}

#[test]
fn byte_length_mismatch_rejected() {
    let dir = tempfile::tempdir().unwrap();
    // Hand-build a malformed file: F32 dtype, shape [4] (16 bytes), but
    // declare only 12 bytes of data.
    let header = serde_json::json!({
        "w": {
            "dtype": "F32",
            "shape": [4],
            "data_offsets": [0, 12],
        }
    });
    let header_str = serde_json::to_string(&header).unwrap();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
    buf.extend_from_slice(header_str.as_bytes());
    buf.extend_from_slice(&[0u8; 12]);
    std::fs::write(dir.path().join("model.safetensors"), &buf).unwrap();

    let err = QuarotTensorReader::open(dir.path()).unwrap_err();
    match err {
        InferenceError::InvalidSafetensors(msg) => {
            assert!(msg.contains("byte length mismatch"), "got: {msg}");
        }
        other => panic!("expected InvalidSafetensors, got {other:?}"),
    }
}

#[test]
fn unsupported_dtype_is_skipped() {
    let dir = tempfile::tempdir().unwrap();
    // Build a file with two tensors: a supported F32 and an unsupported
    // I64. The I64 must be ignored, not error.
    let header = serde_json::json!({
        "supported": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        },
        "ignored": {
            "dtype": "I64",
            "shape": [1],
            "data_offsets": [4, 12],
        }
    });
    let header_str = serde_json::to_string(&header).unwrap();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
    buf.extend_from_slice(header_str.as_bytes());
    buf.extend_from_slice(&7f32.to_le_bytes());
    buf.extend_from_slice(&42i64.to_le_bytes());
    std::fs::write(dir.path().join("model.safetensors"), &buf).unwrap();

    let reader = QuarotTensorReader::open(dir.path()).unwrap();
    assert!(reader.has_tensor("supported"));
    assert!(!reader.has_tensor("ignored"));
    let (data, _) = reader.read_tensor_f64("supported").unwrap();
    assert_eq!(data, vec![7.0]);
}

#[test]
fn fnuz_dtype_tensors_are_skipped_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    // F8_E4M3FNUZ / F8_E5M2FNUZ are official SafeTensors dtypes this
    // reader does not decode. Opening a checkpoint that merely contains
    // one must still succeed and expose the readable tensors — it must
    // not error out as an unrecognized dtype.
    let header = serde_json::json!({
        "supported": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        },
        "e4m3fnuz": {
            "dtype": "F8_E4M3FNUZ",
            "shape": [1],
            "data_offsets": [4, 5],
        },
        "e5m2fnuz": {
            "dtype": "F8_E5M2FNUZ",
            "shape": [1],
            "data_offsets": [5, 6],
        },
    });
    let header_str = serde_json::to_string(&header).unwrap();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
    buf.extend_from_slice(header_str.as_bytes());
    buf.extend_from_slice(&7f32.to_le_bytes());
    buf.extend_from_slice(&[0u8, 0u8]);
    std::fs::write(dir.path().join("model.safetensors"), &buf).unwrap();

    let reader = QuarotTensorReader::open(dir.path()).unwrap();
    assert!(reader.has_tensor("supported"));
    assert!(!reader.has_tensor("e4m3fnuz"));
    assert!(!reader.has_tensor("e5m2fnuz"));
    let (data, _) = reader.read_tensor_f64("supported").unwrap();
    assert_eq!(data, vec![7.0]);
}

#[test]
fn single_file_wins_when_both_layouts_present() {
    // Mirrors the runtime loaders at qwen35/model.rs:25 and qwen.rs:425,
    // which both check `model.safetensors` before the sharded index.
    // The converter must agree, otherwise step 3c's forward-equivalence
    // check would target a different checkpoint than the runtime serves.
    let dir = tempfile::tempdir().unwrap();
    write_safetensors(
        &dir.path().join("model.safetensors"),
        &[("real", FixtureDType::F32, vec![1], &[1.0])],
    );
    write_safetensors(
        &dir.path().join("model-00001-of-00001.safetensors"),
        &[("decoy", FixtureDType::F32, vec![1], &[999.0])],
    );
    let index = serde_json::json!({
        "metadata": {"total_size": 4usize},
        "weight_map": {"decoy": "model-00001-of-00001.safetensors"},
    });
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        serde_json::to_string(&index).unwrap(),
    )
    .unwrap();

    let reader = QuarotTensorReader::open(dir.path()).unwrap();
    assert!(reader.has_tensor("real"));
    assert!(!reader.has_tensor("decoy"));
    let (data, _) = reader.read_tensor_f64("real").unwrap();
    assert_eq!(data, vec![1.0]);
}

/// Helper for malformed-layout tests: writes a hand-crafted SafeTensors
/// file with a caller-supplied header object and raw data section.
fn write_raw_safetensors(path: &Path, header_json: &Value, data: &[u8]) {
    let header_str = serde_json::to_string(header_json).unwrap();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
    buf.extend_from_slice(header_str.as_bytes());
    buf.extend_from_slice(data);
    std::fs::write(path, &buf).unwrap();
}

/// Two F32 tensors declared at the same `[0, 4)` byte range alias each
/// other — the official safetensors parser rejects this via
/// `InvalidOffset`, and so must we, because a converter that reads
/// both names would silently rotate the same bytes twice.
#[test]
fn overlapping_offsets_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let header = serde_json::json!({
        "a": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] },
        "b": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] },
    });
    write_raw_safetensors(
        &dir.path().join("model.safetensors"),
        &header,
        &1.0f32.to_le_bytes(),
    );
    let err = QuarotTensorReader::open(dir.path()).unwrap_err();
    match err {
        InferenceError::InvalidSafetensors(msg) => {
            assert!(msg.contains("non-contiguous"), "got: {msg}");
        }
        other => panic!("expected InvalidSafetensors, got {other:?}"),
    }
}

/// First tensor starts at offset 4 — the data section has a 4-byte
/// leading hole. The official parser rejects this; we must too.
#[test]
fn leading_hole_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let header = serde_json::json!({
        "w": { "dtype": "F32", "shape": [1], "data_offsets": [4, 8] },
    });
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(&[0u8; 4]);
    data.extend_from_slice(&7.0f32.to_le_bytes());
    write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &data);
    let err = QuarotTensorReader::open(dir.path()).unwrap_err();
    match err {
        InferenceError::InvalidSafetensors(msg) => {
            assert!(msg.contains("non-contiguous"), "got: {msg}");
        }
        other => panic!("expected InvalidSafetensors, got {other:?}"),
    }
}

/// Gap between the end of tensor `a` and the start of tensor `b` —
/// 4 stray bytes in the middle of the data section.
#[test]
fn internal_hole_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let header = serde_json::json!({
        "a": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] },
        "b": { "dtype": "F32", "shape": [1], "data_offsets": [8, 12] },
    });
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&[0u8; 4]);
    data.extend_from_slice(&2.0f32.to_le_bytes());
    write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &data);
    let err = QuarotTensorReader::open(dir.path()).unwrap_err();
    match err {
        InferenceError::InvalidSafetensors(msg) => {
            assert!(msg.contains("non-contiguous"), "got: {msg}");
        }
        other => panic!("expected InvalidSafetensors, got {other:?}"),
    }
}

/// Data section has 4 extra trailing bytes beyond the last tensor's
/// declared end. The official parser rejects this; we must too.
#[test]
fn trailing_payload_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let header = serde_json::json!({
        "w": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] },
    });
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&[0u8; 4]);
    write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &data);
    let err = QuarotTensorReader::open(dir.path()).unwrap_err();
    match err {
        InferenceError::InvalidSafetensors(msg) => {
            assert!(msg.contains("trailing or missing payload"), "got: {msg}");
        }
        other => panic!("expected InvalidSafetensors, got {other:?}"),
    }
}

/// An unsupported-dtype (I64) tensor at offset 4 must still fail the
/// global contiguity check — offset validation applies to every
/// declared tensor, not just the F32/F16/BF16 ones the reader decodes.
#[test]
fn unsupported_dtype_with_bad_offsets_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let header = serde_json::json!({
        "x": { "dtype": "I64", "shape": [1], "data_offsets": [4, 12] },
    });
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(&[0u8; 4]);
    data.extend_from_slice(&42i64.to_le_bytes());
    write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &data);
    let err = QuarotTensorReader::open(dir.path()).unwrap_err();
    match err {
        InferenceError::InvalidSafetensors(msg) => {
            assert!(msg.contains("non-contiguous"), "got: {msg}");
        }
        other => panic!("expected InvalidSafetensors, got {other:?}"),
    }
}

/// An entirely unrecognized dtype string is rejected at open time,
/// not silently skipped. The runtime / official parser would reject
/// the same bytes; the converter must not bless what the runtime
/// wouldn't.
#[test]
fn unknown_dtype_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let header = serde_json::json!({
        "x": { "dtype": "FUTURE_DTYPE", "shape": [1], "data_offsets": [0, 4] },
    });
    write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &[0u8; 4]);
    let err = QuarotTensorReader::open(dir.path()).unwrap_err();
    match err {
        InferenceError::InvalidSafetensors(msg) => {
            assert!(msg.contains("unrecognized SafeTensors dtype"), "got: {msg}");
        }
        other => panic!("expected InvalidSafetensors, got {other:?}"),
    }
}

/// F8_E8M0 is 8 bits per element; shape [2] requires 2 bytes.
/// Declaring a 1-byte range fails the byte-length check, even though
/// F8_E8M0 isn't a converter-decoded dtype.
#[test]
fn f8_e8m0_byte_length_mismatch_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let header = serde_json::json!({
        "x": { "dtype": "F8_E8M0", "shape": [2], "data_offsets": [0, 1] },
    });
    write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &[0u8; 1]);
    let err = QuarotTensorReader::open(dir.path()).unwrap_err();
    match err {
        InferenceError::InvalidSafetensors(msg) => {
            assert!(msg.contains("byte length mismatch"), "got: {msg}");
        }
        other => panic!("expected InvalidSafetensors, got {other:?}"),
    }
}

/// C64 is complex64 — 64 bits = 8 bytes per element. Shape [1]
/// requires 8 bytes; declaring a 4-byte range fails.
#[test]
fn c64_byte_length_mismatch_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let header = serde_json::json!({
        "z": { "dtype": "C64", "shape": [1], "data_offsets": [0, 4] },
    });
    write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &[0u8; 4]);
    let err = QuarotTensorReader::open(dir.path()).unwrap_err();
    match err {
        InferenceError::InvalidSafetensors(msg) => {
            assert!(msg.contains("byte length mismatch"), "got: {msg}");
        }
        other => panic!("expected InvalidSafetensors, got {other:?}"),
    }
}

/// F4 stores 4 bits per element. Shape [3] = 12 bits, which is not
/// byte-aligned — the file is unreadable regardless of declared
/// offsets. The reader rejects this at open time.
#[test]
fn f4_sub_byte_misalignment_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let header = serde_json::json!({
        "x": { "dtype": "F4", "shape": [3], "data_offsets": [0, 2] },
    });
    write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &[0u8; 2]);
    let err = QuarotTensorReader::open(dir.path()).unwrap_err();
    match err {
        InferenceError::InvalidSafetensors(msg) => {
            assert!(msg.contains("not byte-aligned"), "got: {msg}");
        }
        other => panic!("expected InvalidSafetensors, got {other:?}"),
    }
}

/// F6_E2M3 stores 6 bits per element. Shape [4] = 24 bits = 3 bytes;
/// declaring a 2-byte range fails the byte-length check.
#[test]
fn f6_e2m3_byte_length_mismatch_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let header = serde_json::json!({
        "x": { "dtype": "F6_E2M3", "shape": [4], "data_offsets": [0, 2] },
    });
    write_raw_safetensors(&dir.path().join("model.safetensors"), &header, &[0u8; 2]);
    let err = QuarotTensorReader::open(dir.path()).unwrap_err();
    match err {
        InferenceError::InvalidSafetensors(msg) => {
            assert!(msg.contains("byte length mismatch"), "got: {msg}");
        }
        other => panic!("expected InvalidSafetensors, got {other:?}"),
    }
}

/// Index declares `required.weight` in a shard whose actual contents
/// do NOT include that tensor. `has_tensor` and `tensor_names` must
/// reflect the readable view, not the manifest, so step 3c's preflight
/// rejects the checkpoint before the rotation pass starts.
#[test]
fn sharded_index_entry_missing_from_shard_is_not_readable() {
    let dir = tempfile::tempdir().unwrap();
    let shard_file = "model-00001-of-00001.safetensors";
    // Shard contains only `other.weight`, not `required.weight`.
    write_safetensors(
        &dir.path().join(shard_file),
        &[("other.weight", FixtureDType::F32, vec![1], &[1.0])],
    );
    let index = serde_json::json!({
        "metadata": {"total_size": 4usize},
        "weight_map": {
            "required.weight": shard_file,
            "other.weight": shard_file,
        },
    });
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        serde_json::to_string(&index).unwrap(),
    )
    .unwrap();

    let reader = QuarotTensorReader::open(dir.path()).unwrap();
    assert!(reader.has_tensor("other.weight"));
    assert!(!reader.has_tensor("required.weight"));
    assert_eq!(reader.tensor_names(), vec!["other.weight".to_string()]);
    let err = reader.read_tensor_f64("required.weight").unwrap_err();
    assert!(matches!(err, InferenceError::MissingTensor(_)));
}

/// #1069: a weight_map entry that escapes the model directory (here via
/// `..` traversal to a structurally valid shard outside it) must be
/// rejected at open time, before anything is memory-mapped.
#[test]
fn sharded_index_entry_escaping_model_dir_is_rejected() {
    let outer = tempfile::tempdir().unwrap();
    let model_dir = outer.path().join("model");
    std::fs::create_dir_all(&model_dir).unwrap();
    write_safetensors(
        &outer.path().join("evil.safetensors"),
        &[("t.weight", FixtureDType::F32, vec![1], &[1.0])],
    );
    let index = serde_json::json!({
        "metadata": {"total_size": 4usize},
        "weight_map": {"t.weight": "../evil.safetensors"},
    });
    std::fs::write(
        model_dir.join("model.safetensors.index.json"),
        serde_json::to_string(&index).unwrap(),
    )
    .unwrap();

    let err = QuarotTensorReader::open(&model_dir).unwrap_err();
    match err {
        InferenceError::InvalidSafetensors(msg) => {
            assert!(
                msg.contains("must stay within the model directory"),
                "got: {msg}"
            );
        }
        other => panic!("expected InvalidSafetensors, got {other:?}"),
    }
}

/// Index declares `required.weight` in a shard that DOES contain a
/// tensor by that name but with an unsupported dtype (I64). The shard
/// parser drops unsupported entries, so the reader must report the
/// tensor as not-readable.
#[test]
fn sharded_index_entry_unsupported_dtype_is_not_readable() {
    let dir = tempfile::tempdir().unwrap();
    let shard_file = "model-00001-of-00001.safetensors";
    // Hand-build a shard with one I64 tensor (unsupported dtype).
    let header = serde_json::json!({
        "required.weight": {
            "dtype": "I64",
            "shape": [1],
            "data_offsets": [0, 8],
        }
    });
    let header_str = serde_json::to_string(&header).unwrap();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
    buf.extend_from_slice(header_str.as_bytes());
    buf.extend_from_slice(&42i64.to_le_bytes());
    std::fs::write(dir.path().join(shard_file), &buf).unwrap();

    let index = serde_json::json!({
        "metadata": {"total_size": 8usize},
        "weight_map": {"required.weight": shard_file},
    });
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        serde_json::to_string(&index).unwrap(),
    )
    .unwrap();

    let reader = QuarotTensorReader::open(dir.path()).unwrap();
    assert!(!reader.has_tensor("required.weight"));
    assert!(reader.tensor_names().is_empty());
    let err = reader.read_tensor_f64("required.weight").unwrap_err();
    assert!(matches!(err, InferenceError::MissingTensor(_)));
}

/// Compile-time check that `qwen_required_tensor_names` is reachable
/// from non-test code paths. If step 3b's promotion ever regresses,
/// this test fails to compile.
#[test]
fn qwen_required_tensor_names_is_publicly_reachable() {
    use crate::model::qwen35::qwen_required_tensor_names;
    use crate::model::qwen35_config::Qwen35Config;
    // We don't construct a config (it has many required fields); we
    // just want a fn pointer to prove visibility.
    let _f: fn(&Qwen35Config) -> Vec<String> = qwen_required_tensor_names;
}

// --- ArtifactVersion / OnlineArtifactDescriptor (issue #703 PR1) ---
//
// `ArtifactVersion`'s `#[serde(rename = ...)]` attributes are the single
// on-disk spelling of each tag; deserializing a JSON string through the
// derived `Deserialize` impl below is the one parser, exercised
// directly rather than through a second hand-written match.

#[test]
fn artifact_version_deserializes_known_tags() {
    assert_eq!(
        serde_json::from_str::<ArtifactVersion>("\"v0-residual\"").unwrap(),
        ArtifactVersion::V0Residual
    );
    assert_eq!(
        serde_json::from_str::<ArtifactVersion>("\"v1-online-r3r4\"").unwrap(),
        ArtifactVersion::V1Online
    );
}

#[test]
fn artifact_version_refuses_unknown_tag() {
    assert!(serde_json::from_str::<ArtifactVersion>("\"v2-future\"").is_err());
}

#[test]
fn artifact_version_refuses_empty_tag() {
    assert!(serde_json::from_str::<ArtifactVersion>("\"\"").is_err());
}

#[test]
fn artifact_version_refuses_case_variant() {
    // Refuses rather than normalizing — an unexpected casing is
    // evidence of a foreign/corrupted writer, not a legitimate variant.
    assert!(serde_json::from_str::<ArtifactVersion>("\"V0-Residual\"").is_err());
}

fn sample_r3_cfg() -> Qwen35Config {
    crate::model::qwen35_config::Qwen35Config::qwen35_0_8b()
}

fn sample_r3_spec() -> OnlineRotationSpec {
    let cfg = sample_r3_cfg();
    // block_size == num_attention_heads (8): one dense cross-head
    // Hadamard covering every head (QuaRot Eq. 9's H_num_heads).
    OnlineRotationSpec::r3_full_attention(&cfg, 42, 8).unwrap()
}

/// Every tensor name `sample_r3_spec()`'s scope (`[3,7,11,15,19,23]`)
/// requires full coverage for — `model.language_model.layers.{i}.self_attn.o_proj.weight`
/// for each scoped layer index.
fn full_r3_layer_names() -> Vec<String> {
    [3usize, 7, 11, 15, 19, 23]
        .iter()
        .map(|i| format!("model.language_model.layers.{i}.self_attn.o_proj.weight"))
        .collect()
}

#[test]
fn v0_residual_with_no_online_rotations_is_valid() {
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V0Residual,
        online_rotations: Vec::new(),
        asymmetric_tensor_names: Vec::new(),
    };
    assert!(descriptor.validate(None).is_ok());
}

/// A V0Residual descriptor must never carry online rotation metadata —
/// feeding it one must be refused loudly rather than silently ignored
/// (which would let a hand-edited index sneak online-rotation intent
/// past a v0-only loader).
#[test]
fn v0_residual_with_online_rotation_is_refused() {
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V0Residual,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: Vec::new(),
    };
    let err = descriptor.validate(None).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("never interpreted as online-capable"),
        "got: {msg}"
    );
}

/// A V0Residual descriptor must never carry
/// asymmetric-Q4 tensor declarations — the V0 contract is symmetric-only
/// Q4 with no R3/R4 metadata, so a nonempty `asymmetric_tensor_names`
/// list on a V0 descriptor is evidence of a corrupted or hand-edited
/// index, same as a nonempty `online_rotations` list.
#[test]
fn v0_residual_with_asymmetric_tensor_names_is_refused() {
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V0Residual,
        online_rotations: Vec::new(),
        asymmetric_tensor_names: vec![
            "model.language_model.layers.3.self_attn.o_proj.weight".to_string(),
        ],
    };
    let err = descriptor.validate(None).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("must not declare any asymmetric_tensor_names"),
        "got: {msg}"
    );
}

/// Mutation/refusal: V1Online with an empty rotation list is a
/// contradiction (an "online" artifact that rotates nothing).
#[test]
fn v1_online_with_zero_rotations_is_refused() {
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: Vec::new(),
        asymmetric_tensor_names: Vec::new(),
    };
    let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("at least one R3/R4"), "got: {msg}");
}

/// `validate` must refuse immediately for a
/// `V1Online` descriptor called without `cfg`, rather than silently
/// running a weaker (per-layer-coverage-skipping) check. This must hold
/// even for a descriptor that would otherwise validate cleanly with
/// `cfg` supplied — a caller cannot opt into the weaker semantics just
/// by omitting the config.
#[test]
fn v1_online_validate_without_cfg_is_refused() {
    let names = full_r3_layer_names();
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: names,
    };
    let err = descriptor.validate(None).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires model config"), "got: {msg}");
    // The same descriptor passes once cfg is supplied — proving the
    // None case above was refused for lack of cfg, not some other
    // reason.
    assert!(descriptor.validate(Some(&sample_r3_cfg())).is_ok());
}

/// Mutation/refusal: a declaration missing one scoped layer's tensor
/// must be refused — the single `asymmetric_tensor_names` declaration
/// must equal the recipe-derived set exactly: this is the same
/// requirement the pre-collapse
/// `affected_tensor_names` cross-check enforced, now checked entirely
/// from the descriptor's own fields).
#[test]
fn v1_online_missing_asymmetric_declaration_is_refused() {
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: vec![
            "model.language_model.layers.3.self_attn.o_proj.weight".to_string(),
        ],
    };
    let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("coverage is incomplete"), "got: {msg}");
    assert!(msg.contains("layer 7"), "got: {msg}");
}

/// Preserved under the single-representation
/// contract: a real rotation recipe with zero asymmetric declarations
/// must be refused internally rather than silently accepted.
#[test]
fn v1_online_empty_asymmetric_list_is_refused() {
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: vec![],
    };
    let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("self-contradictory"), "got: {msg}");
}

/// A descriptor whose caller-supplied vectors exceed the validation cap is
/// rejected up front, before any per-entry scan, so validation cost stays
/// bounded regardless of how large an untrusted descriptor is.
#[test]
fn validate_rejects_descriptor_exceeding_entry_cap() {
    let over_cap = OnlineArtifactDescriptor::MAX_VALIDATED_ENTRIES + 1;
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: (0..over_cap).map(|i| format!("t{i}")).collect(),
    };
    let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("exceeding the maximum"), "got: {msg}");
}

/// Regression test, preserved under the
/// single-representation contract: a real R3 recipe declaring an
/// unrelated, nonempty `asymmetric_tensor_names` entry must be rejected
/// for not matching any tensor the recipe's derived expected set
/// contains — an unrelated string must not satisfy the fail-closed
/// requirement just by being nonempty.
#[test]
fn v1_online_unrelated_asymmetric_declaration_is_refused() {
    // Full, correct per-layer coverage PLUS one unrelated extra entry —
    // isolates the extra-direction (unrelated) check from the
    // missing-direction check exercised by
    // `v1_online_missing_asymmetric_declaration_is_refused` above.
    let mut names = full_r3_layer_names();
    names.push("unrelated".to_string());
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: names,
    };
    let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unrelated") && msg.contains("does not match any tensor"),
        "got: {msg}"
    );
}

/// A declared name with the right suffix but an out-of-scope layer
/// (layer 0 is GDN, outside `sample_r3_spec()`'s full-attention scope)
/// must also be rejected as "extra" — exact-set equality, not a
/// suffix-only membership check.
#[test]
fn v1_online_out_of_scope_layer_declaration_is_refused() {
    let mut names = full_r3_layer_names();
    names.push("model.language_model.layers.0.self_attn.o_proj.weight".to_string());
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: names,
    };
    let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("layers.0.self_attn.o_proj.weight")
            && msg.contains("does not match any tensor"),
        "got: {msg}"
    );
}

#[test]
fn v1_online_with_full_asymmetric_coverage_is_valid() {
    // Regression test: a complete, exact declaration
    // (every scoped layer's tensor, nothing extra) must pass.
    let names = full_r3_layer_names();
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: names,
    };
    // cfg is mandatory for V1Online — see
    // `v1_online_validate_without_cfg_is_refused` for the None case.
    assert!(descriptor.validate(Some(&sample_r3_cfg())).is_ok());
}

// --- OnlineRotationSpec::validate ---

#[test]
fn online_rotation_spec_accepts_constructor_built_specs() {
    let cfg = sample_r3_cfg();
    assert!(sample_r3_spec().validate(None).is_ok());
    assert!(sample_r3_spec().validate(Some(&cfg)).is_ok());
    let r4 = OnlineRotationSpec::r4_dense_mlp(&cfg, 9, 256).unwrap();
    assert!(r4.validate(None).is_ok());
    assert!(r4.validate(Some(&cfg)).is_ok());
}

/// The exact malformed spec: an R3 id
/// with `OutputSide`, `block_size=3` (not a power of two), and
/// `layer_scope=None` (R3 requires Some). Directly constructed —
/// bypassing `r3_full_attention` — because that is precisely how a
/// hand-edited or corrupted index could produce this. Must be rejected
/// both standalone and via the descriptor path.
#[test]
fn online_rotation_spec_rejects_the_reported_malformed_r3_spec() {
    let malformed = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::OutputSide,
        seed: 1,
        block_size: 3,
        layer_scope: None,
    };
    assert!(malformed.validate(None).is_err());

    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![malformed],
        asymmetric_tensor_names: vec![
            "model.language_model.layers.3.self_attn.o_proj.weight".to_string(),
        ],
    };
    let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
    assert!(
        format!("{err}").contains("InputSide"),
        "descriptor validation must surface the spec-level rejection, got: {err}"
    );
}

#[test]
fn online_rotation_spec_rejects_wrong_side() {
    let spec = OnlineRotationSpec {
        id: RotationId::MlpDownR4,
        side: AbsorptionSide::OutputSide,
        seed: 1,
        block_size: 64,
        layer_scope: None,
    };
    let err = spec.validate(None).unwrap_err();
    assert!(format!("{err}").contains("InputSide"));
}

#[test]
fn online_rotation_spec_rejects_non_power_of_two_block_size() {
    let spec = OnlineRotationSpec {
        id: RotationId::MlpDownR4,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 96,
        layer_scope: None,
    };
    let err = spec.validate(None).unwrap_err();
    assert!(format!("{err}").contains("power-of-two"));
}

#[test]
fn online_rotation_spec_rejects_zero_block_size() {
    let spec = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 0,
        layer_scope: Some(vec![3]),
    };
    let err = spec.validate(None).unwrap_err();
    assert!(format!("{err}").contains("power-of-two"));
}

#[test]
fn online_rotation_spec_rejects_r3_with_none_layer_scope() {
    let spec = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 8,
        layer_scope: None,
    };
    let err = spec.validate(None).unwrap_err();
    assert!(format!("{err}").contains("layer_scope"));
}

#[test]
fn online_rotation_spec_rejects_r3_with_empty_layer_scope() {
    let spec = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 8,
        layer_scope: Some(vec![]),
    };
    let err = spec.validate(None).unwrap_err();
    assert!(format!("{err}").contains("must not be empty"));
}

#[test]
fn online_rotation_spec_rejects_r4_with_some_layer_scope() {
    let spec = OnlineRotationSpec {
        id: RotationId::MlpDownR4,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 64,
        layer_scope: Some(vec![0]),
    };
    let err = spec.validate(None).unwrap_err();
    assert!(format!("{err}").contains("must be None"));
}

#[test]
fn online_rotation_spec_rejects_residual_stream_id() {
    let spec = OnlineRotationSpec {
        id: RotationId::ResidualStream,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 8,
        layer_scope: None,
    };
    let err = spec.validate(None).unwrap_err();
    assert!(format!("{err}").contains("ResidualStream"));
}

#[test]
fn online_rotation_spec_cfg_aware_checks_catch_bad_divisibility_and_scope() {
    let cfg = sample_r3_cfg();
    // block_size divides nothing meaningful without cfg, but with cfg
    // it must divide num_attention_heads (8) — 4 does. layer_scope must
    // be the config's full-attention layers exactly, so this uses the
    // full set (not a subset — see the equality test below).
    let good_block = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 4, // power of two, divides 8 — valid without extra checks
        layer_scope: Some(vec![3, 7, 11, 15, 19, 23]),
    };
    assert!(good_block.validate(Some(&cfg)).is_ok());

    // A layer_scope entry that is not actually a full-attention layer
    // (layer 0 is GDN in qwen35_0_8b) must be rejected once cfg is known.
    let bad_scope = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 8,
        layer_scope: Some(vec![0]),
    };
    assert!(!cfg.is_full_attention(0), "sanity: layer 0 is GDN");
    let err = bad_scope.validate(Some(&cfg)).unwrap_err();
    assert!(format!("{err}").contains("layer 0"));
}

/// The cfg-aware check must reject a
/// `layer_scope` that is a strict subset of the config's full-attention
/// layers — membership alone is not sufficient, coverage must be
/// complete. For `qwen35_0_8b`, full-attention layers are
/// `[3,7,11,15,19,23]`; a scope of just `[3]` must be rejected, naming
/// the missing layers.
#[test]
fn online_rotation_spec_rejects_r3_scope_missing_full_attention_layers() {
    let cfg = sample_r3_cfg();
    let partial_scope = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 8,
        layer_scope: Some(vec![3]),
    };
    let err = partial_scope.validate(Some(&cfg)).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("missing"), "got: {msg}");
    for missing in ["7", "11", "15", "19", "23"] {
        assert!(msg.contains(missing), "expected {missing} in: {msg}");
    }

    // The full set passes.
    let full_scope = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 8,
        layer_scope: Some(vec![3, 7, 11, 15, 19, 23]),
    };
    assert!(full_scope.validate(Some(&cfg)).is_ok());

    // A superset naming a non-full-attention layer is still rejected
    // (the membership direction, unaffected by the equality fix).
    let superset_with_non_member = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 8,
        layer_scope: Some(vec![0, 3, 7, 11, 15, 19, 23]),
    };
    let err = superset_with_non_member.validate(Some(&cfg)).unwrap_err();
    assert!(format!("{err}").contains("layer 0"));
}

/// A scope with a duplicate layer index (the
/// exact `[3,3,7,11,15,19,23]` example from the review) must be
/// rejected outright — a naive set-equality check would silently dedup
/// it and accept. The check is cfg-independent, so it must also fire
/// with `cfg: None`.
#[test]
fn online_rotation_spec_rejects_duplicate_r3_layer_index() {
    let duplicate_scope = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 8,
        layer_scope: Some(vec![3, 3, 7, 11, 15, 19, 23]),
    };
    let err = duplicate_scope.validate(None).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("strictly sorted ascending") && msg.contains("duplicates"),
        "got: {msg}"
    );
    let err = duplicate_scope
        .validate(Some(&sample_r3_cfg()))
        .unwrap_err();
    assert!(
        format!("{err}").contains("strictly sorted ascending"),
        "duplicate rejection must fire before/independent of cfg checks"
    );
}

/// An unsorted (but duplicate-free) scope must
/// also be rejected — the canonical form is strictly ascending, not
/// merely a valid set.
#[test]
fn online_rotation_spec_rejects_unsorted_r3_layer_scope() {
    let unsorted_scope = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 8,
        layer_scope: Some(vec![7, 3, 11, 15, 19, 23]),
    };
    let err = unsorted_scope.validate(None).unwrap_err();
    assert!(
        format!("{err}").contains("strictly sorted ascending"),
        "got: {err}"
    );
}

/// A V1Online descriptor must not accept two
/// online rotation specs targeting the same runtime site — two
/// `AttentionOutputR3` entries (even with different seeds) leave the
/// future runtime to invent duplicate-site semantics, so the second one
/// is rejected.
#[test]
fn v1_online_rejects_two_specs_targeting_the_same_site() {
    let cfg = sample_r3_cfg();
    let first = OnlineRotationSpec::r3_full_attention(&cfg, 42, 8).unwrap();
    let second = OnlineRotationSpec::r3_full_attention(&cfg, 99, 8).unwrap();
    let names = full_r3_layer_names();
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![first, second],
        asymmetric_tensor_names: names,
    };
    let err = descriptor.validate(Some(&cfg)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("same runtime site") && msg.contains("AttentionOutputPreOProj"),
        "got: {msg}"
    );
}

/// A well-formed single-spec-per-site descriptor (R3 + R4 together)
/// must still pass — the same-site rejection must not false-positive
/// on two specs that target *different* sites.
#[test]
fn v1_online_with_distinct_sites_is_valid() {
    let cfg = sample_r3_cfg();
    let r3 = OnlineRotationSpec::r3_full_attention(&cfg, 42, 8).unwrap();
    let r4 = OnlineRotationSpec::r4_dense_mlp(&cfg, 9, 256).unwrap();
    let mut names = full_r3_layer_names();
    for i in 0..cfg.num_hidden_layers {
        names.push(format!(
            "{}.mlp.down_proj.weight",
            crate::model::qwen35::qwen_layer_tensor_prefix(i)
        ));
    }
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![r3, r4],
        asymmetric_tensor_names: names,
    };
    assert!(descriptor.validate(Some(&cfg)).is_ok());
}

// --- canonical Qwen3.5 tensor namespace ---

/// The canonical Qwen3.5 tensor namespace is
/// `model.language_model.layers.{idx}.{suffix}` (matching
/// `qwen_required_tensor_names` and the QuaRot converter), not
/// `model.layers.{idx}.{suffix}`. A descriptor declaring the canonical
/// names for the full R3 scope must pass; the same descriptor rewritten
/// with the old, non-canonical prefix must now fail per-layer coverage.
#[test]
fn v1_online_per_layer_coverage_uses_canonical_qwen_namespace() {
    let cfg = sample_r3_cfg();
    let canonical_names = full_r3_layer_names();
    for name in &canonical_names {
        assert!(
            name.starts_with("model.language_model.layers."),
            "got: {name}"
        );
    }
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: canonical_names,
    };
    assert!(descriptor.validate(Some(&cfg)).is_ok());

    let old_namespace_names: Vec<String> = [3usize, 7, 11, 15, 19, 23]
        .iter()
        .map(|i| format!("model.layers.{i}.self_attn.o_proj.weight"))
        .collect();
    let old_descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: old_namespace_names,
    };
    let err = old_descriptor.validate(Some(&cfg)).unwrap_err();
    assert!(
        format!("{err}").contains("coverage is incomplete"),
        "old model.layers.* namespace must be rejected as incomplete coverage"
    );
}

// --- per-layer tensor coverage ---

/// The exact scenario from the finding: an R3 recipe scoped to layers
/// `[3,7,11,15,19,23]` must not validate against a declaration that only
/// contains layer 0's tensor (suffix matches, layer doesn't). Must fail
/// naming a missing scoped layer.
#[test]
fn v1_online_layer_scope_coverage_rejects_wrong_layer_declaration() {
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: vec![
            "model.language_model.layers.0.self_attn.o_proj.weight".to_string(),
        ],
    };
    let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("layer 3") && msg.contains("coverage is incomplete"),
        "got: {msg}"
    );
}

/// Partial coverage — every scoped layer but one declared — must still
/// be rejected, naming the missing layer.
#[test]
fn v1_online_layer_scope_coverage_rejects_partial_declaration() {
    let mut names = full_r3_layer_names();
    names.pop(); // drop layer 23
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: names,
    };
    let err = descriptor.validate(Some(&sample_r3_cfg())).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("layer 23") && msg.contains("coverage is incomplete"),
        "got: {msg}"
    );
}

/// R4 (`layer_scope: None`, "every layer"
/// semantics) validation without `cfg` is refused outright — the prior
/// behavior of silently *skipping* the per-layer coverage check for a
/// `None`-scoped rotation (because the full layer count was
/// "unknowable") was the exact bug the review escalated: a V1/R4
/// descriptor declaring only `model.language_model.layers.0.mlp.down_proj.weight` must
/// no longer pass any validation path that claims completeness, with or
/// without cfg games. With `cfg` supplied, the same partial declaration
/// is rejected for missing coverage of the layers `cfg` says exist.
#[test]
fn v1_online_r4_partial_declaration_rejected_regardless_of_cfg() {
    let cfg = sample_r3_cfg();
    let r4 = OnlineRotationSpec::r4_dense_mlp(&cfg, 9, 256).unwrap();
    let partial_name = "model.language_model.layers.0.mlp.down_proj.weight".to_string();
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![r4],
        asymmetric_tensor_names: vec![partial_name],
    };
    // Without cfg: refused immediately — no path claims completeness
    // without model config.
    let err = descriptor.validate(None).unwrap_err();
    assert!(format!("{err}").contains("requires model config"));
    // With cfg: full-layer coverage is enforced and layer 1..N are
    // missing, so validation must fail here too.
    let err = descriptor.validate(Some(&cfg)).unwrap_err();
    assert!(format!("{err}").contains("coverage is incomplete"));
}

// --- verify_affected (narrow converter cross-check entry point) ---

#[test]
fn verify_affected_accepts_exact_match() {
    let cfg = sample_r3_cfg();
    let names = full_r3_layer_names();
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: names.clone(),
    };
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    assert!(descriptor.verify_affected(&cfg, &refs).is_ok());
}

#[test]
fn verify_affected_rejects_missing_and_extra() {
    let cfg = sample_r3_cfg();
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![sample_r3_spec()],
        asymmetric_tensor_names: full_r3_layer_names(),
    };
    // Missing: converter reports transforming nothing.
    assert!(descriptor.verify_affected(&cfg, &[]).is_err());
    // Extra: converter reports an unrelated tensor alongside full coverage.
    let mut names = full_r3_layer_names();
    names.push("unrelated".to_string());
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    assert!(descriptor.verify_affected(&cfg, &refs).is_err());
}
