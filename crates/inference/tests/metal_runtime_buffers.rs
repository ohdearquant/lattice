#![cfg(all(target_os = "macos", feature = "metal-gpu"))]

use lattice_inference_core::InferenceError;
use lattice_metal_runtime::{BufferPolicy, UploadCopy, ZeroLengthPolicy, allocate, upload};
use metal::{Buffer, Device};

fn policy(zero_length: ZeroLengthPolicy) -> BufferPolicy {
    BufferPolicy {
        zero_length,
        invalid: InferenceError::InvalidInput,
        runtime: InferenceError::Inference,
    }
}

fn expect_empty(_: &Buffer, values: &[f32]) -> Result<(), InferenceError> {
    assert!(values.is_empty(), "the empty upload copied {values:?}");
    Ok(())
}

#[test]
fn zero_length_requests_follow_their_policy_on_the_device() {
    let _gpu = lattice_inference::measurement::gpu_test_lock();
    let Some(device) = Device::system_default() else {
        assert!(
            std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_none(),
            "Metal device required under enforcement"
        );
        eprintln!("SKIP metal_runtime: no Metal device");
        return;
    };
    let minimum = policy(ZeroLengthPolicy::MinimumF32);
    let buffer = allocate(&device, 0, "zero-minimum", minimum).expect("one f32 is allocated");
    assert!(buffer.length() >= 4, "length {}", buffer.length());
    let buffer = upload(
        &device,
        &[],
        "empty-minimum",
        minimum,
        UploadCopy::Checked(expect_empty),
    )
    .expect("an empty upload allocates one f32");
    assert!(buffer.length() >= 4, "length {}", buffer.length());

    let exact = policy(ZeroLengthPolicy::Exact);
    match allocate(&device, 0, "zero-exact", exact).map(|buffer| buffer.length()) {
        Err(InferenceError::Inference(reason)) => {
            assert_eq!(reason, "zero-exact allocation failed")
        }
        other => panic!("an exact zero-length request must be refused, got {other:?}"),
    }
    match upload(&device, &[], "empty-exact", exact, UploadCopy::Direct)
        .map(|buffer| buffer.length())
    {
        Err(InferenceError::Inference(reason)) => {
            assert_eq!(reason, "empty-exact allocation failed")
        }
        other => panic!("an exact empty upload must be refused, got {other:?}"),
    }
}
