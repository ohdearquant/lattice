use super::*;
use serde_json::{Map, json};

fn fixture(tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)>) -> SafetensorsFile {
    let mut entries = Map::new();
    let mut payload = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = payload.len();
        payload.extend_from_slice(&bytes);
        entries.insert(
            name.to_string(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
        );
    }
    let header = serde_json::to_vec(&entries).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&payload);
    SafetensorsFile::from_bytes(file).unwrap()
}

fn pair(
    block_shape: Vec<usize>,
    scale_shape: Vec<usize>,
    block_dtype: &str,
    scale_dtype: &str,
) -> SafetensorsFile {
    fixture(vec![
        ("proj_blocks", block_dtype, block_shape, vec![0x21; 64]),
        (
            "proj_scales",
            scale_dtype,
            scale_shape,
            vec![127, 128, 126, 127],
        ),
    ])
}

#[test]
fn hf_gpt_oss_pair_decodes_all_rows_and_blocks_in_checkpoint_order() {
    let file = pair(vec![2, 1, 2, 16], vec![2, 1, 2], "U8", "U8");
    let mut output = [0.0; 128];
    dequantize_mxfp4_safetensors(&file, "proj", &mut output).unwrap();
    for (group, (low, high)) in [(1.0, 2.0), (2.0, 4.0), (0.5, 1.0), (1.0, 2.0)]
        .into_iter()
        .enumerate()
    {
        for pair in output[group * 32..(group + 1) * 32].chunks_exact(2) {
            assert_eq!(pair, &[low, high]);
        }
    }
}

#[test]
fn rejects_wrong_shape_dtype_and_output_without_writes() {
    let bad = [
        pair(vec![64], vec![4], "U8", "U8"),
        pair(vec![2, 1, 4, 8], vec![2, 1, 2], "U8", "U8"),
        pair(vec![2, 1, 2, 16], vec![1, 2, 2], "U8", "U8"),
        pair(vec![2, 1, 2, 16], vec![2, 1, 2], "I8", "U8"),
        pair(vec![2, 1, 2, 16], vec![2, 1, 2], "U8", "I8"),
    ];
    for file in &bad {
        let mut output = [9.0; 128];
        assert!(dequantize_mxfp4_safetensors(file, "proj", &mut output).is_err());
        assert_eq!(output, [9.0; 128]);
    }
    let file = pair(vec![2, 1, 2, 16], vec![2, 1, 2], "U8", "U8");
    let mut short = [9.0; 127];
    assert!(dequantize_mxfp4_safetensors(&file, "proj", &mut short).is_err());
    assert_eq!(short, [9.0; 127]);
}

#[test]
fn missing_or_misnamed_pair_and_empty_stem_are_rejected() {
    let file = fixture(vec![("proj_blocks", "U8", vec![1, 16], vec![0x21; 16])]);
    let mut output = [9.0; 32];
    for stem in ["proj", "other", ""] {
        assert!(dequantize_mxfp4_safetensors(&file, stem, &mut output).is_err());
        assert_eq!(output, [9.0; 32]);
    }
}

#[test]
fn empty_matrices_and_overflow_are_checked() {
    let empty = fixture(vec![
        ("proj_blocks", "U8", vec![0, 2, 16], vec![]),
        ("proj_scales", "U8", vec![0, 2], vec![]),
    ]);
    dequantize_mxfp4_safetensors(&empty, "proj", &mut []).unwrap();
    let no_groups = fixture(vec![
        ("proj_blocks", "U8", vec![2, 0, 16], vec![]),
        ("proj_scales", "U8", vec![2, 0], vec![]),
    ]);
    dequantize_mxfp4_safetensors(&no_groups, "proj", &mut []).unwrap();
    let overflow = fixture(vec![
        ("proj_blocks", "U8", vec![0, usize::MAX, 16], vec![]),
        ("proj_scales", "U8", vec![0, usize::MAX], vec![]),
    ]);
    assert!(dequantize_mxfp4_safetensors(&overflow, "proj", &mut []).is_err());
}

#[test]
fn reserved_scale_propagates_nan_as_documented() {
    let file = fixture(vec![
        ("proj_blocks", "U8", vec![1, 16], vec![0x21; 16]),
        ("proj_scales", "U8", vec![1], vec![255]),
    ]);
    let mut output = [0.0; 32];
    dequantize_mxfp4_safetensors(&file, "proj", &mut output).unwrap();
    assert!(output.iter().all(|value| value.is_nan()));
}

#[test]
fn on_disk_safetensors_pair_loads_via_the_existing_file_reader() {
    use std::io::Write;

    let header = br#"{"proj_blocks":{"dtype":"U8","shape":[1,16],"data_offsets":[0,16]},"proj_scales":{"dtype":"U8","shape":[1],"data_offsets":[16,17]}}"#;
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header).unwrap();
    file.write_all(&[0x21; 16]).unwrap();
    file.write_all(&[127]).unwrap();
    file.flush().unwrap();
    let parsed = SafetensorsFile::open(file.path()).unwrap();
    let mut output = [0.0; 32];
    dequantize_mxfp4_safetensors(&parsed, "proj", &mut output).unwrap();
    for pair in output.chunks_exact(2) {
        assert_eq!(pair, &[1.0, 2.0]);
    }
}
