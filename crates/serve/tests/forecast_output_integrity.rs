use std::path::Path;
use std::process::Command;

fn tiny_chronos_checkpoint(dir: &Path) {
    std::fs::create_dir(dir).expect("create model directory");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "d_model":2,"d_kv":2,"num_heads":1,"d_ff":2,"num_layers":0,
            "layer_norm_epsilon":0.000001,"rope_theta":10000.0,"dense_act_fn":"relu",
            "chronos_config":{
                "context_length":4,"input_patch_size":1,"input_patch_stride":1,
                "output_patch_size":1,"max_output_patches":2,"quantiles":[0.5],
                "use_arcsinh":true,"use_reg_token":false
            }
        }"#,
    )
    .expect("write config");

    let mut header = serde_json::Map::new();
    let mut data_len = 0usize;
    let mut tensor = |name: &str, shape: &[usize]| {
        let start = data_len;
        data_len += shape.iter().product::<usize>() * 4;
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,data_len]}),
        );
    };
    tensor("shared.weight", &[1, 2]);
    for (prefix, input, output) in [
        ("input_patch_embedding", 3, 2),
        ("output_patch_embedding", 2, 1),
    ] {
        for (layer, in_dim, out_dim) in [
            ("hidden_layer", input, 2),
            ("output_layer", 2, output),
            ("residual_layer", input, output),
        ] {
            tensor(&format!("{prefix}.{layer}.weight"), &[out_dim, in_dim]);
            tensor(&format!("{prefix}.{layer}.bias"), &[out_dim]);
        }
    }
    tensor("encoder.final_layer_norm.weight", &[2]);
    let mut header = serde_json::to_vec(&header).expect("serialize tensor header");
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.resize(file.len() + data_len, 0);
    std::fs::write(dir.join("model.safetensors"), file).expect("write weights");
}

fn run_forecast(model: &Path, input: &Path, output: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_lattice"))
        .arg("forecast")
        .arg("--model")
        .arg(model)
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(output)
        .output()
        .expect("run lattice forecast")
}

#[test]
fn later_forecast_error_preserves_existing_output() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let model = temp.path().join("model");
    tiny_chronos_checkpoint(&model);
    let input = temp.path().join("input.jsonl");
    let output = temp.path().join("existing-output.jsonl");
    std::fs::write(
        &input,
        b"{\"target\":[1.0],\"horizon\":1}\n{\"target\":[1.0],\"horizon\":0}\n",
    )
    .expect("write input");
    let sentinel = b"previous forecast output\n";
    std::fs::write(&output, sentinel).expect("write existing output");

    let result = run_forecast(&model, &input, &output);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "invalid horizon must fail");
    assert!(
        stderr.contains("request 2") && stderr.contains("horizon 0"),
        "fixture must load and forecast its first request before refusing the second: {stderr}"
    );
    assert_eq!(std::fs::read(&output).expect("read output"), &sentinel[..]);
}

#[test]
fn forecast_error_preserves_input_when_input_is_output() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let model = temp.path().join("model");
    tiny_chronos_checkpoint(&model);
    let input = temp.path().join("same.jsonl");
    let request = b"{\"target\":[1.0],\"horizon\":0}\n";
    std::fs::write(&input, request).expect("write input");

    let result = run_forecast(&model, &input, &input);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "invalid horizon must fail");
    assert!(
        stderr.contains("request 1") && stderr.contains("horizon 0"),
        "fixture must load and reach forecast validation: {stderr}"
    );
    assert_eq!(std::fs::read(&input).expect("read input"), &request[..]);
}
