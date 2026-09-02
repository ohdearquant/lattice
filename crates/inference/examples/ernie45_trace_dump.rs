//! Dev instrument: dump the full ERNIE-4.5 forward trace for one goldens case
//! as raw little-endian f32 files, for offline diffing against the HF
//! activation archive. Not part of any gate.
use lattice_inference::model::ernie45::{Ernie45Config, Ernie45Model, Ernie45Weights};
use lattice_inference::weights::SafetensorsFile;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args
        .get(1)
        .expect("usage: ernie45_trace_dump <case_id> <out_dir>");
    let out_dir = PathBuf::from(args.get(2).expect("out dir"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let home = std::env::var("HOME").unwrap();
    let dir = PathBuf::from(home).join(".lattice/models/paddleocr-vl-1.6");
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/paddleocr_vl/decoder/decoder_goldens.json");
    let golden: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture).unwrap()).unwrap();
    let ids: Vec<u32> = golden["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == *case)
        .expect("case exists")["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let cfg = Ernie45Config::from_config_json(&dir.join("config.json")).unwrap();
    let mut source = SafetensorsFile::open(&dir.join("model.safetensors")).unwrap();
    let weights = Ernie45Weights::load(&mut source, &cfg).unwrap();
    let model = Ernie45Model::new(cfg, weights).unwrap();
    let trace = model.forward_trace(&ids).unwrap();

    let write = |name: &str, buf: &[f32]| {
        let mut f = std::fs::File::create(out_dir.join(format!("{name}.f32"))).unwrap();
        let bytes: Vec<u8> = buf.iter().flat_map(|v| v.to_le_bytes()).collect();
        f.write_all(&bytes).unwrap();
    };
    write("embed", &trace.embed);
    for (i, l) in trace.layer_outputs.iter().enumerate() {
        write(&format!("layer_{i}"), l);
    }
    write("final_norm", &trace.final_norm);
    write("logits", &trace.logits);
    println!("dumped {} ids to {}", ids.len(), out_dir.display());
}
