use super::{
    BertModel, MAX_ENCODE_WORKING_SET_BYTES, checkpoint_support, packed_encode_working_set_bytes,
};
use std::fs::OpenOptions;
use std::io::Write;

#[test]
#[ignore = "writes exact embedding bits; set LATTICE_BERT_SCRATCH_DUMP to a new file"]
fn dump_packed_attention_embeddings() {
    let destination = std::env::var_os("LATTICE_BERT_SCRATCH_DUMP")
        .expect("set LATTICE_BERT_SCRATCH_DUMP to a new output file");
    let checkpoint = checkpoint_support::tiny_bert_checkpoint(3, 2);
    let model = BertModel::from_directory(checkpoint.path()).unwrap();
    let mut cases = Vec::new();
    for (name, batch, capped) in [
        ("public-b1", 1, false),
        ("public-b3", 3, false),
        ("public-b17", 17, false),
        ("cap-split-b17", 17, true),
    ] {
        let texts = &checkpoint_support::MIXED_INPUTS[..batch];
        let lengths: Vec<_> = model
            .tokenizer
            .tokenize_batch(texts)
            .iter()
            .map(|input| input.real_length)
            .collect();
        assert!(lengths.iter().all(|&length| length > 0 && length <= 32));
        let longest = *lengths.iter().max().unwrap();
        let whole =
            packed_encode_working_set_bytes(Some(lengths.iter().sum()), longest, model.config())
                .unwrap();
        let cap = if capped {
            let cap = packed_encode_working_set_bytes(Some(longest * 2), longest, model.config())
                .unwrap();
            assert!(whole > cap, "{name}: fixture must require multiple chunks");
            for &length in &lengths {
                assert!(
                    packed_encode_working_set_bytes(Some(length), length, model.config()).unwrap()
                        <= cap,
                    "{name}: one sequence must fit"
                );
            }
            cap
        } else {
            assert!(whole <= MAX_ENCODE_WORKING_SET_BYTES);
            MAX_ENCODE_WORKING_SET_BYTES
        };
        let outputs = if capped {
            model.encode_batch_packed_capped(texts, cap).unwrap()
        } else {
            model.encode_batch(texts).unwrap()
        };
        assert_eq!(outputs.len(), batch, "{name}: output rows");
        let mut bits = Vec::with_capacity(batch);
        for (row, values) in outputs.iter().enumerate() {
            assert_eq!(values.len(), 4, "{name}: row {row} columns");
            assert!(
                values.iter().all(|value| value.is_finite()),
                "{name}: row {row}"
            );
            bits.push(
                values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            );
        }
        if batch > 1 {
            assert!(lengths.windows(2).any(|pair| pair[0] > pair[1]));
            assert!(
                bits.windows(2).any(|pair| pair[0] != pair[1]),
                "{name}: distinct outputs"
            );
        }
        cases.push(serde_json::json!({
            "case": name, "rows": batch, "columns": 4,
            "route": if batch == 1 { "single" } else { "packed" },
            "cap_bytes": cap, "requires_multiple_chunks": capped,
            "whole_packed_bytes": whole, "token_lengths": lengths, "f32_bits": bits,
        }));
    }
    assert_eq!(cases.len(), 4);
    let record = serde_json::json!({
        "schema": "bert-packed-attention-bits-v1", "layers": 3, "heads": 2,
        "hidden_size": 4, "case_count": 4, "row_count": 38, "value_count": 152,
        "cases": cases,
    });
    let bytes = serde_json::to_vec_pretty(&record).unwrap();
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .expect("dump destination must be a new writable file");
    output.write_all(&bytes).unwrap();
    output.write_all(b"\n").unwrap();
    output.sync_all().unwrap();
    println!("wrote 4 cases, 38 rows, 152 exact f32 bit patterns");
}
