use super::super::pre_tokenize;
use crate::common::load_tokenizer_with_report;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use unicode_categories::UnicodeCategories;

const SCALAR_COUNT: usize = 1_112_064;
const INTERACTIONS: &[(&str, &str)] = &[
    ("clean_before_marks", "A\u{200d}\u{0483}B"),
    ("canonical_order", "A\u{0301}\u{0323}B"),
    ("nonspacing_and_enclosing", "A\u{0345}\u{20dd}B"),
    ("lowercase_expansion", "İ ΟΣ ẞ"),
    ("canonical_not_compatibility", "Æ ß Ø œ ı ﬁ"),
    ("hangul_and_jamo", "가 각 가"),
    (
        "cjk_before_decomposition",
        "a\u{f900}b a\u{2b820}b a\u{2b920}b",
    ),
    (
        "control_boundaries",
        "a\tb\nc\rd a\u{000b}b a\u{000c}b a\u{0085}b",
    ),
    (
        "unassigned_and_replacement",
        "a\u{0378}b a\u{1aff}b a\u{fffd}b",
    ),
    ("decomposed_nonlatin", "α\u{0301} \u{1e9b} カ\u{3099}"),
];
const CORPUS: &[&str] = &[
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "Rust 2024: arrays, slices, and strings!",
    "query: efficient local search",
    "multiple\tspaces\nand\rline breaks",
    "cafe resume naive",
    "中文 日本語",
];

fn scalars(text: &str) -> Vec<u32> {
    text.chars().map(u32::from).collect()
}

fn pieces(text: &str) -> Vec<Vec<u32>> {
    pre_tokenize(text)
        .iter()
        .map(|word| scalars(word))
        .collect()
}

fn new_dump() -> std::io::Result<BufWriter<File>> {
    let path = PathBuf::from(
        std::env::var_os("LATTICE_WORDPIECE_DUMP").expect("set LATTICE_WORDPIECE_DUMP"),
    );
    assert!(path.is_absolute(), "dump destination must be absolute");
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map(BufWriter::new)
}

fn complete(mut output: BufWriter<File>) -> std::io::Result<()> {
    output.flush()?;
    output.get_ref().sync_all()
}

#[test]
#[ignore = "writes the complete Unicode-scalar preprocessing artifact"]
fn dump_wordpiece_scalar_pieces() -> std::io::Result<()> {
    let mut output = new_dump()?;
    let (major, minor, patch) = std::char::UNICODE_VERSION;
    let unicode = [major, minor, patch];
    writeln!(
        output,
        "{{\"type\":\"header\",\"schema\":\"lattice-wordpiece-pieces-v1\",\"scalar_start\":0,\"scalar_end\":1114111,\"surrogates\":[55296,57343],\"scalar_count\":{SCALAR_COUNT},\"rust_unicode\":{unicode:?},\"interaction_count\":{}}}",
        INTERACTIONS.len()
    )?;
    let mut count = 0usize;
    for cp in 0..=0x10ffff {
        let Some(ch) = char::from_u32(cp) else {
            continue;
        };
        let classes = [
            ch.is_other_control(),
            ch.is_other_format(),
            ch.is_other_private_use(),
            ch.is_mark_nonspacing(),
        ];
        writeln!(
            output,
            "{{\"type\":\"scalar\",\"cp\":{cp},\"classes\":{classes:?},\"pieces\":{:?}}}",
            pieces(&format!("a{ch}b"))
        )?;
        count += 1;
    }
    assert_eq!(count, SCALAR_COUNT);
    for (index, (label, text)) in INTERACTIONS.iter().enumerate() {
        writeln!(
            output,
            "{{\"type\":\"interaction\",\"case\":{index},\"label\":{:?},\"input\":{:?},\"pieces\":{:?}}}",
            scalars(label),
            scalars(text),
            pieces(text)
        )?;
    }
    writeln!(
        output,
        "{{\"type\":\"complete\",\"scalar_count\":{count},\"interaction_count\":{}}}",
        INTERACTIONS.len()
    )?;
    complete(output)
}

#[test]
#[ignore = "requires a fixed real WordPiece tokenizer directory and new dump destination"]
fn dump_wordpiece_fixed_corpus() -> std::io::Result<()> {
    let directory = PathBuf::from(
        std::env::var_os("LATTICE_WORDPIECE_MODEL_DIR").expect("set LATTICE_WORDPIECE_MODEL_DIR"),
    );
    assert!(directory.is_absolute(), "model directory must be absolute");
    let (tokenizer, report) = load_tokenizer_with_report(&directory).unwrap();
    assert!(matches!(
        report.format,
        "wordpiece-json" | "wordpiece-vocab-txt"
    ));
    assert!(!report.inputs.is_empty());
    let mut output = new_dump()?;
    writeln!(
        output,
        "{{\"type\":\"header\",\"schema\":\"lattice-wordpiece-corpus-v1\",\"case_count\":{},\"artifact_count\":{},\"format\":\"{}\",\"max_seq_len\":{}}}",
        CORPUS.len(),
        report.inputs.len(),
        report.format,
        tokenizer.max_seq_len()
    )?;
    for (index, input) in report.inputs.iter().enumerate() {
        let digest = Sha256::digest(&input.bytes).to_vec();
        writeln!(
            output,
            "{{\"type\":\"artifact\",\"index\":{index},\"name\":{:?},\"bytes\":{},\"sha256\":{digest:?}}}",
            scalars(&input.logical_identifier),
            input.bytes.len()
        )?;
    }
    for (index, text) in CORPUS.iter().enumerate() {
        let input = tokenizer.tokenize(text);
        assert!(input.real_length > 0);
        assert_eq!(input.input_ids.len(), input.attention_mask.len());
        assert_eq!(input.input_ids.len(), input.token_type_ids.len());
        assert!(input.real_length <= input.input_ids.len());
        assert_eq!(
            input.pre_truncation_len, input.real_length,
            "corpus must not truncate"
        );
        writeln!(
            output,
            "{{\"type\":\"case\",\"case\":{index},\"input\":{:?},\"ids\":{:?},\"mask\":{:?},\"types\":{:?},\"real_length\":{},\"pre_truncation_len\":{}}}",
            scalars(text),
            input.input_ids,
            input.attention_mask,
            input.token_type_ids,
            input.real_length,
            input.pre_truncation_len
        )?;
    }
    writeln!(
        output,
        "{{\"type\":\"complete\",\"case_count\":{}}}",
        CORPUS.len()
    )?;
    complete(output)
}
