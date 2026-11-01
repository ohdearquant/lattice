use std::io::{self, Write};
use std::path::PathBuf;

#[derive(clap::Args, Debug)]
pub(super) struct Args {
    /// Existing local BERT-compatible checkpoint directory
    #[arg(long, value_name = "DIR")]
    pub(super) model: PathBuf,
    /// One literal text to embed, without reading stdin or adding a role prefix
    #[arg(long, value_name = "TEXT")]
    pub(super) text: String,
}

pub(super) fn run(args: &Args) -> Result<(), String> {
    let plan = lattice_serve::serving_provider::plan_text_embeddings(&args.model, None)
        .map_err(|error| format!("embedding model selection failed: {error}"))?;
    let model = plan.load()?;
    let embeddings = model
        .encode_batch(std::slice::from_ref(&args.text))
        .map_err(|error| format!("embedding failed: {error}"))?;
    let stdout = io::stdout();
    let mut output = stdout.lock();
    write_embedding(embeddings, &mut output)
}

fn write_embedding(embeddings: Vec<Vec<f32>>, output: &mut impl Write) -> Result<(), String> {
    let [embedding] = embeddings.as_slice() else {
        return Err(format!(
            "embedding model returned {} vectors for one input",
            embeddings.len()
        ));
    };
    if embedding.iter().any(|value| !value.is_finite()) {
        return Err("embedding contains a non-finite value".into());
    }
    serde_json::to_writer(&mut *output, embedding)
        .map_err(|error| format!("failed to serialize embedding output: {error}"))?;
    output
        .write_all(b"\n")
        .map_err(|error| format!("failed to finish embedding output: {error}"))?;
    output
        .flush()
        .map_err(|error| format!("failed to flush embedding output: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Command};
    use clap::{Parser, error::ErrorKind};

    #[test]
    fn parser_preserves_literal_text_and_model_directory() {
        let text = "  query: literal\n\ttext  ";
        let cli = Cli::try_parse_from([
            "lattice",
            "embed",
            "--model",
            "/models/local-bert",
            "--text",
            text,
        ])
        .expect("embed arguments should parse");
        let Command::Embed { args } = cli.command else {
            panic!("expected the embed command");
        };
        assert_eq!(args.model, PathBuf::from("/models/local-bert"));
        assert_eq!(args.text, text);
    }

    #[test]
    fn parser_preserves_empty_and_whitespace_only_text() {
        for text in ["", " \t\n "] {
            let cli = Cli::try_parse_from([
                "lattice",
                "embed",
                "--model",
                "/models/local-bert",
                "--text",
                text,
            ])
            .expect("literal empty or whitespace text should parse");
            let Command::Embed { args } = cli.command else {
                panic!("expected the embed command");
            };
            assert_eq!(args.text, text);
        }
    }

    #[test]
    fn parser_requires_model_and_text() {
        for arguments in [
            vec!["lattice", "embed"],
            vec!["lattice", "embed", "--model", "/models/local-bert"],
            vec!["lattice", "embed", "--text", "literal"],
        ] {
            let error = Cli::try_parse_from(arguments)
                .err()
                .expect("missing required arguments should fail");
            assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
        }
    }

    #[test]
    fn parser_rejects_an_unimplemented_role_flag() {
        let error = Cli::try_parse_from([
            "lattice",
            "embed",
            "--model",
            "/models/local-bert",
            "--text",
            "literal",
            "--role",
            "query",
        ])
        .err()
        .expect("unknown flags should fail");
        assert_eq!(error.kind(), ErrorKind::UnknownArgument);
    }

    #[test]
    fn output_contains_every_value_and_one_trailing_newline() {
        let embedding = vec![0.25_f32, -1.5, -0.0, 2.75, 3.0, 4.5, 5.0, 6.25, 7.0, 8.5];
        let mut output = Vec::new();
        write_embedding(vec![embedding.clone()], &mut output).expect("output should serialize");
        assert_eq!(output.last(), Some(&b'\n'));
        assert_eq!(output.iter().filter(|&&byte| byte == b'\n').count(), 1);
        let actual: Vec<f32> = serde_json::from_slice(&output).expect("one JSON vector");
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            embedding
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn output_rejects_unexpected_cardinality_before_writing() {
        for embeddings in [vec![], vec![vec![1.0], vec![2.0]]] {
            let mut output = Vec::new();
            let error = write_embedding(embeddings, &mut output).expect_err("invalid cardinality");
            assert!(error.contains("vectors for one input"));
            assert!(output.is_empty());
        }
    }

    #[test]
    fn output_rejects_non_finite_values_before_writing() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut output = Vec::new();
            let error =
                write_embedding(vec![vec![value]], &mut output).expect_err("non-finite value");
            assert!(error.contains("non-finite"));
            assert!(output.is_empty());
        }
    }

    #[derive(Clone, Copy)]
    enum FailureStage {
        Json,
        Newline,
        Flush,
    }

    struct FailingWriter(FailureStage);

    impl Write for FailingWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            if matches!(self.0, FailureStage::Json)
                || (matches!(self.0, FailureStage::Newline) && buffer == b"\n")
            {
                return Err(io::Error::other("intentional output failure"));
            }
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            if matches!(self.0, FailureStage::Flush) {
                return Err(io::Error::other("intentional output failure"));
            }
            Ok(())
        }
    }

    #[test]
    fn output_propagates_json_newline_and_flush_errors() {
        for (stage, context) in [
            (FailureStage::Json, "serialize"),
            (FailureStage::Newline, "finish"),
            (FailureStage::Flush, "flush"),
        ] {
            let error = write_embedding(vec![vec![0.25]], &mut FailingWriter(stage))
                .expect_err("output failure should propagate");
            assert!(error.contains(context));
            assert!(error.contains("intentional output failure"));
        }
    }
}
