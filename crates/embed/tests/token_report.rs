//! Token-limit reports from `NativeEmbeddingService`, against real checkpoints.
//!
//! `count_tokens` and `embed_with_report` must describe the sequence the model embeds:
//! exact at, one under and one over the length limit for each model family, and with a
//! role instruction counted. Inputs are built by token count, not by bytes, and the
//! expected lengths come from a tokenizer loaded here and from the fixed number of special
//! tokens each family wraps around a sequence, not from the service under test.
//!
//! Each test needs checkpoint files under `~/.lattice/models/` (or the service's own
//! override variables). A missing checkpoint records `LATTICE_TOKEN_REPORT_SKIPPED` on
//! standard error and returns, so the run reports which families were never exercised.
//! Set `LATTICE_TOKEN_REPORT_ENFORCE` to turn a missing checkpoint into a failure.
//!
//! Run:
//!   cargo test -p lattice-embed --test token_report -- --nocapture

#![cfg(feature = "native")]

use lattice_embed::{EmbeddingModel, EmbeddingRole, EmbeddingService, NativeEmbeddingService};
use lattice_inference::model::embeddinggemma2::EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS;
use lattice_inference::{GemmaBpeTokenizer, Tokenizer, load_tokenizer};
use std::path::PathBuf;

const QWEN_EOS_TOKEN_ID: u32 = 151_643;

/// A word every tokenizer here keeps as exactly one token, in or out of a sentence.
const WORD: &str = "a";

fn text_of_words(count: usize) -> String {
    vec![WORD; count].join(" ")
}

fn skip(test: &str, what: &str) {
    if std::env::var("LATTICE_TOKEN_REPORT_ENFORCE").is_ok() {
        panic!("{test}: checkpoint missing despite LATTICE_TOKEN_REPORT_ENFORCE: {what}");
    }
    eprintln!("LATTICE_TOKEN_REPORT_SKIPPED test={test} reason=missing_weights what={what}");
}

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".lattice")
        .join("models")
}

/// The checkpoint directory the service would load, when its weights are present.
fn checkpoint(env_override: Option<&str>, slug: &str) -> Option<PathBuf> {
    let dir = match env_override.and_then(|name| std::env::var(name).ok()) {
        Some(dir) => PathBuf::from(dir),
        None => models_dir().join(slug),
    };
    let has_weights =
        dir.join("model.safetensors").exists() || dir.join("model.safetensors.index.json").exists();
    has_weights.then_some(dir)
}

/// Model-visible length of a text, from a tokenizer of the test's own.
struct Reference {
    /// Tokens the family wraps around every sequence.
    overhead: usize,
    /// The model's sequence limit.
    limit: usize,
    length: Box<dyn Fn(&str) -> usize>,
}

impl Reference {
    /// Checks the length function against the family's fixed overhead before it is used
    /// to build boundary inputs, so a tokenizer that splits `WORD` differently fails here.
    fn calibrated(self) -> Self {
        for words in [1usize, 2, 7, 50] {
            assert_eq!(
                (self.length)(&text_of_words(words)),
                words + self.overhead,
                "{words} words must be one token each plus {} wrapping tokens",
                self.overhead
            );
        }
        self
    }
}

fn bert_reference(dir: &std::path::Path) -> Reference {
    let tokenizer = load_tokenizer(dir).expect("tokenizer loads");
    Reference {
        overhead: 2,
        limit: tokenizer.max_seq_len(),
        length: Box::new(move |text| tokenizer.tokenize(text).pre_truncation_len),
    }
    .calibrated()
}

fn qwen_reference(dir: &std::path::Path) -> Reference {
    let tokenizer = load_tokenizer(dir).expect("tokenizer loads");
    Reference {
        overhead: 1,
        limit: tokenizer.max_seq_len(),
        length: Box::new(move |text| {
            let tokens = tokenizer.tokenize(text);
            let ids = &tokens.input_ids[..tokens.real_length];
            tokens.pre_truncation_len + usize::from(ids.last() != Some(&QWEN_EOS_TOKEN_ID))
        }),
    }
    .calibrated()
}

fn gemma_reference(dir: &std::path::Path) -> Reference {
    let json = std::fs::read_to_string(dir.join("tokenizer.json")).expect("tokenizer.json reads");
    let tokenizer = GemmaBpeTokenizer::from_tokenizer_json_str(&json)
        .expect("tokenizer loads")
        .with_max_seq_len(usize::MAX);
    Reference {
        overhead: 2,
        limit: EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS,
        length: Box::new(move |text| {
            // Batch tokenization pads to the batch's own length; a single `tokenize` would
            // pad to the unbounded sequence limit set above.
            let tokens = tokenizer.tokenize_batch(&[text]);
            tokens.first().map_or(0, |t| t.pre_truncation_len) + 2
        }),
    }
    .calibrated()
}

/// limit - 1, limit and limit + 1 tokens, built by token count.
async fn assert_exact_at_the_limit(
    service: &NativeEmbeddingService,
    model: EmbeddingModel,
    reference: &Reference,
) {
    for model_visible in [reference.limit - 1, reference.limit, reference.limit + 1] {
        let text = text_of_words(model_visible - reference.overhead);
        assert_eq!(
            (reference.length)(&text),
            model_visible,
            "the input must be {model_visible} model-visible tokens"
        );
        let counts = service
            .count_tokens(std::slice::from_ref(&text), model, EmbeddingRole::Generic)
            .await
            .expect("count_tokens succeeds");
        assert_eq!(counts.len(), 1);
        let expected_embedded = model_visible.min(reference.limit);
        assert_eq!(
            counts[0].before_truncation, model_visible,
            "{model:?} before_truncation at {model_visible}"
        );
        assert_eq!(
            counts[0].embedded, expected_embedded,
            "{model:?} embedded at {model_visible}"
        );
        assert_eq!(
            counts[0].truncated(),
            model_visible > reference.limit,
            "{model:?} truncated() at {model_visible}"
        );
        println!(
            "[{model:?}] limit={} visible={model_visible} before={} embedded={} truncated={}",
            reference.limit,
            counts[0].before_truncation,
            counts[0].embedded,
            counts[0].truncated()
        );
    }
}

/// A query-role count exceeds the generic one by exactly the instruction's token length.
async fn assert_role_instruction_is_counted(
    service: &NativeEmbeddingService,
    model: EmbeddingModel,
    reference: &Reference,
) {
    let texts = vec![text_of_words(5)];
    let generic = service
        .count_tokens(&texts, model, EmbeddingRole::Generic)
        .await
        .expect("generic count");
    let plain = service
        .count_tokens(&texts, model, EmbeddingRole::Passage)
        .await
        .expect("passage count");
    let query = service
        .count_tokens(&texts, model, EmbeddingRole::Query)
        .await
        .expect("query count");
    let instruction = model
        .query_instruction()
        .expect("this test needs a query instruction");
    let instruction_tokens = (reference.length)(instruction.trim_end()) - reference.overhead;
    assert!(instruction_tokens > 0);
    assert_eq!(
        query[0].before_truncation - generic[0].before_truncation,
        instruction_tokens,
        "{model:?} query count must exceed the generic count by the instruction length"
    );
    assert_eq!(
        query[0].embedded - generic[0].embedded,
        instruction_tokens,
        "{model:?} embedded count must include the instruction"
    );
    let passage_instruction_tokens = model
        .document_instruction()
        .map_or(0, |i| (reference.length)(i.trim_end()) - reference.overhead);
    assert_eq!(
        plain[0].before_truncation - generic[0].before_truncation,
        passage_instruction_tokens,
        "{model:?} passage count must exceed the generic count by its instruction length"
    );
    println!(
        "[{model:?}] generic={} query={} passage={} instruction_tokens={instruction_tokens}",
        generic[0].before_truncation, query[0].before_truncation, plain[0].before_truncation
    );
}

fn bits(vectors: &[Vec<f32>]) -> Vec<Vec<u32>> {
    vectors
        .iter()
        .map(|v| v.iter().map(|x| x.to_bits()).collect())
        .collect()
}

/// `embed_with_report` returns the vectors `embed_with_role` returns, bit for bit, and
/// reports the counts `count_tokens` reports.
async fn assert_report_matches_role_embedding(
    service: &NativeEmbeddingService,
    model: EmbeddingModel,
    role: EmbeddingRole,
) {
    let texts: Vec<String> = [
        "a",
        "a a a a a a",
        "the quick brown fox jumps over the lazy dog",
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    let report = service
        .embed_with_report(&texts, model, role)
        .await
        .expect("embed_with_report succeeds");
    let plain = service
        .embed_with_role(&texts, model, role)
        .await
        .expect("embed_with_role succeeds");
    let counts = service
        .count_tokens(&texts, model, role)
        .await
        .expect("count_tokens succeeds");
    assert_eq!(report.embeddings.len(), texts.len());
    assert_eq!(report.token_counts.len(), texts.len());
    assert_eq!(
        bits(&report.embeddings),
        bits(&plain),
        "{model:?} {role:?} vectors"
    );
    assert_eq!(report.token_counts, counts, "{model:?} {role:?} counts");
    println!("[{model:?}] {role:?} report == role embedding, bit-identical");
}

const BERT_FAMILY: [(EmbeddingModel, &str); 4] = [
    (EmbeddingModel::BgeSmallEnV15, "bge-small-en-v1.5"),
    (EmbeddingModel::MultilingualE5Small, "multilingual-e5-small"),
    (EmbeddingModel::AllMiniLmL6V2, "all-minilm-l6-v2"),
    (
        EmbeddingModel::ParaphraseMultilingualMiniLmL12V2,
        "paraphrase-multilingual-minilm-l12-v2",
    ),
];

#[tokio::test]
async fn bert_family_counts_are_exact_at_the_limit() {
    let mut ran = 0;
    for (model, slug) in BERT_FAMILY {
        let Some(dir) = checkpoint(None, slug) else {
            skip("bert_family_counts_are_exact_at_the_limit", slug);
            continue;
        };
        ran += 1;
        let service = NativeEmbeddingService::with_model(model);
        let reference = bert_reference(&dir);
        assert_exact_at_the_limit(&service, model, &reference).await;
    }
    println!(
        "bert family checkpoints exercised: {ran}/{}",
        BERT_FAMILY.len()
    );
}

#[tokio::test]
async fn bert_family_role_instruction_is_counted() {
    for (model, slug) in BERT_FAMILY {
        if model.query_instruction().is_none() {
            continue;
        }
        let Some(dir) = checkpoint(None, slug) else {
            skip("bert_family_role_instruction_is_counted", slug);
            continue;
        };
        let service = NativeEmbeddingService::with_model(model);
        assert_role_instruction_is_counted(&service, model, &bert_reference(&dir)).await;
    }
}

#[tokio::test]
async fn bert_family_report_matches_role_embedding() {
    for (model, slug) in BERT_FAMILY {
        if checkpoint(None, slug).is_none() {
            skip("bert_family_report_matches_role_embedding", slug);
            continue;
        }
        let service = NativeEmbeddingService::with_model(model);
        for role in [EmbeddingRole::Generic, EmbeddingRole::Query] {
            assert_report_matches_role_embedding(&service, model, role).await;
        }
    }
}

#[tokio::test]
async fn qwen3_counts_are_exact_at_the_limit() {
    let Some(dir) = checkpoint(Some("LATTICE_QWEN_MODEL_DIR"), "qwen3-embedding-0.6b") else {
        skip(
            "qwen3_counts_are_exact_at_the_limit",
            "qwen3-embedding-0.6b",
        );
        return;
    };
    let model = EmbeddingModel::Qwen3Embedding0_6B;
    let service = NativeEmbeddingService::with_model(model);
    let reference = qwen_reference(&dir);
    assert_exact_at_the_limit(&service, model, &reference).await;
    assert_role_instruction_is_counted(&service, model, &reference).await;
}

#[tokio::test]
async fn qwen3_report_matches_role_embedding() {
    if checkpoint(Some("LATTICE_QWEN_MODEL_DIR"), "qwen3-embedding-0.6b").is_none() {
        skip(
            "qwen3_report_matches_role_embedding",
            "qwen3-embedding-0.6b",
        );
        return;
    }
    let model = EmbeddingModel::Qwen3Embedding0_6B;
    let service = NativeEmbeddingService::with_model(model);
    for role in [EmbeddingRole::Generic, EmbeddingRole::Query] {
        assert_report_matches_role_embedding(&service, model, role).await;
    }
}

#[tokio::test]
async fn embeddinggemma2_counts_are_exact_at_the_limit() {
    let Some(dir) = checkpoint(
        Some("LATTICE_EMBEDDINGGEMMA2_MODEL_DIR"),
        "embeddinggemma-2",
    ) else {
        skip(
            "embeddinggemma2_counts_are_exact_at_the_limit",
            "embeddinggemma-2",
        );
        return;
    };
    let model = EmbeddingModel::EmbeddingGemma2;
    let service = NativeEmbeddingService::with_model(model);
    let reference = gemma_reference(&dir);
    assert_exact_at_the_limit(&service, model, &reference).await;
    assert_role_instruction_is_counted(&service, model, &reference).await;
}

#[tokio::test]
async fn embeddinggemma2_report_matches_role_embedding() {
    if checkpoint(
        Some("LATTICE_EMBEDDINGGEMMA2_MODEL_DIR"),
        "embeddinggemma-2",
    )
    .is_none()
    {
        skip(
            "embeddinggemma2_report_matches_role_embedding",
            "embeddinggemma-2",
        );
        return;
    }
    let model = EmbeddingModel::EmbeddingGemma2;
    let service = NativeEmbeddingService::with_model(model);
    for role in [EmbeddingRole::Generic, EmbeddingRole::Query] {
        assert_report_matches_role_embedding(&service, model, role).await;
    }
}
