// Native (napi-rs) Node.js bindings for `lattice-embed`'s BERT-family text
// embedding path (MiniLM, BGE, E5). Unlike the `lattice-embed-wasm` package
// (which is limited to in-memory byte buffers, see crates/embed/src/wasm.rs),
// this binding loads models directly from a local directory on disk via
// `lattice_inference::BertModel::from_directory`, the same real, production
// entry point `lattice_embed::service::native::NativeEmbeddingService` itself
// calls for BERT-family models (see that module's `load_model_sync`).
//
// v0 scope (see the design note for the full rationale): local-directory
// loading only, no remote model-id resolution/download tier (that is the
// wasm package's `resolve.mjs` job, not this one, and is explicitly
// deferred here). Pooling (CLS for BGE, mean for E5/MiniLM) is resolved
// automatically from the model family, exactly as
// `lattice_embed::EmbeddingModel::bert_pooling()` defines it, so callers
// never have to pick a pooling strategy by hand (contrast
// `crates/embed/src/wasm.rs`'s `LatticeEmbedder::use_cls_pooling()`, which
// requires the JS caller to opt in to CLS explicitly).
//
// Known v0 limitation (flagged loudly here and in the delivery report):
// `BertModel::encode`/`encode_batch` unconditionally L2-normalize their
// output (see crates/inference/src/model/bert.rs; the `pool()` doc comment
// confirms normalization always happens in the caller, i.e. `encode`/
// `encode_batch`, and there is no public non-normalizing path). There is
// therefore no way to honor `normalize: false` without adding a new public
// method to `lattice-inference` (skipping this crate's own napi surface
// entirely). Rather than silently ignoring an explicit `normalize: false`
// and returning a normalized vector anyway (a silent-partial-success), this
// binding rejects `normalize: false` with `FL_EMBED_BAD_OPTIONS` so the
// caller gets an honest, actionable error instead of a lie.

// Import, rather than redefine, the production service's own OOM
// guardrails (`crates/embed/src/service/mod.rs`) so this binding and
// `NativeEmbeddingService`/`CachedEmbeddingService` stay a single source of
// truth for what "too large" means -- both are `pub` and re-exported
// unconditionally at the `lattice_embed` crate root (not gated behind the
// `native` feature), so this crate can depend on them directly.
use lattice_embed::{DEFAULT_MAX_BATCH_SIZE, EmbeddingModel as ModelFamily, MAX_TEXT_CHARS};
use lattice_inference::BertModel;
use napi::bindgen_prelude::{AsyncTask, Float32Array, Task};
use napi::{Env, Error, Result, Status};
use napi_derive::napi;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

/// Options for `loadModel`/`loadModelSync`.
///
/// `modelPath` is required: this v0 binding only supports loading a model
/// already present as a local directory (a `model.safetensors` +
/// `config.json` + a WordPiece `vocab.txt` or `tokenizer.json`), the same
/// three-file shape `BertModel::from_directory` expects. There is no
/// network-fetch tier here (contrast the wasm package's `resolve.mjs`).
///
/// `modelId` is an optional override for the model-family lookup
/// (pooling strategy + expected dimension). If omitted, the family is
/// inferred from `modelPath`'s final path component, which works
/// automatically for any directory named after its canonical slug (e.g.
/// `all-minilm-l6-v2`, `bge-small-en-v1.5` -- exactly the directory names
/// `lattice-inference`'s own model cache convention uses, see
/// `default_cache_dir` in crates/inference/src/lib.rs).
// A wrong JS type for `modelId`/`normalize` (e.g. a number or a stringified
// boolean) never reaches `from_options` below: napi-rs converts the JS
// object into this struct's typed fields as part of argument marshalling,
// before any function body runs, and rejects a type mismatch with its own
// napi status ("StringExpected", "BooleanExpected") rather than our stable
// `FL_EMBED_*` codes. There is no Rust-side hook earlier than this to
// intercept that conversion, so the authoritative guard for malformed
// optional-field *types* lives in JS (`normalizeOptions()` in index.js),
// which validates before ever calling into native code. This struct (and
// `from_options`) still validates *values* that survive conversion with the
// right type but a bad value (e.g. `normalize: false`, an empty `modelPath`).
#[derive(Clone)]
#[napi(object)]
#[allow(non_snake_case)]
pub struct LoadOptions {
  pub modelPath: String,
  pub modelId: Option<String>,
  pub normalize: Option<bool>,
}

#[napi(object)]
pub struct EmbeddingBatch {
  pub data: Float32Array,
  pub rows: u32,
  pub dimensions: u32,
  pub normalized: bool,
}

#[napi]
pub struct EmbeddingModel {
  model: Arc<BertModel>,
  dimension: u32,
}

#[napi]
impl EmbeddingModel {
  #[napi(constructor)]
  pub fn new(options: LoadOptions) -> Result<Self> {
    Self::from_options(options)
  }

  #[napi(getter)]
  pub fn dimension(&self) -> u32 {
    self.dimension
  }

  /// Always `true` in this v0 binding; see the module doc comment's
  /// "Known v0 limitation" section for why `normalize: false` is rejected
  /// at load time rather than silently ignored here.
  #[napi(getter)]
  pub fn normalized(&self) -> bool {
    true
  }

  #[napi]
  pub fn embed_sync(&self, text: String) -> Result<Float32Array> {
    validate_text(&text, 0)?;
    let output = encode_one(&self.model, &text)?;
    Ok(output.into())
  }

  #[napi]
  pub fn embed(&self, text: String) -> Result<AsyncTask<EmbedTask>> {
    validate_text(&text, 0)?;
    Ok(AsyncTask::new(EmbedTask {
      model: Arc::clone(&self.model),
      text,
    }))
  }

  #[napi]
  pub fn embed_batch_sync(&self, texts: Vec<String>) -> Result<EmbeddingBatch> {
    validate_texts(&texts)?;
    let output = encode_many(&self.model, &texts, self.dimension)?;
    Ok(output.into_napi())
  }

  #[napi]
  pub fn embed_batch(&self, texts: Vec<String>) -> Result<AsyncTask<BatchTask>> {
    validate_texts(&texts)?;
    Ok(AsyncTask::new(BatchTask {
      model: Arc::clone(&self.model),
      texts,
      dimension: self.dimension,
    }))
  }
}

impl EmbeddingModel {
  fn from_options(options: LoadOptions) -> Result<Self> {
    if options.modelPath.trim().is_empty() {
      return Err(invalid_arg("FL_EMBED_BAD_OPTIONS", "modelPath must not be empty"));
    }

    // `normalize: false` cannot be honored honestly in this v0 binding --
    // see the module doc comment. Any other value (omitted, or explicitly
    // `true`) proceeds; the engine always L2-normalizes regardless.
    if options.normalize == Some(false) {
      return Err(invalid_arg(
        "FL_EMBED_BAD_OPTIONS",
        "normalize: false is not supported in this v0 binding -- \
         BertModel::encode/encode_batch always L2-normalize their output \
         and there is no public non-normalizing path in lattice-inference \
         yet; omit `normalize` or set it to true",
      ));
    }

    let model_path = Path::new(&options.modelPath);
    if !model_path.is_dir() {
      return Err(invalid_arg(
        "FL_EMBED_BAD_MODEL",
        format!(
          "modelPath does not exist or is not a directory: {}",
          options.modelPath
        ),
      ));
    }

    // Family (-> pooling strategy) is resolved from an explicit `modelId`
    // override if given, else from modelPath's final path component. Both
    // go through the same `lattice_embed::EmbeddingModel::from_str`, the
    // real production parser (case-insensitive, accepts display names,
    // short names, and HuggingFace ids -- see crates/embed/src/model.rs).
    let family_hint = options
      .modelId
      .as_deref()
      .filter(|s| !s.trim().is_empty())
      .map(|s| s.to_string())
      .or_else(|| {
        model_path
          .file_name()
          .and_then(|name| name.to_str())
          .map(|s| s.to_string())
      })
      .ok_or_else(|| {
        invalid_arg(
          "FL_EMBED_BAD_MODEL",
          "could not determine a model identifier from modelId or modelPath's \
           final path component",
        )
      })?;

    let family = ModelFamily::from_str(&family_hint).map_err(|err| {
      invalid_arg(
        "FL_EMBED_BAD_MODEL",
        format!("unrecognized model identifier \"{family_hint}\": {err}"),
      )
    })?;

    let pooling = family.bert_pooling().ok_or_else(|| {
      invalid_arg(
        "FL_EMBED_BAD_MODEL",
        format!(
          "model family \"{family}\" is not a BERT-family encoder model; \
           this v0 native binding only supports BGE/E5/MiniLM-family models \
           (Qwen and remote-API models are out of scope)"
        ),
      )
    })?;

    let mut model = BertModel::from_directory(model_path).map_err(|err| {
      invalid_arg(
        "FL_EMBED_BAD_MODEL",
        format!("failed to load model from {}: {err}", options.modelPath),
      )
    })?;
    model.set_pooling(pooling);

    let dimension = u32::try_from(model.dimensions()).map_err(|_| {
      invalid_arg(
        "FL_EMBED_BAD_MODEL",
        "model dimension does not fit in a u32",
      )
    })?;

    Ok(Self {
      model: Arc::new(model),
      dimension,
    })
  }
}

pub struct LoadModelTask {
  options: LoadOptions,
}

#[napi]
impl Task for LoadModelTask {
  type Output = EmbeddingModel;
  type JsValue = EmbeddingModel;

  fn compute(&mut self) -> Result<Self::Output> {
    EmbeddingModel::from_options(self.options.clone())
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

#[napi]
pub fn load_model_sync(options: LoadOptions) -> Result<EmbeddingModel> {
  EmbeddingModel::from_options(options)
}

#[napi]
pub fn load_model(options: LoadOptions) -> Result<AsyncTask<LoadModelTask>> {
  Ok(AsyncTask::new(LoadModelTask { options }))
}

pub struct EmbedTask {
  model: Arc<BertModel>,
  text: String,
}

#[napi]
impl Task for EmbedTask {
  type Output = Vec<f32>;
  type JsValue = Float32Array;

  fn compute(&mut self) -> Result<Self::Output> {
    encode_one(&self.model, &self.text)
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output.into())
  }
}

pub struct BatchTask {
  model: Arc<BertModel>,
  texts: Vec<String>,
  dimension: u32,
}

#[napi]
impl Task for BatchTask {
  type Output = BatchOutput;
  type JsValue = EmbeddingBatch;

  fn compute(&mut self) -> Result<Self::Output> {
    encode_many(&self.model, &self.texts, self.dimension)
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output.into_napi())
  }
}

pub struct BatchOutput {
  data: Vec<f32>,
  rows: u32,
  dimensions: u32,
  normalized: bool,
}

impl BatchOutput {
  fn into_napi(self) -> EmbeddingBatch {
    EmbeddingBatch {
      data: self.data.into(),
      rows: self.rows,
      dimensions: self.dimensions,
      normalized: self.normalized,
    }
  }
}

/// Encodes a single text through the real engine. `BertModel::encode` is
/// `&self` and allocates its own per-call scratch (`AttentionBuffers`, see
/// crates/inference/src/model/bert.rs) rather than caching anything on
/// `self`, so concurrent calls through the same `Arc<BertModel>` (e.g. many
/// in-flight `embed()` AsyncTasks on napi's worker pool) never race.
fn encode_one(model: &BertModel, text: &str) -> Result<Vec<f32>> {
  model
    .encode(text)
    .map_err(|err| invalid_arg("FL_EMBED_BAD_MODEL", format!("encode failed: {err}")))
}

/// Encodes a batch through the real engine's fused batched forward path
/// (`BertModel::encode_batch`). `dimension` is the model's fixed output
/// width, cached at load time (`BertModel::dimensions()`); it is used only
/// for the reported `EmbeddingBatch.dimensions` metadata field, not to
/// reshape the actual data, so a caller inspecting `data.length ===
/// rows * dimensions` is checking real agreement between the engine's
/// actual per-row output length and this wrapper's own bookkeeping, not a
/// value this function assumes.
fn encode_many(model: &BertModel, texts: &[String], dimension: u32) -> Result<BatchOutput> {
  let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
  let rows_out = model
    .encode_batch(&text_refs)
    .map_err(|err| invalid_arg("FL_EMBED_BAD_MODEL", format!("encode_batch failed: {err}")))?;

  let rows = u32::try_from(rows_out.len()).map_err(|_| {
    invalid_arg(
      "FL_EMBED_BAD_BATCH",
      "batch has more rows than can be represented as u32",
    )
  })?;

  let mut data = Vec::with_capacity(rows_out.len() * dimension as usize);
  for row in &rows_out {
    data.extend_from_slice(row);
  }

  Ok(BatchOutput {
    data,
    rows,
    dimensions: dimension,
    normalized: true,
  })
}

fn validate_text(text: &str, index: usize) -> Result<()> {
  if text.is_empty() {
    return Err(invalid_arg(
      "FL_EMBED_EMPTY_INPUT",
      format!("text at index {index} must not be empty"),
    ));
  }

  // `MAX_TEXT_CHARS`'s own doc comment says "characters", but the
  // production enforcement point this binding mirrors
  // (`crates/embed/src/service/native.rs`, `if text.len() > MAX_TEXT_CHARS`)
  // actually measures `str::len()`, i.e. UTF-8 BYTE length, not
  // `chars().count()`. Match that exactly -- not the doc comment -- so a
  // multi-byte string near the boundary is accepted/rejected identically
  // by this binding and by `NativeEmbeddingService`/`CachedEmbeddingService`.
  if text.len() > MAX_TEXT_CHARS {
    return Err(invalid_arg(
      "FL_EMBED_INPUT_TOO_LARGE",
      format!(
        "text at index {index} is {} bytes, exceeding the maximum of {MAX_TEXT_CHARS} bytes",
        text.len()
      ),
    ));
  }

  Ok(())
}

fn validate_texts(texts: &[String]) -> Result<()> {
  if texts.is_empty() {
    return Err(invalid_arg(
      "FL_EMBED_BAD_BATCH",
      "texts must contain at least one item",
    ));
  }

  // Checked before per-item validation below so an oversized batch fails
  // fast on item count alone, without walking every item first.
  if texts.len() > DEFAULT_MAX_BATCH_SIZE {
    return Err(invalid_arg(
      "FL_EMBED_BAD_BATCH",
      format!(
        "batch has {} items, exceeding the maximum of {DEFAULT_MAX_BATCH_SIZE}",
        texts.len()
      ),
    ));
  }

  for (index, text) in texts.iter().enumerate() {
    validate_text(text, index)?;
  }

  Ok(())
}

fn invalid_arg(code: &str, message: impl AsRef<str>) -> Error {
  Error::new(Status::InvalidArg, format!("{code}: {}", message.as_ref()))
}
