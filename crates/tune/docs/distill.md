# Distillation

The `distill` module is the boundary between conversational source material and
the embedding-based `TrainingExample` values used by the rest of
`lattice-tune`. It has three responsibilities:

1. describe a teacher and the endpoint policy that constrains it;
2. turn a `RawExample` into a bounded, sanitized prompt and record a labeling
   result; and
3. join successful labels with embeddings supplied by the caller.

The module intentionally does not create embeddings. It also does **not**
currently issue HTTP requests: `DistillationPipeline::label_single` formats a
prompt but then returns a simulated label distribution. The configuration and
result types define the intended integration boundary; callers must not treat
the current pipeline as a live teacher client.

For the provider-selection decision and its alternatives, see
[ADR-001: Multi-Provider Teacher Strategy](ADR-001-teacher-providers.md). The
retired crate-local distillation ADR points to its maintained counterpart in
the repository-wide ADR series.

## End-to-end flow

```text
Raw text
  │
  ▼
RawExample ──sanitize + format──► teacher prompt
  │                                  │
  │                                  └── current implementation: simulated result
  ▼
LabelingResult ──successful only──► caller-provided embeddings
                                         │
                                         ▼
                                  TrainingExample
                                         │
                                         ▼
                                      Dataset / train
```

A `RawExample` has an ID, chronological context strings, one current message,
and optional metadata. A labeling result preserves the raw ID, so the caller
can align label results with its independently generated context and message
embeddings. `to_training_examples` constructs the training records only after
that alignment has been supplied.

## Teacher configuration

`TeacherConfig` makes the teacher selection, request policy, prompt, and
endpoint controls explicit. It derives serialization only with the `serde`
feature.

| Field                            | Meaning                                                    | Validation or behavior                                                                              |
| -------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `provider`                       | `Claude`, `OpenAI`, `Gemini`, `Local`, or `Custom(String)` | Controls default endpoint and display name.                                                         |
| `model_id`                       | Provider-specific model identifier                         | Must not be empty.                                                                                  |
| `endpoint`                       | Optional custom base endpoint                              | When absent, `get_endpoint` chooses a provider default. A custom endpoint is checked by `validate`. |
| `api_key_env`                    | Environment-variable name for credentials                  | Configuration only; the current pipeline never reads it.                                            |
| `temperature`                    | Generation temperature                                     | Must lie in `0.0..=2.0`.                                                                            |
| `max_tokens`                     | Response-token budget                                      | Must be nonzero.                                                                                    |
| `timeout_ms`                     | Per-request timeout                                        | Must be nonzero.                                                                                    |
| `max_retries` / `retry_delay_ms` | Retry policy                                               | Stored for a future client; currently not executed.                                                 |
| `system_prompt`                  | Optional classification instruction                        | Presets use the built-in intent-classification prompt; the builder can replace or clear it.         |
| `security`                       | `EndpointSecurity` policy                                  | Validates a custom endpoint during `validate`.                                                      |

### Provider defaults and presets

The default configuration is `TeacherConfig::claude_sonnet()`. The convenience
constructors select the following values:

| Constructor              | Provider | Model ID                    | Endpoint        | Key variable        | Timeout | Retries / delay |
| ------------------------ | -------- | --------------------------- | --------------- | ------------------- | ------: | --------------- |
| `claude_sonnet()`        | Claude   | `claude-sonnet-4-20250514`  | Claude default  | `ANTHROPIC_API_KEY` |    30 s | 3 / 1 s         |
| `claude_haiku()`         | Claude   | `claude-3-5-haiku-20241022` | Claude default  | `ANTHROPIC_API_KEY` |    15 s | 3 / 500 ms      |
| `gpt4()`                 | OpenAI   | `gpt-4-turbo-preview`       | OpenAI default  | `OPENAI_API_KEY`    |    30 s | 3 / 1 s         |
| `gemini_pro()`           | Gemini   | `gemini-pro`                | Gemini default  | `GOOGLE_API_KEY`    |    30 s | 3 / 1 s         |
| `local(model, endpoint)` | Local    | Caller supplied             | Caller supplied | empty               |    60 s | 2 / 500 ms      |

All presets use temperature `0.3`, a 1,024-token response budget, the default
system prompt, and a security preset appropriate to a remote or local
endpoint. The endpoint defaults returned by `get_endpoint` are:

| Provider | Default endpoint                               |
| -------- | ---------------------------------------------- |
| Claude   | `https://api.anthropic.com/v1`                 |
| OpenAI   | `https://api.openai.com/v1`                    |
| Gemini   | `https://generativelanguage.googleapis.com/v1` |
| Local    | `http://localhost:11434`                       |
| Custom   | empty string                                   |

`TeacherProvider` displays as `claude`, `openai`, `gemini`, `local`, or
`custom:<name>`. `TeacherConfig::display_name` combines that display form with
the model ID as `provider:model_id`; the pipeline saves this value in example
metadata.

### The builder

`TeacherConfig::builder()` starts from the Sonnet preset. Its setters replace
individual fields and `build()` returns the resulting configuration without
validating it. Validate before creating a connection or pass the value to
`DistillationPipeline::new`, which validates it itself.

```rust,ignore
use lattice_tune::distill::{
    DistillationConfig, DistillationPipeline, EndpointSecurity, TeacherConfig,
    TeacherProvider,
};

let teacher = TeacherConfig::builder()
    .provider(TeacherProvider::Local)
    .model_id("my-local-teacher")
    .endpoint("http://localhost:11434")
    .security(EndpointSecurity::for_local())
    .max_tokens(1_024)
    .build();

teacher.validate()?;
let pipeline = DistillationPipeline::new(teacher, DistillationConfig::default())?;
```

The default system prompt asks for a JSON object with six intent scores:
`continuation`, `topic_shift`, `explicit_query`, `person_lookup`,
`health_check`, and `task_status`. It asks scores to sum approximately to one
and asks the teacher to return JSON only. There is no response parser yet, so
this is a contract for a future teacher-client implementation rather than a
live parsing guarantee.

## Endpoint security policy

`EndpointSecurity` expresses endpoint policy independently of a concrete HTTP
client:

| Control                     | Purpose                                  | What the current code checks                                                                   |
| --------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `require_tls`               | Disallow non-HTTPS remote endpoints      | `verify_endpoint` rejects endpoints whose lowercased string does not start with `https://`.    |
| `allowed_domains`           | Host allowlist                           | The extracted host must exactly equal one configured domain.                                   |
| `expected_cert_fingerprint` | Expected SHA-256 certificate fingerprint | `validate_cert_fingerprint` checks only that the configured value has 64 ASCII-hex characters. |
| `model_checksum`            | Expected checksum for local weights      | `verify_model_checksum` compares an actual checksum provided by its caller.                    |

`EndpointSecurity::default_secure()` requires TLS and allows only
`api.anthropic.com`, `api.openai.com`, and
`generativelanguage.googleapis.com`. `EndpointSecurity::for_local()` does not
require TLS and allows only `localhost` and `127.0.0.1`. `allow_domain` appends
to the allowlist, while `with_cert_fingerprint` and `with_model_checksum` set
the corresponding optional expectations.

### What verification does and does not do

`TeacherConfig::validate` checks model ID, temperature, response budget, and
timeout. It applies `security.verify_endpoint` only when an explicit custom
endpoint is present. `TeacherConfig::verify_endpoint` resolves either a custom
or provider-default endpoint, applies the endpoint policy, and validates the
_format_ of a configured certificate fingerprint.

Neither method opens a network connection. In particular, certificate
fingerprint comparison must happen in the future HTTP/TLS client after it has
the peer certificate; the empty value passed by `verify_endpoint` is not an
actual certificate fingerprint. Similarly, local weight checking occurs only
when an integration calls `verify_model_checksum` with a computed value.
Creating a `DistillationPipeline` calls `validate`, not
`verify_endpoint`.

The current host extraction is deliberately simple: it removes a lowercase
`http://` or `https://` prefix, takes the text before the next slash, then
drops a port. It is policy input validation, not a general URL parser. Supply
a lowercase scheme and canonical lowercase endpoint host when using an
allowlist.

## Raw examples and prompt bounds

`RawExample` is the pre-embedding input:

```rust
pub struct RawExample {
    pub id: Uuid,
    pub context: Vec<String>,      // oldest to newest
    pub message: String,           // item to classify
    pub metadata: Option<ExampleMetadata>,
}
```

`new` assigns a fresh UUID; `with_id` preserves a caller-controlled one; and
`with_metadata` attaches optional source information. The pipeline uses the ID
for result correlation but does not currently copy `RawExample::metadata` into
the resulting training example.

### Prompt layout

For nonempty context, `to_prompt` emits:

```text
Context (previous messages):
1. <oldest sanitized message>
2. <next sanitized message>

Current message to classify:
<sanitized current message>
```

The formatter applies the same sanitization to each context item and to the
current message:

1. It limits an individual input to `MAX_MESSAGE_LENGTH` (10,000 bytes),
   stopping at a UTF-8 character boundary.
2. It removes control characters except newline, tab, and carriage return.
3. After assembling the labeled prompt, it truncates the content to
   `MAX_PROMPT_LENGTH` (50,000 bytes) and appends `\n[truncated]` when the
   limit was exceeded. The marker is appended after truncation, so the final
   output includes the marker in addition to the capped content.

The limits prevent unbounded prompt construction before an eventual network
request. They do not interpret user text or provide a semantic defense against
prompt injection; the context and current message remain teacher-visible
content. Treat the resulting text as untrusted data when implementing a
provider client.

## Pipeline configuration

`DistillationConfig` controls the labeling policy. Its default is a batch size
of 10, concurrency of 5, softmax normalization enabled, no confidence
threshold, no intermediate persistence, and a progress interval of 100.

| Field               | Default | Current use                                                                            |
| ------------------- | ------: | -------------------------------------------------------------------------------------- |
| `batch_size`        |      10 | Validated nonzero; not used to schedule the synchronous loop.                          |
| `concurrency`       |       5 | Validated nonzero; not used to run parallel requests yet.                              |
| `normalize_labels`  |    true | Applies `IntentLabels::softmax_normalize` to the simulated scores.                     |
| `min_confidence`    |    none | Rejects a result whose confidence is below the threshold.                              |
| `save_intermediate` |   false | Stored only.                                                                           |
| `output_dir`        |    none | `output_dir(...)` sets this and enables `save_intermediate`; no files are written yet. |
| `progress_interval` |     100 | Stored only.                                                                           |

`fast()` selects a 20-item batch, concurrency 10, and progress every 50
examples. `quality()` selects a 5-item batch, concurrency 3, a `0.5` minimum
confidence, intermediate persistence enabled, and progress every 20 examples.
Both keep normalization enabled. Validation requires nonzero batch size and
concurrency; when present, the confidence threshold must lie in `0.0..=1.0`.

## Labeling behavior and accounting

### A single item

`DistillationPipeline::new` validates both the teacher and pipeline
configuration, then starts with empty `DistillationStats`. `with_teacher` uses
the default pipeline configuration. `label_single`:

1. starts a latency timer and formats the raw prompt;
2. currently creates fixed scores
   `[0.4, 0.1, 0.3, 0.1, 0.05, 0.05]` instead of invoking a provider;
3. applies softmax if requested;
4. assigns a simulated confidence of `0.85`;
5. rejects scores below `min_confidence`, when configured; otherwise records
   a successful `LabelingResult` and updates statistics.

Therefore, a threshold above `0.85` causes the current placeholder to return
`TuneError::Validation`. On that direct error path the pipeline increments
`stats.skipped` but does not increment `total_processed`; `label_batch` turns
the error into a failed result and supplies that second accounting update.

### A batch

`label_batch` processes inputs synchronously in input order. It returns one
`LabelingResult` per raw input, including failures. An error from
`label_single` becomes `LabelingResult::failure(raw.id, error, 0)` and is
included in the statistics. Failed results carry default all-zero labels,
zero confidence, no raw response, and the error string. The optional raw
response is currently populated only if code constructs a result with
`with_raw_response`.

### Statistics

`DistillationStats::update` is the one normal result-accounting path:

- It increments `total_processed` and adds latency for every result.
- Success increments `successful` and updates average confidence using only
  successful items.
- Failure increments `failed`.
- Average latency is total latency divided by every processed item, including
  failures.
- On the first success, `label_distribution` becomes six zero counters. Each
  successful result increments the counter for its dominant intent; it is not
  a sum of soft probabilities.

`success_rate()` is `successful / total_processed` and returns `0.0` for an
empty statistics value. `reset_stats()` discards all accumulated counts and
averages. `skipped` is a separate counter for confidence-threshold rejects;
when a reject flows through `label_batch` it is also represented as a failure.

## Converting labels to training data

`to_training_examples` accepts parallel slices of results, context embedding
matrices, and current-message embeddings. All three slices must have the same
length. The method checks results against each embedding slice, but the
dimension-mismatch error reports the context slice's length even when the
message slice is the mismatched input.

For each successful result it constructs:

```text
TrainingExample {
  id: result.example_id,
  context_embeddings: caller-provided context_embeddings[i],
  message_embedding: caller-provided message_embeddings[i],
  labels: result.labels,
  metadata: {
    source_id: result.example_id as text,
    teacher_model: teacher.display_name(),
    labeled_at: current UTC time,
    teacher_confidence: result.confidence,
  },
}
```

Failures are skipped, so output length can be smaller than input length. The
method does not validate embedding shapes or label ranges; validate the
resulting `TrainingExample` values before handing them to a training consumer
that requires those guarantees. See [data.md](data.md) for that format and
validation contract.

## Implementing a live teacher client

A provider implementation should preserve the public contracts above while
filling in the missing network step:

1. validate the configuration and resolve the effective endpoint;
2. enforce endpoint policy before connection, then perform real TLS
   verification and any configured certificate pin comparison in the client;
3. obtain credentials from the configured environment variable without
   including their value in results, errors, or logs;
4. send the system prompt and bounded `RawExample::to_prompt` output;
5. apply timeout and retry settings with provider-specific error handling;
6. parse the teacher response into the fixed `IntentLabels` order, reject
   malformed or non-finite scores, and preserve an optional raw response only
   when its retention is appropriate; and
7. route every terminal outcome through the established statistics rules.

The provider client should remain an implementation detail of `distill`. The
embedding and student-training concerns stay outside this module, connected
only through the data types documented here and in [data.md](data.md).

## RawExample::to_prompt

`RawExample::to_prompt` is the text-to-prompt boundary used by the current
placeholder and by any future provider client. It renders chronological
context, when present, as a numbered `Context (previous messages)` block and
then renders the current message under `Current message to classify`.

Before it formats either kind of text, the method limits every input string to
`MAX_MESSAGE_LENGTH` and removes control characters other than newline, tab,
and carriage return. It then limits the assembled prompt content to
`MAX_PROMPT_LENGTH` and appends `[truncated]` on overflow. These are resource
bounds, not a prompt-injection defense: user-provided strings remain untrusted
teacher-visible content.

## TeacherConfig::verify_endpoint

`TeacherConfig::verify_endpoint` resolves the configured endpoint or the
provider default, then applies `EndpointSecurity::verify_endpoint`. That local
policy checks the required scheme and, when configured, the exact allowlisted
host. It also calls `validate_cert_fingerprint`, which checks only that a
configured SHA-256 fingerprint has 64 ASCII-hex characters.

The method opens no connection, compares no peer certificate, and does not
calculate a local model checksum. A live provider client must perform TLS
verification and certificate-pin comparison after it has the peer certificate;
an integration that loads local weights must separately pass its calculated
checksum to `verify_model_checksum`.

## DistillationPipeline::label_single

`label_single` is deliberately a provider-shaped placeholder. It starts timing
and materializes the bounded prompt, but it does not send the prompt, read the
configured API-key environment variable, retry, or parse a response. Instead,
it emits its fixed score vector, optionally softmax-normalizes it, assigns
confidence `0.85`, and applies the configured confidence threshold.

This preserves the eventual client boundary: a real implementation should
replace only the simulated result with provider I/O and response validation,
while retaining the prompt limits, label order, threshold behavior, and result
accounting described above.
