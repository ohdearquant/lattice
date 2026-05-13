# ADR-028: Multi-Provider Teacher Strategy

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-tune

## Context

lattice-tune implements knowledge distillation where soft labels from LLM teachers train
lightweight student classifiers. The choice of teacher provider directly impacts:

1. **Vendor Flexibility** - Avoiding lock-in to a single LLM provider ensures
   negotiating leverage and resilience against service disruptions.

2. **Cost Optimization** - Different providers offer varying price/performance
   trade-offs (e.g., Claude Haiku for fast/cheap labeling vs. Claude Sonnet for
   high-quality labeling).

3. **Local Development** - Developers need to iterate without incurring API costs or
   requiring network access. Ollama-compatible local models enable offline development.

4. **Regulatory Compliance** - Some deployments require data to remain on-premises,
   mandating local model support.

5. **Quality Variance** - Different models exhibit different strengths in intent
   classification. The ability to switch teachers allows selecting the best model for
   specific domains.

## Decision

Support multiple LLM teacher providers through an extensible `TeacherProvider` enum:

```rust
pub enum TeacherProvider {
    Claude,              // Anthropic Claude models
    OpenAI,              // OpenAI GPT models
    Gemini,              // Google Gemini models
    Local,               // Ollama-compatible local models
    Custom(String),      // Custom provider name for future extensibility
}
```

Each provider integrates through a unified `TeacherConfig` structure with:

- Provider-specific endpoint defaults
- Configurable authentication via environment variables
- Security settings (TLS enforcement, domain whitelisting)
- Pre-configured presets for common use cases

**Pre-configured Presets**:

| Preset                   | Model               | Timeout | Use Case              |
| ------------------------ | ------------------- | ------- | --------------------- |
| `claude_sonnet()`        | claude-sonnet-4     | 30s     | High-quality labeling |
| `claude_haiku()`         | claude-3-5-haiku    | 15s     | Fast/cheap labeling   |
| `gpt4()`                 | gpt-4-turbo-preview | 30s     | Alternative teacher   |
| `gemini_pro()`           | gemini-pro          | 30s     | Google ecosystem      |
| `local(model, endpoint)` | Custom              | 60s     | Privacy-first         |

## Consequences

### Positive

- **Flexibility**: Switch providers without code changes, only configuration.
- **Cost Control**: Choose Claude Haiku for bulk labeling, Claude Sonnet for high-stakes
  classification.
- **Offline Development**: Local preset enables development without network
  dependencies.
- **Future-Proof**: `Custom(String)` variant allows integrating new providers without
  schema changes.
- **Security**: Domain whitelisting per provider prevents data exfiltration to
  unauthorized endpoints.

### Negative

- **Configuration Complexity**: Users must manage API keys for each provider they use.
- **Quality Variance**: Different teachers produce different soft label distributions;
  models trained on mixed providers may exhibit inconsistent behavior.
- **Testing Burden**: Each provider requires integration testing to ensure
  compatibility.
- **Documentation Overhead**: Must document provider-specific quirks and optimal
  configurations.

### Mitigations

- Provide sensible defaults via presets to reduce configuration burden.
- Record `teacher_model` in `ExampleMetadata` for lineage tracking.
- Normalize labels via softmax to reduce distribution variance across providers.

## Alternatives Considered

### Alternative 1: Single Provider (Claude-only)

**Approach**: Support only Anthropic Claude models.

**Pros**:

- Simpler implementation
- Consistent label quality
- Reduced testing surface

**Cons**:

- Vendor lock-in
- No offline development
- Unable to leverage cost differences between providers

**Rejected because**: Vendor lock-in contradicts the design principle of
extensibility. The slight implementation complexity is outweighed by operational
flexibility.

### Alternative 2: Provider Abstraction Layer

**Approach**: Define a `TeacherTrait` and implement provider adapters.

```rust
pub trait Teacher {
    async fn label(&self, example: &RawExample) -> Result<LabelingResult>;
}
```

**Pros**:

- Clean separation of concerns
- Easy to add new providers
- Testable via mock implementations

**Cons**:

- Over-engineering for current scope (4 providers)
- Runtime dispatch overhead
- Added abstraction layer to maintain

**Rejected because**: The current enum-based approach is sufficient for the known
provider set. If provider count exceeds 6-8, this alternative should be revisited. The
`Custom(String)` variant provides escape hatch for unforeseen providers.

## References

- TeacherConfig: `crates/tune/src/distill/teacher.rs`
- EndpointSecurity: `crates/tune/src/distill/teacher.rs`
