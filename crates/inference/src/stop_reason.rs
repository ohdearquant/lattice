/// Why autoregressive generation terminated.
///
/// Returned as `GenerateOutput::stop_reason`. All real generation exit paths set `Some(…)`.
/// `None` is reserved for non-generation returns (empty prompt, stub code paths) that do not
/// correspond to any of these causes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// EOS token, configured stop-token id, or stop-string match.
    Eos,
    /// Grammar-constrained generation reached a state with no valid continuation.
    Grammar,
    /// `max_new_tokens` budget (or answer-budget cap) exhausted before a stop condition.
    Length,
    /// KV cache / context capacity reached before a stop condition.
    KvFull,
    /// Streaming callback returned `false`, requesting cancellation.
    Interrupt,
}
