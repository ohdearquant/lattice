//! Experimental lazy mask engine. This does not implement strict-serving admission.

use std::mem::size_of;
use std::sync::Mutex;

use crate::engine::GrammarError;
use crate::gbnf::parse_gbnf;
use crate::json_schema::compile;
use crate::pda::{
    CompiledGrammar, GrammarState, StackFrame, StepResult, advance_byte, initial_grammar_state,
};
use crate::spec::GrammarSpec;
use crate::trie::ByteTrie;

const MAX_ENTRIES: usize = 256;

struct Key {
    stacks: Box<[Box<[StackFrame]>]>,
    complete: bool,
}

impl Key {
    fn matches(&self, state: &GrammarState) -> bool {
        self.complete == state.complete
            && self.stacks.len() == state.stacks.len()
            && self
                .stacks
                .iter()
                .zip(&state.stacks)
                .all(|(left, right)| left.as_ref() == right.as_slice())
    }
}

struct Entry {
    key: Key,
    allowed: Box<[u64]>,
    bytes: usize,
}

struct Cache {
    entries: Vec<Entry>,
    retained: usize,
    budget: usize,
}

impl Cache {
    fn new(budget: usize) -> Self {
        let capacity = (budget / (2 * size_of::<Entry>())).min(MAX_ENTRIES);
        let mut entries = Vec::with_capacity(capacity);
        if entries.capacity() * size_of::<Entry>() > budget {
            entries = Vec::new();
        }
        let retained = entries.capacity() * size_of::<Entry>();
        Self {
            entries,
            retained,
            budget,
        }
    }

    fn entry_bytes(state: &GrammarState, words: usize) -> Option<usize> {
        let stacks = state
            .stacks
            .len()
            .checked_mul(size_of::<Box<[StackFrame]>>())?;
        let frames = state.stacks.iter().try_fold(0usize, |sum, stack| {
            sum.checked_add(stack.len().checked_mul(size_of::<StackFrame>())?)
        })?;
        stacks
            .checked_add(frames)?
            .checked_add(words.checked_mul(size_of::<u64>())?)
    }

    fn insert(&mut self, state: &GrammarState, allowed: Vec<u64>) {
        let Some(bytes) = Self::entry_bytes(state, allowed.len()) else {
            return;
        };
        let slots = self.entries.capacity();
        let fixed = slots * size_of::<Entry>();
        if slots == 0 || bytes > self.budget.saturating_sub(fixed) {
            return;
        }
        if self.entries.iter().any(|entry| entry.key.matches(state)) {
            return;
        }
        while self.entries.len() == slots || bytes > self.budget - self.retained {
            let removed = self.entries.remove(0);
            self.retained -= removed.bytes;
        }
        let stacks = state
            .stacks
            .iter()
            .map(|stack| stack.clone().into_boxed_slice())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.entries.push(Entry {
            key: Key {
                stacks,
                complete: state.complete,
            },
            allowed: allowed.into_boxed_slice(),
            bytes,
        });
        self.retained += bytes;
    }
}

/// A separate, opt-in exact trie-mask engine with an engine-local bounded cache.
///
/// Construction skips eager state enumeration and partitions. The cache budget
/// accounts for entry slots, state-key payloads and mask words; allocator
/// metadata, the vocabulary trie, temporary misses and concurrent callers are
/// outside that budget. Misses can be slow. This type does not replace the
/// strict-serving 256-state admission policy.
pub struct LazyGrammarEngine {
    grammar: CompiledGrammar,
    vocab_bytes: Vec<Vec<u8>>,
    trie: ByteTrie,
    cache: Mutex<Cache>,
    #[cfg(test)]
    hits: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    misses: std::sync::atomic::AtomicUsize,
}

impl LazyGrammarEngine {
    /// Compile a grammar and trie without enumerating grammar states.
    ///
    /// `cache_budget` is an upper bound on accounted cache bytes. A budget too
    /// small for an entry disables caching while keeping exact trie masking.
    ///
    /// # Errors
    ///
    /// Returns the schema or GBNF compilation error.
    pub fn new(
        spec: &GrammarSpec,
        vocab_bytes: Vec<Vec<u8>>,
        cache_budget: usize,
    ) -> Result<Self, GrammarError> {
        let grammar = match spec {
            GrammarSpec::JsonSchema(schema) => compile(schema)?,
            GrammarSpec::Gbnf(gbnf) => parse_gbnf(gbnf)?,
        };
        let trie = ByteTrie::build(&vocab_bytes);
        Ok(Self {
            grammar,
            vocab_bytes,
            trie,
            cache: Mutex::new(Cache::new(cache_budget)),
            #[cfg(test)]
            hits: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            misses: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Create the initial state for a sequence.
    pub fn initial_state(&self) -> GrammarState {
        initial_grammar_state(&self.grammar)
    }

    /// Mask disallowed tokens, preserving allowed logits and the logits tail.
    ///
    /// # Errors
    ///
    /// Returns an error if logits are too short, the matcher exceeds its limits,
    /// or the cache mutex has been poisoned. A matcher error leaves logits intact.
    pub fn mask_logits(
        &self,
        state: &GrammarState,
        logits: &mut [f32],
    ) -> Result<(), GrammarError> {
        let vocab_size = self.vocab_bytes.len();
        if logits.len() < vocab_size {
            return Err(GrammarError(format!(
                "logits length {} is shorter than vocabulary size {vocab_size}",
                logits.len()
            )));
        }
        {
            let cache = self
                .cache
                .lock()
                .map_err(|_| GrammarError("lazy mask cache poisoned".into()))?;
            if let Some(entry) = cache.entries.iter().find(|entry| entry.key.matches(state)) {
                #[cfg(test)]
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                apply_mask(&entry.allowed, logits, vocab_size);
                return Ok(());
            }
        }
        #[cfg(test)]
        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut classified = vec![0.0; vocab_size];
        self.trie
            .mask(state, &self.grammar, vocab_size, &mut classified)?;
        let mut allowed = vec![0u64; vocab_size.div_ceil(64)];
        for (id, value) in classified.into_iter().enumerate() {
            if value != f32::NEG_INFINITY {
                allowed[id / 64] |= 1 << (id % 64);
            }
        }
        apply_mask(&allowed, logits, vocab_size);
        self.cache
            .lock()
            .map_err(|_| GrammarError("lazy mask cache poisoned".into()))?
            .insert(state, allowed);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn cache_counts(&self) -> (usize, usize) {
        use std::sync::atomic::Ordering;
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    /// Advance a state over a vocabulary token.
    ///
    /// # Errors
    ///
    /// Returns an error when the matcher exceeds its limits; the state may have
    /// advanced past earlier bytes in that token, as with `GrammarEngine`.
    pub fn advance(&self, state: &mut GrammarState, token_id: u32) -> Result<bool, GrammarError> {
        let Some(bytes) = self.vocab_bytes.get(token_id as usize) else {
            return Ok(false);
        };
        for &byte in bytes {
            match advance_byte(state, &self.grammar, byte) {
                StepResult::Accepted => {}
                StepResult::Rejected => return Ok(false),
                StepResult::StackLimitExceeded => return Err(crate::pda::StackLimitError.into()),
            }
        }
        Ok(true)
    }
}

fn apply_mask(allowed: &[u64], logits: &mut [f32], vocab_size: usize) {
    for (id, logit) in logits[..vocab_size].iter_mut().enumerate() {
        if allowed[id / 64] & (1 << (id % 64)) == 0 {
            *logit = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests;
