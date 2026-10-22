//! Grammar facade over `lattice-grammar`.
//!
//! Every public item keeps the path it had before the grammar engine moved into
//! its own crate. The re-exports are named rather than globbed so the facade
//! never widens the supported API.

pub mod engine {
    #[cfg(feature = "bench-internals")]
    #[doc(inline)]
    pub use lattice_grammar::engine::{BfsDedupProfile, profile_reachable_states};
    #[doc(inline)]
    pub use lattice_grammar::engine::{
        BuildProfile, GrammarEngine, GrammarError, MaskProfile, context_recheck_candidates,
        context_recheck_simulated, enable_mask_profiling, last_build_profile,
        probe_reachable_states, take_mask_profile,
    };
}

pub mod gbnf {
    #[doc(inline)]
    pub use lattice_grammar::gbnf::{GbnfError, parse_gbnf};
}

pub mod json_schema {
    #[doc(inline)]
    pub use lattice_grammar::json_schema::{SchemaError, compile, compile_json_schema};
}

pub mod pda {
    #[doc(inline)]
    pub use lattice_grammar::pda::{
        Alt, BuilderError, CompiledGrammar, GrammarBuilder, GrammarState, Rule, SimResult,
        StackFrame, StackLimitError, StepResult, Symbol, UNCHOSEN_ALT, advance_byte,
        simulate_token,
    };
}

pub mod spec {
    #[doc(inline)]
    pub use lattice_grammar::spec::GrammarSpec;
}

pub mod trie {
    #[doc(inline)]
    pub use lattice_grammar::trie::ByteTrie;
}

pub mod vocab_partition {
    #[doc(inline)]
    pub use lattice_grammar::vocab_partition::{MAX_GRAMMAR_STATES, VocabPartition};
}

#[doc(no_inline)]
pub use engine::{GrammarEngine, GrammarError};
#[doc(no_inline)]
pub use spec::GrammarSpec;
