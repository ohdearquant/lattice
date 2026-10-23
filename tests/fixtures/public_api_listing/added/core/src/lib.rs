//! Items that the facade re-exports, with one added inherent method.

mod tokenizer {
    /// A tokenizer with both inherent and trait methods.
    pub struct BpeTokenizer {
        /// Number of merges loaded by the tokenizer.
        pub merges: usize,
        vocabulary_size: usize,
    }

    impl BpeTokenizer {
        /// Creates a tokenizer with the given merge count.
        pub fn new(merges: usize) -> Self {
            Self {
                merges,
                vocabulary_size: merges + 1,
            }
        }

        /// Returns the vocabulary size.
        pub fn vocabulary_size(&self) -> usize {
            self.vocabulary_size
        }

        /// Added method exercised by the acceptance control.
        pub fn is_empty(&self) -> bool {
            self.merges == 0
        }

        /// Maximum sequence length accepted by the tokenizer.
        pub const MAX_SEQUENCE: usize = 512;
    }

    impl Default for BpeTokenizer {
        fn default() -> Self {
            Self::new(0)
        }
    }

    /// A token pair used to exercise tuple fields.
    pub struct TokenPair(pub u32, u8);

    /// Errors produced by the tokenizer.
    #[derive(Debug)]
    pub enum TokenizerError {
        /// The input was empty.
        EmptyInput,
        /// The requested sequence length exceeds the maximum.
        TooLong(usize),
    }

    impl core::fmt::Display for TokenizerError {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::EmptyInput => formatter.write_str("empty input"),
                Self::TooLong(length) => write!(formatter, "input of {length} tokens is too long"),
            }
        }
    }

    impl core::error::Error for TokenizerError {}

    /// Tokenizer operations used by callers.
    pub trait Tokenizer {
        /// Encodes text into token identifiers.
        fn encode(&self, text: &str) -> Vec<u32>;

        /// Decodes identifiers back into text.
        fn decode(&self, ids: &[u32]) -> String {
            format!("{ids:?}")
        }
    }

    impl Tokenizer for BpeTokenizer {
        fn encode(&self, text: &str) -> Vec<u32> {
            text.bytes().map(u32::from).collect()
        }
    }

    /// The default tokenizer type.
    pub type DefaultTokenizer = BpeTokenizer;

    /// Default maximum sequence length.
    pub const DEFAULT_MAX_SEQUENCE: usize = 512;

    /// Process-wide default tokenizer name.
    pub static DEFAULT_TOKENIZER_NAME: &str = "bpe";

    /// Returns a tokenizer with the default configuration.
    pub fn default_tokenizer() -> BpeTokenizer {
        BpeTokenizer::default()
    }
}

#[doc(inline)]
pub use tokenizer::{
    BpeTokenizer, DEFAULT_MAX_SEQUENCE, DEFAULT_TOKENIZER_NAME,
    DEFAULT_TOKENIZER_NAME as TOKENIZER_NAME, DefaultTokenizer, TokenPair, Tokenizer,
    TokenizerError, default_tokenizer,
};

/// Returns the default tokenizer name.
#[macro_export]
macro_rules! tokenizer_name {
    () => {
        "bpe"
    };
}

/// Implementation detail retained for engine callers.
#[doc(hidden)]
pub fn internal_padding() -> usize {
    0
}

/// Engine entry points exposed through the facade.
pub mod engine {
    /// Runs the engine.
    pub fn run() -> u32 {
        1
    }
}
