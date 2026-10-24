# lattice-tokenizer

Pure Rust tokenizers for Lattice inference: WordPiece, SentencePiece, Qwen-style
byte-level BPE, and the additive Gemma BPE path. It also provides the
`Tokenizer` trait and model-directory tokenizer auto-detection.

## Features

The default `parallel` feature runs WordPiece batches larger than 16 texts on
multiple threads when the current Rayon pool has more than one worker. Existing
pool bounds and input ordering are preserved. Other tokenizer implementations
are unaffected.

To use serial WordPiece batches without this crate's normal Rayon dependency:

```toml
lattice-tokenizer = { version = "0.15.0", default-features = false }
```

All tokenizer APIs remain available. Cargo features are additive, so another
consumer can reenable `parallel`. Rayon remains a development dependency for
existing pool-based parity tests; test builds and other workspace crates may
therefore still include it.
