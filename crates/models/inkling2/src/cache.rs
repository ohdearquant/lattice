//! Key, value and convolution caches, one slot per layer application.

use lattice_arch::inkling2_config::{AttentionSpan, Inkling2TextConfig};

use crate::conv::ConvHistory;

/// Decoding state of one sequence: a cache slot for each stored layer, one more for each loop
/// layer in every pass after the first, and the number of tokens processed.
#[derive(Debug, Clone)]
pub struct Inkling2Cache {
    pub(crate) slots: Vec<CacheSlot>,
    pub(crate) position: usize,
    hidden: usize,
    kernel: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct CacheSlot {
    pub(crate) keys: Vec<f32>,
    pub(crate) values: Vec<f32>,
    pub(crate) first_position: usize,
    pub(crate) width: usize,
    pub(crate) window: Option<usize>,
    pub(crate) key_history: ConvHistory,
    pub(crate) value_history: ConvHistory,
    pub(crate) attention_history: ConvHistory,
    pub(crate) mlp_history: ConvHistory,
}

fn slot_count(config: &Inkling2TextConfig) -> usize {
    let lc = config.loop_config;
    config.num_hidden_layers + (lc.num_passes - 1) * (lc.core_end - lc.core_start)
}

/// Key/value width and window of a slot. Slots past the stored layers serve the loop layers,
/// one run of them per extra pass.
fn slot_shape(config: &Inkling2TextConfig, slot: usize) -> (usize, Option<usize>) {
    let lc = config.loop_config;
    let layer = if slot < config.num_hidden_layers {
        slot
    } else {
        lc.core_start + (slot - config.num_hidden_layers) % (lc.core_end - lc.core_start)
    };
    let span = config.layer_spans[layer];
    let heads = config.heads(span);
    let window = match span {
        AttentionSpan::Global => None,
        AttentionSpan::Sliding => Some(config.sliding_window_size),
    };
    (heads.num_key_value_heads * heads.head_dim, window)
}

impl Inkling2Cache {
    pub(crate) fn new(config: &Inkling2TextConfig) -> Self {
        let hidden = config.hidden_size;
        let kernel = config.conv_kernel_size;
        let slots = (0..slot_count(config))
            .map(|slot| {
                let (width, window) = slot_shape(config, slot);
                CacheSlot {
                    keys: Vec::new(),
                    values: Vec::new(),
                    first_position: 0,
                    width,
                    window,
                    key_history: ConvHistory::new(width, kernel),
                    value_history: ConvHistory::new(width, kernel),
                    attention_history: ConvHistory::new(hidden, kernel),
                    mlp_history: ConvHistory::new(hidden, kernel),
                }
            })
            .collect();
        Self {
            slots,
            position: 0,
            hidden,
            kernel,
        }
    }

    /// Whether the cache has the slots `config` needs.
    pub(crate) fn fits(&self, config: &Inkling2TextConfig) -> bool {
        self.hidden == config.hidden_size
            && self.kernel == config.conv_kernel_size
            && self.slots.len() == slot_count(config)
            && self
                .slots
                .iter()
                .enumerate()
                .all(|(index, slot)| (slot.width, slot.window) == slot_shape(config, index))
    }

    /// Tokens processed so far, which is the position of the next token.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Number of slots: the stored layers, plus the loop layers once for each pass after the
    /// first.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Keys held in slot `slot`, or `None` past the last slot. A sliding slot keeps the last
    /// `window - 1` tokens between calls; a global slot keeps every token.
    pub fn cached_tokens(&self, slot: usize) -> Option<usize> {
        self.slots.get(slot).map(CacheSlot::token_count)
    }
}

impl CacheSlot {
    pub(crate) fn token_count(&self) -> usize {
        self.keys.len() / self.width
    }

    /// Drops the keys and values no later query can see: a sliding slot keeps the last
    /// `window - 1`.
    pub(crate) fn evict(&mut self) {
        let Some(window) = self.window else {
            return;
        };
        let keep = window - 1;
        let count = self.token_count();
        if count > keep {
            let dropped = count - keep;
            self.keys.drain(..dropped * self.width);
            self.values.drain(..dropped * self.width);
            self.first_position += dropped;
        }
    }
}
