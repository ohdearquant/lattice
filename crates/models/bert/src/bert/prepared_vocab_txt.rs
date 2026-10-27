//! Dormant, layout-neutral prepared-BERT `vocab.txt` facts.
//!
//! This module mirrors only the line-to-ID substrate shared by the legacy
//! WordPiece `vocab.txt` and BPE `vocab.txt` + `merges.txt` loaders. It does
//! not require special tokens. It records optional exact known-token IDs, and
//! a separate validator enforces the legacy WordPiece specials only after the
//! caller has selected `VocabTxt`. A separate resolver derives the raw
//! `VocabTxtMerges` BPE control IDs without inspecting `merges.txt`. This
//! module does not validate merges, emitted-ID or config ranges, or any live
//! file.

use std::cmp::Ordering;
use std::mem::size_of;
use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertVocabTxtLimitAxis {
    VocabTxtBytes,
    VocabEntries,
    SpanScratchBytes,
    ParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertVocabTxtExpression {
    BaseParseWorkBytes,
    SpanScratchBytes,
    EntryCount,
    HeapIndex,
    ParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertVocabTxtAllocationArena {
    TokenSpans,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertVocabTxtError {
    PlatformUnrepresentable {
        axis: PreparedBertVocabTxtLimitAxis,
        value: u64,
    },
    Exceeded {
        axis: PreparedBertVocabTxtLimitAxis,
        actual: u64,
        limit: u64,
    },
    ArithmeticOverflow(PreparedBertVocabTxtExpression),
    InvalidUtf8 {
        valid_up_to: usize,
    },
    EmptyVocabulary,
    TokenIdUnrepresentable {
        cardinality: u64,
    },
    DuplicateToken {
        first_id: u32,
        duplicate_id: u32,
    },
    AllocationFailed {
        arena: PreparedBertVocabTxtAllocationArena,
        requested_bytes: u64,
    },
    SecondPassCensusMismatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertVocabTxtLimits {
    max_vocab_txt_bytes: NonZeroU64,
    max_vocab_entries: NonZeroU64,
    max_span_scratch_bytes: NonZeroU64,
    max_parse_work_bytes: NonZeroU64,
}

impl PreparedBertVocabTxtLimits {
    pub(super) fn try_new(
        max_vocab_txt_bytes: NonZeroU64,
        max_vocab_entries: NonZeroU64,
        max_span_scratch_bytes: NonZeroU64,
        max_parse_work_bytes: NonZeroU64,
    ) -> Result<Self, PreparedBertVocabTxtError> {
        let platform_span_max = (usize::MAX as u64).min(isize::MAX as u64);
        Self::try_new_with_platform_max(
            max_vocab_txt_bytes,
            max_vocab_entries,
            max_span_scratch_bytes,
            max_parse_work_bytes,
            platform_span_max,
        )
    }

    fn try_new_with_platform_max(
        max_vocab_txt_bytes: NonZeroU64,
        max_vocab_entries: NonZeroU64,
        max_span_scratch_bytes: NonZeroU64,
        max_parse_work_bytes: NonZeroU64,
        platform_span_max: u64,
    ) -> Result<Self, PreparedBertVocabTxtError> {
        for (axis, value) in [
            (
                PreparedBertVocabTxtLimitAxis::VocabTxtBytes,
                max_vocab_txt_bytes.get(),
            ),
            (
                PreparedBertVocabTxtLimitAxis::VocabEntries,
                max_vocab_entries.get(),
            ),
            (
                PreparedBertVocabTxtLimitAxis::SpanScratchBytes,
                max_span_scratch_bytes.get(),
            ),
            (
                PreparedBertVocabTxtLimitAxis::ParseWorkBytes,
                max_parse_work_bytes.get(),
            ),
        ] {
            if value > platform_span_max {
                return Err(PreparedBertVocabTxtError::PlatformUnrepresentable { axis, value });
            }
        }
        let max_dense_entries = u64::from(u32::MAX) + 1;
        if max_vocab_entries.get() > max_dense_entries {
            return Err(PreparedBertVocabTxtError::TokenIdUnrepresentable {
                cardinality: max_vocab_entries.get(),
            });
        }
        Ok(Self {
            max_vocab_txt_bytes,
            max_vocab_entries,
            max_span_scratch_bytes,
            max_parse_work_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertVocabTxtKnownTokenIds {
    cls: Option<u32>,
    sep: Option<u32>,
    pad: Option<u32>,
    unk: Option<u32>,
    mask: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertBpeVocabTxtKnownTokenIds {
    pipe_pad: Option<u32>,
    pad: Option<u32>,
    endoftext: Option<u32>,
    slash_s: Option<u32>,
    unk: Option<u32>,
    bos: Option<u32>,
    s: Option<u32>,
    eos: Option<u32>,
}

impl PreparedBertBpeVocabTxtKnownTokenIds {
    pub(super) fn pipe_pad(self) -> Option<u32> {
        self.pipe_pad
    }

    pub(super) fn pad(self) -> Option<u32> {
        self.pad
    }

    pub(super) fn endoftext(self) -> Option<u32> {
        self.endoftext
    }

    pub(super) fn slash_s(self) -> Option<u32> {
        self.slash_s
    }

    pub(super) fn unk(self) -> Option<u32> {
        self.unk
    }

    pub(super) fn bos(self) -> Option<u32> {
        self.bos
    }

    pub(super) fn s(self) -> Option<u32> {
        self.s
    }

    pub(super) fn eos(self) -> Option<u32> {
        self.eos
    }
}

impl PreparedBertVocabTxtKnownTokenIds {
    pub(super) fn cls(self) -> Option<u32> {
        self.cls
    }

    pub(super) fn sep(self) -> Option<u32> {
        self.sep
    }

    pub(super) fn pad(self) -> Option<u32> {
        self.pad
    }

    pub(super) fn unk(self) -> Option<u32> {
        self.unk
    }

    pub(super) fn mask(self) -> Option<u32> {
        self.mask
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertVocabTxtFacts {
    vocab_txt_bytes: u64,
    vocabulary_cardinality: NonZeroU64,
    max_token_id: u32,
    span_scratch_bytes: u64,
    logical_parse_work_bytes: u64,
    known_token_ids: PreparedBertVocabTxtKnownTokenIds,
    bpe_known_token_ids: PreparedBertBpeVocabTxtKnownTokenIds,
}

impl PreparedBertVocabTxtFacts {
    pub(super) fn vocab_txt_bytes(self) -> u64 {
        self.vocab_txt_bytes
    }

    pub(super) fn vocabulary_cardinality(self) -> NonZeroU64 {
        self.vocabulary_cardinality
    }

    pub(super) fn max_token_id(self) -> u32 {
        self.max_token_id
    }

    pub(super) fn span_scratch_bytes(self) -> u64 {
        self.span_scratch_bytes
    }

    pub(super) fn logical_parse_work_bytes(self) -> u64 {
        self.logical_parse_work_bytes
    }

    pub(super) fn known_token_ids(self) -> PreparedBertVocabTxtKnownTokenIds {
        self.known_token_ids
    }

    pub(super) fn bpe_known_token_ids(self) -> PreparedBertBpeVocabTxtKnownTokenIds {
        self.bpe_known_token_ids
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertWordPieceSpecialToken {
    Cls,
    Sep,
    Pad,
    Unk,
    Mask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertWordPieceVocabTxtError {
    MissingSpecialToken(PreparedBertWordPieceSpecialToken),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertWordPieceVocabTxtFacts {
    vocab_txt: PreparedBertVocabTxtFacts,
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    unk_id: u32,
    mask_id: u32,
}

impl PreparedBertWordPieceVocabTxtFacts {
    pub(super) fn vocab_txt(self) -> PreparedBertVocabTxtFacts {
        self.vocab_txt
    }

    pub(super) fn cls_id(self) -> u32 {
        self.cls_id
    }

    pub(super) fn sep_id(self) -> u32 {
        self.sep_id
    }

    pub(super) fn pad_id(self) -> u32 {
        self.pad_id
    }

    pub(super) fn unk_id(self) -> u32 {
        self.unk_id
    }

    pub(super) fn mask_id(self) -> u32 {
        self.mask_id
    }
}

pub(super) fn validate_prepared_bert_wordpiece_vocab_txt(
    vocab_txt: PreparedBertVocabTxtFacts,
) -> Result<PreparedBertWordPieceVocabTxtFacts, PreparedBertWordPieceVocabTxtError> {
    let known = vocab_txt.known_token_ids;
    let cls_id = known
        .cls
        .ok_or(PreparedBertWordPieceVocabTxtError::MissingSpecialToken(
            PreparedBertWordPieceSpecialToken::Cls,
        ))?;
    let sep_id = known
        .sep
        .ok_or(PreparedBertWordPieceVocabTxtError::MissingSpecialToken(
            PreparedBertWordPieceSpecialToken::Sep,
        ))?;
    let pad_id = known
        .pad
        .ok_or(PreparedBertWordPieceVocabTxtError::MissingSpecialToken(
            PreparedBertWordPieceSpecialToken::Pad,
        ))?;
    let unk_id = known
        .unk
        .ok_or(PreparedBertWordPieceVocabTxtError::MissingSpecialToken(
            PreparedBertWordPieceSpecialToken::Unk,
        ))?;
    let mask_id = known
        .mask
        .ok_or(PreparedBertWordPieceVocabTxtError::MissingSpecialToken(
            PreparedBertWordPieceSpecialToken::Mask,
        ))?;
    Ok(PreparedBertWordPieceVocabTxtFacts {
        vocab_txt,
        cls_id,
        sep_id,
        pad_id,
        unk_id,
        mask_id,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertBpeVocabTxtFacts {
    vocab_txt: PreparedBertVocabTxtFacts,
    pad_id: u32,
    unk_id: Option<u32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
}

impl PreparedBertBpeVocabTxtFacts {
    pub(super) fn vocab_txt(self) -> PreparedBertVocabTxtFacts {
        self.vocab_txt
    }

    pub(super) fn pad_id(self) -> u32 {
        self.pad_id
    }

    pub(super) fn unk_id(self) -> Option<u32> {
        self.unk_id
    }

    pub(super) fn bos_id(self) -> Option<u32> {
        self.bos_id
    }

    pub(super) fn eos_id(self) -> Option<u32> {
        self.eos_id
    }
}

/// Resolve the raw `vocab.txt` + `merges.txt` BPE constructor's control IDs.
///
/// The selected `merges.txt` is deliberately not inspected by this function.
pub(super) fn resolve_prepared_bert_bpe_vocab_txt(
    vocab_txt: PreparedBertVocabTxtFacts,
) -> PreparedBertBpeVocabTxtFacts {
    let wordpiece = vocab_txt.known_token_ids;
    let bpe = vocab_txt.bpe_known_token_ids;
    let pad_id = bpe
        .pipe_pad
        .or(bpe.pad)
        .or(wordpiece.pad)
        .or(bpe.endoftext)
        .or(bpe.slash_s)
        .unwrap_or(0);
    let unk_id = bpe.unk.or(wordpiece.unk);
    let bos_id = bpe.bos.or(bpe.s);
    let eos_id = bpe.eos.or(bpe.slash_s).or(bpe.endoftext);
    PreparedBertBpeVocabTxtFacts {
        vocab_txt,
        pad_id,
        unk_id,
        bos_id,
        eos_id,
    }
}

#[derive(Clone, Copy)]
struct TokenSpan<'a> {
    token: &'a [u8],
    id: u32,
}

#[derive(Clone, Copy)]
enum KnownTokenSlot {
    SlashS,
    Bos,
    Eos,
    Pad,
    S,
    Unk,
    Endoftext,
    PipePad,
    Cls,
    Mask,
    BracketPad,
    Sep,
    BracketUnk,
}

const KNOWN_TOKEN_TARGETS: [(&[u8], KnownTokenSlot); 13] = [
    (b"</s>", KnownTokenSlot::SlashS),
    (b"<bos>", KnownTokenSlot::Bos),
    (b"<eos>", KnownTokenSlot::Eos),
    (b"<pad>", KnownTokenSlot::Pad),
    (b"<s>", KnownTokenSlot::S),
    (b"<unk>", KnownTokenSlot::Unk),
    (b"<|endoftext|>", KnownTokenSlot::Endoftext),
    (b"<|pad|>", KnownTokenSlot::PipePad),
    (b"[CLS]", KnownTokenSlot::Cls),
    (b"[MASK]", KnownTokenSlot::Mask),
    (b"[PAD]", KnownTokenSlot::BracketPad),
    (b"[SEP]", KnownTokenSlot::Sep),
    (b"[UNK]", KnownTokenSlot::BracketUnk),
];

struct WorkMeter {
    used: u64,
    limit: u64,
}

impl WorkMeter {
    fn with_base(base: u64, limit: u64) -> Result<Self, PreparedBertVocabTxtError> {
        if base > limit {
            return Err(PreparedBertVocabTxtError::Exceeded {
                axis: PreparedBertVocabTxtLimitAxis::ParseWorkBytes,
                actual: base,
                limit,
            });
        }
        Ok(Self { used: base, limit })
    }

    fn charge(&mut self, amount: u64) -> Result<(), PreparedBertVocabTxtError> {
        let next =
            self.used
                .checked_add(amount)
                .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
                    PreparedBertVocabTxtExpression::ParseWorkBytes,
                ))?;
        if next > self.limit {
            return Err(PreparedBertVocabTxtError::Exceeded {
                axis: PreparedBertVocabTxtLimitAxis::ParseWorkBytes,
                actual: next,
                limit: self.limit,
            });
        }
        self.used = next;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ReserveMode {
    Actual,
    #[cfg(test)]
    Fail,
}

pub(super) fn parse_prepared_bert_vocab_txt(
    bytes: &[u8],
    limits: &PreparedBertVocabTxtLimits,
) -> Result<PreparedBertVocabTxtFacts, PreparedBertVocabTxtError> {
    parse_prepared_bert_vocab_txt_with_reserve(bytes, limits, ReserveMode::Actual)
}

fn parse_prepared_bert_vocab_txt_with_reserve(
    bytes: &[u8],
    limits: &PreparedBertVocabTxtLimits,
    reserve_mode: ReserveMode,
) -> Result<PreparedBertVocabTxtFacts, PreparedBertVocabTxtError> {
    let vocab_txt_bytes = u64::try_from(bytes.len()).map_err(|_| {
        PreparedBertVocabTxtError::PlatformUnrepresentable {
            axis: PreparedBertVocabTxtLimitAxis::VocabTxtBytes,
            value: u64::MAX,
        }
    })?;
    enforce_limit(
        PreparedBertVocabTxtLimitAxis::VocabTxtBytes,
        vocab_txt_bytes,
        limits.max_vocab_txt_bytes.get(),
    )?;
    // Charge one complete input span for UTF-8 validation, one for each of the
    // two line scans, and one for worst-case trailing-CR normalization. The
    // data-dependent duplicate proof adds its own charge below.
    let base_work =
        vocab_txt_bytes
            .checked_mul(4)
            .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
                PreparedBertVocabTxtExpression::BaseParseWorkBytes,
            ))?;
    let mut work = WorkMeter::with_base(base_work, limits.max_parse_work_bytes.get())?;
    let text =
        std::str::from_utf8(bytes).map_err(|error| PreparedBertVocabTxtError::InvalidUtf8 {
            valid_up_to: error.valid_up_to(),
        })?;

    let mut entry_count = 0_u64;
    for _ in text.lines() {
        entry_count =
            entry_count
                .checked_add(1)
                .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
                    PreparedBertVocabTxtExpression::EntryCount,
                ))?;
        enforce_limit(
            PreparedBertVocabTxtLimitAxis::VocabEntries,
            entry_count,
            limits.max_vocab_entries.get(),
        )?;
    }
    let vocabulary_cardinality =
        NonZeroU64::new(entry_count).ok_or(PreparedBertVocabTxtError::EmptyVocabulary)?;
    let max_token_id = checked_max_token_id(entry_count)?;
    let requested_span_bytes = entry_count
        .checked_mul(u64::try_from(size_of::<TokenSpan<'_>>()).map_err(|_| {
            PreparedBertVocabTxtError::ArithmeticOverflow(
                PreparedBertVocabTxtExpression::SpanScratchBytes,
            )
        })?)
        .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
            PreparedBertVocabTxtExpression::SpanScratchBytes,
        ))?;
    enforce_limit(
        PreparedBertVocabTxtLimitAxis::SpanScratchBytes,
        requested_span_bytes,
        limits.max_span_scratch_bytes.get(),
    )?;
    let entry_capacity = usize::try_from(entry_count).map_err(|_| {
        PreparedBertVocabTxtError::PlatformUnrepresentable {
            axis: PreparedBertVocabTxtLimitAxis::VocabEntries,
            value: entry_count,
        }
    })?;

    let mut spans = Vec::new();
    match reserve_mode {
        ReserveMode::Actual => spans.try_reserve_exact(entry_capacity).map_err(|_| {
            PreparedBertVocabTxtError::AllocationFailed {
                arena: PreparedBertVocabTxtAllocationArena::TokenSpans,
                requested_bytes: requested_span_bytes,
            }
        })?,
        #[cfg(test)]
        ReserveMode::Fail => {
            return Err(PreparedBertVocabTxtError::AllocationFailed {
                arena: PreparedBertVocabTxtAllocationArena::TokenSpans,
                requested_bytes: requested_span_bytes,
            });
        }
    }
    let span_scratch_bytes = u64::try_from(spans.capacity())
        .ok()
        .and_then(|capacity| capacity.checked_mul(u64::try_from(size_of::<TokenSpan<'_>>()).ok()?))
        .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
            PreparedBertVocabTxtExpression::SpanScratchBytes,
        ))?;
    enforce_limit(
        PreparedBertVocabTxtLimitAxis::SpanScratchBytes,
        span_scratch_bytes,
        limits.max_span_scratch_bytes.get(),
    )?;

    for (id, line) in text.lines().enumerate() {
        if spans.len() >= spans.capacity() {
            return Err(PreparedBertVocabTxtError::SecondPassCensusMismatch);
        }
        let id =
            u32::try_from(id).map_err(|_| PreparedBertVocabTxtError::TokenIdUnrepresentable {
                cardinality: entry_count,
            })?;
        spans.push(TokenSpan {
            token: line.trim_end_matches('\r').as_bytes(),
            id,
        });
    }
    if spans.len() != entry_capacity {
        return Err(PreparedBertVocabTxtError::SecondPassCensusMismatch);
    }

    heap_sort_token_spans(&mut spans, &mut work)?;
    for pair in spans.windows(2) {
        if compare_token_bytes(pair[0].token, pair[1].token, &mut work)? == Ordering::Equal {
            return Err(PreparedBertVocabTxtError::DuplicateToken {
                first_id: pair[0].id.min(pair[1].id),
                duplicate_id: pair[0].id.max(pair[1].id),
            });
        }
    }
    let (known_token_ids, bpe_known_token_ids) = observe_known_token_ids(&spans, &mut work)?;

    Ok(PreparedBertVocabTxtFacts {
        vocab_txt_bytes,
        vocabulary_cardinality,
        max_token_id,
        span_scratch_bytes,
        logical_parse_work_bytes: work.used,
        known_token_ids,
        bpe_known_token_ids,
    })
}

fn observe_known_token_ids(
    spans: &[TokenSpan<'_>],
    work: &mut WorkMeter,
) -> Result<
    (
        PreparedBertVocabTxtKnownTokenIds,
        PreparedBertBpeVocabTxtKnownTokenIds,
    ),
    PreparedBertVocabTxtError,
> {
    let mut wordpiece = PreparedBertVocabTxtKnownTokenIds {
        cls: None,
        sep: None,
        pad: None,
        unk: None,
        mask: None,
    };
    let mut bpe = PreparedBertBpeVocabTxtKnownTokenIds {
        pipe_pad: None,
        pad: None,
        endoftext: None,
        slash_s: None,
        unk: None,
        bos: None,
        s: None,
        eos: None,
    };
    let mut span_index = 0;
    let mut known_index = 0;
    while span_index < spans.len() && known_index < KNOWN_TOKEN_TARGETS.len() {
        let span = spans[span_index];
        let (known, slot) = KNOWN_TOKEN_TARGETS[known_index];
        match compare_token_bytes(span.token, known, work)? {
            Ordering::Less => span_index += 1,
            Ordering::Greater => known_index += 1,
            Ordering::Equal => {
                match slot {
                    KnownTokenSlot::SlashS => bpe.slash_s = Some(span.id),
                    KnownTokenSlot::Bos => bpe.bos = Some(span.id),
                    KnownTokenSlot::Eos => bpe.eos = Some(span.id),
                    KnownTokenSlot::Pad => bpe.pad = Some(span.id),
                    KnownTokenSlot::S => bpe.s = Some(span.id),
                    KnownTokenSlot::Unk => bpe.unk = Some(span.id),
                    KnownTokenSlot::Endoftext => bpe.endoftext = Some(span.id),
                    KnownTokenSlot::PipePad => bpe.pipe_pad = Some(span.id),
                    KnownTokenSlot::Cls => wordpiece.cls = Some(span.id),
                    KnownTokenSlot::Mask => wordpiece.mask = Some(span.id),
                    KnownTokenSlot::BracketPad => wordpiece.pad = Some(span.id),
                    KnownTokenSlot::Sep => wordpiece.sep = Some(span.id),
                    KnownTokenSlot::BracketUnk => wordpiece.unk = Some(span.id),
                }
                span_index += 1;
                known_index += 1;
            }
        }
    }
    Ok((wordpiece, bpe))
}

fn enforce_limit(
    axis: PreparedBertVocabTxtLimitAxis,
    actual: u64,
    limit: u64,
) -> Result<(), PreparedBertVocabTxtError> {
    if actual > limit {
        return Err(PreparedBertVocabTxtError::Exceeded {
            axis,
            actual,
            limit,
        });
    }
    Ok(())
}

fn checked_max_token_id(cardinality: u64) -> Result<u32, PreparedBertVocabTxtError> {
    let last = cardinality
        .checked_sub(1)
        .ok_or(PreparedBertVocabTxtError::EmptyVocabulary)?;
    u32::try_from(last)
        .map_err(|_| PreparedBertVocabTxtError::TokenIdUnrepresentable { cardinality })
}

fn compare_token_bytes(
    left: &[u8],
    right: &[u8],
    work: &mut WorkMeter,
) -> Result<Ordering, PreparedBertVocabTxtError> {
    work.charge(1)?;
    for (&left_byte, &right_byte) in left.iter().zip(right) {
        work.charge(2)?;
        match left_byte.cmp(&right_byte) {
            Ordering::Equal => {}
            ordering => return Ok(ordering),
        }
    }
    Ok(left.len().cmp(&right.len()))
}

fn compare_spans(
    left: TokenSpan<'_>,
    right: TokenSpan<'_>,
    work: &mut WorkMeter,
) -> Result<Ordering, PreparedBertVocabTxtError> {
    let token_order = compare_token_bytes(left.token, right.token, work)?;
    Ok(token_order.then_with(|| left.id.cmp(&right.id)))
}

fn heap_sort_token_spans(
    spans: &mut [TokenSpan<'_>],
    work: &mut WorkMeter,
) -> Result<(), PreparedBertVocabTxtError> {
    if spans.len() < 2 {
        return Ok(());
    }
    for root in (0..(spans.len() / 2)).rev() {
        work.charge(1)?;
        sift_down(spans, root, spans.len(), work)?;
    }
    for end in (1..spans.len()).rev() {
        work.charge(1)?;
        spans.swap(0, end);
        sift_down(spans, 0, end, work)?;
    }
    Ok(())
}

fn sift_down(
    spans: &mut [TokenSpan<'_>],
    mut root: usize,
    end: usize,
    work: &mut WorkMeter,
) -> Result<(), PreparedBertVocabTxtError> {
    loop {
        work.charge(1)?;
        let child = root
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
                PreparedBertVocabTxtExpression::HeapIndex,
            ))?;
        if child >= end {
            return Ok(());
        }
        let mut largest = root;
        if compare_spans(spans[largest], spans[child], work)? == Ordering::Less {
            largest = child;
        }
        let right = child
            .checked_add(1)
            .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
                PreparedBertVocabTxtExpression::HeapIndex,
            ))?;
        if right < end && compare_spans(spans[largest], spans[right], work)? == Ordering::Less {
            largest = right;
        }
        if largest == root {
            return Ok(());
        }
        spans.swap(root, largest);
        root = largest;
    }
}

#[cfg(test)]
mod tests;
