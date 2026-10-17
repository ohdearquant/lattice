use super::super::prepared_tokenizer_selection::{
    PreparedBertTokenizerLayout, PreparedBertTokenizerSelectionError, SentencePieceCandidate,
};
use super::*;

/// Bit order matches `RawPreparedBertFilePresence::mask()`: model_safetensors,
/// config_json, tokenizer_config_json, tokenizer_json, vocab_json, merges_txt,
/// vocab_txt, tokenizer_model, sentencepiece_bpe_model, spiece_model (MSB to
/// LSB, 10 bits).
fn presence(mask: u16) -> RawPreparedBertFilePresence {
    RawPreparedBertFilePresence::new(
        mask & 0b10_0000_0000 != 0,
        mask & 0b01_0000_0000 != 0,
        mask & 0b00_1000_0000 != 0,
        mask & 0b00_0100_0000 != 0,
        mask & 0b00_0010_0000 != 0,
        mask & 0b00_0001_0000 != 0,
        mask & 0b00_0000_1000 != 0,
        mask & 0b00_0000_0100 != 0,
        mask & 0b00_0000_0010 != 0,
        mask & 0b00_0000_0001 != 0,
    )
}

#[test]
fn required_files_fail_in_model_config_tokenizer_config_order() {
    for (mask, expected) in [
        (
            0b00_0100_0000,
            PreparedBertFilePresenceError::MissingModelSafetensors,
        ),
        (
            0b10_0100_0000,
            PreparedBertFilePresenceError::MissingConfigJson,
        ),
        (
            0b11_0100_0000,
            PreparedBertFilePresenceError::MissingTokenizerConfigJson,
        ),
    ] {
        assert_eq!(
            validate_prepared_bert_file_presence(presence(mask)),
            Err(expected)
        );
    }
}

#[test]
fn tokenizer_validation_and_precedence_are_delegated_to_the_selector() {
    use PreparedBertTokenizerLayout::{TokenizerJson, VocabJsonMerges, VocabTxt, VocabTxtMerges};

    for (mask, expected) in [
        (0b11_1111_1111, Ok(TokenizerJson)),
        (0b11_1011_0000, Ok(VocabJsonMerges)),
        (0b11_1001_1000, Ok(VocabTxtMerges)),
        (0b11_1000_1000, Ok(VocabTxt)),
        (
            0b11_1000_0100,
            Err(PreparedBertFilePresenceError::Tokenizer(
                PreparedBertTokenizerSelectionError::SentencePieceSelected(
                    SentencePieceCandidate::TokenizerDotModel,
                ),
            )),
        ),
        (
            0b11_1000_0010,
            Err(PreparedBertFilePresenceError::Tokenizer(
                PreparedBertTokenizerSelectionError::SentencePieceSelected(
                    SentencePieceCandidate::SentencePieceBpe,
                ),
            )),
        ),
        (
            0b11_1000_0001,
            Err(PreparedBertFilePresenceError::Tokenizer(
                PreparedBertTokenizerSelectionError::SentencePieceSelected(
                    SentencePieceCandidate::Spiece,
                ),
            )),
        ),
        (
            0b11_1001_0000,
            Err(PreparedBertFilePresenceError::Tokenizer(
                PreparedBertTokenizerSelectionError::MergesWithoutVocab,
            )),
        ),
        (
            0b11_1010_0000,
            Err(PreparedBertFilePresenceError::Tokenizer(
                PreparedBertTokenizerSelectionError::VocabJsonWithoutMerges,
            )),
        ),
    ] {
        let actual = validate_prepared_bert_file_presence(presence(mask))
            .map(PreparedBertFilePresence::tokenizer_layout);
        assert_eq!(actual, expected, "recognized-file mask {mask:010b}");
    }
}

#[test]
fn validated_output_preserves_all_ten_raw_presence_facts() {
    for mask in [
        0b11_1111_1111u16,
        0b11_1011_0000,
        0b11_1001_1000,
        0b11_1000_1000,
    ] {
        let raw = presence(mask);
        let validated = validate_prepared_bert_file_presence(raw).unwrap();
        assert_eq!(validated.presence(), raw);
        assert_eq!(validated.presence().mask(), mask);
    }
}

/// Every present recognized SentencePiece candidate is still carried in the
/// raw presence facts even though selecting one is rejected: the inventory
/// evidence (this struct) and the selection outcome are independent.
#[test]
fn a_rejected_sentencepiece_selection_does_not_erase_the_original_presence_mask() {
    let mask = 0b11_1000_0111u16; // required files + all three SentencePiece candidates
    let raw = presence(mask);
    assert_eq!(raw.mask(), mask);
    assert_eq!(
        validate_prepared_bert_file_presence(raw),
        Err(PreparedBertFilePresenceError::Tokenizer(
            PreparedBertTokenizerSelectionError::SentencePieceSelected(
                SentencePieceCandidate::TokenizerDotModel,
            ),
        ))
    );
    // The caller retains the original Copy value regardless of the error.
    assert_eq!(raw.mask(), mask);
}
