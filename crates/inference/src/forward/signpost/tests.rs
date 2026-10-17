use super::Label;
use super::Scope;
use super::SignpostMode;
use super::interval;
use super::recorder::{Event, RecordingInterval};

// Pure parsing, platform- and feature-independent (runs under every
// `cargo test` combination the gates use): unset/"auto"/anything
// unrecognized default to the idle-inert mode; only the literal string
// "always" opts into the ordinary category the xctrace CLI path needs.
// Deliberately does not touch `std::env` — see the in-process
// `os_signpost_enabled` regression tests in `imp::tests` (macOS +
// `signpost` feature only) for the category-behavior half of this
// guard.
#[test]
fn signpost_mode_from_env_value() {
    assert_eq!(SignpostMode::from_env_value(None), SignpostMode::Auto);
    assert_eq!(SignpostMode::from_env_value(Some("")), SignpostMode::Auto);
    assert_eq!(
        SignpostMode::from_env_value(Some("auto")),
        SignpostMode::Auto
    );
    assert_eq!(
        SignpostMode::from_env_value(Some("Always")),
        SignpostMode::Auto
    );
    assert_eq!(
        SignpostMode::from_env_value(Some("bogus")),
        SignpostMode::Auto
    );
    assert_eq!(
        SignpostMode::from_env_value(Some("always")),
        SignpostMode::Always
    );
}

// Mutation-sensitive: the mode -> category string mapping `decode_log()`
// relies on. Also platform/feature independent (`category()` compiles
// under every combination; only its callers are macOS+signpost-gated).
// Breaking this mapping (e.g. always returning "DynamicTracing") makes
// this test fail.
#[test]
fn signpost_mode_category_mapping() {
    assert_eq!(SignpostMode::Auto.category(), "DynamicTracing");
    assert_eq!(SignpostMode::Always.category(), "decode");
}

// Exercises the exact call-site shape used in the decode path (bind a
// guard, let it drop at end of scope). Compiles and runs identically
// whether this crate is built with `--features signpost` (a real
// interval on macOS) or without it (the no-op fallback) -- there is no
// separate call-site variant to test per configuration.
#[test]
fn interval_guard_compiles_and_drops_cleanly() {
    let _guard = interval(Label::DecodeStep);
    {
        let _nested = interval(Label::DecodeCbCommit);
    }
}

// Mutation-sensitive: asserts the actual begin/end emission order for a
// nested two-interval scope via the recording seam, independent of the
// real (macOS-only) FFI path exercised by the smoke test above.
#[test]
fn nested_intervals_emit_begin_end_in_order() {
    super::recorder::clear();
    {
        let _outer = RecordingInterval::begin(Label::DecodeStep);
        {
            let _inner = RecordingInterval::begin(Label::DecodeCbCommit);
        }
    }
    assert_eq!(
        super::recorder::events(),
        vec![
            Event::Begin(Label::DecodeStep),
            Event::Begin(Label::DecodeCbCommit),
            Event::End(Label::DecodeCbCommit),
            Event::End(Label::DecodeStep),
        ]
    );
}

// Mutation-sensitive: pairing regression for `Label::DecodeSample`, the
// exact label `sample_decode_traced` in `metal_qwen35.rs` opens --
// that call site is the *only* place any
// of the five decode loops open a `decode.sample` interval (previously
// five independent copies, two of which had already drifted with no
// interval at all). This test fails if `RecordingInterval::begin`
// stops emitting the paired begin/end events.
#[test]
fn decode_sample_label_records_begin_end_pair() {
    super::recorder::clear();
    {
        let _guard = RecordingInterval::begin(Label::DecodeSample);
    }
    assert_eq!(
        super::recorder::events(),
        vec![
            Event::Begin(Label::DecodeSample),
            Event::End(Label::DecodeSample)
        ]
    );
}

// Mutation-sensitive: drives the `Scope` discriminator both ways through
// the exact `RecordingInterval::begin_in` seam `interval_in` mirrors.
// `Scope::NotDecode` (MTP verify / prefill / diagnostic call sites) must
// record zero events; `Scope::Decode` must record the full begin/end
// pair. Reverting the discriminator (emitting unconditionally, as the
// shared helper did before this fix) makes the `NotDecode` assertion
// fail.
#[test]
fn scope_discriminator_silences_non_decode_and_records_decode() {
    super::recorder::clear();
    {
        let _not_decode = RecordingInterval::begin_in(Scope::NotDecode, Label::DecodeStep);
    }
    assert_eq!(
        super::recorder::events(),
        vec![],
        "Scope::NotDecode must record nothing"
    );

    super::recorder::clear();
    {
        let _decode = RecordingInterval::begin_in(Scope::Decode, Label::DecodeStep);
    }
    assert_eq!(
        super::recorder::events(),
        vec![
            Event::Begin(Label::DecodeStep),
            Event::End(Label::DecodeStep)
        ],
        "Scope::Decode must record the begin/end pair"
    );
}
