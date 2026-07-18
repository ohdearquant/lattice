//! `os_signpost` instrumentation for the Metal decode path (feature `signpost`,
//! default OFF).
//!
//! Every call site is an RAII interval guard: `signpost::interval(Label::X)`
//! begins an `os_signpost` interval and ends it when the guard drops. When the
//! `signpost` feature is disabled, or the target isn't macOS, [`interval`]
//! returns a zero-sized no-op type and the call compiles to nothing — the
//! call sites in the decode path never need a `#[cfg(...)]` of their own.
//!
//! The label set is closed (see [`Label`]) and each variant is backed by a
//! `static` byte string placed in the binary's `__TEXT,__oslogstring`
//! section — the same section Clang places `os_signpost_interval_begin`'s
//! string-literal `name`/`format` arguments into. Apple's underscored
//! `_os_signpost_emit_with_name_impl` entry point (what the public macros
//! expand to) requires its `name`/`format` pointers to live there; a heap
//! `CString` does not qualify and is recorded as an empty name in Instruments.
//!
//! See `crates/inference/METAL_TRACE.md` for how to capture a Metal System
//! Trace with these signposts and the label glossary.

/// Closed set of decode-path signpost interval labels. Adding a new call site
/// means adding a variant here (and its backing static in `imp`, macOS-only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)] // "Decode" mirrors the decode.* signpost namespace, not redundant noise
pub(crate) enum Label {
    DecodeStep,
    DecodeCbCommit,
    DecodeCbWait,
    DecodeHostScalarRead,
    DecodeGrammarMask,
    DecodeSample,
}

/// Runtime selector for the `os_log` category `decode_log()` creates its
/// handle with, read once from `LATTICE_SIGNPOST_MODE` at first use.
///
/// The two documented capture paths need opposite `os_signpost_enabled`
/// defaults: Instruments.app (GUI) attaches as an Instruments-style tool
/// session, so `DynamicTracing` sees it and stays idle-inert the rest of the
/// time. `xcrun xctrace record --instrument os_signpost` (CLI) does not —
/// on this machine's macOS 26, a direct `os_signpost_enabled` probe under
/// that exact invocation reported `DynamicTracing=0` while an ordinary
/// category reported `decode=1`, so the CLI path needs the ordinary
/// category to observe anything at all. One process cannot satisfy both
/// defaults with a compile-time-fixed category, hence the runtime switch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SignpostMode {
    /// `DynamicTracing` category (`OS_LOG_CATEGORY_DYNAMIC_TRACING`):
    /// idle-inert, `os_signpost_enabled` is only true while an
    /// Instruments-style tool session is attached. Default (unset or
    /// `"auto"`), preserves round 3's idle-inertness fix.
    Auto,
    /// Ordinary `"decode"` category: `os_signpost_enabled` is true whenever
    /// any collector could observe it, including the `xcrun xctrace` CLI
    /// capture path. Opt-in via `LATTICE_SIGNPOST_MODE=always`.
    Always,
}

impl SignpostMode {
    pub(crate) const ENV_VAR: &'static str = "LATTICE_SIGNPOST_MODE";

    /// Pure parsing, independent of the actual process environment so it can
    /// be unit-tested without mutating global env state: `"always"` selects
    /// [`SignpostMode::Always`]; unset, `"auto"`, or anything else defaults
    /// to [`SignpostMode::Auto`] (fail-safe toward the idle-inert default).
    pub(crate) fn from_env_value(value: Option<&str>) -> Self {
        match value {
            Some("always") => SignpostMode::Always,
            _ => SignpostMode::Auto,
        }
    }

    #[cfg_attr(not(all(feature = "signpost", target_os = "macos")), allow(dead_code))]
    fn from_process_env() -> Self {
        let value = std::env::var(Self::ENV_VAR).ok();
        Self::from_env_value(value.as_deref())
    }

    /// The `os_log_create` category string for this mode. `Auto` is
    /// `<os/signpost.h>`'s `OS_LOG_CATEGORY_DYNAMIC_TRACING`
    /// (`/Library/Developer/CommandLineTools/SDKs/MacOSX26.2.sdk/usr/include/os/signpost.h:332`):
    /// "signposts emitted to the resulting log handle should be disabled by
    /// default... will only return 'true' when a performance tool like
    /// Instruments.app is recording." `Always` is a plain category with no
    /// such contract, matching what the `xcrun xctrace` CLI workflow needs.
    #[cfg_attr(not(all(feature = "signpost", target_os = "macos")), allow(dead_code))]
    fn category(self) -> &'static str {
        match self {
            SignpostMode::Auto => "DynamicTracing",
            SignpostMode::Always => "decode",
        }
    }
}

#[cfg(all(feature = "signpost", target_os = "macos"))]
mod imp {
    use super::Label;
    use std::ffi::{CString, c_char, c_void};
    use std::sync::OnceLock;

    #[allow(non_camel_case_types)]
    type os_log_t = *mut c_void;
    #[allow(non_camel_case_types)]
    type os_signpost_id_t = u64;

    const OS_SIGNPOST_INTERVAL_BEGIN: u8 = 1;
    const OS_SIGNPOST_INTERVAL_END: u8 = 2;

    unsafe extern "C" {
        fn os_log_create(subsystem: *const c_char, category: *const c_char) -> os_log_t;
        fn os_signpost_id_generate(log: os_log_t) -> os_signpost_id_t;
        fn os_signpost_enabled(log: os_log_t) -> bool;
        static __dso_handle: c_void;

        // The public `os_signpost_interval_begin`/`_end` macros in
        // <os/signpost.h> are Clang-only (they rely on `__builtin_os_log_format`
        // to build the log's argument buffer at compile time, and require
        // `name`/`format` to be string literals placed by the compiler into
        // the `__TEXT,__oslogstring` section). Called from Rust, the
        // equivalent is the underscored-but-exported libsystem_trace entry
        // point those macros expand to for a static, no-format-argument name
        // — the same technique other non-Clang os_signpost bindings use. We
        // supply `name`/`format` pointers into statics placed in that same
        // section ourselves (see the `oslogstring!` statics below) so this
        // satisfies the same compiled-in-binary-string-table contract Clang's
        // macro expansion satisfies; a heap `CString` does not.
        // SDK header (`os/signpost.h:378-380`, all installed CommandLineTools
        // SDKs incl. MacOSX26.2): `void _os_signpost_emit_with_name_impl(void
        // *dso, os_log_t log, os_signpost_type_t type, os_signpost_id_t spid,
        // const char *name, const char *format, uint8_t *buf, uint32_t
        // size);` — `buf` is a writable `uint8_t *`, not `const`. The prior
        // Rust declaration typed it `*const u8` and passed a pointer derived
        // from an immutable `static`/stack array; if the OS implementation
        // ever writes through `buf` (its own log-packet buffer), that was UB
        // over immutable memory. `buf` is declared `*mut u8` here and every
        // call site passes a mutable stack scratch buffer instead.
        fn _os_signpost_emit_with_name_impl(
            dso: *const c_void,
            log: os_log_t,
            signpost_type: u8,
            spid: os_signpost_id_t,
            name: *const c_char,
            format: *const c_char,
            buf: *mut u8,
            size: u32,
        );
    }

    /// Declares a `static` NUL-terminated byte array placed in the
    /// `__TEXT,__oslogstring,cstring_literals` Mach-O section — verified
    /// against this machine's Xcode/CommandLineTools SDK to be the same
    /// section Clang places `os_log`/`os_signpost` string-literal name and
    /// format arguments into (confirmed via `clang -S -emit-llvm` on a
    /// minimal `os_signpost_interval_begin` call: both the name and the
    /// empty-format constant land in a `section "__TEXT,__oslogstring,cstring_literals"`
    /// global). `#[used]` keeps the linker from discarding it as dead data;
    /// only its address (not a load of its contents through normal Rust
    /// code) is ever taken.
    macro_rules! oslogstring {
        ($name:ident, $s:literal) => {
            #[used]
            #[unsafe(link_section = "__TEXT,__oslogstring,cstring_literals")]
            static $name: [u8; $s.len() + 1] = {
                let src = $s.as_bytes();
                let mut buf = [0u8; $s.len() + 1];
                let mut i = 0;
                while i < src.len() {
                    buf[i] = src[i];
                    i += 1;
                }
                buf
            };
        };
    }

    oslogstring!(NAME_DECODE_STEP, "decode.step");
    oslogstring!(NAME_DECODE_CB_COMMIT, "decode.cb_commit");
    oslogstring!(NAME_DECODE_CB_WAIT, "decode.cb_wait");
    oslogstring!(NAME_DECODE_HOST_SCALAR_READ, "decode.host_scalar_read");
    oslogstring!(NAME_DECODE_GRAMMAR_MASK, "decode.grammar_mask");
    oslogstring!(NAME_DECODE_SAMPLE, "decode.sample");
    // Zero-argument log payload: empty format string + a 2-byte header
    // {flags=0, argc=0}, matching what Clang's macro emits when there are no
    // dynamic values to log.
    oslogstring!(EMPTY_FORMAT, "");

    impl Label {
        fn name_ptr(self) -> *const c_char {
            let bytes: &'static [u8] = match self {
                Label::DecodeStep => &NAME_DECODE_STEP,
                Label::DecodeCbCommit => &NAME_DECODE_CB_COMMIT,
                Label::DecodeCbWait => &NAME_DECODE_CB_WAIT,
                Label::DecodeHostScalarRead => &NAME_DECODE_HOST_SCALAR_READ,
                Label::DecodeGrammarMask => &NAME_DECODE_GRAMMAR_MASK,
                Label::DecodeSample => &NAME_DECODE_SAMPLE,
            };
            bytes.as_ptr().cast()
        }
    }

    /// Creates the `os_log_t` handle for an explicit category string. Kept
    /// separate from [`decode_log`] so tests can construct a log for each
    /// `SignpostMode` category directly and assert on `os_signpost_enabled`
    /// without touching process-global environment state (the mode→category
    /// mapping is instead tested in isolation via
    /// `SignpostMode::from_env_value`, which takes an `Option<&str>`).
    ///
    /// This category name is not one of `Label`'s `__TEXT,__oslogstring`
    /// statics: it is passed to `os_log_create`, not to
    /// `_os_signpost_emit_with_name_impl`'s `name`/`format` parameters, so it
    /// does not need compiler-string-table placement — only those two
    /// parameters are read from `__TEXT,__oslogstring` by the trace decoder.
    /// A heap `CString`, one-time-allocated per call, is the same
    /// category-string mechanism the subsystem string already used.
    fn create_log(category: &str) -> usize {
        let subsystem = CString::new("ai.lattice.inference").expect("static subsystem string");
        let category = CString::new(category).expect("signpost category string has no NUL byte");
        // SAFETY: both CStrings outlive this call; os_log_create copies what
        // it needs internally per Apple's os_log contract.
        unsafe { os_log_create(subsystem.as_ptr(), category.as_ptr()) as usize }
    }

    /// `os_signpost_enabled`'s default for the resulting log handle depends
    /// on the category `SignpostMode` selects (see [`super::SignpostMode`]):
    /// `DynamicTracing` (mode `Auto`, default) is idle-inert until an
    /// Instruments-style tool session attaches; the ordinary `"decode"`
    /// category (mode `Always`, opt-in via `LATTICE_SIGNPOST_MODE=always`)
    /// is enabled whenever any collector could observe it, including the
    /// `xcrun xctrace` CLI capture path. Read once and `OnceLock`-cached —
    /// not a per-interval cost.
    fn decode_log() -> os_log_t {
        static LOG: OnceLock<usize> = OnceLock::new();
        let ptr =
            *LOG.get_or_init(|| create_log(super::SignpostMode::from_process_env().category()));
        ptr as os_log_t
    }

    /// Emits without re-checking `os_signpost_enabled` — callers must already
    /// know signposts are enabled (only reached from a live [`Interval`]
    /// whose `begin` observed `os_signpost_enabled(log) == true`).
    fn emit_unchecked(
        log: os_log_t,
        signpost_type: u8,
        spid: os_signpost_id_t,
        name: *const c_char,
    ) {
        // Mutable stack scratch buffer, not a `&'static`/immutable source:
        // the header types `buf` as writable `uint8_t *` (see the SAFETY
        // note above the extern block), so a pointer derived from immutable
        // memory would be UB if the OS implementation ever wrote through it.
        // Zero-argument payload semantics (empty format, 2-byte {flags=0,
        // argc=0} header) are unchanged.
        let mut buf: [u8; 2] = [0, 0];
        // SAFETY: `log` is a live os_log_t from `decode_log()`; `name` and
        // the format pointer are `'static` addresses into the
        // `__TEXT,__oslogstring` section; `buf` is a live mutable stack
        // buffer describing a zero-argument log payload matching what
        // Clang's macro emits with no dynamic values.
        unsafe {
            _os_signpost_emit_with_name_impl(
                &__dso_handle as *const c_void,
                log,
                signpost_type,
                spid,
                name,
                EMPTY_FORMAT.as_ptr().cast(),
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
        }
    }

    pub struct Interval {
        // `None` when `os_signpost_enabled` was false at `begin` time (the
        // common case with tracing off) — `Drop` then does no FFI work at
        // all, rather than repeating the enabled check.
        state: Option<(os_log_t, os_signpost_id_t, Label)>,
    }

    impl Interval {
        pub fn begin(label: Label) -> Self {
            let log = decode_log();
            // First operation: the enabled check. Only on true do we pay for
            // ID generation and emission — statics mean the label lookup
            // itself is always zero-allocation, but we still skip the FFI
            // calls entirely when nothing is recording.
            // SAFETY: `log` is a live os_log_t from `decode_log()`.
            if !unsafe { os_signpost_enabled(log) } {
                return Interval { state: None };
            }
            // SAFETY: `log` is a live os_log_t from `decode_log()`.
            let spid = unsafe { os_signpost_id_generate(log) };
            emit_unchecked(log, OS_SIGNPOST_INTERVAL_BEGIN, spid, label.name_ptr());
            Interval {
                state: Some((log, spid, label)),
            }
        }

        /// Same zero-FFI-cost `state: None` guard `begin` returns when
        /// `os_signpost_enabled` observed false — used for a label that is
        /// suppressed for the current [`super::Scope`] rather than for the
        /// tracing-off case. No `decode_log()`/FFI call at all: the caller
        /// already knows this scope must stay silent.
        pub fn not_recording() -> Self {
            Interval { state: None }
        }
    }

    impl Drop for Interval {
        fn drop(&mut self) {
            if let Some((log, spid, label)) = self.state {
                emit_unchecked(log, OS_SIGNPOST_INTERVAL_END, spid, label.name_ptr());
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // Regression for finding 4 (round-4 review): the two documented
        // capture paths need opposite `os_signpost_enabled` defaults, and
        // both are observable in-process with no Instruments/xctrace tool
        // attached — no external tooling needed to guard this. Constructs
        // logs from an explicit category string (never touches process env)
        // so it can run concurrently with other tests without env races;
        // the env->mode parsing itself is tested separately in the outer
        // `mod tests` via `SignpostMode::from_env_value`.
        #[test]
        fn always_mode_category_is_enabled_with_no_tool_attached() {
            let log = create_log("decode") as os_log_t;
            assert!(
                unsafe { os_signpost_enabled(log) },
                "ordinary category must report enabled even with no tool attached \
                 (mode=always exists precisely so the xcrun xctrace CLI path observes it)"
            );
        }

        #[test]
        fn auto_mode_category_is_idle_inert_with_no_tool_attached() {
            let log = create_log("DynamicTracing") as os_log_t;
            assert!(
                !unsafe { os_signpost_enabled(log) },
                "DynamicTracing category must report disabled with no tool attached \
                 (round 3's idle-inertness property, preserved as the default mode)"
            );
        }
    }
}

#[cfg(not(all(feature = "signpost", target_os = "macos")))]
mod imp {
    use super::Label;

    /// Zero-sized no-op: optimizes away entirely so the default build (feature
    /// off, or non-macOS) is byte-identical to a build with no signpost calls.
    pub struct Interval;

    impl Interval {
        #[inline(always)]
        pub fn begin(_label: Label) -> Self {
            Interval
        }

        #[inline(always)]
        pub fn not_recording() -> Self {
            Interval
        }
    }
}

pub(crate) use imp::Interval;

/// Discriminates which decode-path invocations of the shared per-token
/// forward helper are in the documented autoregressive-decode scope. Only
/// `Scope::Decode` call sites emit `decode.*` signposts; MTP verification,
/// prompt-prefill (multimodal or plain), and diagnostic replays pass
/// `Scope::NotDecode` and stay silent — the same shared helper otherwise
/// reports their work as decode, contradicting the label glossary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    Decode,
    NotDecode,
}

/// Begin a signpost interval for `label`, ending it when the returned guard
/// drops. No-op (zero-sized, inlined away) unless built with `--features
/// signpost` on macOS.
#[inline(always)]
pub(crate) fn interval(label: Label) -> Interval {
    Interval::begin(label)
}

/// Same as [`interval`], but only for `Scope::Decode`; `Scope::NotDecode`
/// returns the same zero-FFI-cost guard a disabled/not-recording build
/// returns, without touching `decode_log()` or any FFI call.
#[inline(always)]
pub(crate) fn interval_in(scope: Scope, label: Label) -> Interval {
    match scope {
        Scope::Decode => Interval::begin(label),
        Scope::NotDecode => Interval::not_recording(),
    }
}

#[cfg(test)]
pub(crate) mod recorder {
    //! Emission-recording seam for tests: records begin/end pairing without
    //! touching the real macOS FFI path, so pairing can be asserted on every
    //! platform/feature combination `cargo test` runs under.
    use super::{Label, Scope};
    use std::cell::RefCell;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Event {
        Begin(Label),
        End(Label),
    }

    thread_local! {
        static LOG: RefCell<Vec<Event>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) fn clear() {
        LOG.with(|log| log.borrow_mut().clear());
    }

    pub(crate) fn events() -> Vec<Event> {
        LOG.with(|log| log.borrow().clone())
    }

    pub(crate) struct RecordingInterval(Option<Label>);

    impl RecordingInterval {
        pub(crate) fn begin(label: Label) -> Self {
            LOG.with(|log| log.borrow_mut().push(Event::Begin(label)));
            RecordingInterval(Some(label))
        }

        /// Mirrors [`super::interval_in`]'s scope gating: `Scope::NotDecode`
        /// records nothing at begin or drop, exactly like the real FFI path's
        /// `Interval::not_recording()`.
        pub(crate) fn begin_in(scope: Scope, label: Label) -> Self {
            match scope {
                Scope::Decode => Self::begin(label),
                Scope::NotDecode => RecordingInterval(None),
            }
        }
    }

    impl Drop for RecordingInterval {
        fn drop(&mut self) {
            if let Some(label) = self.0 {
                LOG.with(|log| log.borrow_mut().push(Event::End(label)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
    // `signpost` feature only) for the category-behavior half of finding 4.
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
    // Breaking this mapping (e.g. always returning "DynamicTracing") is
    // exactly the round-4 finding-4 regression -- see fix-round-4 report
    // for the mutation run.
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
    // finding 1's round-4 fix made that one call site the *only* place any
    // of the five decode loops open a `decode.sample` interval (previously
    // five independent copies, two of which had already drifted with no
    // interval at all). See fix-round-4 report for the mutation run
    // breaking `RecordingInterval::begin` and observing this test fail.
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
    // fail — see fix-round-3 report for the mutation run.
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
}
