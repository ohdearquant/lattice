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
        fn _os_signpost_emit_with_name_impl(
            dso: *const c_void,
            log: os_log_t,
            signpost_type: u8,
            spid: os_signpost_id_t,
            name: *const c_char,
            format: *const c_char,
            buf: *const u8,
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

    /// The category string is `<os/signpost.h>`'s `OS_LOG_CATEGORY_DYNAMIC_TRACING`:
    /// ```c
    /// #define OS_LOG_CATEGORY_DYNAMIC_TRACING "DynamicTracing"
    /// ```
    /// (`/Library/Developer/CommandLineTools/SDKs/MacOSX26.2.sdk/usr/include/os/signpost.h:332`,
    /// under the header's own `@const OS_LOG_CATEGORY_DYNAMIC_TRACING` doc comment:
    /// "signposts emitted to the resulting log handle should be disabled by
    /// default, reducing the runtime overhead. os_signpost_enabled calls on the
    /// resulting log handle will only return 'true' when a performance tool
    /// like Instruments.app is recording.") A plain `"decode"` category (the
    /// prior value) has no such contract — `os_signpost_enabled` on it can be
    /// `true` with no tool attached, so every decode interval paid ID
    /// generation plus begin/end FFI even in an idle signpost-enabled build.
    /// This category name is not one of `Label`'s `__TEXT,__oslogstring`
    /// statics: it is passed to `os_log_create`, not to
    /// `_os_signpost_emit_with_name_impl`'s `name`/`format` parameters, so it
    /// does not need compiler-string-table placement — only those two
    /// parameters are read from `__TEXT,__oslogstring` by the trace decoder.
    /// A heap `CString`, one-time-allocated and `OnceLock`-cached below (not
    /// a per-interval cost), is the same category-string mechanism the
    /// subsystem string already used before this fix.
    fn decode_log() -> os_log_t {
        static LOG: OnceLock<usize> = OnceLock::new();
        let ptr = *LOG.get_or_init(|| {
            let subsystem = CString::new("ai.lattice.inference").expect("static subsystem string");
            let category =
                CString::new("DynamicTracing").expect("static dynamic-tracing category string");
            // SAFETY: both CStrings outlive this call; os_log_create copies
            // what it needs internally per Apple's os_log contract. This is a
            // one-time (OnceLock-cached) allocation, not a per-interval cost;
            // subsystem/category are not the compile-time-provenance strings
            // the emit-impl's name/format arguments must be.
            unsafe { os_log_create(subsystem.as_ptr(), category.as_ptr()) as usize }
        });
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
        let buf: [u8; 2] = [0, 0];
        // SAFETY: `log` is a live os_log_t from `decode_log()`; `name` and
        // the format pointer are `'static` addresses into the
        // `__TEXT,__oslogstring` section; `buf` describes a zero-argument
        // log payload matching what Clang's macro emits with no dynamic
        // values.
        unsafe {
            _os_signpost_emit_with_name_impl(
                &__dso_handle as *const c_void,
                log,
                signpost_type,
                spid,
                name,
                EMPTY_FORMAT.as_ptr().cast(),
                buf.as_ptr(),
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
    use super::interval;
    use super::recorder::{Event, RecordingInterval};

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
