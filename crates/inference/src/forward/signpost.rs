//! `os_signpost` instrumentation for the Metal decode path (feature `signpost`,
//! default OFF).
//!
//! Every call site is an RAII interval guard: `signpost::interval("label")`
//! begins an `os_signpost` interval and ends it when the guard drops. When the
//! `signpost` feature is disabled, or the target isn't macOS, [`interval`]
//! returns a zero-sized no-op type and the call compiles to nothing — the
//! call sites in the decode path never need a `#[cfg(...)]` of their own.
//!
//! See `docs/metal-trace.md` for how to capture a Metal System Trace with
//! these signposts and the label glossary.

#[cfg(all(feature = "signpost", target_os = "macos"))]
mod imp {
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
        // to build the log's argument buffer at compile time). Called from
        // Rust, the equivalent is the underscored-but-exported libsystem_trace
        // entry point those macros expand to for a static, no-format-argument
        // name — the same technique other non-Clang os_signpost bindings use.
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

    fn decode_log() -> os_log_t {
        static LOG: OnceLock<usize> = OnceLock::new();
        let ptr = *LOG.get_or_init(|| {
            let subsystem = CString::new("ai.lattice.inference").expect("static subsystem string");
            let category = CString::new("decode").expect("static category string");
            // SAFETY: both CStrings outlive this call; os_log_create copies
            // what it needs internally per Apple's os_log contract.
            unsafe { os_log_create(subsystem.as_ptr(), category.as_ptr()) as usize }
        });
        ptr as os_log_t
    }

    fn emit(log: os_log_t, signpost_type: u8, spid: os_signpost_id_t, name: &CString) {
        // SAFETY: `log` is a live os_log_t from `decode_log()`; `name` is a
        // valid NUL-terminated C string held on the stack for this call;
        // `format`/`buf` describe a zero-argument log payload (empty format
        // string, 2-byte header {flags=0, argc=0}), matching what Clang's
        // macro emits when there are no dynamic values to log.
        unsafe {
            if !os_signpost_enabled(log) {
                return;
            }
            let format = c"";
            let buf: [u8; 2] = [0, 0];
            _os_signpost_emit_with_name_impl(
                &__dso_handle as *const c_void,
                log,
                signpost_type,
                spid,
                name.as_ptr(),
                format.as_ptr(),
                buf.as_ptr(),
                buf.len() as u32,
            );
        }
    }

    pub struct Interval {
        log: os_log_t,
        spid: os_signpost_id_t,
        name: CString,
    }

    impl Interval {
        pub fn begin(label: &str) -> Self {
            let log = decode_log();
            let name = CString::new(label).unwrap_or_else(|_| {
                CString::new("invalid-signpost-label").expect("static fallback string")
            });
            // SAFETY: `log` is a live os_log_t from `decode_log()`.
            let spid = unsafe { os_signpost_id_generate(log) };
            emit(log, OS_SIGNPOST_INTERVAL_BEGIN, spid, &name);
            Interval { log, spid, name }
        }
    }

    impl Drop for Interval {
        fn drop(&mut self) {
            emit(self.log, OS_SIGNPOST_INTERVAL_END, self.spid, &self.name);
        }
    }
}

#[cfg(not(all(feature = "signpost", target_os = "macos")))]
mod imp {
    /// Zero-sized no-op: optimizes away entirely so the default build (feature
    /// off, or non-macOS) is byte-identical to a build with no signpost calls.
    pub struct Interval;

    impl Interval {
        #[inline(always)]
        pub fn begin(_label: &str) -> Self {
            Interval
        }
    }
}

pub use imp::Interval;

/// Begin a signpost interval named `label`, ending it when the returned guard
/// drops. No-op (zero-sized, inlined away) unless built with `--features
/// signpost` on macOS.
#[inline(always)]
pub fn interval(label: &str) -> Interval {
    Interval::begin(label)
}

#[cfg(test)]
mod tests {
    use super::interval;

    // Exercises the exact call-site shape used in the decode path (bind a
    // guard, let it drop at end of scope). Compiles and runs identically
    // whether this crate is built with `--features signpost` (a real
    // interval on macOS) or without it (the no-op fallback) -- there is no
    // separate call-site variant to test per configuration.
    #[test]
    fn interval_guard_compiles_and_drops_cleanly() {
        let _guard = interval("decode.step");
        {
            let _nested = interval("decode.cb_commit");
        }
    }
}
