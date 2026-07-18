//! Machine-wide Metal GPU test serialization, exposed through the `test-utils` seam (PR #1021
//! review round 6, issue 6) so a separate-compilation-unit integration test (a `tests/*.rs`
//! binary, which cannot see this crate's own `#[cfg(test)]`-only internals -- see
//! `model::qwen35::test_support`'s doc comment for the same cross-compilation-unit reasoning)
//! can serialize against the exact same lock every in-crate Metal test uses, instead of
//! maintaining an independent copy of the lock path, timeout, and polling protocol.
//!
//! Mirrors `forward::metal_qwen35`'s private `inner::tests::gpu_test_lock` byte for byte
//! (same path, same timeout, same advisory-`flock`-with-bounded-poll protocol) -- that helper
//! stays private to its own `#[cfg(test)]` module (it is exercised by every in-crate Metal
//! test already); this is the one place both sides of the machine-wide convention are defined,
//! so a future change to the protocol cannot update one copy and silently leave the other
//! stale.

use std::fs::File;
use std::time::{Duration, Instant};

const GPU_MACHINE_LOCK_PATH: &str = "/tmp/lion-metal-gpu-test.lock";
const GPU_MACHINE_LOCK_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Held for the guard's lifetime; dropping the `File` closes the fd, releasing the flock.
pub struct GpuTestLockGuard {
    _file: File,
}

/// Acquire the machine-wide advisory `flock` on `/tmp/lion-metal-gpu-test.lock`. Blocks (polling
/// `try_lock`) for up to 30 minutes, then panics with an `lsof` hint rather than hanging
/// silently -- matching `forward::metal_qwen35`'s in-crate `gpu_test_lock()` exactly.
pub fn gpu_test_lock() -> GpuTestLockGuard {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(GPU_MACHINE_LOCK_PATH)
        .unwrap_or_else(|e| panic!("gpu_test_lock: cannot open {GPU_MACHINE_LOCK_PATH}: {e}"));
    let deadline = Instant::now() + GPU_MACHINE_LOCK_TIMEOUT;
    loop {
        match file.try_lock() {
            Ok(()) => break,
            Err(std::fs::TryLockError::WouldBlock) => {
                if Instant::now() >= deadline {
                    panic!(
                        "gpu_test_lock: another process has held {GPU_MACHINE_LOCK_PATH} for \
                         over {}s -- inspect `lsof {GPU_MACHINE_LOCK_PATH}`",
                        GPU_MACHINE_LOCK_TIMEOUT.as_secs()
                    );
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(std::fs::TryLockError::Error(e)) => {
                panic!("gpu_test_lock: flock on {GPU_MACHINE_LOCK_PATH} failed: {e}")
            }
        }
    }
    GpuTestLockGuard { _file: file }
}
