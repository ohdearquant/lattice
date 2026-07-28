//! Internal support for tests and measurement targets.
//!
//! Cargo compiles integration tests, benches, examples, and binaries as crates
//! separate from `lattice-inference`, so their shared machine-level guards must
//! cross the library boundary. This module is public only for those repository
//! targets: it is hidden from generated documentation and is not a supported
//! production API.
//!
//! Concurrent Metal work corrupts both timing and numerics: confirmed
//! contention inflated top-k boundary margins roughly threefold and produced
//! false failures (#628, #629). The guard therefore combines a process-local
//! mutex with the fleet-wide advisory file lock. Mutex-before-file acquisition
//! keeps at most one thread per process contending for the machine lock, and
//! the returned guard owns both for its full lifetime.

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

const GPU_MACHINE_LOCK_PATH: &str = "/tmp/lion-metal-gpu-test.lock";
const GPU_MACHINE_LOCK_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const GPU_MACHINE_LOCK_POLL_INTERVAL: Duration = Duration::from_millis(500);

static GPU_LOCK: Mutex<()> = Mutex::new(());

struct GpuTestGuard {
    _process: MutexGuard<'static, ()>,
    _machine: File,
}

/// Serialize a GPU-driving test or measurement on the shared Metal device.
///
/// The returned opaque guard must remain in scope for the entire GPU operation.
/// Acquisition waits for at most 30 minutes, then panics with the fleet lock
/// path and an `lsof` diagnostic rather than hanging indefinitely.
#[doc(hidden)]
#[must_use = "the guard must remain in scope for the entire Metal operation"]
pub fn gpu_test_lock() -> impl Sized {
    let process = GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let file = acquire_machine_lock(
        Path::new(GPU_MACHINE_LOCK_PATH),
        GPU_MACHINE_LOCK_TIMEOUT,
        GPU_MACHINE_LOCK_POLL_INTERVAL,
    );

    GpuTestGuard {
        _process: process,
        _machine: file,
    }
}

fn acquire_machine_lock(lock_path: &Path, timeout: Duration, poll_interval: Duration) -> File {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(lock_path)
        .unwrap_or_else(|e| panic!("gpu_test_lock: cannot open {}: {e}", lock_path.display()));
    let deadline = Instant::now() + timeout;
    loop {
        match file.try_lock() {
            Ok(()) => break,
            Err(std::fs::TryLockError::WouldBlock) => {
                if Instant::now() >= deadline {
                    panic!(
                        "gpu_test_lock: another process has held \
                         {} for over {}s — a Metal \
                         test run elsewhere on this machine is wedged or \
                         genuinely that long; inspect `lsof {}`",
                        lock_path.display(),
                        timeout.as_secs(),
                        lock_path.display()
                    );
                }
                std::thread::sleep(poll_interval);
            }
            Err(std::fs::TryLockError::Error(e)) => {
                panic!(
                    "gpu_test_lock: flock on {} failed: {e}",
                    lock_path.display()
                )
            }
        }
    }
    file
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machine_lock_excludes_another_process() {
        const CHILD_ENV: &str = "LATTICE_GPU_LOCK_TEST_CHILD";
        const READY_ENV: &str = "LATTICE_GPU_LOCK_TEST_READY";
        const LOCK_ENV: &str = "LATTICE_GPU_LOCK_TEST_PATH";

        if std::env::var_os(CHILD_ENV).is_some() {
            let ready = std::env::var_os(READY_ENV).expect("child ready path");
            let lock_path = std::env::var_os(LOCK_ENV).expect("child lock path");
            std::fs::write(ready, b"ready").expect("publish child ready marker");
            let _guard = acquire_machine_lock(
                Path::new(&lock_path),
                Duration::from_secs(5),
                Duration::from_millis(10),
            );
            return;
        }

        let temp = tempfile::tempdir().expect("temporary lock-test directory");
        let lock_path = temp.path().join("machine-lock");
        let guard = acquire_machine_lock(
            &lock_path,
            Duration::from_secs(5),
            Duration::from_millis(10),
        );
        let ready = temp.path().join("child-ready");
        let mut child = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "measurement::tests::machine_lock_excludes_another_process",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .env(READY_ENV, &ready)
            .env(LOCK_ENV, &lock_path)
            .spawn()
            .expect("spawn lock contender");

        let ready_deadline = Instant::now() + Duration::from_secs(5);
        while !ready.exists() {
            if Instant::now() >= ready_deadline {
                child.kill().expect("kill unready lock contender");
                child.wait().expect("reap unready lock contender");
                panic!("child did not reach the lock acquisition");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            child.try_wait().expect("poll lock contender").is_none(),
            "a second process acquired the machine lock while the parent held it"
        );

        drop(guard);
        let exit_deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(status) = child.try_wait().expect("poll released contender") {
                assert!(status.success(), "lock contender failed after release");
                break;
            }
            if Instant::now() >= exit_deadline {
                child.kill().expect("kill wedged lock contender");
                child.wait().expect("reap wedged lock contender");
                panic!("lock contender did not acquire after the parent released");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
