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

use std::ffi::{OsStr, OsString};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::{AsFd, AsRawFd};
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

const GPU_MACHINE_LOCK_PATH: &str = "/tmp/lion-metal-gpu-test.lock";
const GPU_MACHINE_LOCK_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const GPU_MACHINE_LOCK_POLL_INTERVAL: Duration = Duration::from_millis(500);
// Set by scripts/lib/bench-locks.py and scripts/lib/bench_supervision.py on
// every launch route where a supervisor holds this lock and does not hand it
// to the child (#1643). Read only on the non-handoff path below: it never
// substitutes for a real handoff, only tells a native acquisition that
// waiting out GPU_MACHINE_LOCK_TIMEOUT is pointless, because the holder is
// our own supervisor rather than an unrelated benchmark elsewhere.
const SUPERVISOR_MARKER_ENV: &str = "LATTICE_GPU_LOCK_SUPERVISOR_PID";
const HANDOFF_TIMEOUT: Duration = Duration::from_secs(120);
const HANDOFF_FRAME_LIMIT: usize = 4096;
const HANDOFF_ENV: [&str; 3] = [
    "LATTICE_GPU_HANDOFF_FD",
    "LATTICE_GPU_HANDOFF_CONTROL",
    "LATTICE_GPU_HANDOFF_TOKEN",
];

#[cfg(test)]
mod lock_file_refresh_tests;

static GPU_LOCK: Mutex<()> = Mutex::new(());

struct GpuTestGuard {
    _process: MutexGuard<'static, ()>,
    // Explicit unlock on drop would also release the supervisor's shared lock.
    _machine: File,
}

#[derive(Debug)]
struct HandoffConfig {
    control: PathBuf,
    token: String,
}

#[derive(serde::Serialize)]
struct HandoffRequest<'a> {
    protocol: u8,
    token: &'a str,
    pid: u32,
    exe: PathBuf,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct HandoffReady {
    protocol: u8,
    token: String,
    status: String,
}

/// Serialize a GPU-driving test or measurement on the shared Metal device.
///
/// The returned opaque guard must remain in scope for the entire GPU operation.
/// Ordinary acquisition waits for at most 30 minutes, unless a
/// `LATTICE_GPU_LOCK_SUPERVISOR_PID` marker names a live process already
/// holding the lock without a handoff, in which case it panics immediately
/// (#1643) rather than waiting out a timeout it can never win. Explicitly
/// admitted benchmarks retain the supervisor's shared lock description and
/// wait for its invocation acknowledgement before measurement; invalid
/// admission panics.
#[doc(hidden)]
#[must_use = "the guard must remain in scope for the entire Metal operation"]
pub fn gpu_test_lock() -> impl Sized {
    gpu_test_lock_for_path(
        Path::new(GPU_MACHINE_LOCK_PATH),
        HANDOFF_ENV.map(std::env::var_os),
        std::env::var_os(SUPERVISOR_MARKER_ENV),
    )
}

fn gpu_test_lock_for_path(
    lock_path: &Path,
    handoff_signals: [Option<OsString>; 3],
    supervisor_marker: Option<OsString>,
) -> GpuTestGuard {
    let process = GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let handoff = handoff_config(handoff_signals)
        .unwrap_or_else(|error| panic!("gpu_test_lock: refusing GPU handoff: {error}"));
    let file = match handoff {
        Some(config) => accept_handoff(&config, lock_path)
            .unwrap_or_else(|error| panic!("gpu_test_lock: refusing GPU handoff: {error}")),
        None => acquire_machine_lock(
            lock_path,
            GPU_MACHINE_LOCK_TIMEOUT,
            GPU_MACHINE_LOCK_POLL_INTERVAL,
            supervisor_marker,
        ),
    };

    GpuTestGuard {
        _process: process,
        _machine: file,
    }
}

fn handoff_config(values: [Option<OsString>; 3]) -> Result<Option<HandoffConfig>, String> {
    if values.iter().all(Option::is_none) {
        return Ok(None);
    }
    let [Some(fd), Some(control), Some(token)] = values else {
        return Err("all three GPU handoff variables are required".into());
    };
    if fd.as_os_str() != OsStr::new("0") {
        return Err("GPU handoff descriptor must be standard input (0)".into());
    }
    let control = PathBuf::from(control);
    if !control.is_absolute() {
        return Err("GPU handoff control path must be absolute".into());
    }
    let token = token
        .into_string()
        .map_err(|_| "GPU handoff token must be UTF-8".to_owned())?;
    if token.len() != 32 || !token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("GPU handoff token must contain exactly 32 hexadecimal digits".into());
    }
    Ok(Some(HandoffConfig { control, token }))
}

fn accept_handoff(config: &HandoffConfig, lock_path: &Path) -> Result<File, String> {
    let stdin = std::io::stdin();
    let descriptor = stdin
        .as_fd()
        .try_clone_to_owned()
        .map_err(|error| format!("cannot duplicate GPU handoff descriptor: {error}"))?;
    let file = File::from(descriptor);
    verify_handoff_lock(&file, lock_path)?;
    refresh_lock_file(&file)
        .map_err(|error| format!("cannot refresh GPU handoff lock timestamps: {error}"))?;
    wait_for_handoff_ready(config, HANDOFF_TIMEOUT)?;
    verify_handoff_lock(&file, lock_path)?;
    Ok(file)
}

fn lock_identity(file: &File, lock_path: &Path) -> Result<(u64, u64), String> {
    let held = file
        .metadata()
        .map_err(|error| format!("cannot inspect GPU handoff descriptor: {error}"))?;
    let canonical = std::fs::metadata(lock_path)
        .map_err(|error| format!("cannot inspect canonical GPU lock: {error}"))?;
    if !held.is_file() || !canonical.is_file() {
        return Err("GPU handoff and canonical lock must name a regular file".into());
    }
    let identity = (held.dev(), held.ino());
    if identity != (canonical.dev(), canonical.ino()) {
        return Err("GPU handoff descriptor does not match the canonical lock inode".into());
    }
    Ok(identity)
}

fn require_contention(file: &File, phase: &str) -> Result<(), String> {
    match file.try_lock() {
        Err(std::fs::TryLockError::WouldBlock) => Ok(()),
        Ok(()) => Err(format!("canonical GPU lock is not held {phase}")),
        Err(std::fs::TryLockError::Error(error)) => {
            Err(format!("cannot probe canonical GPU lock {phase}: {error}"))
        }
    }
}

fn verify_handoff_lock(file: &File, lock_path: &Path) -> Result<(), String> {
    let identity = lock_identity(file, lock_path)?;
    let probe = OpenOptions::new()
        .read(true)
        .write(true)
        .open(lock_path)
        .map_err(|error| format!("cannot open canonical GPU lock for verification: {error}"))?;
    if lock_identity(&probe, lock_path)? != identity {
        return Err("canonical GPU lock changed before handoff verification".into());
    }
    // Reversing these probes would let an unheld descriptor create its own evidence.
    require_contention(&probe, "before handoff")?;
    match file.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => {
            return Err("GPU handoff descriptor is not the exclusive lock owner".into());
        }
        Err(std::fs::TryLockError::Error(error)) => {
            return Err(format!("cannot verify shared GPU lock ownership: {error}"));
        }
    }
    require_contention(&probe, "after handoff")?;
    if lock_identity(file, lock_path)? != identity || lock_identity(&probe, lock_path)? != identity
    {
        return Err("canonical GPU lock changed during handoff verification".into());
    }
    // This is a retained capability, not authentication or proof of past ownership.
    // Explicit unlock on any duplicate would release the supervisor's lock too.
    Ok(())
}

fn wait_for_handoff_ready(config: &HandoffConfig, timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    let request = HandoffRequest {
        protocol: 1,
        token: &config.token,
        pid: std::process::id(),
        exe: std::env::current_exe()
            .map_err(|error| format!("cannot identify GPU handoff executable: {error}"))?,
    };
    let mut frame = serde_json::to_vec(&request)
        .map_err(|error| format!("cannot encode GPU handoff request: {error}"))?;
    frame.push(b'\n');
    if frame.len() > HANDOFF_FRAME_LIMIT {
        return Err("GPU handoff request exceeds the frame limit".into());
    }
    let path = config.control.clone();
    let mut control = connect_handoff_with(move || UnixStream::connect(path), deadline)?;
    write_handoff_frame(&mut control, &frame, deadline)?;
    let response = read_handoff_frame(&mut control, deadline)?;
    let ready: HandoffReady = serde_json::from_slice(&response)
        .map_err(|_| "malformed GPU handoff acknowledgement".to_owned())?;
    if ready.protocol != 1 || ready.token != config.token || ready.status != "ready" {
        return Err("GPU handoff acknowledgement did not admit this invocation".into());
    }
    Ok(())
}

fn handoff_remaining(deadline: Instant) -> Result<Duration, String> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err("GPU handoff deadline expired".into());
    }
    Ok(remaining)
}

fn connect_handoff_with<F>(connector: F, deadline: Instant) -> Result<UnixStream, String>
where
    F: FnOnce() -> std::io::Result<UnixStream> + Send + 'static,
{
    let (sender, receiver) = std::sync::mpsc::channel();
    // A pending connect may outlive the timeout until process exit. The worker
    // owns no lock descriptor, and its late socket closes when sending fails.
    let _worker = std::thread::Builder::new()
        .name("gpu-handoff-connect".into())
        .spawn(move || {
            let _ = sender.send(connector());
        })
        .map_err(|error| format!("cannot start GPU handoff connector: {error}"))?;
    receiver
        .recv_timeout(handoff_remaining(deadline)?)
        .map_err(|error| format!("GPU handoff connection did not complete: {error}"))?
        .map_err(|error| format!("cannot connect to GPU handoff supervisor: {error}"))
}

fn write_handoff_frame(
    control: &mut UnixStream,
    mut frame: &[u8],
    deadline: Instant,
) -> Result<(), String> {
    control
        .set_nonblocking(true)
        .map_err(|error| format!("cannot make GPU handoff writes nonblocking: {error}"))?;
    while !frame.is_empty() {
        handoff_remaining(deadline)?;
        match control.write(frame) {
            Ok(0) => return Err("GPU handoff closed while sending its request".into()),
            Ok(count) => frame = &frame[count..],
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                wait_for_handoff_io(deadline)?;
            }
            Err(error) => return Err(format!("cannot send GPU handoff request: {error}")),
        }
    }
    Ok(())
}

fn read_handoff_frame(control: &mut UnixStream, deadline: Instant) -> Result<Vec<u8>, String> {
    // Darwin rejects SO_RCVTIMEO changes after peer closure, even with a complete
    // acknowledgement buffered. Nonblocking reads can still drain that frame.
    control
        .set_nonblocking(true)
        .map_err(|error| format!("cannot make GPU handoff reads nonblocking: {error}"))?;
    let mut frame = Vec::with_capacity(256);
    while frame.len() < HANDOFF_FRAME_LIMIT {
        handoff_remaining(deadline)?;
        let mut byte = [0];
        match control.read(&mut byte) {
            Ok(0) => return Err("GPU handoff closed before a complete acknowledgement".into()),
            Ok(_) => {
                frame.push(byte[0]);
                if byte[0] == b'\n' {
                    return Ok(frame);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                wait_for_handoff_io(deadline)?;
            }
            Err(error) => return Err(format!("cannot read GPU handoff acknowledgement: {error}")),
        }
    }
    Err("GPU handoff acknowledgement exceeds the frame limit".into())
}

fn wait_for_handoff_io(deadline: Instant) -> Result<(), String> {
    std::thread::sleep(handoff_remaining(deadline)?.min(Duration::from_millis(5)));
    Ok(())
}

/// The pid `marker` names, if it parses to a positive pid and that pid is
/// still alive. Absent, unparsable, non-positive, or naming a pid that
/// `libc::kill(pid, 0)` reports as gone all return `None`: every one of those
/// is a reason to fall back to today's bounded wait rather than to refuse, so
/// the caller does not need to distinguish them.
fn live_supervisor_pid(marker: Option<&OsStr>) -> Option<libc::pid_t> {
    let pid: libc::pid_t = marker?.to_str()?.trim().parse().ok()?;
    if pid <= 0 {
        return None;
    }
    // SAFETY: signal 0 sends nothing; `kill` with it only probes whether
    // `pid` names a process we could otherwise signal, which is exactly the
    // liveness question here and has no other precondition.
    (unsafe { libc::kill(pid, 0) } == 0).then_some(pid)
}

fn refresh_lock_file(file: &File) -> std::io::Result<()> {
    // On macOS, `set_times` and `futimens` require ownership even for the current time;
    // `futimes` permits write access to refresh a lock file owned by another user.
    // SAFETY: `file` owns a valid descriptor for this call, and a null `times` pointer is the
    // documented form for setting both timestamps to now.
    if unsafe { libc::futimes(file.as_raw_fd(), std::ptr::null()) } != 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn acquire_machine_lock(
    lock_path: &Path,
    timeout: Duration,
    poll_interval: Duration,
    supervisor_marker: Option<OsString>,
) -> File {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(lock_path)
        .unwrap_or_else(|e| panic!("gpu_test_lock: cannot open {}: {e}", lock_path.display()));
    // Refresh the opened inode even while another process holds it; do not
    // replace the path and split contenders across different lock files.
    refresh_lock_file(&file).unwrap_or_else(|e| {
        panic!(
            "gpu_test_lock: cannot refresh timestamps for {}: {e}",
            lock_path.display()
        )
    });
    match file.try_lock() {
        Ok(()) => return file,
        Err(std::fs::TryLockError::WouldBlock) => {
            if let Some(pid) = live_supervisor_pid(supervisor_marker.as_deref()) {
                panic!(
                    "gpu_test_lock: the benchmark supervisor (pid {pid}) running this \
                     command holds {} and did not hand it to this process; this target \
                     takes the GPU lock itself, so it can never acquire it here. Run it \
                     through the admitted `--gpu-handoff` route for cargo bench targets, \
                     or see the guidance in CLAUDE.md for other targets.",
                    lock_path.display()
                );
            }
        }
        Err(std::fs::TryLockError::Error(e)) => {
            panic!(
                "gpu_test_lock: flock on {} failed: {e}",
                lock_path.display()
            )
        }
    }
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
mod tests;
