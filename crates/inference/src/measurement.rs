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
use std::os::fd::AsFd;
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

const GPU_MACHINE_LOCK_PATH: &str = "/tmp/lion-metal-gpu-test.lock";
const GPU_MACHINE_LOCK_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const GPU_MACHINE_LOCK_POLL_INTERVAL: Duration = Duration::from_millis(500);
const HANDOFF_TIMEOUT: Duration = Duration::from_secs(120);
const HANDOFF_FRAME_LIMIT: usize = 4096;
const HANDOFF_ENV: [&str; 3] = [
    "LATTICE_GPU_HANDOFF_FD",
    "LATTICE_GPU_HANDOFF_CONTROL",
    "LATTICE_GPU_HANDOFF_TOKEN",
];

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
/// Ordinary acquisition waits for at most 30 minutes. Explicitly admitted
/// benchmarks retain the supervisor's shared lock description and wait for its
/// invocation acknowledgement before measurement; invalid admission panics.
#[doc(hidden)]
#[must_use = "the guard must remain in scope for the entire Metal operation"]
pub fn gpu_test_lock() -> impl Sized {
    gpu_test_lock_for_path(
        Path::new(GPU_MACHINE_LOCK_PATH),
        HANDOFF_ENV.map(std::env::var_os),
    )
}

fn gpu_test_lock_for_path(
    lock_path: &Path,
    handoff_signals: [Option<OsString>; 3],
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

    #[test]
    fn handoff_signal_requires_complete_valid_metadata() {
        assert!(handoff_config([None, None, None]).unwrap().is_none());
        let complete = [
            Some(OsString::from("0")),
            Some(OsString::from("/tmp/lattice-test-control.sock")),
            Some(OsString::from("0123456789abcdef0123456789abcdef")),
        ];
        for mask in 1..7 {
            let partial = std::array::from_fn(|index| {
                if mask & (1 << index) != 0 {
                    complete[index].clone()
                } else {
                    None
                }
            });
            assert!(handoff_config(partial).is_err(), "partial mask {mask}");
        }
        assert!(handoff_config(complete.clone()).unwrap().is_some());
        for (index, invalid) in [
            (0, ""),
            (0, "3"),
            (0, "00"),
            (1, ""),
            (1, "relative.sock"),
            (2, ""),
            (2, "short"),
            (2, "g123456789abcdef0123456789abcdef"),
        ] {
            let mut values = complete.clone();
            values[index] = Some(OsString::from(invalid));
            assert!(handoff_config(values).is_err(), "invalid field {index}");
        }
    }

    #[test]
    fn handoff_proof_discriminates_lock_ownership() {
        const CASE_ENV: &str = "LATTICE_GPU_PROOF_TEST_CASE";
        const PATH_ENV: &str = "LATTICE_GPU_PROOF_TEST_PATH";
        if let Some(case) = std::env::var_os(CASE_ENV) {
            let path = PathBuf::from(std::env::var_os(PATH_ENV).unwrap());
            let stdin = std::io::stdin();
            let proof = File::from(stdin.as_fd().try_clone_to_owned().unwrap());
            let result = verify_handoff_lock(&proof, &path);
            if case == "owned" {
                result.unwrap();
            } else {
                assert!(result.is_err(), "unexpected admission for {case:?}");
            }
            return;
        }

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("lock");
        let owner = acquire_machine_lock(&path, Duration::from_secs(5), Duration::from_millis(10));
        let independent = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let wrong = File::create(temp.path().join("wrong-inode")).unwrap();
        let run = |case: &str, file: &File| {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "measurement::tests::handoff_proof_discriminates_lock_ownership",
                    "--nocapture",
                ])
                .env(CASE_ENV, case)
                .env(PATH_ENV, &path)
                .stdin(std::process::Stdio::from(file.try_clone().unwrap()))
                .status()
                .unwrap();
            assert!(status.success(), "handoff proof case {case} failed");
        };
        run("owned", &owner);
        assert!(matches!(
            independent.try_lock(),
            Err(std::fs::TryLockError::WouldBlock)
        ));
        run("unrelated", &independent);
        run("wrong-inode", &wrong);
        drop(owner);
        run("unheld", &independent);
        independent.try_lock().unwrap();
    }

    #[test]
    fn handoff_readiness_waits_for_matching_acknowledgement() {
        let temp = tempfile::tempdir().unwrap();
        let config = HandoffConfig {
            control: temp.path().join("ready.sock"),
            token: "0123456789abcdef0123456789abcdef".into(),
        };
        let listener = std::os::unix::net::UnixListener::bind(&config.control).unwrap();
        let token = config.token.clone();
        let (seen_tx, seen_rx) = std::sync::mpsc::channel();
        let (allow_tx, allow_rx) = std::sync::mpsc::channel();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let frame =
                read_handoff_frame(&mut stream, Instant::now() + Duration::from_secs(5)).unwrap();
            let request: serde_json::Value = serde_json::from_slice(&frame).unwrap();
            assert_eq!(request["protocol"], 1);
            assert_eq!(request["token"], token);
            assert_eq!(request["pid"], std::process::id());
            assert_eq!(
                PathBuf::from(request["exe"].as_str().unwrap()),
                std::env::current_exe().unwrap()
            );
            seen_tx.send(()).unwrap();
            allow_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let response = serde_json::json!({"protocol": 1, "token": token, "status": "ready"});
            writeln!(stream, "{response}").unwrap();
        });
        let target =
            std::thread::spawn(move || wait_for_handoff_ready(&config, Duration::from_secs(5)));
        seen_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            !target.is_finished(),
            "target entered before acknowledgement"
        );
        allow_tx.send(()).unwrap();
        target.join().unwrap().unwrap();
        server.join().unwrap();
    }

    #[test]
    fn handoff_readiness_rejects_invalid_acknowledgements() {
        let token = "0123456789abcdef0123456789abcdef";
        let responses = [
            format!("{{\"protocol\":2,\"token\":\"{token}\",\"status\":\"ready\"}}\n"),
            "{\"protocol\":1,\"token\":\"ffffffffffffffffffffffffffffffff\",\"status\":\"ready\"}\n".into(),
            format!("{{\"protocol\":1,\"token\":\"{token}\",\"status\":\"refused\"}}\n"),
            format!("{{\"protocol\":1,\"token\":\"{token}\",\"status\":\"ready\",\"extra\":1}}\n"),
            "{\"protocol\":1}\n".into(),
            "not-json\n".into(),
            "".into(),
            format!("{{\"protocol\":1,\"token\":\"{token}\",\"status\":\"ready\"}}"),
            format!("{}\n", "x".repeat(HANDOFF_FRAME_LIMIT)),
        ];
        for response in responses {
            let temp = tempfile::tempdir().unwrap();
            let config = HandoffConfig {
                control: temp.path().join("invalid.sock"),
                token: token.into(),
            };
            let listener = std::os::unix::net::UnixListener::bind(&config.control).unwrap();
            let server = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                read_handoff_frame(&mut stream, Instant::now() + Duration::from_secs(5)).unwrap();
                let _ = stream.write_all(response.as_bytes());
            });
            assert!(wait_for_handoff_ready(&config, Duration::from_secs(5)).is_err());
            server.join().unwrap();
        }
    }

    #[test]
    fn handoff_readiness_deadline_is_enforced() {
        let (mut reader, writer) = UnixStream::pair().unwrap();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let peer = std::thread::spawn(move || {
            let _ = release_rx.recv_timeout(Duration::from_millis(500));
            drop(writer);
        });
        let error = read_handoff_frame(&mut reader, Instant::now() + Duration::from_millis(20))
            .unwrap_err();
        release_tx.send(()).unwrap();
        assert!(
            !error.contains("closed before"),
            "deadline did not precede EOF"
        );
        peer.join().unwrap();
    }

    #[test]
    fn handoff_readiness_accepts_buffered_frame_after_peer_close() {
        let (mut reader, mut writer) = UnixStream::pair().unwrap();
        let token = "0123456789abcdef0123456789abcdef";
        let response = serde_json::json!({"protocol": 1, "token": token, "status": "ready"});
        writeln!(writer, "{response}").unwrap();
        drop(writer);
        let frame =
            read_handoff_frame(&mut reader, Instant::now() + Duration::from_secs(5)).unwrap();
        let ready: HandoffReady = serde_json::from_slice(&frame).unwrap();
        assert_eq!(ready.protocol, 1);
        assert_eq!(ready.token, token);
        assert_eq!(ready.status, "ready");
    }

    #[test]
    fn handoff_connection_timeout_closes_a_late_socket() {
        let (client, mut peer) = UnixStream::pair().unwrap();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let result = connect_handoff_with(
            move || {
                release_rx.recv_timeout(Duration::from_millis(500)).unwrap();
                Ok(client)
            },
            Instant::now() + Duration::from_millis(20),
        );
        assert!(
            result.is_err(),
            "connector exceeded its deadline without refusal"
        );
        release_tx.send(()).unwrap();
        peer.set_read_timeout(Some(Duration::from_millis(500)))
            .unwrap();
        assert_eq!(peer.read(&mut [0]).unwrap(), 0, "late socket was retained");
    }

    struct TestChild(std::process::Child);

    impl Drop for TestChild {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    fn wait_for_test_marker(child: &mut std::process::Child, marker: &Path) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !marker.exists() {
            assert!(
                child.try_wait().unwrap().is_none(),
                "child exited before marker {}",
                marker.display()
            );
            assert!(
                Instant::now() < deadline,
                "child did not enter the expected phase"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn handoff_dispatch_refuses_invalid_signals_before_acquisition() {
        const CHILD_ENV: &str = "LATTICE_GPU_INVALID_DISPATCH_CHILD";
        const ROOT_ENV: &str = "LATTICE_GPU_INVALID_DISPATCH_ROOT";
        if std::env::var_os(CHILD_ENV).is_some() {
            let root = PathBuf::from(std::env::var_os(ROOT_ENV).unwrap());
            std::fs::write(root.join("started"), b"started").unwrap();
            let _guard =
                gpu_test_lock_for_path(&root.join("lock"), HANDOFF_ENV.map(std::env::var_os));
            std::fs::write(root.join("entered"), b"entered").unwrap();
            return;
        }

        for partial in [true, false] {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path();
            let path = root.join("lock");
            let owner =
                acquire_machine_lock(&path, Duration::from_secs(5), Duration::from_millis(10));
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "measurement::tests::handoff_dispatch_refuses_invalid_signals_before_acquisition",
                    "--nocapture",
                ])
                .env_clear()
                .env(CHILD_ENV, "1")
                .env(ROOT_ENV, root)
                .env(HANDOFF_ENV[0], "0")
                .env(HANDOFF_ENV[1], root.join("invalid.sock"))
                .stdin(std::process::Stdio::from(owner.try_clone().unwrap()))
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::piped());
            if !partial {
                command.env(HANDOFF_ENV[2], "invalid-token");
            }
            let mut child = TestChild(command.spawn().unwrap());
            drop(command);
            let deadline = Instant::now() + Duration::from_secs(5);
            let status = loop {
                if let Some(status) = child.0.try_wait().unwrap() {
                    break status;
                }
                assert!(
                    Instant::now() < deadline,
                    "invalid handoff waited for native acquisition instead of refusing; partial={partial}"
                );
                std::thread::sleep(Duration::from_millis(10));
            };
            let mut error = String::new();
            child
                .0
                .stderr
                .as_mut()
                .unwrap()
                .read_to_string(&mut error)
                .unwrap();
            assert!(
                root.join("started").exists(),
                "child did not reach dispatch"
            );
            assert!(
                !status.success(),
                "invalid handoff succeeded; partial={partial}"
            );
            assert!(error.contains("refusing GPU handoff"), "{error}");
            assert!(
                !root.join("entered").exists(),
                "invalid handoff entered measurement"
            );
            let probe = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            assert!(matches!(
                probe.try_lock(),
                Err(std::fs::TryLockError::WouldBlock)
            ));
        }
    }

    #[test]
    fn handoff_dispatch_uses_verified_and_native_paths() {
        const CHILD_ENV: &str = "LATTICE_GPU_DISPATCH_TEST_CHILD";
        const ROOT_ENV: &str = "LATTICE_GPU_DISPATCH_TEST_ROOT";
        if std::env::var_os(CHILD_ENV).is_some() {
            let root = PathBuf::from(std::env::var_os(ROOT_ENV).unwrap());
            std::fs::write(root.join("started"), b"started").unwrap();
            let _guard =
                gpu_test_lock_for_path(&root.join("lock"), HANDOFF_ENV.map(std::env::var_os));
            std::fs::write(root.join("entered"), b"entered").unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while !root.join("release").exists() {
                assert!(
                    Instant::now() < deadline,
                    "parent did not release test child"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            return;
        }

        for (admitted, retain_owner) in [(true, true), (true, false), (false, false)] {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path();
            let path = root.join("lock");
            let mut owner = Some(acquire_machine_lock(
                &path,
                Duration::from_secs(5),
                Duration::from_millis(10),
            ));
            let probe = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let control = root.join("dispatch.sock");
            let listener = std::os::unix::net::UnixListener::bind(&control).unwrap();
            listener.set_nonblocking(true).unwrap();
            let token = "0123456789abcdef0123456789abcdef";
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "measurement::tests::handoff_dispatch_uses_verified_and_native_paths",
                    "--nocapture",
                ])
                .env(CHILD_ENV, "1")
                .env(ROOT_ENV, root);
            for name in HANDOFF_ENV {
                command.env_remove(name);
            }
            if admitted {
                command
                    .env(HANDOFF_ENV[0], "0")
                    .env(HANDOFF_ENV[1], &control)
                    .env(HANDOFF_ENV[2], token)
                    .stdin(std::process::Stdio::from(
                        owner.as_ref().unwrap().try_clone().unwrap(),
                    ));
            }
            let mut child = TestChild(command.spawn().unwrap());
            drop(command);
            wait_for_test_marker(&mut child.0, &root.join("started"));
            if admitted {
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                        Err(error) => panic!("cannot accept test handoff: {error}"),
                    }
                    assert!(
                        child.0.try_wait().unwrap().is_none(),
                        "admitted child exited"
                    );
                    assert!(
                        Instant::now() < deadline,
                        "admitted child stalled before READY"
                    );
                    std::thread::sleep(Duration::from_millis(10));
                };
                let frame =
                    read_handoff_frame(&mut stream, Instant::now() + Duration::from_secs(5))
                        .unwrap();
                let request: serde_json::Value = serde_json::from_slice(&frame).unwrap();
                assert_eq!(request["protocol"], 1);
                assert_eq!(request["token"], token);
                assert_eq!(request["pid"], child.0.id());
                assert_eq!(
                    PathBuf::from(request["exe"].as_str().unwrap()),
                    std::env::current_exe().unwrap()
                );
                assert!(!root.join("entered").exists(), "child entered before ACK");
                let response =
                    serde_json::json!({"protocol": 1, "token": token, "status": "ready"});
                writeln!(stream, "{response}").unwrap();
                wait_for_test_marker(&mut child.0, &root.join("entered"));
            } else {
                std::thread::sleep(Duration::from_millis(100));
                assert!(
                    !root.join("entered").exists(),
                    "native acquisition did not wait"
                );
                assert!(child.0.try_wait().unwrap().is_none());
            }
            if !retain_owner {
                drop(owner.take());
            }
            wait_for_test_marker(&mut child.0, &root.join("entered"));
            assert!(matches!(
                probe.try_lock(),
                Err(std::fs::TryLockError::WouldBlock)
            ));
            std::fs::write(root.join("release"), b"release").unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                if let Some(status) = child.0.try_wait().unwrap() {
                    assert!(status.success());
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "child did not finish its protected span"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            if retain_owner {
                assert!(
                    matches!(probe.try_lock(), Err(std::fs::TryLockError::WouldBlock)),
                    "target exit explicitly unlocked the retained supervisor lock"
                );
                drop(owner.take());
            }
            probe.try_lock().unwrap();
        }
    }
}
