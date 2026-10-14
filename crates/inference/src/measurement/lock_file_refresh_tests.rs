use super::*;
use std::fs::Metadata;
use std::os::unix::net::UnixListener;
use std::process::{Child, Command, Stdio};
use std::time::UNIX_EPOCH;

const OLD_AGE: Duration = Duration::from_secs(4 * 24 * 60 * 60);
const TEST_TIMEOUT: Duration = Duration::from_secs(30);

fn aged_lock(path: &Path) -> File {
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let old = SystemTime::now() - OLD_AGE;
    file.set_times(FileTimes::new().set_accessed(old).set_modified(old))
        .unwrap();
    let metadata = file.metadata().unwrap();
    for timestamp in [metadata.accessed().unwrap(), metadata.modified().unwrap()] {
        assert!(
            SystemTime::now().duration_since(timestamp).unwrap() > OLD_AGE - Duration::from_secs(1)
        );
    }
    file
}

fn assert_refreshed(file: &File, path: &Path, before: &Metadata, started: SystemTime) {
    let held = file.metadata().unwrap();
    let canonical = std::fs::metadata(path).unwrap();
    let identity = (before.dev(), before.ino());
    assert_eq!((held.dev(), held.ino()), identity, "held inode changed");
    assert_eq!(
        (canonical.dev(), canonical.ino()),
        identity,
        "path inode changed"
    );
    let earliest = started - Duration::from_secs(2);
    let latest = SystemTime::now() + Duration::from_secs(2);
    // ctime cannot be backdated; atime/mtime establish mutation sensitivity.
    for metadata in [&held, &canonical] {
        let changed = UNIX_EPOCH
            + Duration::new(
                u64::try_from(metadata.ctime()).unwrap(),
                u32::try_from(metadata.ctime_nsec()).unwrap(),
            );
        for (name, timestamp) in [
            ("atime", metadata.accessed().unwrap()),
            ("mtime", metadata.modified().unwrap()),
            ("ctime", changed),
        ] {
            assert!(
                earliest <= timestamp && timestamp <= latest,
                "{name} was not refreshed on the existing inode: {timestamp:?}"
            );
        }
    }
}

#[test]
fn native_acquisition_refreshes_old_lock_timestamps_without_replacing_inode() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("lock");
    let file = aged_lock(&path);
    let before = file.metadata().unwrap();
    let started = SystemTime::now();
    let guard = gpu_test_lock_for_path(&path, [None, None, None], None);
    assert_refreshed(&guard._machine, &path, &before, started);
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

struct RefreshTestChild(Child);

impl Drop for RefreshTestChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn handoff_refreshes_old_lock_timestamps_before_ready_without_replacing_inode() {
    const CHILD_ROOT: &str = "LATTICE_LOCK_REFRESH_CHILD_ROOT";
    const TOKEN: &str = "0123456789abcdef0123456789abcdef";
    const TEST_NAME: &str = "measurement::lock_file_refresh_tests::handoff_refreshes_old_lock_timestamps_before_ready_without_replacing_inode";
    if let Some(root) = std::env::var_os(CHILD_ROOT) {
        let root = PathBuf::from(root);
        let path = root.join("lock");
        let before = std::fs::metadata(&path).unwrap();
        let started = SystemTime::now();
        let guard = gpu_test_lock_for_path(
            &path,
            [
                Some(OsString::from("0")),
                Some(root.join("s").into_os_string()),
                Some(OsString::from(TOKEN)),
            ],
            None,
        );
        assert_refreshed(&guard._machine, &path, &before, started);
        return;
    }

    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("lock");
    let owner = aged_lock(&path);
    owner.try_lock().unwrap();
    let before = owner.metadata().unwrap();
    let listener = UnixListener::bind(temporary.path().join("s")).unwrap();
    listener.set_nonblocking(true).unwrap();
    let started = SystemTime::now();
    let mut child = RefreshTestChild(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST_NAME, "--nocapture"])
            .env(CHILD_ROOT, temporary.path())
            .stdin(Stdio::from(owner.try_clone().unwrap()))
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + TEST_TIMEOUT;
    let mut stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("cannot accept refresh test handoff: {error}"),
        }
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "handoff child exited before READY"
        );
        assert!(
            Instant::now() < deadline,
            "handoff child never requested READY"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    let frame = read_handoff_frame(&mut stream, deadline).unwrap();
    let request: serde_json::Value = serde_json::from_slice(&frame).unwrap();
    assert_eq!(request["protocol"], 1);
    assert_eq!(request["token"], TOKEN);
    assert_eq!(request["pid"], child.0.id());
    assert_refreshed(&owner, &path, &before, started);
    let probe = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    assert!(matches!(
        probe.try_lock(),
        Err(std::fs::TryLockError::WouldBlock)
    ));
    let ready = serde_json::json!({"protocol": 1, "token": TOKEN, "status": "ready"});
    write_handoff_frame(&mut stream, format!("{ready}\n").as_bytes(), deadline).unwrap();
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(
                status.success(),
                "handoff child failed after READY: {status}"
            );
            break;
        }
        assert!(Instant::now() < deadline, "handoff child did not finish");
        std::thread::sleep(Duration::from_millis(10));
    }
}
