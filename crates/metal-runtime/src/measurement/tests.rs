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
            None,
        );
        return;
    }

    let temp = tempfile::tempdir().expect("temporary lock-test directory");
    let lock_path = temp.path().join("machine-lock");
    let guard = acquire_machine_lock(
        &lock_path,
        Duration::from_secs(5),
        Duration::from_millis(10),
        None,
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
fn supervisor_marker_with_live_pid_refuses_fast_on_contention() {
    let temp = tempfile::tempdir().expect("temporary lock-test directory");
    let lock_path = temp.path().join("lock");
    // A second open of the same path is a separate open-file description,
    // so its flock contends with `acquire_machine_lock`'s own open below
    // without needing a second process, the same pattern
    // `handoff_proof_discriminates_lock_ownership` uses elsewhere in this
    // module.
    let holder = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .expect("open lock holder");
    holder.try_lock().expect("hold the lock for contention");

    let marker = Some(OsString::from(std::process::id().to_string()));
    let started = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        acquire_machine_lock(
            &lock_path,
            Duration::from_secs(60),
            Duration::from_millis(10),
            marker,
        )
    }));
    let elapsed = started.elapsed();

    assert!(result.is_err(), "a live supervisor marker did not panic");
    assert!(
        elapsed < Duration::from_secs(10),
        "did not refuse fast: waited {elapsed:?} against a 60 s timeout"
    );
    let message = result
        .unwrap_err()
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(
        message.contains("benchmark supervisor"),
        "unexpected panic message: {message}"
    );
    drop(holder);
}

#[test]
fn supervisor_marker_absent_waits_out_the_short_timeout() {
    let temp = tempfile::tempdir().expect("temporary lock-test directory");
    let lock_path = temp.path().join("lock");
    let holder = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .expect("open lock holder");
    holder.try_lock().expect("hold the lock for contention");

    let short_timeout = Duration::from_millis(200);
    let started = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        acquire_machine_lock(&lock_path, short_timeout, Duration::from_millis(10), None)
    }));
    let elapsed = started.elapsed();

    assert!(
        result.is_err(),
        "expected the bounded wait to panic once its injected timeout elapsed"
    );
    assert!(
        elapsed >= short_timeout,
        "returned before the injected timeout: waited {elapsed:?}, timeout {short_timeout:?}"
    );
    drop(holder);
}

#[test]
fn supervisor_marker_with_dead_pid_waits_out_the_short_timeout() {
    let temp = tempfile::tempdir().expect("temporary lock-test directory");
    let lock_path = temp.path().join("lock");
    let holder = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .expect("open lock holder");
    holder.try_lock().expect("hold the lock for contention");

    // No real process holds this pid on Linux (pid_max is well under
    // 2^31) or macOS (PID_MAX 99999), so this names a definitely-dead pid
    // without racing an actual process's exit against being reaped.
    let marker = Some(OsString::from(i32::MAX.to_string()));
    let short_timeout = Duration::from_millis(200);
    let started = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        acquire_machine_lock(&lock_path, short_timeout, Duration::from_millis(10), marker)
    }));
    let elapsed = started.elapsed();

    assert!(
        result.is_err(),
        "expected the bounded wait to panic once its injected timeout elapsed"
    );
    assert!(
        elapsed >= short_timeout,
        "a dead-pid marker refused instead of waiting: waited {elapsed:?}, timeout {short_timeout:?}"
    );
    drop(holder);
}

#[test]
fn supervisor_marker_with_live_pid_acquires_when_uncontended() {
    let temp = tempfile::tempdir().expect("temporary lock-test directory");
    let lock_path = temp.path().join("lock");
    let marker = Some(OsString::from(std::process::id().to_string()));

    let file = acquire_machine_lock(
        &lock_path,
        Duration::from_secs(5),
        Duration::from_millis(10),
        marker,
    );

    let probe = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&lock_path)
        .expect("open a second description on the same lock");
    assert!(
        matches!(probe.try_lock(), Err(std::fs::TryLockError::WouldBlock)),
        "an uncontended acquisition with a live marker set did not hold the lock"
    );
    drop(file);
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
    let owner = acquire_machine_lock(
        &path,
        Duration::from_secs(5),
        Duration::from_millis(10),
        None,
    );
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
        "{\"protocol\":1,\"token\":\"ffffffffffffffffffffffffffffffff\",\"status\":\"ready\"}\n"
            .into(),
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
    let error =
        read_handoff_frame(&mut reader, Instant::now() + Duration::from_millis(20)).unwrap_err();
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
    let frame = read_handoff_frame(&mut reader, Instant::now() + Duration::from_secs(5)).unwrap();
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
            gpu_test_lock_for_path(&root.join("lock"), HANDOFF_ENV.map(std::env::var_os), None);
        std::fs::write(root.join("entered"), b"entered").unwrap();
        return;
    }

    for partial in [true, false] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let path = root.join("lock");
        let owner = acquire_machine_lock(
            &path,
            Duration::from_secs(5),
            Duration::from_millis(10),
            None,
        );
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
            gpu_test_lock_for_path(&root.join("lock"), HANDOFF_ENV.map(std::env::var_os), None);
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
            None,
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
                read_handoff_frame(&mut stream, Instant::now() + Duration::from_secs(5)).unwrap();
            let request: serde_json::Value = serde_json::from_slice(&frame).unwrap();
            assert_eq!(request["protocol"], 1);
            assert_eq!(request["token"], token);
            assert_eq!(request["pid"], child.0.id());
            assert_eq!(
                PathBuf::from(request["exe"].as_str().unwrap()),
                std::env::current_exe().unwrap()
            );
            assert!(!root.join("entered").exists(), "child entered before ACK");
            let response = serde_json::json!({"protocol": 1, "token": token, "status": "ready"});
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
