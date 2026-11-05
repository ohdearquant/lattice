use super::*;

const MODEL_FIXTURE: &[u8] = b"model fixture";
const MODEL_FIXTURE_SHA256: &str =
    "21249a290a4255a0f3ee6685ff7933bffa241c3e35bb13a52dc0c7a679bed3b2";
const TOKENIZER_FIXTURE: &[u8] = b"tokenizer fixture";
const TOKENIZER_FIXTURE_SHA256: &str =
    "c882d95b612f23378308ac1207596a83cf97ed59d57e1b18e12a7085a10bf8e2";
const TOKENIZER_CASES: [(&str, &str); 7] = [
    ("bge-small-en-v1.5", "vocab.txt"),
    ("bge-base-en-v1.5", "vocab.txt"),
    ("bge-large-en-v1.5", "vocab.txt"),
    ("multilingual-e5-small", "tokenizer.json"),
    ("multilingual-e5-base", "tokenizer.json"),
    ("all-minilm-l6-v2", "vocab.txt"),
    ("paraphrase-multilingual-minilm-l12-v2", "tokenizer.json"),
];

#[test]
fn cache_presence_requires_each_models_tokenizer() {
    let root = tempfile::tempdir().unwrap();
    for (name, required) in TOKENIZER_CASES {
        assert_eq!(tokenizer_filename(name), Some(required), "{name}");
        assert_eq!(
            expected_checksums(name).tokenizer.unwrap().0,
            required,
            "{name}"
        );
        let wrong = if required == "vocab.txt" {
            "tokenizer.json"
        } else {
            "vocab.txt"
        };
        let directory = root.path().join(name);
        std::fs::create_dir(&directory).unwrap();
        assert!(
            !has_cached_model_files(&directory),
            "{name}: empty directory"
        );
        std::fs::write(directory.join("model.safetensors"), MODEL_FIXTURE).unwrap();
        assert!(!has_cached_model_files(&directory), "{name}: no tokenizer");
        std::fs::write(directory.join(wrong), TOKENIZER_FIXTURE).unwrap();
        assert!(
            !has_cached_model_files(&directory),
            "{name}: wrong tokenizer alone"
        );
        std::fs::write(directory.join(required), TOKENIZER_FIXTURE).unwrap();
        assert!(
            has_cached_model_files(&directory),
            "{name}: both tokenizers"
        );
        std::fs::remove_file(directory.join(wrong)).unwrap();
        assert!(
            has_cached_model_files(&directory),
            "{name}: correct tokenizer alone"
        );
        std::fs::remove_file(directory.join("model.safetensors")).unwrap();
        assert!(
            !has_cached_model_files(&directory),
            "{name}: missing weights"
        );
    }
}

#[test]
fn unknown_cache_names_do_not_inherit_a_tokenizer_choice() {
    let root = tempfile::tempdir().unwrap();
    for name in [
        "custom-model",
        "custom-e5-model",
        "custom-multilingual-model",
        "prefix-bge-small-en-v1.5",
        "bge-small-en-v1.5-copy",
        "BGE-small-en-v1.5",
        "bge-small-en-v1.5/custom",
        "all-MiniLM-L6-v2",
    ] {
        let directory = root.path().join(name);
        std::fs::create_dir_all(&directory).unwrap();
        for file in ["model.safetensors", "vocab.txt", "tokenizer.json"] {
            std::fs::write(directory.join(file), b"present").unwrap();
        }
        assert!(!has_cached_model_files(&directory), "{name}");
        assert!(
            matches!(
                ensure_model_files_inner(name, root.path(), true),
                Err(InferenceError::UnsupportedModel(_))
            ),
            "{name}"
        );
    }
}

#[test]
fn wrong_tokenizer_is_an_offline_cache_miss_before_checksum_verification() {
    let root = tempfile::tempdir().unwrap();
    for (name, required, wrong) in [
        ("bge-small-en-v1.5", "vocab.txt", "tokenizer.json"),
        ("multilingual-e5-small", "tokenizer.json", "vocab.txt"),
    ] {
        let directory = root.path().join(name);
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), MODEL_FIXTURE).unwrap();
        std::fs::write(directory.join(wrong), TOKENIZER_FIXTURE).unwrap();
        let expected = ExpectedChecksums {
            model_safetensors: Some(MODEL_FIXTURE_SHA256),
            tokenizer: Some((required, TOKENIZER_FIXTURE_SHA256)),
        };
        let result = ensure_model_files_inner_with_checksums(name, root.path(), true, expected);
        assert!(
            matches!(result, Err(InferenceError::ModelNotFound(_))),
            "{name}: wrong tokenizer must take the offline cache-miss branch, got {result:?}"
        );
        assert!(!directory.join(required).exists());
        assert_eq!(
            std::fs::read(directory.join(wrong)).unwrap(),
            TOKENIZER_FIXTURE
        );
        std::fs::rename(directory.join(wrong), directory.join(required)).unwrap();
        assert_eq!(
            ensure_model_files_inner_with_checksums(name, root.path(), true, expected).unwrap(),
            directory,
            "{name}: correct tokenizer must remain a verified offline hit"
        );
    }
}

#[cfg(unix)]
#[test]
fn non_utf8_cache_paths_are_not_hits() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let directory = Path::new(OsStr::from_bytes(b"bge-small-en-v1.5\xff"));
    assert!(directory.file_name().unwrap().to_str().is_none());
    assert!(!has_cached_model_files(directory));
}

// The macOS CI filesystem rejects these names before the cache predicate runs.
#[cfg(target_os = "linux")]
#[test]
fn non_utf8_cache_names_are_not_hits() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    let root = tempfile::tempdir().unwrap();
    let directory = root
        .path()
        .join(OsString::from_vec(b"bge-small-en-v1.5\xff".to_vec()));
    std::fs::create_dir(&directory).unwrap();
    for file in ["model.safetensors", "vocab.txt", "tokenizer.json"] {
        std::fs::write(directory.join(file), b"present").unwrap();
    }
    assert!(!has_cached_model_files(&directory));
}

#[cfg(unix)]
#[test]
fn cache_model_names_are_lexical_even_for_directory_symlinks() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("custom-target");
    std::fs::create_dir(&target).unwrap();
    std::fs::write(target.join("model.safetensors"), MODEL_FIXTURE).unwrap();
    std::fs::write(target.join("vocab.txt"), TOKENIZER_FIXTURE).unwrap();
    for (name, expected) in [
        ("bge-small-en-v1.5", true),
        ("multilingual-e5-small", false),
        ("custom-link", false),
    ] {
        let link = root.path().join(name);
        symlink(&target, &link).unwrap();
        assert_eq!(has_cached_model_files(&link), expected, "{name}");
    }
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn every_supported_model_downloads_from_a_pinned_commit_with_checksums() {
    for (name, tokenizer) in TOKENIZER_CASES {
        assert_eq!(canonical_model_name(name).expect("supported"), name);
        let (repo, revision) = hf_source(name).expect("source");
        assert_eq!(canonical_model_name(repo).expect("repo alias"), name);
        assert!(
            revision.len() == 40
                && revision
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "{name}: revision {revision:?} is not a commit id"
        );
        let checksums = expected_checksums(name);
        assert!(
            checksums.model_safetensors.is_some(),
            "{name}: no model checksum"
        );
        assert!(
            checksums.tokenizer.is_some(),
            "{name}: no tokenizer checksum"
        );
        assert_eq!(checksums.tokenizer.unwrap().0, tokenizer, "{name}");
    }
    assert!(hf_source("not-a-model").is_err());
}

#[cfg(all(unix, feature = "download", not(target_arch = "wasm32")))]
#[test]
fn download_paths_are_not_group_writable_under_a_group_writable_umask() {
    use std::os::unix::fs::PermissionsExt;
    const ROOT_ENV: &str = "LATTICE_DOWNLOAD_UMASK_CHILD_ROOT";
    if let Some(root) = std::env::var_os(ROOT_ENV) {
        // SAFETY: `umask` has no preconditions; this branch runs only in the child
        // process spawned below, so no other test observes the changed mask.
        unsafe { libc::umask(0o002) };
        let root = PathBuf::from(root);
        let model_dir = root.join("cache").join("all-minilm-l6-v2");
        create_model_dir(&model_dir).expect("create model dir");
        let stale = model_dir.join("model.part");
        std::fs::write(&stale, b"stale").expect("write stale part");
        std::fs::set_permissions(&stale, std::fs::Permissions::from_mode(0o664))
            .expect("make stale part group-writable");
        let (stage, file) = create_download_target(&model_dir.join("model.safetensors"))
            .expect("create download target");
        drop(file);
        for path in [root.join("cache"), model_dir.clone(), stage.path.clone()] {
            let mode = std::fs::metadata(&path).expect("stat").permissions().mode();
            assert_eq!(
                mode & 0o022,
                0,
                "{} is mode {:o}",
                path.display(),
                mode & 0o777
            );
        }
        assert_eq!(std::fs::metadata(&stage.path).expect("stat").len(), 0);
        assert!(!stale.exists());
        let stage_path = stage.path.clone();
        drop(stage);
        assert!(!stage_path.exists());
        return;
    }
    let temp = tempfile::tempdir().expect("tempdir");
    let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "download::tests::download_paths_are_not_group_writable_under_a_group_writable_umask",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ROOT_ENV, temp.path())
        .status()
        .expect("spawn child test");
    assert!(status.success(), "child test under umask 0002 failed");
}

fn fixture_checksums() -> ExpectedChecksums {
    ExpectedChecksums {
        model_safetensors: Some(MODEL_FIXTURE_SHA256),
        tokenizer: Some(("vocab.txt", TOKENIZER_FIXTURE_SHA256)),
    }
}

fn write_fixture_cache(test_name: &str, model: &[u8], tokenizer: &[u8]) -> (PathBuf, PathBuf) {
    let tmp = std::env::temp_dir().join(format!("lattice_{test_name}_{}", std::process::id()));
    let model_dir = tmp.join("all-minilm-l6-v2");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&model_dir).expect("create temp model dir");
    std::fs::write(model_dir.join("model.safetensors"), model).expect("write safetensors fixture");
    std::fs::write(model_dir.join("vocab.txt"), tokenizer).expect("write vocab fixture");
    (tmp, model_dir)
}

/// Offline mode + cache miss must error immediately, never attempting a download.
#[test]
fn offline_cache_miss_errors_without_download() {
    let tmp = std::env::temp_dir().join(format!("lattice_offline_miss_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let res = ensure_model_files_inner("all-minilm-l6-v2", &tmp, true);
    assert!(
        matches!(res, Err(InferenceError::ModelNotFound(_))),
        "offline + cache miss must return ModelNotFound, got {res:?}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Serializes tests in this module that mutate `LATTICE_OFFLINE` in the real
/// process environment. `set_var`/`remove_var` are `unsafe` because they can race
/// with a read on another thread; this is the same per-variable lock convention
/// `forward/metal_qwen35/inner/tests.rs`'s `with_self_spec_env` uses for
/// `LATTICE_SELF_SPEC`. Restores
/// the prior value (including "was unset") on the way out, even if `f` panics.
fn with_lattice_offline_env<R>(value: &str, f: impl FnOnce() -> R) -> R {
    use std::sync::Mutex;
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prior = std::env::var("LATTICE_OFFLINE").ok();
    // SAFETY: serialized by `ENV_LOCK` above — this lock only guards writers of
    // LATTICE_OFFLINE against each other; it does not stop other tests from
    // reading the environment concurrently.
    unsafe {
        std::env::set_var("LATTICE_OFFLINE", value);
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    // SAFETY: see above.
    unsafe {
        match &prior {
            Some(v) => std::env::set_var("LATTICE_OFFLINE", v),
            None => std::env::remove_var("LATTICE_OFFLINE"),
        }
    }
    match result {
        Ok(r) => r,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// `ensure_model_files` reads `LATTICE_OFFLINE` through `env_switch_enabled`, by
/// value, not presence (#1614): `=0` must mean "downloads are fine", not enable
/// fail-closed offline mode by merely being present. `cache_dir` here is a plain
/// file, not a directory, so a cache miss that falls through to the download
/// attempt fails immediately and deterministically at `create_dir_all` — no
/// network access is made either way, so the test cannot hang or flake on it.
#[test]
fn lattice_offline_is_read_by_value_not_presence() {
    let not_a_dir =
        std::env::temp_dir().join(format!("lattice_offline_value_{}", std::process::id()));
    let _ = std::fs::remove_file(&not_a_dir);
    std::fs::write(&not_a_dir, b"not a directory").expect("write stand-in file");

    let res = with_lattice_offline_env("0", || ensure_model_files("all-minilm-l6-v2", &not_a_dir));

    let _ = std::fs::remove_file(&not_a_dir);

    match res {
        Err(InferenceError::ModelNotFound(msg)) if msg.contains("LATTICE_OFFLINE") => {
            panic!("LATTICE_OFFLINE=0 must not enable offline mode by presence, got: {msg}");
        }
        Err(_) => {
            // Any other error (here: an Io error from the broken cache_dir) proves
            // the offline branch was not taken, which is what this test checks.
        }
        Ok(_) => panic!("a broken cache_dir must not report success"),
    }
}

/// Offline mode still serves a populated cache — it blocks the network, not cache reads.
#[test]
fn verified_offline_cache_hit_succeeds() {
    let (tmp, model_dir) =
        write_fixture_cache("offline_verified_hit", MODEL_FIXTURE, TOKENIZER_FIXTURE);
    let res = ensure_model_files_inner_with_checksums(
        "all-minilm-l6-v2",
        &tmp,
        true,
        fixture_checksums(),
    );
    assert_eq!(
        res.expect("verified offline cache hit must succeed"),
        model_dir
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn corrupt_cached_model_fails_before_online_download() {
    let (tmp, model_dir) = write_fixture_cache(
        "corrupt_cached_model",
        b"corrupt model fixture",
        TOKENIZER_FIXTURE,
    );
    let model_path = model_dir.join("model.safetensors");
    let res = ensure_model_files_inner_with_checksums(
        "all-minilm-l6-v2",
        &tmp,
        false,
        fixture_checksums(),
    );
    assert!(
        matches!(res, Err(InferenceError::ChecksumMismatch { .. })),
        "corrupt cache hit must fail before download, got {res:?}"
    );
    assert!(
        model_path.exists(),
        "cache-hit verification must preserve caller-managed artifacts"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn corrupt_cached_tokenizer_fails_offline_without_deleting_it() {
    let (tmp, model_dir) = write_fixture_cache(
        "corrupt_cached_tokenizer",
        MODEL_FIXTURE,
        b"corrupt tokenizer fixture",
    );
    let tokenizer_path = model_dir.join("vocab.txt");
    let res = ensure_model_files_inner_with_checksums(
        "all-minilm-l6-v2",
        &tmp,
        true,
        fixture_checksums(),
    );
    assert!(
        matches!(res, Err(InferenceError::ChecksumMismatch { .. })),
        "corrupt offline cache hit must fail closed, got {res:?}"
    );
    assert!(
        tokenizer_path.exists(),
        "offline verification must preserve the prefetched artifact"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn downloaded_checksum_mismatch_removes_artifact() {
    let (tmp, model_dir) = write_fixture_cache(
        "corrupt_downloaded_model",
        b"corrupt model fixture",
        TOKENIZER_FIXTURE,
    );
    let model_path = model_dir.join("model.safetensors");
    let res = verify_file_checksum(
        &model_path,
        MODEL_FIXTURE_SHA256,
        ChecksumFailureAction::Remove,
    );
    assert!(
        matches!(res, Err(InferenceError::ChecksumMismatch { .. })),
        "corrupt downloaded artifact must fail verification, got {res:?}"
    );
    assert!(
        !model_path.exists(),
        "failed download verification must retain its self-healing cleanup"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn download_agent_keeps_the_connect_timeout_and_redirect_limit_ureq_2_had() {
    let agent = download_agent();
    let config = agent.config();
    assert_eq!(
        config.timeouts().connect,
        Some(std::time::Duration::from_secs(30)),
        "a stalled connect must give up after 30 s, not wait for the OS"
    );
    assert_eq!(config.max_redirects(), 5);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn serve_loopback(response: Vec<u8>) -> u16 {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                match stream.read(&mut byte) {
                    Ok(1) => request.push(byte[0]),
                    _ => break,
                }
            }
            let _ = stream.write_all(&response);
        }
    });
    port
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn loopback_reply(status: &str, body: &[u8]) -> Vec<u8> {
    let mut reply = format!(
        "HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    reply.extend_from_slice(body);
    reply
}

/// Checkpoint files run to gigabytes, far past the 10 MB the HTTP client's
/// own buffered readers stop at, so the body is streamed to disk whole.
#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn download_file_writes_a_body_past_ten_megabytes_whole() {
    let body = vec![b'x'; 11 * 1024 * 1024];
    let port = serve_loopback(loopback_reply("200 OK", &body));
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("model.safetensors");
    download_file(&format!("http://127.0.0.1:{port}/model"), &path).expect("download");
    let written = std::fs::read(&path).expect("downloaded file");
    assert_eq!(written.len(), body.len());
    assert!(written == body);
    assert!(!path.with_extension("part").exists());
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 2);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn different_targets_have_independent_live_stages() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    let legacy = path.with_extension("part");
    std::fs::write(&legacy, b"legacy stage").unwrap();
    let (first, first_file) = create_download_target(&path).unwrap();
    let (second, second_file) =
        create_download_target(&temp.path().join("tokenizer.json")).unwrap();
    drop(first_file);
    drop(second_file);
    let first_path = first.path.clone();
    let second_path = second.path.clone();
    assert_ne!(first_path, second_path);
    assert!(first_path.exists());
    assert!(second_path.exists());
    assert!(!legacy.exists());
    drop(first);
    assert!(!first_path.exists());
    assert!(second_path.exists());
    drop(second);
    assert!(!second_path.exists());
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 2);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
struct PausedDownloadReader {
    bytes: &'static [u8],
    offset: usize,
    ready: Option<std::sync::mpsc::Sender<()>>,
    resume: std::sync::mpsc::Receiver<()>,
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
impl std::io::Read for PausedDownloadReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if buffer.is_empty() || self.offset == self.bytes.len() {
            return Ok(0);
        }
        if self.offset != 0
            && let Some(ready) = self.ready.take()
        {
            ready.send(()).map_err(std::io::Error::other)?;
            self.resume
                .recv_timeout(std::time::Duration::from_secs(30))
                .map_err(std::io::Error::other)?;
        }
        let count = buffer
            .len()
            .min(self.bytes.len() - self.offset)
            .min(if self.offset == 0 { 3 } else { usize::MAX });
        buffer[..count].copy_from_slice(&self.bytes[self.offset..self.offset + count]);
        self.offset += count;
        Ok(count)
    }
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn concurrent_downloads_publish_only_their_own_complete_bodies() {
    use std::sync::mpsc::channel;
    use std::time::Duration;

    const FIRST: &[u8] = b"first complete download";
    const SECOND: &[u8] = b"second complete download";
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    let (first_ready_tx, first_ready_rx) = channel();
    let (second_ready_tx, second_ready_rx) = channel();
    let (first_resume_tx, first_resume_rx) = channel();
    let (second_resume_tx, second_resume_rx) = channel();
    let (first_done_tx, first_done_rx) = channel();
    let first_path = path.clone();
    let first = std::thread::spawn(move || {
        let result = write_download_body(
            PausedDownloadReader {
                bytes: FIRST,
                offset: 0,
                ready: Some(first_ready_tx),
                resume: first_resume_rx,
            },
            &first_path,
        );
        let _ = first_done_tx.send(());
        result
    });
    let first_ready = first_ready_rx.recv_timeout(Duration::from_secs(30));
    let second_path = path.clone();
    let second = std::thread::spawn(move || {
        write_download_body(
            PausedDownloadReader {
                bytes: SECOND,
                offset: 0,
                ready: Some(second_ready_tx),
                resume: second_resume_rx,
            },
            &second_path,
        )
    });
    let probe = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(download_lock_path(&path).unwrap())
        .unwrap();
    let contended = matches!(probe.try_lock(), Err(std::fs::TryLockError::WouldBlock));
    drop(probe);
    let _ = first_resume_tx.send(());
    let first_done = first_done_rx.recv_timeout(Duration::from_secs(30));
    let second_ready = second_ready_rx.recv_timeout(Duration::from_secs(30));
    let first_published = std::fs::read(&path);
    let _ = second_resume_tx.send(());
    let first_result = first.join();
    let second_result = second.join();

    assert!(contended, "live writer must hold the target lock");
    assert!(first_ready.is_ok(), "first writer did not pause");
    assert!(second_ready.is_ok(), "second writer did not pause");
    assert!(first_done.is_ok(), "first writer did not finish");
    first_result.unwrap().unwrap();
    assert_eq!(first_published.unwrap(), FIRST);
    second_result.unwrap().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), SECOND);
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 2);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn interrupted_download_removes_only_its_own_stage() {
    struct Interrupted(bool);
    impl std::io::Read for Interrupted {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            if buffer.is_empty() {
                return Ok(0);
            }
            if self.0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "interrupted fixture",
                ));
            }
            self.0 = true;
            buffer[0] = b'x';
            Ok(1)
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    let legacy = path.with_extension("part");
    std::fs::write(&path, b"existing artifact").unwrap();
    std::fs::write(&legacy, b"legacy stage").unwrap();
    let (other, other_file) = create_download_target(&temp.path().join("tokenizer.json")).unwrap();
    drop(other_file);
    let result = write_download_body(Interrupted(false), &path);
    assert!(
        matches!(result, Err(InferenceError::Io(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof)
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"existing artifact");
    assert!(!legacy.exists());
    assert!(other.path.exists());
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 4);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn failed_download_publication_removes_its_stage() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("keep"), b"existing directory").unwrap();
    let result = write_download_body(&b"complete body"[..], &path);
    assert!(matches!(result, Err(InferenceError::Io(_))));
    assert_eq!(
        std::fs::read(path.join("keep")).unwrap(),
        b"existing directory"
    );
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 2);
}

#[cfg(all(feature = "download", target_os = "linux"))]
#[test]
fn non_utf8_download_destination_is_preserved() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let temp = tempfile::tempdir().unwrap();
    let path = temp
        .path()
        .join(OsStr::from_bytes(b"model\xff.safetensors"));
    write_download_body(&b"complete body"[..], &path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"complete body");
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 2);
}

/// An error status is a failed download, never a file holding the error page.
#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn download_file_refuses_an_error_status_and_writes_nothing() {
    let port = serve_loopback(loopback_reply("404 Not Found", b"missing"));
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("model.safetensors");
    let result = download_file(&format!("http://127.0.0.1:{port}/model"), &path);
    assert!(
        matches!(result, Err(InferenceError::Download(_))),
        "{result:?}"
    );
    assert!(!path.exists());
    assert!(!path.with_extension("part").exists());
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
const PROXY_CHILD_PORT: &str = "LATTICE_DOWNLOAD_PROXY_CHILD_PORT";

/// Proxy settings in the environment are not applied to downloads. The check
/// runs in a child process so the variables are set only there. In the child,
/// a client built with the library's default configuration must fail through
/// the dead proxy, which shows the variables were read.
#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn download_file_ignores_proxy_settings_in_the_environment() {
    if let Ok(port) = std::env::var(PROXY_CHILD_PORT) {
        let url = format!("http://127.0.0.1:{port}/model");
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("model.safetensors");
        download_file(&url, &path).expect("download through the proxy settings");
        assert_eq!(std::fs::read(&path).expect("downloaded file"), b"weights");
        assert!(
            ureq::Agent::new_with_defaults().get(&url).call().is_err(),
            "a default client reached the server, so the proxy variables were not read"
        );
        return;
    }
    let port = serve_loopback(loopback_reply("200 OK", b"weights"));
    let dead_proxy = "http://127.0.0.1:9";
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "download::tests::download_file_ignores_proxy_settings_in_the_environment",
            "--exact",
            "--test-threads=1",
            "--nocapture",
        ])
        .env(PROXY_CHILD_PORT, port.to_string())
        .env("ALL_PROXY", dead_proxy)
        .env("HTTP_PROXY", dead_proxy)
        .env("http_proxy", dead_proxy)
        .env_remove("NO_PROXY")
        .env_remove("no_proxy")
        .output()
        .expect("child test process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "child failed: {stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
