use super::*;

const MODEL_FIXTURE: &[u8] = b"model fixture";
const MODEL_FIXTURE_SHA256: &str =
    "21249a290a4255a0f3ee6685ff7933bffa241c3e35bb13a52dc0c7a679bed3b2";
const TOKENIZER_FIXTURE: &[u8] = b"tokenizer fixture";
const TOKENIZER_FIXTURE_SHA256: &str =
    "c882d95b612f23378308ac1207596a83cf97ed59d57e1b18e12a7085a10bf8e2";

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn every_supported_model_downloads_from_a_pinned_commit_with_checksums() {
    const MODELS: [&str; 7] = [
        "bge-small-en-v1.5",
        "bge-base-en-v1.5",
        "bge-large-en-v1.5",
        "multilingual-e5-small",
        "multilingual-e5-base",
        "all-minilm-l6-v2",
        "paraphrase-multilingual-minilm-l12-v2",
    ];
    for name in MODELS {
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
        drop(create_download_target(&stale).expect("create download target"));
        for path in [root.join("cache"), model_dir.clone(), stale.clone()] {
            let mode = std::fs::metadata(&path).expect("stat").permissions().mode();
            assert_eq!(
                mode & 0o022,
                0,
                "{} is mode {:o}",
                path.display(),
                mode & 0o777
            );
        }
        assert_eq!(std::fs::metadata(&stale).expect("stat").len(), 0);
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
/// `metal_qwen35.rs`'s `with_self_spec_env` uses for `LATTICE_SELF_SPEC`. Restores
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
