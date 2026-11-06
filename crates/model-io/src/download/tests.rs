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

#[test]
fn full_cache_missing_pins_refuse_without_removing_artifacts() {
    for expected in [
        ExpectedChecksums {
            model_safetensors: None,
            ..fixture_checksums()
        },
        ExpectedChecksums {
            tokenizer: None,
            ..fixture_checksums()
        },
    ] {
        let temp = tempfile::tempdir().unwrap();
        let model_dir = temp.path().join("all-minilm-l6-v2");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.safetensors"), MODEL_FIXTURE).unwrap();
        std::fs::write(model_dir.join("vocab.txt"), TOKENIZER_FIXTURE).unwrap();
        for offline in [false, true] {
            let result = ensure_model_files_inner_with_checksums(
                "all-minilm-l6-v2",
                temp.path(),
                offline,
                expected,
            );
            assert!(
                matches!(result, Err(InferenceError::Download(_))),
                "{result:?}"
            );
            assert_eq!(
                std::fs::read(model_dir.join("model.safetensors")).unwrap(),
                MODEL_FIXTURE
            );
            assert_eq!(
                std::fs::read(model_dir.join("vocab.txt")).unwrap(),
                TOKENIZER_FIXTURE
            );
        }
    }
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn rejected_download_never_publishes_unverified_bytes() {
    for (filename, expected, rejected) in [
        ("model.safetensors", MODEL_FIXTURE_SHA256, TOKENIZER_FIXTURE),
        ("vocab.txt", TOKENIZER_FIXTURE_SHA256, MODEL_FIXTURE),
        ("tokenizer.json", TOKENIZER_FIXTURE_SHA256, MODEL_FIXTURE),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join(filename);
        let sentinel = temp.path().join("unrelated");
        std::fs::write(&sentinel, b"keep").unwrap();
        let sync_calls = std::cell::Cell::new(0);
        let result = download_with_sync(
            &path,
            expected,
            || Ok(std::io::Cursor::new(rejected)),
            |file| {
                sync_calls.set(sync_calls.get() + 1);
                file.sync_all()
            },
        );
        assert!(
            !path.exists(),
            "rejected bytes were published at the final artifact path"
        );
        assert_eq!(
            sync_calls.get(),
            0,
            "verification must precede synchronization"
        );
        match result {
            Err(InferenceError::ChecksumMismatch {
                file,
                expected: reported,
                actual,
            }) => {
                assert_eq!(file, path.display().to_string());
                assert_eq!(reported, expected);
                assert_eq!(actual, fixture_digest(rejected));
            }
            other => panic!("expected a staged checksum refusal, got {other:?}"),
        }
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"keep");
        assert_download_entries(temp.path(), &[sentinel, download_lock_path(&path).unwrap()]);
    }
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn download_sync_failure_prevents_publication_and_cleans_stage() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    let sentinel = temp.path().join("unrelated");
    std::fs::write(&sentinel, b"keep").unwrap();
    let sync_calls = std::cell::Cell::new(0);
    let result = download_with_sync(
        &path,
        MODEL_FIXTURE_SHA256,
        || Ok(std::io::Cursor::new(MODEL_FIXTURE)),
        |file| {
            sync_calls.set(sync_calls.get() + 1);
            assert_eq!(file.metadata()?.len(), MODEL_FIXTURE.len() as u64);
            Err(std::io::Error::other(
                "injected file synchronization failure",
            ))
        },
    );
    assert!(
        !path.exists(),
        "a synchronization failure published the final artifact"
    );
    assert_eq!(sync_calls.get(), 1, "synchronization must be attempted");
    match result {
        Err(InferenceError::Io(error)) => {
            assert_eq!(error.to_string(), "injected file synchronization failure")
        }
        other => panic!("expected the synchronization error, got {other:?}"),
    }
    assert_eq!(std::fs::read(&sentinel).unwrap(), b"keep");
    assert_download_entries(temp.path(), &[sentinel, download_lock_path(&path).unwrap()]);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn existing_download_is_verified_without_fetching() {
    for expected in [MODEL_FIXTURE_SHA256, TOKENIZER_FIXTURE_SHA256] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.safetensors");
        std::fs::write(&path, MODEL_FIXTURE).unwrap();
        let fetches = std::cell::Cell::new(0);
        let result = download_with(&path, expected, || {
            fetches.set(fetches.get() + 1);
            Ok(std::io::Cursor::new(MODEL_FIXTURE))
        });
        assert_eq!(
            fetches.get(),
            0,
            "an existing artifact must not trigger a fetch"
        );
        if expected == MODEL_FIXTURE_SHA256 {
            result.unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), MODEL_FIXTURE);
        } else {
            assert!(
                matches!(result, Err(InferenceError::ChecksumMismatch { .. })),
                "an existing artifact bypassed its pin: {result:?}"
            );
            assert!(
                !path.exists(),
                "partial-cache mismatch must retain removal policy"
            );
        }
    }
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn existing_nonregular_download_target_is_not_a_fetch_candidate() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    std::fs::create_dir(&path).unwrap();
    let fetches = std::cell::Cell::new(0);
    let result = download_with(&path, MODEL_FIXTURE_SHA256, || {
        fetches.set(fetches.get() + 1);
        Ok(std::io::Cursor::new(MODEL_FIXTURE))
    });
    assert_eq!(fetches.get(), 0);
    assert!(matches!(result, Err(InferenceError::Io(_))), "{result:?}");
    assert!(path.is_dir());
}

#[cfg(all(feature = "download", unix, not(target_arch = "wasm32")))]
#[test]
fn dangling_download_target_is_replaced_with_verified_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    let missing = temp.path().join("missing");
    std::os::unix::fs::symlink(&missing, &path).unwrap();
    let fetches = std::cell::Cell::new(0);
    let result = download_with(&path, MODEL_FIXTURE_SHA256, || {
        fetches.set(fetches.get() + 1);
        Ok(std::io::Cursor::new(MODEL_FIXTURE))
    });
    result.unwrap();
    assert_eq!(fetches.get(), 1);
    assert!(std::fs::symlink_metadata(&path).unwrap().is_file());
    assert_eq!(std::fs::read(&path).unwrap(), MODEL_FIXTURE);
    assert!(!missing.exists());
    assert_download_entries(temp.path(), &[download_lock_path(&path).unwrap(), path]);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn partial_cache_rejects_corrupt_model_before_tokenizer_fetch() {
    let temp = tempfile::tempdir().unwrap();
    let model_dir = temp.path().join("all-minilm-l6-v2");
    std::fs::create_dir(&model_dir).unwrap();
    let model_path = model_dir.join("model.safetensors");
    std::fs::write(&model_path, b"corrupt model").unwrap();
    let mut visited = Vec::new();
    let fetches = std::cell::Cell::new(0);
    let result = ensure_partial_cache_with(
        "all-minilm-l6-v2",
        &model_dir,
        fixture_checksums(),
        |_, path, pin| {
            visited.push(path.to_owned());
            download_with(path, pin, || {
                fetches.set(fetches.get() + 1);
                Ok(std::io::Cursor::new(TOKENIZER_FIXTURE))
            })
        },
    );
    assert!(
        matches!(result, Err(InferenceError::ChecksumMismatch { .. })),
        "{result:?}"
    );
    assert_eq!(visited, std::slice::from_ref(&model_path));
    assert_eq!(
        fetches.get(),
        0,
        "corrupt partial model must fail before fetching its missing tokenizer"
    );
    assert!(!model_path.exists());
    assert!(!model_dir.join("vocab.txt").exists());
    assert_download_entries(&model_dir, &[download_lock_path(&model_path).unwrap()]);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn partial_cache_downloads_with_each_artifacts_own_pin() {
    for (model_name, tokenizer_name) in TOKENIZER_CASES {
        let temp = tempfile::tempdir().unwrap();
        let model_dir = temp.path().join(model_name);
        let expected = ExpectedChecksums {
            model_safetensors: Some(MODEL_FIXTURE_SHA256),
            tokenizer: Some((tokenizer_name, TOKENIZER_FIXTURE_SHA256)),
        };
        let mut visits = Vec::new();
        let mut fetches = 0;
        ensure_partial_cache_with(model_name, &model_dir, expected, |url, path, pin| {
            visits.push((url.to_owned(), path.to_owned(), pin.to_owned()));
            let payload = if path.file_name().unwrap() == "model.safetensors" {
                MODEL_FIXTURE
            } else {
                TOKENIZER_FIXTURE
            };
            download_with(path, pin, || {
                fetches += 1;
                Ok(std::io::Cursor::new(payload))
            })
        })
        .unwrap();
        let (repo, revision) = hf_source(model_name).unwrap();
        let expected_visits: Vec<_> = [
            ("model.safetensors", MODEL_FIXTURE_SHA256),
            (tokenizer_name, TOKENIZER_FIXTURE_SHA256),
        ]
        .into_iter()
        .map(|(name, pin)| {
            (
                format!("https://huggingface.co/{repo}/resolve/{revision}/{name}"),
                model_dir.join(name),
                pin.to_owned(),
            )
        })
        .collect();
        assert_eq!(visits, expected_visits);
        assert_eq!(fetches, 2);
        assert_eq!(
            std::fs::read(model_dir.join("model.safetensors")).unwrap(),
            MODEL_FIXTURE
        );
        assert_eq!(
            std::fs::read(model_dir.join(tokenizer_name)).unwrap(),
            TOKENIZER_FIXTURE
        );
    }
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn partial_cache_removes_corrupt_tokenizer_after_verified_model_download() {
    let temp = tempfile::tempdir().unwrap();
    let model_dir = temp.path().join("all-minilm-l6-v2");
    std::fs::create_dir(&model_dir).unwrap();
    std::fs::write(model_dir.join("vocab.txt"), b"corrupt tokenizer").unwrap();
    let mut fetches = Vec::new();
    let result = ensure_partial_cache_with(
        "all-minilm-l6-v2",
        &model_dir,
        fixture_checksums(),
        |_, path, pin| {
            download_with(path, pin, || {
                fetches.push(path.to_owned());
                Ok(std::io::Cursor::new(MODEL_FIXTURE))
            })
        },
    );
    assert!(
        matches!(result, Err(InferenceError::ChecksumMismatch { .. })),
        "{result:?}"
    );
    assert_eq!(fetches, [model_dir.join("model.safetensors")]);
    assert_eq!(
        std::fs::read(model_dir.join("model.safetensors")).unwrap(),
        MODEL_FIXTURE
    );
    assert!(!model_dir.join("vocab.txt").exists());
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn missing_download_pins_refuse_before_fetch_or_publication() {
    for expected in [
        ExpectedChecksums {
            model_safetensors: None,
            ..fixture_checksums()
        },
        ExpectedChecksums {
            tokenizer: None,
            ..fixture_checksums()
        },
    ] {
        let temp = tempfile::tempdir().unwrap();
        let model_dir = temp.path().join("all-minilm-l6-v2");
        let mut downloads = 0;
        let result =
            ensure_partial_cache_with("all-minilm-l6-v2", &model_dir, expected, |_, _, _| {
                downloads += 1;
                Ok(())
            });
        assert!(
            matches!(result, Err(InferenceError::Download(_))),
            "{result:?}"
        );
        assert_eq!(downloads, 0);
        assert!(!model_dir.exists());
    }
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

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn fixture_digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn assert_download_entries(dir: &Path, expected: &[PathBuf]) {
    let mut actual: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    actual.sort();
    let mut expected = expected.to_vec();
    expected.sort();
    assert_eq!(actual, expected, "unexpected download directory entries");
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
    download_file(
        &format!("http://127.0.0.1:{port}/model"),
        &path,
        &fixture_digest(&body),
    )
    .expect("download");
    let written = std::fs::read(&path).expect("downloaded file");
    assert_eq!(written.len(), body.len());
    assert!(written == body);
    assert_download_entries(temp.path(), &[download_lock_path(&path).unwrap(), path]);
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
    let contended = download_lock_path(&path).and_then(|lock_path| {
        let probe = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(lock_path)?;
        Ok(matches!(
            probe.try_lock(),
            Err(std::fs::TryLockError::WouldBlock)
        ))
    });
    let _ = first_resume_tx.send(());
    let first_done = first_done_rx.recv_timeout(Duration::from_secs(30));
    let second_ready = second_ready_rx.recv_timeout(Duration::from_secs(30));
    let first_published = std::fs::read(&path);
    let _ = second_resume_tx.send(());
    let first_result = first.join();
    let second_result = second.join();

    assert!(contended.unwrap(), "live writer must hold the target lock");
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
    assert_download_entries(
        temp.path(),
        &[
            path.clone(),
            download_lock_path(&path).unwrap(),
            download_lock_path(&temp.path().join("tokenizer.json")).unwrap(),
            other.path.clone(),
        ],
    );
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

// The macOS CI filesystem rejects this invalid UTF-8 name before staging runs.
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
    let result = download_file(
        &format!("http://127.0.0.1:{port}/model"),
        &path,
        MODEL_FIXTURE_SHA256,
    );
    assert!(
        matches!(result, Err(InferenceError::Download(_))),
        "{result:?}"
    );
    assert!(!path.exists());
    assert_download_entries(temp.path(), &[download_lock_path(&path).unwrap()]);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn verified_download_waiter_checks_the_published_digest() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::channel;
    use std::time::Duration;

    for waiter_pin in [MODEL_FIXTURE_SHA256, TOKENIZER_FIXTURE_SHA256] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.safetensors");
        let calls = Arc::new(AtomicUsize::new(0));
        let sync_calls = Arc::new(AtomicUsize::new(0));
        let (ready_tx, ready_rx) = channel();
        let (resume_tx, resume_rx) = channel();
        let (event_tx, event_rx) = channel();
        let first_path = path.clone();
        let first_calls = Arc::clone(&calls);
        let first_sync_calls = Arc::clone(&sync_calls);
        let first = std::thread::spawn(move || {
            download_with_sync(
                &first_path,
                MODEL_FIXTURE_SHA256,
                || {
                    first_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(std::io::Cursor::new(MODEL_FIXTURE))
                },
                |file| {
                    first_sync_calls.fetch_add(1, Ordering::SeqCst);
                    ready_tx.send(()).map_err(std::io::Error::other)?;
                    resume_rx
                        .recv_timeout(Duration::from_secs(30))
                        .map_err(std::io::Error::other)?;
                    file.sync_all()
                },
            )
        });
        let ready = ready_rx.recv_timeout(Duration::from_secs(30));
        let second_path = path.clone();
        let second_calls = Arc::clone(&calls);
        let second = std::thread::spawn(move || {
            let observer_tx = event_tx.clone();
            staging::observe_next_lock(move |contended| {
                let _ = observer_tx.send(WaiterEvent::Contention(contended));
            });
            download_with(&second_path, waiter_pin, || {
                second_calls.fetch_add(1, Ordering::SeqCst);
                let _ = event_tx.send(WaiterEvent::Fetch);
                Ok(std::io::Cursor::new(MODEL_FIXTURE))
            })
        });
        let event = event_rx.recv_timeout(Duration::from_secs(30));
        let final_visible_before_sync = path.exists();
        let stage_payloads = (|| -> std::io::Result<Vec<Vec<u8>>> {
            let mut payloads = Vec::new();
            for entry in std::fs::read_dir(temp.path())? {
                let entry = entry?;
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".model.safetensors.part-")
                {
                    payloads.push(std::fs::read(entry.path())?);
                }
            }
            Ok(payloads)
        })();
        let calls_before_release = calls.load(Ordering::SeqCst);
        let _ = resume_tx.send(());
        let first_result = first.join();
        let second_result = second.join();

        assert!(
            ready.is_ok(),
            "verified stage did not reach synchronization"
        );
        assert!(
            matches!(event, Ok(WaiterEvent::Contention(true))),
            "waiter must observe contention during finalization"
        );
        assert!(
            !final_visible_before_sync,
            "final artifact became visible before synchronization completed"
        );
        assert_eq!(
            stage_payloads.unwrap(),
            [MODEL_FIXTURE.to_vec()],
            "synchronization must see the complete staged artifact"
        );
        assert_eq!(
            sync_calls.load(Ordering::SeqCst),
            1,
            "successful publication requires real file synchronization"
        );
        assert_eq!(calls_before_release, 1);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "waiter must verify the published artifact without refetching"
        );
        first_result.unwrap().unwrap();
        let second_result = second_result.unwrap();
        let mut entries = vec![download_lock_path(&path).unwrap()];
        if waiter_pin == MODEL_FIXTURE_SHA256 {
            second_result.unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), MODEL_FIXTURE);
            entries.push(path);
        } else {
            match second_result {
                Err(InferenceError::ChecksumMismatch {
                    file,
                    expected,
                    actual,
                }) => {
                    assert_eq!(file, path.display().to_string());
                    assert_eq!(expected, waiter_pin);
                    assert_eq!(actual, MODEL_FIXTURE_SHA256);
                }
                other => panic!("waiter accepted a publication with the wrong pin: {other:?}"),
            }
            assert!(!path.exists());
        }
        assert_download_entries(temp.path(), &entries);
    }
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
enum WaiterEvent {
    Contention(bool),
    Fetch,
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
const CONTENDED_FIRST_BODY: &[u8] = b"first complete response";

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
const CONTENDED_SECOND_BODY: &[u8] = b"second complete response";

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn contended_download_observations() -> (bool, usize, usize, Vec<u8>) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::channel;
    use std::time::Duration;

    assert_ne!(
        CONTENDED_FIRST_BODY, CONTENDED_SECOND_BODY,
        "contenders must have distinct fixture bodies"
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let calls = Arc::new(AtomicUsize::new(0));
    let (ready_tx, ready_rx) = channel();
    let (resume_tx, resume_rx) = channel();
    let (event_tx, event_rx) = channel();
    let first_path = path.clone();
    let first_calls = Arc::clone(&calls);
    let first = std::thread::spawn(move || {
        download_with(&first_path, &fixture_digest(CONTENDED_FIRST_BODY), || {
            first_calls.fetch_add(1, Ordering::SeqCst);
            Ok(PausedDownloadReader {
                bytes: CONTENDED_FIRST_BODY,
                offset: 0,
                ready: Some(ready_tx),
                resume: resume_rx,
            })
        })
    });
    let ready = ready_rx.recv_timeout(Duration::from_secs(30));
    let second_path = path.clone();
    let second_calls = Arc::clone(&calls);
    let second = std::thread::spawn(move || {
        let observer_tx = event_tx.clone();
        staging::observe_next_lock(move |contended| {
            let _ = observer_tx.send(WaiterEvent::Contention(contended));
        });
        download_with(&second_path, &fixture_digest(CONTENDED_FIRST_BODY), || {
            second_calls.fetch_add(1, Ordering::SeqCst);
            let _ = event_tx.send(WaiterEvent::Fetch);
            Ok(std::io::Cursor::new(CONTENDED_SECOND_BODY))
        })
    });
    let event = event_rx.recv_timeout(Duration::from_secs(30));
    let before_release = calls.load(Ordering::SeqCst);
    let _ = resume_tx.send(());
    let first_result = first.join();
    let second_result = second.join();

    assert!(
        ready.is_ok(),
        "first body did not reach its held-lock pause"
    );
    let event = event.expect("waiter must report contention or an early fetch");
    first_result.unwrap().unwrap();
    second_result.unwrap().unwrap();
    let contended_before_fetch = matches!(event, WaiterEvent::Contention(true));
    let count = calls.load(Ordering::SeqCst);
    let published = std::fs::read(&path).unwrap();
    (contended_before_fetch, before_release, count, published)
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn assert_first_contender_published(published: &[u8]) {
    assert_eq!(
        published, CONTENDED_FIRST_BODY,
        "publication must preserve the first contender's body"
    );
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn same_target_waiter_skips_fetch_after_publication() {
    let (contended, _, calls, published) = contended_download_observations();
    assert!(
        contended,
        "waiter must observe WouldBlock while the first body is held"
    );
    assert_eq!(
        calls, 1,
        "the waiter must reuse the first complete publication"
    );
    assert_first_contender_published(&published);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn download_request_starts_only_after_target_lock() {
    let (contended, before_release, calls, published) = contended_download_observations();
    assert_eq!(
        before_release, 1,
        "a second fetch began before the first released its target lock"
    );
    assert!(
        contended,
        "the independently opened waiter handle must report WouldBlock"
    );
    assert_eq!(calls, 1);
    assert_first_contender_published(&published);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn successful_download_reclaims_stale_unique_and_legacy_stages() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    for name in [
        ".model.safetensors.part-999999-0",
        ".model.safetensors.part-2-17",
        "model.part",
    ] {
        std::fs::write(temp.path().join(name), b"stale body").unwrap();
    }
    write_download_body(&b"new complete body"[..], &path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"new complete body");
    assert_download_entries(temp.path(), &[download_lock_path(&path).unwrap(), path]);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn successful_download_preserves_other_targets_and_nonnumeric_decoys() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    let names = [
        ".tokenizer.json.part-43-2",
        "tokenizer.part",
        ".model.bin.part-43-2",
        ".model.safetensors.part-abc",
        ".model.safetensors.part-1-",
        ".model.safetensors.part--2",
        ".model.safetensors.part-1-2-3",
        ".model.safetensors.part-1-2x",
        "model.partial",
    ];
    for name in names {
        std::fs::write(temp.path().join(name), b"keep me").unwrap();
    }
    write_download_body(&b"new body"[..], &path).unwrap();
    let mut expected = vec![download_lock_path(&path).unwrap(), path];
    for name in names {
        let survivor = temp.path().join(name);
        assert_eq!(std::fs::read(&survivor).unwrap(), b"keep me", "{name}");
        expected.push(survivor);
    }
    assert_download_entries(temp.path(), &expected);
}

#[cfg(all(feature = "download", unix, not(target_arch = "wasm32")))]
#[test]
fn stale_stage_symlinks_are_unlinked_without_touching_their_targets() {
    use std::os::unix::fs::symlink;
    let temp = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let victim = outside.path().join("keep");
    std::fs::write(&victim, b"outside data").unwrap();
    symlink(&victim, temp.path().join(".model.safetensors.part-3-4")).unwrap();
    symlink(outside.path(), temp.path().join("model.part")).unwrap();
    let path = temp.path().join("model.safetensors");
    write_download_body(&b"new body"[..], &path).unwrap();
    assert_eq!(std::fs::read(victim).unwrap(), b"outside data");
    assert_download_entries(temp.path(), &[download_lock_path(&path).unwrap(), path]);
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn stale_stage_removal_error_preserves_the_destination() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    std::fs::write(&path, b"existing model").unwrap();
    let stale = temp.path().join(".model.safetensors.part-3-4");
    std::fs::create_dir(&stale).unwrap();
    std::fs::write(stale.join("keep"), b"nested data").unwrap();
    assert!(matches!(
        write_download_body(&b"new body"[..], &path),
        Err(InferenceError::Io(_))
    ));
    assert_eq!(std::fs::read(&path).unwrap(), b"existing model");
    assert_eq!(std::fs::read(stale.join("keep")).unwrap(), b"nested data");
    assert_download_entries(
        temp.path(),
        &[download_lock_path(&path).unwrap(), path, stale],
    );
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
#[test]
fn download_target_lock_contends_across_processes_and_releases_on_drop() {
    const ROOT: &str = "LATTICE_DOWNLOAD_LOCK_CHILD_ROOT";
    const HELD: &str = "LATTICE_DOWNLOAD_LOCK_CHILD_HELD";
    if let Some(root) = std::env::var_os(ROOT) {
        let path = PathBuf::from(root).join("model.safetensors");
        let probe = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(download_lock_path(&path).unwrap())
            .unwrap();
        if std::env::var_os(HELD).is_some() {
            assert!(matches!(
                probe.try_lock(),
                Err(std::fs::TryLockError::WouldBlock)
            ));
        } else {
            probe.try_lock().unwrap();
            drop(probe);
            write_download_body(&b"child complete body"[..], &path).unwrap();
        }
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    let (stage, mut file) = create_download_target(&path).unwrap();
    std::io::Write::write_all(&mut file, b"live parent stage").unwrap();
    drop(file);
    let run_child = |held| {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["download::tests::download_target_lock_contends_across_processes_and_releases_on_drop", "--exact", "--nocapture", "--test-threads=1"])
            .env(ROOT, temp.path()).env_remove(HELD);
        if held {
            command.env(HELD, "1");
        }
        let output = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "child failed: {stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    };
    run_child(true);
    assert_eq!(std::fs::read(&stage.path).unwrap(), b"live parent stage");
    drop(stage);
    run_child(false);
    assert_eq!(std::fs::read(&path).unwrap(), b"child complete body");
    assert_download_entries(temp.path(), &[download_lock_path(&path).unwrap(), path]);
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
        download_file(&url, &path, &fixture_digest(b"weights"))
            .expect("download through the proxy settings");
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
