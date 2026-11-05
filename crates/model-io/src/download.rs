//! Model-file ensure/download flow, canonical model names, checksum verification, and download helper.
use lattice_inference_core::InferenceError;
use std::path::{Path, PathBuf};

mod config;
mod staging;
pub use config::{ConfigMaterialization, materialize_pinned_config};
#[cfg(all(test, feature = "download", not(target_arch = "wasm32")))]
use staging::create_target as create_download_target;
#[cfg(all(test, feature = "download", not(target_arch = "wasm32")))]
use staging::lock_path as download_lock_path;

/// Whether a directory has the files required for a model cache hit.
///
/// The directory's lexical basename must be a canonical supported model name.
/// Unknown or non-UTF8 basenames return false. The required tokenizer filename
/// comes from that model's checksum entry; another tokenizer file is insufficient.
/// This checks file presence only; it does not verify checksums or model validity.
/// Marker symlinks are followed, without resolving the directory's model name
/// through its symlink target.
pub fn has_cached_model_files(model_dir: &Path) -> bool {
    let Some(tokenizer) = model_dir
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(tokenizer_filename)
    else {
        return false;
    };
    model_dir.join("model.safetensors").exists() && model_dir.join(tokenizer).exists()
}

fn tokenizer_filename(model_name: &str) -> Option<&'static str> {
    expected_checksums(model_name)
        .tokenizer
        .map(|(filename, _)| filename)
}

/// **Unstable**: model file caching and conditional download; download feature flag and supported
/// model list are subject to change.
///
/// Ensure that the model files exist locally, downloading them if needed.
/// Cached artifacts are checksum-verified even when automatic download is unavailable.
pub fn ensure_model_files(model_name: &str, cache_dir: &Path) -> Result<PathBuf, InferenceError> {
    // Offline gate: when LATTICE_OFFLINE is enabled, never touch the network — a cache
    // miss fails fast instead of implicitly fetching from Hugging Face. Downstream
    // consumers (khive CI, sandboxed builds) set this. Reading the env here keeps the
    // public signature stable while `ensure_model_files_inner` stays env-free and
    // unit-testable. By value, not presence: `LATTICE_OFFLINE=0` must mean "downloads
    // are fine", not silently enable fail-closed offline mode.
    let offline = crate::env_switch_enabled("LATTICE_OFFLINE");
    ensure_model_files_inner(model_name, cache_dir, offline)
}

fn ensure_model_files_inner(
    model_name: &str,
    cache_dir: &Path,
    offline: bool,
) -> Result<PathBuf, InferenceError> {
    let model_name = canonical_model_name(model_name)?;
    ensure_model_files_inner_with_checksums(
        model_name,
        cache_dir,
        offline,
        expected_checksums(model_name),
    )
}

fn ensure_model_files_inner_with_checksums(
    model_name: &str,
    cache_dir: &Path,
    offline: bool,
    expected: ExpectedChecksums,
) -> Result<PathBuf, InferenceError> {
    let model_dir = cache_dir.join(model_name);

    if has_cached_model_files(&model_dir) {
        // Prefetched caches may be offline or read-only, so reject corrupt bytes
        // without deleting the caller-managed artifact.
        verify_checksums(
            model_name,
            &model_dir,
            ChecksumFailureAction::Preserve,
            expected,
        )?;
        tracing::debug!(path = %model_dir.display(), "model files already cached");
        return Ok(model_dir);
    }

    // Offline mode blocks the network entirely: a cache miss is a hard, clear error
    // rather than a download attempt. Applies whether or not the `download` feature
    // is compiled in.
    if offline {
        return Err(InferenceError::ModelNotFound(format!(
            "model files not found at {} and LATTICE_OFFLINE is set (offline mode: no \
             download attempted). Pre-fetch the model into the cache, or unset \
             LATTICE_OFFLINE to allow downloading.",
            model_dir.display()
        )));
    }

    // Downloads are unavailable when the `download` feature is off, and also on
    // wasm32 even with the feature on: this crate gates its ureq/rustls/ring stack
    // to non-wasm targets (Cargo.toml), so the fetch path below is compiled out
    // there. This complementary gate is what makes a wasm consumer that forwards
    // `lattice-inference/download` resolve download-free instead of failing.
    #[cfg(not(all(feature = "download", not(target_arch = "wasm32"))))]
    {
        Err(InferenceError::ModelNotFound(format!(
            "model files not found at {} and automatic download is unavailable in this \
             build (the `download` feature is off, or this is a wasm target). Pre-fetch \
             the model files into the cache.",
            model_dir.display()
        )))
    }

    #[cfg(all(feature = "download", not(target_arch = "wasm32")))]
    {
        let safetensors_path = model_dir.join("model.safetensors");
        let tokenizer_file = tokenizer_filename(model_name)
            .ok_or_else(|| InferenceError::UnsupportedModel(model_name.to_owned()))?;
        let tokenizer_path = model_dir.join(tokenizer_file);
        create_model_dir(&model_dir)?;

        let (hf_model_id, revision) = hf_source(model_name)?;

        let base_url = format!("https://huggingface.co/{hf_model_id}/resolve/{revision}");

        if !safetensors_path.exists() {
            let url = format!("{base_url}/model.safetensors");
            tracing::info!(%url, "downloading model.safetensors");
            download_file(&url, &safetensors_path)?;
        }

        if !tokenizer_path.exists() {
            let url = format!("{base_url}/{tokenizer_file}");
            tracing::info!(%url, "downloading tokenizer");
            download_file(&url, &tokenizer_path)?;
        }

        verify_checksums(
            model_name,
            &model_dir,
            ChecksumFailureAction::Remove,
            expected,
        )?;
        Ok(model_dir)
    }
}

// The Hugging Face repository and pinned commit for each supported model.
// Each revision is the commit whose files match `expected_checksums`. Fetching a
// branch name instead would turn any upstream change into a checksum refusal on
// first download, so a revision moves only together with its checksums.
#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn hf_source(model_name: &str) -> Result<(&'static str, &'static str), InferenceError> {
    match model_name {
        "bge-small-en-v1.5" => Ok((
            "BAAI/bge-small-en-v1.5",
            "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        )),
        "bge-base-en-v1.5" => Ok((
            "BAAI/bge-base-en-v1.5",
            "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        )),
        "bge-large-en-v1.5" => Ok((
            "BAAI/bge-large-en-v1.5",
            "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        )),
        "multilingual-e5-small" => Ok((
            "intfloat/multilingual-e5-small",
            "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        )),
        "multilingual-e5-base" => Ok((
            "intfloat/multilingual-e5-base",
            "d128750597153bb5987e10b1c3493a34e5a4502a",
        )),
        "all-minilm-l6-v2" => Ok((
            "sentence-transformers/all-MiniLM-L6-v2",
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        )),
        "paraphrase-multilingual-minilm-l12-v2" => Ok((
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        )),
        other => Err(InferenceError::UnsupportedModel(other.to_string())),
    }
}

fn canonical_model_name(model_name: &str) -> Result<&str, InferenceError> {
    match model_name {
        "bge-small-en-v1.5" | "BAAI/bge-small-en-v1.5" => Ok("bge-small-en-v1.5"),
        "bge-base-en-v1.5" | "BAAI/bge-base-en-v1.5" => Ok("bge-base-en-v1.5"),
        "bge-large-en-v1.5" | "BAAI/bge-large-en-v1.5" => Ok("bge-large-en-v1.5"),
        "multilingual-e5-small" | "intfloat/multilingual-e5-small" => Ok("multilingual-e5-small"),
        "multilingual-e5-base" | "intfloat/multilingual-e5-base" => Ok("multilingual-e5-base"),
        "all-minilm-l6-v2" | "sentence-transformers/all-MiniLM-L6-v2" => Ok("all-minilm-l6-v2"),
        "paraphrase-multilingual-minilm-l12-v2"
        | "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" => {
            Ok("paraphrase-multilingual-minilm-l12-v2")
        }
        other => Err(InferenceError::UnsupportedModel(other.to_string())),
    }
}

#[derive(Clone, Copy)]
enum ChecksumFailureAction {
    Preserve,
    #[cfg(all(feature = "download", not(target_arch = "wasm32")))]
    Remove,
}

impl ChecksumFailureAction {
    fn removes_file(self) -> bool {
        #[cfg(all(feature = "download", not(target_arch = "wasm32")))]
        {
            matches!(self, Self::Remove)
        }
        #[cfg(not(all(feature = "download", not(target_arch = "wasm32"))))]
        {
            let _ = self;
            false
        }
    }
}

fn verify_checksums(
    model_name: &str,
    model_dir: &Path,
    failure_action: ChecksumFailureAction,
    expected: ExpectedChecksums,
) -> Result<(), InferenceError> {
    if let Some(expected_model_sha) = expected.model_safetensors {
        verify_file_checksum(
            &model_dir.join("model.safetensors"),
            expected_model_sha,
            failure_action,
        )?;
    } else {
        // Refuse to proceed without a checksum -- silent acceptance of unverified
        // downloads is a supply-chain risk. When the real SHA-256 is obtained from
        // Hugging Face, add it to `expected_checksums()`.
        return Err(InferenceError::Download(format!(
            "No checksum available for model '{model_name}' model.safetensors. \
             Cannot verify integrity. Add the SHA-256 hash to expected_checksums()."
        )));
    }

    if let Some((tokenizer_file, expected_sha)) = expected.tokenizer {
        verify_file_checksum(
            &model_dir.join(tokenizer_file),
            expected_sha,
            failure_action,
        )?;
    } else {
        return Err(InferenceError::Download(format!(
            "No checksum available for model '{model_name}' tokenizer. \
             Cannot verify integrity. Add the SHA-256 hash to expected_checksums()."
        )));
    }

    Ok(())
}

fn verify_file_checksum(
    path: &Path,
    expected: &str,
    failure_action: ChecksumFailureAction,
) -> Result<(), InferenceError> {
    let actual = sha256_hex(path)?;
    if actual != expected {
        if failure_action.removes_file() {
            let _ = std::fs::remove_file(path);
        }
        return Err(InferenceError::ChecksumMismatch {
            file: path.display().to_string(),
            expected: expected.to_string(),
            actual,
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ExpectedChecksums {
    model_safetensors: Option<&'static str>,
    /// vocab.txt for WordPiece models (BGE), tokenizer.json for SentencePiece models (E5).
    tokenizer: Option<(&'static str, &'static str)>, // (filename, sha256)
}

fn expected_checksums(model_name: &str) -> ExpectedChecksums {
    match model_name {
        // model.safetensors SHA-256 from the Hugging Face LFS pointer.
        // vocab.txt SHA-256 computed from the raw file.
        "bge-small-en-v1.5" => ExpectedChecksums {
            model_safetensors: Some(
                "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
            ),
            tokenizer: Some((
                "vocab.txt",
                "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
            )),
        },
        "bge-base-en-v1.5" => ExpectedChecksums {
            model_safetensors: Some(
                "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7",
            ),
            tokenizer: Some((
                "vocab.txt",
                "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
            )),
        },
        "bge-large-en-v1.5" => ExpectedChecksums {
            model_safetensors: Some(
                "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7",
            ),
            tokenizer: Some((
                "vocab.txt",
                "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
            )),
        },
        "multilingual-e5-small" => ExpectedChecksums {
            model_safetensors: Some(
                "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477",
            ),
            tokenizer: Some((
                "tokenizer.json",
                "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
            )),
        },
        "multilingual-e5-base" => ExpectedChecksums {
            model_safetensors: Some(
                "a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7",
            ),
            tokenizer: Some((
                "tokenizer.json",
                "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
            )),
        },
        "all-minilm-l6-v2" => ExpectedChecksums {
            model_safetensors: Some(
                "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
            ),
            tokenizer: Some((
                "vocab.txt",
                "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
            )),
        },
        "paraphrase-multilingual-minilm-l12-v2" => ExpectedChecksums {
            model_safetensors: Some(
                "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b",
            ),
            tokenizer: Some((
                "tokenizer.json",
                "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
            )),
        },
        _ => ExpectedChecksums {
            model_safetensors: None,
            tokenizer: None,
        },
    }
}

/// The client ureq 2 gave this module by default: no proxy from the
/// environment, a 30 s connect timeout, and at most 5 redirects. ureq 3
/// reads proxies from the environment, has no connect timeout and follows
/// 10 redirects unless told otherwise.
#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn download_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .proxy(None)
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .max_redirects(5)
        .build()
        .into()
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn download_file(url: &str, path: &Path) -> Result<(), InferenceError> {
    download_with(path, || {
        let agent = download_agent();
        let response = agent
            .get(url)
            .header("User-Agent", "lattice-inference/0.3.3")
            .call()
            .map_err(|e| InferenceError::Download(format!("{url}: {e}")))?;

        Ok(response.into_body().into_reader())
    })
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn download_with<R: std::io::Read>(
    path: &Path,
    fetch: impl FnOnce() -> Result<R, InferenceError>,
) -> Result<(), InferenceError> {
    let lock = staging::lock_target(path)?;
    if path.is_file() {
        return Ok(());
    }
    let reader = fetch()?;
    write_download_body_locked(reader, path, lock)
}

#[cfg(all(test, feature = "download", not(target_arch = "wasm32")))]
fn write_download_body(reader: impl std::io::Read, path: &Path) -> Result<(), InferenceError> {
    write_download_body_locked(reader, path, staging::lock_target(path)?)
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn write_download_body_locked(
    mut reader: impl std::io::Read,
    path: &Path,
    lock: staging::TargetLock,
) -> Result<(), InferenceError> {
    use std::io::Write;

    let (stage, mut file) = lock.create_stage()?;
    let written = std::io::copy(&mut reader, &mut file).and_then(|_| file.flush());
    drop(file);
    written?;
    stage.replace(path)?;
    Ok(())
}

// The mmap trust gate refuses a checkpoint whose file or any ancestor directory is
// group- or other-writable. The default creation modes (0o777 and 0o666 minus the
// umask) produce exactly that under a group-writable umask such as 0002, so the
// download would be refused by the loader it was fetched for.
#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn create_model_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o755)
            .create(dir)
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(dir)
    }
}

fn sha256_hex(path: &Path) -> Result<String, InferenceError> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    let digest = hasher.finalize();
    let bytes: &[u8] = digest.as_ref();
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{byte:02x}");
    }
    Ok(hex)
}

#[cfg(test)]
mod tests;
