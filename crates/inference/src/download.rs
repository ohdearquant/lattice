//! Model-file ensure/download flow, canonical model names, checksum verification, and download helper.
use crate::error::InferenceError;
use std::path::{Path, PathBuf};

/// **Unstable**: model file caching and conditional download; download feature flag and supported
/// model list are subject to change.
///
/// Ensure that the model files exist locally, downloading them if needed.
pub fn ensure_model_files(model_name: &str, cache_dir: &Path) -> Result<PathBuf, InferenceError> {
    // Offline gate: when LATTICE_OFFLINE is set, never touch the network — a cache miss
    // fails fast instead of implicitly fetching from Hugging Face. Downstream consumers
    // (khive CI, sandboxed builds) set this. Reading the env here keeps the public
    // signature stable while `ensure_model_files_inner` stays env-free and unit-testable.
    let offline = std::env::var_os("LATTICE_OFFLINE").is_some();
    ensure_model_files_inner(model_name, cache_dir, offline)
}

fn ensure_model_files_inner(
    model_name: &str,
    cache_dir: &Path,
    offline: bool,
) -> Result<PathBuf, InferenceError> {
    let model_name = canonical_model_name(model_name)?;
    let model_dir = cache_dir.join(model_name);
    let safetensors_path = model_dir.join("model.safetensors");
    let vocab_path = model_dir.join("vocab.txt");
    let tokenizer_json_path = model_dir.join("tokenizer.json");

    // Check if model files are cached — accept either vocab.txt or tokenizer.json
    let has_tokenizer = vocab_path.exists() || tokenizer_json_path.exists();
    if safetensors_path.exists() && has_tokenizer {
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
    // wasm32 even with the feature on: inference gates its ureq/rustls/ring stack
    // to non-wasm targets (Cargo.toml), so the fetch path below is compiled out
    // there. This complementary gate is what makes a wasm consumer that forwards
    // `lattice-inference/download` resolve download-free instead of failing.
    #[cfg(not(all(feature = "download", not(target_arch = "wasm32"))))]
    {
        return Err(InferenceError::ModelNotFound(format!(
            "model files not found at {} and automatic download is unavailable in this \
             build (the `download` feature is off, or this is a wasm target). Pre-fetch \
             the model files into the cache.",
            model_dir.display()
        )));
    }

    #[cfg(all(feature = "download", not(target_arch = "wasm32")))]
    {
        std::fs::create_dir_all(&model_dir)?;

        let hf_model_id = match model_name {
            "bge-small-en-v1.5" => "BAAI/bge-small-en-v1.5",
            "bge-base-en-v1.5" => "BAAI/bge-base-en-v1.5",
            "bge-large-en-v1.5" => "BAAI/bge-large-en-v1.5",
            "multilingual-e5-small" => "intfloat/multilingual-e5-small",
            "multilingual-e5-base" => "intfloat/multilingual-e5-base",
            "all-minilm-l6-v2" => "sentence-transformers/all-MiniLM-L6-v2",
            "paraphrase-multilingual-minilm-l12-v2" => {
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            }
            other => return Err(InferenceError::UnsupportedModel(other.to_string())),
        };

        let base_url = format!("https://huggingface.co/{hf_model_id}/resolve/main");

        if !safetensors_path.exists() {
            let url = format!("{base_url}/model.safetensors");
            tracing::info!(%url, "downloading model.safetensors");
            download_file(&url, &safetensors_path)?;
        }

        // E5 and multilingual models use tokenizer.json (sentencepiece); BGE/MiniLM use vocab.txt (WordPiece)
        let uses_tokenizer_json = model_name.contains("e5-") || model_name.contains("multilingual");
        if uses_tokenizer_json {
            if !tokenizer_json_path.exists() {
                let url = format!("{base_url}/tokenizer.json");
                tracing::info!(%url, "downloading tokenizer.json");
                download_file(&url, &tokenizer_json_path)?;
            }
        } else if !vocab_path.exists() {
            let url = format!("{base_url}/vocab.txt");
            tracing::info!(%url, "downloading vocab.txt");
            download_file(&url, &vocab_path)?;
        }

        verify_checksums(model_name, &model_dir)?;
        Ok(model_dir)
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

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn verify_checksums(model_name: &str, model_dir: &Path) -> Result<(), InferenceError> {
    let expected = expected_checksums(model_name);

    if let Some(expected_model_sha) = expected.model_safetensors {
        verify_file_checksum(&model_dir.join("model.safetensors"), expected_model_sha)?;
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
        verify_file_checksum(&model_dir.join(tokenizer_file), expected_sha)?;
    } else {
        return Err(InferenceError::Download(format!(
            "No checksum available for model '{model_name}' tokenizer. \
             Cannot verify integrity. Add the SHA-256 hash to expected_checksums()."
        )));
    }

    Ok(())
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn verify_file_checksum(path: &Path, expected: &str) -> Result<(), InferenceError> {
    let actual = sha256_hex(path)?;
    if actual != expected {
        let _ = std::fs::remove_file(path);
        return Err(InferenceError::ChecksumMismatch {
            file: path.display().to_string(),
            expected: expected.to_string(),
            actual,
        });
    }
    Ok(())
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
struct ExpectedChecksums {
    model_safetensors: Option<&'static str>,
    /// vocab.txt for WordPiece models (BGE), tokenizer.json for SentencePiece models (E5).
    tokenizer: Option<(&'static str, &'static str)>, // (filename, sha256)
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
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

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn download_file(url: &str, path: &Path) -> Result<(), InferenceError> {
    use std::io::Write;

    let response = ureq::get(url)
        .set("User-Agent", "lattice-inference/0.3.3")
        .call()
        .map_err(|e| InferenceError::Download(format!("{url}: {e}")))?;

    let tmp_path = path.with_extension("part");
    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(&tmp_path)?;
    std::io::copy(&mut reader, &mut file)?;
    file.flush()?;
    drop(file);
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
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
mod tests {
    use super::*;

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

    /// Offline mode still serves a populated cache — it blocks the network, not cache reads.
    #[test]
    fn offline_cache_hit_succeeds() {
        let tmp = std::env::temp_dir().join(format!("lattice_offline_hit_{}", std::process::id()));
        let model_dir = tmp.join("all-minilm-l6-v2");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&model_dir).expect("create temp model dir");
        std::fs::write(model_dir.join("model.safetensors"), b"stub")
            .expect("write safetensors stub");
        std::fs::write(model_dir.join("vocab.txt"), b"stub").expect("write vocab stub");
        let res = ensure_model_files_inner("all-minilm-l6-v2", &tmp, true);
        assert!(res.is_ok(), "offline + cache hit must succeed, got {res:?}");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
