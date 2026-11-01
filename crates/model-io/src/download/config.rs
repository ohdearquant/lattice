//! Explicit, verified materialization of a missing managed model config.

use super::canonical_model_name;
use lattice_inference_core::InferenceError;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// SHA-256 of `config.json` at each model's pinned `hf_source` revision.
// A missing pin is an explicit refusal, never permission to accept unverified bytes.
const CONFIG_SHA256: [(&str, Option<&str>); 7] = [
    (
        "bge-small-en-v1.5",
        Some("094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750"),
    ),
    (
        "bge-base-en-v1.5",
        Some("bc00af31a4a31b74040d73370aa83b62da34c90b75eb77bfa7db039d90abd591"),
    ),
    (
        "bge-large-en-v1.5",
        Some("446712fac367857b4b1302762fe1cd7bfa8b3c4b77b4dc5d77c4025407660896"),
    ),
    (
        "multilingual-e5-small",
        Some("69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959"),
    ),
    (
        "multilingual-e5-base",
        Some("9dab198f24c8c0879e481cf7822005d5ecbceedbacb390ffafa594e28d31bac4"),
    ),
    (
        "all-minilm-l6-v2",
        Some("953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41"),
    ),
    (
        "paraphrase-multilingual-minilm-l12-v2",
        Some("6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e"),
    ),
];

// Match the existing checkpoint configuration reader's 8 MiB limit without adding a dependency.
const MAX_CONFIG_BYTES: u64 = 8_388_608;
static CONFIG_TEMP_SEQ: AtomicU64 = AtomicU64::new(0);
const TEMP_ATTEMPTS: usize = 64;

/// Outcome of an explicitly requested, checksum-verified config materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigMaterialization {
    /// A matching existing config was verified and left in place.
    AlreadyVerified,
    /// A verified config was atomically published into a previously empty path.
    Created,
}

/// Materialize a missing `config.json` for a managed model from its pinned revision.
///
/// This is opt-in; [`super::ensure_model_files`] and normal model loading do not call it.
/// Existing config files are never overwritten or removed. An existing file must match the
/// pinned digest to return `AlreadyVerified`. A missing digest, offline cache miss, unavailable
/// download, checksum mismatch or unsupported atomic publication returns an error.
/// Adding a config can change a later model's effective settings and embeddings.
///
/// # Errors
///
/// Returns an error for unsupported models, missing pins, invalid or mismatched files,
/// network or I/O errors, or when no-replace publication fails.
pub fn materialize_pinned_config(
    model_name: &str,
    cache_dir: &Path,
) -> Result<ConfigMaterialization, InferenceError> {
    let model_name = canonical_model_name(model_name)?;
    let expected = CONFIG_SHA256
        .iter()
        .find(|(name, _)| *name == model_name)
        .and_then(|(_, hash)| *hash);
    let offline = crate::env_switch_enabled("LATTICE_OFFLINE");
    let model_dir = cache_dir.join(model_name);
    materialize_with(&model_dir, expected, offline, || {
        #[cfg(all(feature = "download", not(target_arch = "wasm32")))]
        {
            let (repo, revision) = super::hf_source(model_name)?;
            let url = format!("https://huggingface.co/{repo}/resolve/{revision}/config.json");
            fetch_config(&url)
        }
        #[cfg(not(all(feature = "download", not(target_arch = "wasm32"))))]
        {
            Err(InferenceError::Download(
                "config download unavailable in this build".into(),
            ))
        }
    })
}

#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
fn fetch_config(url: &str) -> Result<Box<dyn Read>, InferenceError> {
    let response = super::download_agent()
        .get(url)
        .header("User-Agent", "lattice-inference/0.3.3")
        .call()
        .map_err(|error| InferenceError::Download(format!("{url}: {error}")))?;
    Ok(Box::new(response.into_body().into_reader()))
}

fn materialize_with(
    model_dir: &Path,
    expected: Option<&str>,
    offline: bool,
    fetch: impl FnOnce() -> Result<Box<dyn Read>, InferenceError>,
) -> Result<ConfigMaterialization, InferenceError> {
    let expected = expected
        .filter(|hash| {
            hash.len() == 64
                && hash
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
        .ok_or_else(|| {
            InferenceError::Download("missing or invalid pinned config SHA-256".into())
        })?;
    let target = model_dir.join("config.json");
    if verify_existing(&target, expected)? {
        return Ok(ConfigMaterialization::AlreadyVerified);
    }
    if offline {
        return Err(InferenceError::Download(
            "config unavailable in offline mode".into(),
        ));
    }
    if !model_dir.is_dir() {
        return Err(InferenceError::Download(format!(
            "managed model directory does not exist: {}",
            model_dir.display()
        )));
    }
    let bytes = read_bounded(fetch()?)?;
    verify_bytes(&bytes, expected, &target)?;
    let (mut file, temp) = create_temp(model_dir)?;
    file.write_all(&bytes)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    match fs::hard_link(&temp.0, &target) {
        Ok(()) => Ok(ConfigMaterialization::Created),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if verify_existing(&target, expected)? {
                Ok(ConfigMaterialization::AlreadyVerified)
            } else {
                Err(InferenceError::Io(error))
            }
        }
        Err(error) => Err(InferenceError::Io(error)),
    }
}

fn read_bounded(reader: impl Read) -> Result<Vec<u8>, InferenceError> {
    let mut bytes = Vec::new();
    reader.take(MAX_CONFIG_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_CONFIG_BYTES {
        return Err(InferenceError::Download(format!(
            "config exceeds {MAX_CONFIG_BYTES} bytes"
        )));
    }
    Ok(bytes)
}

fn verify_bytes(bytes: &[u8], expected: &str, path: &Path) -> Result<(), InferenceError> {
    if bytes.is_empty() {
        return Err(InferenceError::Download("empty config is invalid".into()));
    }
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        return Err(InferenceError::ChecksumMismatch {
            file: path.display().to_string(),
            expected: expected.into(),
            actual,
        });
    }
    Ok(())
}

fn verify_existing(path: &Path, expected: &str) -> Result<bool, InferenceError> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(InferenceError::Io(error)),
        Ok(metadata) if !metadata.is_file() => {
            return Err(InferenceError::Download(format!(
                "config is not a regular file: {}",
                path.display()
            )));
        }
        Ok(_) => {}
    }
    let file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(InferenceError::Download(format!(
            "config is not a regular file: {}",
            path.display()
        )));
    }
    verify_bytes(&read_bounded(file)?, expected, path)?;
    Ok(true)
}

struct TempFile(PathBuf);
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn create_temp(dir: &Path) -> Result<(File, TempFile), InferenceError> {
    for _ in 0..TEMP_ATTEMPTS {
        let sequence = CONFIG_TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(
            ".config.json.part-{}-{sequence}",
            std::process::id()
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o644);
        }
        match options.open(&path) {
            Ok(file) => return Ok((file, TempFile(path))),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(InferenceError::Io(error)),
        }
    }
    Err(InferenceError::Download(
        "could not allocate a unique config staging file".into(),
    ))
}

#[cfg(test)]
mod tests;
