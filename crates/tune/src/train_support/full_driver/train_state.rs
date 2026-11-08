//! Checkpoint of a training run between invocations: the optimizer moments and
//! the step cursor, bound to the data, the configuration and the adapter they
//! were saved beside.
//!
//! The adapter file alone is not enough to continue a run. It carries the LoRA
//! weights, but not Adam's first and second moments or the per-key step counts,
//! so a run restarted from the adapter takes its next steps with bias correction
//! reset to step one and the momentum gone: a different optimizer trajectory
//! from the uninterrupted run. This file carries the rest.
//!
//! Every binding is checked on load, and a mismatch refuses rather than
//! continuing: a state saved for different data, a different configuration or a
//! different adapter would silently produce a run that is none of the runs it
//! looks like.

use std::io::Write;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::lora::AdamState;
use crate::lora::train_core::{Dims, GdnLoraParams, GdnModuleSelection, LoraParams};
use crate::train_support::Sample;

const MAGIC: &[u8; 8] = b"LTTRSTAT";
const VERSION: u32 = 2;
const DIGEST_LEN: usize = 32;
/// Magic, version, next step, train count and three fingerprints.
const HEADER_LEN: usize = 8 + 4 + 8 + 8 + 3 * DIGEST_LEN;

/// Saved state of one training run, read back by `--load-state`.
pub(crate) struct TrainState {
    /// Global 1-based index of the next step to run.
    pub next_step: u64,
    /// Number of training samples the run cycled through.
    pub train_count: u64,
    pub data: [u8; DIGEST_LEN],
    pub config: [u8; DIGEST_LEN],
    pub adapter: [u8; DIGEST_LEN],
    pub adam: AdamState,
}

impl TrainState {
    /// Encode the state with a trailing SHA-256 over every preceding byte.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let adam = self.adam.to_bytes();
        let mut out = Vec::with_capacity(HEADER_LEN + 8 + adam.len() + DIGEST_LEN);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.next_step.to_le_bytes());
        out.extend_from_slice(&self.train_count.to_le_bytes());
        out.extend_from_slice(&self.data);
        out.extend_from_slice(&self.config);
        out.extend_from_slice(&self.adapter);
        out.extend_from_slice(&(adam.len() as u64).to_le_bytes());
        out.extend_from_slice(&adam);
        let digest = Sha256::digest(&out);
        out.extend_from_slice(&digest);
        out
    }

    /// Decode a state written by [`TrainState::to_bytes`], refusing a damaged,
    /// truncated or foreign file before any field is trusted.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_LEN + 8 + DIGEST_LEN {
            return Err(format!(
                "training state is {} bytes, shorter than its fixed header",
                bytes.len()
            ));
        }
        let (body, digest) = bytes.split_at(bytes.len() - DIGEST_LEN);
        if Sha256::digest(body).as_slice() != digest {
            return Err("training state checksum does not match its contents".to_string());
        }
        if &body[..8] != MAGIC {
            return Err("file is not a training state".to_string());
        }
        let u32_at =
            |at: usize| u32::from_le_bytes([body[at], body[at + 1], body[at + 2], body[at + 3]]);
        let u64_at = |at: usize| {
            let mut a = [0u8; 8];
            a.copy_from_slice(&body[at..at + 8]);
            u64::from_le_bytes(a)
        };
        let digest_at = |at: usize| {
            let mut a = [0u8; DIGEST_LEN];
            a.copy_from_slice(&body[at..at + DIGEST_LEN]);
            a
        };
        let version = u32_at(8);
        if version == 1 {
            return Err(
                "training state version 1 predates checkpoint binding; start a new run".into(),
            );
        }
        if version != VERSION {
            return Err(format!(
                "training state version {version} is not the supported version {VERSION}"
            ));
        }
        let next_step = u64_at(12);
        if next_step == 0 {
            return Err("training state names step 0 as the next step".to_string());
        }
        let train_count = u64_at(20);
        let data = digest_at(28);
        let config = digest_at(28 + DIGEST_LEN);
        let adapter = digest_at(28 + 2 * DIGEST_LEN);
        let adam_len = u64_at(HEADER_LEN);
        let adam_bytes = &body[HEADER_LEN + 8..];
        if adam_len != adam_bytes.len() as u64 {
            return Err(format!(
                "training state declares {adam_len} optimizer bytes and carries {}",
                adam_bytes.len()
            ));
        }
        let adam = AdamState::from_bytes(adam_bytes).map_err(|e| e.to_string())?;
        Ok(Self {
            next_step,
            train_count,
            data,
            config,
            adapter,
            adam,
        })
    }

    /// Write the state beside `path` and rename it into place, so an interrupted
    /// write leaves the previous file intact rather than a truncated one.
    pub(crate) fn write_atomic(&self, path: &Path) -> Result<(), String> {
        let mut tmp_name = path
            .file_name()
            .ok_or_else(|| format!("state path {} has no file name", path.display()))?
            .to_os_string();
        tmp_name.push(".partial");
        let tmp = path.with_file_name(tmp_name);
        let bytes = self.to_bytes();
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW).mode(0o600);
        }
        let mut file = options
            .open(&tmp)
            .map_err(|e| format!("create {}: {e}", tmp.display()))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|e| format!("write {}: {e}", tmp.display()))?;
        drop(file);
        std::fs::rename(&tmp, path)
            .map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), path.display()))
    }

    /// Read and decode the state at `path`.
    pub(crate) fn read(path: &Path) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("read state {}: {e}", path.display()))?;
        Self::from_bytes(&bytes).map_err(|e| format!("state {}: {e}", path.display()))
    }

    pub(crate) fn validate_config(&self, config: &[u8; DIGEST_LEN]) -> Result<(), String> {
        if &self.config != config {
            return Err(
                "saved by a run with a different base checkpoint or different settings \
                 (layers, rank, alpha, lr, seed, GDN modules, geometry or Adam parameters)"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Bind the config bytes and the sorted shard names and lengths without reading
/// full weights. Same-length weight changes are not detected.
pub(crate) fn checkpoint_fingerprint(model_dir: &Path) -> Result<[u8; DIGEST_LEN], String> {
    use lattice_inference::weights::{contained_shard_path, parse_index};

    let config_path = model_dir.join("config.json");
    let config = std::fs::read(&config_path)
        .map_err(|e| format!("read checkpoint config {}: {e}", config_path.display()))?;
    let names: std::collections::BTreeSet<String> = if model_dir.join("model.safetensors").exists()
    {
        ["model.safetensors".to_string()].into_iter().collect()
    } else {
        parse_index(model_dir)
            .map_err(|e| format!("read checkpoint index {}: {e}", model_dir.display()))?
            .weight_map
            .into_values()
            .collect()
    };
    let mut shards = Vec::with_capacity(names.len());
    for name in names {
        let path = contained_shard_path(model_dir, &name).map_err(|e| e.to_string())?;
        let metadata = std::fs::metadata(&path)
            .map_err(|e| format!("read checkpoint shard metadata {}: {e}", path.display()))?;
        if !metadata.is_file() {
            return Err(format!("checkpoint shard {} is not a file", path.display()));
        }
        shards.push((name, metadata.len()));
    }
    let mut h = Sha256::new();
    h.update(b"lattice-train-checkpoint-v1");
    h.update(Sha256::digest(&config));
    h.update((shards.len() as u64).to_le_bytes());
    for (name, len) in shards {
        h.update((name.len() as u64).to_le_bytes());
        h.update(name.as_bytes());
        h.update(len.to_le_bytes());
    }
    Ok(h.finalize().into())
}

/// Fingerprint of the loaded training samples, in load order: every token and
/// every completion boundary.
pub(crate) fn data_fingerprint(samples: &[Sample]) -> [u8; DIGEST_LEN] {
    let mut h = Sha256::new();
    h.update(b"lattice-train-data-v1");
    h.update((samples.len() as u64).to_le_bytes());
    for s in samples {
        h.update((s.tokens.len() as u64).to_le_bytes());
        for t in &s.tokens {
            h.update(t.to_le_bytes());
        }
        h.update((s.completion_start as u64).to_le_bytes());
    }
    h.finalize().into()
}

/// The run settings a continuation must share with the run it continues.
pub(crate) struct RunIdentity<'a> {
    pub checkpoint: [u8; DIGEST_LEN],
    pub first_layer: usize,
    pub rank: usize,
    pub alpha: f32,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps_adam: f32,
    pub seed: u64,
    pub gdn_modules: GdnModuleSelection,
    pub dims: &'a Dims,
    pub slot_layers: &'a [usize],
    pub gdn_slot_layers: &'a [usize],
}

/// Fingerprint of the settings that define the run's optimizer trajectory.
pub(crate) fn config_fingerprint(id: &RunIdentity<'_>) -> [u8; DIGEST_LEN] {
    let mut h = Sha256::new();
    h.update(b"lattice-train-config-v2");
    h.update(id.checkpoint);
    for v in [
        id.first_layer,
        id.rank,
        id.dims.hidden,
        id.dims.vocab,
        id.dims.num_q_heads,
        id.dims.num_kv_heads,
        id.dims.head_dim,
        id.dims.rope_dim,
        id.dims.inter,
        id.dims.q_dim,
        id.dims.kv_dim,
    ] {
        h.update((v as u64).to_le_bytes());
    }
    h.update(id.alpha.to_bits().to_le_bytes());
    h.update(id.lr.to_bits().to_le_bytes());
    h.update(id.dims.eps.to_bits().to_le_bytes());
    h.update(id.beta1.to_bits().to_le_bytes());
    h.update(id.beta2.to_bits().to_le_bytes());
    h.update(id.eps_adam.to_bits().to_le_bytes());
    h.update(id.seed.to_le_bytes());
    for m in id.gdn_modules.modules() {
        h.update((m.len() as u64).to_le_bytes());
        h.update(m.as_bytes());
    }
    for layers in [id.slot_layers, id.gdn_slot_layers] {
        h.update((layers.len() as u64).to_le_bytes());
        for &l in layers {
            h.update((l as u64).to_le_bytes());
        }
    }
    h.finalize().into()
}

/// Fingerprint of every adapter value the run trains and saves, in slot order.
///
/// Covers the same projections the adapter file carries, so a state checked
/// against an adapter loaded from disk compares exactly what that file holds.
pub(crate) fn adapter_fingerprint(
    loras: &[LoraParams],
    gdn_loras: &[GdnLoraParams],
    selection: GdnModuleSelection,
) -> [u8; DIGEST_LEN] {
    fn put(h: &mut Sha256, values: &[f32]) {
        h.update((values.len() as u64).to_le_bytes());
        for v in values {
            h.update(v.to_bits().to_le_bytes());
        }
    }
    let mut h = Sha256::new();
    h.update(b"lattice-train-adapter-v1");
    for p in loras {
        for values in [&p.a_q, &p.b_q, &p.a_v, &p.b_v] {
            put(&mut h, values);
        }
    }
    for g in gdn_loras {
        for values in [&g.a_qkv, &g.b_qkv, &g.a_z, &g.b_z, &g.a_out, &g.b_out] {
            put(&mut h, values);
        }
        if selection.trains_fused_kernel_projections() {
            for values in [&g.a_b, &g.b_b, &g.a_a, &g.b_a] {
                put(&mut h, values);
            }
        }
    }
    h.finalize().into()
}

/// Indices of the training samples steps `start ..= start + steps - 1` visit,
/// ascending, each once. Step `k` (1-based, global) trains sample
/// `(k - 1) % count`; with no steps the window is the sample the next step
/// would visit, so the consistency check still has a sample to run on.
pub(crate) fn step_window(start: usize, steps: usize, count: usize) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    let mut window: Vec<usize> = (0..steps.min(count).max(1))
        .map(|k| (start - 1 + k) % count)
        .collect();
    window.sort_unstable();
    window
}

#[cfg(test)]
mod tests;
