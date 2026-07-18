//! Stream quantizer: converts sharded BF16 safetensors → Q4_0 `.q4` files.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --bin quantize_q4 -- \
//!   --model-dir ~/.lattice/models/qwen3.6-27b \
//!   --output-dir ~/.lattice/models/qwen3.6-27b-q4
//! ```
//!
//! # Memory budget
//!
//! At any point only one tensor's decoded `f64` values are live in RAM
//! alongside its `f32` downcast and Q4 output.

use lattice_inference::quant::quarot::QuarotTensorReader;
use lattice_inference::quant::quarot::convert::encode_f16_payload;
use lattice_inference::weights::q4_weights::{Q4_BLOCK_BYTES, q4_file_bytes, quantize_f32_to_q4};
use std::ffi::CString;
use std::fs;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Tensor classification: should_quantize
// ---------------------------------------------------------------------------

/// Returns `true` for large weight matrices that benefit from Q4_0 quantization.
///
/// Rule: quantize weight matrices for projections, MLP layers, embeddings, and lm_head.
/// Keep scalars, norms, biases, conv1d weights, and Mamba-specific parameters in f16.
fn should_quantize(name: &str) -> bool {
    // MoE routed-expert tensors are stored as one fused array per layer,
    // shape `[num_experts, out_features, in_features]`, WITHOUT a trailing
    // `.weight` suffix (e.g. `model.language_model.layers.0.mlp.experts
    // .gate_up_proj` / `.down_proj`, and the analogous `mtp.layers.N.mlp
    // .experts.*` tensors). They fail the `.weight`-suffix gate below and
    // were silently skipped, even though they hold the large majority of
    // parameters in a MoE checkpoint (~92% for a 256-expert model). Match
    // them explicitly before the suffix gate. The sibling `shared_expert.*`
    // and `shared_expert_gate` tensors already carry a `.weight` suffix and
    // are already covered by the rules below.
    if name.ends_with(".experts.gate_up_proj") || name.ends_with(".experts.down_proj") {
        return true;
    }

    // Must be a weight tensor.
    if !name.ends_with(".weight") && !name.ends_with("lm_head.weight") {
        return false;
    }

    // Always quantize these large matrices.
    if name.ends_with("_proj.weight")
        || name.ends_with("_proj_a.weight")
        || name.ends_with("_proj_b.weight")
        || name.ends_with("_proj_qkv.weight")
        || name.ends_with("_proj_z.weight")
        || name.ends_with("gate_proj.weight")
        || name.ends_with("up_proj.weight")
        || name.ends_with("down_proj.weight")
        || name.ends_with("lm_head.weight")
        || name.ends_with("embed_tokens.weight")
    {
        return true;
    }

    // Keep small / special tensors in f16 (norms, conv1d, biases).
    // These checks shadow the weight-check above for norm weights, which are small.
    if name.contains("norm.weight")
        || name.contains("norm_")
        || name.ends_with(".bias")
        || name.ends_with("A_log")
        || name.ends_with("dt_bias")
        || name.ends_with("conv1d.weight")
    {
        return false;
    }

    // Default: quantize unknown weight matrices.
    true
}

// ---------------------------------------------------------------------------
// Output index
// ---------------------------------------------------------------------------

/// Index entry recorded in `quantize_index.json`.
#[derive(serde::Serialize)]
struct IndexEntry {
    /// Source tensor name.
    name: String,
    /// Output file stem (relative to output directory).
    file: String,
    /// Whether the tensor was quantized (true) or saved as f16 (false).
    quantized: bool,
    /// Original shape.
    shape: Vec<usize>,
    /// Number of original elements.
    numel: usize,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn print_usage_and_exit() -> ! {
    eprintln!("Usage: quantize_q4 --model-dir <DIR> --output-dir <DIR> [--dry-run]");
    eprintln!();
    eprintln!("  --model-dir   directory containing model.safetensors[.index.json]");
    eprintln!("  --output-dir  directory to write .q4 and index files");
    eprintln!("  --dry-run     read tensors but skip writing output");
    std::process::exit(1);
}

fn main() {
    if let Err(e) = run() {
        eprintln!("quantize_q4 failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut dry_run = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                model_dir = Some(PathBuf::from(args.get(i).unwrap_or_else(|| {
                    eprintln!("--model-dir requires an argument");
                    print_usage_and_exit();
                })));
            }
            "--output-dir" => {
                i += 1;
                output_dir = Some(PathBuf::from(args.get(i).unwrap_or_else(|| {
                    eprintln!("--output-dir requires an argument");
                    print_usage_and_exit();
                })));
            }
            "--dry-run" => dry_run = true,
            other => {
                eprintln!("Unknown argument: {other}");
                print_usage_and_exit();
            }
        }
        i += 1;
    }

    let model_dir = model_dir.unwrap_or_else(|| {
        eprintln!("--model-dir is required");
        print_usage_and_exit();
    });
    let output_dir = output_dir.unwrap_or_else(|| {
        eprintln!("--output-dir is required");
        print_usage_and_exit();
    });

    quantize_model(&model_dir, &output_dir, dry_run)
}

/// Open `dir` as a directory fd bound to that exact inode, refusing to
/// follow a symlink.
///
/// This is the fd-bind primitive every staging-directory write in this file
/// routes through: a `Path` re-resolves at the moment of each syscall, so an
/// attacker who can replace `dir` (or a child within it) between `mkdir` and
/// a later path-based write redirects that write to wherever the path now
/// points. A directory fd, once opened, is bound to the inode — later
/// `openat` calls against this fd always land inside the directory that was
/// actually created, regardless of what the path currently resolves to.
/// `O_NOFOLLOW` additionally rejects the open outright if `dir` was swapped
/// for a symlink between `mkdir` and this call.
fn open_dir_nofollow(dir: &Path) -> Result<OwnedFd, Box<dyn std::error::Error>> {
    let cpath = CString::new(dir.as_os_str().as_bytes()).map_err(|e| {
        format!(
            "staging directory path {} contains an interior NUL byte: {e}",
            dir.display()
        )
    })?;
    // SAFETY: `cpath` is a valid NUL-terminated C string owned for the
    // duration of this call. `libc::open` returns either a valid owned fd
    // (>= 0) or -1 with errno set; the -1 case is checked immediately below
    // before any use of `raw`.
    let raw = unsafe {
        libc::open(
            cpath.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(format!(
            "failed to open staging directory {} (O_DIRECTORY|O_NOFOLLOW): {}",
            dir.display(),
            std::io::Error::last_os_error()
        )
        .into());
    }
    // SAFETY: `raw` was just returned by `libc::open` above, checked
    // non-negative, and is not owned or closed anywhere else — `OwnedFd`
    // becomes its sole owner and will close it on drop.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// Read `name` (a plain file name) from the directory bound to `dirfd` via
/// `openat(..., O_NOFOLLOW)`, never a path-based `fs::read`. `fs::read`
/// follows a symlink at the final path component; a hostile or
/// concurrently-swapped model directory could plant one at `config.json`
/// pointing at an arbitrary readable file, which `fs::read` would then copy
/// into the output directory.
fn read_file_at_nofollow(dirfd: RawFd, name: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let cname = CString::new(name)
        .map_err(|e| format!("file name {name} contains an interior NUL byte: {e}"))?;
    // SAFETY: `dirfd` is a valid open directory fd borrowed for the
    // duration of this call; `cname` is a valid NUL-terminated C string.
    // `openat` returns either a valid fd (>= 0) or -1 with errno set,
    // checked immediately below.
    let raw = unsafe {
        libc::openat(
            dirfd,
            cname.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(format!(
            "failed to open {name} (O_NOFOLLOW): {}",
            std::io::Error::last_os_error()
        )
        .into());
    }
    // SAFETY: `raw` was just returned by `libc::openat` above, checked
    // non-negative, and is not owned or closed anywhere else.
    let mut file = unsafe { fs::File::from(OwnedFd::from_raw_fd(raw)) };
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|e| format!("failed to read {name}: {e}"))?;
    Ok(buf)
}

/// Publish directory for a conversion run: a fresh temp dir on success (`dry_run`
/// leaves `output_dir` itself as the target, since nothing is ever written), or
/// `output_dir` directly when there is nothing to atomically publish.
///
/// The `Drop` impl removes the temp dir unless [`TempPublishDir::publish`] has
/// consumed it — so any early return via `?` between creation and publish
/// discards the partially-written temp dir and leaves `output_dir` untouched.
struct TempPublishDir {
    /// `Some(temp_path)` when writes are staged in a temp dir pending an
    /// atomic rename; `None` for dry runs, where there is nothing to publish.
    temp: Option<PathBuf>,
    /// Directory fd for `temp`, opened `O_DIRECTORY|O_NOFOLLOW` immediately
    /// after `mkdir` and held for the lifetime of the conversion. Every
    /// staging-directory write in [`quantize_model`] goes through
    /// [`TempPublishDir::create_file`] against this fd — never a path-based
    /// `File::create`/`fs::write` — so the write always lands in the
    /// directory that was actually created, not wherever the staging path
    /// resolves to at write time. `None` for dry runs (nothing is staged).
    dir_fd: Option<OwnedFd>,
    output_dir: PathBuf,
}

impl TempPublishDir {
    /// Refuse a pre-existing non-empty `output_dir` (fail before any write),
    /// then create a fresh temp dir adjacent to it (same parent => same
    /// filesystem, so the final rename is atomic). A fresh dir has no
    /// pre-planted symlinks. Immediately opens a directory fd bound to that
    /// staging dir (`open_dir_nofollow`) — held for the rest of the run so
    /// every subsequent write can bind to the fd instead of re-resolving
    /// the staging path.
    fn create(output_dir: &Path, dry_run: bool) -> Result<Self, Box<dyn std::error::Error>> {
        if dry_run {
            return Ok(Self {
                temp: None,
                dir_fd: None,
                output_dir: output_dir.to_path_buf(),
            });
        }

        if let Ok(mut entries) = fs::read_dir(output_dir)
            && entries.next().is_some()
        {
            return Err(format!(
                "output directory {} already exists and is not empty; \
                 refusing to overwrite (remove it first if this is intentional)",
                output_dir.display()
            )
            .into());
        }

        let parent = output_dir.parent().ok_or_else(|| {
            format!(
                "output directory {} has no parent to stage a temp dir in",
                output_dir.display()
            )
        })?;
        // Restore the parent-creation behavior `fs::create_dir_all(output_dir)`
        // used to provide before `TempPublishDir` replaced it: a valid
        // `--output-dir` whose parent doesn't exist yet must still succeed
        // (the adjacent temp dir and the final rename target both resolve
        // once the parent exists). The refuse-non-empty-output check above
        // already ran, so this can't paper over a stale non-empty target.
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create output directory parent {}: {e}",
                parent.display()
            )
        })?;
        let dir_name = output_dir
            .file_name()
            .ok_or_else(|| format!("output directory {} has no file name", output_dir.display()))?
            .to_string_lossy();
        let temp_dir = parent.join(format!(".{dir_name}.quantize-tmp-{}", std::process::id()));
        fs::create_dir(&temp_dir).map_err(|e| {
            format!(
                "failed to create staging directory {}: {e}",
                temp_dir.display()
            )
        })?;
        let dir_fd = open_dir_nofollow(&temp_dir)?;

        Ok(Self {
            temp: Some(temp_dir),
            dir_fd: Some(dir_fd),
            output_dir: output_dir.to_path_buf(),
        })
    }

    /// Directory that writes should target: the temp dir when staging, or
    /// `output_dir` directly for a dry run. Used only for path *display* and
    /// bookkeeping now — actual writes go through [`TempPublishDir::create_file`],
    /// bound to `dir_fd`, never through this path.
    fn write_dir(&self) -> &Path {
        self.temp.as_deref().unwrap_or(&self.output_dir)
    }

    /// Create (and open for writing) a new file directly inside the staging
    /// directory via `openat` against the held `dir_fd` — never a path-based
    /// `File::create`. `filename` must be a plain file name (no separators);
    /// every call site passes a literal or a sanitized tensor-name stem.
    ///
    /// `O_EXCL` refuses a pre-existing entry (including one an attacker
    /// planted after `mkdir` but before this call); `O_NOFOLLOW` refuses to
    /// follow it even if it is a symlink. Mode `0o600`: staging artifacts
    /// are private to the invoking user until the final rename publishes
    /// them under `output_dir`.
    fn create_file(&self, filename: &str) -> Result<fs::File, Box<dyn std::error::Error>> {
        let dir_fd = self.dir_fd.as_ref().ok_or_else(|| {
            format!(
                "staging directory {} has no open directory fd (dry run?)",
                self.write_dir().display()
            )
        })?;
        let cname = CString::new(filename)
            .map_err(|e| format!("file name {filename} contains an interior NUL byte: {e}"))?;
        // SAFETY: `dir_fd` is a valid open directory fd held for the
        // lifetime of `self`; `cname` is a valid NUL-terminated C string
        // owned for the duration of this call. `openat` returns either a
        // valid owned fd (>= 0) or -1 with errno set; the -1 case is
        // checked immediately below before any use of `raw`.
        let raw = unsafe {
            libc::openat(
                dir_fd.as_raw_fd(),
                cname.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_WRONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        };
        if raw < 0 {
            return Err(format!(
                "failed to create {filename} in staging directory {}: {}",
                self.write_dir().display(),
                std::io::Error::last_os_error()
            )
            .into());
        }
        // SAFETY: `raw` was just returned by `libc::openat` above, checked
        // non-negative, and is not owned or closed anywhere else —
        // `OwnedFd`/`File` becomes its sole owner and will close it on drop.
        let owned = unsafe { OwnedFd::from_raw_fd(raw) };
        Ok(fs::File::from(owned))
    }

    /// fsync every held file handle written during this run, fsync the
    /// staging directory itself (via the held `dir_fd`, not a path-based
    /// reopen), then atomically rename the temp dir onto `output_dir`.
    /// No-op for a dry run (nothing was staged).
    ///
    /// Consumes `written_files` (path + open file handle pairs, for fsync
    /// and error-message display) rather than reopening each path — the
    /// files were already opened bound to `dir_fd` by
    /// [`TempPublishDir::create_file`], so fsyncing the held handle keeps
    /// the whole write+durability path fd-bound end to end.
    ///
    /// Ownership of the staging dir/fd is released (`self.temp`/`self.dir_fd`
    /// set to `None`) only *after* the rename succeeds — an earlier `?`
    /// return (fsync failure) leaves them populated so `Drop` still cleans
    /// up the orphaned staging directory instead of leaking it.
    fn publish(
        mut self,
        written_files: Vec<(PathBuf, fs::File)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(temp_dir) = self.temp.clone() else {
            return Ok(());
        };
        for (path, file) in &written_files {
            file.sync_all()
                .map_err(|e| format!("failed to fsync {}: {e}", path.display()))?;
        }
        let dir_fd = self.dir_fd.as_ref().ok_or_else(|| {
            format!(
                "staging directory {} has no open directory fd to fsync",
                temp_dir.display()
            )
        })?;
        // SAFETY: `dir_fd` is a valid open fd held for the lifetime of
        // `self`. `libc::fsync` returns 0 on success or -1 with errno set.
        if unsafe { libc::fsync(dir_fd.as_raw_fd()) } != 0 {
            return Err(format!(
                "failed to fsync staging directory {}: {}",
                temp_dir.display(),
                std::io::Error::last_os_error()
            )
            .into());
        }
        // `fs::rename` below is necessarily path-based (POSIX has no
        // "rename this fd" call) — but `dir_fd` was bound to the inode
        // `mkdir` actually created, so it is the one piece of ground truth
        // an attacker cannot redirect. Compare it against a fresh stat of
        // `temp_dir` immediately before the rename: if the path was swapped
        // for a different directory (a plain directory substitution passes
        // `O_NOFOLLOW`, which only rejects a symlink) between the last
        // write and this call, the inode numbers diverge and the rename is
        // refused instead of publishing the attacker's substituted
        // directory as `output_dir`.
        let mut fd_stat: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: `dir_fd` is a valid open fd for the lifetime of `self`;
        // `fd_stat` is a valid, fully-initialized (zeroed) `libc::stat` the
        // kernel writes into. `fstat` returns 0 on success or -1 with errno
        // set, checked immediately below.
        if unsafe { libc::fstat(dir_fd.as_raw_fd(), &raw mut fd_stat) } != 0 {
            return Err(format!(
                "failed to fstat staging directory {}: {}",
                temp_dir.display(),
                std::io::Error::last_os_error()
            )
            .into());
        }
        let path_meta = fs::symlink_metadata(&temp_dir).map_err(|e| {
            format!(
                "failed to stat staging directory {} before publish: {e}",
                temp_dir.display()
            )
        })?;
        use std::os::unix::fs::MetadataExt;
        if path_meta.dev() != fd_stat.st_dev as u64 || path_meta.ino() != fd_stat.st_ino {
            return Err(format!(
                "staging directory {} was replaced before publish (inode mismatch: \
                 held fd is dev={} ino={}, path now resolves to dev={} ino={}); \
                 refusing to publish a substituted directory",
                temp_dir.display(),
                fd_stat.st_dev,
                fd_stat.st_ino,
                path_meta.dev(),
                path_meta.ino(),
            )
            .into());
        }
        fs::rename(&temp_dir, &self.output_dir).map_err(|e| {
            format!(
                "failed to publish {} to {}: {e}",
                temp_dir.display(),
                self.output_dir.display()
            )
        })?;
        self.temp = None;
        self.dir_fd = None;
        Ok(())
    }
}

impl Drop for TempPublishDir {
    /// Recursively deletes `temp_dir` only after proving, via the fd `mkdir`
    /// bound at creation, that the path still resolves to the same inode
    /// this run actually created. `fs::remove_dir_all` is necessarily
    /// path-based (POSIX has no "delete relative to this fd" call for a
    /// whole subtree), so without this check an attacker who replaces
    /// `temp_dir` with a directory or symlink after the last write (a
    /// failed conversion, a panic mid-run) makes a bare `remove_dir_all`
    /// recursively delete a directory this process never created. On
    /// identity mismatch, the staging directory is deliberately leaked
    /// rather than deleted — reported to stderr for manual cleanup.
    fn drop(&mut self) {
        let Some(temp_dir) = self.temp.take() else {
            return;
        };
        let identity_confirmed = self.dir_fd.as_ref().is_some_and(|dir_fd| {
            let mut fd_stat: libc::stat = unsafe { std::mem::zeroed() };
            // SAFETY: `dir_fd` is a valid open fd held for the lifetime of
            // `self`; `fd_stat` is a valid, fully-initialized (zeroed)
            // `libc::stat`. `fstat` returns 0 on success or -1 with errno
            // set, checked immediately below.
            if unsafe { libc::fstat(dir_fd.as_raw_fd(), &raw mut fd_stat) } != 0 {
                return false;
            }
            let Ok(path_meta) = fs::symlink_metadata(&temp_dir) else {
                return false;
            };
            use std::os::unix::fs::MetadataExt;
            path_meta.dev() == fd_stat.st_dev as u64 && path_meta.ino() == fd_stat.st_ino
        });
        if identity_confirmed {
            let _ = fs::remove_dir_all(&temp_dir);
        } else {
            eprintln!(
                "quantize_q4: staging directory {} was replaced before cleanup \
                 could run (inode identity check failed); leaving it in place \
                 instead of recursively deleting a directory this run may not \
                 have created — remove it manually if it is safe to do so",
                temp_dir.display()
            );
        }
    }
}

fn quantize_model(
    model_dir: &Path,
    output_dir: &Path,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let publish_dir = TempPublishDir::create(output_dir, dry_run)?;
    let write_dir = publish_dir.write_dir().to_path_buf();
    let mut written_files: Vec<(PathBuf, fs::File)> = Vec::new();

    // Checkpoint validation (`QuarotTensorReader::open` parses and validates
    // every SafeTensors header) runs BEFORE config.json is ever read or
    // copied. Copying config.json first would let a hostile or
    // concurrently-swapped model directory exfiltrate an arbitrary readable
    // file into the output before the checkpoint is known to be valid.
    let reader = QuarotTensorReader::open(model_dir)?;
    let mut tensor_names = reader.tensor_names();
    tensor_names.sort();
    let n_tensors = tensor_names.len();

    if !dry_run {
        let model_dir_fd = open_dir_nofollow(model_dir)?;
        let config_bytes = read_file_at_nofollow(model_dir_fd.as_raw_fd(), "config.json")
            .map_err(|e| format!("failed to read {}/config.json: {e}", model_dir.display()))?;
        let output_config = write_dir.join("config.json");
        let mut f = publish_dir.create_file("config.json")?;
        f.write_all(&config_bytes).map_err(|e| {
            format!(
                "failed to write {} to staging directory: {e}",
                output_config.display()
            )
        })?;
        written_files.push((output_config, f));
    }

    eprintln!("=== quantize_q4: SafeTensors → Q4_0 ===");
    eprintln!("Model dir:  {}", model_dir.display());
    eprintln!("Output dir: {}", output_dir.display());
    eprintln!("Tensors:    {n_tensors}");
    if dry_run {
        eprintln!("Mode:       DRY RUN (no files written)");
    }
    eprintln!();

    let global_start = Instant::now();
    let mut index_entries: Vec<IndexEntry> = Vec::new();
    let mut total_tensors = 0usize;
    let mut total_quantized = 0usize;
    let mut total_kept_f16 = 0usize;
    let mut total_bytes_in = 0u64;
    let mut total_bytes_out = 0u64;

    for (tensor_idx, tensor_name) in tensor_names.iter().enumerate() {
        let tensor_start = Instant::now();
        let bytes_in = reader.tensor_byte_len(tensor_name)?;
        let source_dtype = reader.source_dtype(tensor_name)?;
        let (data_f64, shape) = reader.read_tensor_f64(tensor_name)?;

        let expected_numel = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| format!("tensor {tensor_name}: shape product overflow for {shape:?}"))?;
        if expected_numel != data_f64.len() {
            return Err(format!(
                "tensor {tensor_name}: shape {shape:?} has {expected_numel} elements, \
                 reader returned {}",
                data_f64.len()
            )
            .into());
        }

        let numel = data_f64.len();
        total_bytes_in += bytes_in;

        let sanitized: String = tensor_name
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();

        if should_quantize(tensor_name) {
            // Reader decodes to f64; the Q4 quantizer works in f32 (ADR-044 step
            // 3c). Only quantized tensors need this widening pass — kept
            // tensors write directly from `data_f64` below.
            let data_f32: Vec<f32> = data_f64.iter().map(|&v| v as f32).collect();
            let q4 = quantize_f32_to_q4(&data_f32, &shape)?;
            let bytes_out = (q4.blocks.len() * Q4_BLOCK_BYTES) as u64;
            total_bytes_out += bytes_out;

            let out_filename = format!("{sanitized}.q4");
            let out_path = write_dir.join(&out_filename);

            if !dry_run {
                let mut f = publish_dir.create_file(&out_filename)?;
                f.write_all(&q4_file_bytes(&q4))
                    .map_err(|e| format!("failed to write {}: {e}", out_path.display()))?;
                written_files.push((out_path, f));
            }

            let elapsed = tensor_start.elapsed();
            eprintln!(
                "  [{}/{n_tensors}] Q4_0  {tensor_name}  shape={shape:?}  \
                 {:.1}MB→{:.1}MB  {:.2}s",
                tensor_idx + 1,
                bytes_in as f64 / 1_048_576.0,
                bytes_out as f64 / 1_048_576.0,
                elapsed.as_secs_f64()
            );

            index_entries.push(IndexEntry {
                name: tensor_name.clone(),
                file: out_filename,
                quantized: true,
                shape: shape.clone(),
                numel,
            });
            total_quantized += 1;
        } else {
            // Kept tensor: reader already decoded to numeric values, so the
            // common path is decoded-value → f16 for every source dtype.
            let bytes_out = (numel * 2) as u64;
            total_bytes_out += bytes_out;

            let out_filename = format!("{sanitized}.f16");
            let out_path = write_dir.join(&out_filename);

            // Validate f16 representability unconditionally: a dry run must
            // reject exactly what the real write below
            // would reject — a finite kept value that overflows f16
            // narrowing (e.g. `f32::MAX`) errors here regardless of
            // `dry_run`, instead of only surfacing on the next real run.
            let framed = encode_f16_payload(
                &out_path.display().to_string(),
                tensor_name,
                &data_f64,
                &shape,
            )?;

            if !dry_run {
                let mut f = publish_dir.create_file(&out_filename)?;
                f.write_all(&framed)
                    .map_err(|e| format!("failed to write {}: {e}", out_path.display()))?;
                written_files.push((out_path, f));
            }

            let elapsed = tensor_start.elapsed();
            eprintln!(
                "  [{}/{n_tensors}] F16   {tensor_name}  shape={shape:?}  \
                 {:.1}MB  dtype={}  {:.3}s",
                tensor_idx + 1,
                bytes_in as f64 / 1_048_576.0,
                source_dtype.name(),
                elapsed.as_secs_f64()
            );

            index_entries.push(IndexEntry {
                name: tensor_name.clone(),
                file: out_filename,
                quantized: false,
                shape: shape.clone(),
                numel,
            });
            total_kept_f16 += 1;
        }

        total_tensors += 1;
    }

    // Write the quantization index, then atomically publish the whole run.
    if !dry_run {
        let index_path = write_dir.join("quantize_index.json");
        let index_json = serde_json::to_string_pretty(&index_entries)
            .map_err(|e| format!("failed to serialize index: {e}"))?;
        let mut f = publish_dir.create_file("quantize_index.json")?;
        f.write_all(index_json.as_bytes())
            .map_err(|e| format!("failed to write {}: {e}", index_path.display()))?;
        written_files.push((index_path, f));
        eprintln!(
            "Index written: {}",
            output_dir.join("quantize_index.json").display()
        );

        publish_dir.publish(written_files)?;
    }

    let total_elapsed = global_start.elapsed();
    let compression = if total_bytes_in > 0 {
        total_bytes_out as f64 / total_bytes_in as f64
    } else {
        1.0
    };

    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!("Tensors processed: {total_tensors}");
    eprintln!("  Quantized (Q4_0): {total_quantized}");
    eprintln!("  Kept (F16):       {total_kept_f16}");
    eprintln!(
        "Input size:   {:.2} GB",
        total_bytes_in as f64 / 1_073_741_824.0
    );
    eprintln!(
        "Output size:  {:.2} GB",
        total_bytes_out as f64 / 1_073_741_824.0
    );
    eprintln!(
        "Ratio:        {:.2}x  ({:.1}%)",
        1.0 / compression,
        compression * 100.0
    );
    eprintln!("Total time:   {:.1}s", total_elapsed.as_secs_f64());

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{quantize_model, should_quantize};
    use std::fs;
    use std::io::Write;
    use std::path::Path;

    fn write_single_f32_tensor(path: &Path, name: &str, values: &[f32]) {
        let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = serde_json::json!({
            name: {
                "dtype": "F32",
                "shape": [values.len()],
                "data_offsets": [0, payload.len()],
            }
        });
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&payload).unwrap();
    }

    #[test]
    fn invalid_source_tensor_is_not_published() {
        for (name, extension) in [("invalid.weight", "q4"), ("invalid.norm.weight", "f16")] {
            let tmp = tempfile::tempdir().unwrap();
            let input = tmp.path().join("input");
            let output = tmp.path().join("output");
            fs::create_dir(&input).unwrap();
            fs::write(input.join("config.json"), b"{}").unwrap();
            write_single_f32_tensor(
                &input.join("model.safetensors"),
                name,
                &[1.0, f32::NAN, 3.0],
            );

            let err = quantize_model(&input, &output, false)
                .expect_err("offline quantizer must reject a non-finite source tensor");
            let sanitized = name.replace('.', "_");
            assert!(err.to_string().contains(name), "unexpected error: {err}");
            assert!(
                !output.join(format!("{sanitized}.{extension}")).exists(),
                "invalid tensor must not have a completed output artifact"
            );
            assert!(!output.join("quantize_index.json").exists());
        }
    }

    #[test]
    #[cfg(unix)]
    fn config_json_symlink_is_rejected_not_followed() {
        // `config.json` planted as a symlink to a file outside the model
        // directory — standing in for a hostile or concurrently-swapped
        // checkpoint trying to exfiltrate arbitrary readable bytes into the
        // quantizer's output. Mutation-sensitive: reverting
        // `read_file_at_nofollow` to a path-based `fs::read` (which follows
        // the final symlink) makes this test fail — the secret contents
        // would be copied into the staging directory instead of the read
        // being refused.
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        fs::create_dir(&input).unwrap();
        write_single_f32_tensor(
            &input.join("model.safetensors"),
            "model.layers.0.input_layernorm.weight",
            &[1.0, 2.0, 3.0],
        );

        let secret = tmp.path().join("secret.txt");
        fs::write(&secret, b"do-not-exfiltrate").unwrap();
        std::os::unix::fs::symlink(&secret, input.join("config.json")).unwrap();

        let err = quantize_model(&input, &output, false)
            .expect_err("a symlinked config.json must be refused, not followed");
        assert!(
            err.to_string().contains("config.json"),
            "unexpected error: {err}"
        );
        assert!(
            !output.exists() || fs::read_dir(&output).unwrap().next().is_none(),
            "no output artifact may exist after a refused config.json read"
        );
    }

    // -----------------------------------------------------------------------
    // Atomic publish + refuse-non-empty-output-dir.
    //
    // Mutation-sensitive: reverting `TempPublishDir` so writes land directly
    // in `output_dir` (as before this fix) makes
    // `atomic_publish_leaves_output_dir_untouched_on_mid_conversion_failure`
    // fail (a partial `.q4`/`config.json` would appear in `output`), and
    // dropping the up-front `fs::read_dir` non-empty check makes both
    // `quantize_model_refuses_preexisting_nonempty_output_dir` and
    // `quantize_model_refuses_output_dir_containing_planted_symlink` fail
    // (the stray file / symlink would silently coexist with fresh output).
    // -----------------------------------------------------------------------

    #[test]
    fn quantize_model_publishes_successful_run_atomically() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        fs::create_dir(&input).unwrap();
        fs::write(input.join("config.json"), b"{}").unwrap();
        write_single_f32_tensor(
            &input.join("model.safetensors"),
            "model.layers.0.input_layernorm.weight",
            &[1.0, 2.0, 3.0],
        );

        quantize_model(&input, &output, false).unwrap();

        assert!(output.join("config.json").exists());
        assert!(output.join("quantize_index.json").exists());
        // No leftover staging directory beside the published output.
        let siblings: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(
            siblings
                .iter()
                .filter(|n| *n != "input" && *n != "output")
                .count(),
            0,
            "no temp staging directory should survive a successful run: {siblings:?}"
        );
    }

    #[test]
    fn atomic_publish_leaves_output_dir_untouched_on_mid_conversion_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        fs::create_dir(&input).unwrap();
        fs::write(input.join("config.json"), b"{}").unwrap();
        write_single_f32_tensor(
            &input.join("model.safetensors"),
            "invalid.norm.weight",
            &[1.0, f32::NAN, 3.0],
        );

        quantize_model(&input, &output, false)
            .expect_err("non-finite source tensor must fail the conversion");

        assert!(
            !output.exists(),
            "a failed conversion must leave output_dir absent, not partially populated"
        );
        // No orphaned staging directory left behind either.
        let siblings: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(
            siblings.iter().filter(|n| *n != "input").count(),
            0,
            "a failed conversion must not leave a temp staging directory: {siblings:?}"
        );
    }

    #[test]
    fn publish_refuses_staging_dir_substituted_with_different_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let output = tmp.path().join("output");

        let staging = super::TempPublishDir::create(&output, false).unwrap();
        let temp_path = staging.write_dir().to_path_buf();
        staging
            .create_file("real.q4")
            .unwrap()
            .write_all(b"legit")
            .unwrap();

        // Substitute the staging directory: remove the one `create()` made
        // and `mkdir` a fresh plain directory at the same path. This is not
        // a symlink, so `O_NOFOLLOW` on the held `dir_fd` (opened against
        // the original inode) does not reject it — only an inode check
        // right before the path-based `rename` can catch the swap.
        fs::remove_dir_all(&temp_path).unwrap();
        fs::create_dir(&temp_path).unwrap();
        fs::write(temp_path.join("attacker-planted.txt"), b"evil").unwrap();

        let err = staging
            .publish(Vec::new())
            .expect_err("publish must refuse a staging directory substituted after creation");
        assert!(
            err.to_string().contains("inode mismatch"),
            "unexpected error: {err}"
        );
        assert!(
            !output.exists(),
            "a substituted staging directory must not be published as output_dir"
        );
    }

    #[test]
    fn quantize_model_refuses_preexisting_nonempty_output_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        fs::create_dir(&input).unwrap();
        fs::write(input.join("config.json"), b"{}").unwrap();
        write_single_f32_tensor(
            &input.join("model.safetensors"),
            "model.layers.0.input_layernorm.weight",
            &[1.0, 2.0, 3.0],
        );
        fs::create_dir(&output).unwrap();
        fs::write(output.join("stray.txt"), b"pre-existing").unwrap();

        let err = quantize_model(&input, &output, false)
            .expect_err("a non-empty pre-existing output_dir must be refused");
        assert!(
            err.to_string().contains("already exists and is not empty"),
            "unexpected error: {err}"
        );
        assert!(output.join("stray.txt").exists());
        assert!(!output.join("quantize_index.json").exists());
    }

    #[test]
    #[cfg(unix)]
    fn quantize_model_refuses_output_dir_containing_planted_symlink() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        fs::create_dir(&input).unwrap();
        fs::write(input.join("config.json"), b"{}").unwrap();
        write_single_f32_tensor(
            &input.join("model.safetensors"),
            "model.layers.0.input_layernorm.weight",
            &[1.0, 2.0, 3.0],
        );

        let escape_target = tmp.path().join("outside_target");
        fs::create_dir(&escape_target).unwrap();
        fs::create_dir(&output).unwrap();
        std::os::unix::fs::symlink(&escape_target, output.join("config.json")).unwrap();

        let err = quantize_model(&input, &output, false)
            .expect_err("an output_dir containing a planted symlink must be refused");
        assert!(
            err.to_string().contains("already exists and is not empty"),
            "unexpected error: {err}"
        );
        // The attacker's symlink target must never have been written into.
        assert_eq!(fs::read_dir(&escape_target).unwrap().count(), 0);
    }

    // -----------------------------------------------------------------------
    // Fold: Drop-ordering — a failed `publish()` must leave the staging
    // handle in `self` so `Drop` can still clean it up.
    //
    // Mutation-sensitive: reverting `publish()` to `.take()` `self.temp` (and
    // `self.dir_fd`) up front makes this test fail — the
    // staging directory would survive (leaked) after a failed rename,
    // because `Drop` would find `self.temp == None` and do nothing.
    // -----------------------------------------------------------------------
    #[test]
    fn publish_failure_still_lets_drop_clean_up_the_staging_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let output = tmp.path().join("output");
        // Pre-create `output` as a regular FILE (not a directory): the
        // non-empty-dir refusal in `create()` only triggers for an existing
        // *directory*, so `create()` succeeds, but the final
        // `fs::rename(temp_dir, output)` in `publish()` fails (renaming a
        // directory onto an existing non-directory is always rejected).
        fs::write(&output, b"not a directory").unwrap();

        let publish_dir = super::TempPublishDir::create(&output, false).unwrap();
        let staging_path = publish_dir.write_dir().to_path_buf();
        assert!(staging_path.is_dir());

        let mut f = publish_dir.create_file("artifact.q4").unwrap();
        f.write_all(b"data").unwrap();

        // `publish` takes `self` by value, so `publish_dir` is dropped
        // inside this call before the `Err` reaches the assertion below —
        // whatever `Drop` does (or fails to do) has already happened by
        // the time `.expect_err` runs.
        let err = publish_dir
            .publish(vec![(staging_path.join("artifact.q4"), f)])
            .expect_err("renaming the staging dir onto an existing file must fail");
        assert!(
            err.to_string().contains("failed to publish"),
            "unexpected error: {err}"
        );

        assert!(
            !staging_path.exists(),
            "a failed publish must not leak the staging directory — Drop must \
             still own cleanup because `self.temp`/`self.dir_fd` stayed \
             populated until the rename actually succeeded"
        );
    }

    #[test]
    #[cfg(unix)]
    fn drop_refuses_to_delete_a_staging_dir_substituted_with_a_different_directory() {
        // Same substitution as `publish_refuses_staging_dir_substituted_
        // with_different_directory`, but exercised through `Drop` (the
        // no-publish / early-error path) rather than `publish()`: the
        // staging path is replaced with a fresh, unrelated directory that
        // holds attacker-planted content, then the guard drops without
        // calling `publish`. Mutation-sensitive: reverting `Drop` to a bare
        // `fs::remove_dir_all(temp_dir)` (no inode check against the held
        // `dir_fd`) makes this test fail — the substituted directory (and
        // the attacker's file inside it) would be recursively deleted even
        // though this run never created it.
        let tmp = tempfile::tempdir().unwrap();
        let output = tmp.path().join("output");

        let staging = super::TempPublishDir::create(&output, false).unwrap();
        let temp_path = staging.write_dir().to_path_buf();

        fs::remove_dir_all(&temp_path).unwrap();
        fs::create_dir(&temp_path).unwrap();
        let planted = temp_path.join("attacker-planted.txt");
        fs::write(&planted, b"evil").unwrap();

        drop(staging);

        assert!(
            temp_path.exists(),
            "a substituted staging directory must be leaked, not deleted"
        );
        assert_eq!(
            fs::read(&planted).unwrap(),
            b"evil",
            "the attacker's substituted directory contents must be untouched"
        );
    }

    // -----------------------------------------------------------------------
    // fd-bind the staging directory (path race).
    //
    // Mutation-sensitive: reverting `TempPublishDir::create_file` to a
    // the previous path-based `File::create(self.write_dir().join(filename))` makes `create_file_writes_land_in_original_dirfd_inode_not_a_
    // post_create_symlink_swap` fail — the write would follow the attacker's
    // symlink into `attacker_dir` instead of landing in the original staging
    // inode, so the "attacker dir stays empty" assertion trips.
    // -----------------------------------------------------------------------
    #[test]
    #[cfg(unix)]
    fn create_file_writes_land_in_original_dirfd_inode_not_a_post_create_symlink_swap() {
        let tmp = tempfile::tempdir().unwrap();
        let output = tmp.path().join("output");
        let publish_dir = super::TempPublishDir::create(&output, false).unwrap();
        let staging_path = publish_dir.write_dir().to_path_buf();
        assert!(
            staging_path.is_dir(),
            "staging dir must exist right after TempPublishDir::create"
        );

        // Attacker replaces the staging PATH with a symlink to a directory
        // they control, after the dir (and its fd) were already created.
        // The original directory is preserved (moved aside via `rename`,
        // which never touches inodes or open fds — only the still-open
        // `dir_fd` keeps pointing at it) rather than `rmdir`'d: an already
        // fully-unlinked (0-link) directory refuses further `openat` calls
        // outright (verified on this platform), which would make the test
        // pass for the wrong reason (ENOENT, not fd-binding) instead of
        // proving the write actually lands in the original inode.
        let attacker_dir = tmp.path().join("attacker_dir");
        let moved_aside = tmp.path().join("staging_moved_aside");
        fs::create_dir(&attacker_dir).unwrap();
        fs::rename(&staging_path, &moved_aside).unwrap();
        std::os::unix::fs::symlink(&attacker_dir, &staging_path).unwrap();

        // Drive a child write through the dirfd-bound API.
        let mut f = publish_dir.create_file("child.q4").unwrap();
        f.write_all(b"trusted-payload").unwrap();
        drop(f);

        // The attacker's directory must never have been written into: a
        // path-based write would have followed the swapped symlink here.
        assert_eq!(
            fs::read_dir(&attacker_dir).unwrap().count(),
            0,
            "attacker-controlled directory must stay empty; a path-based \
             write would have redirected the child write here"
        );

        // Affirmative check: the write landed in the ORIGINAL staging
        // inode — reachable at `moved_aside` (where the directory now
        // lives) — not wherever `staging_path` currently resolves.
        let landed = fs::read(moved_aside.join("child.q4"))
            .expect("child write must land in the original (moved-aside) staging directory");
        assert_eq!(landed, b"trusted-payload");
    }

    // -----------------------------------------------------------------------
    // dry-run must validate f16 representability identically to
    // the real run.
    //
    // Mutation-sensitive: hoisting the `encode_f16_payload` call back inside
    // the `if !dry_run` block makes
    // `dry_run_rejects_f16_overflow_that_the_real_run_would_also_reject`
    // pass silently through dry-run (no error), diverging from the real-run
    // sibling assertion below it.
    // -----------------------------------------------------------------------
    #[test]
    fn dry_run_rejects_f16_overflow_that_the_real_run_would_also_reject() {
        for dry_run in [true, false] {
            let tmp = tempfile::tempdir().unwrap();
            let input = tmp.path().join("input");
            let output = tmp.path().join("output");
            fs::create_dir(&input).unwrap();
            fs::write(input.join("config.json"), b"{}").unwrap();
            // A finite kept (non-quantized) tensor whose value overflows f16
            // narrowing to +inf: `f32::MAX` is finite in f32/f64 but has no
            // finite f16 representation.
            write_single_f32_tensor(
                &input.join("model.safetensors"),
                "model.layers.0.input_layernorm.weight",
                &[f32::MAX, 1.0, 2.0],
            );

            let err = quantize_model(&input, &output, dry_run).expect_err(&format!(
                "a kept tensor with an f16-unrepresentable value must be rejected \
                 (dry_run={dry_run})"
            ));
            assert!(
                err.to_string().contains("non-finite"),
                "unexpected error (dry_run={dry_run}): {err}"
            );
            assert!(
                !output.exists(),
                "a rejected conversion must not create output_dir (dry_run={dry_run})"
            );
        }
    }

    // -----------------------------------------------------------------------
    // restore missing-parent-directory creation.
    //
    // Mutation-sensitive: removing the `fs::create_dir_all(parent)` call
    // added to `TempPublishDir::create` makes
    // `quantize_model_creates_missing_output_parent_directories` fail with
    // "failed to create staging directory" (no such file or directory),
    // while the non-empty-output-dir refusal sibling stays green throughout.
    // -----------------------------------------------------------------------
    #[test]
    fn quantize_model_creates_missing_output_parent_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        // `output`'s parent ("nested/does/not/exist") does not exist yet.
        let output = tmp
            .path()
            .join("nested")
            .join("does")
            .join("not")
            .join("exist")
            .join("out");
        fs::create_dir(&input).unwrap();
        fs::write(input.join("config.json"), b"{}").unwrap();
        write_single_f32_tensor(
            &input.join("model.safetensors"),
            "model.layers.0.input_layernorm.weight",
            &[1.0, 2.0, 3.0],
        );

        quantize_model(&input, &output, false)
            .expect("a valid output path with a missing parent directory must succeed");

        assert!(output.join("config.json").exists());
        assert!(output.join("quantize_index.json").exists());
    }

    // -----------------------------------------------------------------------
    // MoE routed-expert tensors (issue #874 regression coverage).
    //
    // Mutation-sensitive: these tensor names have no `.weight` suffix, so the
    // pre-fix gate (`if !name.ends_with(".weight") ... return false`) rejects
    // them before reaching any of the quantize-candidate checks. Reverting
    // the `.experts.gate_up_proj` / `.experts.down_proj` special-case added
    // in this fix makes every assertion below fail.
    // -----------------------------------------------------------------------

    #[test]
    fn should_quantize_accepts_routed_expert_tensors_main_layers() {
        assert!(should_quantize(
            "model.language_model.layers.0.mlp.experts.gate_up_proj"
        ));
        assert!(should_quantize(
            "model.language_model.layers.0.mlp.experts.down_proj"
        ));
        assert!(should_quantize(
            "model.language_model.layers.39.mlp.experts.gate_up_proj"
        ));
        assert!(should_quantize(
            "model.language_model.layers.39.mlp.experts.down_proj"
        ));
    }

    #[test]
    fn should_quantize_accepts_routed_expert_tensors_mtp_layer() {
        // The speculative-decode MTP head has its own MoE layer under a
        // different name prefix; the suffix match must not be anchored to
        // the `model.language_model.` prefix.
        assert!(should_quantize("mtp.layers.0.mlp.experts.gate_up_proj"));
        assert!(should_quantize("mtp.layers.0.mlp.experts.down_proj"));
    }

    #[test]
    fn should_quantize_rejects_names_that_merely_contain_experts() {
        // Precision check: the match is a name-suffix match, not a substring
        // match, so a hypothetical `.experts.gate_up_proj_extra` (or any
        // other name that merely contains "experts") does not get swept in
        // by accident.
        assert!(!should_quantize(
            "model.language_model.layers.0.mlp.experts.gate_up_proj_extra"
        ));
    }

    // -----------------------------------------------------------------------
    // Dense / already-suffixed tensors — must be unaffected by the fix.
    // -----------------------------------------------------------------------

    #[test]
    fn should_quantize_accepts_dense_projection_weights() {
        for name in [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            "model.language_model.layers.0.self_attn.v_proj.weight",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
        ] {
            assert!(should_quantize(name), "expected quantize=true for {name}");
        }
    }

    #[test]
    fn should_quantize_accepts_shared_expert_weights() {
        // The shared (always-on) expert and its gate already carry a
        // `.weight` suffix and were already quantized correctly pre-fix;
        // this pins that they remain quantized post-fix.
        for name in [
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
        ] {
            assert!(should_quantize(name), "expected quantize=true for {name}");
        }
    }

    #[test]
    fn should_quantize_accepts_router_gate_weight() {
        // The per-layer MoE router (`mlp.gate.weight`, shape [num_experts,
        // hidden]) is small relative to the routed experts but already
        // falls through to the "quantize unknown weight matrices" default;
        // this pins that unrelated behavior stays unchanged.
        assert!(should_quantize(
            "model.language_model.layers.0.mlp.gate.weight"
        ));
    }

    // -----------------------------------------------------------------------
    // Norms / biases / Mamba-specific scalars — must stay excluded.
    // -----------------------------------------------------------------------

    #[test]
    fn should_quantize_rejects_norms_biases_and_mamba_scalars() {
        for name in [
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.norm.weight",
            "model.language_model.layers.0.self_attn.q_norm.weight",
            "model.language_model.layers.0.self_attn.o_proj.bias",
            "model.language_model.layers.0.mamba.A_log",
            "model.language_model.layers.0.mamba.dt_bias",
            "model.language_model.layers.0.mamba.conv1d.weight",
        ] {
            assert!(!should_quantize(name), "expected quantize=false for {name}");
        }
    }

    #[test]
    fn should_quantize_rejects_non_weight_non_expert_tensors() {
        // Anything that isn't a `.weight` tensor and isn't a recognized
        // suffix-less expert tensor must be rejected outright.
        assert!(!should_quantize(
            "model.language_model.layers.0.self_attn.rotary_emb.inv_freq"
        ));
    }
}
