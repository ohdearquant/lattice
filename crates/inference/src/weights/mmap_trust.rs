//! Format-neutral mmap trust-boundary gate (#1037).
//!
//! Every checkpoint tensor lattice mmaps read-only and hands to the GPU
//! no-copy (`Q4WeightBuf`, `Q3WeightBuf`, the MoE f16/f32 dequant-on-load
//! buffers, and `ExpertByteTable`'s lazily-re-read expert cache) shares one
//! trust concern: the file must not be mutable by any principal other than
//! the process itself (or root, for shared root-deployed model directories)
//! for as long as the mmap is alive, because a mutation after validation but
//! before -- or during -- GPU reads bypasses every shape/bounds check this
//! crate performs. [`open_trusted_mmap_file`] is the single chokepoint all
//! five load sites route through so that guarantee holds identically for
//! each of them.
//!
//! POSIX mode bits + uid alone are not sufficient on macOS: an extended ACL
//! (`chmod +a "someuser allow write" <path>`) can grant another principal
//! write access to a file whose `st_mode` reports a locked-down `0600` --
//! entirely invisible to the mode/uid check below. On macOS this module
//! additionally inspects the file's extended ACL (`ACL_TYPE_EXTENDED`) and
//! rejects any file carrying an ACE that grants a permission which confers
//! write, directly or transitively, to *any* principal -- not just
//! non-owner, non-root principals. That includes the two meta-rights
//! `ACL_WRITE_SECURITY` (rewrite the ACL itself, then grant yourself write)
//! and `ACL_CHANGE_OWNER` (take ownership, then re-grant): neither mutates
//! file content directly, but both let a principal escalate to content
//! write after this check has already passed, so they are rejected
//! identically to the direct-write permissions. See `macos_acl`'s
//! `REJECTED_PERMS`/`HARMLESS_PERMS` for the full classification of every
//! `acl_perm_t` value.
//!
//! Resolving an ACE's qualifier (a `guid_t`) back to a uid to discriminate
//! "owner/root" from "foreign" requires `mbr_uuid_to_id` (Directory
//! Services) and materially widens the FFI surface for a narrower rejection
//! set; the conservative any-write-ACE policy is simpler to get right and a
//! locked-down model file legitimately has no extended ACL entries at all,
//! so it costs nothing in the documented deployment shapes (owner-only or
//! root-owned, non-ACL'd files).
//!
//! # Parent-directory chain
//!
//! A file-only check is not sufficient: a principal that cannot write the
//! checkpoint file itself can still rename-replace it out from under an
//! owner-locked-down file if any ancestor directory is writable to them, and
//! `unlink`/`rename` are directory operations gated on the *directory's*
//! write permission, not the target file's. [`reject_if_mmap_parent_directory_chain_weak`]
//! walks every ancestor of the canonicalized checkpoint path and applies the
//! same group/other-writable-or-foreign-owner rule
//! [`mmap_file_trust_boundary_issue`] applies to the file, with one
//! deliberate exception: a world-writable directory that also carries the
//! sticky bit (`S_ISVTX`, the `/tmp` pattern) is accepted despite being
//! group/other-writable. The sticky bit restricts `unlink`/`rename` within
//! that directory to the entry's owner, the directory's owner, or root --
//! exactly the rename-replace this walk exists to stop -- so a foreign,
//! non-privileged principal holding write access to a sticky world-writable
//! directory still cannot rename-replace a checkpoint file this module's
//! file-level check has already confirmed is owned by the process uid or
//! root. Rejecting sticky world-writable directories anyway would make
//! `/tmp`-hosted checkpoints unloadable for no additional protection, since
//! the sticky bit already closes the exact race this walk defends against.
//! A non-sticky group/other-writable ancestor, or one owned by a uid that is
//! neither the process uid nor root, is refused.
//!
//! # Trust-boundary scope (what this gate defends against, and what it does not)
//!
//! This gate -- [`open_trusted_mmap_file`]'s regular-file/O_NONBLOCK/O_NOFOLLOW
//! guard, the mode/uid/ACL checks, [`reject_if_mmap_parent_directory_chain_weak`]'s
//! ancestor-directory walk, and [`verify_mmap_target_unchanged`]'s post-map
//! fstat recheck together -- defends against a **foreign principal**:
//! another uid mutating, truncating, replacing, or blocking on the
//! checkpoint file between validation and the GPU reading the mapped pages,
//! planting a FIFO/device at the model path to hang the open before any
//! check runs, substituting a different file via a symlink planted at the
//! model path's *final* component (`O_NOFOLLOW` rejects that final component
//! symlink outright; a symlink in a *parent* directory is still followed --
//! final-component-only scope, see [`open_trusted_mmap_file`]'s doc
//! comment), or rename-replacing the checkpoint file via a writable ancestor
//! directory (closed by the parent-directory chain walk above, with the
//! sticky-bit exception documented there). It also closes the
//! **validate-then-truncate race**: a writer (foreign or, in principle, the
//! same process racing itself) that shortens the file in the window between
//! the pre-map stat and the `mmap()` call.
//!
//! It does **not**, and cannot, defend against **same-UID mutation of an
//! already-mapped `MAP_SHARED` region**: the owning process can always write
//! through its own fd (or a second fd it opens) to pages already handed to
//! the GPU, and no fstat-based recheck -- this one or any other -- observes
//! a write to already-mapped pages without abandoning zero-copy mmap
//! entirely (reading the multi-GiB payload into an owned buffer instead).
//! That is a structural limitation of every mmap-based model loader in the
//! industry, not a gap specific to this implementation.
//!
//! The same limitation covers a closely related shape worth naming
//! explicitly: **a foreign process that already held a writable fd to the
//! checkpoint file before this gate's permission/ACL hardening took
//! effect** (e.g. it opened the file, or the file was made writable-to-that-
//! principal, before the file was locked down to owner-only). A writable fd
//! opened earlier remains writable for as long as it stays open regardless
//! of later permission changes on the path, so such a process can still
//! write or truncate the mapped pages after this gate's post-map fstat
//! recheck runs -- the recheck detects the mutation's *effects* (size/mtime/
//! ino/dev mismatch) but cannot prevent a write to pages already mapped
//! `MAP_SHARED`. This is **accepted residual risk**, the same structural
//! class as same-UID mutation above, not a distinct gap: a zero-copy
//! `MAP_SHARED` mapping cannot observe or block a write to already-mapped
//! pages without abandoning zero-copy mmap entirely.
//!
//! This is this project's settled engineering position on the trust
//! boundary: with the parent-directory chain now enforced above, the sole
//! remaining residual is a principal that already held a writable fd to the
//! checkpoint before this gate's permission/ACL/directory hardening took
//! effect, plus same-UID mutation through a second fd the owning process
//! itself opens -- both an accepted trust-root assumption for any zero-copy
//! `MAP_SHARED` mmap loader, not specific to this implementation. The named
//! mitigation for both is the same: an opt-in read-into-owned-buffer mode
//! for untrusted roots, a possible future addition and explicitly out of
//! scope here.
//!
//! # fd-binding invariant
//!
//! [`reject_if_mmap_file_trust_boundary_weak`] takes the already-open
//! `&std::fs::File` that the caller is about to mmap, not a path. Every
//! check inside -- mode/uid (`file.metadata()`, fstat) and, on macOS, the
//! extended-ACL scan (`acl_get_fd_np`) -- resolves against that same open
//! descriptor. The path is accepted only for error messages and is never
//! re-resolved (re-`stat`'d, re-`open`'d, or passed to a path-based ACL
//! call) after the caller's original `open`. This closes a parent-directory
//! rename race: a pathname-based check (`acl_get_file`/`stat(path)`) can
//! observe a *different* inode than the fd being mapped if the path is
//! renamed out from under it between check and mmap; an fd-bound check
//! cannot, because the fd always refers to the inode it was opened against
//! regardless of what the path later resolves to.

use std::fs::File;
use std::path::Path;

/// Pure predicate: does this `(mode, file_uid, process_uid)` triple fall
/// outside lattice's mode/uid trust boundary for a mmap'd model file? Split
/// out from the metadata-reading wrapper so it is unit-testable without
/// touching the filesystem.
///
/// Root-owned files (`file_uid == 0`) are accepted even when
/// `process_uid != 0`: a root-deployed, owner-read-only model directory
/// shared across service accounts is a documented, legitimate deployment
/// shape -- the actual trust concern is mutability (group/other-writable),
/// not the specific non-root owning uid.
#[cfg(unix)]
#[cfg_attr(not(feature = "metal-gpu"), allow(dead_code))]
fn mmap_file_trust_boundary_issue(mode: u32, file_uid: u32, process_uid: u32) -> bool {
    let group_or_other_writable = mode & 0o022 != 0;
    let foreign_non_root_owner = file_uid != process_uid && file_uid != 0;
    group_or_other_writable || foreign_non_root_owner
}

/// The sticky bit (`S_ISVTX`), identical value on Darwin and Linux
/// (`0o1000`, `bsd/sys/stat.h` / `asm-generic/stat.h`). See
/// [`mmap_parent_dir_trust_boundary_issue`] for why this exempts a
/// world-writable directory from the ancestor-chain rejection.
#[cfg(unix)]
const S_ISVTX: u32 = 0o1000;

/// Pure predicate: does this ancestor directory `(mode, dir_uid,
/// process_uid)` triple fall outside lattice's trust boundary for a
/// checkpoint's parent-directory chain? Split out from the walking wrapper
/// so it is unit-testable without touching the filesystem, mirroring
/// [`mmap_file_trust_boundary_issue`].
///
/// A directory writable by group or other lets any principal with that
/// access rename or unlink entries inside it -- including replacing the
/// checkpoint file with a different one -- regardless of the checkpoint
/// file's own permissions, because `unlink`/`rename` are gated on the
/// *directory's* write permission, not the target file's. A directory owned
/// by a uid that is neither the process uid nor root is rejected
/// unconditionally: as owner, that uid can always chmod its own directory
/// to grant itself write regardless of the mode bits observed here.
///
/// The one deliberate exception: a world-writable directory that also
/// carries the sticky bit (`S_ISVTX` -- the `/tmp` pattern) is accepted.
/// The sticky bit restricts `unlink`/`rename` within the directory to the
/// entry's own owner, the directory's owner, or root, which is exactly the
/// rename-replace this check exists to prevent -- so a foreign,
/// non-privileged principal holding write access to a sticky world-writable
/// directory still cannot rename-replace a checkpoint file this module's
/// file-level check has already confirmed is owned by the process uid or
/// root. See this module's "Parent-directory chain" doc comment for the
/// full reasoning behind this exception.
#[cfg(unix)]
#[cfg_attr(not(feature = "metal-gpu"), allow(dead_code))]
fn mmap_parent_dir_trust_boundary_issue(mode: u32, dir_uid: u32, process_uid: u32) -> bool {
    let foreign_non_root_owner = dir_uid != process_uid && dir_uid != 0;
    let group_or_other_writable = mode & 0o022 != 0;
    let sticky = mode & S_ISVTX != 0;
    foreign_non_root_owner || (group_or_other_writable && !sticky)
}

/// Current process's real user id, via a direct `libc::getuid()` FFI call
/// (no `libc`/`nix` crate dependency -- `getuid()` always links against the
/// platform's libc, takes no arguments, and cannot fail).
#[cfg(unix)]
#[cfg_attr(not(feature = "metal-gpu"), allow(dead_code))]
fn current_uid() -> u32 {
    unsafe extern "C" {
        fn getuid() -> u32;
    }
    // SAFETY: `getuid()` is a pure, argument-free POSIX syscall wrapper
    // with no preconditions and no failure mode.
    unsafe { getuid() }
}

/// Walk every ancestor directory of `path` and refuse the load if any of
/// them falls outside [`mmap_parent_dir_trust_boundary_issue`]'s trust
/// boundary, or -- macOS only -- carries a macOS extended ACL entry that
/// grants a directory-mutating permission to any principal. `path` is
/// canonicalized first so the walk follows the real filesystem chain
/// (resolving any symlinked ancestor) rather than the literal,
/// possibly-relative components the caller supplied -- the walk must inspect
/// the directories the kernel will actually resolve through, not a textual
/// approximation of them.
///
/// The macOS ACL check reuses [`macos_acl::acl_grants_rejected_permission`] --
/// the same scan the target file itself goes through in
/// [`reject_if_mmap_file_trust_boundary_weak`] -- applied to each ancestor's
/// own fd instead of the checkpoint file's. `REJECTED_PERMS` is the correct
/// set for a directory as-is: `ACL_WRITE_DATA`/`ACL_APPEND_DATA` are the
/// same bits macOS's `sys/acl.h` aliases to `ACL_ADD_FILE`/
/// `ACL_ADD_SUBDIRECTORY` when the ACE's subject is a directory, so the
/// permission that lets a foreign principal create a replacement entry (the
/// mechanism a rename-replace needs) is already covered by the identical
/// bit this module rejects on a file for direct content-write. Combined
/// with `ACL_DELETE`/`ACL_DELETE_CHILD` (unlink the existing entry or a
/// child), `ACL_WRITE_ATTRIBUTES`/`ACL_WRITE_EXTATTRIBUTES` (retarget the
/// directory's own metadata), and the two transitive meta-rights
/// `ACL_WRITE_SECURITY`/`ACL_CHANGE_OWNER`, this is the full
/// directory-mutating set: every permission that lets a principal replace,
/// remove, or redirect an entry beneath the directory, or escalate to one of
/// those, without needing the process uid's own write access. Reusing the
/// same scan (rather than a second one) also means it inherits that scan's
/// existing failure-mode coverage for free: deny entries never widen access
/// so are skipped without weakening the any-allow-rejects policy, entries
/// materialized on this directory by ACL inheritance are enumerated exactly
/// like any other literal entry, and no principal/qualifier (user, group, or
/// a synthetic principal such as `everyone`) is special-cased -- an ACE
/// granting a rejected permission is rejected regardless of who it names.
/// Anything the FFI cannot enumerate or classify -- an unrecognized
/// `acl_get_entry`/`acl_get_perm_np` return -- fails the whole check closed
/// rather than treating the unparsed entry as benign; see
/// [`macos_acl::acl_grants_rejected_permission`]'s doc comment for the
/// itemized fail-closed behavior this inherits verbatim.
///
/// See this module's "Parent-directory chain" doc comment for what this
/// closes and the sticky-bit exception's rationale.
#[cfg(unix)]
#[cfg_attr(not(feature = "metal-gpu"), allow(dead_code))]
pub(crate) fn reject_if_mmap_parent_directory_chain_weak(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;

    let canonical = path.canonicalize().map_err(|e| {
        format!(
            "failed to canonicalize {} for parent-directory trust check: {e}",
            path.display()
        )
    })?;
    let process_uid = current_uid();

    let mut dir = canonical.parent();
    while let Some(d) = dir {
        let meta = std::fs::metadata(d)
            .map_err(|e| format!("failed to stat parent directory {}: {e}", d.display()))?;
        let mode = meta.mode();
        let dir_uid = meta.uid();
        if mmap_parent_dir_trust_boundary_issue(mode, dir_uid, process_uid) {
            return Err(format!(
                "refusing to load {}: parent directory {} is outside lattice's \
                 trust boundary (mode {:o}, dir uid {dir_uid}, process uid \
                 {process_uid}) -- writable by group/other without the sticky \
                 bit, or owned by a uid that is neither the process's uid nor \
                 root. A writable ancestor directory permits a rename-replace \
                 of the checkpoint file regardless of the file's own \
                 permissions. Fix the directory's permissions (not \
                 group/other writable, unless sticky) and, unless it is \
                 root-owned, its ownership before retrying.",
                path.display(),
                d.display(),
                mode & 0o1777,
            ));
        }

        #[cfg(target_os = "macos")]
        {
            let dir_file = std::fs::File::open(d).map_err(|e| {
                format!(
                    "failed to open parent directory {} for extended-ACL check: {e}",
                    d.display()
                )
            })?;
            if let Some(perm_name) = macos_acl::acl_grants_rejected_permission(&dir_file, d)? {
                return Err(format!(
                    "refusing to load {}: parent directory {} carries a macOS \
                     extended ACL entry granting {perm_name}, a permission that \
                     confers directory mutation (directly or transitively) to \
                     another principal. Extended ACLs are additive to POSIX \
                     mode bits and invisible to the mode/uid check above, so a \
                     0700 process-owned directory can still carry an ACE that \
                     lets a foreign principal create, delete, or rename-replace \
                     an entry beneath it -- including the checkpoint file \
                     itself -- before this gate's checkpoint is opened. Remove \
                     the ACL (`chmod -N {}`) before retrying.",
                    path.display(),
                    d.display(),
                    d.display(),
                ));
            }
        }

        dir = d.parent();
    }
    Ok(())
}

#[cfg(not(unix))]
#[cfg_attr(not(feature = "metal-gpu"), allow(dead_code))]
pub(crate) fn reject_if_mmap_parent_directory_chain_weak(_path: &Path) -> Result<(), String> {
    Ok(())
}

/// Refuse to trust a checkpoint file for read-only no-copy mmap when it
/// falls outside lattice's trust boundary: writable by group or other (POSIX
/// mode bits), owned by a uid that is neither the process's uid nor root, or
/// -- macOS only -- carrying a macOS extended ACL entry that grants a
/// permission conferring write (directly or transitively) to any principal.
/// This file is loaded via a read-only no-copy mmap, so a principal who can
/// write to it could truncate or replace it in place between this check and
/// the GPU reading the mapped pages; removing that write access is what
/// closes the race, not merely warning about it. Unix-only: there is no
/// portable equivalent of POSIX mode/uid bits to gate on elsewhere, so
/// non-Unix targets accept unconditionally.
///
/// `file` must be the same already-open descriptor the caller is about to
/// mmap -- see the module doc comment's "fd-binding invariant". `path` is
/// used only to name the file in error messages; it is never re-resolved.
#[cfg(unix)]
#[cfg_attr(not(feature = "metal-gpu"), allow(dead_code))]
pub(crate) fn reject_if_mmap_file_trust_boundary_weak(
    file: &File,
    path: &Path,
) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;

    let meta = file
        .metadata()
        .map_err(|e| format!("failed to stat {}: {e}", path.display()))?;
    let mode = meta.mode();
    let file_uid = meta.uid();
    let process_uid = current_uid();
    if mmap_file_trust_boundary_issue(mode, file_uid, process_uid) {
        return Err(format!(
            "refusing to load {}: checkpoint file is outside lattice's trust \
             boundary (mode {:o}, file uid {file_uid}, process uid {process_uid}) \
             -- writable by group/other, or owned by a uid that is neither the \
             process's uid nor root. This file is loaded via a read-only no-copy \
             mmap, so a principal able to write to it could truncate or replace \
             it in place between validation and the GPU reading the mapped \
             pages; shape/bounds validation alone cannot defend against that. \
             Fix the file's permissions (not group/other writable) and, unless \
             it is root-owned, its ownership (owned by the deploying user) \
             before retrying.",
            path.display(),
            mode & 0o777,
        ));
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(perm_name) = macos_acl::acl_grants_rejected_permission(file, path)? {
            return Err(format!(
                "refusing to load {}: file carries a macOS extended ACL entry \
                 granting {perm_name}, a permission that confers write access \
                 (directly or transitively) to another principal. Extended ACLs \
                 are invisible to POSIX mode bits and can grant this even to an \
                 otherwise owner-locked-down file, bypassing the mode/uid check \
                 above. This file is loaded via a read-only no-copy mmap; remove \
                 the ACL (`chmod -N {}`) before retrying.",
                path.display(),
                path.display(),
            ));
        }
    }

    Ok(())
}

#[cfg(not(unix))]
#[cfg_attr(not(feature = "metal-gpu"), allow(dead_code))]
pub(crate) fn reject_if_mmap_file_trust_boundary_weak(
    _file: &File,
    _path: &Path,
) -> Result<(), String> {
    Ok(())
}

/// `O_NONBLOCK`, verified against this platform's `fcntl.h` (no `libc` crate
/// dependency, mirroring [`current_uid`]'s direct `getuid()` FFI pattern).
/// Darwin: `0x00000004` (`bsd/sys/fcntl.h`). Linux: `0o4000` octal --
/// architecture-independent for every target this workspace builds
/// (`asm-generic/fcntl.h`); the handful of Linux ports with a divergent
/// value (sparc, mips, alpha, parisc) are not lattice build targets.
#[cfg(target_os = "macos")]
const O_NONBLOCK: i32 = 0x0000_0004;
#[cfg(all(unix, not(target_os = "macos")))]
const O_NONBLOCK: i32 = 0o4000;

/// `O_NOFOLLOW`, verified the same way as [`O_NONBLOCK`] above (no `libc`
/// crate dependency, direct per-platform header value). Darwin:
/// `0x00000100` (`bsd/sys/fcntl.h`). Linux: `0o0400000` octal
/// (`asm-generic/fcntl.h`), architecture-independent for every target this
/// workspace builds -- same caveat as `O_NONBLOCK` above for the handful of
/// non-target Linux ports with a divergent value.
#[cfg(target_os = "macos")]
const O_NOFOLLOW: i32 = 0x0000_0100;
#[cfg(all(unix, not(target_os = "macos")))]
const O_NOFOLLOW: i32 = 0o400000;

/// `ELOOP` ("too many levels of symbolic links" -- also the errno `open()`
/// returns for an `O_NOFOLLOW` open whose final path component is a
/// symlink), verified per-platform the same way as `O_NOFOLLOW` above:
/// `std::io::ErrorKind::FilesystemLoop` is not yet stable
/// (rust-lang/rust#86442), so this compares `raw_os_error()` directly.
/// Darwin: `62` (`sys/errno.h`). Linux: `40` (`asm-generic/errno.h`),
/// architecture-independent for every target this workspace builds.
#[cfg(target_os = "macos")]
const ELOOP: i32 = 62;
#[cfg(all(unix, not(target_os = "macos")))]
const ELOOP: i32 = 40;

// POSIX-common `fcntl` command values, identical across every unix this
// workspace targets.
#[cfg(unix)]
const F_GETFL: i32 = 3;
#[cfg(unix)]
const F_SETFL: i32 = 4;

#[cfg(unix)]
unsafe extern "C" {
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
}

/// Open `path` for a trusted, no-copy mmap and return the still-open
/// `(File, Metadata)` pair every one of this crate's five mmap loaders needs
/// -- callers never re-open or re-stat `path` afterward (see this module's
/// fd-binding invariant).
///
/// # FIFO / device DoS
///
/// A plain `File::open` on a FIFO with no writer at the other end blocks the
/// calling thread indefinitely; the path here names a checkpoint file on
/// disk that is fully attacker-controlled up to this point (nothing has
/// validated it yet), so opening blindly lets an attacker-planted FIFO or
/// device node at the model path hang model loading before the trust gate
/// ever runs. Opening with `O_NONBLOCK` makes `open()` on a FIFO return
/// immediately (POSIX) instead of blocking; the regular-file check below
/// then rejects it -- and any device, socket, or directory -- outright.
/// `O_NONBLOCK` is cleared via `fcntl` before this returns so the caller's
/// mmap/read of the confirmed regular file behaves exactly as if the flag
/// had never been set (it has no effect on regular-file I/O either way --
/// this clears it purely for hygiene, so nothing downstream needs to know
/// this function ever used it).
///
/// # Final-component symlink substitution (#1037)
///
/// Without `O_NOFOLLOW`, a symlink planted at the *final* path component
/// (the checkpoint filename itself, not a parent directory) is followed
/// transparently by `open()`: every check this function and its caller run
/// afterward -- regular-file, mode/uid, ACL, shape, post-map identity --
/// then validates and maps whatever *that symlink points at*, not the path
/// the caller asked for. Opening with `O_NOFOLLOW` alongside `O_NONBLOCK`
/// (both set in the one `custom_flags` call above -- `custom_flags` replaces
/// the flag bits on each call rather than OR-ing across calls, so setting
/// them separately would silently drop one) makes `open()` fail with
/// `ELOOP` when the final component is a symlink, which this function
/// detects via `raw_os_error()` (`std::io::ErrorKind::FilesystemLoop` is not
/// yet stable, rust-lang/rust#86442) and reports with a dedicated message.
/// Because the `open()` itself fails,
/// there is no fd for a symlinked final component, so `is_file()` and the
/// mode/uid/ACL checks below are simply never reached for it -- rejection
/// happens at `open()`, not by a later check catching a symlink after the
/// fact.
///
/// Scope: this protects only the *final* path component. A symlink earlier
/// in the path -- i.e. one of the checkpoint's parent directories -- is
/// still followed by the kernel when resolving the rest of the path, same
/// as the parent-directory rename-replace gap this module's doc comment
/// already documents as residual, unshipped hardening.
///
/// Runs the parent-directory chain walk
/// ([`reject_if_mmap_parent_directory_chain_weak`]) and the mode/uid/ACL
/// trust checks ([`reject_if_mmap_file_trust_boundary_weak`]) on the same fd
/// last, after the regular-file guard, so a FIFO/device never reaches the
/// ACL FFI calls at all.
#[cfg(unix)]
#[cfg_attr(not(feature = "metal-gpu"), allow(dead_code))]
pub(crate) fn open_trusted_mmap_file(path: &Path) -> Result<(File, std::fs::Metadata), String> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;

    // `custom_flags` REPLACES the custom-flag bits on each call rather than
    // OR-ing across calls, so O_NONBLOCK and O_NOFOLLOW must both be set in
    // this one call -- a second `custom_flags` call here would silently
    // drop O_NONBLOCK.
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NONBLOCK | O_NOFOLLOW)
        .open(path)
        .map_err(|e| {
            if e.raw_os_error() == Some(ELOOP) {
                format!(
                    "refusing to open {}: final path component is a symlink -- \
                     checkpoint loads must name a regular file directly. \
                     O_NOFOLLOW rejects this before any trust or shape check \
                     runs, so a symlink planted at the final path component \
                     cannot redirect validation and mapping to a different, \
                     attacker-controlled file. This guards only the final \
                     path component -- a symlink earlier in the path (a \
                     parent directory) is still followed.",
                    path.display()
                )
            } else {
                format!("failed to open {}: {e}", path.display())
            }
        })?;

    let meta = file
        .metadata()
        .map_err(|e| format!("failed to stat {}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!(
            "refusing to load {}: not a regular file -- checkpoint loads \
             must be a plain file on disk. FIFOs, devices, sockets, and \
             directories are rejected here, before any trust or shape \
             check runs, so an attacker-planted node at the model path \
             cannot block or misdirect the read-only load.",
            path.display()
        ));
    }

    let fd = file.as_raw_fd();
    // SAFETY: `fd` is a valid, open descriptor for the duration of these
    // two calls; `F_GETFL`/`F_SETFL` take/return an `int` and do not retain
    // the fd past the call.
    let flags = unsafe { fcntl(fd, F_GETFL) };
    if flags == -1 {
        return Err(format!(
            "failed to read fd flags for {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: same fd, clearing exactly the `O_NONBLOCK` bit this function
    // set when opening.
    if unsafe { fcntl(fd, F_SETFL, flags & !O_NONBLOCK) } == -1 {
        return Err(format!(
            "failed to clear O_NONBLOCK for {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }

    reject_if_mmap_parent_directory_chain_weak(path)?;
    reject_if_mmap_file_trust_boundary_weak(&file, path)?;

    Ok((file, meta))
}

#[cfg(not(unix))]
#[cfg_attr(not(feature = "metal-gpu"), allow(dead_code))]
pub(crate) fn open_trusted_mmap_file(path: &Path) -> Result<(File, std::fs::Metadata), String> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let meta = file
        .metadata()
        .map_err(|e| format!("failed to stat {}: {e}", path.display()))?;
    Ok((file, meta))
}

/// Post-map integrity recheck, replacing a previous
/// `mmap.len()`-based "recheck" that shipped for #1037:
/// `mmap.len()` only echoes the length `memmap2` *requested* when it derived
/// the mapping's size via its own internal `fstat`, called *before*
/// `mmap()` -- it is not a second, independent observation of the file, so
/// comparing a header-derived bound against it can never disagree with a
/// stat taken moments earlier by the same code path and detects nothing.
///
/// This instead fstats `file`'s fd again -- the *same* still-open fd the
/// mapping was made from -- strictly after the `mmap()` call, and compares
/// it against `prior`, the stat [`open_trusted_mmap_file`] captured before
/// mapping. Checks `st_size`, `st_ino`, `st_dev`, and `st_mtime` (seconds +
/// nanoseconds): size/mtime catch an in-place truncate or rewrite that
/// landed in the validate-then-map window; ino/dev are a defense-in-depth
/// cross-check on the fd's own identity (the fd-binding invariant already
/// means this function's `file` cannot itself have been swapped to a
/// different inode by a path-level rename-replace -- see the module doc
/// comment -- but comparing them here costs nothing and catches a
/// same-identity assumption violation should one ever creep in upstream).
///
/// See this module's "Trust-boundary scope" doc comment for what this does
/// and does not defend against.
#[cfg(unix)]
#[cfg_attr(not(feature = "metal-gpu"), allow(dead_code))]
pub(crate) fn verify_mmap_target_unchanged(
    file: &File,
    prior: &std::fs::Metadata,
    path: &Path,
) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;

    let now = file
        .metadata()
        .map_err(|e| format!("failed to re-stat {} after mmap: {e}", path.display()))?;
    if now.size() != prior.size()
        || now.ino() != prior.ino()
        || now.dev() != prior.dev()
        || now.mtime() != prior.mtime()
        || now.mtime_nsec() != prior.mtime_nsec()
    {
        return Err(format!(
            "refusing to trust mapped {}: file identity/content changed \
             between pre-map validation and mmap (size {}->{}, ino {}->{}, \
             dev {}->{}, mtime {}.{:09}->{}.{:09}) -- a truncate-or-replace \
             race landed in the validate-then-map window",
            path.display(),
            prior.size(),
            now.size(),
            prior.ino(),
            now.ino(),
            prior.dev(),
            now.dev(),
            prior.mtime(),
            prior.mtime_nsec(),
            now.mtime(),
            now.mtime_nsec(),
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
#[cfg_attr(not(feature = "metal-gpu"), allow(dead_code))]
pub(crate) fn verify_mmap_target_unchanged(
    _file: &File,
    _prior: &std::fs::Metadata,
    _path: &Path,
) -> Result<(), String> {
    Ok(())
}

/// macOS extended-ACL inspection, isolated in its own submodule so the FFI
/// surface (opaque `acl_t`/`acl_entry_t`/`acl_permset_t` pointers, ACL
/// constants) does not leak into the rest of this file. Direct libc FFI --
/// no new crate dependency, mirroring [`current_uid`]'s `getuid()` pattern.
#[cfg(target_os = "macos")]
mod macos_acl {
    use std::fs::File;
    use std::os::fd::AsRawFd;
    use std::os::raw::c_void;
    use std::path::Path;

    // `sys/acl.h` (Darwin, read from this machine's active Xcode/CLT SDK --
    // `xcrun --show-sdk-path`). Opaque handles are treated as `*mut c_void`;
    // only pointer identity matters to this code, never their layout.
    const ACL_TYPE_EXTENDED: i32 = 0x0000_0100;
    const ACL_FIRST_ENTRY: i32 = 0;
    const ACL_NEXT_ENTRY: i32 = -1;
    const ACL_EXTENDED_ALLOW: i32 = 1;

    // Full `acl_perm_t` enum, verified value-by-value against this machine's
    // `sys/acl.h` (not trusted from memory -- a previous version of this
    // module shipped an `ACL_TYPE_EXTENDED` value that silently no-op'd the
    // whole check).
    // `ACL_LIST_DIRECTORY`, `ACL_ADD_FILE`, `ACL_SEARCH`, and
    // `ACL_ADD_SUBDIRECTORY` are header-defined aliases of `ACL_READ_DATA`,
    // `ACL_WRITE_DATA`, `ACL_EXECUTE`, and `ACL_APPEND_DATA` respectively --
    // omitted here since they duplicate values already classified.
    // Read-only/execute/sync perms are only referenced from `HARMLESS_PERMS`
    // below, which is `#[cfg(test)]`-only (see its doc comment) -- so these
    // are legitimately dead in a non-test build.
    #[cfg_attr(not(test), allow(dead_code))]
    const ACL_READ_DATA: u32 = 1 << 1;
    const ACL_WRITE_DATA: u32 = 1 << 2;
    #[cfg_attr(not(test), allow(dead_code))]
    const ACL_EXECUTE: u32 = 1 << 3;
    const ACL_DELETE: u32 = 1 << 4;
    const ACL_APPEND_DATA: u32 = 1 << 5;
    const ACL_DELETE_CHILD: u32 = 1 << 6;
    #[cfg_attr(not(test), allow(dead_code))]
    const ACL_READ_ATTRIBUTES: u32 = 1 << 7;
    const ACL_WRITE_ATTRIBUTES: u32 = 1 << 8;
    #[cfg_attr(not(test), allow(dead_code))]
    const ACL_READ_EXTATTRIBUTES: u32 = 1 << 9;
    const ACL_WRITE_EXTATTRIBUTES: u32 = 1 << 10;
    #[cfg_attr(not(test), allow(dead_code))]
    const ACL_READ_SECURITY: u32 = 1 << 11;
    const ACL_WRITE_SECURITY: u32 = 1 << 12;
    const ACL_CHANGE_OWNER: u32 = 1 << 13;
    #[cfg_attr(not(test), allow(dead_code))]
    const ACL_SYNCHRONIZE: u32 = 1 << 20;

    /// The full `acl_perm_t` partition, rejected half: permissions that
    /// confer write on the mmap'd file's content directly (`WRITE_DATA`,
    /// `APPEND_DATA`, `DELETE`, `DELETE_CHILD`, `WRITE_ATTRIBUTES`,
    /// `WRITE_EXTATTRIBUTES`) plus the two meta-rights that confer it
    /// transitively (`WRITE_SECURITY` -- rewrite the ACL to grant yourself
    /// write; `CHANGE_OWNER` -- take ownership, then re-grant). An ACE
    /// granting any one of these to any principal is rejected. See the
    /// `classification_tests` module below for the full perm-by-perm
    /// classification table.
    const REJECTED_PERMS: [(u32, &str); 8] = [
        (ACL_WRITE_DATA, "ACL_WRITE_DATA"),
        (ACL_APPEND_DATA, "ACL_APPEND_DATA"),
        (ACL_DELETE, "ACL_DELETE"),
        (ACL_DELETE_CHILD, "ACL_DELETE_CHILD"),
        (ACL_WRITE_ATTRIBUTES, "ACL_WRITE_ATTRIBUTES"),
        (ACL_WRITE_EXTATTRIBUTES, "ACL_WRITE_EXTATTRIBUTES"),
        (ACL_WRITE_SECURITY, "ACL_WRITE_SECURITY"),
        (ACL_CHANGE_OWNER, "ACL_CHANGE_OWNER"),
    ];

    /// The partition's other half: read/execute/sync permissions that confer
    /// no write capability, direct or transitive. Not consulted at runtime
    /// (an ACE not in `REJECTED_PERMS` is accepted by omission) -- listed so
    /// a test can assert `REJECTED_PERMS ∪ HARMLESS_PERMS` covers the full
    /// 14-value `acl_perm_t` enum exhaustively, with no perm left
    /// unclassified and no overlap between the two halves.
    #[cfg(test)]
    const HARMLESS_PERMS: [(u32, &str); 6] = [
        (ACL_READ_DATA, "ACL_READ_DATA"),
        (ACL_EXECUTE, "ACL_EXECUTE"),
        (ACL_READ_ATTRIBUTES, "ACL_READ_ATTRIBUTES"),
        (ACL_READ_EXTATTRIBUTES, "ACL_READ_EXTATTRIBUTES"),
        (ACL_READ_SECURITY, "ACL_READ_SECURITY"),
        (ACL_SYNCHRONIZE, "ACL_SYNCHRONIZE"),
    ];

    // POSIX-common errno values, identical on every unix `errno.h`
    // (including Darwin) for these specific codes.
    const ENOENT: i32 = 2;
    const EINVAL: i32 = 22;

    unsafe extern "C" {
        fn acl_get_fd_np(fd: i32, acl_type: i32) -> *mut c_void;
        fn acl_get_entry(acl: *mut c_void, entry_id: i32, entry_p: *mut *mut c_void) -> i32;
        fn acl_get_tag_type(entry: *mut c_void, tag_type_p: *mut i32) -> i32;
        fn acl_get_permset(entry: *mut c_void, permset_p: *mut *mut c_void) -> i32;
        fn acl_get_perm_np(permset: *mut c_void, perm: u32) -> i32;
        fn acl_free(obj: *mut c_void) -> i32;
    }

    /// `Some(perm_name)` if `file`'s extended ACL carries at least one
    /// `ACL_EXTENDED_ALLOW` entry granting a permission in `REJECTED_PERMS`,
    /// to any principal (conservative policy -- see the module doc comment
    /// for why this does not discriminate owner/root from foreign
    /// qualifiers). `file` is the caller's already-open descriptor for the
    /// same file it is about to mmap -- see the module doc comment's
    /// "fd-binding invariant"; the ACL is read via `acl_get_fd_np`, not
    /// `acl_get_file`, so no path is resolved here at all.
    ///
    /// A file with no extended ACL at all (`acl_get_fd_np` fails with
    /// `ENOENT` or `EINVAL` -- verified empirically on this machine against
    /// both `/tmp` and `$TMPDIR`, an APFS `/var/folders` volume, to report
    /// `ENOENT`; `EINVAL` is accepted too since this may vary by filesystem
    /// or macOS version) is `Ok(None)`. Every other FFI failure -- from
    /// `acl_get_fd_np` itself, or from enumerating entries, tags, permsets,
    /// or perms once an ACL handle is obtained -- fails closed as an `Err`,
    /// including an `acl_get_entry` return that is not the one documented
    /// end-of-list sentinel (`man acl_get_entry`: `-1`/`EINVAL` on an
    /// `ACL_NEXT_ENTRY` call after at least one successful entry). Silently
    /// treating an unrecognized error as "no more entries" or "permission
    /// not granted" would let a genuinely write-granting ACE slip past an
    /// inspection failure instead of rejecting the file.
    pub(super) fn acl_grants_rejected_permission(
        file: &File,
        path: &Path,
    ) -> Result<Option<&'static str>, String> {
        // SAFETY: `file` is a valid, open file descriptor for the duration
        // of this call; `acl_get_fd_np` does not retain the fd past the
        // call.
        let acl = unsafe { acl_get_fd_np(file.as_raw_fd(), ACL_TYPE_EXTENDED) };
        if acl.is_null() {
            let err = std::io::Error::last_os_error();
            if matches!(err.raw_os_error(), Some(ENOENT) | Some(EINVAL)) {
                // No extended ACL on this file -- the common, expected case.
                return Ok(None);
            }
            return Err(format!(
                "failed to read extended ACL for {}: {err}",
                path.display()
            ));
        }

        let result = scan_acl_for_rejected_perm(acl, path);

        // SAFETY: `acl` is the same non-null handle returned by
        // `acl_get_fd_np` above, freed exactly once here regardless of
        // `scan_acl_for_rejected_perm`'s outcome.
        unsafe {
            acl_free(acl);
        }

        result
    }

    /// Walk every entry of an already-fetched `acl` handle looking for an
    /// `ACL_EXTENDED_ALLOW` entry granting a `REJECTED_PERMS` permission.
    /// Split out from [`acl_grants_rejected_permission`] so every early
    /// return here still passes through that function's single `acl_free`
    /// call rather than needing one at each fail-closed exit.
    fn scan_acl_for_rejected_perm(
        acl: *mut c_void,
        path: &Path,
    ) -> Result<Option<&'static str>, String> {
        let mut entry: *mut c_void = std::ptr::null_mut();
        let mut entry_id = ACL_FIRST_ENTRY;
        loop {
            let is_next_entry_call = entry_id == ACL_NEXT_ENTRY;
            // SAFETY: `acl` is a valid handle owned by the caller for the
            // duration of this call, and `entry` is a valid out-pointer.
            let rc = unsafe { acl_get_entry(acl, entry_id, &mut entry) };
            if rc != 0 {
                let err = std::io::Error::last_os_error();
                if is_next_entry_call && err.raw_os_error() == Some(EINVAL) {
                    // The one documented end-of-list sentinel (`man
                    // acl_get_entry`): an `ACL_NEXT_ENTRY` call after the
                    // last entry has already been returned.
                    break;
                }
                return Err(format!(
                    "failed to enumerate ACL entries for {}: acl_get_entry \
                     returned {rc} ({err}), which is not the documented \
                     end-of-list sentinel -- failing closed rather than \
                     silently truncating ACL enumeration",
                    path.display()
                ));
            }
            entry_id = ACL_NEXT_ENTRY;

            let mut tag_type: i32 = 0;
            // SAFETY: `entry` was just populated by a successful
            // `acl_get_entry` call above.
            if unsafe { acl_get_tag_type(entry, &mut tag_type) } != 0 {
                return Err(format!(
                    "failed to read ACL entry tag type for {}: {} -- \
                     failing closed",
                    path.display(),
                    std::io::Error::last_os_error()
                ));
            }
            // Only ALLOW entries grant a permission; DENY entries (and any
            // other tag type) cannot widen access, so skipping them here
            // does not weaken the fail-closed posture.
            if tag_type != ACL_EXTENDED_ALLOW {
                continue;
            }

            let mut permset: *mut c_void = std::ptr::null_mut();
            // SAFETY: `entry` is the same valid, just-populated entry as
            // above.
            if unsafe { acl_get_permset(entry, &mut permset) } != 0 {
                return Err(format!(
                    "failed to read ACL entry permission set for {}: {} -- \
                     failing closed",
                    path.display(),
                    std::io::Error::last_os_error()
                ));
            }

            for (perm, name) in REJECTED_PERMS {
                // SAFETY: `permset` was just populated by a successful
                // `acl_get_permset` call above.
                let rc = unsafe { acl_get_perm_np(permset, perm) };
                match rc {
                    1 => return Ok(Some(name)),
                    0 => continue,
                    _ => {
                        return Err(format!(
                            "failed to check ACL permission {name} for {}: \
                             acl_get_perm_np returned {rc} ({}) -- failing \
                             closed",
                            path.display(),
                            std::io::Error::last_os_error()
                        ));
                    }
                }
            }
        }

        Ok(None)
    }

    #[cfg(test)]
    mod fail_closed_tests {
        use super::*;
        use std::os::fd::AsRawFd;

        // Proves `acl_get_entry` failing with the SAME errno
        // (`EINVAL`) our end-of-list sentinel checks for is not, by itself,
        // proof of end-of-list -- the sentinel additionally requires the
        // call to have been an `ACL_NEXT_ENTRY` continuation, not a first
        // lookup. `man acl_get_entry` documents `EINVAL` for two distinct
        // conditions: (a) the genuine end-of-list case
        // (`ACL_NEXT_ENTRY` after the last entry) and (b) `entry_id` not
        // being `ACL_FIRST_ENTRY`, `ACL_NEXT_ENTRY`, or a valid entry
        // index. This test reproduces (b) directly against a real
        // extended-ACL handle (an out-of-range numeric `entry_id`, which is
        // condition (b) verbatim) and confirms `scan_acl_for_rejected_perm`'s
        // `is_next_entry_call` guard is what tells the two apart, not the
        // errno alone -- if the guard were dropped (checking only
        // `errno == EINVAL`), this exact rc/errno pair would be
        // misclassified as end-of-list and the scan would silently return
        // `Ok(None)` instead of failing closed.
        //
        // This is the closest reproduction of an "ACL inspection error"
        // achievable from safe, portable test setup: forcing
        // `acl_get_tag_type`/`acl_get_permset`/`acl_get_perm_np` themselves
        // to fail requires a corrupted/deallocated handle, which is
        // undefined behavior to construct deliberately.
        #[test]
        fn out_of_range_entry_id_reproduces_a_non_end_of_list_einval() {
            let tmp = tempfile::tempdir().expect("tempdir create");
            let path = tmp.path().join("fail_closed_probe.q4");
            std::fs::write(&path, b"not a real checkpoint, only the ACL matters here")
                .expect("write tempfile");
            let status = std::process::Command::new("chmod")
                .arg("+a")
                .arg("everyone allow read")
                .arg(&path)
                .status()
                .expect("run chmod +a");
            assert!(
                status.success(),
                "chmod +a must succeed to set up this test"
            );

            let file = File::open(&path).expect("open tempfile");
            // SAFETY: `file` is a valid, open file descriptor for the
            // duration of this call.
            let acl = unsafe { acl_get_fd_np(file.as_raw_fd(), ACL_TYPE_EXTENDED) };
            assert!(
                !acl.is_null(),
                "acl_get_fd_np must succeed: the ACE set up above must be present"
            );

            let bogus_entry_id = 12345;
            assert_ne!(
                bogus_entry_id, ACL_NEXT_ENTRY,
                "sanity: the probe id must not accidentally equal the sentinel id"
            );
            let mut entry: *mut c_void = std::ptr::null_mut();
            // SAFETY: `acl` was just checked non-null above; `entry` is a
            // valid out-pointer. An out-of-range `entry_id` is documented
            // (`man acl_get_entry`) to fail rather than read out of bounds.
            let rc = unsafe { acl_get_entry(acl, bogus_entry_id, &mut entry) };
            let err = std::io::Error::last_os_error();

            // SAFETY: `acl` is the same non-null handle from `acl_get_fd_np`
            // above, freed exactly once here.
            unsafe {
                acl_free(acl);
            }

            assert_ne!(rc, 0, "an out-of-range entry_id must fail, not succeed");
            assert_eq!(
                err.raw_os_error(),
                Some(EINVAL),
                "documented failure mode for an invalid entry_id is EINVAL -- \
                 the same errno the genuine end-of-list sentinel uses, which is \
                 exactly why `scan_acl_for_rejected_perm` must not treat \
                 errno==EINVAL alone as sufficient"
            );
            let is_next_entry_call = bogus_entry_id == ACL_NEXT_ENTRY;
            assert!(
                !is_next_entry_call,
                "this call is not the ACL_NEXT_ENTRY continuation the sentinel \
                 requires, so `scan_acl_for_rejected_perm` must classify this \
                 exact (rc, errno) pair as a failure to fail closed on, not as \
                 end-of-list"
            );
        }
    }

    #[cfg(test)]
    mod classification_tests {
        use super::*;
        use std::collections::HashSet;

        // The 14 distinct values the Darwin `acl_perm_t` enum defines
        // (`sys/acl.h`), excluding header-defined aliases that duplicate an
        // already-listed value (`ACL_LIST_DIRECTORY`, `ACL_ADD_FILE`,
        // `ACL_SEARCH`, `ACL_ADD_SUBDIRECTORY`). Independent of
        // `REJECTED_PERMS`/`HARMLESS_PERMS` above so this test cannot pass
        // by construction if this module's classification silently drops or
        // duplicates a perm.
        const ALL_ACL_PERM_T_VALUES: [u32; 14] = [
            1 << 1,
            1 << 2,
            1 << 3,
            1 << 4,
            1 << 5,
            1 << 6,
            1 << 7,
            1 << 8,
            1 << 9,
            1 << 10,
            1 << 11,
            1 << 12,
            1 << 13,
            1 << 20,
        ];

        // Regression test for #1037: an earlier write-class list covered 5
        // of the 6 direct-content-write perms (missing `DELETE_CHILD`) and
        // both meta-rights (`WRITE_SECURITY`, `CHANGE_OWNER`) were not
        // classified anywhere, so an ACE granting only `WRITE_SECURITY` or
        // only `CHANGE_OWNER` passed the old check. This test proves the
        // classification is a complete, non-overlapping partition of the
        // full enum, so that failure mode cannot recur one perm at a time.
        #[test]
        fn rejected_and_harmless_perms_exactly_partition_the_full_acl_perm_t_enum() {
            let rejected: HashSet<u32> = REJECTED_PERMS.iter().map(|(v, _)| *v).collect();
            let harmless: HashSet<u32> = HARMLESS_PERMS.iter().map(|(v, _)| *v).collect();
            let all: HashSet<u32> = ALL_ACL_PERM_T_VALUES.iter().copied().collect();

            assert_eq!(
                rejected.len(),
                REJECTED_PERMS.len(),
                "REJECTED_PERMS must not contain a duplicate value"
            );
            assert_eq!(
                harmless.len(),
                HARMLESS_PERMS.len(),
                "HARMLESS_PERMS must not contain a duplicate value"
            );
            assert!(
                rejected.is_disjoint(&harmless),
                "a perm classified as both rejected and harmless is a \
                 contradiction, not just redundant"
            );
            let union: HashSet<u32> = rejected.union(&harmless).copied().collect();
            assert_eq!(
                union, all,
                "REJECTED_PERMS ∪ HARMLESS_PERMS must cover every acl_perm_t \
                 value exactly once -- a value present in `all` but missing \
                 here is an unclassified perm (must default-reject per this \
                 round's ruling, i.e. belong in REJECTED_PERMS unless proven \
                 harmless); a value here but absent from `all` is stale"
            );

            // Mechanized SDK-header binding (#1037 fold): the assertion
            // above compares this module's classification against
            // `ALL_ACL_PERM_T_VALUES`, a hand-copied literal -- a Darwin SDK
            // perm addition would not trip it, since nothing here re-derives
            // that list from anything. This second pass re-derives the
            // "full enum" side from the SDK header itself
            // (`$(xcrun --show-sdk-path)/usr/include/sys/acl.h`) so a future
            // SDK adding an `__DARWIN_ACL_*` perm fails this test instead of
            // silently landing unclassified. Skips (not fails) when `xcrun`
            // or the header is unavailable -- this must not turn "no Xcode
            // SDK on this machine" into a spurious CI failure.
            let Some(sdk_path) = std::process::Command::new("xcrun")
                .args(["--show-sdk-path"])
                .output()
                .ok()
                .filter(|out| out.status.success())
                .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
            else {
                eprintln!(
                    "SKIP: `xcrun --show-sdk-path` unavailable -- skipping SDK-header-bound \
                     acl_perm_t classification check"
                );
                return;
            };
            let header_path = std::path::Path::new(&sdk_path).join("usr/include/sys/acl.h");
            let Ok(header) = std::fs::read_to_string(&header_path) else {
                eprintln!(
                    "SKIP: {} not present -- skipping SDK-header-bound acl_perm_t \
                     classification check",
                    header_path.display()
                );
                return;
            };

            // Resolve every `#define __DARWIN_ACL_<NAME> ...` macro to a
            // `u32` value, in two passes so alias defines (which reference
            // another `__DARWIN_ACL_*` name instead of a literal shift, e.g.
            // `__DARWIN_ACL_LIST_DIRECTORY __DARWIN_ACL_READ_DATA`) resolve
            // through to the same value as their target. This header also
            // defines unrelated `__DARWIN_ACL_*` macros outside the
            // `acl_perm_t` enum (ACL entry-flag bits like
            // `__DARWIN_ACL_FLAG_NO_INHERIT`, extended-ACL type tags like
            // `__DARWIN_ACL_EXTENDED_ALLOW`) that must NOT be treated as
            // perms merely for sharing the `__DARWIN_ACL_` prefix -- scoping
            // to just the `typedef enum { ... } acl_perm_t;` block below,
            // rather than every `__DARWIN_ACL_*` `#define` in the file, is
            // what keeps those out.
            const PREFIX: &str = "__DARWIN_ACL_";
            let mut defines: std::collections::HashMap<&str, u32> =
                std::collections::HashMap::new();
            for line in header.lines() {
                let Some(rest) = line
                    .trim()
                    .strip_prefix("#define ")
                    .and_then(|s| s.strip_prefix(PREFIX))
                else {
                    continue;
                };
                let Some((name, value_str)) = rest.split_once(char::is_whitespace) else {
                    continue;
                };
                let value_str = value_str.trim();
                if let Some(shift_str) = value_str
                    .strip_prefix("(1<<")
                    .and_then(|s| s.strip_suffix(')'))
                    && let Ok(shift) = shift_str.trim().parse::<u32>()
                {
                    defines.insert(name, 1u32 << shift);
                }
            }
            for line in header.lines() {
                let Some(rest) = line
                    .trim()
                    .strip_prefix("#define ")
                    .and_then(|s| s.strip_prefix(PREFIX))
                else {
                    continue;
                };
                let Some((name, value_str)) = rest.split_once(char::is_whitespace) else {
                    continue;
                };
                if defines.contains_key(name) {
                    continue;
                }
                if let Some(alias) = value_str.trim().strip_prefix(PREFIX)
                    && let Some(&v) = defines.get(alias)
                {
                    defines.insert(name, v);
                }
            }

            // Scope to exactly the `acl_perm_t` enum block, not every
            // `__DARWIN_ACL_*` macro in the header.
            let block_end = header.find("} acl_perm_t;").unwrap_or_else(|| {
                panic!(
                    "could not find `}} acl_perm_t;` in {} -- the header's enum \
                     format changed and this test's parser needs updating",
                    header_path.display()
                )
            });
            let block_start = header[..block_end]
                .rfind("typedef enum {")
                .unwrap_or_else(|| {
                    panic!(
                        "found `}} acl_perm_t;` in {} but no preceding `typedef enum {{` -- \
                     the header's enum format changed and this test's parser needs updating",
                        header_path.display()
                    )
                });
            let enum_block = &header[block_start..block_end];

            let mut header_perms: HashSet<u32> = HashSet::new();
            for line in enum_block.lines() {
                let Some(idx) = line.find(PREFIX) else {
                    continue;
                };
                let rest = &line[idx + PREFIX.len()..];
                let ident_end = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .unwrap_or(rest.len());
                let name = &rest[..ident_end];
                if let Some(&v) = defines.get(name) {
                    header_perms.insert(v);
                }
            }

            assert!(
                !header_perms.is_empty(),
                "parsed zero acl_perm_t perms out of {} -- the header's enum/macro \
                 format changed and this test's parser needs updating, not silently \
                 passing vacuously",
                header_path.display()
            );
            assert_eq!(
                union,
                header_perms,
                "this module's REJECTED_PERMS ∪ HARMLESS_PERMS must classify EXACTLY \
                 the acl_perm_t perms {} currently defines -- a perm the header defines \
                 but this module doesn't classify is a real gap (a Darwin SDK addition \
                 landing unclassified, silently treated as harmless-by-omission); a \
                 perm classified here but absent from the header is stale and should \
                 be removed",
                header_path.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CPU-only unit test for `mmap_file_trust_boundary_issue` (#1037): the
    // pure predicate behind `reject_if_mmap_file_trust_boundary_weak`, the
    // fail-closed check applied when a checkpoint file is mmap'd no-copy for
    // GPU dispatch and is group/other-writable or owned by a non-root,
    // non-process uid. No filesystem needed.
    #[cfg(unix)]
    #[test]
    fn mmap_file_trust_boundary_issue_fires_on_group_or_other_writable_or_foreign_non_root_owner() {
        assert!(
            mmap_file_trust_boundary_issue(0o664, 1000, 1000),
            "group-writable (0o664) must trigger the trust-boundary check"
        );
        assert!(
            mmap_file_trust_boundary_issue(0o646, 1000, 1000),
            "other-writable (0o646) must trigger the trust-boundary check"
        );
        assert!(
            !mmap_file_trust_boundary_issue(0o644, 1000, 1000),
            "owner-writable-only, group/other read-only must not trigger the trust-boundary check"
        );
        assert!(
            !mmap_file_trust_boundary_issue(0o600, 1000, 1000),
            "owner-only rwx must not trigger the trust-boundary check"
        );
        assert!(
            mmap_file_trust_boundary_issue(0o400, 999, 1000),
            "a file not owned by the current uid (and not root) must trigger the \
             trust-boundary check even when read-only"
        );
        assert!(
            !mmap_file_trust_boundary_issue(0o644, 0, 1000),
            "a root-owned, owner-read-only file must be accepted for a \
             non-root process uid -- root-owned shared model directories are \
             a documented deployment shape, not an attack"
        );
        assert!(
            mmap_file_trust_boundary_issue(0o664, 0, 1000),
            "root ownership does not excuse group/other-writability"
        );
    }

    // Regression test for #1037: the trust-boundary check must reject, not
    // merely warn about, a weak file. This test calls the public wrapper
    // directly against a real group/other-writable tempfile, so it is
    // Device-free and runs on every CPU-only leg.
    #[cfg(unix)]
    #[test]
    fn reject_if_mmap_file_trust_boundary_weak_fails_closed_on_writable_file() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().expect("tempdir create");
        let path = tmp.path().join("world_writable.q4");
        std::fs::write(
            &path,
            b"not a real checkpoint, only permissions matter here",
        )
        .expect("write tempfile");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666))
            .expect("chmod 0o666");
        let file = File::open(&path).expect("open tempfile");

        let result = reject_if_mmap_file_trust_boundary_weak(&file, &path);
        assert!(
            result.is_err(),
            "a group/other-writable checkpoint file must be refused, not merely warned about"
        );
        let msg = result.expect_err("checked is_err above");
        assert!(
            msg.contains("refusing to load") && msg.contains(&path.display().to_string()),
            "error must state the load is refused and name the offending path; got: {msg}"
        );

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .expect("chmod 0o600");
        let private_file = File::open(&path).expect("reopen tempfile after chmod");
        assert!(
            reject_if_mmap_file_trust_boundary_weak(&private_file, &path).is_ok(),
            "an owner-only file owned by the current process must be accepted"
        );
    }

    // Regression test for #1037: a checkpoint whose parent directory is
    // group- or other-writable, without the sticky bit, must be refused --
    // a writable ancestor directory permits a rename-replace of the
    // checkpoint file regardless of the file's own permissions, since
    // `unlink`/`rename` are gated on the *directory's* write permission, not
    // the target file's.
    //
    // This test requires `open_trusted_mmap_file` to run the
    // parent-directory-chain check in addition to the file-level check --
    // the fixture's checkpoint file itself is owner-only 0600, so only the
    // directory-chain check catches the writable ancestor.
    #[cfg(unix)]
    #[test]
    fn open_trusted_mmap_file_rejects_a_writable_parent_directory() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().expect("tempdir create");
        let writable_dir = tmp.path().join("writable_parent");
        std::fs::create_dir(&writable_dir).expect("create parent dir fixture");
        std::fs::set_permissions(&writable_dir, std::fs::Permissions::from_mode(0o777))
            .expect("chmod parent dir 0o777 (no sticky bit)");

        let path = writable_dir.join("checkpoint.q4");
        std::fs::write(
            &path,
            b"not a real checkpoint, only the parent dir matters here",
        )
        .expect("write tempfile");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .expect("chmod checkpoint file 0o600");

        let err = open_trusted_mmap_file(&path).expect_err(
            "a checkpoint under a group/other-writable, non-sticky parent directory \
             must be refused even though the file itself is owner-only",
        );
        assert!(
            err.contains("parent directory"),
            "error must name the parent directory as the rejection cause; got: {err}"
        );
    }

    // Control: a world-writable parent directory that ALSO carries the
    // sticky bit (the `/tmp` pattern) must be accepted -- the sticky bit
    // restricts rename/unlink within the directory to the entry's owner,
    // the directory's owner, or root, closing the exact race a plain
    // world-writable directory would permit.
    #[cfg(unix)]
    #[test]
    fn open_trusted_mmap_file_accepts_a_sticky_world_writable_parent_directory() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().expect("tempdir create");
        let sticky_dir = tmp.path().join("sticky_parent");
        std::fs::create_dir(&sticky_dir).expect("create parent dir fixture");
        std::fs::set_permissions(&sticky_dir, std::fs::Permissions::from_mode(0o1777))
            .expect("chmod parent dir 0o1777 (world-writable + sticky)");

        let path = sticky_dir.join("checkpoint.q4");
        std::fs::write(
            &path,
            b"not a real checkpoint, only the parent dir matters here",
        )
        .expect("write tempfile");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .expect("chmod checkpoint file 0o600");

        open_trusted_mmap_file(&path).expect(
            "a checkpoint under a sticky world-writable parent directory must be \
             accepted -- the sticky bit already prevents the rename-replace this \
             check exists to stop",
        );
    }

    // A 0700, process-owned ancestor directory is invisible to the mode/uid
    // check above, but on macOS can still carry an extended ACL granting a
    // foreign principal a directory-mutating permission -- the ancestor-walk
    // analogue of `macos_acl_regression_tests::content_write_grant_is_rejected`
    // below, which covers the same gap on the target file itself. Skips
    // cleanly (rather than failing) if this sandbox cannot set an extended
    // ACL via `chmod +a` at all, since that is an environment limitation, not
    // an assertion about the code under test.
    #[cfg(target_os = "macos")]
    #[test]
    fn reject_if_mmap_parent_directory_chain_weak_fails_closed_on_ancestor_acl_grant() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().expect("tempdir create");
        let ancestor = tmp.path().join("acl_ancestor");
        std::fs::create_dir(&ancestor).expect("create ancestor dir fixture");
        std::fs::set_permissions(&ancestor, std::fs::Permissions::from_mode(0o700))
            .expect("chmod ancestor dir 0o700 (owner-only, passes the mode/uid check)");

        let status = std::process::Command::new("chmod")
            .arg("+a")
            .arg("everyone allow add_file")
            .arg(&ancestor)
            .status()
            .expect("run chmod +a");
        if !status.success() {
            eprintln!(
                "skipping reject_if_mmap_parent_directory_chain_weak_fails_closed_on_ancestor_acl_grant: \
                 this sandbox cannot set an extended ACL via chmod +a"
            );
            return;
        }

        let path = ancestor.join("checkpoint.q4");
        std::fs::write(
            &path,
            b"not a real checkpoint, only the ancestor ACL matters here",
        )
        .expect("write tempfile");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .expect("chmod checkpoint file 0o600");

        let result = reject_if_mmap_parent_directory_chain_weak(&path);
        assert!(
            result.is_err(),
            "a 0700 process-owned ancestor directory carrying an extended ACL that \
             grants a foreign principal add_file (directory-mutating) rights must \
             still be refused -- that ACE is invisible to the mode/uid check alone \
             and lets the grantee replace the checkpoint entry before it is opened"
        );
        let msg = result.expect_err("checked is_err above");
        assert!(
            msg.contains("extended ACL") && msg.contains(&ancestor.display().to_string()),
            "error must name the extended-ACL cause and the offending ancestor; got: {msg}"
        );
    }

    // Regression test for #1037: a FIFO planted at the model path must be
    // rejected by `open_trusted_mmap_file` -- and rejected FAST, not by
    // hanging the calling thread. Before the O_NONBLOCK + regular-file guard
    // existed, `File::open` on a FIFO with no writer at the other end blocks
    // indefinitely; this test proves the fix by asserting the call returns
    // (any return, not just an error) inside a generous deadline, then
    // separately asserts it specifically rejected the FIFO as non-regular.
    // `mkfifo` (POSIX, not a macOS-only tool) keeps this test portable to
    // the Linux CI legs `cargo test -p lattice-inference --lib mmap_trust`
    // runs on.
    #[test]
    fn open_trusted_mmap_file_rejects_a_fifo_without_blocking() {
        let tmp = tempfile::tempdir().expect("tempdir create");
        let path = tmp.path().join("planted.q4");
        let status = std::process::Command::new("mkfifo")
            .arg(&path)
            .status()
            .expect("run mkfifo");
        assert!(status.success(), "mkfifo must succeed to set up this test");

        let (tx, rx) = std::sync::mpsc::channel();
        let probe_path = path.clone();
        // A pre-fix blocking `File::open` on this FIFO (no writer ever
        // connects) would hang this thread forever; running the call on a
        // detached thread and racing it against a deadline on `rx.recv_timeout`
        // turns "hangs forever" into an observable test failure instead of an
        // actually-hung test process.
        std::thread::spawn(move || {
            let result = open_trusted_mmap_file(&probe_path);
            let _ = tx.send(result);
        });
        let result = rx.recv_timeout(std::time::Duration::from_secs(5)).expect(
            "open_trusted_mmap_file did not return within 5s -- it blocked on the \
                 planted FIFO, meaning the O_NONBLOCK open-time guard regressed",
        );
        let err = result.expect_err("a FIFO must be rejected, not accepted, for a mmap load");
        assert!(
            err.contains("not a regular file"),
            "error must name the FIFO as the non-regular-file rejection cause; got: {err}"
        );
    }

    // A directory at the model path must be rejected the same way as a FIFO
    // -- both are non-regular-file nodes an attacker (or a misconfigured
    // deployment) could plant at the path a checkpoint is expected at.
    #[test]
    fn open_trusted_mmap_file_rejects_a_directory() {
        let tmp = tempfile::tempdir().expect("tempdir create");
        let dir_path = tmp.path().join("planted_dir.q4");
        std::fs::create_dir(&dir_path).expect("create planted directory");

        let err = open_trusted_mmap_file(&dir_path)
            .expect_err("a directory must be rejected, not accepted, for a mmap load");
        assert!(
            err.contains("not a regular file"),
            "error must name the directory as the non-regular-file rejection cause; got: {err}"
        );
    }

    // A normal regular file must still be accepted end-to-end through
    // `open_trusted_mmap_file` -- proves the regular-file/O_NONBLOCK guard
    // added in front of the existing trust checks does not itself reject
    // the common case.
    #[test]
    fn open_trusted_mmap_file_accepts_a_regular_owner_only_file() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().expect("tempdir create");
        let path = tmp.path().join("real.q4");
        std::fs::write(
            &path,
            b"not a real checkpoint, just needs to be a regular file",
        )
        .expect("write tempfile");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .expect("chmod 0o600");

        let (file, meta) =
            open_trusted_mmap_file(&path).expect("a regular owner-only file must be accepted");
        assert!(meta.is_file());
        assert_eq!(
            file.metadata().expect("stat returned file").len(),
            meta.len()
        );
    }

    // Regression test for #1037: a symlink planted at the *final* path
    // component must be rejected outright, not
    // followed to whatever it points at. Proves both directions:
    // `open_trusted_mmap_file` on the symlink path fails, and the real file
    // it points at is still loadable directly (the guard rejects the
    // symlink, not the target's contents or permissions).
    //
    // This test requires `open_trusted_mmap_file`'s `O_NOFOLLOW` open flag
    // to reject the symlink at `open()` -- the target's own
    // regular-file/mode/uid checks would otherwise pass, since the symlink
    // resolves to a valid checkpoint.
    #[test]
    fn open_trusted_mmap_file_rejects_a_symlinked_final_component() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().expect("tempdir create");
        let real_path = tmp.path().join("real.q4");
        std::fs::write(&real_path, b"a real checkpoint-shaped regular file")
            .expect("write real fixture");
        std::fs::set_permissions(&real_path, std::fs::Permissions::from_mode(0o600))
            .expect("chmod 0o600");

        let symlink_path = tmp.path().join("planted_symlink.q4");
        std::os::unix::fs::symlink(&real_path, &symlink_path).expect("create symlink fixture");

        let err = open_trusted_mmap_file(&symlink_path).expect_err(
            "a symlink at the final path component must be rejected, not followed to its target",
        );
        assert!(
            err.contains("symlink"),
            "error must name the symlink as the rejection cause; got: {err}"
        );

        open_trusted_mmap_file(&real_path)
            .expect("the real (non-symlink) file must still be accepted directly");
    }

    // Regression test for #1037: proves `verify_mmap_target_unchanged`
    // (the real fstat recheck replacing the `mmap.len()` theater) actually
    // detects a truncate-after-validate race, by driving the exact sequence
    // the doc comment describes -- open, stat (via `open_trusted_mmap_file`),
    // mmap, truncate, recheck.
    #[test]
    fn verify_mmap_target_unchanged_rejects_a_truncate_after_validate_race() {
        let tmp = tempfile::tempdir().expect("tempdir create");
        let path = tmp.path().join("truncate_race.q4");
        std::fs::write(&path, vec![0xABu8; 4096]).expect("write fixture");

        let (file, prior_meta) =
            open_trusted_mmap_file(&path).expect("open + trust-gate the fixture");
        // SAFETY: read-only mmap of a tempfile this test alone controls.
        let _mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.expect("mmap fixture");

        // Simulate a writer truncating the file in the window between the
        // pre-map stat captured above and this recheck -- the exact race
        // `verify_mmap_target_unchanged` exists to catch.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("reopen fixture for truncation")
            .set_len(4096 - 8)
            .expect("truncate fixture in place");

        let result = verify_mmap_target_unchanged(&file, &prior_meta, &path);
        assert!(
            result.is_err(),
            "a size change between the pre-map stat and this recheck must be rejected"
        );
        let msg = result.expect_err("checked is_err above");
        assert!(
            msg.contains("refusing to trust mapped") && msg.contains("size"),
            "error must state the mapping is untrusted and cite the size mismatch; got: {msg}"
        );
    }

    // Control for the above: an UNCHANGED file between the pre-map stat and
    // the recheck must be accepted -- proves the recheck is not simply
    // "always reject", the same shape `harmless_read_only_grant_is_accepted`
    // proves for the ACL check below.
    #[test]
    fn verify_mmap_target_unchanged_accepts_an_unmodified_file() {
        let tmp = tempfile::tempdir().expect("tempdir create");
        let path = tmp.path().join("unchanged.q4");
        std::fs::write(&path, vec![0xCDu8; 4096]).expect("write fixture");

        let (file, prior_meta) =
            open_trusted_mmap_file(&path).expect("open + trust-gate the fixture");
        let _mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.expect("mmap fixture");

        assert!(
            verify_mmap_target_unchanged(&file, &prior_meta, &path).is_ok(),
            "a file untouched between the pre-map stat and the post-map recheck must be accepted"
        );
    }

    // macOS-only regression tests for the extended-ACL bypass this module
    // closes: a 0600 owner-matched file (passes the mode/uid check above)
    // that ALSO carries an extended ACL granting a *different* principal a
    // rejected permission must still be refused -- that ACE is invisible to
    // `st_mode`/`st_uid` alone. Uses the `chmod` CLI (`chmod +a`) rather
    // than the `acl_set_file` FFI to keep the test's ACL-authoring path
    // independent of the ACL-reading code under test.
    #[cfg(target_os = "macos")]
    mod macos_acl_regression_tests {
        use super::*;
        use std::os::unix::fs::PermissionsExt;

        /// Build a fresh 0600 owner-matched tempfile, `chmod +a <ace>` it,
        /// and return the still-open `File` handle the caller passes to
        /// `reject_if_mmap_file_trust_boundary_weak` -- proving the check
        /// operates on that handle (fd-bound), not a re-derived path lookup:
        /// nothing here re-opens or re-resolves `path` after this point.
        fn tempfile_with_ace(
            dir: &std::path::Path,
            name: &str,
            ace: &str,
        ) -> (std::path::PathBuf, File) {
            let path = dir.join(name);
            std::fs::write(&path, b"not a real checkpoint, only the ACL matters here")
                .expect("write tempfile");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
                .expect("chmod 0o600");
            let status = std::process::Command::new("chmod")
                .arg("+a")
                .arg(ace)
                .arg(&path)
                .status()
                .expect("run chmod +a");
            assert!(
                status.success(),
                "chmod +a must succeed to set up this test: {ace}"
            );
            let file = File::open(&path).expect("open tempfile after chmod +a");
            (path, file)
        }

        // Control: a 0600 owner-matched file with no extended ACL at all
        // must be accepted.
        #[test]
        fn no_acl_is_accepted() {
            let tmp = tempfile::tempdir().expect("tempdir create");
            let path = tmp.path().join("no_acl.q4");
            std::fs::write(&path, b"not a real checkpoint, only the ACL matters here")
                .expect("write tempfile");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
                .expect("chmod 0o600");
            let file = File::open(&path).expect("open tempfile");
            assert!(
                reject_if_mmap_file_trust_boundary_weak(&file, &path).is_ok(),
                "a 0600 owner-matched file with no extended ACL must be accepted (control)"
            );
        }

        // A content-write grant is still rejected, kept under the fd-bound
        // signature.
        #[test]
        fn content_write_grant_is_rejected() {
            let tmp = tempfile::tempdir().expect("tempdir create");
            let (path, file) = tempfile_with_ace(
                tmp.path(),
                "content_write.q4",
                "everyone allow write,writeattr,writeextattr,delete",
            );
            let result = reject_if_mmap_file_trust_boundary_weak(&file, &path);
            assert!(
                result.is_err(),
                "a file carrying an extended ACL that grants content-write access to \
                 another principal must be refused even though st_mode reports 0600 \
                 owner-only"
            );
            let msg = result.expect_err("checked is_err above");
            assert!(
                msg.contains("extended ACL"),
                "error must name the extended-ACL cause; got: {msg}"
            );
        }

        // An ACE granting ONLY `writesecurity` (ACL_WRITE_SECURITY) --
        // no direct content-write perm at all -- must still be rejected: it
        // lets the grantee rewrite the ACL itself and grant themselves write
        // afterward, so it transitively confers write even though it is not
        // itself a content-write permission.
        #[test]
        fn write_security_only_grant_is_rejected() {
            let tmp = tempfile::tempdir().expect("tempdir create");
            let (path, file) = tempfile_with_ace(
                tmp.path(),
                "write_security_only.q4",
                "everyone allow writesecurity",
            );
            let result = reject_if_mmap_file_trust_boundary_weak(&file, &path);
            assert!(
                result.is_err(),
                "an ACE granting only ACL_WRITE_SECURITY must be rejected -- it lets \
                 the grantee rewrite the ACL to grant themselves write later"
            );
            assert!(
                result
                    .expect_err("checked is_err above")
                    .contains("ACL_WRITE_SECURITY")
            );
        }

        // An ACE granting ONLY `chown` (ACL_CHANGE_OWNER) must also be
        // rejected -- it lets the grantee take ownership of the file and
        // then re-grant themselves write as the new owner.
        #[test]
        fn change_owner_only_grant_is_rejected() {
            let tmp = tempfile::tempdir().expect("tempdir create");
            let (path, file) =
                tempfile_with_ace(tmp.path(), "change_owner_only.q4", "everyone allow chown");
            let result = reject_if_mmap_file_trust_boundary_weak(&file, &path);
            assert!(
                result.is_err(),
                "an ACE granting only ACL_CHANGE_OWNER must be rejected -- it lets \
                 the grantee take ownership and then re-grant themselves write"
            );
            assert!(
                result
                    .expect_err("checked is_err above")
                    .contains("ACL_CHANGE_OWNER")
            );
        }

        // Proves the allowlist admits benign ACEs and the policy is not
        // "reject any ACE whatsoever": a read-only grant to another
        // principal carries no write capability, direct or transitive, and
        // must be accepted.
        #[test]
        fn harmless_read_only_grant_is_accepted() {
            let tmp = tempfile::tempdir().expect("tempdir create");
            let (path, file) = tempfile_with_ace(tmp.path(), "read_only.q4", "everyone allow read");
            assert!(
                reject_if_mmap_file_trust_boundary_weak(&file, &path).is_ok(),
                "a harmless read-only ACE must be accepted -- the policy allowlists \
                 harmless perms, it does not reject every ACE unconditionally"
            );
        }
    }
}
