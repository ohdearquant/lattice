pub(super) fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Presence-only boolean flag (no value): `--preload-vision`, not
/// `--preload-vision true`.
pub(super) fn parse_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// Parses `--max-pending`, rejecting a malformed or negative value
/// outright instead of silently substituting the default (issue #939):
/// the prior `.and_then(|s| s.parse().ok()).unwrap_or(DEFAULT)` chain
/// could not distinguish "flag omitted" from "flag given garbage" --
/// `--max-pending -1` silently became `32`. Range validation (zero, or
/// above `Semaphore::MAX_PERMITS`) is deliberately NOT duplicated here:
/// it happens once, downstream, in `MetalWorker::spawn`
/// (`StartupError::InvalidMaxPending`), shared with `lattice.rs`.
pub(super) fn parse_max_pending(args: &[String]) -> Result<usize, String> {
    match parse_arg(args, "--max-pending") {
        Some(raw) => raw.parse::<usize>().map_err(|_| {
            format!("--max-pending: invalid value {raw:?} (expected a positive integer)")
        }),
        None => Ok(lattice_inference::serve::metal_worker::DEFAULT_MAX_PENDING_JOBS),
    }
}

pub(super) fn parse_residency_limits(
    args: &[String],
) -> Result<lattice_inference::serve::lora::ResidencyLimits, String> {
    use lattice_inference::serve::lora::{ResidencyLimits, parse_resident_limit};
    let parse = |flag: &str, default: usize| {
        let Some(index) = args.iter().position(|arg| arg == flag) else {
            return Ok(default);
        };
        let raw = args
            .get(index + 1)
            .ok_or_else(|| format!("{flag}: expected a positive integer"))?;
        parse_resident_limit(raw).map_err(|err| format!("{flag}: {err}"))
    };
    let defaults = ResidencyLimits::default();
    Ok(ResidencyLimits {
        max_adapters: parse("--max-resident-adapters", defaults.max_adapters)?,
        max_bytes: parse("--max-resident-adapter-bytes", defaults.max_bytes)?,
    })
}
