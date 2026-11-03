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

fn parse_optional_limit<'a>(args: &'a [String], flag: &str) -> Result<Option<&'a str>, String> {
    let Some(index) = args.iter().position(|arg| arg == flag) else {
        return Ok(None);
    };
    args.get(index + 1)
        .map(|value| Some(value.as_str()))
        .ok_or_else(|| format!("{flag}: expected a positive integer"))
}

pub(super) fn parse_max_context_args(args: &[String]) -> Result<Option<usize>, String> {
    lattice_serve::serving_provider::parse_max_context(parse_optional_limit(args, "--max-context")?)
}

pub(super) fn parse_port(args: &[String], env: Option<&std::ffi::OsStr>) -> Result<u16, String> {
    if let Some(index) = args.iter().position(|arg| arg == "--port") {
        let raw = args
            .get(index + 1)
            .ok_or_else(|| "--port: expected a port number".to_owned())?;
        return raw
            .parse::<u16>()
            .map_err(|_| format!("--port: invalid port value {raw:?}"));
    }
    match env {
        None => Ok(11435),
        Some(value) if value.is_empty() => Ok(11435),
        Some(value) => value
            .to_str()
            .and_then(|raw| raw.parse::<u16>().ok())
            .ok_or_else(|| format!("LATTICE_SERVE_PORT: invalid port value {value:?}")),
    }
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
    match parse_optional_limit(args, "--max-pending")? {
        Some(raw) => raw.parse::<usize>().map_err(|_| {
            format!("--max-pending: invalid value {raw:?} (expected a positive integer)")
        }),
        None => Ok(lattice_serve::serve::metal_worker::DEFAULT_MAX_PENDING_JOBS),
    }
}

pub(super) fn parse_residency_limits(
    args: &[String],
) -> Result<lattice_serve::serve::lora::ResidencyLimits, String> {
    use lattice_serve::serve::lora::{ResidencyLimits, parse_resident_limit};
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| (*part).to_owned()).collect()
    }

    #[test]
    fn trailing_max_pending_requires_value() {
        let error = parse_max_pending(&args(&["lattice_serve", "--max-pending"]))
            .expect_err("present flag must require a value");
        assert!(error.contains("--max-pending"), "{error}");
    }

    #[test]
    fn trailing_max_context_requires_value() {
        let argv = args(&["lattice_serve", "--max-context"]);
        let error = parse_max_context_args(&argv).expect_err("present flag must require a value");
        assert!(error.contains("--max-context"), "{error}");
    }

    #[test]
    fn omitted_and_valid_limits_keep_existing_behavior() {
        let empty = args(&["lattice_serve"]);
        assert_eq!(
            parse_max_pending(&empty).unwrap(),
            lattice_serve::serve::metal_worker::DEFAULT_MAX_PENDING_JOBS
        );
        assert_eq!(parse_max_context_args(&empty).unwrap(), None);
        let valid = args(&[
            "lattice_serve",
            "--max-pending",
            "4",
            "--max-context",
            "128",
        ]);
        assert_eq!(parse_max_pending(&valid).unwrap(), 4);
        assert_eq!(parse_max_context_args(&valid).unwrap(), Some(128));
    }
}

#[cfg(test)]
mod port_tests {
    use super::parse_port;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| (*part).to_owned()).collect()
    }

    fn port(args: &[String], env: Option<&str>) -> Result<u16, String> {
        parse_port(args, env.map(std::ffi::OsStr::new))
    }

    #[test]
    fn standalone_port_rejects_invalid_explicit_value() {
        for raw in ["nope", "-1", "65536"] {
            let argv = args(&["lattice_serve", "--port", raw]);
            let error = port(&argv, Some("10001")).expect_err("invalid port must fail");
            assert!(error.contains("--port"), "{error}");
        }
    }

    #[test]
    fn standalone_port_rejects_missing_explicit_value() {
        let argv = args(&["lattice_serve", "--port"]);
        let error = port(&argv, Some("10001")).expect_err("missing port must fail");
        assert!(error.contains("--port"), "{error}");
    }

    #[test]
    fn standalone_port_rejects_invalid_environment() {
        let argv = args(&["lattice_serve"]);
        for raw in ["nope", "-1", "65536"] {
            let error = port(&argv, Some(raw)).expect_err("invalid environment must fail");
            assert!(
                error.contains("LATTICE_SERVE_PORT") && error.contains(raw),
                "{error}"
            );
        }
    }

    #[test]
    fn standalone_port_empty_environment_defaults_and_cli_wins() {
        let absent = args(&["lattice_serve"]);
        assert_eq!(port(&absent, Some("")).unwrap(), 11435);
        let cli = args(&["lattice_serve", "--port", "12345"]);
        assert_eq!(port(&cli, Some("invalid")).unwrap(), 12345);
    }

    #[test]
    fn standalone_port_keeps_cli_env_default_precedence() {
        let cli = args(&["lattice_serve", "--port", "12345"]);
        assert_eq!(port(&cli, Some("10001")).unwrap(), 12345);
        let ephemeral = args(&["lattice_serve", "--port", "0"]);
        assert_eq!(port(&ephemeral, Some("10001")).unwrap(), 0);
        let absent = args(&["lattice_serve"]);
        assert_eq!(port(&absent, Some("10001")).unwrap(), 10001);
        assert_eq!(port(&absent, None).unwrap(), 11435);
    }
    #[cfg(unix)]
    #[test]
    fn standalone_port_rejects_nonunicode_environment_unless_cli_wins() {
        use std::os::unix::ffi::OsStrExt;
        let invalid = std::ffi::OsStr::from_bytes(&[0xff]);
        let absent = args(&["lattice_serve"]);
        let error = parse_port(&absent, Some(invalid)).expect_err("non-UTF-8 env must fail");
        assert!(error.contains("LATTICE_SERVE_PORT"), "{error}");
        let cli = args(&["lattice_serve", "--port", "12345"]);
        assert_eq!(parse_port(&cli, Some(invalid)).unwrap(), 12345);
    }
}
