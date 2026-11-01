//! Guard the host/device agreement that bounds Metal MoE route scratch arrays.

const ENGINE: &str = include_str!("../src/forward/metal_qwen35/inner/engine.rs");
const SHADER: &str = include_str!("../src/forward/shaders/qwen35.metal");
const NAMES: [&str; 2] = ["MOE_ROUTE_MAX_EXPERTS", "MOE_ROUTE_MAX_TOP_K"];

fn declaration_value(source: &str, name: &str, shader: bool) -> Result<u64, String> {
    let mut declarations = Vec::new();
    let mut pending = String::new();
    for line in source.lines() {
        let code = line.split_once("//").map_or(line, |(code, _)| code).trim();
        if pending.is_empty() {
            let words: Vec<_> = code
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .filter(|word| !word.is_empty())
                .collect();
            let keyword = if shader { "constexpr" } else { "const" };
            let Some(pos) = words.iter().position(|word| *word == keyword) else {
                continue;
            };
            if !shader && words.get(pos + 1) == Some(&"fn") {
                continue;
            }
        }
        pending.push_str(code);
        pending.push(' ');
        if code.contains(';') {
            if pending
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .any(|word| word == name)
            {
                declarations.push(pending.trim().to_string());
            }
            pending.clear();
        }
    }
    if !pending.is_empty()
        && pending
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .any(|word| word == name)
    {
        return Err(format!("unterminated declaration for {name}"));
    }
    let line = match declarations.as_slice() {
        [] => return Err(format!("missing declaration for {name}")),
        [line] => line.as_str(),
        _ => return Err(format!("duplicate declarations for {name}")),
    };
    let expected = if shader {
        format!("constant constexpr uint {name} = ")
    } else {
        format!("pub(super) const {name}: usize = ")
    };
    let value = line
        .strip_prefix(&expected)
        .and_then(|value| value.strip_suffix(';'))
        .ok_or_else(|| format!("unrecognized declaration for {name}: {line}"))?;
    let value = if shader {
        value
            .strip_suffix('u')
            .ok_or_else(|| format!("nonliteral value for {name}: {line}"))?
    } else {
        value
    };
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("nonliteral value for {name}: {line}"));
    }
    value
        .parse()
        .map_err(|_| format!("out-of-range value for {name}: {line}"))
}

fn check_bounds(engine: &str, shader: &str) -> Result<(), String> {
    for name in NAMES {
        let host = declaration_value(engine, name, false)?;
        let device = declaration_value(shader, name, true)?;
        if host != device {
            return Err(format!("{name} differs: host {host}, Metal {device}"));
        }
    }
    Ok(())
}

#[test]
fn moe_host_and_metal_route_bounds_agree() {
    assert_eq!(check_bounds(ENGINE, SHADER), Ok(()));
}

#[test]
fn guard_rejects_one_sided_bound_changes() {
    let host = ENGINE.replacen(
        "pub(super) const MOE_ROUTE_MAX_EXPERTS: usize = 1024;",
        "pub(super) const MOE_ROUTE_MAX_EXPERTS: usize = 1025;",
        1,
    );
    let shader = SHADER.replacen(
        "constant constexpr uint MOE_ROUTE_MAX_TOP_K = 16u;",
        "constant constexpr uint MOE_ROUTE_MAX_TOP_K = 17u;",
        1,
    );
    assert_ne!(host, ENGINE);
    assert_ne!(shader, SHADER);
    assert!(check_bounds(&host, SHADER).unwrap_err().contains(NAMES[0]));
    assert!(
        check_bounds(ENGINE, &shader)
            .unwrap_err()
            .contains(NAMES[1])
    );
}

#[test]
fn guard_rejects_missing_duplicate_and_nonliteral_declarations() {
    let rust_declaration = "pub(super) const MOE_ROUTE_MAX_EXPERTS: usize = 1024;";
    let shader_declaration = "constant constexpr uint MOE_ROUTE_MAX_TOP_K = 16u;";
    let missing = ENGINE.replacen(rust_declaration, "", 1);
    assert_ne!(missing, ENGINE);
    assert!(
        check_bounds(&missing, SHADER)
            .unwrap_err()
            .contains("missing")
    );

    let duplicate = format!("{SHADER}\n{shader_declaration}\n");
    assert!(
        check_bounds(ENGINE, &duplicate)
            .unwrap_err()
            .contains("duplicate")
    );

    let multiline_duplicate =
        format!("{SHADER}\nconstant constexpr uint\nMOE_ROUTE_MAX_TOP_K = 16u;\n");
    assert!(
        check_bounds(ENGINE, &multiline_duplicate)
            .unwrap_err()
            .contains("duplicate")
    );

    let expression = SHADER.replacen(
        shader_declaration,
        "constant constexpr uint MOE_ROUTE_MAX_TOP_K = 8u + 8u;",
        1,
    );
    assert_ne!(expression, SHADER);
    assert!(
        check_bounds(ENGINE, &expression)
            .unwrap_err()
            .contains("nonliteral")
    );
}
