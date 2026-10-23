use super::public_scheduling_entry_point_tests::strip_comments_and_strings;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Exemptions from the pooling requirement, as `(file, function)` pairs so a
/// same-named function in a different file is not exempted by accident.
///
/// `run_once`: a bench fixture, not a serving path — the process that runs it exits
/// after the measurement, and pooling inside it would change a timed region. It would
/// stop being exempt the moment anything long-lived called it.
const EXEMPT: &[(&str, &str)] = &[("forward/metal_qwen35.rs", "run_once")];

/// Files this guard must find `new_command_buffer` in on every run — a floor under
/// the directory walk, not its ceiling: any other file the walk turns up is still
/// scanned and enforced. Catches the walk itself breaking (an exclusion rule drawn
/// too wide, a rename) before that reads as "nothing left to fix".
const EXPECTED_FILES: &[&str] = &[
    "forward/metal.rs",
    "forward/metal_gemm.rs",
    "forward/metal_qwen35.rs",
];

/// Finds the end of the brace-delimited block opening at or after `from`, by depth
/// count. Returns `None` rather than panicking when no such block exists (a
/// bodyless `mod name;` item) or the count never returns to zero (this scanner's
/// documented blind spot: it does not understand raw strings, so a raw string
/// containing its own `"` and brace characters — e.g. an embedded JSON fixture —
/// can desync its notion of what is source vs. string content for the rest of the
/// file). Both cases are directory-wide-walk realities the single hand-scoped
/// region this guard used before lattice#1630 never had to face: callers treat an
/// unresolvable span as "not a function/module worth blanking" and move on, since
/// the file that produced it is judged on its own content, not on this guess.
fn body_end(source: &str, from: usize) -> Option<usize> {
    let open = from + source[from..].find('{')?;
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + offset + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Overwrites every `#[cfg(..test..)] mod NAME { .. }` with spaces, preserving length
/// so byte offsets still address the same source positions. Test helpers drive Metal
/// from short-lived test binaries and are not the subject.
fn blank_test_modules(source: &str) -> String {
    let mut out = source.as_bytes().to_vec();
    let mut index = 0usize;
    while let Some(found) = source[index..].find("#[cfg(") {
        let start = index + found;
        let Some(close) = source[start..].find(")]") else {
            break;
        };
        let attribute = &source[start..start + close];
        index = start + close + 2;
        if !attribute.contains("test") {
            continue;
        }
        let after = &source[index..];
        let trimmed = after.trim_start();
        let skipped = after.len() - trimmed.len();
        let declaration = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
        if !declaration.starts_with("mod ") {
            continue;
        }
        let decl_start = index + skipped;
        // `mod name;` declares an external file with no inline body here to hide a
        // command-buffer call in; that file is its own entry in the directory walk,
        // judged on its own content. Only `mod name { .. }` has inline text to blank.
        let brace_pos = source[decl_start..].find('{');
        let semi_pos = source[decl_start..].find(';');
        let is_external = match (brace_pos, semi_pos) {
            (Some(b), Some(s)) => s < b,
            (None, Some(_)) => true,
            _ => false,
        };
        if is_external {
            index = decl_start + semi_pos.unwrap_or(0) + 1;
            continue;
        }
        let Some(end) = body_end(source, decl_start) else {
            // Can't bound this module; leave it unblanked rather than guess. Only
            // matters if the file goes on to match `new_command_buffer`, which the
            // caller checks independently of what this pass managed to blank.
            continue;
        };
        for byte in out[start..end].iter_mut() {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
        index = end;
    }
    String::from_utf8(out).expect("blanking preserves UTF-8 boundaries")
}

fn enclosing_functions(source: &str) -> Vec<(&str, usize, usize)> {
    let mut functions = Vec::new();
    let mut index = 0usize;
    while let Some(found) = source[index..].find("fn ") {
        let start = index + found;
        let preceded_by_word = source[..start]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        if preceded_by_word {
            index = start + 3;
            continue;
        }
        let name_start = start + 3;
        let Some(name_len) = source[name_start..].find(['(', '<']) else {
            break;
        };
        let name = source[name_start..name_start + name_len].trim();
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            index = name_start;
            continue;
        }
        // A "fn " match with no resolvable body is not a real function declaration
        // in this scan (a bodyless trait/extern signature, or fallout from the raw-
        // string blind spot noted on `body_end`); skip it rather than treat a parse
        // failure elsewhere in the file as this guard's failure.
        let Some(end) = body_end(source, name_start + name_len) else {
            index = name_start + name_len;
            continue;
        };
        functions.push((name, start, end));
        index = end;
    }
    functions
}

/// Recursively collects every `.rs` file under `dir`, skipping directories named
/// `tests`, `benches`, or `examples`: those hold harnesses whose process exits after
/// the measurement, exactly the population this guard has always excluded ("the
/// remaining `new_command_buffer` sites in the tree are in benches, examples and
/// tests, whose processes exit after the measurement" — lattice#1630).
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("failed to read directory {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| {
            panic!(
                "failed to read a directory entry under {}: {e}",
                dir.display()
            )
        });
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(name, "tests" | "benches" | "examples") {
                continue;
            }
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            // A `tests.rs` file is the file form of the `tests/` directory skipped
            // above: both carry test modules, whose process exits after the
            // measurement. The inline form of the same module is blanked by
            // `blank_test_modules`; this keeps the external form out of the
            // production population it was never meant to enter.
            if path.file_name().and_then(|n| n.to_str()) == Some("tests.rs") {
                continue;
            }
            out.push(path);
        }
    }
}

#[test]
fn every_production_command_buffer_creator_runs_inside_an_autorelease_pool() {
    let src_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs_files(&src_root, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "directory walk under {} found no .rs files at all: the walk, not the \
         source, broke",
        src_root.display()
    );

    let mut discovered: BTreeSet<String> = BTreeSet::new();
    let mut total_sites = 0usize;

    for path in &files {
        let relative = path.strip_prefix(&src_root).unwrap_or(path);
        let label = relative.to_string_lossy().replace('\\', "/");
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

        // `metal_qwen35.rs` carries a large non-Metal prelude and suffix (chat
        // template helpers, MTP resolution, target-independent numerical support)
        // around its production `mod inner { .. }` Metal implementation; scope to
        // that region, as this guard always has, so unrelated code — or this guard's
        // own source below — can't be mistaken for (or hide) a dispatch site.
        let scoped: String = if label == "forward/metal_qwen35.rs" {
            let (_, rest) = source
                .split_once("mod inner {")
                .expect("metal_qwen35.rs must still declare its production `mod inner`");
            let (region, _) = rest
                .split_once("// Target-independent numerical support follows")
                .expect("metal_qwen35.rs must still carry its stable end-of-Metal marker");
            region.to_string()
        } else {
            source
        };

        let scrubbed = blank_test_modules(&strip_comments_and_strings(&scoped));
        assert_eq!(
            scrubbed.len(),
            scoped.len(),
            "scrubbing must preserve offsets ({label})"
        );

        if !scrubbed.contains("new_command_buffer") {
            continue;
        }
        discovered.insert(label.clone());

        let functions = enclosing_functions(&scrubbed);
        let mut creators: BTreeSet<&str> = BTreeSet::new();
        let mut cursor = 0usize;
        while let Some(found) = scrubbed[cursor..].find("new_command_buffer") {
            let at = cursor + found;
            cursor = at + "new_command_buffer".len();
            total_sites += 1;
            let owner = functions
                .iter()
                .rev()
                .find(|(_, start, end)| *start <= at && at < *end)
                .unwrap_or_else(|| {
                    panic!("{label}: command buffer creation at byte {at} sits in no function")
                });
            creators.insert(owner.0);
        }

        for name in creators {
            if EXEMPT.contains(&(label.as_str(), name)) {
                continue;
            }
            let base = name.strip_suffix("_dispatch").unwrap_or_else(|| {
                panic!(
                    "{label}: {name} creates a Metal command buffer with no autorelease \
                     pool (lattice#1630). Rename it to {name}_dispatch and add a wrapper \
                     `fn {name}(..) {{ objc::rc::autoreleasepool(|| self.{name}_dispatch(..)) }}`, \
                     or add it to EXEMPT with the reason it cannot leak."
                )
            });
            let wrapper_start = scrubbed
                .find(&format!("fn {base}("))
                .unwrap_or_else(|| panic!("{label}: {name} has no wrapper named {base}"));
            let wrapper_end = body_end(&scrubbed, wrapper_start + 3)
                .unwrap_or_else(|| panic!("{label}: wrapper {base} has no resolvable body"));
            let wrapper = &scrubbed[wrapper_start..wrapper_end];
            assert!(
                wrapper.contains("objc::rc::autoreleasepool")
                    && wrapper.contains(&format!("{name}(")),
                "{label}: {base} must be the pooling wrapper that calls {name} (lattice#1630)"
            );
        }
    }

    assert!(
        total_sites > 0,
        "found no command buffer creation at all across the discovered files: the \
         scanner, not the source, changed"
    );
    for expected in EXPECTED_FILES {
        assert!(
            discovered.contains(*expected),
            "discovery did not find a `new_command_buffer` call in {expected}: the \
             walk is broken (expected at least: {EXPECTED_FILES:?}, discovered: \
             {discovered:?})"
        );
    }
}
