//! Regression guard for #923/#942 (round 2): every example binary that loads a
//! Q4 model directory must resolve its architecture config via
//! `Qwen35Config::from_model_dir`, which fails closed on a missing
//! `config.json` instead of silently substituting a guessed preset.
//!
//! #942 round 1 fixed the library loaders but left nine example `main()`s on
//! the old `dir.join("config.json").exists() { from_config_json } else {
//! qwenXX_preset() }` pattern — a sibling-invocation-path miss of the exact
//! class the fix was meant to close. This test greps `examples/*.rs` for the
//! anti-pattern so a reintroduction fails CI instead of silently reappearing.

use std::fs;
use std::path::Path;

#[test]
fn examples_never_reintroduce_config_json_preset_fallback() {
    let examples_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let mut offenders = Vec::new();

    for entry in fs::read_dir(&examples_dir).expect("read crates/inference/examples") {
        let entry = entry.expect("read dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        if src.contains("join(\"config.json\").exists()") {
            offenders.push(path.display().to_string());
        }
    }

    assert!(
        offenders.is_empty(),
        "example loader(s) reintroduced the config.json-exists()-then-preset \
         silent fallback (the class of bug fixed in #923/#942): {offenders:?}. \
         Route model-directory config loading through \
         Qwen35Config::from_model_dir(dir), which fails closed on a missing \
         config.json instead of guessing an architecture preset."
    );
}
