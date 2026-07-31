use std::collections::BTreeSet;
use std::ops::Range;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum CallSelector {
    Path(&'static [&'static str]),
    Method(&'static str),
}

const DEVICE_SYSTEM_DEFAULT: CallSelector = CallSelector::Path(&["Device", "system_default"]);
const NEW_COMMAND_BUFFER: CallSelector = CallSelector::Method("new_command_buffer");
const SHARED_LOCK_SELECTOR: CallSelector =
    CallSelector::Path(&["lattice_inference", "measurement", "gpu_test_lock"]);
const LOCAL_LOCK_SELECTOR: CallSelector = CallSelector::Path(&["gpu_test_lock"]);
const RAW_GPU_SELECTORS: &[CallSelector] = &[DEVICE_SYSTEM_DEFAULT, NEW_COMMAND_BUFFER];
const SHARED_LOCK_CALL: &str = "lattice_inference::measurement::gpu_test_lock()";
const RAW_HARNESS_ENTRYPOINTS: &[(&str, CallSelector)] = &[
    (
        "benches/decode_attn_bench.rs",
        CallSelector::Path(&["bench_flash_decode"]),
    ),
    (
        "benches/topk_readback.rs",
        CallSelector::Path(&["bench_topk_parity"]),
    ),
    ("examples/bench_concurrent.rs", CallSelector::Path(&["run"])),
    ("examples/bench_dispatch.rs", DEVICE_SYSTEM_DEFAULT),
    ("examples/bench_dispatch2.rs", DEVICE_SYSTEM_DEFAULT),
    ("examples/bench_mps_gemm.rs", DEVICE_SYSTEM_DEFAULT),
    ("examples/bench_simdgroup.rs", DEVICE_SYSTEM_DEFAULT),
    ("examples/profile_metal_decode.rs", DEVICE_SYSTEM_DEFAULT),
];
const ADDITIONAL_GUARDED_ENTRYPOINTS: &[(&str, CallSelector)] = &[
    (
        "examples/bench_embed_quality.rs",
        CallSelector::Path(&["run_bench"]),
    ),
    (
        "examples/bench_metal.rs",
        CallSelector::Path(&["MetalForwardPass", "new"]),
    ),
];
const REVIEWED_TOP_LEVEL_METAL_TARGETS: &[&str] = &[
    "benches/cross_turn_prefix_cache_bench.rs",
    "benches/decode_attn_bench.rs",
    "benches/lm_head_bench.rs",
    "benches/metal_decode_bench.rs",
    "benches/mtp_decode.rs",
    "benches/topk_readback.rs",
    "examples/bench_concurrent.rs",
    "examples/bench_dispatch.rs",
    "examples/bench_dispatch2.rs",
    "examples/bench_embed_quality.rs",
    "examples/bench_embedding.rs",
    "examples/bench_gdn_decode.rs",
    "examples/bench_gdn_prefill_ab.rs",
    "examples/bench_gdn_state.rs",
    "examples/bench_gpu.rs",
    "examples/bench_metal.rs",
    "examples/bench_mps_gemm.rs",
    "examples/bench_persistent_state.rs",
    "examples/bench_profile.rs",
    "examples/bench_pruning.rs",
    "examples/bench_q4_prefill.rs",
    "examples/bench_q8_prefill.rs",
    "examples/bench_quality.rs",
    "examples/bench_simdgroup.rs",
    "examples/bench_stability.rs",
    "examples/bench_suite.rs",
    "examples/decode_profile.rs",
    "examples/layer_sweep.rs",
    "examples/profile_metal.rs",
    "examples/profile_metal_decode.rs",
    "src/bin/backfill_qwen3.rs",
    "src/bin/bench_decode_ab.rs",
    "src/bin/bench_decode_slopefit.rs",
    "src/bin/bench_gdn_prefill_ab.rs",
    "src/bin/bench_logit_dump.rs",
    "src/bin/bench_lora_mixture.rs",
    "src/bin/chat_metal.rs",
    "src/bin/dump_quarot_q4_golden.rs",
    "src/bin/eval_perplexity.rs",
    "src/bin/gramperf_profile.rs",
    "src/bin/lattice.rs",
    "src/bin/lattice/prune_score.rs",
    "src/bin/lattice_serve.rs",
    "src/bin/ppl_metal.rs",
];
const REVIEWED_NON_METAL_TOP_LEVEL_TARGETS: &[&str] = &[
    "benches/attention_dispatch_bench.rs",
    "benches/attn_opt_bench.rs",
    "benches/compute_attention_bench.rs",
    "benches/differential_attention_bench.rs",
    "benches/e2e_bench.rs",
    "benches/elementwise_bench.rs",
    "benches/elementwise_cpu_bench.rs",
    "benches/f16_convert_bench.rs",
    "benches/gated_attention_bench.rs",
    "benches/inference_bench.rs",
    "benches/inference_perf.rs",
    "benches/kv_cache_f16_bench.rs",
    "benches/kv_cache_layout_bench.rs",
    "benches/metrics_bench.rs",
    "benches/native_sparse_attention_bench.rs",
    "benches/pruning_bench.rs",
    "benches/quarot_hadamard_bench.rs",
    "benches/tokenizer_bench.rs",
    "examples/bench_gdn.rs",
    "examples/diff_attn_layer23.rs",
    "examples/diff_gdn_layer.rs",
    "src/bin/moe_admission_sim.rs",
    "src/bin/quantize_q4.rs",
    "src/bin/quantize_quarot.rs",
    "src/bin/qwen35_debug.rs",
    "src/bin/qwen35_generate.rs",
];
const IN_CRATE_COMMAND_BUFFER_TESTS: &[&str] = &[
    "src/forward/metal.rs::rms_norm_matches_pre_854_oracle_on_finite_input",
    "src/forward/metal_qwen35.rs::test_gemv_decode_numerical",
    "src/forward/metal_qwen35.rs::test_gpu_argmax_parity_k1",
    "src/forward/metal_qwen35.rs::decode_attention_reference_fails_closed_on_nan_q",
    "src/forward/metal_qwen35.rs::gemv_q3_decode_matches_cpu_reference_across_shapes",
    "src/forward/metal_qwen35.rs::gemm_q3_tiled_matches_cpu_reference_at_tile_boundaries",
    "src/forward/metal_qwen35.rs::gemm_q3_tiled_differential_fails_closed_on_nan_scale",
    "src/forward/metal_qwen35.rs::gemv_q3_decode_mutation_sensitive_high_plane_bit",
    "src/forward/metal_qwen35.rs::lora_gemv_kernel_matches_cpu_reference",
    "src/forward/metal_qwen35.rs::load_adapter_and_dispatch_lora_if_active",
    "src/forward/metal_qwen35.rs::dispatch_matmul_q4_writes_all_rows",
];
const IN_CRATE_COMMAND_BUFFER_EXEMPTIONS: &[(&str, &str)] = &[(
    "src/forward/metal_qwen35.rs::test_gpu_argmax_parity_k1",
    "existing raw-dispatch test; migration is tracked with the remaining Metal lock work",
)];

const CONSTRUCTION_SELECTORS: &[CallSelector] = &[
    CallSelector::Path(&["MetalQwen35State", "new"]),
    CallSelector::Path(&["MetalQwen35State", "from_q4_dir"]),
];
const LEGACY_CRITERION: &str =
    "legacy Criterion target; source-level locking needs separate benchmark evidence";
const LEGACY_EXAMPLE: &str =
    "legacy manually launched measurement example; lock migration is tracked separately";
const LONG_RUNNING: &str =
    "long-running process; a lifetime lock would starve measurements or exceed its bounded wait";
const CONSTRUCTION_EXEMPTIONS: &[(&str, &str)] = &[
    ("benches/cross_turn_prefix_cache_bench.rs", LEGACY_CRITERION),
    ("benches/lm_head_bench.rs", LEGACY_CRITERION),
    ("benches/metal_decode_bench.rs", LEGACY_CRITERION),
    ("benches/mtp_decode.rs", LEGACY_CRITERION),
    ("examples/bench_gdn_decode.rs", LEGACY_EXAMPLE),
    ("examples/bench_gdn_prefill_ab.rs", LEGACY_EXAMPLE),
    ("examples/bench_gdn_state.rs", LEGACY_EXAMPLE),
    ("examples/bench_persistent_state.rs", LEGACY_EXAMPLE),
    ("examples/bench_pruning.rs", LEGACY_EXAMPLE),
    ("examples/bench_q4_prefill.rs", LEGACY_EXAMPLE),
    ("examples/bench_q8_prefill.rs", LEGACY_EXAMPLE),
    ("examples/bench_quality.rs", LEGACY_EXAMPLE),
    ("examples/bench_stability.rs", LEGACY_EXAMPLE),
    ("examples/bench_suite.rs", LEGACY_EXAMPLE),
    ("examples/decode_profile.rs", LEGACY_EXAMPLE),
    ("examples/profile_metal.rs", LEGACY_EXAMPLE),
    ("src/bin/chat_metal.rs", LONG_RUNNING),
    ("src/bin/lattice.rs", LONG_RUNNING),
    ("src/bin/lattice/prune_score.rs", LONG_RUNNING),
    ("src/bin/lattice_serve.rs", LONG_RUNNING),
    (
        "tests/quarot_q4_composed_golden.rs",
        "opt-in real-model gate runs as a serialized step on an isolated CI runner",
    ),
];

fn rust_sources_under(root: &Path) -> Vec<PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    let mut sources = Vec::new();
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("read source directory") {
            let path = entry.expect("read source entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                sources.push(path);
            }
        }
    }
    sources.sort();
    sources
}

fn assert_construction_inventory_classified(
    discovered: &BTreeSet<String>,
    classified: &BTreeSet<String>,
) {
    assert_eq!(
        discovered, classified,
        "MetalQwen35State construction inventory changed; every site must acquire the shared \
         lock before construction or have an explicit, justified exemption"
    );
}

fn assert_guard_fixture_rejected(source: &str, context: &str) {
    let result = StructuredSource::parse(context, source, false).and_then(|parsed| {
        parsed.validate_selector(
            0..parsed.tokens.len(),
            DEVICE_SYSTEM_DEFAULT,
            SHARED_LOCK_SELECTOR,
            GuardRequirement::Lexical,
        )
    });
    let Err(message) = result else {
        panic!("source contract accepted {context}");
    };
    assert!(
        message.contains(context),
        "source contract failure did not name {context}: {message}"
    );
}

fn assert_command_buffer_fixture_rejected(source: &str, context: &str) {
    let result = StructuredSource::parse(context, source, true).and_then(|parsed| {
        parsed.reject_selector_in_macros(NEW_COMMAND_BUFFER)?;
        let mut found = false;
        for function in parsed.test_functions() {
            let calls = parsed.call_sites(function.body.clone(), NEW_COMMAND_BUFFER, true)?;
            if calls.is_empty() {
                continue;
            }
            found = true;
            parsed.validate_work_sites(
                &calls,
                LOCAL_LOCK_SELECTOR,
                GuardRequirement::Function {
                    closing_brace: function.body.end,
                },
            )?;
        }
        if found {
            Ok(())
        } else {
            Err(format!("{context}: protected work was not classified"))
        }
    });
    let Err(message) = result else {
        panic!("source contract accepted {context}");
    };
    assert!(
        message.contains(context),
        "source contract failure did not name {context}: {message}"
    );
}

#[derive(Clone, Debug)]
enum RustTokenKind {
    Ident { text: String, raw: bool },
    String(String),
    Punct(char),
    Other,
}

#[derive(Clone, Debug)]
struct RustToken {
    kind: RustTokenKind,
    offset: usize,
}

impl RustToken {
    fn ident(&self) -> Option<&str> {
        match &self.kind {
            RustTokenKind::Ident { text, .. } => Some(text),
            _ => None,
        }
    }

    fn is_raw_ident(&self) -> bool {
        matches!(self.kind, RustTokenKind::Ident { raw: true, .. })
    }

    fn is_punct(&self, expected: char) -> bool {
        matches!(self.kind, RustTokenKind::Punct(actual) if actual == expected)
    }
}

fn raw_string_at(source: &str, start: usize) -> Option<Result<(usize, String), String>> {
    let bytes = source.as_bytes();
    let mut cursor = start;
    if matches!(bytes.get(cursor), Some(b'b' | b'c')) {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'r') {
        return None;
    }
    cursor += 1;
    let hashes_start = cursor;
    while bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'"') {
        return None;
    }
    let hashes = cursor - hashes_start;
    let content_start = cursor + 1;
    cursor = content_start;
    while cursor < bytes.len() {
        if bytes[cursor] == b'"'
            && bytes.get(cursor + 1..cursor + 1 + hashes)
                == Some(&bytes[hashes_start..hashes_start + hashes])
        {
            return Some(Ok((
                cursor + 1 + hashes,
                source[content_start..cursor].to_string(),
            )));
        }
        cursor += 1;
    }
    Some(Err(format!("unterminated raw string at byte {start}")))
}

fn quoted_string_at(source: &str, quote: usize) -> Result<(usize, String), String> {
    let bytes = source.as_bytes();
    let mut cursor = quote + 1;
    while cursor < bytes.len() {
        if bytes[cursor] == b'\\' {
            cursor = (cursor + 2).min(bytes.len());
        } else if bytes[cursor] == b'"' {
            return Ok((cursor + 1, source[quote + 1..cursor].to_string()));
        } else {
            cursor += 1;
        }
    }
    Err(format!("unterminated string at byte {quote}"))
}

fn char_literal_end(source: &str, quote: usize) -> Option<Result<usize, String>> {
    let bytes = source.as_bytes();
    if bytes.get(quote) != Some(&b'\'') {
        return None;
    }
    let mut cursor = quote + 1;
    if bytes.get(cursor) == Some(&b'\\') {
        cursor += 1;
        while cursor < bytes.len() {
            if bytes[cursor] == b'\'' {
                return Some(Ok(cursor + 1));
            }
            cursor += 1;
        }
        return Some(Err(format!(
            "unterminated character literal at byte {quote}"
        )));
    }
    let character = source[cursor..].chars().next()?;
    cursor += character.len_utf8();
    (bytes.get(cursor) == Some(&b'\'')).then_some(Ok(cursor + 1))
}

fn rust_tokens(source: &str) -> Result<Vec<RustToken>, String> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut offset = 0usize;
    while offset < bytes.len() {
        if bytes[offset].is_ascii_whitespace() {
            offset += 1;
            continue;
        }
        if bytes[offset..].starts_with(b"//") {
            offset = bytes[offset..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |length| offset + length + 1);
            continue;
        }
        if bytes[offset..].starts_with(b"/*") {
            let start = offset;
            offset += 2;
            let mut depth = 1usize;
            while offset < bytes.len() && depth > 0 {
                if bytes[offset..].starts_with(b"/*") {
                    depth += 1;
                    offset += 2;
                } else if bytes[offset..].starts_with(b"*/") {
                    depth -= 1;
                    offset += 2;
                } else {
                    offset += 1;
                }
            }
            if depth != 0 {
                return Err(format!("unterminated block comment at byte {start}"));
            }
            continue;
        }
        if let Some(raw) = raw_string_at(source, offset) {
            let (end, value) = raw?;
            tokens.push(RustToken {
                kind: RustTokenKind::String(value),
                offset,
            });
            offset = end;
            continue;
        }
        let quote = if bytes[offset] == b'"' {
            Some(offset)
        } else if matches!(bytes[offset], b'b' | b'c') && bytes.get(offset + 1) == Some(&b'"') {
            Some(offset + 1)
        } else {
            None
        };
        if let Some(quote) = quote {
            let (end, value) = quoted_string_at(source, quote)?;
            tokens.push(RustToken {
                kind: RustTokenKind::String(value),
                offset,
            });
            offset = end;
            continue;
        }
        let char_quote = if bytes[offset] == b'\'' {
            Some(offset)
        } else if bytes[offset] == b'b' && bytes.get(offset + 1) == Some(&b'\'') {
            Some(offset + 1)
        } else {
            None
        };
        if let Some(quote) = char_quote
            && let Some(end) = char_literal_end(source, quote)
        {
            tokens.push(RustToken {
                kind: RustTokenKind::Other,
                offset,
            });
            offset = end?;
            continue;
        }
        if bytes[offset..].starts_with(b"r#")
            && source[offset + 2..]
                .chars()
                .next()
                .is_some_and(|character| character == '_' || character.is_alphabetic())
        {
            let mut end = offset + 2;
            for character in source[end..].chars() {
                if character == '_' || character.is_alphanumeric() {
                    end += character.len_utf8();
                } else {
                    break;
                }
            }
            tokens.push(RustToken {
                kind: RustTokenKind::Ident {
                    text: source[offset + 2..end].to_string(),
                    raw: true,
                },
                offset,
            });
            offset = end;
            continue;
        }
        let character = source[offset..]
            .chars()
            .next()
            .ok_or_else(|| format!("invalid UTF-8 boundary at byte {offset}"))?;
        if character == '_' || character.is_alphabetic() {
            let mut end = offset + character.len_utf8();
            for character in source[end..].chars() {
                if character == '_' || character.is_alphanumeric() {
                    end += character.len_utf8();
                } else {
                    break;
                }
            }
            tokens.push(RustToken {
                kind: RustTokenKind::Ident {
                    text: source[offset..end].to_string(),
                    raw: false,
                },
                offset,
            });
            offset = end;
            continue;
        }
        tokens.push(RustToken {
            kind: if character.is_ascii_punctuation() {
                RustTokenKind::Punct(character)
            } else {
                RustTokenKind::Other
            },
            offset,
        });
        offset += character.len_utf8();
    }
    Ok(tokens)
}

type DelimiterStructure = (Vec<Option<usize>>, Vec<Option<usize>>);

fn delimiter_structure(tokens: &[RustToken]) -> Result<DelimiterStructure, String> {
    let mut pairs = vec![None; tokens.len()];
    let mut parents = vec![None; tokens.len()];
    let mut stack = Vec::<(char, usize)>::new();
    for (index, token) in tokens.iter().enumerate() {
        match token.kind {
            RustTokenKind::Punct(open @ ('(' | '[' | '{')) => {
                parents[index] = stack.last().map(|(_, parent)| *parent);
                stack.push((open, index));
            }
            RustTokenKind::Punct(close @ (')' | ']' | '}')) => {
                let Some((open, opening)) = stack.pop() else {
                    return Err(format!("unmatched `{close}` at byte {}", token.offset));
                };
                let expected = match open {
                    '(' => ')',
                    '[' => ']',
                    '{' => '}',
                    _ => unreachable!(),
                };
                if close != expected {
                    return Err(format!(
                        "mismatched `{open}` and `{close}` at byte {}",
                        token.offset
                    ));
                }
                pairs[opening] = Some(index);
                pairs[index] = Some(opening);
                parents[index] = parents[opening];
            }
            _ => parents[index] = stack.last().map(|(_, parent)| *parent),
        }
    }
    if let Some((open, opening)) = stack.pop() {
        return Err(format!(
            "unclosed `{open}` at byte {}",
            tokens[opening].offset
        ));
    }
    Ok((pairs, parents))
}

#[derive(Clone, Debug)]
enum CfgFormula {
    True,
    False,
    Atom(String),
    Not(Box<CfgFormula>),
    All(Vec<CfgFormula>),
    Any(Vec<CfgFormula>),
}

impl CfgFormula {
    fn all(formulas: impl IntoIterator<Item = CfgFormula>) -> Self {
        let mut combined = Vec::new();
        for formula in formulas {
            match formula {
                Self::True => {}
                Self::False => return Self::False,
                Self::All(nested) => combined.extend(nested),
                other => combined.push(other),
            }
        }
        match combined.len() {
            0 => Self::True,
            1 => combined.pop().unwrap_or(Self::True),
            _ => Self::All(combined),
        }
    }

    fn any(formulas: impl IntoIterator<Item = CfgFormula>) -> Self {
        let mut combined = Vec::new();
        for formula in formulas {
            match formula {
                Self::False => {}
                Self::True => return Self::True,
                Self::Any(nested) => combined.extend(nested),
                other => combined.push(other),
            }
        }
        match combined.len() {
            0 => Self::False,
            1 => combined.pop().unwrap_or(Self::False),
            _ => Self::Any(combined),
        }
    }

    fn not(formula: CfgFormula) -> Self {
        match formula {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Not(inner) => *inner,
            other => Self::Not(Box::new(other)),
        }
    }

    fn collect_atoms(&self, atoms: &mut BTreeSet<String>) {
        match self {
            Self::Atom(atom) => {
                atoms.insert(atom.clone());
            }
            Self::Not(formula) => formula.collect_atoms(atoms),
            Self::All(formulas) | Self::Any(formulas) => {
                for formula in formulas {
                    formula.collect_atoms(atoms);
                }
            }
            Self::True | Self::False => {}
        }
    }

    fn evaluate(&self, values: &std::collections::BTreeMap<String, bool>) -> bool {
        match self {
            Self::True => true,
            Self::False => false,
            Self::Atom(atom) => values.get(atom).copied().unwrap_or(false),
            Self::Not(formula) => !formula.evaluate(values),
            Self::All(formulas) => formulas.iter().all(|formula| formula.evaluate(values)),
            Self::Any(formulas) => formulas.iter().any(|formula| formula.evaluate(values)),
        }
    }

    fn satisfiable(&self) -> Result<bool, String> {
        let mut atoms = BTreeSet::new();
        self.collect_atoms(&mut atoms);
        if atoms.len() > 16 {
            return Err(format!(
                "cfg expression has {} independent atoms; refusing an unbounded classification",
                atoms.len()
            ));
        }
        let atoms = atoms.into_iter().collect::<Vec<_>>();
        for assignment in 0usize..(1usize << atoms.len()) {
            let values = atoms
                .iter()
                .enumerate()
                .map(|(index, atom)| (atom.clone(), assignment & (1 << index) != 0))
                .collect::<std::collections::BTreeMap<_, _>>();
            if self.evaluate(&values) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn implies(&self, other: &CfgFormula) -> Result<bool, String> {
        Ok(!Self::all([self.clone(), Self::not(other.clone())]).satisfiable()?)
    }
}

#[derive(Clone, Debug)]
struct AttributeSpec {
    content: Range<usize>,
}

#[derive(Clone, Debug)]
struct FunctionSpec {
    name: String,
    body: Range<usize>,
    attributes: Vec<AttributeSpec>,
}

#[derive(Clone, Debug)]
struct ScopeSpec {
    body: Range<usize>,
    attributes: Vec<AttributeSpec>,
}

struct StructuredSource {
    context: String,
    tokens: Vec<RustToken>,
    pairs: Vec<Option<usize>>,
    parents: Vec<Option<usize>>,
    macro_ranges: Vec<Range<usize>>,
    functions: Vec<FunctionSpec>,
    scopes: Vec<ScopeSpec>,
    file_attributes: Vec<AttributeSpec>,
    test_cfg: bool,
    external_cfg: CfgFormula,
}

impl CallSelector {
    fn label(self) -> String {
        match self {
            Self::Path(path) => format!("{}()", path.join("::")),
            Self::Method(method) => format!(".{method}()"),
        }
    }

    fn final_name(self) -> &'static str {
        match self {
            Self::Path(path) => path.last().copied().unwrap_or(""),
            Self::Method(method) => method,
        }
    }
}

impl StructuredSource {
    fn parse(context: &str, source: &str, test_cfg: bool) -> Result<Self, String> {
        let tokens = rust_tokens(source).map_err(|reason| format!("{context}: {reason}"))?;
        let (pairs, parents) =
            delimiter_structure(&tokens).map_err(|reason| format!("{context}: {reason}"))?;
        let mut parsed = Self {
            context: context.to_string(),
            tokens,
            pairs,
            parents,
            macro_ranges: Vec::new(),
            functions: Vec::new(),
            scopes: Vec::new(),
            file_attributes: Vec::new(),
            test_cfg,
            external_cfg: CfgFormula::True,
        };
        parsed.macro_ranges = parsed.find_macro_ranges();
        parsed.functions = parsed.find_functions()?;
        parsed.scopes = parsed.find_scopes()?;
        parsed.file_attributes = parsed.inner_attributes_at(0).0;
        Ok(parsed)
    }

    fn find_macro_ranges(&self) -> Vec<Range<usize>> {
        let mut ranges = Vec::new();
        for index in 0..self.tokens.len() {
            if !self.tokens[index].is_punct('!') {
                continue;
            }
            let mut opening = index + 1;
            if self
                .tokens
                .get(opening)
                .and_then(RustToken::ident)
                .is_some()
                && self
                    .tokens
                    .get(index.wrapping_sub(1))
                    .and_then(RustToken::ident)
                    == Some("macro_rules")
            {
                opening += 1;
            }
            if self.tokens.get(opening).is_some_and(|token| {
                token.is_punct('(') || token.is_punct('[') || token.is_punct('{')
            }) && let Some(closing) = self.pairs.get(opening).and_then(|pair| *pair)
            {
                ranges.push(opening..closing + 1);
            }
        }
        ranges
    }

    fn in_macro(&self, index: usize) -> bool {
        self.macro_ranges.iter().any(|range| range.contains(&index))
    }

    fn attribute_at(&self, start: usize, inner: bool) -> Option<(AttributeSpec, usize)> {
        if !self.tokens.get(start)?.is_punct('#') {
            return None;
        }
        let opening = if inner {
            if !self.tokens.get(start + 1)?.is_punct('!') {
                return None;
            }
            start + 2
        } else {
            start + 1
        };
        if !self.tokens.get(opening)?.is_punct('[') {
            return None;
        }
        let closing = self.pairs.get(opening).and_then(|pair| *pair)?;
        Some((
            AttributeSpec {
                content: opening + 1..closing,
            },
            closing + 1,
        ))
    }

    fn inner_attributes_at(&self, mut cursor: usize) -> (Vec<AttributeSpec>, usize) {
        let mut attributes = Vec::new();
        while let Some((attribute, next)) = self.attribute_at(cursor, true) {
            attributes.push(attribute);
            cursor = next;
        }
        (attributes, cursor)
    }

    fn outer_attributes_at(&self, mut cursor: usize) -> (Vec<AttributeSpec>, usize) {
        let mut attributes = Vec::new();
        while let Some((attribute, next)) = self.attribute_at(cursor, false) {
            attributes.push(attribute);
            cursor = next;
        }
        (attributes, cursor)
    }

    fn outer_attributes_before(&self, mut cursor: usize) -> (Vec<AttributeSpec>, usize) {
        let mut attributes = Vec::new();
        while cursor > 0 && self.tokens[cursor - 1].is_punct(']') {
            let closing = cursor - 1;
            let Some(opening) = self.pairs[closing] else {
                break;
            };
            let Some(hash) = opening.checked_sub(1) else {
                break;
            };
            if !self.tokens[hash].is_punct('#') || hash > 0 && self.tokens[hash - 1].is_punct('!') {
                break;
            }
            attributes.push(AttributeSpec {
                content: opening + 1..closing,
            });
            cursor = hash;
        }
        attributes.reverse();
        (attributes, cursor)
    }

    fn item_attributes_before(&self, item: usize) -> Vec<AttributeSpec> {
        let mut cursor = item;
        loop {
            let Some(previous) = cursor.checked_sub(1) else {
                break;
            };
            if self.tokens[previous].ident().is_some_and(|ident| {
                matches!(
                    ident,
                    "async" | "const" | "default" | "extern" | "pub" | "unsafe"
                )
            }) {
                cursor = previous;
                continue;
            }
            if matches!(self.tokens[previous].kind, RustTokenKind::String(_))
                && previous > 0
                && self.tokens[previous - 1].ident() == Some("extern")
            {
                cursor = previous - 1;
                continue;
            }
            if self.tokens[previous].is_punct(')')
                && let Some(opening) = self.pairs[previous]
                && opening > 0
                && self.tokens[opening - 1].ident() == Some("pub")
            {
                cursor = opening - 1;
                continue;
            }
            break;
        }
        self.outer_attributes_before(cursor).0
    }

    fn find_body_opening(&self, item: usize) -> Option<usize> {
        let parent = self.parents[item];
        for index in item + 1..self.tokens.len() {
            if self.parents[index] != parent {
                continue;
            }
            if self.tokens[index].is_punct(';') {
                return None;
            }
            if self.tokens[index].is_punct('{') {
                return Some(index);
            }
        }
        None
    }

    fn find_functions(&self) -> Result<Vec<FunctionSpec>, String> {
        let mut functions = Vec::new();
        for index in 0..self.tokens.len() {
            if self.tokens[index].ident() != Some("fn") || self.in_macro(index) {
                continue;
            }
            let Some(name) = self.tokens.get(index + 1).and_then(RustToken::ident) else {
                continue;
            };
            let Some(opening) = self.find_body_opening(index) else {
                continue;
            };
            let closing = self.pairs[opening].ok_or_else(|| {
                format!(
                    "{}: function `{name}` has no brace-balanced body",
                    self.context
                )
            })?;
            functions.push(FunctionSpec {
                name: name.to_string(),
                body: opening + 1..closing,
                attributes: self.item_attributes_before(index),
            });
        }
        Ok(functions)
    }

    fn find_scopes(&self) -> Result<Vec<ScopeSpec>, String> {
        let mut scopes = self
            .functions
            .iter()
            .map(|function| ScopeSpec {
                body: function.body.clone(),
                attributes: function.attributes.clone(),
            })
            .collect::<Vec<_>>();
        for index in 0..self.tokens.len() {
            if self.tokens[index].ident() == Some("mod") && !self.in_macro(index) {
                let Some(opening) = self.find_body_opening(index) else {
                    continue;
                };
                let closing = self.pairs[opening].ok_or_else(|| {
                    format!("{}: inline module has no closing brace", self.context)
                })?;
                scopes.push(ScopeSpec {
                    body: opening + 1..closing,
                    attributes: self.item_attributes_before(index),
                });
            }
        }
        for opening in 0..self.tokens.len() {
            if !self.tokens[opening].is_punct('{') {
                continue;
            }
            let Some(closing) = self.pairs[opening] else {
                continue;
            };
            let enclosing = self.construct_attributes(opening);
            if !enclosing.is_empty() {
                scopes.push(ScopeSpec {
                    body: opening + 1..closing,
                    attributes: enclosing,
                });
            }
            let inner = self.inner_attributes_at(opening + 1).0;
            if !inner.is_empty() {
                scopes.push(ScopeSpec {
                    body: opening + 1..closing,
                    attributes: inner,
                });
            }
        }
        Ok(scopes)
    }

    fn construct_attributes(&self, opening: usize) -> Vec<AttributeSpec> {
        let parent = self.parents[opening];
        let initial = parent.map_or(0, |parent| parent + 1);
        let start = (initial..opening)
            .rev()
            .find(|&index| {
                self.parents[index] == parent
                    && (self.tokens[index].is_punct(';') || self.tokens[index].is_punct('}'))
            })
            .map_or(initial, |index| index + 1);
        self.outer_attributes_at(start).0
    }

    fn split_arguments(&self, range: Range<usize>) -> Vec<Range<usize>> {
        let mut arguments = Vec::new();
        let mut start = range.start;
        let mut stack = Vec::<char>::new();
        for index in range.clone() {
            match self.tokens[index].kind {
                RustTokenKind::Punct(open @ ('(' | '[' | '{')) => stack.push(open),
                RustTokenKind::Punct(')' | ']' | '}') => {
                    stack.pop();
                }
                RustTokenKind::Punct(',') if stack.is_empty() => {
                    arguments.push(start..index);
                    start = index + 1;
                }
                _ => {}
            }
        }
        arguments.push(start..range.end);
        arguments
    }

    fn normalized_tokens(&self, range: Range<usize>) -> String {
        let mut normalized = String::new();
        for token in &self.tokens[range] {
            match &token.kind {
                RustTokenKind::Ident { text, raw } => {
                    if *raw {
                        normalized.push_str("r#");
                    }
                    normalized.push_str(text);
                }
                RustTokenKind::String(value) => {
                    normalized.push('"');
                    normalized.push_str(value);
                    normalized.push('"');
                }
                RustTokenKind::Punct(punct) => normalized.push(*punct),
                RustTokenKind::Other => normalized.push('?'),
            }
        }
        normalized
    }

    fn cfg_predicate(&self, range: Range<usize>) -> Result<CfgFormula, String> {
        if range.is_empty() {
            return Err(format!("{}: empty cfg predicate", self.context));
        }
        if let Some(name) = self.tokens[range.start].ident()
            && range.start + 1 < range.end
            && self.tokens[range.start + 1].is_punct('(')
            && self.pairs[range.start + 1] == Some(range.end - 1)
        {
            let arguments = self.split_arguments(range.start + 2..range.end - 1);
            return match name {
                "all" => Ok(CfgFormula::all(
                    arguments
                        .into_iter()
                        .filter(|argument| !argument.is_empty())
                        .map(|argument| self.cfg_predicate(argument))
                        .collect::<Result<Vec<_>, _>>()?,
                )),
                "any" => Ok(CfgFormula::any(
                    arguments
                        .into_iter()
                        .filter(|argument| !argument.is_empty())
                        .map(|argument| self.cfg_predicate(argument))
                        .collect::<Result<Vec<_>, _>>()?,
                )),
                "not" if arguments.len() == 1 => {
                    Ok(CfgFormula::not(self.cfg_predicate(arguments[0].clone())?))
                }
                "not" => Err(format!(
                    "{}: cfg(not(...)) needs one predicate",
                    self.context
                )),
                _ => Ok(CfgFormula::Atom(self.normalized_tokens(range))),
            };
        }
        if range.len() == 1 {
            let Some(name) = self.tokens[range.start].ident() else {
                return Err(format!(
                    "{}: unsupported cfg predicate `{}`",
                    self.context,
                    self.normalized_tokens(range)
                ));
            };
            return Ok(match name {
                "test" => {
                    if self.test_cfg {
                        CfgFormula::True
                    } else {
                        CfgFormula::False
                    }
                }
                "unix" => CfgFormula::True,
                "windows" => CfgFormula::False,
                _ => CfgFormula::Atom(name.to_string()),
            });
        }
        if range.len() == 3 && self.tokens[range.start + 1].is_punct('=') {
            let Some(key) = self.tokens[range.start].ident() else {
                return Err(format!(
                    "{}: unsupported cfg key `{}`",
                    self.context,
                    self.normalized_tokens(range)
                ));
            };
            let RustTokenKind::String(value) = &self.tokens[range.start + 2].kind else {
                return Err(format!(
                    "{}: cfg value for `{key}` is not a string",
                    self.context
                ));
            };
            return Ok(match (key, value.as_str()) {
                ("target_os", "macos") | ("target_family", "unix") => CfgFormula::True,
                ("target_os", _) | ("target_family", "windows") => CfgFormula::False,
                ("feature", "metal-gpu") => CfgFormula::True,
                _ => CfgFormula::Atom(format!("{key}={value}")),
            });
        }
        Err(format!(
            "{}: unsupported cfg predicate `{}`",
            self.context,
            self.normalized_tokens(range)
        ))
    }

    fn meta_cfg_effect(&self, range: Range<usize>) -> Result<CfgFormula, String> {
        let Some(name) = self.tokens.get(range.start).and_then(RustToken::ident) else {
            return Ok(CfgFormula::True);
        };
        if !matches!(name, "cfg" | "cfg_attr") {
            return Ok(CfgFormula::True);
        }
        let opening = range.start + 1;
        if !self
            .tokens
            .get(opening)
            .is_some_and(|token| token.is_punct('('))
            || self.pairs.get(opening).and_then(|pair| *pair) != Some(range.end - 1)
        {
            return Err(format!("{}: malformed `{name}` attribute", self.context));
        }
        let arguments = self.split_arguments(opening + 1..range.end - 1);
        if name == "cfg" {
            if arguments.len() != 1 {
                return Err(format!(
                    "{}: cfg attribute needs one predicate",
                    self.context
                ));
            }
            return self.cfg_predicate(arguments[0].clone());
        }
        if arguments.len() < 2 {
            return Err(format!(
                "{}: cfg_attr attribute needs a predicate and emitted attribute",
                self.context
            ));
        }
        let predicate = self.cfg_predicate(arguments[0].clone())?;
        let mut effects = Vec::new();
        for emitted in arguments.into_iter().skip(1) {
            let emitted_effect = self.meta_cfg_effect(emitted)?;
            effects.push(CfgFormula::any([
                CfgFormula::not(predicate.clone()),
                emitted_effect,
            ]));
        }
        Ok(CfgFormula::all(effects))
    }

    fn attributes_formula(&self, attributes: &[AttributeSpec]) -> Result<CfgFormula, String> {
        Ok(CfgFormula::all(
            attributes
                .iter()
                .map(|attribute| self.meta_cfg_effect(attribute.content.clone()))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }

    fn meta_registers_test(&self, range: Range<usize>) -> bool {
        if range.len() == 1 {
            return self.tokens[range.start].ident() == Some("test");
        }
        if self.tokens[range.start].ident() != Some("cfg_attr") {
            return false;
        }
        let opening = range.start + 1;
        if !self
            .tokens
            .get(opening)
            .is_some_and(|token| token.is_punct('('))
            || self.pairs.get(opening).and_then(|pair| *pair) != Some(range.end - 1)
        {
            return false;
        }
        self.split_arguments(opening + 1..range.end - 1)
            .into_iter()
            .skip(1)
            .any(|emitted| self.meta_registers_test(emitted))
    }

    fn attributes_register_test(&self, attributes: &[AttributeSpec]) -> bool {
        attributes
            .iter()
            .any(|attribute| self.meta_registers_test(attribute.content.clone()))
    }

    fn nearest_brace(&self, index: usize) -> Option<usize> {
        let mut parent = self.parents.get(index).and_then(|parent| *parent);
        while let Some(opening) = parent {
            if self.tokens[opening].is_punct('{') {
                return Some(opening);
            }
            parent = self.parents[opening];
        }
        None
    }

    fn statement_attributes(&self, site: usize) -> Vec<AttributeSpec> {
        let Some(block) = self.nearest_brace(site) else {
            return Vec::new();
        };
        let mut start = block + 1;
        for index in block + 1..site {
            let same_block =
                self.nearest_brace(index) == Some(block) || self.parents[index] == Some(block);
            if same_block && (self.tokens[index].is_punct(';') || self.tokens[index].is_punct('}'))
            {
                start = index + 1;
            }
        }
        self.outer_attributes_at(start).0
    }

    fn formula_at(&self, site: usize) -> Result<CfgFormula, String> {
        let mut formulas = vec![self.external_cfg.clone()];
        formulas.push(self.attributes_formula(&self.file_attributes)?);
        for scope in &self.scopes {
            if scope.body.contains(&site) {
                formulas.push(self.attributes_formula(&scope.attributes)?);
            }
        }
        formulas.push(self.attributes_formula(&self.statement_attributes(site))?);
        Ok(CfgFormula::all(formulas))
    }

    fn path_start(&self, final_index: usize, path: &[&str]) -> Option<usize> {
        if self.tokens.get(final_index)?.ident() != path.last().copied() {
            return None;
        }
        let mut cursor = final_index;
        for segment in path[..path.len().saturating_sub(1)].iter().rev() {
            if cursor < 3
                || !self.tokens[cursor - 1].is_punct(':')
                || !self.tokens[cursor - 2].is_punct(':')
                || self.tokens[cursor - 3].ident() != Some(*segment)
            {
                return None;
            }
            cursor -= 3;
        }
        Some(cursor)
    }

    fn selector_candidate(&self, index: usize, selector: CallSelector) -> Option<usize> {
        match selector {
            CallSelector::Path(path) => {
                let start = self.path_start(index, path)?;
                if path.len() == 1
                    && start > 0
                    && (self.tokens[start - 1].is_punct('.')
                        || self.tokens[start - 1].is_punct(':'))
                {
                    return None;
                }
                Some(start)
            }
            CallSelector::Method(method) => {
                if self.tokens[index].ident() == Some(method)
                    && index > 0
                    && self.tokens[index - 1].is_punct('.')
                {
                    Some(index)
                } else {
                    None
                }
            }
        }
    }

    fn call_sites(
        &self,
        range: Range<usize>,
        selector: CallSelector,
        fail_closed: bool,
    ) -> Result<Vec<usize>, String> {
        let mut calls = Vec::new();
        for index in range {
            if self.tokens[index].ident() != Some(selector.final_name()) {
                continue;
            }
            let Some(start) = self.selector_candidate(index, selector) else {
                continue;
            };
            if matches!(selector, CallSelector::Path(path) if path.len() == 1)
                && start > 0
                && self.tokens[start - 1].ident() == Some("fn")
            {
                continue;
            }
            if self.in_macro(index) {
                if fail_closed {
                    return Err(format!(
                        "{}: unclassifiable protected-work syntax containing `{}`",
                        self.context,
                        selector.label()
                    ));
                }
                continue;
            }
            if self
                .tokens
                .get(index + 1)
                .is_some_and(|token| token.is_punct('('))
            {
                calls.push(index);
            } else if fail_closed {
                return Err(format!(
                    "{}: unclassifiable protected-work syntax near `{}`",
                    self.context,
                    selector.label()
                ));
            }
        }
        Ok(calls)
    }

    fn has_selector(&self, selector: CallSelector) -> Result<bool, String> {
        self.reject_selector_in_macros(selector)?;
        Ok(!self
            .call_sites(0..self.tokens.len(), selector, false)?
            .is_empty())
    }

    fn reject_selector_in_macros(&self, selector: CallSelector) -> Result<(), String> {
        for range in &self.macro_ranges {
            for index in range.clone() {
                if self.tokens[index].ident() == Some(selector.final_name())
                    && self.selector_candidate(index, selector).is_some()
                {
                    return Err(format!(
                        "{}: unclassifiable protected-work syntax containing `{}`",
                        self.context,
                        selector.label()
                    ));
                }
            }
        }
        Ok(())
    }

    fn has_code_identifier(&self, identifier: &str) -> bool {
        self.tokens
            .iter()
            .enumerate()
            .any(|(index, token)| token.ident() == Some(identifier) && !self.in_macro(index))
    }

    fn test_functions(&self) -> Vec<&FunctionSpec> {
        self.functions
            .iter()
            .filter(|function| self.attributes_register_test(&function.attributes))
            .collect()
    }

    fn path_attribute(&self, attributes: &[AttributeSpec]) -> Option<String> {
        attributes.iter().find_map(|attribute| {
            let range = attribute.content.clone();
            if range.len() != 3
                || self.tokens[range.start].ident() != Some("path")
                || !self.tokens[range.start + 1].is_punct('=')
            {
                return None;
            }
            match &self.tokens[range.start + 2].kind {
                RustTokenKind::String(path) => Some(path.clone()),
                _ => None,
            }
        })
    }

    fn external_modules(&self, parent_path: &Path) -> Result<Vec<(PathBuf, CfgFormula)>, String> {
        let mut modules = Vec::new();
        for index in 0..self.tokens.len() {
            if self.tokens[index].ident() != Some("mod") || self.in_macro(index) {
                continue;
            }
            let Some(name) = self.tokens.get(index + 1).and_then(RustToken::ident) else {
                continue;
            };
            let parent = self.parents[index];
            let Some(semicolon) = (index + 2..self.tokens.len()).find(|cursor| {
                self.parents[*cursor] == parent
                    && (self.tokens[*cursor].is_punct(';') || self.tokens[*cursor].is_punct('{'))
            }) else {
                continue;
            };
            if !self.tokens[semicolon].is_punct(';') {
                continue;
            }
            let attributes = self.item_attributes_before(index);
            let target = if let Some(path) = self.path_attribute(&attributes) {
                parent_path
                    .parent()
                    .unwrap_or_else(|| Path::new(""))
                    .join(path)
            } else {
                let filename = parent_path.file_name().and_then(|name| name.to_str());
                let base = if matches!(filename, Some("lib.rs" | "main.rs" | "mod.rs")) {
                    parent_path
                        .parent()
                        .unwrap_or_else(|| Path::new(""))
                        .to_path_buf()
                } else {
                    parent_path.with_extension("")
                };
                let flat = base.join(format!("{name}.rs"));
                if flat.exists() {
                    flat
                } else {
                    base.join(name).join("mod.rs")
                }
            };
            modules.push((
                target,
                CfgFormula::all([
                    self.formula_at(index)?,
                    self.attributes_formula(&attributes)?,
                ]),
            ));
        }
        Ok(modules)
    }

    fn parse_path(
        manifest_dir: &Path,
        path: &Path,
        source: &str,
        test_cfg: bool,
    ) -> Result<Self, String> {
        let relative = path
            .strip_prefix(manifest_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        let mut parsed = Self::parse(&relative, source, test_cfg)?;
        parsed.external_cfg =
            external_module_formula(manifest_dir, path, test_cfg, &mut BTreeSet::new())?;
        Ok(parsed)
    }
}

fn module_parent_candidates(manifest_dir: &Path, path: &Path) -> Vec<PathBuf> {
    let src = manifest_dir.join("src");
    let Ok(relative) = path.strip_prefix(&src) else {
        return Vec::new();
    };
    let components = relative.components().collect::<Vec<_>>();
    if components.len() == 2 && components[0].as_os_str() == "bin" {
        return Vec::new();
    }
    let Some(filename) = path.file_name().and_then(|name| name.to_str()) else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    if filename == "mod.rs" {
        let Some(module_dir) = path.parent() else {
            return candidates;
        };
        let Some(parent_dir) = module_dir.parent() else {
            return candidates;
        };
        if parent_dir == src {
            candidates.extend([src.join("lib.rs"), src.join("main.rs")]);
        } else {
            candidates.extend([parent_dir.join("mod.rs"), module_dir.with_extension("rs")]);
        }
    } else if let Some(directory) = path.parent() {
        if directory == src {
            candidates.extend([src.join("lib.rs"), src.join("main.rs")]);
        } else {
            candidates.extend([directory.join("mod.rs"), directory.with_extension("rs")]);
        }
    }
    candidates.retain(|candidate| candidate.exists() && candidate != path);
    candidates.sort();
    candidates.dedup();
    candidates
}

fn external_module_formula(
    manifest_dir: &Path,
    path: &Path,
    test_cfg: bool,
    visiting: &mut BTreeSet<PathBuf>,
) -> Result<CfgFormula, String> {
    let canonical = std::fs::canonicalize(path)
        .map_err(|error| format!("could not resolve module path {}: {error}", path.display()))?;
    if !visiting.insert(canonical.clone()) {
        return Err(format!(
            "module context cycle while classifying {}",
            path.display()
        ));
    }
    let mut contexts = Vec::new();
    for parent in module_parent_candidates(manifest_dir, path) {
        let parent_source = std::fs::read_to_string(&parent).map_err(|error| {
            format!("could not read module parent {}: {error}", parent.display())
        })?;
        let parent_relative = parent
            .strip_prefix(manifest_dir)
            .unwrap_or(&parent)
            .to_string_lossy();
        let parent_parsed = StructuredSource::parse(&parent_relative, &parent_source, test_cfg)?;
        for (target, declaration) in parent_parsed.external_modules(&parent)? {
            let Ok(target) = std::fs::canonicalize(target) else {
                continue;
            };
            if target != canonical {
                continue;
            }
            let parent_context =
                external_module_formula(manifest_dir, &parent, test_cfg, visiting)?;
            contexts.push(CfgFormula::all([parent_context, declaration]));
        }
    }
    visiting.remove(&canonical);
    if contexts.is_empty() {
        Ok(CfgFormula::True)
    } else {
        Ok(CfgFormula::any(contexts))
    }
}

#[derive(Clone)]
struct StructuralGuard {
    protected: Range<usize>,
    binding: String,
    formula: CfgFormula,
    used_after_binding: bool,
}

#[derive(Clone, Copy)]
enum GuardRequirement {
    Lexical,
    Function { closing_brace: usize },
}

impl StructuredSource {
    fn statement_bounds(&self, site: usize) -> Option<(usize, usize, usize)> {
        let block = self.nearest_brace(site)?;
        let mut start = block + 1;
        for index in block + 1..site {
            let same_block =
                self.nearest_brace(index) == Some(block) || self.parents[index] == Some(block);
            if same_block && (self.tokens[index].is_punct(';') || self.tokens[index].is_punct('}'))
            {
                start = index + 1;
            }
        }
        let closing = self.pairs[block]?;
        let end = (site + 1..closing).find(|index| {
            self.tokens[*index].is_punct(';')
                && (self.nearest_brace(*index) == Some(block)
                    || self.parents[*index] == Some(block))
        })?;
        Some((start, end, block))
    }

    fn direct_guard_binding(
        &self,
        call: usize,
        selector: CallSelector,
    ) -> Result<Option<StructuralGuard>, String> {
        let Some((statement_start, statement_end, block)) = self.statement_bounds(call) else {
            return Ok(None);
        };
        let (_, content_start) = self.outer_attributes_at(statement_start);
        if self.tokens.get(content_start).and_then(RustToken::ident) != Some("let") {
            return Ok(None);
        }
        let Some(binding_token) = self.tokens.get(content_start + 1) else {
            return Ok(None);
        };
        let Some(binding) = binding_token.ident() else {
            return Ok(None);
        };
        if binding == "_" || binding_token.is_raw_ident() {
            return Ok(None);
        }
        if !self
            .tokens
            .get(content_start + 2)
            .is_some_and(|token| token.is_punct('='))
        {
            return Ok(None);
        }
        let call_start = match selector {
            CallSelector::Path(path) => self.path_start(call, path),
            CallSelector::Method(_) => Some(call),
        };
        if call_start != Some(content_start + 3) {
            return Ok(None);
        }
        let opening = call + 1;
        if !self
            .tokens
            .get(opening)
            .is_some_and(|token| token.is_punct('('))
        {
            return Ok(None);
        }
        let Some(closing) = self.pairs[opening] else {
            return Ok(None);
        };
        if closing != opening + 1 || statement_end != closing + 1 {
            return Ok(None);
        }
        let scope_end = self.pairs[block].ok_or_else(|| {
            format!(
                "{}: guard binding has no brace-balanced scope",
                self.context
            )
        })?;
        let used_after_binding = self.tokens[statement_end + 1..scope_end]
            .iter()
            .any(|token| token.ident() == Some(binding));
        Ok(Some(StructuralGuard {
            protected: statement_end + 1..scope_end,
            binding: binding.to_string(),
            formula: self.formula_at(call)?,
            used_after_binding,
        }))
    }

    fn guard_bindings(
        &self,
        range: Range<usize>,
        selector: CallSelector,
    ) -> Result<Vec<StructuralGuard>, String> {
        self.call_sites(range, selector, true)?
            .into_iter()
            .filter_map(|call| self.direct_guard_binding(call, selector).transpose())
            .collect()
    }

    fn validate_work_sites(
        &self,
        work_sites: &[usize],
        lock_selector: CallSelector,
        requirement: GuardRequirement,
    ) -> Result<(), String> {
        if work_sites.is_empty() {
            return Err(format!("{}: protected work was not found", self.context));
        }
        let guards = self.guard_bindings(0..self.tokens.len(), lock_selector)?;
        let mut live_work = 0usize;
        for work in work_sites {
            let work_formula = self.formula_at(*work)?;
            if !work_formula.satisfiable()? {
                continue;
            }
            live_work += 1;
            let candidates = guards.iter().filter(|guard| {
                guard.protected.contains(work)
                    && !guard.used_after_binding
                    && match requirement {
                        GuardRequirement::Lexical => true,
                        GuardRequirement::Function { closing_brace } => {
                            guard.protected.end == closing_brace
                        }
                    }
            });
            let guard_formula = CfgFormula::any(candidates.map(|guard| guard.formula.clone()));
            if !work_formula.implies(&guard_formula)? {
                let moved = guards
                    .iter()
                    .find(|guard| guard.protected.contains(work) && guard.used_after_binding);
                let reason = moved.map_or_else(
                    || "no live shared-lock binding encloses this work".to_string(),
                    |guard| {
                        format!(
                            "shared-lock binding `{}` is used before its scope ends and may be moved, dropped, or shadowed",
                            guard.binding
                        )
                    },
                );
                return Err(format!(
                    "{}: {} at token byte {}",
                    self.context, reason, self.tokens[*work].offset
                ));
            }
        }
        if live_work == 0 {
            return Err(format!(
                "{}: protected work has no live occurrence in the Metal configuration",
                self.context
            ));
        }
        Ok(())
    }

    fn validate_selector(
        &self,
        range: Range<usize>,
        selector: CallSelector,
        lock_selector: CallSelector,
        requirement: GuardRequirement,
    ) -> Result<(), String> {
        let work_sites = self.call_sites(range, selector, true)?;
        self.validate_work_sites(&work_sites, lock_selector, requirement)
    }

    fn construction_sites(&self, range: Range<usize>) -> Result<Vec<usize>, String> {
        let mut sites = Vec::new();
        for selector in CONSTRUCTION_SELECTORS {
            sites.extend(self.call_sites(range.clone(), *selector, true)?);
        }
        sites.sort_unstable();
        Ok(sites)
    }

    fn has_top_level_metal_marker(&self) -> Result<bool, String> {
        for selector in RAW_GPU_SELECTORS {
            if self.has_selector(*selector)? {
                return Ok(true);
            }
        }
        Ok(["MetalQwen35State", "MetalForwardPass", "QwenModel"]
            .iter()
            .any(|marker| self.has_code_identifier(marker)))
    }
}

#[test]
fn raw_metal_measurement_harnesses_use_live_lock_bindings_across_the_entrypoint() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut actual = BTreeSet::new();

    for relative_dir in ["benches", "examples", "src/bin"] {
        for path in rust_sources_under(&manifest_dir.join(relative_dir)) {
            let source = std::fs::read_to_string(&path).expect("read measurement source");
            let relative = path
                .strip_prefix(manifest_dir)
                .expect("source under manifest directory")
                .to_string_lossy()
                .into_owned();
            let parsed = StructuredSource::parse(&relative, &source, false)
                .unwrap_or_else(|reason| panic!("{reason}"));
            let mut has_raw_gpu_work = false;
            for selector in RAW_GPU_SELECTORS {
                has_raw_gpu_work |= parsed
                    .has_selector(*selector)
                    .unwrap_or_else(|reason| panic!("{reason}"));
            }
            if !has_raw_gpu_work {
                continue;
            }
            actual.insert(relative.clone());

            let protected_entrypoint = RAW_HARNESS_ENTRYPOINTS
                .iter()
                .find_map(|(path, marker)| (*path == relative).then_some(*marker))
                .unwrap_or_else(|| panic!("raw Metal harness {relative} is not classified"));
            parsed
                .validate_selector(
                    0..parsed.tokens.len(),
                    protected_entrypoint,
                    SHARED_LOCK_SELECTOR,
                    GuardRequirement::Lexical,
                )
                .unwrap_or_else(|reason| panic!("{reason}"));
        }
    }

    let expected = RAW_HARNESS_ENTRYPOINTS
        .iter()
        .map(|(path, _)| (*path).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual, expected,
        "raw Metal harness inventory changed; classify every added or removed path explicitly"
    );
}

#[test]
fn top_level_metal_entrypoint_inventory_is_explicit_and_fail_closed() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut discovered = BTreeSet::new();
    let mut all_targets = BTreeSet::new();
    for relative_dir in ["benches", "examples", "src/bin"] {
        for path in rust_sources_under(&manifest_dir.join(relative_dir)) {
            let source = std::fs::read_to_string(&path).expect("read measurement source");
            let relative = path
                .strip_prefix(manifest_dir)
                .expect("source under manifest directory")
                .to_string_lossy()
                .into_owned();
            all_targets.insert(relative.clone());
            let parsed = StructuredSource::parse(&relative, &source, false)
                .unwrap_or_else(|reason| panic!("{reason}"));
            if parsed
                .has_top_level_metal_marker()
                .unwrap_or_else(|reason| panic!("{reason}"))
            {
                discovered.insert(relative);
            }
        }
    }

    let reviewed = REVIEWED_TOP_LEVEL_METAL_TARGETS
        .iter()
        .map(|path| (*path).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        discovered, reviewed,
        "top-level Metal entrypoint inventory changed; classify direct, helper-mediated, and \
         alternate-state-family paths explicitly"
    );
    let reviewed_non_metal = REVIEWED_NON_METAL_TOP_LEVEL_TARGETS
        .iter()
        .map(|path| (*path).to_string())
        .collect::<BTreeSet<_>>();
    assert!(
        reviewed.is_disjoint(&reviewed_non_metal),
        "top-level targets must have exactly one classification"
    );
    let all_reviewed = reviewed
        .union(&reviewed_non_metal)
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        all_targets, all_reviewed,
        "top-level target inventory changed; every new bench, example, or binary must be \
         classified before it can pass the contract"
    );
}

#[test]
fn alternate_and_helper_mediated_entrypoints_hold_the_shared_lock() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (relative, entrypoint) in ADDITIONAL_GUARDED_ENTRYPOINTS {
        let source = std::fs::read_to_string(manifest_dir.join(relative))
            .expect("read guarded Metal entrypoint");
        let parsed = StructuredSource::parse(relative, &source, false)
            .unwrap_or_else(|reason| panic!("{reason}"));
        parsed
            .validate_selector(
                0..parsed.tokens.len(),
                *entrypoint,
                SHARED_LOCK_SELECTOR,
                GuardRequirement::Lexical,
            )
            .unwrap_or_else(|reason| panic!("{reason}"));
    }
}

#[test]
fn in_crate_raw_command_buffer_tests_are_explicit_and_guarded() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let exemptions = IN_CRATE_COMMAND_BUFFER_EXEMPTIONS
        .iter()
        .map(|(name, reason)| {
            assert!(
                !reason.trim().is_empty(),
                "{name} needs an exemption reason"
            );
            (*name, *reason)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut discovered = BTreeSet::new();
    let mut violations = Vec::new();
    for path in rust_sources_under(&manifest_dir.join("src")) {
        let source = std::fs::read_to_string(&path).expect("read in-crate Rust source");
        let relative = path
            .strip_prefix(manifest_dir)
            .expect("source under manifest directory")
            .to_string_lossy()
            .into_owned();
        let parsed = StructuredSource::parse_path(manifest_dir, &path, &source, true)
            .unwrap_or_else(|reason| panic!("{reason}"));
        parsed
            .reject_selector_in_macros(NEW_COMMAND_BUFFER)
            .unwrap_or_else(|reason| panic!("{reason}"));
        for function in parsed.test_functions() {
            let command_buffers = parsed
                .call_sites(function.body.clone(), NEW_COMMAND_BUFFER, true)
                .unwrap_or_else(|reason| panic!("{reason}"));
            if command_buffers.is_empty() {
                continue;
            }
            let site = format!("{relative}::{}", function.name);
            discovered.insert(site.clone());
            if exemptions.contains_key(site.as_str()) {
                continue;
            }
            if let Err(reason) = parsed.validate_work_sites(
                &command_buffers,
                LOCAL_LOCK_SELECTOR,
                GuardRequirement::Function {
                    closing_brace: function.body.end,
                },
            ) {
                violations.push(format!("{site}: {reason}"));
            }
        }
    }

    let reviewed = IN_CRATE_COMMAND_BUFFER_TESTS
        .iter()
        .map(|site| (*site).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        discovered, reviewed,
        "in-crate raw command-buffer inventory changed; classify every test explicitly"
    );
    assert!(
        violations.is_empty(),
        "in-crate raw command-buffer tests without a function-lifetime shared lock:\n{}",
        violations.join("\n")
    );
}

#[test]
fn cfg_attr_test_function_is_included_in_raw_dispatch_inventory() {
    let source = r#"
#[cfg_attr(test, test)]
fn cfg_attr_registered_test() {
    let command_buffer = queue.new_command_buffer();
}
"#;
    let parsed = StructuredSource::parse("fixtures/cfg_attr_test.rs", source, true)
        .expect("parse cfg_attr test fixture");
    let discovered = parsed
        .test_functions()
        .into_iter()
        .filter(|function| {
            !parsed
                .call_sites(function.body.clone(), NEW_COMMAND_BUFFER, true)
                .expect("classify cfg_attr test work")
                .is_empty()
        })
        .map(|function| function.name.as_str())
        .collect::<BTreeSet<_>>();

    assert_eq!(discovered, BTreeSet::from(["cfg_attr_registered_test"]));
}

#[test]
fn guard_lifetime_rejects_an_immediate_drop_before_metal_work() {
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    drop(gpu_guard);
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a guard dropped before Metal work");
}

#[test]
fn guard_lifetime_rejects_a_wildcard_that_drops_the_guard_immediately() {
    let source = r#"
fn measurement() {
    let _ = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a wildcard guard binding");
}

#[test]
fn guard_lifetime_rejects_a_cfg_elided_acquisition() {
    let source = r#"
fn measurement() {
    #[cfg(any())]
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a cfg-elided guard acquisition");
}

#[test]
fn guard_lifetime_rejects_a_stacked_cfg_elided_acquisition() {
    let source = r#"
fn measurement() {
    #[cfg(any())]
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a stacked cfg-elided guard acquisition");
}

#[test]
fn guard_lifetime_rejects_an_active_cfg_attr_elided_acquisition() {
    let source = r#"
fn measurement() {
    #[cfg_attr(all(), cfg(any()))]
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "an active cfg_attr-elided guard acquisition");
}

#[test]
fn guard_lifetime_rejects_a_move_before_metal_work() {
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let moved_guard = gpu_guard;
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a moved guard binding");
}

#[test]
fn guard_lifetime_rejects_shadowing_before_metal_work() {
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let gpu_guard = ();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a shadowed guard binding");
}

#[test]
fn guard_lifetime_rejects_a_guard_in_a_nested_scope() {
    let source = r#"
fn measurement() {
    {
        let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    }
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a guard dropped with a nested scope");
}

#[test]
fn guard_lifetime_rejects_an_enclosing_cfg_elided_function() {
    let source = r#"
#[cfg(any())]
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/enclosing_false_cfg.rs");
}

#[test]
fn guard_lifetime_rejects_an_enclosing_cfg_elided_module() {
    let source = r#"
#[cfg(any())]
mod hidden {
    fn measurement() {
        let gpu_guard = lattice_inference::measurement::gpu_test_lock();
        let _device = Device::system_default();
    }
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/enclosing_false_module.rs");
}

#[test]
fn guard_lifetime_rejects_an_out_of_line_cfg_elided_module() {
    let fixture = tempfile::tempdir().expect("create module-context fixture");
    let src = fixture.path().join("src");
    std::fs::create_dir(&src).expect("create fixture source directory");
    std::fs::write(src.join("lib.rs"), "#[cfg(any())]\npub mod child;\n")
        .expect("write fixture module parent");
    let child = src.join("child.rs");
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    std::fs::write(&child, source).expect("write fixture module child");
    let result =
        StructuredSource::parse_path(fixture.path(), &child, source, false).and_then(|parsed| {
            parsed.validate_selector(
                0..parsed.tokens.len(),
                DEVICE_SYSTEM_DEFAULT,
                SHARED_LOCK_SELECTOR,
                GuardRequirement::Lexical,
            )
        });
    let Err(message) = result else {
        panic!("source contract accepted src/child.rs");
    };
    assert!(
        message.contains("src/child.rs"),
        "source contract failure did not name src/child.rs: {message}"
    );
}

#[test]
fn guard_lifetime_rejects_an_out_of_line_module_in_a_cfg_elided_parent() {
    let fixture = tempfile::tempdir().expect("create nested module-context fixture");
    let src = fixture.path().join("src");
    std::fs::create_dir(&src).expect("create nested fixture source directory");
    std::fs::write(
        src.join("lib.rs"),
        "#[cfg(any())]\nmod hidden {\n    #[path = \"child.rs\"]\n    pub mod child;\n}\n",
    )
    .expect("write nested fixture module parent");
    let child = src.join("child.rs");
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    std::fs::write(&child, source).expect("write nested fixture module child");
    let result =
        StructuredSource::parse_path(fixture.path(), &child, source, false).and_then(|parsed| {
            parsed.validate_selector(
                0..parsed.tokens.len(),
                DEVICE_SYSTEM_DEFAULT,
                SHARED_LOCK_SELECTOR,
                GuardRequirement::Lexical,
            )
        });
    let Err(message) = result else {
        panic!("source contract accepted src/child.rs");
    };
    assert!(
        message.contains("src/child.rs"),
        "source contract failure did not name src/child.rs: {message}"
    );
}

#[test]
fn guard_lifetime_rejects_an_enclosing_cfg_elided_block() {
    let source = r#"
fn measurement() {
    #[cfg(any())]
    unsafe {
        let gpu_guard = lattice_inference::measurement::gpu_test_lock();
        let _device = Device::system_default();
    }
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/enclosing_false_block.rs");
}

#[test]
fn guard_lifetime_rejects_a_recursively_cfg_elided_function() {
    let source = r#"
#[cfg_attr(all(), cfg_attr(all(), cfg(any())))]
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/nested_false_cfg_attr.rs");
}

#[test]
fn guard_lifetime_checks_every_work_occurrence_after_a_false_gated_decoy() {
    let source = r#"
#[cfg(any())]
fn guarded_decoy() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}

fn live_work() {
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/false_gated_decoy.rs");
}

#[test]
fn guard_lifetime_ignores_a_comment_decoy_before_live_work() {
    let source = r#"
fn measurement() {
    {
        let gpu_guard = lattice_inference::measurement::gpu_test_lock();
        // Device::system_default()
    }
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/comment_decoy.rs");
}

#[test]
fn raw_dispatch_rejects_source_trivia_before_call_parentheses() {
    let source = r#"
#[test]
fn trivia_dispatch() {
    let command_buffer = queue.new_command_buffer /* comment */ ();
}
"#;
    assert_command_buffer_fixture_rejected(source, "src/forward/trivia_dispatch.rs");
}

#[test]
fn guard_lifetime_uses_braces_for_an_irregularly_indented_nested_scope() {
    let source = r#"
fn measurement() {
    {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
   }
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/irregular_nested_guard.rs");
}

#[test]
fn raw_dispatch_rejects_unclassifiable_macro_work() {
    let source = r#"
#[test]
fn macro_dispatch() {
    let gpu_guard = gpu_test_lock();
    dispatch!(queue.new_command_buffer());
}
"#;
    assert_command_buffer_fixture_rejected(source, "src/forward/macro_dispatch.rs");
}

#[test]
fn raw_dispatch_rejects_macro_generated_test_work() {
    let source = r#"
macro_rules! raw_test {
    () => {
        #[test]
        fn generated() {
            let command_buffer = queue.new_command_buffer();
        }
    };
}
raw_test!();
"#;
    assert_command_buffer_fixture_rejected(source, "src/forward/macro_generated_test.rs");
}

#[test]
fn guard_lifetime_rejects_stacked_cfg_separated_by_docs_and_blank_lines() {
    let source = r#"
fn measurement() {
    #[cfg(any())]

    /// The active Metal acquisition.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "stacked cfg with intervening documentation");
}

#[test]
fn guard_lifetime_rejects_a_raw_identifier_binding() {
    let source = r#"
fn measurement() {
    let r#gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a raw-identifier guard binding");
}

#[test]
fn guard_lifetime_rejects_a_let_else_derived_binding() {
    let source = r#"
fn measurement() {
    let Some(gpu_guard) = Some(lattice_inference::measurement::gpu_test_lock()) else {
        return;
    };
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a let-else-derived guard binding");
}

#[test]
fn guard_lifetime_rejects_an_if_let_derived_binding() {
    let source = r#"
fn measurement() {
    if let Some(gpu_guard) = Some(lattice_inference::measurement::gpu_test_lock()) {
        let _device = Device::system_default();
    }
}
"#;
    assert_guard_fixture_rejected(source, "an if-let-derived guard binding");
}

#[test]
fn guard_lifetime_rejects_a_match_derived_binding() {
    let source = r#"
fn measurement() {
    let gpu_guard = match () {
        () => lattice_inference::measurement::gpu_test_lock(),
    };
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a match-derived guard binding");
}

#[test]
fn metal_qwen35_state_construction_tests_use_function_lifetime_lock_bindings() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let exemptions = CONSTRUCTION_EXEMPTIONS
        .iter()
        .map(|(path, reason)| {
            assert!(
                !reason.trim().is_empty(),
                "construction exemption {path} needs a justification"
            );
            ((*path).to_string(), *reason)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(
        exemptions.len(),
        CONSTRUCTION_EXEMPTIONS.len(),
        "construction exemptions must not contain duplicate paths"
    );

    let mut discovered = BTreeSet::new();
    let mut classified = exemptions.keys().cloned().collect::<BTreeSet<_>>();
    let mut violations = Vec::new();
    for relative_dir in ["benches", "examples", "src/bin", "tests"] {
        for path in rust_sources_under(&manifest_dir.join(relative_dir)) {
            let source = std::fs::read_to_string(&path).expect("read construction source");
            let relative = path
                .strip_prefix(manifest_dir)
                .expect("source under manifest directory")
                .to_string_lossy()
                .into_owned();
            let parsed = if path.starts_with(manifest_dir.join("src")) {
                StructuredSource::parse_path(manifest_dir, &path, &source, relative_dir == "tests")
            } else {
                StructuredSource::parse(&relative, &source, relative_dir == "tests")
            }
            .unwrap_or_else(|reason| panic!("{reason}"));
            let construction_sites = parsed
                .construction_sites(0..parsed.tokens.len())
                .unwrap_or_else(|reason| panic!("{reason}"));
            if construction_sites.is_empty() {
                continue;
            }
            discovered.insert(relative.clone());

            if exemptions.contains_key(&relative) {
                continue;
            }
            let direct_guard = parsed.validate_work_sites(
                &construction_sites,
                SHARED_LOCK_SELECTOR,
                GuardRequirement::Lexical,
            );
            let protected_wrapper = RAW_HARNESS_ENTRYPOINTS
                .iter()
                .find_map(|(path, marker)| (*path == relative).then_some(*marker))
                .is_some_and(|marker| {
                    parsed
                        .validate_selector(
                            0..parsed.tokens.len(),
                            marker,
                            SHARED_LOCK_SELECTOR,
                            GuardRequirement::Lexical,
                        )
                        .is_ok()
                });
            match direct_guard {
                Ok(()) => {
                    classified.insert(relative);
                }
                Err(_) if protected_wrapper => {
                    classified.insert(relative);
                }
                Err(reason) => violations.push(reason),
            }
        }
    }

    let in_crate_relative = "src/forward/metal_qwen35.rs";
    let in_crate_source =
        std::fs::read_to_string(manifest_dir.join(in_crate_relative)).expect("read in-crate tests");
    let in_crate = StructuredSource::parse_path(
        manifest_dir,
        &manifest_dir.join(in_crate_relative),
        &in_crate_source,
        true,
    )
    .unwrap_or_else(|reason| panic!("{reason}"));
    for function in in_crate.test_functions() {
        let construction_sites = in_crate
            .construction_sites(function.body.clone())
            .unwrap_or_else(|reason| panic!("{reason}"));
        if construction_sites.is_empty() {
            continue;
        }
        let site = format!("{in_crate_relative}::{}", function.name);
        discovered.insert(site.clone());
        let result = in_crate.validate_work_sites(
            &construction_sites,
            LOCAL_LOCK_SELECTOR,
            GuardRequirement::Function {
                closing_brace: function.body.end,
            },
        );
        match result {
            Ok(()) => {
                classified.insert(site);
            }
            Err(reason) => violations.push(format!("{site}: {reason}")),
        }
    }

    assert!(
        violations.is_empty(),
        "MetalQwen35State construction sites without a live shared-lock binding:\n{}",
        violations.join("\n")
    );

    assert_construction_inventory_classified(&discovered, &classified);
}

#[test]
fn construction_inventory_comparison_rejects_an_unclassified_site() {
    let discovered = BTreeSet::from(["src/bin/known.rs".to_string(), "src/bin/new.rs".to_string()]);
    let classified = BTreeSet::from(["src/bin/known.rs".to_string()]);

    let result = std::panic::catch_unwind(|| {
        assert_construction_inventory_classified(&discovered, &classified);
    });
    assert!(
        result.is_err(),
        "inventory comparison accepted an unclassified site"
    );
}

#[test]
fn rust_targets_have_one_gpu_lock_definition_and_path() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let definition = ["fn gpu", "_test_lock"].concat();
    let lock_path = ["/tmp/lion-metal", "-gpu-test.lock"].concat();
    let mut definitions = BTreeSet::new();
    let mut paths = BTreeSet::new();

    for path in rust_sources_under(manifest_dir) {
        let source = std::fs::read_to_string(&path).expect("read Rust source");
        let relative = path
            .strip_prefix(manifest_dir)
            .expect("source under manifest directory")
            .to_string_lossy()
            .into_owned();
        if source.contains(&definition) {
            definitions.insert(relative.clone());
        }
        if source.contains(&lock_path) {
            paths.insert(relative);
        }
    }

    let only_shared_module = BTreeSet::from(["src/measurement.rs".to_string()]);
    assert_eq!(
        definitions, only_shared_module,
        "GPU lock behavior must have exactly one Rust definition"
    );
    assert_eq!(
        paths, only_shared_module,
        "the fleet GPU lock path must have exactly one Rust source of truth"
    );
}

#[test]
fn prior_lock_owners_and_existing_bench_use_the_shared_module() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let callers = [
        (
            "src/forward/metal.rs",
            "use crate::measurement::gpu_test_lock;",
        ),
        (
            "src/forward/metal_qwen35.rs",
            "use crate::measurement::gpu_test_lock;",
        ),
        (
            "tests/vision_s3b_vit_metal_gate_test.rs",
            "use lattice_inference::measurement::gpu_test_lock;",
        ),
        ("src/bin/bench_gdn_prefill_ab.rs", SHARED_LOCK_CALL),
    ];

    for (relative, shared_reference) in callers {
        let source =
            std::fs::read_to_string(manifest_dir.join(relative)).expect("read migrated caller");
        assert!(
            source.contains(shared_reference),
            "{relative} no longer references the one shared GPU lock"
        );
    }
}
