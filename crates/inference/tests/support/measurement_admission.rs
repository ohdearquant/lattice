use super::*;
use sha2::{Digest, Sha256};
use syn::spanned::Spanned;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContractKind {
    Raw,
    Additional,
    State,
    Synchronous,
    Indirect,
    Worker,
    NonMeasurement,
}

#[derive(Clone, Copy)]
enum MeasurementContract {
    Raw(CallSelector),
    Additional(CallSelector),
    State,
    Synchronous(&'static SyncWorkTarget<'static>),
    Indirect,
    Worker,
}

impl MeasurementContract {
    fn kind(self) -> ContractKind {
        match self {
            Self::Raw(_) => ContractKind::Raw,
            Self::Additional(_) => ContractKind::Additional,
            Self::State => ContractKind::State,
            Self::Synchronous(_) => ContractKind::Synchronous,
            Self::Indirect => ContractKind::Indirect,
            Self::Worker => ContractKind::Worker,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NonMeasurementPurpose {
    DatabaseMigration,
    ChatProduct,
    ProductCli,
    ServerProduct,
    CorrectnessDump,
}

impl NonMeasurementPurpose {
    fn path(self) -> &'static str {
        match self {
            Self::DatabaseMigration => "src/bin/backfill_qwen3.rs",
            Self::ChatProduct => "src/bin/chat_metal.rs",
            Self::ProductCli => "src/bin/lattice/main.rs",
            Self::ServerProduct => "src/bin/lattice_serve.rs",
            Self::CorrectnessDump => "examples/dump_embeddinggemma2.rs",
        }
    }
}

struct ReviewedSource {
    path: &'static str,
    sha256: &'static str,
}

#[derive(Clone, Copy)]
struct NonMeasurementContract {
    purpose: NonMeasurementPurpose,
    reason: &'static str,
    sources: &'static [ReviewedSource],
    constructions: usize,
}

const BACKFILL_SOURCES: &[ReviewedSource] = &[ReviewedSource {
    path: "src/bin/backfill_qwen3.rs",
    sha256: "e3dce92a789e59e27ef301dbec61e548bb7eef6c95ade634276551a755843d1e",
}];

const CHAT_SOURCES: &[ReviewedSource] = &[ReviewedSource {
    path: "src/bin/chat_metal.rs",
    sha256: "1cfee06fc670f03e20c16a7b1afed5546520f69a0cdc17e174b6b90da401250a",
}];

const CLI_SOURCES: &[ReviewedSource] = &[
    ReviewedSource {
        path: "src/bin/lattice/main.rs",
        sha256: "777ea06c2698ddb05883b825b6eeaaaac00fbe10dbc469345c773bd775dd6a4e",
    },
    ReviewedSource {
        path: "src/bin/lattice/chat.rs",
        sha256: "7a4b122fee6b86fd4946e83ba20242f1f1ff9a97fcce84374c25e920ff7c5546",
    },
    ReviewedSource {
        path: "src/bin/lattice/doctor.rs",
        sha256: "a3bdb5a421de7d633616f4c2a24fa3b4eec2c3da1f4bee352a0b4a20cf14ac6d",
    },
    ReviewedSource {
        path: "src/bin/lattice/prune_score.rs",
        sha256: "92dacd183c52e6662335146bcce6c2aa574f9211840d1d4c310f82391c2d68f6",
    },
    ReviewedSource {
        path: "src/bin/lattice/router.rs",
        sha256: "a9495551664e1df928bf7081330c6096e49f7001472fda51898dc1555243798f",
    },
    ReviewedSource {
        path: "src/bin/lattice/serve.rs",
        sha256: "49cefc85aa78fc7d3465f190535a91e04004d97fcbfc877175f60c7d0616a1aa",
    },
];

const SERVER_SOURCES: &[ReviewedSource] = &[ReviewedSource {
    path: "src/bin/lattice_serve.rs",
    sha256: "896356be491f0d7c86a35dff10398cb58a75b44f46258d3544518a88f3cc25a6",
}];

const DUMP_SOURCES: &[ReviewedSource] = &[ReviewedSource {
    path: "examples/dump_embeddinggemma2.rs",
    sha256: "348c2db6229d0060ea48ed933973e832af02e46c32e8e2ecb689cd012da00ebb",
}];

const NON_MEASUREMENT_TARGETS: &[NonMeasurementContract] = &[
    NonMeasurementContract {
        purpose: NonMeasurementPurpose::DatabaseMigration,
        reason: "Database embedding migration with progress/rate/ETA telemetry, not a controlled measurement harness; retain its exact batch construction exemption.",
        sources: BACKFILL_SOURCES,
        constructions: 1,
    },
    NonMeasurementContract {
        purpose: NonMeasurementPurpose::ChatProduct,
        reason: "Interactive, one-shot and persistent JSON chat product; latency/throughput events are product telemetry. Both exact construction exemptions remain required.",
        sources: CHAT_SOURCES,
        constructions: 2,
    },
    NonMeasurementContract {
        purpose: NonMeasurementPurpose::ProductCli,
        reason: "Product CLI for chat, serving, inspection and pruning artifacts; its two chat/worker construction exemptions and two guarded prune-score constructions remain required across the reviewed module closure.",
        sources: CLI_SOURCES,
        constructions: 4,
    },
    NonMeasurementContract {
        purpose: NonMeasurementPurpose::ServerProduct,
        reason: "Long-running HTTP product server with request telemetry; both model-loader construction exemptions remain required.",
        sources: SERVER_SOURCES,
        constructions: 2,
    },
    NonMeasurementContract {
        purpose: NonMeasurementPurpose::CorrectnessDump,
        reason: "Offline JSON token/embedding correctness dump without timed measurements; optional Metal construction retains its live shared-lock check.",
        sources: DUMP_SOURCES,
        constructions: 1,
    },
];

#[derive(Clone, Copy)]
enum TargetClass {
    Measurement(MeasurementContract),
    NonMeasurement(&'static NonMeasurementContract),
}

impl TargetClass {
    fn kind(self) -> ContractKind {
        match self {
            Self::Measurement(contract) => contract.kind(),
            Self::NonMeasurement(_) => ContractKind::NonMeasurement,
        }
    }
}

#[derive(Clone, Copy)]
struct TargetAdmission {
    path: &'static str,
    class: TargetClass,
}

struct DispatchReceipt {
    path: String,
    kind: ContractKind,
}

struct TargetIdentity<'a> {
    kind: &'a str,
    name: &'a str,
    relative: &'a str,
}

fn target_admissions() -> Vec<TargetAdmission> {
    let mut rows = Vec::new();
    for &(path, selector) in RAW_HARNESS_ENTRYPOINTS {
        if path.starts_with("examples/") || path.starts_with("src/bin/") {
            rows.push(TargetAdmission {
                path,
                class: TargetClass::Measurement(MeasurementContract::Raw(selector)),
            });
        }
    }
    for &(path, selector) in ADDITIONAL_GUARDED_ENTRYPOINTS {
        rows.push(TargetAdmission {
            path,
            class: TargetClass::Measurement(MeasurementContract::Additional(selector)),
        });
    }
    for &path in TARGETS_WITH_STATE_DRIVEN_WORK {
        rows.push(TargetAdmission {
            path,
            class: TargetClass::Measurement(MeasurementContract::State),
        });
    }
    for contract in SYNC_WORK_TARGETS {
        rows.push(TargetAdmission {
            path: contract.path,
            class: TargetClass::Measurement(MeasurementContract::Synchronous(contract)),
        });
    }
    for &path in measurement_indirect::TARGETS {
        rows.push(TargetAdmission {
            path,
            class: TargetClass::Measurement(MeasurementContract::Indirect),
        });
    }
    rows.push(TargetAdmission {
        path: "examples/bench_serve_prepare.rs",
        class: TargetClass::Measurement(MeasurementContract::Worker),
    });
    for contract in NON_MEASUREMENT_TARGETS {
        rows.push(TargetAdmission {
            path: contract.purpose.path(),
            class: TargetClass::NonMeasurement(contract),
        });
    }
    rows
}

fn classification_index(rows: &[TargetAdmission]) -> Result<BTreeMap<&str, TargetClass>, String> {
    let mut index = BTreeMap::new();
    for row in rows {
        if !(row.path.starts_with("examples/") || row.path.starts_with("src/bin/")) {
            return Err(format!("non-example/bin admission row {}", row.path));
        }
        if index.insert(row.path, row.class).is_some() {
            return Err(format!(
                "duplicate or overlapping target admission {}",
                row.path
            ));
        }
    }
    Ok(index)
}

fn validate_receipts(
    discovered: &BTreeSet<String>,
    rows: &[TargetAdmission],
    receipts: &[DispatchReceipt],
) -> Result<(), String> {
    let index = classification_index(rows)?;
    let classified = index
        .keys()
        .map(|path| (*path).to_string())
        .collect::<BTreeSet<_>>();
    if &classified != discovered {
        return Err("discovered and classified Metal example/bin roots disagree".into());
    }
    let mut completed = BTreeSet::new();
    for receipt in receipts {
        let class = index
            .get(receipt.path.as_str())
            .ok_or_else(|| format!("receipt has no target classification {}", receipt.path))?;
        if class.kind() != receipt.kind {
            return Err(format!("wrong validator receipt for {}", receipt.path));
        }
        if !completed.insert(receipt.path.clone()) {
            return Err(format!("duplicate validator receipt for {}", receipt.path));
        }
    }
    if &completed != discovered {
        return Err("a discovered Metal target has no completed validator receipt".into());
    }
    Ok(())
}

#[derive(Default)]
struct ImmediateSyntax<'a> {
    calls: Vec<&'a syn::ExprCall>,
    locals: Vec<&'a syn::Local>,
    glob_import: bool,
}

impl<'a> Visit<'a> for ImmediateSyntax<'a> {
    fn visit_expr_call(&mut self, call: &'a syn::ExprCall) {
        self.calls.push(call);
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_local(&mut self, local: &'a syn::Local) {
        self.locals.push(local);
        syn::visit::visit_local(self, local);
    }

    fn visit_use_glob(&mut self, _glob: &'a syn::UseGlob) {
        self.glob_import = true;
    }
}

fn exact_call_path(call: &syn::ExprCall, names: &[&str]) -> bool {
    let syn::Expr::Path(path) = &*call.func else {
        return false;
    };
    path.attrs.is_empty()
        && path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == names.len()
        && path.path.segments.iter().zip(names).all(|(segment, name)| {
            segment.ident == *name && matches!(segment.arguments, syn::PathArguments::None)
        })
}

fn identifier_offsets(parsed: &StructuredSource, name: &str) -> BTreeSet<usize> {
    parsed
        .tokens
        .iter()
        .filter(|token| token.ident() == Some(name))
        .map(|token| token.offset)
        .collect()
}

fn token_at(parsed: &StructuredSource, offset: usize) -> Result<usize, String> {
    parsed
        .tokens
        .iter()
        .position(|token| token.offset == offset)
        .ok_or_else(|| format!("unmapped immediate-work syntax at byte {offset}"))
}

fn immediate_return(output: &syn::ReturnType) -> bool {
    let syn::ReturnType::Type(_, ty) = output else {
        return true;
    };
    if matches!(&**ty, syn::Type::Tuple(tuple) if tuple.elems.is_empty()) {
        return true;
    }
    let syn::Type::Path(path) = &**ty else {
        return false;
    };
    if path.qself.is_some() || path.path.leading_colon.is_some() || path.path.segments.len() != 1 {
        return false;
    }
    let segment = &path.path.segments[0];
    if segment.ident == "f64" && matches!(segment.arguments, syn::PathArguments::None) {
        return true;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return false;
    };
    segment.ident == "Result"
        && arguments.args.len() == 2
        && matches!(&arguments.args[0], syn::GenericArgument::Type(syn::Type::Tuple(tuple)) if tuple.elems.is_empty())
        && matches!(&arguments.args[1], syn::GenericArgument::Type(syn::Type::Path(path)) if path.qself.is_none() && path.path.is_ident("String"))
}

fn plain_helper(
    parsed: &StructuredSource,
    syntax: &syn::File,
    name: &str,
    sites: &[usize],
) -> Result<(), String> {
    let functions = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == name => Some(function),
            _ => None,
        })
        .collect::<Vec<_>>();
    if functions.len() != 1 {
        return Err(format!("expected one reviewed top-level helper {name}"));
    }
    let function = functions[0];
    if function.sig.asyncness.is_some()
        || function.sig.unsafety.is_some()
        || function.sig.abi.is_some()
        || !function.sig.generics.params.is_empty()
        || !immediate_return(&function.sig.output)
    {
        return Err(format!(
            "helper {name} is not an ordinary synchronous function"
        ));
    }
    let definition = function.sig.ident.span().byte_range().start;
    let mut expected = sites
        .iter()
        .map(|site| parsed.tokens[*site].offset)
        .collect::<BTreeSet<_>>();
    expected.insert(definition);
    if identifier_offsets(parsed, name) != expected {
        return Err(format!(
            "helper {name} has an alias, shadow, or unreviewed use"
        ));
    }
    let function_token = token_at(parsed, function.sig.fn_token.span.byte_range().start)?;
    let definition_formula = CfgFormula::all([
        parsed.formula_at(token_at(parsed, definition)?)?,
        parsed.attributes_formula(&parsed.item_attributes_before(function_token))?,
    ]);
    for site in sites {
        if !parsed.formula_at(*site)?.implies(&definition_formula)? {
            return Err(format!(
                "helper {name} is unavailable for a caller configuration"
            ));
        }
    }
    Ok(())
}

struct ImmediateWork {
    expected: BTreeSet<usize>,
    seen: BTreeSet<usize>,
    allowed_closures: BTreeSet<usize>,
    function_depth: usize,
    deferred: bool,
    error: Option<String>,
}

impl ImmediateWork {
    fn record(&mut self, offset: usize) {
        if self.expected.contains(&offset) {
            self.seen.insert(offset);
            if self.deferred || self.function_depth == 0 {
                self.error = Some("protected work crosses an unreviewed deferred boundary".into());
            }
        }
    }
}

impl<'ast> Visit<'ast> for ImmediateWork {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*call.func
            && let Some(segment) = path.path.segments.last()
        {
            self.record(segment.ident.span().byte_range().start);
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.record(call.method.span().byte_range().start);
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        let previous = self.deferred;
        self.deferred |= !self
            .allowed_closures
            .contains(&closure.span().byte_range().start);
        syn::visit::visit_expr_closure(self, closure);
        self.deferred = previous;
    }

    fn visit_expr_async(&mut self, expression: &'ast syn::ExprAsync) {
        let previous = self.deferred;
        self.deferred = true;
        syn::visit::visit_expr_async(self, expression);
        self.deferred = previous;
    }

    fn visit_item(&mut self, item: &'ast syn::Item) {
        let previous = self.deferred;
        self.deferred |= self.function_depth > 0;
        syn::visit::visit_item(self, item);
        self.deferred = previous;
    }

    fn visit_item_fn(&mut self, function: &'ast ItemFn) {
        let previous = self.deferred;
        self.deferred |= function.sig.asyncness.is_some();
        self.function_depth += 1;
        syn::visit::visit_item_fn(self, function);
        self.function_depth -= 1;
        self.deferred = previous;
    }
}

fn validate_immediate_work(
    relative: &str,
    parsed: &StructuredSource,
    sites: &[usize],
) -> Result<(), String> {
    let syntax = syn::parse_file(&parsed.source).map_err(|reason| reason.to_string())?;
    let mut shapes = ImmediateSyntax::default();
    shapes.visit_file(&syntax);
    let mut work = ImmediateWork {
        expected: sites
            .iter()
            .map(|site| parsed.tokens[*site].offset)
            .collect(),
        seen: BTreeSet::new(),
        allowed_closures: BTreeSet::new(),
        function_depth: 0,
        deferred: false,
        error: None,
    };
    if relative == "examples/profile_metal.rs" {
        for (name, count) in [("measure_prefill", 2), ("measure_steps", 4)] {
            let locals = shapes
                .locals
                .iter()
                .filter_map(|local| match &local.pat {
                    syn::Pat::Ident(binding)
                        if binding.ident == name
                            && binding.subpat.is_none()
                            && binding.mutability.is_none()
                            && binding.by_ref.is_none() =>
                    {
                        Some((*local, binding))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            if locals.len() != 1 {
                return Err(format!("missing unique closure {name}"));
            }
            let (local, binding) = locals[0];
            let Some(init) = &local.init else {
                return Err("closure has no initializer".into());
            };
            let syn::Expr::Closure(closure) = &*init.expr else {
                return Err("closure initializer changed".into());
            };
            if !local.attrs.is_empty()
                || init.diverge.is_some()
                || closure.asyncness.is_some()
                || closure.capture.is_some()
                || closure.movability.is_some()
                || !closure.attrs.is_empty()
            {
                return Err("reviewed closure is no longer synchronous and borrowed".into());
            }
            let definition = binding.ident.span().byte_range().start;
            if parsed.enclosing_function_path(token_at(parsed, definition)?) != "main" {
                return Err("reviewed closure moved outside main".into());
            }
            let calls = shapes
                .calls
                .iter()
                .filter(|call| exact_call_path(call, &[name]))
                .collect::<Vec<_>>();
            if calls.len() != count {
                return Err(format!("closure {name} invocation count changed"));
            }
            let mut identifiers = BTreeSet::from([definition]);
            let mut calls_to_guard = Vec::new();
            for call in calls {
                let offset = call.func.span().byte_range().start;
                let site = token_at(parsed, offset)?;
                if offset <= closure.span().byte_range().end
                    || parsed.enclosing_function_path(site) != "main"
                    || !parsed.formula_at(site)?.satisfiable()?
                {
                    return Err("reviewed closure call moved or became inactive".into());
                }
                identifiers.insert(offset);
                work.expected.insert(offset);
                calls_to_guard.push(site);
            }
            if identifier_offsets(parsed, name) != identifiers {
                return Err(format!(
                    "closure {name} is aliased, shadowed, assigned, or escapes"
                ));
            }
            parsed.validate_work_sites(
                &calls_to_guard,
                SHARED_LOCK_SELECTOR,
                GuardRequirement::Lexical,
            )?;
            work.allowed_closures
                .insert(closure.span().byte_range().start);
        }
    }
    if relative == "examples/rss_autorelease_probe.rs" {
        if shapes.glob_import {
            return Err("an unreviewed glob may replace the autoreleasepool namespace".into());
        }
        let mut namespaces = BTreeSet::new();
        for call in &shapes.calls {
            if !exact_call_path(call, &["objc", "rc", "autoreleasepool"]) {
                continue;
            }
            namespaces.insert(call.func.span().byte_range().start);
            if call.args.len() != 1 {
                return Err("autoreleasepool argument contract changed".into());
            }
            let Some(syn::Expr::Closure(closure)) = call.args.first() else {
                return Err("autoreleasepool must immediately consume its closure".into());
            };
            if !closure.inputs.is_empty()
                || closure.asyncness.is_some()
                || closure.movability.is_some()
                || closure.capture.is_some()
                || !closure.attrs.is_empty()
            {
                return Err(
                    "autoreleasepool callback is not an ordinary synchronous closure".into(),
                );
            }
            work.allowed_closures
                .insert(closure.span().byte_range().start);
        }
        if identifier_offsets(parsed, "objc") != namespaces {
            return Err("autoreleasepool namespace has an unreviewed binding or use".into());
        }
    }
    work.visit_file(&syntax);
    if let Some(error) = work.error {
        return Err(error);
    }
    if work.seen != work.expected {
        return Err("protected work has no direct AST call".into());
    }
    Ok(())
}

struct EntrypointHelperContract {
    path: &'static str,
    helper: &'static str,
    source_sha256: &'static str,
    reason: &'static str,
    calls: usize,
    work: &'static [(CallSelector, usize)],
}

// These hashes bind the transitive helper bodies and their namespace bindings.
// Any source edit requires renewed review of completion before guard release.
const ENTRYPOINT_HELPERS: &[EntrypointHelperContract] = &[
    EntrypointHelperContract {
        path: "examples/bench_concurrent.rs",
        helper: "run",
        source_sha256: "26b2d46923d86986db4b9a76184e881bca743530e6f45b2d11f3a50fe2abc0fd",
        reason: "The guarded run call completes both synchronous session measurement helpers: one device construction and six forward-step sites; no worker is detached.",
        calls: 1,
        work: &[
            (DEVICE_SYSTEM_DEFAULT, 1),
            (CallSelector::Method("forward_step"), 6),
        ],
    },
    EntrypointHelperContract {
        path: "examples/bench_gemm_shapes.rs",
        helper: "run",
        source_sha256: "3107d065be087a515d60e807ffcd6812f2971a320297196b7621920803c43eaf",
        reason: "The guarded run call completes the selected kernel measurements; the shared command helper invokes its encoder callback and waits for its command buffer before returning.",
        calls: 1,
        work: &[(DEVICE_SYSTEM_DEFAULT, 1), (NEW_COMMAND_BUFFER, 1)],
    },
    EntrypointHelperContract {
        path: "examples/bench_embed_quality.rs",
        helper: "run_bench",
        source_sha256: "71ed69f73f4bfa600709aa7e75a9114b1f8fa1ad1a25276270f176fa85a29df2",
        reason: "The Metal branch holds the guard through run_bench; the CPU fallback is cfg-disjoint. Qwen construction and two Qwen encode sites complete inline or through immediate collect; two BGE encode sites are CPU work.",
        calls: 2,
        work: &[
            (CallSelector::Path(&["QwenModel", "from_directory"]), 1),
            (CallSelector::Method("encode"), 4),
        ],
    },
];

fn validate_entrypoint_provenance(
    relative: &str,
    parsed: &StructuredSource,
    helper: &str,
    sites: &[usize],
) -> Result<(), String> {
    let contract = ENTRYPOINT_HELPERS
        .iter()
        .find(|contract| contract.path == relative && contract.helper == helper)
        .ok_or("named entrypoint has no reviewed source contract")?;
    if contract.reason.trim().is_empty() || contract.work.is_empty() || contract.calls == 0 {
        return Err("named entrypoint source contract is empty".into());
    }
    if format!("{:x}", Sha256::digest(parsed.source.as_bytes())) != contract.source_sha256 {
        return Err(
            "named entrypoint source provenance changed; re-review synchronous completion".into(),
        );
    }
    if sites.len() != contract.calls {
        return Err("named entrypoint invocation census changed".into());
    }
    for &(selector, count) in contract.work {
        let work = parsed.call_sites(0..parsed.tokens.len(), selector, true)?;
        if count == 0 || work.len() != count {
            return Err("named entrypoint protected-work census changed".into());
        }
        for site in work {
            if !parsed.formula_at(site)?.satisfiable()? {
                return Err("named entrypoint protected work is inactive".into());
            }
        }
    }
    Ok(())
}

fn validate_entrypoint_shape(
    relative: &str,
    sources: &[ParsedModuleSource],
    selector: CallSelector,
) -> Result<(), String> {
    if sources.len() != 1 || sources[0].relative != relative {
        return Err(format!(
            "unreviewed entrypoint module closure for {relative}"
        ));
    }
    let parsed = &sources[0].parsed;
    let sites = parsed.call_sites(0..parsed.tokens.len(), selector, true)?;
    if sites.is_empty() {
        return Err(format!("entrypoint contract checked no work in {relative}"));
    }
    validate_immediate_work(relative, parsed, &sites)?;
    if let CallSelector::Path([name]) = selector {
        let syntax = syn::parse_file(&parsed.source).map_err(|reason| reason.to_string())?;
        plain_helper(parsed, &syntax, name, &sites)?;
    }
    parsed.validate_work_sites(&sites, SHARED_LOCK_SELECTOR, GuardRequirement::Lexical)
}

fn validate_entrypoint(
    relative: &str,
    sources: &[ParsedModuleSource],
    selector: CallSelector,
) -> Result<(), String> {
    if sources.len() != 1 || sources[0].relative != relative {
        return Err(format!(
            "unreviewed entrypoint module closure for {relative}"
        ));
    }
    if let CallSelector::Path([name]) = selector {
        let parsed = &sources[0].parsed;
        let sites = parsed.call_sites(0..parsed.tokens.len(), selector, true)?;
        validate_entrypoint_provenance(relative, parsed, name, &sites)?;
    }
    validate_entrypoint_shape(relative, sources, selector)
}

fn validate_state_target(relative: &str, sources: &[ParsedModuleSource]) -> Result<(), String> {
    let helpers = STATE_WORK_HELPER_ENTRYPOINTS
        .iter()
        .filter(|(path, _)| *path == relative)
        .map(|(_, selector)| *selector)
        .collect::<Vec<_>>();
    let expected_helpers = helpers
        .iter()
        .map(|selector| selector.final_name().to_string())
        .collect::<BTreeSet<_>>();
    let mut found_helpers = BTreeSet::new();
    let mut checked = 0;
    for source in sources {
        let parsed = &source.parsed;
        let mut direct = Vec::new();
        let mut mediated = BTreeSet::new();
        let mut immediate = Vec::new();
        for selector in STATE_WORK_SELECTORS {
            for site in parsed.call_sites(0..parsed.tokens.len(), *selector, true)? {
                immediate.push(site);
                let owner = parsed.enclosing_function_path(site);
                if helpers.iter().any(|helper| helper.final_name() == owner) {
                    mediated.insert(owner);
                } else {
                    direct.push(site);
                }
            }
        }
        if !direct.is_empty() {
            parsed.validate_work_sites(&direct, SHARED_LOCK_SELECTOR, GuardRequirement::Lexical)?;
            checked += direct.len();
        }
        for owner in mediated {
            let selector = helpers
                .iter()
                .find(|selector| selector.final_name() == owner)
                .ok_or_else(|| format!("unreviewed state helper {owner}"))?;
            let sites = parsed.call_sites(0..parsed.tokens.len(), *selector, true)?;
            if sites.is_empty() {
                return Err(format!("state helper {owner} has no guarded invocation"));
            }
            let syntax = syn::parse_file(&parsed.source).map_err(|reason| reason.to_string())?;
            plain_helper(parsed, &syntax, selector.final_name(), &sites)?;
            immediate.extend(sites.iter().copied());
            parsed.validate_work_sites(&sites, SHARED_LOCK_SELECTOR, GuardRequirement::Lexical)?;
            checked += sites.len();
            found_helpers.insert(owner);
        }
        validate_immediate_work(relative, parsed, &immediate)?;
    }
    if checked == 0 || found_helpers != expected_helpers {
        return Err(format!(
            "state-work coverage is empty or a reviewed helper changed in {relative}"
        ));
    }
    Ok(())
}

pub(super) fn validate_target_constructions(
    kind: &str,
    name: &str,
    relative: &str,
    sources: &[ParsedModuleSource],
) -> Result<usize, String> {
    let prefix = format!("{kind}:{name}:{relative}");
    let mut exemptions = BTreeMap::new();
    for exemption in CONSTRUCTION_EXEMPTIONS {
        if exemption.reason.trim().is_empty() {
            return Err(format!(
                "construction exemption {} has no reason",
                exemption.site
            ));
        }
        if exemption.site.starts_with(&format!("{prefix}=>"))
            && exemptions
                .insert(exemption.site.to_string(), exemption)
                .is_some()
        {
            return Err(format!(
                "duplicate construction exemption {}",
                exemption.site
            ));
        }
    }
    let expected = exemptions.keys().cloned().collect::<BTreeSet<_>>();
    let mut tally = ConstructionTally {
        exemptions: &exemptions,
        discovered: BTreeSet::new(),
        classified: BTreeSet::new(),
        used_exemptions: BTreeSet::new(),
        moved_exemptions: Vec::new(),
    };
    for source in sources {
        let parsed = &source.parsed;
        let mut ordinals = BTreeMap::new();
        for (construction, selector) in parsed.construction_sites(0..parsed.tokens.len())? {
            let Some(site) = tally.record(parsed, &prefix, construction, selector, &mut ordinals)
            else {
                continue;
            };
            parsed.validate_work_sites(
                &[construction],
                SHARED_LOCK_SELECTOR,
                GuardRequirement::Lexical,
            )?;
            tally.classified.insert(site);
        }
    }
    if !tally.moved_exemptions.is_empty() {
        return Err(tally.moved_exemptions.join("\n"));
    }
    if tally.used_exemptions != expected || tally.discovered != tally.classified {
        return Err(format!(
            "construction/exemption coverage changed for {relative}"
        ));
    }
    Ok(tally.discovered.len())
}

fn validate_nonmeasurement(
    identity: &TargetIdentity<'_>,
    sources: &[ParsedModuleSource],
    contract: &NonMeasurementContract,
) -> Result<(), String> {
    if identity.relative != contract.purpose.path() {
        return Err("nonmeasurement purpose does not authorize this exact target".into());
    }
    if contract.reason.trim().is_empty() {
        return Err("nonmeasurement target has no reviewed reason".into());
    }
    let mut expected = BTreeMap::new();
    for source in contract.sources {
        if expected.insert(source.path, source.sha256).is_some() {
            return Err("duplicate reviewed nonmeasurement source".into());
        }
    }
    let actual = sources
        .iter()
        .map(|source| source.relative.as_str())
        .collect::<BTreeSet<_>>();
    if actual.len() != sources.len() || actual != expected.keys().copied().collect::<BTreeSet<_>>()
    {
        return Err("nonmeasurement source closure changed".into());
    }
    for source in sources {
        let fingerprint = format!("{:x}", Sha256::digest(source.parsed.source.as_bytes()));
        if expected.get(source.relative.as_str()).copied() != Some(fingerprint.as_str()) {
            return Err(format!(
                "nonmeasurement purpose/source needs re-review: {}",
                source.relative
            ));
        }
    }
    let constructions =
        validate_target_constructions(identity.kind, identity.name, identity.relative, sources)?;
    if contract.constructions == 0 || constructions != contract.constructions {
        return Err("nonmeasurement construction population changed".into());
    }
    Ok(())
}

fn dispatch_target(
    identity: &TargetIdentity<'_>,
    sources: &[ParsedModuleSource],
    row: &TargetAdmission,
) -> Result<DispatchReceipt, String> {
    if row.path != identity.relative {
        return Err("classifier row and target identity disagree".into());
    }
    match row.class {
        TargetClass::Measurement(contract) => match contract {
            MeasurementContract::Raw(selector) | MeasurementContract::Additional(selector) => {
                validate_entrypoint(identity.relative, sources, selector)?;
            }
            MeasurementContract::State => validate_state_target(identity.relative, sources)?,
            MeasurementContract::Synchronous(contract) => validate_sync_work(sources, contract)?,
            MeasurementContract::Indirect => {
                measurement_indirect::validate_target(identity.relative, sources)?
            }
            MeasurementContract::Worker => measurement_worker::validate_target(sources)?,
        },
        TargetClass::NonMeasurement(contract) => {
            validate_nonmeasurement(identity, sources, contract)?
        }
    }
    Ok(DispatchReceipt {
        path: identity.relative.to_string(),
        kind: row.class.kind(),
    })
}

#[test]
fn every_discovered_metal_example_and_binary_has_a_completed_contract() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let rows = target_admissions();
    let index = classification_index(&rows).expect("closed target classification");
    let mut discovered = BTreeSet::new();
    let mut receipts = Vec::new();
    for target in cargo_targets(manifest_dir, &["example", "bin"]).expect("discover Cargo targets")
    {
        let relative = target
            .path
            .strip_prefix(manifest_dir)
            .expect("target in crate")
            .to_string_lossy()
            .into_owned();
        let sources = parsed_module_closure(manifest_dir, &target.path, false)
            .expect("parse target source closure");
        let mut metal = false;
        for source in &sources {
            metal |= source
                .parsed
                .has_top_level_metal_marker()
                .expect("classifiable Metal markers");
        }
        if !metal {
            continue;
        }
        assert!(
            discovered.insert(relative.clone()),
            "duplicate Cargo target root"
        );
        let class = *index
            .get(relative.as_str())
            .expect("Metal target must have a classification");
        let row = rows
            .iter()
            .find(|row| row.path == relative)
            .expect("classified row");
        assert_eq!(class.kind(), row.class.kind());
        let identity = TargetIdentity {
            kind: &target.kind,
            name: &target.name,
            relative: &relative,
        };
        receipts.push(
            dispatch_target(&identity, &sources, row)
                .unwrap_or_else(|reason| panic!("{relative}: {reason}")),
        );
    }
    validate_receipts(&discovered, &rows, &receipts).expect("every target validator completed");
}

fn fixture_source(relative: &str, source: &str) -> Vec<ParsedModuleSource> {
    vec![ParsedModuleSource {
        relative: relative.to_string(),
        parsed: StructuredSource::parse(relative, source, false).expect("parse classifier fixture"),
    }]
}

fn raw_fixture_row() -> TargetAdmission {
    TargetAdmission {
        path: "examples/classifier_fixture.rs",
        class: TargetClass::Measurement(MeasurementContract::Raw(DEVICE_SYSTEM_DEFAULT)),
    }
}

#[test]
fn classifier_rejects_missing_stale_duplicate_and_overlapping_rows() {
    let row = raw_fixture_row();
    let roots = BTreeSet::from([row.path.to_string()]);
    let receipt = DispatchReceipt {
        path: row.path.to_string(),
        kind: ContractKind::Raw,
    };
    assert!(validate_receipts(&roots, &[row], &[receipt]).is_ok());
    assert!(validate_receipts(&roots, &[], &[]).is_err());
    assert!(validate_receipts(&BTreeSet::new(), &[row], &[]).is_err());
    assert!(classification_index(&[row, row]).is_err());
    let overlap = TargetAdmission {
        path: row.path,
        class: TargetClass::Measurement(MeasurementContract::State),
    };
    assert!(classification_index(&[row, overlap]).is_err());
}

#[test]
fn classifier_rejects_unexecuted_duplicate_and_wrong_validator_receipts() {
    let row = raw_fixture_row();
    let roots = BTreeSet::from([row.path.to_string()]);
    assert!(validate_receipts(&roots, &[row], &[]).is_err());
    let wrong = DispatchReceipt {
        path: row.path.to_string(),
        kind: ContractKind::State,
    };
    assert!(validate_receipts(&roots, &[row], &[wrong]).is_err());
    let make = || DispatchReceipt {
        path: row.path.to_string(),
        kind: ContractKind::Raw,
    };
    assert!(validate_receipts(&roots, &[row], &[make(), make()]).is_err());
}

#[test]
fn classifier_dispatches_the_real_validator_and_rejects_empty_work() {
    let row = raw_fixture_row();
    let identity = TargetIdentity {
        kind: "example",
        name: "classifier_fixture",
        relative: row.path,
    };
    let guarded = fixture_source(
        row.path,
        r#"
fn main() {
    let gpu = lattice_inference::measurement::gpu_test_lock();
    let device = Device::system_default();
}
"#,
    );
    let receipt = dispatch_target(&identity, &guarded, &row).expect("guarded real dispatch");
    validate_receipts(&BTreeSet::from([row.path.to_string()]), &[row], &[receipt]).unwrap();
    let unguarded = fixture_source(
        row.path,
        "fn main() { let device = Device::system_default(); }",
    );
    assert!(dispatch_target(&identity, &unguarded, &row).is_err());
    let empty = fixture_source(row.path, "fn main() {}");
    assert!(dispatch_target(&identity, &empty, &row).is_err());
    let state = TargetAdmission {
        path: row.path,
        class: TargetClass::Measurement(MeasurementContract::State),
    };
    assert!(dispatch_target(&identity, &empty, &state).is_err());
}

#[test]
fn classifier_cannot_relabel_measurement_as_a_purpose_exception() {
    let row = raw_fixture_row();
    let identity = TargetIdentity {
        kind: "example",
        name: "classifier_fixture",
        relative: row.path,
    };
    let sources = fixture_source(row.path, "fn main() {}");
    let false_exception = TargetAdmission {
        path: row.path,
        class: TargetClass::NonMeasurement(&NON_MEASUREMENT_TARGETS[4]),
    };
    assert!(dispatch_target(&identity, &sources, &false_exception).is_err());
    let mut missing_reason = NON_MEASUREMENT_TARGETS[4];
    missing_reason.reason = " ";
    let dump_identity = TargetIdentity {
        kind: "example",
        name: "dump_embeddinggemma2",
        relative: missing_reason.purpose.path(),
    };
    assert!(
        validate_nonmeasurement(&dump_identity, &sources, &missing_reason)
            .unwrap_err()
            .contains("reason")
    );
}

#[test]
fn classifier_purpose_fingerprint_rejects_source_and_module_drift() {
    let contract = &NON_MEASUREMENT_TARGETS[4];
    let relative = contract.purpose.path();
    let identity = TargetIdentity {
        kind: "example",
        name: "dump_embeddinggemma2",
        relative,
    };
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources =
        parsed_module_closure(manifest_dir, &manifest_dir.join(relative), false).unwrap();
    validate_nonmeasurement(&identity, &sources, contract).expect("reviewed correctness purpose");
    sources[0]
        .parsed
        .source
        .push_str("\n// source purpose changed\n");
    assert!(validate_nonmeasurement(&identity, &sources, contract).is_err());
    let sources = fixture_source("examples/unreviewed_module.rs", "fn work() {}");
    assert!(validate_nonmeasurement(&identity, &sources, contract).is_err());
}

#[test]
fn classifier_construction_checks_are_executed_for_purpose_exceptions() {
    let mut contract = NON_MEASUREMENT_TARGETS[4];
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = contract.purpose.path();
    let identity = TargetIdentity {
        kind: "example",
        name: "dump_embeddinggemma2",
        relative: target,
    };
    let sources = parsed_module_closure(root, &root.join(target), false).unwrap();
    validate_nonmeasurement(&identity, &sources, &contract).unwrap();
    contract.constructions += 1;
    assert!(validate_nonmeasurement(&identity, &sources, &contract).is_err());

    let relative = "examples/classifier_fixture.rs";
    let guarded = fixture_source(
        relative,
        r#"
fn main() {
    let gpu = lattice_inference::measurement::gpu_test_lock();
    let state = MetalQwen35State::new();
}
"#,
    );
    assert_eq!(
        validate_target_constructions("example", "classifier_fixture", relative, &guarded).unwrap(),
        1
    );
    let unguarded = fixture_source(
        relative,
        "fn main() { let state = MetalQwen35State::new(); }",
    );
    assert!(
        validate_target_constructions("example", "classifier_fixture", relative, &unguarded)
            .is_err()
    );
}

#[test]
fn classifier_never_admits_measurement_by_table_membership() {
    for row in target_admissions() {
        if !matches!(row.class, TargetClass::Measurement(_)) {
            continue;
        }
        let identity = TargetIdentity {
            kind: if row.path.starts_with("examples/") {
                "example"
            } else {
                "bin"
            },
            name: "classifier_fixture",
            relative: row.path,
        };
        let sources = fixture_source(row.path, "fn main() {}");
        assert!(
            dispatch_target(&identity, &sources, &row).is_err(),
            "{} was accepted without executing protected work validation",
            row.path
        );
    }
}

#[test]
fn state_dispatch_validates_helper_calls_without_requiring_direct_work() {
    let relative = "examples/bench_gdn_prefill_ab.rs";
    let sources = fixture_source(
        relative,
        r#"
fn run() {
    let gpu = lattice_inference::measurement::gpu_test_lock();
    run_one();
}
fn run_one() { state.forward_prefill(); }
"#,
    );
    validate_state_target(relative, &sources).expect("guarded synchronous helper");
    let escaped = fixture_source(
        relative,
        r#"
fn run() { run_one(); }
fn run_one() { state.forward_prefill(); }
"#,
    );
    assert!(validate_state_target(relative, &escaped).is_err());
}

#[test]
fn entrypoint_dispatch_rejects_deferred_calls_and_async_helpers() {
    let relative = "examples/classifier_fixture.rs";
    for (selector, expression) in [
        (DEVICE_SYSTEM_DEFAULT, "Device::system_default()"),
        (
            CallSelector::Path(&["MetalForwardPass", "new"]),
            "MetalForwardPass::new()",
        ),
    ] {
        let direct = format!(
            "fn main() {{ let gpu = {SHARED_LOCK_CALL}; for _ in 0..2 {{ {expression}; }} }}"
        );
        validate_entrypoint(relative, &fixture_source(relative, &direct), selector).unwrap();
        for deferred in [
            format!("std::thread::spawn(|| {{ {expression}; }});"),
            format!("let later = || {{ {expression}; }};"),
            format!("let later = async {{ {expression}; }};"),
            format!("fn later() {{ {expression}; }} later();"),
        ] {
            let source = format!("fn main() {{ let gpu = {SHARED_LOCK_CALL}; {deferred} }}");
            assert!(
                validate_entrypoint(relative, &fixture_source(relative, &source), selector)
                    .is_err(),
                "{deferred}"
            );
        }
    }
    let guarded = format!("fn main() {{ let gpu = {SHARED_LOCK_CALL}; run(); }} fn run() {{}}");
    let selector = CallSelector::Path(&["run"]);
    validate_entrypoint_shape(relative, &fixture_source(relative, &guarded), selector).unwrap();
    let matching_cfg = format!(
        "#[cfg(feature = \"extra-helper\")] {}",
        guarded.replace("fn run()", "#[cfg(feature = \"extra-helper\")] fn run()")
    );
    validate_entrypoint_shape(relative, &fixture_source(relative, &matching_cfg), selector)
        .unwrap();
    assert!(
        validate_entrypoint(relative, &fixture_source(relative, &guarded), selector)
            .unwrap_err()
            .contains("source contract")
    );
    for mutant in [
        guarded.replace("fn run()", "async fn run()"),
        guarded.replace("fn run() {}", "fn run() -> impl FnOnce() { || {} }"),
        guarded.replace("fn run()", "#[cfg(any())] fn run()"),
        guarded.replace("fn run()", "#[cfg(feature = \"extra-helper\")] fn run()"),
        guarded.replace("run();", "let alias = run; alias();"),
        guarded.replace("run();", "std::thread::spawn(|| run());"),
        guarded.replace("run();", "let run = || {}; run();"),
    ] {
        assert!(
            validate_entrypoint_shape(relative, &fixture_source(relative, &mutant), selector)
                .is_err()
        );
    }
}

#[test]
fn state_dispatch_rejects_deferred_work_and_helper_bypasses() {
    let relative = "examples/bench_gdn_prefill_ab.rs";
    let guarded = format!(
        "fn run() {{ let gpu = {SHARED_LOCK_CALL}; run_one(); }} fn run_one() {{ state.forward_prefill(); }}"
    );
    validate_state_target(relative, &fixture_source(relative, &guarded)).unwrap();
    for mutant in [
        guarded.replace("fn run_one()", "async fn run_one()"),
        guarded.replace("fn run_one()", "#[cfg(any())] fn run_one()"),
        guarded.replace(
            "fn run_one()",
            "#[cfg(feature = \"extra-helper\")] fn run_one()",
        ),
        guarded.replace("run_one();", "let alias = run_one; alias();"),
        guarded.replace("run_one();", "std::thread::spawn(|| run_one());"),
        guarded.replace(
            "state.forward_prefill();",
            "std::thread::spawn(|| { state.forward_prefill(); });",
        ),
        guarded.replace(
            "state.forward_prefill();",
            "let later = async { state.forward_prefill(); };",
        ),
        guarded.replace(
            "state.forward_prefill();",
            "fn later() { state.forward_prefill(); } later();",
        ),
        format!("{guarded} fn unguarded() {{ run_one(); }}"),
    ] {
        assert!(
            validate_state_target(relative, &fixture_source(relative, &mutant)).is_err(),
            "{mutant}"
        );
    }
    let relative = "examples/bench_gdn_decode.rs";
    let direct = format!("fn run() {{ let gpu = {SHARED_LOCK_CALL}; state.forward_step(1, 0); }}");
    validate_state_target(relative, &fixture_source(relative, &direct)).unwrap();
    let deferred = direct.replace(
        "state.forward_step(1, 0);",
        "let later = || { state.forward_step(1, 0); };",
    );
    assert!(validate_state_target(relative, &fixture_source(relative, &deferred)).is_err());
}

#[test]
fn state_dispatch_proves_local_closure_calls_and_rejects_escape() {
    let relative = "examples/profile_metal.rs";
    let guarded = r#"
fn main() {
    let gpu = lattice_inference::measurement::gpu_test_lock();
    let measure_prefill = |state: &mut State| { state.forward_prefill(&[1]); };
    let measure_steps = |state: &mut State| { state.forward_step(1, 0); };
    let a = measure_prefill(&mut state);
    let b = measure_prefill(&mut state);
    let c = measure_steps(&mut state);
    let d = measure_steps(&mut state);
    let e = measure_steps(&mut state);
    let f = measure_steps(&mut state);
}
"#;
    validate_state_target(relative, &fixture_source(relative, guarded)).unwrap();
    for mutant in [
        guarded.replace(
            "let a = measure_prefill(&mut state);",
            "let a = std::thread::spawn(|| measure_prefill(&mut state));",
        ),
        guarded.replace(
            "let a = measure_prefill(&mut state);",
            "let alias = measure_prefill; let a = measure_prefill(&mut state);",
        ),
        guarded.replace(
            "let a = measure_prefill(&mut state);",
            "drop(gpu); let a = measure_prefill(&mut state);",
        ),
        guarded.replace(
            "state.forward_step(1, 0);",
            "std::thread::spawn(|| { state.forward_step(1, 0); });",
        ),
        guarded.replace("|state: &mut State|", "async |state: &mut State|"),
        guarded.replace("let a = measure_prefill(&mut state);", ""),
    ] {
        assert!(
            validate_state_target(relative, &fixture_source(relative, &mutant)).is_err(),
            "{mutant}"
        );
    }
}

#[test]
fn state_dispatch_only_accepts_the_immediate_autoreleasepool_callback() {
    let relative = "examples/rss_autorelease_probe.rs";
    let guarded = r#"
fn run_decode_arm() {
    let gpu = lattice_inference::measurement::gpu_test_lock();
    objc::rc::autoreleasepool(|| { state.forward_step(1, 0); });
}
"#;
    validate_state_target(relative, &fixture_source(relative, guarded)).unwrap();
    for mutant in [
        guarded.replace("objc::rc::autoreleasepool", "std::thread::spawn"),
        guarded.replace("|| { state", "async || { state"),
        guarded.replace(
            "state.forward_step(1, 0);",
            "let later = || { state.forward_step(1, 0); };",
        ),
        guarded
            .replace(
                "objc::rc::autoreleasepool(||",
                "std::thread::spawn(|| objc::rc::autoreleasepool(||",
            )
            .replace("; });", "; }));"),
        format!("mod objc {{}} {guarded}"),
        format!("use alternate::*; {guarded}"),
    ] {
        assert!(
            validate_state_target(relative, &fixture_source(relative, &mutant)).is_err(),
            "{mutant}"
        );
    }
}

#[test]
fn named_entrypoint_contracts_reject_empty_and_detached_helper_bodies() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let rows = target_admissions();
    for contract in ENTRYPOINT_HELPERS {
        let source = std::fs::read_to_string(root.join(contract.path)).unwrap();
        let row = rows.iter().find(|row| row.path == contract.path).unwrap();
        let identity = TargetIdentity {
            kind: "example",
            name: "named_entrypoint_fixture",
            relative: contract.path,
        };
        dispatch_target(&identity, &fixture_source(contract.path, &source), row)
            .expect("reviewed synchronous source and complete work census");
        let syntax = syn::parse_file(&source).unwrap();
        let function = syntax
            .items
            .iter()
            .find_map(|item| match item {
                syn::Item::Fn(function) if function.sig.ident == contract.helper => Some(function),
                _ => None,
            })
            .unwrap();
        let body_range = function.block.span().byte_range();
        let original_body = &source[body_range.clone()];
        let return_value = if matches!(function.sig.output, syn::ReturnType::Default) {
            ""
        } else {
            "Ok(())"
        };
        let mut detached = source.clone();
        detached.replace_range(
            body_range.clone(),
            &format!("{{ std::thread::spawn(move || {original_body}); {return_value} }}"),
        );
        let mut empty = source.clone();
        empty.replace_range(body_range.clone(), &format!("{{ {return_value} }}"));
        let original_function =
            &source[function.sig.fn_token.span.byte_range().start..body_range.end];
        empty.push_str("\n#[cfg(any())]\n");
        empty.push_str(&original_function.replacen(
            &format!("fn {}", contract.helper),
            "fn retained_metal_markers",
            1,
        ));
        for mutant in [detached, empty] {
            let sources = fixture_source(contract.path, &mutant);
            assert!(sources[0].parsed.has_top_level_metal_marker().unwrap());
            let Err(reason) = dispatch_target(&identity, &sources, row) else {
                panic!("changed helper body must refuse");
            };
            assert!(reason.contains("source provenance"));
        }
    }
}
