#[path = "support/metal_qwen35_population.rs"]
#[allow(dead_code)]
mod metal_qwen35_population;
#[path = "support/metal_source_guards.rs"]
#[allow(dead_code)]
mod metal_source_guards;
#[path = "support/source_graph.rs"]
#[allow(dead_code)]
mod source_graph;

use metal_source_guards::{CallSpec, GuardPopulation, OwnerKind, OwnerSpec, Scope};
use proc_macro2::TokenTree;
use std::collections::BTreeSet;
use std::path::Path;
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{Expr, Lit, Token};

const STATE: &str = "crate::forward::metal_qwen35::MetalQwen35State";
const SESSION: &str = "crate::decoder::qwen_metal::QwenMetalSession";

const fn function(id: &'static str, path: &'static str) -> OwnerSpec {
    OwnerSpec {
        id,
        path,
        kind: OwnerKind::Function,
        scope: Scope::Metal,
    }
}

const fn method(id: &'static str, self_ty: &'static str, path: &'static str) -> OwnerSpec {
    OwnerSpec {
        id,
        path,
        kind: OwnerKind::InherentMethod { self_ty },
        scope: Scope::Metal,
    }
}

const fn state(id: &'static str, path: &'static str) -> OwnerSpec {
    method(id, STATE, path)
}

const fn session(id: &'static str, path: &'static str) -> OwnerSpec {
    method(id, SESSION, path)
}

const fn trait_method(
    id: &'static str,
    self_ty: &'static str,
    trait_path: &'static str,
    path: &'static str,
) -> OwnerSpec {
    OwnerSpec {
        id,
        path,
        kind: OwnerKind::TraitMethod {
            self_ty,
            trait_path,
        },
        scope: Scope::Metal,
    }
}

const OWNERS: &[OwnerSpec] = &[
    function(
        "sample",
        "crate::forward::metal_qwen35::sample_decode_traced",
    ),
    state("multimodal", "generate_multimodal"),
    state("vision", "generate_multimodal_vision_impl"),
    trait_method(
        "select",
        SESSION,
        "crate::decoder::DecoderSession",
        "select",
    ),
    state("mtp", "generate_greedy_mtp"),
    function(
        "batch_guard",
        "crate::forward::metal_qwen35::use_batch_gemm_verifier",
    ),
    function(
        "pipelines",
        "crate::forward::metal_qwen35::inner::build_pipelines",
    ),
    method(
        "engine_new",
        "crate::forward::metal_qwen35::inner::MetalQwen35Engine",
        "new",
    ),
    state("q4", "from_q4_dir"),
    state("gqa", "encode_gqa_layer"),
    state("lora", "dispatch_lora_if_active"),
    state("mlp", "encode_mlp_block"),
    state("generate", "generate"),
    state(
        "prefix",
        "generate_streaming_with_prefix_cache_and_cancel_inner",
    ),
    state("stream", "generate_streaming_with_cancel"),
    function("plan", "crate::forward::metal_qwen35::plan_sampling_route"),
    function(
        "apply",
        "crate::forward::metal_qwen35::apply_sampling_route_plan",
    ),
    function(
        "block_route",
        "crate::forward::metal_qwen35::inner::choose_gpu_block_topk_route",
    ),
    session("construct", "construct"),
    session("handoff", "with_route_environment"),
    session("restored", "over_restored_state"),
    session("session_new", "new"),
    state("injected", "forward_step_inner_impl_dispatch"),
    state("verify_dispatch", "verify_tokens_batch_gemm_dispatch"),
    state("prefill_dispatch", "forward_prefill_batched_chunk_dispatch"),
    state("head", "encode_final_head"),
    state("capture", "encode_capture_final_hidden"),
    state("try_step", "try_forward_step"),
    state("hidden_step", "forward_step_with_hidden"),
    state("step", "forward_step"),
    state("try_prefill", "try_forward_prefill"),
    state("prefill", "forward_prefill"),
    state("score", "score_option_letters"),
    state("hidden_prefill", "forward_prefill_with_hidden"),
    state("all_logits", "forward_prefill_all_logits"),
    state("prefill_from", "forward_prefill_from"),
    state("live_cursor", "check_live_cursor"),
    state("raw_fresh", "check_raw_prefill_fresh_session"),
    state("hidden_fresh", "check_hidden_prefill_fresh_session"),
    state("reject_moe", "reject_moe_batched"),
    state("rollback_state", "rollback_speculative_state_to"),
    trait_method(
        "verify",
        STATE,
        "crate::speculative::MtpTargetVerifier",
        "verify_tokens",
    ),
    trait_method(
        "rollback",
        STATE,
        "crate::speculative::MtpTargetVerifier",
        "rollback_cache_to",
    ),
    method(
        "prepare_hidden",
        "crate::forward::metal_qwen35::inner::bench_support::LmHeadBenchFixture",
        "prepare_hidden_for_bench",
    ),
    function(
        "isolated",
        "crate::forward::metal_qwen35::inner::bench_support::forward_prefill_gdn_isolated_chunk",
    ),
    function(
        "isolated_capture",
        "crate::forward::metal_qwen35::inner::bench_support::forward_prefill_gdn_isolated_chunk_capture",
    ),
    function(
        "production_chunk",
        "crate::forward::metal_qwen35::inner::bench_support::forward_prefill_production_chunk",
    ),
    function(
        "bench_preflight",
        "crate::forward::metal_qwen35::inner::bench_support::preflight_bench_prefill",
    ),
    trait_method("drop", SESSION, "std::ops::Drop", "drop"),
    trait_method(
        "finish",
        SESSION,
        "crate::decoder::DecoderSession",
        "finish",
    ),
    session("disengage", "disengage_route"),
    state("disengage_compact", "disengage_compact_route"),
];

const fn call(caller: &'static str, callee: &'static str, count: usize) -> CallSpec {
    CallSpec {
        caller,
        callee,
        count,
    }
}

const SAMPLING_CALLS: &[CallSpec] = &[
    call("multimodal", "sample", 1),
    call("vision", "sample", 1),
    call("select", "sample", 2),
];
const PIPELINE_CALLS: &[CallSpec] = &[
    call("engine_new", "pipelines", 1),
    call("q4", "pipelines", 1),
];
const ROUTE_CALLS: &[CallSpec] = &[
    call("generate", "plan", 1),
    call("construct", "plan", 1),
    call("generate", "apply", 1),
    call("construct", "apply", 1),
    call("plan", "block_route", 1),
];
const PREFLIGHT_CALLS: &[CallSpec] = &[
    call("try_step", "live_cursor", 1),
    call("hidden_step", "live_cursor", 1),
    call("step", "try_step", 1),
    call("try_prefill", "raw_fresh", 1),
    call("prefill", "try_prefill", 1),
    call("score", "try_prefill", 1),
    call("hidden_prefill", "hidden_fresh", 1),
    call("all_logits", "raw_fresh", 1),
    call("prefill_from", "reject_moe", 1),
    call("generate", "handoff", 1),
    call("stream", "session_new", 1),
    call("prefix", "restored", 1),
    call("verify", "live_cursor", 1),
    call("rollback", "rollback_state", 1),
    call("prepare_hidden", "step", 1),
    call("isolated", "bench_preflight", 1),
    call("isolated_capture", "bench_preflight", 1),
    call("production_chunk", "bench_preflight", 1),
];
const RETIRED: &[&str] = &[
    "full_attention_layer_step_by_idx",
    "gdn_layer_step_by_idx",
    "gdn_cpu_recurrence",
    "append_kv",
    "dispatch_add",
    "softplus",
];

fn population() -> GuardPopulation {
    GuardPopulation::load(Path::new(env!("CARGO_MANIFEST_DIR")), OWNERS)
        .expect("guarded owners and production sources must reconcile")
}

fn body(population: &GuardPopulation, id: &str) -> syn::Block {
    syn::parse_str(
        population
            .owner_body(id)
            .expect("declared owner must resolve"),
    )
    .expect("production owner body must parse")
}

fn count(population: &GuardPopulation, caller: &str, callee: &str, expected: usize) {
    assert_eq!(
        population
            .call_sites(caller, callee)
            .expect("calls must resolve")
            .len(),
        expected,
        "{caller} must have {expected} calls to {callee}"
    );
}

fn path(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(expr) if expr.qself.is_none() => Some(source_graph::path_label(&expr.path)),
        Expr::Paren(expr) => path(&expr.expr),
        Expr::Group(expr) => path(&expr.expr),
        _ => None,
    }
}

fn string(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Lit(expr) => match &expr.lit {
            Lit::Str(value) => Some(value.value()),
            _ => None,
        },
        _ => None,
    }
}

fn path_ends(expr: &Expr, suffix: &str) -> bool {
    path(expr).is_some_and(|label| label == suffix || label.ends_with(&format!("::{suffix}")))
}

fn expression_list(tokens: proc_macro2::TokenStream, context: &str) -> Vec<Expr> {
    Punctuated::<Expr, Token![,]>::parse_terminated
        .parse2(tokens)
        .unwrap_or_else(|error| panic!("{context} must parse: {error}"))
        .into_iter()
        .collect()
}

fn reference_to(expr: &Expr, name: &str) -> bool {
    matches!(expr, Expr::Reference(reference) if path(&reference.expr).as_deref() == Some(name))
}

#[derive(Default)]
struct Calls<'a> {
    functions: Vec<&'a syn::ExprCall>,
    methods: Vec<&'a syn::ExprMethodCall>,
}

impl<'ast> Visit<'ast> for Calls<'ast> {
    fn visit_expr_call(&mut self, expr: &'ast syn::ExprCall) {
        self.functions.push(expr);
        visit::visit_expr_call(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        self.methods.push(expr);
        visit::visit_expr_method_call(self, expr);
    }
}

fn calls(block: &syn::Block) -> Calls<'_> {
    let mut result = Calls::default();
    result.visit_block(block);
    result
}

#[derive(Default)]
struct Functions<'a> {
    items: Vec<(&'a syn::Signature, &'a syn::Block, &'a syn::Visibility)>,
}

impl<'ast> Visit<'ast> for Functions<'ast> {
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.items.push((&item.sig, &item.block, &item.vis));
        visit::visit_item_fn(self, item);
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.items.push((&item.sig, &item.block, &item.vis));
        visit::visit_impl_item_fn(self, item);
    }
}

fn definitions(population: &GuardPopulation, name: &str) -> usize {
    population
        .production_sources()
        .map(|(_, source)| {
            let syntax = syn::parse_file(source).expect("production view must parse");
            let mut functions = Functions::default();
            functions.visit_file(&syntax);
            functions
                .items
                .iter()
                .filter(|(sig, _, _)| sig.ident == name)
                .count()
        })
        .sum()
}

fn has_named_call(block: &syn::Block, name: &str) -> bool {
    let calls = calls(block);
    calls
        .functions
        .iter()
        .any(|call| path_ends(&call.func, name))
        || calls.methods.iter().any(|call| call.method == name)
}

#[test]
fn sample_decode_traced_opens_the_decode_sample_interval() {
    let population = population();
    let block = body(&population, "sample");
    assert!(
        calls(&block).functions.iter().any(|call| {
            path(&call.func).as_deref() == Some("crate::forward::signpost::interval")
                && call.args.len() == 1
                && path(call.args.first().expect("one argument")).as_deref()
                    == Some("crate::forward::signpost::Label::DecodeSample")
        }),
        "the shared sampler must open the decode sample interval"
    );
}

#[test]
fn mtp_verify_phase_selector_goes_through_adapter_guard() {
    let population = population();
    let mut readers = 0;
    for (_, source) in population.production_sources() {
        let syntax = syn::parse_file(source).expect("production source must parse");
        let mut scan = Calls::default();
        scan.visit_file(&syntax);
        readers += scan
            .functions
            .iter()
            .filter(|call| {
                path_ends(&call.func, "env_switch_enabled")
                    && call.args.first().and_then(string).as_deref() == Some("LATTICE_MTP_BATCH")
            })
            .count();
    }
    assert_eq!(
        readers, 1,
        "the batch verifier switch must have one production reader"
    );
    count(&population, "mtp", "batch_guard", 1);
    let block = body(&population, "mtp");
    assert!(
        calls(&block).functions.iter().any(|guard| {
            path_ends(&guard.func, "use_batch_gemm_verifier")
                && guard.args.iter().any(|arg| {
                    matches!(arg, Expr::Call(read)
                if path_ends(&read.func, "env_switch_enabled")
                    && read.args.first().and_then(string).as_deref() == Some("LATTICE_MTP_BATCH"))
                })
        }),
        "the switch read must be an argument of the adapter-aware verifier guard"
    );
}

#[test]
fn all_five_decode_loops_call_sample_decode_traced() {
    let population = population();
    assert_eq!(definitions(&population, "sample_decode_traced"), 1);
    population
        .require_calls(SAMPLING_CALLS)
        .expect("every decode sampler caller must match its manifest");
    let block = body(&population, "select");
    struct ReadbackArms(BTreeSet<String>);
    impl<'ast> Visit<'ast> for ReadbackArms {
        fn visit_arm(&mut self, arm: &'ast syn::Arm) {
            let name = match &arm.pat {
                syn::Pat::TupleStruct(pat) => Some(source_graph::path_label(&pat.path)),
                syn::Pat::Path(pat) => Some(source_graph::path_label(&pat.path)),
                _ => None,
            };
            if let Some(name) = name {
                let mut nested = Calls::default();
                nested.visit_expr(&arm.body);
                if nested
                    .functions
                    .iter()
                    .any(|call| path_ends(&call.func, "sample_decode_traced"))
                {
                    self.0.insert(name);
                }
            }
            visit::visit_arm(self, arm);
        }
    }
    let mut arms = ReadbackArms(BTreeSet::new());
    arms.visit_block(&block);
    assert_eq!(
        arms.0,
        BTreeSet::from(["Readback::Dense".into(), "Readback::Compact".into()])
    );
}

#[derive(Default)]
struct RetiredPipeline {
    present: bool,
}

impl<'ast> Visit<'ast> for RetiredPipeline {
    fn visit_field(&mut self, field: &'ast syn::Field) {
        if field.ident.as_ref().is_some_and(|ident| ident == "add")
            && matches!(&field.ty, syn::Type::Path(ty)
                if ty.path.segments.last().is_some_and(|segment| segment.ident == "ComputePipelineState"))
        {
            self.present = true;
        }
        visit::visit_field(self, field);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if path_ends(&call.func, "make_pipeline")
            && call.args.first().and_then(string).as_deref() == Some("add_buf")
        {
            self.present = true;
        }
        visit::visit_expr_call(self, call);
    }
}

fn assembled_shader(population: &GuardPopulation) -> String {
    #[derive(Default)]
    struct Sources<'a>(Vec<&'a syn::ItemConst>);
    impl<'ast> Visit<'ast> for Sources<'ast> {
        fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
            if item.ident == "MSL_SOURCE" {
                self.0.push(item);
            }
            visit::visit_item_const(self, item);
        }
    }
    let mut assemblies = Vec::new();
    for (file, source) in population.production_sources() {
        let syntax = syn::parse_file(source).expect("production source must parse");
        let mut constants = Sources::default();
        constants.visit_file(&syntax);
        for constant in constants.0 {
            let Expr::Macro(mac) = constant.expr.as_ref() else {
                panic!("main shader source must have an explicit assembly");
            };
            assert!(
                mac.mac.path.is_ident("concat"),
                "main shader assembly must be explicit"
            );
            let fragments = expression_list(mac.mac.tokens.clone(), "shader assembly arguments");
            assert!(
                !fragments.is_empty(),
                "shader assembly must contain source fragments"
            );
            let mut assembly = String::new();
            for fragment in fragments {
                let Expr::Macro(include) = fragment else {
                    panic!("shader assembly must name its source fragments");
                };
                assert!(include.mac.path.is_ident("include_str"));
                let relative = syn::parse2::<syn::LitStr>(include.mac.tokens)
                    .expect("shader fragment path must be literal");
                let owner = Path::new(env!("CARGO_MANIFEST_DIR")).join(file);
                let path = owner
                    .parent()
                    .expect("shader owner has a directory")
                    .join(relative.value());
                assembly.push_str(&std::fs::read_to_string(&path).unwrap_or_else(|error| {
                    panic!("cannot read shader {}: {error}", path.display())
                }));
            }
            assemblies.push(assembly);
        }
    }
    assert_eq!(
        assemblies.len(),
        1,
        "main shader assembly must have one owner"
    );
    assemblies.pop().expect("one shader assembly")
}

#[test]
fn retired_closed_call_chains_stay_deleted() {
    let population = population();
    population
        .reject_declared_functions(RETIRED)
        .expect("retired declarations must remain absent");
    for (file, source) in population.production_sources() {
        let syntax = syn::parse_file(source).expect("production source must parse");
        let mut retired = RetiredPipeline::default();
        retired.visit_file(&syntax);
        assert!(
            !retired.present,
            "retired standalone add pipeline is present in {file}"
        );
    }
    assert!(
        !assembled_shader(&population).contains("kernel void add_buf("),
        "the retired standalone add kernel must remain absent from the assembled shader"
    );

    let block = body(&population, "injected");
    assert!(
        calls(&block).functions.iter().any(|call| {
            path_ends(&call.func, "env_switch_enabled")
                && call.args.first().and_then(string).as_deref() == Some("LATTICE_GDN_CPU")
        }),
        "the release decode switch refusal must remain"
    );
    #[derive(Default)]
    struct Refusal(bool);
    impl<'ast> Visit<'ast> for Refusal {
        fn visit_macro(&mut self, mac: &'ast syn::Macro) {
            if mac.path.is_ident("panic") {
                let args = expression_list(mac.tokens.clone(), "panic arguments");
                self.0 |= args.first().and_then(string).as_deref()
                    == Some("LATTICE_GDN_CPU=1 is debug-only and cannot be used in release decode");
            }
        }
    }
    let mut refusal = Refusal::default();
    refusal.visit_block(&block);
    assert!(
        refusal.0,
        "release decode must retain the explicit GDN CPU refusal"
    );

    for name in [
        "encode_gqa_layer",
        "dispatch_lora_if_active",
        "encode_mlp_block",
    ] {
        assert_eq!(
            definitions(&population, name),
            1,
            "{name} must have one production owner"
        );
    }
    let gqa = population.owner("gqa").expect("GQA owner");
    for helper in ["lora", "mlp"] {
        assert_ne!(
            gqa.file,
            population.owner(helper).expect("shared helper owner").file,
            "the GQA encoder must not duplicate its shared helpers"
        );
    }
    assert_eq!(
        population.owner("lora").unwrap().file,
        population.owner("mlp").unwrap().file,
        "shared layer helpers must retain one common owner module"
    );
    count(&population, "gqa", "lora", 4);
    count(&population, "gqa", "mlp", 1);
    let block = body(&population, "gqa");
    let projections = calls(&block)
        .methods
        .iter()
        .filter(|call| call.method == "dispatch_lora_if_active")
        .map(|call| {
            call.args
                .last()
                .and_then(string)
                .expect("projection must be named")
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        projections,
        BTreeSet::from([
            "q_proj".into(),
            "k_proj".into(),
            "v_proj".into(),
            "o_proj".into()
        ])
    );
}

#[test]
fn metal_constructors_share_one_pipeline_builder() {
    let population = population();
    assert_eq!(definitions(&population, "build_pipelines"), 1);
    population
        .require_calls(PIPELINE_CALLS)
        .expect("pipeline constructors must match their manifest");
    #[derive(Default)]
    struct PrivateBuilder(bool);
    impl<'ast> Visit<'ast> for PrivateBuilder {
        fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
            self.0 |= call.method == "new_library_with_source";
            visit::visit_expr_method_call(self, call);
        }
        fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
            self.0 |= pat.ident == "make_pipeline";
            visit::visit_pat_ident(self, pat);
        }
        fn visit_expr_struct(&mut self, expr: &'ast syn::ExprStruct) {
            self.0 |= expr
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "MetalQwen35Pipelines");
            visit::visit_expr_struct(self, expr);
        }
    }
    for owner in ["engine_new", "q4"] {
        let block = body(&population, owner);
        let mut private = PrivateBuilder::default();
        private.visit_block(&block);
        assert!(
            !private.0,
            "{owner} must not contain a private pipeline builder"
        );
        struct BuildCall(usize);
        impl<'ast> Visit<'ast> for BuildCall {
            fn visit_expr_try(&mut self, expr: &'ast syn::ExprTry) {
                if let Expr::Call(call) = expr.expr.as_ref()
                    && path_ends(&call.func, "build_pipelines")
                    && call.args.len() == 2
                    && reference_to(&call.args[0], "device")
                    && path(&call.args[1]).as_deref() == Some("cfg")
                {
                    self.0 += 1;
                }
                visit::visit_expr_try(self, expr);
            }
        }
        let mut build = BuildCall(0);
        build.visit_block(&block);
        assert_eq!(
            build.0, 1,
            "{owner} must propagate the shared builder result for device and cfg"
        );
    }
    let mut compilations = 0;
    for (_, source) in population.production_sources() {
        let syntax = syn::parse_file(source).expect("production source must parse");
        let mut scan = Calls::default();
        scan.visit_file(&syntax);
        compilations += scan
            .methods
            .iter()
            .filter(|call| {
                call.method == "new_library_with_source"
                    && call
                        .args
                        .first()
                        .is_some_and(|arg| path_ends(arg, "MSL_SOURCE"))
            })
            .count();
    }
    assert_eq!(
        compilations, 1,
        "the main shader must have one production compilation site"
    );
    let block = body(&population, "pipelines");
    assert_eq!(
        calls(&block)
            .methods
            .iter()
            .filter(|call| call.method == "new_library_with_source"
                && call.args.len() == 2
                && reference_to(&call.args[1], "opts")
                && call
                    .args
                    .first()
                    .is_some_and(|arg| path_ends(arg, "MSL_SOURCE")))
            .count(),
        1
    );
}

#[test]
fn sampling_route_configurator_owns_all_three_generation_call_sites() {
    let population = population();
    population
        .reject_declared_functions(&["configure_sampling_route"])
        .expect("the retired sampling configurator must remain absent");
    for (_, source) in population.production_sources() {
        let syntax = syn::parse_file(source).expect("production source must parse");
        struct ConfiguratorName(bool);
        impl<'ast> Visit<'ast> for ConfiguratorName {
            fn visit_ident(&mut self, ident: &'ast proc_macro2::Ident) {
                self.0 |= ident == "configure_sampling_route";
            }
            fn visit_macro(&mut self, mac: &'ast syn::Macro) {
                let mut names = BTreeSet::new();
                token_identifiers(mac.tokens.clone(), &mut names);
                self.0 |= names.contains("configure_sampling_route");
            }
        }
        let mut scan = ConfiguratorName(false);
        scan.visit_file(&syntax);
        assert!(
            !scan.0,
            "the retired sampling configurator must have no production references"
        );
    }
    population
        .require_calls(ROUTE_CALLS)
        .expect("route helper callers must match their manifest");
    count(&population, "prefix", "plan", 0);
    count(&population, "prefix", "apply", 0);
    count(&population, "prefix", "restored", 1);
    count(&population, "generate", "handoff", 1);
    let block = body(&population, "plan");
    struct RouteBinding(bool);
    impl<'ast> Visit<'ast> for RouteBinding {
        fn visit_local(&mut self, local: &'ast syn::Local) {
            if matches!(&local.pat, syn::Pat::Ident(pat) if pat.ident == "block_route")
                && local.init.as_ref().is_some_and(|init| {
                    matches!(init.expr.as_ref(), Expr::Call(call)
                    if path_ends(&call.func, "choose_gpu_block_topk_route"))
                })
            {
                self.0 = true;
            }
            visit::visit_local(self, local);
        }
    }
    let mut binding = RouteBinding(false);
    binding.visit_block(&block);
    assert!(
        binding.0,
        "the planner must own the block route composition"
    );
}

fn token_identifiers(tokens: proc_macro2::TokenStream, names: &mut BTreeSet<String>) {
    for token in tokens {
        match token {
            TokenTree::Ident(ident) => {
                names.insert(ident.to_string());
            }
            TokenTree::Group(group) => token_identifiers(group.stream(), names),
            _ => {}
        }
    }
}

#[test]
fn injected_embedding_never_falls_back_to_token_lookup() {
    let population = population();
    let block = body(&population, "injected");
    struct Injection {
        arms: usize,
        forbidden: BTreeSet<String>,
    }
    impl<'ast> Visit<'ast> for Injection {
        fn visit_expr_match(&mut self, expr: &'ast syn::ExprMatch) {
            if path(&expr.expr).as_deref() == Some("injected_embedding") {
                for arm in &expr.arms {
                    if matches!(&arm.pat, syn::Pat::TupleStruct(pat)
                        if pat.path.is_ident("Some") && pat.elems.len() == 1)
                    {
                        self.arms += 1;
                        struct Names(BTreeSet<String>);
                        impl<'ast> Visit<'ast> for Names {
                            fn visit_ident(&mut self, ident: &'ast proc_macro2::Ident) {
                                self.0.insert(ident.to_string());
                            }
                            fn visit_macro(&mut self, mac: &'ast syn::Macro) {
                                token_identifiers(mac.tokens.clone(), &mut self.0);
                            }
                        }
                        let mut names = Names(BTreeSet::new());
                        names.visit_expr(&arm.body);
                        self.forbidden.extend(
                            names
                                .0
                                .into_iter()
                                .filter(|name| name == "embed_tokens" || name == "token_id"),
                        );
                    }
                }
            }
            visit::visit_expr_match(self, expr);
        }
    }
    let mut injected = Injection {
        arms: 0,
        forbidden: BTreeSet::new(),
    };
    injected.visit_block(&block);
    assert_eq!(injected.arms, 1, "the injected row arm must have one owner");
    assert!(
        injected.forbidden.is_empty(),
        "the injected row must replace token lookup: {:?}",
        injected.forbidden
    );
}

fn field_chain(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Field(expr) => {
            let syn::Member::Named(member) = &expr.member else {
                return None;
            };
            Some(format!("{}::{member}", field_chain(&expr.base)?))
        }
        _ => path(expr),
    }
}

fn final_norm_argument(expr: &Expr) -> bool {
    match expr {
        Expr::Reference(reference) => {
            field_chain(&reference.expr).as_deref() == Some("self::engine::final_norm")
        }
        Expr::Call(call) if path_ends(&call.func, "Some") && call.args.len() == 1 => {
            final_norm_argument(call.args.first().expect("one argument"))
        }
        _ => false,
    }
}

#[test]
fn hidden_state_every_final_norm_dispatch_is_followed_by_a_capture() {
    let population = population();
    for owner in ["verify_dispatch", "prefill_dispatch", "head"] {
        count(&population, owner, "capture", 1);
    }
    let mut total = 0;
    let mut uncovered = Vec::new();
    for (file, source) in population.production_sources() {
        let syntax = syn::parse_file(source).expect("production source must parse");
        let mut functions = Functions::default();
        functions.visit_file(&syntax);
        for (sig, block, _) in functions.items {
            let scan = calls(block);
            let mut sites = Vec::new();
            for call in &scan.methods {
                sites.extend(
                    call.args
                        .iter()
                        .filter(|arg| final_norm_argument(arg))
                        .map(|arg| arg.span().start().line),
                );
            }
            for call in &scan.functions {
                if !path_ends(&call.func, "Some") {
                    sites.extend(
                        call.args
                            .iter()
                            .filter(|arg| final_norm_argument(arg))
                            .map(|arg| arg.span().start().line),
                    );
                }
            }
            let captures = scan
                .methods
                .iter()
                .filter(|call| call.method == "encode_capture_final_hidden")
                .map(|call| call.span().start().line)
                .collect::<Vec<_>>();
            total += sites.len();
            for site in sites {
                if !captures
                    .iter()
                    .any(|line| *line >= site && *line < site + 14)
                {
                    uncovered.push(format!("{file}:{}:{site}", sig.ident));
                }
            }
        }
    }
    assert!(
        total >= 3,
        "expected at least three final-norm argument sites, found {total}"
    );
    assert!(
        uncovered.is_empty(),
        "final-norm dispatches lack a capture within fourteen lines: {uncovered:?}"
    );
}

#[test]
fn source_consistency_call_matching_ignores_comments_and_strings() {
    for source in [
        "{ // self.check_live_cursor();\n }",
        "{ let text = \"self.check_live_cursor();\"; }",
        "{ let text = r###\"self.check_live_cursor(); mod tests {\"###; }",
        "{ /* self.check_live_cursor(); */ }",
    ] {
        let block = syn::parse_str(source).expect("decoy block must parse");
        assert!(!has_named_call(&block, "check_live_cursor"));
    }
    let block =
        syn::parse_str("{ self.check_live_cursor(); }").expect("real call block must parse");
    assert!(has_named_call(&block, "check_live_cursor"));
}

#[test]
fn contract_test_bodies_avoid_syn_token_macros() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("metal_qwen35_source_contract.rs");
    let source = std::fs::read_to_string(&path).expect("read the contract source");
    let syntax = syn::parse_file(&source).expect("contract source must parse");
    let mut functions = Vec::new();
    source_graph::module_functions(&syntax.items, &mut functions);
    let mut offenders = Vec::new();
    scan_for_token_macros(&functions, &mut offenders);
    assert!(
        offenders.is_empty(),
        "test bodies must use Token in file-level helpers: {offenders:?}"
    );
}

fn scan_for_token_macros(functions: &[&syn::ItemFn], offenders: &mut Vec<String>) {
    #[derive(Default)]
    struct TokenMacroScan(bool);

    impl<'ast> Visit<'ast> for TokenMacroScan {
        fn visit_macro(&mut self, mac: &'ast syn::Macro) {
            self.0 |= mac
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "Token");
            visit::visit_macro(self, mac);
        }
    }

    for function in functions {
        if !function
            .attrs
            .iter()
            .any(|attr| attr.path().is_ident("test"))
        {
            continue;
        }
        let mut found = TokenMacroScan::default();
        found.visit_block(function.block.as_ref());
        if found.0 {
            offenders.push(function.sig.ident.to_string());
        }
    }
}

#[test]
fn every_public_stateful_scheduling_entry_point_has_preflight() {
    let population = population();
    let sinks = [
        "forward_step_inner",
        "forward_prefill_impl",
        "forward_prefill_batched_chunk",
        "forward_prefill_chunk_gdn_isolated",
        "verify_tokens_batched",
    ];
    let mut discovered = BTreeSet::new();
    for (_, source) in population.production_sources() {
        let syntax = syn::parse_file(source).expect("production source must parse");
        let mut functions = Functions::default();
        functions.visit_file(&syntax);
        for (sig, block, visibility) in functions.items {
            if !matches!(visibility, syn::Visibility::Public(_)) {
                continue;
            }
            let position = sig.inputs.iter().any(|input| {
                matches!(input,
                syn::FnArg::Typed(arg) if matches!(arg.pat.as_ref(), syn::Pat::Ident(pat)
                    if pat.ident == "position" || pat.ident == "start_pos")
                    && matches!(arg.ty.as_ref(), syn::Type::Path(ty) if ty.path.is_ident("usize")))
            });
            let name = sig.ident.to_string();
            if name.starts_with("forward_prefill")
                || position
                || sinks.iter().any(|sink| has_named_call(block, sink))
            {
                discovered.insert(name);
            }
        }
    }
    discovered.extend(["rollback_cache_to".into(), "verify_tokens".into()]);
    let expected = [
        "forward_prefill",
        "forward_prefill_all_logits",
        "forward_prefill_gdn_isolated_chunk",
        "forward_prefill_gdn_isolated_chunk_capture",
        "forward_prefill_production_chunk",
        "forward_prefill_with_hidden",
        "forward_step",
        "forward_step_with_hidden",
        "prepare_hidden_for_bench",
        "rollback_cache_to",
        "try_forward_prefill",
        "try_forward_step",
        "verify_tokens",
    ];
    assert_eq!(
        discovered,
        expected.into_iter().map(str::to_owned).collect(),
        "public scheduling population changed"
    );
    for obligation in PREFLIGHT_CALLS {
        count(
            &population,
            obligation.caller,
            obligation.callee,
            obligation.count,
        );
    }
    let block = body(&population, "generate");
    struct CompactBranch(bool);
    impl<'ast> Visit<'ast> for CompactBranch {
        fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
            self.0 |= path(&expr.cond).as_deref() == Some("use_compact");
            visit::visit_expr_if(self, expr);
        }
    }
    let mut compact = CompactBranch(false);
    compact.visit_block(&block);
    assert!(
        !compact.0,
        "direct generation must leave compact routing to the session"
    );
    count(&population, "generate", "disengage_compact", 0);
    count(&population, "drop", "disengage", 1);
    count(&population, "finish", "disengage", 1);
    count(&population, "disengage", "disengage_compact", 1);
}
