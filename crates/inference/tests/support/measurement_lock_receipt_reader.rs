use super::*;

// A receipt reader may name the lock only as protocol data; it does not acquire it.
const RECEIPT_READER_PACKAGE: &str = "../serve";
const RECEIPT_READER_PATH: &str = "examples/bench_serve_http.rs";
pub(super) const RECEIPT_READER_REASON: &str =
    "reads a supervisor receipt; does not acquire the GPU lock";

fn validate_receipt_reader(source: &str, lock_path: &str) -> Result<(), String> {
    let tokens = rust_tokens(source)?;
    let mut covered = 0;
    for token in &tokens {
        if let RustTokenKind::String(value) = &token.kind {
            let count = value.matches(lock_path).count();
            if count != 0 {
                let expected = format!("lock=Metal GPU ({lock_path})");
                if count != 1 || !value.starts_with(&expected) {
                    return Err("GPU lock path is not a receipt literal".into());
                }
                covered += count;
            }
        }
        if matches!(token.ident(), Some("gpu_test_lock" | "flock")) {
            return Err("receipt reader contains GPU lock acquisition".into());
        }
    }
    if covered == 0 || covered != source.matches(lock_path).count() {
        return Err("missing receipt path or unclassified path occurrence".into());
    }

    // The example reads the receipt file and its Linux test opens a temporary
    // executable. No other open is admitted without updating this contract.
    struct Opens {
        current: Option<String>,
        error: Option<&'static str>,
    }
    impl<'ast> Visit<'ast> for Opens {
        fn visit_item_fn(&mut self, item: &'ast ItemFn) {
            let old = self.current.replace(item.sig.ident.to_string());
            syn::visit::visit_item_fn(self, item);
            self.current = old;
        }
        fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
            if let syn::Expr::Path(path) = &*call.func {
                let names: Vec<_> = path
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                if names.ends_with(&["File".into(), "open".into()]) {
                    let ok = self.current.as_deref() == Some("supervision_admission")
                        && call.args.len() == 1
                        && matches!(call.args.first(), Some(syn::Expr::Path(p)) if p.path.is_ident("path"));
                    if !ok {
                        self.error = Some("unapproved File::open in receipt reader");
                    }
                }
            }
            syn::visit::visit_expr_call(self, call);
        }
        fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
            if call.method == "open" {
                let ok = self.current.as_deref()
                    == Some("spawn_retry_real_text_busy_exec_recovers_after_writer_closes")
                    && call.args.len() == 1
                    && matches!(call.args.first(), Some(syn::Expr::Reference(r)) if matches!(&*r.expr, syn::Expr::Path(p) if p.path.is_ident("bin")));
                if !ok {
                    self.error = Some("unapproved open method in receipt reader");
                }
            }
            syn::visit::visit_expr_method_call(self, call);
        }
        fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
            if path.path.segments.iter().any(|s| s.ident == "OpenOptions")
                && self.current.as_deref()
                    != Some("spawn_retry_real_text_busy_exec_recovers_after_writer_closes")
            {
                self.error = Some("unapproved OpenOptions in receipt reader");
            }
            syn::visit::visit_expr_path(self, path);
        }
    }
    let syntax =
        syn::parse_file(source).map_err(|e| format!("receipt reader parse failed: {e}"))?;
    let mut opens = Opens {
        current: None,
        error: None,
    };
    opens.visit_file(&syntax);
    if let Some(error) = opens.error {
        return Err(error.into());
    }
    Ok(())
}

pub(super) fn lock_source_inventory(
    sources: &[(String, String)],
    definition: &str,
    lock_path: &str,
) -> Result<(BTreeSet<String>, BTreeSet<String>), String> {
    let reader = package_path_key(RECEIPT_READER_PACKAGE, RECEIPT_READER_PATH);
    let mut definitions = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut reader_seen = false;
    for (relative, source) in sources {
        if source.contains(definition) {
            definitions.insert(relative.clone());
        }
        if *relative == reader {
            if reader_seen {
                return Err("duplicate receipt-reader exemption".into());
            }
            reader_seen = true;
            validate_receipt_reader(source, lock_path)?;
        } else if source.contains(lock_path) {
            paths.insert(relative.clone());
        }
    }
    if !reader_seen {
        return Err("stale receipt-reader exemption".into());
    }
    let owner = BTreeSet::from([package_path_key(
        LOCK_DEFINITION_PACKAGE,
        "src/measurement.rs",
    )]);
    if definitions != owner {
        return Err("GPU lock behavior must have exactly one Rust definition".into());
    }
    if paths != owner {
        return Err("the fleet GPU lock path must have exactly one Rust source of truth".into());
    }
    Ok((definitions, paths))
}

#[test]
fn receipt_reader_exemption_rejects_acquisition_and_unlisted_paths() {
    let lock_path = ["/tmp/lion-metal", "-gpu-test.lock"].concat();
    let definition = ["fn gpu", "_test_lock"].concat();
    let owner = package_path_key(LOCK_DEFINITION_PACKAGE, "src/measurement.rs");
    let reader = package_path_key(RECEIPT_READER_PACKAGE, RECEIPT_READER_PATH);
    let owner_source = format!("{definition}() {{}} const PATH: &str = \"{lock_path}\";");
    let valid = format!("fn check() {{ let _ = \"lock=Metal GPU ({lock_path}): acquired\"; }}");
    let entries = vec![
        (owner.clone(), owner_source.clone()),
        (reader.clone(), valid.clone()),
    ];
    let (definitions, paths) = lock_source_inventory(&entries, &definition, &lock_path).unwrap();
    assert_eq!(definitions, BTreeSet::from([owner.clone()]));
    assert_eq!(paths, BTreeSet::from([owner.clone()]));

    let opens =
        format!("fn bad() {{ let _ = std::fs::File::open(\"lock=Metal GPU ({lock_path})\"); }}");
    let error = lock_source_inventory(
        &[
            (owner.clone(), owner_source.clone()),
            (reader.clone(), opens),
        ],
        &definition,
        &lock_path,
    )
    .unwrap_err();
    assert!(error.contains("File::open"), "{error}");
    let extra = vec![
        (owner.clone(), owner_source.clone()),
        (reader.clone(), valid.clone()),
        ("../serve:examples/other.rs".to_string(), valid.clone()),
    ];
    let error = lock_source_inventory(&extra, &definition, &lock_path).unwrap_err();
    assert_eq!(
        error,
        "the fleet GPU lock path must have exactly one Rust source of truth"
    );
    let stale = lock_source_inventory(
        &[(owner.clone(), owner_source.clone())],
        &definition,
        &lock_path,
    )
    .unwrap_err();
    assert_eq!(stale, "stale receipt-reader exemption");
    for invalid in [
        "fn empty() {}".to_string(),
        format!("// {lock_path}\n{valid}"),
        format!("fn bad() {{ let _ = \"wrong ({lock_path})\"; }}"),
        format!("fn bad() {{ let _ = \"lock=Metal GPU ({lock_path})\"; flock(0); }}"),
        format!(
            "fn bad() {{ let _ = \"lock=Metal GPU ({lock_path})\"; {}(); }}",
            ["gpu", "_test_lock"].concat()
        ),
        format!(
            "fn bad() {{ let _ = std::fs::OpenOptions::new().open(\"lock=Metal GPU ({lock_path})\"); }}"
        ),
    ] {
        assert!(
            lock_source_inventory(
                &[
                    (owner.clone(), owner_source.clone()),
                    (reader.clone(), invalid)
                ],
                &definition,
                &lock_path
            )
            .is_err()
        );
    }
}
