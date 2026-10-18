use super::index::span_range;
use crate::source_graph::{CfgFormula, item_attributes, syn_attributes_formula_with_test};
use std::ops::Range;
use syn::spanned::Spanned;
use syn::visit::Visit;

pub(super) fn production_source(
    file: &str,
    source: &str,
    inherited: CfgFormula,
) -> Result<String, String> {
    let syntax =
        syn::parse_file(source).map_err(|error| format!("{file}: invalid Rust: {error}"))?;
    let mut reader = Masker {
        file,
        source,
        cfg: inherited,
        masked: Vec::new(),
        error: None,
    };
    reader.visit_file(&syntax);
    if let Some(error) = reader.error {
        return Err(error);
    }
    let mut bytes = source.as_bytes().to_vec();
    for range in reader.masked {
        for byte in &mut bytes[range] {
            if *byte != b'\n' && *byte != b'\r' {
                *byte = b' ';
            }
        }
    }
    let result = String::from_utf8(bytes).map_err(|error| error.to_string())?;
    syn::parse_file(&result)
        .map_err(|error| format!("{file}: production mask is not valid Rust: {error}"))?;
    Ok(result)
}

pub(super) fn expression_attributes(expression: &syn::Expr) -> &[syn::Attribute] {
    match expression {
        syn::Expr::Array(e) => &e.attrs,
        syn::Expr::Assign(e) => &e.attrs,
        syn::Expr::Async(e) => &e.attrs,
        syn::Expr::Await(e) => &e.attrs,
        syn::Expr::Binary(e) => &e.attrs,
        syn::Expr::Block(e) => &e.attrs,
        syn::Expr::Break(e) => &e.attrs,
        syn::Expr::Call(e) => &e.attrs,
        syn::Expr::Cast(e) => &e.attrs,
        syn::Expr::Closure(e) => &e.attrs,
        syn::Expr::Const(e) => &e.attrs,
        syn::Expr::Continue(e) => &e.attrs,
        syn::Expr::Field(e) => &e.attrs,
        syn::Expr::ForLoop(e) => &e.attrs,
        syn::Expr::Group(e) => &e.attrs,
        syn::Expr::If(e) => &e.attrs,
        syn::Expr::Index(e) => &e.attrs,
        syn::Expr::Infer(e) => &e.attrs,
        syn::Expr::Let(e) => &e.attrs,
        syn::Expr::Lit(e) => &e.attrs,
        syn::Expr::Loop(e) => &e.attrs,
        syn::Expr::Macro(e) => &e.attrs,
        syn::Expr::Match(e) => &e.attrs,
        syn::Expr::MethodCall(e) => &e.attrs,
        syn::Expr::Paren(e) => &e.attrs,
        syn::Expr::Path(e) => &e.attrs,
        syn::Expr::Range(e) => &e.attrs,
        syn::Expr::RawAddr(e) => &e.attrs,
        syn::Expr::Reference(e) => &e.attrs,
        syn::Expr::Repeat(e) => &e.attrs,
        syn::Expr::Return(e) => &e.attrs,
        syn::Expr::Struct(e) => &e.attrs,
        syn::Expr::Try(e) => &e.attrs,
        syn::Expr::TryBlock(e) => &e.attrs,
        syn::Expr::Tuple(e) => &e.attrs,
        syn::Expr::Unary(e) => &e.attrs,
        syn::Expr::Unsafe(e) => &e.attrs,
        syn::Expr::While(e) => &e.attrs,
        syn::Expr::Yield(e) => &e.attrs,
        _ => &[],
    }
}

struct Masker<'a> {
    file: &'a str,
    source: &'a str,
    cfg: CfgFormula,
    masked: Vec<Range<usize>>,
    error: Option<String>,
}

impl Masker<'_> {
    fn trailing_comma(&mut self, previous: usize, span: proc_macro2::Span) {
        if self.masked.len() == previous {
            return;
        }
        let Ok(own_range) = span_range(self.source, span) else {
            return;
        };
        let Some(range) = self.masked.last_mut() else {
            return;
        };
        if *range != own_range {
            return;
        }
        let mut end = range.end;
        while self
            .source
            .as_bytes()
            .get(end)
            .is_some_and(u8::is_ascii_whitespace)
        {
            end += 1;
        }
        if self.source.as_bytes().get(end) == Some(&b',') {
            range.end = end + 1;
        }
    }
    fn scoped(
        &mut self,
        attrs: &[syn::Attribute],
        span: proc_macro2::Span,
        visit: impl FnOnce(&mut Self),
    ) {
        if self.error.is_some() {
            return;
        }
        let result = (|| {
            let cfg = CfgFormula::all([
                self.cfg.clone(),
                syn_attributes_formula_with_test(attrs, None, self.file)?,
            ]);
            let test_marker = attrs.iter().any(|attr| {
                attr.path()
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "test")
            });
            if test_marker || !cfg.satisfiable_with_test(Some(false))? {
                self.masked.push(span_range(self.source, span)?);
                return Ok(());
            }
            let previous = std::mem::replace(&mut self.cfg, cfg);
            visit(self);
            self.cfg = previous;
            Ok::<(), String>(())
        })();
        if let Err(error) = result {
            self.error = Some(format!("{}: {error}", self.file));
        }
    }
}

impl<'ast> Visit<'ast> for Masker<'_> {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        self.scoped(item_attributes(item), item.span(), |reader| {
            syn::visit::visit_item(reader, item)
        });
    }
    fn visit_impl_item(&mut self, item: &'ast syn::ImplItem) {
        let attrs: &[syn::Attribute] = match item {
            syn::ImplItem::Const(item) => &item.attrs,
            syn::ImplItem::Fn(item) => &item.attrs,
            syn::ImplItem::Type(item) => &item.attrs,
            syn::ImplItem::Macro(item) => &item.attrs,
            _ => &[],
        };
        self.scoped(attrs, item.span(), |reader| {
            syn::visit::visit_impl_item(reader, item)
        });
    }
    fn visit_trait_item(&mut self, item: &'ast syn::TraitItem) {
        let attrs: &[syn::Attribute] = match item {
            syn::TraitItem::Const(item) => &item.attrs,
            syn::TraitItem::Fn(item) => &item.attrs,
            syn::TraitItem::Type(item) => &item.attrs,
            syn::TraitItem::Macro(item) => &item.attrs,
            _ => &[],
        };
        self.scoped(attrs, item.span(), |reader| {
            syn::visit::visit_trait_item(reader, item)
        });
    }
    fn visit_foreign_item(&mut self, item: &'ast syn::ForeignItem) {
        let attrs: &[syn::Attribute] = match item {
            syn::ForeignItem::Fn(item) => &item.attrs,
            syn::ForeignItem::Static(item) => &item.attrs,
            syn::ForeignItem::Type(item) => &item.attrs,
            syn::ForeignItem::Macro(item) => &item.attrs,
            _ => &[],
        };
        self.scoped(attrs, item.span(), |reader| {
            syn::visit::visit_foreign_item(reader, item)
        });
    }
    fn visit_stmt(&mut self, stmt: &'ast syn::Stmt) {
        let attrs = match stmt {
            syn::Stmt::Local(local) => local.attrs.as_slice(),
            syn::Stmt::Item(item) => item_attributes(item),
            syn::Stmt::Expr(expression, _) => expression_attributes(expression),
            syn::Stmt::Macro(mac) => mac.attrs.as_slice(),
        };
        self.scoped(attrs, stmt.span(), |reader| {
            syn::visit::visit_stmt(reader, stmt)
        });
    }
    fn visit_expr(&mut self, expr: &'ast syn::Expr) {
        let previous = self.masked.len();
        self.scoped(expression_attributes(expr), expr.span(), |reader| {
            syn::visit::visit_expr(reader, expr)
        });
        self.trailing_comma(previous, expr.span());
    }
    fn visit_arm(&mut self, arm: &'ast syn::Arm) {
        let previous = self.masked.len();
        self.scoped(&arm.attrs, arm.span(), |reader| {
            syn::visit::visit_arm(reader, arm)
        });
        self.trailing_comma(previous, arm.span());
    }
    fn visit_field(&mut self, field: &'ast syn::Field) {
        let previous = self.masked.len();
        self.scoped(&field.attrs, field.span(), |reader| {
            syn::visit::visit_field(reader, field)
        });
        self.trailing_comma(previous, field.span());
    }
}
