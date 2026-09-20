use crate::model::ConcreteModel;
use crate::neutral::PlainOptions;

// MUST be reported: a `select!`-shaped macro body is opaque to syn (it
// never parses as items or exprs), so the ONLY way this origin-type
// reference is seen is the token-shaped fallback scanning the raw macro
// token stream. Bare (unqualified) macro name deliberately, so the
// fixture -- which has no Cargo.toml and so no declared external crates
// -- does not also exercise the UNRELATED pre-existing "macro's own
// invocation path is itself unresolved" case that a crate-qualified name
// like `tokio::select!` would trip in a fixture with no `tokio`
// dependency to recognize it as external.
fn consume() {
    select! {
        result = ConcreteModel::from_directory() => handle(result),
        _ = other_future() => {}
    }
}

// Negative twin: identical macro shape and nesting depth, a non-origin
// reference. Must NOT be reported.
fn consume_neutral() {
    select! {
        result = PlainOptions::from_directory() => handle(result),
        _ = other_future() => {}
    }
}
