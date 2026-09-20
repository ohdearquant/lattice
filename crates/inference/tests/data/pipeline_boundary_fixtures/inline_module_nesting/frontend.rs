// Both the `use` declaring the alias table entry AND the reference that
// consumes it live one level deeper than the file's top level, inside an
// INLINE `mod inner { .. }` -- as opposed to a file-based `mod inner;`,
// which would be indexed and scanned as its own separate file with its
// own home path built from scratch. `home_module_path` must be threaded
// one level deeper here on BOTH the indexing side (so the alias is
// recorded under `inner`, not the file's top level) and the scanning
// side (so this reference is resolved against `inner`'s own scope), or
// the two sides disagree and the reference either resolves to the wrong
// thing or falls through to unresolved.
mod inner {
    use crate::model::ConcreteModel;
    use crate::neutral::PlainOptions;

    pub fn uses_origin(_model: ConcreteModel) {}

    // Negative twin at the SAME nesting depth (same inline module, same
    // resolution scope), a non-origin type. Must NOT be reported --
    // proves this is real inline-module scoping, not "anything found
    // inside any mod counts".
    pub fn uses_neutral(_options: PlainOptions) {}
}
