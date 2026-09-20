// Declares the alias in a DIFFERENT module than the one that consumes it,
// as a PLAIN (non-`pub`) `use` -- deliberately NOT `pub use`, so this
// cannot resolve through the pre-existing `NamespaceIndex.reexports`
// chase and must instead go through `Resolver::lookup`'s alias-table
// branch (Decision B: a `use` that points at another module's alias,
// not a real definition or a `pub use` re-export). A consumer resolving
// `crate::alias_owner::Renamed` is this scanner's structural analogue of
// that case; it does not model Rust's real privacy rules.
use crate::model::ConcreteModel as Renamed;
use crate::neutral::PlainOptions as RenamedNeutral;
