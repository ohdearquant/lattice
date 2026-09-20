use crate::model::ConcreteModel as Renamed;
use crate::neutral::PlainOptions;

pub fn consume(_model: Renamed, _options: PlainOptions) {}

// Cross-module arm: the alias is declared in `alias_owner`, not here, and
// referenced through its full `crate::`-qualified path -- keeping the
// same-file arm above unchanged rather than replacing it.
pub fn consume_via_alias_owner(
    _model: crate::alias_owner::Renamed,
    _options: crate::alias_owner::RenamedNeutral,
) {
}
