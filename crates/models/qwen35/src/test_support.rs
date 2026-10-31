//! Unsupported Qwen3.5 fixtures for version-coupled consumers.

/// The existing scoring witness, without duplicating its private fixture factory.
pub fn reference_scoring_fixture() -> (
    lattice_arch::qwen35_config::Qwen35Config,
    crate::model::qwen35::ModelWeights,
) {
    crate::reference_scoring_fixture::option_scoring_fixture()
}
