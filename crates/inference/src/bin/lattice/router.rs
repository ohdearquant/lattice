//! `lattice router init` (ADR-096 decision 2).
//!
//! The decision is in `lattice_inference::router_init`, which has no model and
//! no CLI in it and is therefore testable without either. What lives here is
//! the part that cannot be: loading the embedding model and measuring the
//! width through the same loader contract the serving path uses.
//!
//! That measurement is the reason this command loads a model at all. The width
//! could be a flag, and then a mistyped number would produce an artifact that
//! refuses at every startup with a message about the gate rather than about
//! the typo. Measuring it means the recorded width is the width this machine's
//! serving path will produce, or the command fails here where the operator is
//! still looking at it.

#[cfg(all(target_os = "macos", feature = "metal-gpu", feature = "mixture"))]
use std::path::Path;

use clap::Args;

#[derive(Debug, Args)]
pub struct InitArgs {
    /// Destination state directory. Created if absent; refused if it already
    /// holds an artifact.
    #[arg(long)]
    pub state: String,
    /// The ordered adapter names, comma-separated. This IS the schema: the
    /// order is the gate's column order and ADR-096 decision 1 makes it
    /// immutable within a lineage.
    ///
    /// Required explicitly: a `Vec` argument is optional to clap by default,
    /// and an omitted one would arrive as an empty schema and be refused a
    /// layer down, naming the specification rather than the missing flag.
    #[arg(long, value_delimiter = ',', required = true)]
    pub adapters: Vec<String>,
    /// The schema identity this lineage is named by.
    #[arg(long)]
    pub schema_id: String,
    /// The base model the adapters attach to.
    #[arg(long)]
    pub base_model: String,
    /// Directory of the f16 embedding model the gate will consume vectors
    /// from. Loaded, not merely named: the input width is measured through it.
    #[arg(long)]
    pub embedding_model: String,
    /// The identity recorded for that embedder. Defaults to the directory's
    /// own resolution, the same one `lattice serve` uses.
    #[arg(long)]
    pub embedding_model_id: Option<String>,
    /// How the embedder's token states are pooled into the context vector.
    /// Validated with the serving path's own parser, so a value this build
    /// cannot reproduce is refused here rather than at the first request.
    #[arg(long, default_value = "mean_visual")]
    pub pooling: String,
    /// Which text of a request the gate routes on. One rule exists today and
    /// it is the default; it is a flag so the artifact records the rule
    /// explicitly and a second rule needs no new surface.
    #[arg(long, default_value = "last_user_message")]
    pub prompt_source: String,
}

/// Run `router init`, returning false when the command refused.
#[cfg(all(target_os = "macos", feature = "metal-gpu", feature = "mixture"))]
pub fn run(args: &InitArgs) -> bool {
    use lattice_inference::router_init::{BootstrapSpec, init_directory};
    use lattice_inference::router_state::TrainedRepresentation;
    use lattice_inference::serve::routing::EmbedderIdentity;

    let embedder_dir = Path::new(&args.embedding_model);
    // Named explicitly, so it fails closed: the operator said where the
    // embedder is, and "it did not load" is an error rather than a downgrade.
    // `lattice serve` applies the same rule to the same flag.
    let embedder =
        match lattice_inference::serve::embeddings::EmbeddingModel::from_directory(embedder_dir) {
            Ok(model) => model,
            Err(err) => {
                eprintln!(
                    "Error: --embedding-model {} cannot serve as an embedding model: {err}",
                    args.embedding_model
                );
                return false;
            }
        };
    let identity = EmbedderIdentity::resolve(embedder_dir, args.embedding_model_id.as_deref());

    // Both validated with the serving path's own parsers. A second list here
    // is how a build mints an artifact naming a rule it cannot reproduce,
    // which passes every check downstream until a request arrives.
    if let Err(err) = lattice_inference::serve::routing::trained_pooling(&args.pooling) {
        eprintln!("Error: {}", err.message());
        return false;
    }
    if let Err(err) = lattice_inference::serve::routing::trained_prompt_source(&args.prompt_source)
    {
        eprintln!("Error: {}", err.message());
        return false;
    }

    let spec = BootstrapSpec {
        adapter_names: args.adapters.clone(),
        representation: TrainedRepresentation {
            embedding_model: identity.as_str().to_string(),
            pooling: args.pooling.clone(),
            prompt_source: args.prompt_source.clone(),
            loader_format: lattice_inference::serve::embeddings::QWEN35_F16_DECODER_LOADER
                .to_string(),
            input_width: embedder.dimensions() as u64,
        },
        schema_id: args.schema_id.clone(),
        base_model: args.base_model.clone(),
    };

    let dir = Path::new(&args.state);
    match init_directory(dir, &spec) {
        Ok(path) => {
            println!(
                "Initialised {} with an UNTRAINED bootstrap gate over {} adapter(s): {}.",
                args.state,
                spec.adapter_names.len(),
                spec.adapter_names.join(", ")
            );
            println!(
                "  manifest:    {}\n  embedder:    {} ({} dimensions, measured)\n  policy:      uniform, training_state=bootstrap",
                path.display(),
                identity.as_str(),
                spec.representation.input_width
            );
            println!(
                "This gate has learned nothing. Serving it is the uniform baseline, and \
                 GET /v1/lora reports training_state=bootstrap so a response from it is \
                 never read as evidence of learning. Enabling learning is a separate \
                 explicit action; this command did not perform it."
            );
            true
        }
        Err(err) => {
            eprintln!("Error: {err}");
            false
        }
    }
}

/// A build that cannot serve a router does not mint one.
///
/// BOTH requirements, and the second was found by running this command's
/// output rather than by reading: `--router-state` is admitted by a macOS
/// Metal build, and then the gate itself refuses without `mixture`. Gating
/// this command on the Metal half alone let a build mint a directory it could
/// not serve, which fails at the next startup, possibly on another machine,
/// with a message about the gate rather than about the build that wrote it.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu", feature = "mixture")))]
pub fn run(_args: &InitArgs) -> bool {
    eprintln!(
        "Error: `router init` requires a macOS Metal build with the `mixture` feature -- the same \
         pair `--router-state` needs to actually run a gate. It measures the gate's input width \
         through the serving path's own embedding loader, and a build that cannot route must not \
         mint a directory it cannot serve."
    );
    false
}
