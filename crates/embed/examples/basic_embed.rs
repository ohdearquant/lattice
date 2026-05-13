//! Basic embedding generation with lattice-embed.
//!
//! Run with:
//!   cargo run -p lattice-embed --example basic_embed
//!
//! On first run the model weights are downloaded from HuggingFace and cached
//! under `~/.lattice/cache/`. Subsequent runs load from cache and are fast.

use lattice_embed::{EmbeddingModel, EmbeddingService, NativeEmbeddingService, SimdConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Print detected SIMD capabilities so the user knows which code path is active.
    let simd = SimdConfig::detect();
    println!("SIMD capabilities:");
    println!("  AVX-512F:    {}", simd.avx512f_enabled);
    println!("  AVX2:        {}", simd.avx2_enabled);
    println!("  NEON:        {}", simd.neon_enabled);
    println!("  Any SIMD:    {}", simd.simd_available());
    println!();

    // --- Single embedding ---
    // NativeEmbeddingService::default() uses BgeSmallEnV15 (384 dims).
    // The model is loaded lazily on the first embed call.
    let service = NativeEmbeddingService::default();
    let model = EmbeddingModel::default(); // BgeSmallEnV15

    println!("Model:      {}", model);
    println!("Dimensions: {}", model.dimensions());
    println!("Max tokens: {}", model.max_input_tokens());
    println!("Local:      {}", model.is_local());
    println!();

    println!("Generating single embedding (model loads on first call)...");
    let text = "The quick brown fox jumps over the lazy dog";
    let embedding = service.embed_one(text, model).await?;

    println!("Text:       \"{text}\"");
    println!("Embedding:  {} dimensions", embedding.len());
    println!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);
    println!();

    // --- Choosing a different model ---
    // Each model variant has a companion NativeEmbeddingService. The service
    // validates that the requested model matches the loaded one.
    println!("Model selection via string parsing:");
    let from_str: EmbeddingModel = "bge-base".parse().unwrap();
    println!(
        "  \"bge-base\" => {:?} ({} dims)",
        from_str,
        from_str.dimensions()
    );

    let from_hf: EmbeddingModel = "BAAI/bge-large-en-v1.5".parse().unwrap();
    println!(
        "  \"BAAI/bge-large-en-v1.5\" => {:?} ({} dims)",
        from_hf,
        from_hf.dimensions()
    );
    println!();

    // --- Batch embeddings ---
    println!("Generating batch of 3 embeddings...");
    let texts = vec![
        "Rust is a systems programming language focused on safety and performance.".to_string(),
        "Python excels at rapid prototyping and data science workflows.".to_string(),
        "WebAssembly enables near-native performance in web browsers.".to_string(),
    ];

    let embeddings = service.embed(&texts, model).await?;
    assert_eq!(embeddings.len(), 3);

    for (text, emb) in texts.iter().zip(&embeddings) {
        println!("  [{} dims] {}", emb.len(), &text[..40]);
    }
    println!();

    // --- E5 model: asymmetric retrieval prefix ---
    // MultilingualE5Small and E5Base expect "query: " / "passage: " prefixes.
    // BGE and MiniLM models have no such requirement.
    let e5 = EmbeddingModel::MultilingualE5Small;
    let query = "machine learning frameworks";
    let prefixed = match e5.query_instruction() {
        Some(prefix) => format!("{prefix}{query}"),
        None => query.to_string(),
    };
    println!("E5 query instruction: {:?}", e5.query_instruction());
    println!("Prefixed query: \"{prefixed}\"");

    println!();
    println!("Done.");
    Ok(())
}
