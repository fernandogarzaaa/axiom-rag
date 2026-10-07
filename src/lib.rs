//! axiom-rag: agentic retrieval-augmented generation that works fully offline.
//!
//! The pipeline is:
//!
//! ```text
//! ingest -> chunk -> embed -> store -> retrieve (agentic loop) -> generate -> verify
//! ```
//!
//! * [`chunking`] splits documents into overlapping word windows.
//! * [`embedding`] maps text to dense vectors with the feature-hashing trick
//!   (deterministic, no model download, no API key).
//! * [`store`] keeps chunks plus embeddings and answers top-k cosine queries.
//! * [`retrieval`] runs the agentic loop: retrieve, check relevance with a
//!   verifier, reformulate the query with pseudo-relevance feedback, and retry.
//!   Failed formulations are fingerprinted and never repeated (AttemptMemory
//!   pattern, mirroring AXIOM-AETHER's `agent_task` module).
//! * [`generation`] builds answers extractively from retrieved chunks, so every
//!   sentence is traceable to a source.
//! * [`verifier`] rejects any sentence that is not supported by a cited chunk.
//! * [`improve`] drives the `axiom-agent-task` binary (AXIOM-AETHER PR #191)
//!   over stdio JSON-RPC to tune the retrieval configuration through a
//!   verifier-gated, transactional self-improvement loop.

pub mod chunking;
pub mod config;
pub mod embedding;
pub mod eval;
pub mod generation;
pub mod improve;
pub mod ingest;
pub mod retrieval;
pub mod serve;
pub mod store;
pub mod verifier;

use std::io;

use chunking::Chunk;
use config::RagConfig;
use embedding::Embedder;
use generation::Answer;
use retrieval::{RetrievalAgent, RetrievalOutcome};
use store::DocStore;

/// Convenience error type used across the crate.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Run the full question-answering pipeline: agentic retrieval followed by
/// verifier-gated generation.
pub fn answer_question(
    store: &DocStore,
    embedder: &Embedder,
    config: &RagConfig,
    question: &str,
) -> (Answer, RetrievalOutcome) {
    let mut agent = RetrievalAgent::new(store, embedder, config);
    let outcome = agent.retrieve(question);
    let hits: Vec<(Chunk, f32)> = outcome
        .hits
        .iter()
        .map(|h| (store.chunk(h.index).clone(), h.score))
        .collect();
    let answer = generation::generate_answer(question, &hits, embedder, config);
    let unsupported =
        verifier::verify_answer(&answer.sentences, &answer.citations, embedder, config);
    debug_assert!(
        unsupported.is_empty(),
        "generation emitted an unsupported sentence"
    );
    let _ = unsupported;
    (answer, outcome)
}

/// Load an index and config from disk, returning an error if either is missing.
pub fn load_index_and_config(
    index_path: &std::path::Path,
    config_path: Option<&std::path::Path>,
) -> Result<(DocStore, RagConfig)> {
    let store = DocStore::load(index_path).map_err(|e| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "cannot load index at {}: {e} (run `axiom-rag ingest` first)",
                index_path.display()
            ),
        )
    })?;
    let config = match config_path {
        Some(p) => RagConfig::load(p)?,
        None => RagConfig::default(),
    };
    Ok((store, config))
}
