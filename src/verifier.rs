//! Claim-level verification: every emitted sentence must be supported.
//!
//! The generator only emits sentences drawn from retrieved chunks, but drawing
//! is not proving. [`verify_answer`] re-checks each sentence against its cited
//! chunk independently, so a bug in ranking or selection cannot smuggle an
//! unsupported claim into the final answer.

use std::collections::HashSet;

use crate::chunking::Chunk;
use crate::config::RagConfig;
use crate::embedding::{tokenize, Embedder};
use crate::generation::Citation;

/// True when `sentence` is supported by `chunk_text`.
///
/// Two independent signals, either suffices:
/// * cosine similarity of hash embeddings >= `threshold`
/// * token recall: at least 60% of the sentence's tokens appear in the chunk
///
/// The lexical fallback matters because hash embeddings are coarse; a short
/// sentence sharing most of its content words with the chunk is clearly
/// supported even when the cosine is modest.
pub fn claim_supported(
    sentence: &str,
    chunk_text: &str,
    embedder: &Embedder,
    threshold: f64,
) -> bool {
    let s_toks: HashSet<String> = tokenize(sentence).into_iter().collect();
    if s_toks.is_empty() {
        return false;
    }
    let s_emb = embedder.embed(sentence);
    let c_emb = embedder.embed(chunk_text);
    if Embedder::cosine(&s_emb, &c_emb) as f64 >= threshold {
        return true;
    }
    let c_toks: HashSet<String> = tokenize(chunk_text).into_iter().collect();
    let shared = s_toks.intersection(&c_toks).count();
    shared as f64 / s_toks.len() as f64 >= 0.6
}

/// Re-verify a generated answer. Returns the indices of sentences that are
/// NOT supported by their cited chunk. An empty vec means the answer verifies.
pub fn verify_answer(
    sentences: &[String],
    citations: &[Citation],
    embedder: &Embedder,
    config: &RagConfig,
) -> Vec<usize> {
    let mut bad = Vec::new();
    for (i, sentence) in sentences.iter().enumerate() {
        let supported = citations
            .iter()
            .find(|c| c.sentence_index == i)
            .map(|c| claim_supported(sentence, &c.chunk_text, embedder, config.claim_threshold))
            .unwrap_or(false);
        if !supported {
            bad.push(i);
        }
    }
    bad
}

/// A citation must point at a real chunk of the index.
pub fn citation_valid(citation: &Citation, chunks: &[Chunk]) -> bool {
    chunks
        .iter()
        .any(|c| c.source == citation.source && c.index == citation.chunk_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::Embedder;

    fn embedder() -> Embedder {
        Embedder::new(256)
    }

    #[test]
    fn sentence_from_chunk_is_supported() {
        let e = embedder();
        let chunk = "The Axiom engine performs test-time training directly on device without any cloud dependency.";
        let sentence = "The Axiom engine performs test-time training directly on device.";
        assert!(claim_supported(sentence, chunk, &e, 0.12));
    }

    #[test]
    fn unrelated_sentence_is_not_supported() {
        let e = embedder();
        let chunk = "The Axiom engine performs test-time training directly on device.";
        let sentence = "Photosynthesis converts sunlight into chemical energy in leaves.";
        assert!(!claim_supported(sentence, chunk, &e, 0.12));
    }

    #[test]
    fn lexical_fallback_catches_near_verbatim() {
        let e = embedder();
        // High threshold defeats the cosine signal; token recall still fires.
        let chunk = "quux zyxw blorp fnord quux zyxw blorp fnord extra words here";
        let sentence = "quux zyxw blorp fnord";
        assert!(claim_supported(sentence, chunk, &e, 0.99));
    }

    #[test]
    fn empty_sentence_never_supported() {
        let e = embedder();
        assert!(!claim_supported("", "some chunk text here", &e, 0.0));
    }
}
