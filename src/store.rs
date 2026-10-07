//! In-memory vector store with brute-force cosine search and JSON persistence.
//!
//! Brute force is the honest choice at this scale: it is exact, simple, and
//! fast enough for tens of thousands of chunks. An HNSW index would only pay
//! off past ~100k vectors and would add an approximation the verifier would
//! then have to reason about.

use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

use crate::chunking::Chunk;
use crate::embedding::Embedder;

/// A chunk bundled with its embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredChunk {
    pub chunk: Chunk,
    pub embedding: Vec<f32>,
}

/// One search result: index into the store plus cosine score.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub index: usize,
    pub score: f32,
}

/// The index itself. Serialized as JSON so it stays human-inspectable.
#[derive(Debug, Serialize, Deserialize)]
pub struct DocStore {
    chunks: Vec<StoredChunk>,
    dim: usize,
}

impl DocStore {
    /// Embed every chunk and build the store.
    pub fn build(chunks: Vec<Chunk>, embedder: &Embedder) -> Self {
        let dim = embedder.dim();
        let stored = chunks
            .into_iter()
            .map(|chunk| {
                let embedding = embedder.embed(&chunk.text);
                StoredChunk { chunk, embedding }
            })
            .collect();
        Self {
            chunks: stored,
            dim,
        }
    }

    /// Number of indexed chunks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// True when no chunks are indexed.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Embedding dimension the store was built with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Borrow a chunk by index.
    pub fn chunk(&self, index: usize) -> &Chunk {
        &self.chunks[index].chunk
    }

    /// Exact top-k cosine search, sorted by descending score.
    ///
    /// Panics if `query_emb` has a different dimension than the store.
    pub fn search(&self, query_emb: &[f32], top_k: usize) -> Vec<SearchHit> {
        assert_eq!(
            query_emb.len(),
            self.dim,
            "query embedding dim {} != store dim {}",
            query_emb.len(),
            self.dim
        );
        let mut hits: Vec<SearchHit> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(index, sc)| SearchHit {
                index,
                score: Embedder::cosine(query_emb, &sc.embedding),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k.max(1).min(hits.len().max(1)));
        if self.chunks.is_empty() {
            hits.clear();
        }
        hits
    }

    /// Persist to `path` as pretty JSON.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let raw = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, raw)
    }

    /// Load a store written by [`DocStore::save`].
    pub fn load(path: &Path) -> io::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        serde_json::from_str(&raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::chunk_text;

    fn sample_store() -> (DocStore, Embedder) {
        let embedder = Embedder::new(128);
        let docs = [
            ("rust ownership borrow checker", "a.md"),
            ("photosynthesis chlorophyll plants", "b.md"),
            ("rust lifetimes and traits", "c.md"),
        ];
        let mut chunks = Vec::new();
        for (text, src) in docs {
            chunks.extend(chunk_text(text, src, 50, 0));
        }
        (DocStore::build(chunks, &embedder), embedder)
    }

    #[test]
    fn search_ranks_related_first() {
        let (store, embedder) = sample_store();
        let q = embedder.embed("rust borrow checker");
        let hits = store.search(&q, 3);
        assert_eq!(hits.len(), 3);
        assert!(hits[0].score >= hits[1].score);
        assert!(hits[1].score >= hits[2].score);
        assert_eq!(store.chunk(hits[0].index).source, "a.md");
    }

    #[test]
    fn search_top_k_bounded() {
        let (store, embedder) = sample_store();
        let q = embedder.embed("rust");
        assert_eq!(store.search(&q, 1).len(), 1);
        assert_eq!(store.search(&q, 100).len(), 3);
    }

    #[test]
    fn save_load_round_trip() {
        let (store, embedder) = sample_store();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("index.json");
        store.save(&p).unwrap();
        let back = DocStore::load(&p).unwrap();
        assert_eq!(back.len(), store.len());
        let q = embedder.embed("rust borrow checker");
        let a = store.search(&q, 2);
        let b = back.search(&q, 2);
        assert_eq!(a[0].index, b[0].index);
        assert!((a[0].score - b[0].score).abs() < 1e-6);
    }

    #[test]
    fn empty_store_searches_empty() {
        let embedder = Embedder::new(64);
        let store = DocStore::build(Vec::new(), &embedder);
        assert!(store.is_empty());
        assert!(store.search(&embedder.embed("x"), 5).is_empty());
    }
}
