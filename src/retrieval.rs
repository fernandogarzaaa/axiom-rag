//! Agentic retrieval loop.
//!
//! Single-shot RAG fails when the first query formulation is poor. Instead of
//! accepting the first result set, [`RetrievalAgent`] runs a verifier-gated
//! loop:
//!
//! 1. Embed the current query formulation and search.
//! 2. Run the relevance verifier: the best score must clear `min_score`.
//! 3. On failure, fingerprint the formulation, record it as rejected, and
//!    reformulate with pseudo-relevance feedback (terms from the retrieved
//!    chunks that were missing from the query).
//! 4. Repeat up to `max_attempts`; never retry a rejected formulation.
//!
//! The fingerprint-and-remember discipline is the AttemptMemory pattern from
//! AXIOM-AETHER's `agent_task` module: the agent learns what did not work and
//! does not burn verifier calls repeating it.

use std::collections::{HashMap, HashSet};

use crate::config::RagConfig;
use crate::embedding::{tokenize, Embedder};
use crate::store::{DocStore, SearchHit};

/// Remembers rejected query formulations per question so the loop never
/// proposes the same formulation twice.
#[derive(Debug, Default)]
pub struct AttemptMemory {
    rejected: HashMap<String, Vec<String>>,
}

impl AttemptMemory {
    /// Empty memory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Was this exact formulation already tried and rejected for `query_fp`?
    pub fn was_rejected(&self, query_fp: &str, formulation_fp: &str) -> bool {
        self.rejected
            .get(query_fp)
            .map(|v| v.iter().any(|f| f == formulation_fp))
            .unwrap_or(false)
    }

    /// Record a formulation as rejected for `query_fp` (idempotent).
    pub fn record_rejected(&mut self, query_fp: &str, formulation_fp: &str) {
        let list = self.rejected.entry(query_fp.to_string()).or_default();
        if !list.iter().any(|f| f == formulation_fp) {
            list.push(formulation_fp.to_string());
        }
    }

    /// Number of distinct rejected formulations for a question.
    pub fn rejected_count(&self, query_fp: &str) -> usize {
        self.rejected.get(query_fp).map(|v| v.len()).unwrap_or(0)
    }
}

/// Deterministic fingerprint of a query formulation: lowercase tokens, sorted
/// and deduplicated, hashed with FNV-1a. Paraphrases with the same content
/// words collide on purpose; that is what makes dedup useful.
pub fn fingerprint(text: &str) -> String {
    let mut toks = tokenize(text);
    toks.sort();
    toks.dedup();
    let joined = toks.join(" ");
    let mut h: u64 = 0xcbf29ce484222325;
    for b in joined.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// A small English stopword list for reformulation term selection.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "have", "were", "been", "are", "was",
    "had", "has", "will", "would", "what", "when", "where", "which", "who", "how", "why", "does",
    "did", "can", "could", "should", "about", "into", "over", "under", "between", "through",
    "during", "such", "than", "then", "them", "they", "their", "there", "these", "those", "its",
    "our", "your", "his", "her", "she", "him", "not", "but", "all", "any",
];

fn is_stopword(tok: &str) -> bool {
    STOPWORDS.contains(&tok)
}

/// The outcome of one agentic retrieval run.
#[derive(Debug)]
pub struct RetrievalOutcome {
    /// Top-k hits of the accepted (or best-effort) formulation.
    pub hits: Vec<SearchHit>,
    /// Query formulations tried, in order.
    pub formulations: Vec<String>,
    /// Whether the relevance verifier accepted the final result set.
    pub passed: bool,
    /// Best cosine score seen across attempts.
    pub best_score: f32,
}

/// Drives the retrieve -> verify -> reformulate loop.
pub struct RetrievalAgent<'a> {
    store: &'a DocStore,
    embedder: &'a Embedder,
    config: &'a RagConfig,
    memory: AttemptMemory,
}

impl<'a> RetrievalAgent<'a> {
    /// Borrow the store, embedder, and config for one question.
    pub fn new(store: &'a DocStore, embedder: &'a Embedder, config: &'a RagConfig) -> Self {
        Self {
            store,
            embedder,
            config,
            memory: AttemptMemory::new(),
        }
    }

    /// Access the attempt memory (useful for tests and introspection).
    pub fn memory(&self) -> &AttemptMemory {
        &self.memory
    }

    /// Run the loop for `question`.
    pub fn retrieve(&mut self, question: &str) -> RetrievalOutcome {
        let query_fp = fingerprint(question);
        let mut formulation = normalize_question(question);
        let mut formulations = Vec::new();
        let mut best_hits: Vec<SearchHit> = Vec::new();
        let mut best_score = 0.0f32;
        let mut passed = false;

        if self.store.is_empty() {
            return RetrievalOutcome {
                hits: Vec::new(),
                formulations,
                passed: false,
                best_score,
            };
        }

        let budget = self.config.max_attempts.max(1);
        for _ in 0..budget {
            let fp = fingerprint(&formulation);
            if self.memory.was_rejected(&query_fp, &fp) {
                break; // would repeat a known-bad formulation
            }
            formulations.push(formulation.clone());

            let emb = self.embedder.embed(&formulation);
            let hits = self.store.search(&emb, self.config.top_k * 2);
            let top = hits.first().map(|h| h.score).unwrap_or(0.0);
            if top > best_score {
                best_score = top;
                best_hits = hits.clone();
            }

            if verify_relevance(&hits, self.config.min_score) {
                best_hits = hits;
                best_score = top;
                passed = true;
                break;
            }

            self.memory.record_rejected(&query_fp, &fp);
            let next = self.reformulate(&formulation, &hits);
            if next == formulation {
                break; // no new terms to try
            }
            formulation = next;
        }

        best_hits.truncate(self.config.top_k.max(1));
        RetrievalOutcome {
            hits: best_hits,
            formulations,
            passed,
            best_score,
        }
    }

    /// Pseudo-relevance feedback: harvest discriminative terms from the top
    /// retrieved chunks that are missing from the query and append the best
    /// few. Falls back to dropping the query's last token to guarantee the
    /// formulation changes.
    fn reformulate(&self, formulation: &str, hits: &[SearchHit]) -> String {
        let have: HashSet<String> = tokenize(formulation).into_iter().collect();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for hit in hits.iter().take(3) {
            for tok in tokenize(&self.store.chunk(hit.index).text) {
                if tok.len() < 3 || is_stopword(&tok) || have.contains(&tok) {
                    continue;
                }
                *counts.entry(tok).or_insert(0) += 1;
            }
        }
        let mut terms: Vec<(String, usize)> = counts.into_iter().collect();
        terms.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let additions: Vec<String> = terms.into_iter().take(3).map(|(t, _)| t).collect();
        if !additions.is_empty() {
            return format!("{formulation} {}", additions.join(" "));
        }
        // No new terms: shorten the query so the formulation still changes.
        let toks = tokenize(formulation);
        if toks.len() > 1 {
            toks[..toks.len() - 1].join(" ")
        } else {
            formulation.to_string()
        }
    }
}

/// The retrieval verifier: a result set is relevant when it is non-empty and
/// its best cosine score clears `min_score`.
pub fn verify_relevance(hits: &[SearchHit], min_score: f64) -> bool {
    match hits.first() {
        Some(h) => h.score as f64 >= min_score,
        None => false,
    }
}

/// Lowercase, collapse whitespace; the canonical form the loop mutates.
fn normalize_question(q: &str) -> String {
    q.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::chunk_text;
    use crate::config::RagConfig;

    fn sample() -> (DocStore, Embedder, RagConfig) {
        let config = RagConfig {
            embedding_dim: 128,
            top_k: 2,
            ..RagConfig::default()
        };
        let embedder = Embedder::new(config.embedding_dim);
        let texts = [
            (
                "the axiom engine performs test time training on device",
                "a.md",
            ),
            (
                "photosynthesis converts sunlight into chemical energy",
                "b.md",
            ),
            ("test time training adapts models during inference", "c.md"),
        ];
        let mut chunks = Vec::new();
        for (t, s) in texts {
            chunks.extend(chunk_text(t, s, 50, 0));
        }
        (DocStore::build(chunks, &embedder), embedder, config)
    }

    #[test]
    fn memory_dedups_rejected_formulations() {
        let mut m = AttemptMemory::new();
        assert!(!m.was_rejected("q", "f1"));
        m.record_rejected("q", "f1");
        m.record_rejected("q", "f1"); // idempotent
        assert!(m.was_rejected("q", "f1"));
        assert!(!m.was_rejected("q", "f2"));
        assert_eq!(m.rejected_count("q"), 1);
    }

    #[test]
    fn fingerprint_ignores_word_order_and_case() {
        assert_eq!(fingerprint("Hello World"), fingerprint("world hello"));
        assert_ne!(fingerprint("hello world"), fingerprint("hello rust"));
    }

    #[test]
    fn retrieve_passes_on_good_first_query() {
        let (store, embedder, config) = sample();
        let mut agent = RetrievalAgent::new(&store, &embedder, &config);
        let out = agent.retrieve("test time training");
        assert!(out.passed, "formulations: {:?}", out.formulations);
        assert_eq!(out.formulations.len(), 1);
        assert!(!out.hits.is_empty());
    }

    #[test]
    fn retrieve_reformulates_until_relevant_or_budget() {
        let (store, embedder, mut config) = sample();
        config.min_score = 0.99; // unreachable: forces the full loop
        config.max_attempts = 3;
        let mut agent = RetrievalAgent::new(&store, &embedder, &config);
        let out = agent.retrieve("quantum tunneling");
        assert!(!out.passed);
        assert_eq!(out.formulations.len(), 3);
        // all tried formulations were remembered as rejected
        assert_eq!(
            agent
                .memory()
                .rejected_count(&fingerprint("quantum tunneling")),
            3
        );
    }

    #[test]
    fn reformulation_adds_missing_terms() {
        let (store, embedder, config) = sample();
        let agent = RetrievalAgent::new(&store, &embedder, &config);
        let emb = embedder.embed("training");
        let hits = store.search(&emb, 4);
        let next = agent.reformulate("training", &hits);
        assert!(next.len() > "training".len(), "next was {next:?}");
        assert!(next.contains("training"));
    }

    #[test]
    fn empty_store_returns_not_passed() {
        let embedder = Embedder::new(64);
        let store = DocStore::build(Vec::new(), &embedder);
        let config = RagConfig::default();
        let mut agent = RetrievalAgent::new(&store, &embedder, &config);
        let out = agent.retrieve("anything");
        assert!(!out.passed);
        assert!(out.hits.is_empty());
    }
}
