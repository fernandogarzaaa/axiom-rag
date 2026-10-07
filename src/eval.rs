//! Offline evaluation: keyword-recall scoring over a question set.
//!
//! Each eval item names keywords the answer must contain. The score is the
//! fraction of expected keywords present in the generated answer text. This
//! is a coarse but fully deterministic signal, good enough to gate the
//! self-improvement loop in [`crate::improve`]: a config change that drops
//! the mean score is a regression, full stop.

use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

use crate::config::RagConfig;
use crate::embedding::Embedder;
use crate::store::DocStore;
use crate::{answer_question, Result};

/// One eval case: a question plus the keywords a correct answer must mention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalItem {
    pub question: String,
    pub expected_keywords: Vec<String>,
}

/// Score detail for one item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemScore {
    pub question: String,
    pub score: f64,
    pub matched: Vec<String>,
    pub missing: Vec<String>,
}

/// Aggregate report over an eval set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub mean_score: f64,
    pub items: Vec<ItemScore>,
}

/// Load an eval set from a JSON file: `[{"question": ..., "expected_keywords": [...]}]`.
pub fn load_eval_set(path: &Path) -> io::Result<Vec<EvalItem>> {
    let raw = std::fs::read_to_string(path)?;
    serde_json::from_str(&raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Score one item: fraction of expected keywords found in the answer.
fn score_item(answer_text: &str, item: &EvalItem) -> ItemScore {
    let lower = answer_text.to_lowercase();
    let mut matched = Vec::new();
    let mut missing = Vec::new();
    for kw in &item.expected_keywords {
        if lower.contains(&kw.to_lowercase()) {
            matched.push(kw.clone());
        } else {
            missing.push(kw.clone());
        }
    }
    let score = if item.expected_keywords.is_empty() {
        1.0
    } else {
        matched.len() as f64 / item.expected_keywords.len() as f64
    };
    ItemScore {
        question: item.question.clone(),
        score,
        matched,
        missing,
    }
}

/// Run the full pipeline for every item and average the scores.
pub fn evaluate(
    store: &DocStore,
    embedder: &Embedder,
    config: &RagConfig,
    items: &[EvalItem],
) -> EvalReport {
    let mut scored = Vec::with_capacity(items.len());
    for item in items {
        let (answer, _) = answer_question(store, embedder, config, &item.question);
        scored.push(score_item(&answer.text, item));
    }
    let mean_score = if scored.is_empty() {
        0.0
    } else {
        scored.iter().map(|s| s.score).sum::<f64>() / scored.len() as f64
    };
    EvalReport {
        mean_score,
        items: scored,
    }
}

/// Evaluate from disk paths. Used by the `eval` CLI and by the
/// self-improvement verifier command.
pub fn evaluate_from_disk(
    index_path: &Path,
    config_path: Option<&Path>,
    eval_set_path: &Path,
) -> Result<EvalReport> {
    let (store, config) = crate::load_index_and_config(index_path, config_path)?;
    let embedder = Embedder::new(config.embedding_dim);
    if store.dim() != embedder.dim() {
        return Err(format!(
            "index was built with embedding dim {}, but config asks for {}",
            store.dim(),
            embedder.dim()
        )
        .into());
    }
    let items = load_eval_set(eval_set_path)?;
    if items.is_empty() {
        return Err("eval set is empty".into());
    }
    Ok(evaluate(&store, &embedder, &config, &items))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::chunk_text;

    fn sample() -> (DocStore, Embedder, RagConfig) {
        let config = RagConfig::default();
        let embedder = Embedder::new(config.embedding_dim);
        let chunks = chunk_text(
            "The Axiom engine performs test-time training on device. It adapts neural networks during inference without any cloud.",
            "a.md",
            50,
            0,
        );
        (DocStore::build(chunks, &embedder), embedder, config)
    }

    #[test]
    fn keyword_recall_scores() {
        let (store, embedder, config) = sample();
        let items = vec![EvalItem {
            question: "What does the Axiom engine do?".into(),
            expected_keywords: vec!["axiom".into(), "test-time training".into(), "device".into()],
        }];
        let report = evaluate(&store, &embedder, &config, &items);
        assert!(
            report.mean_score > 0.5,
            "mean score was {}",
            report.mean_score
        );
    }

    #[test]
    fn missing_keywords_score_zero() {
        let (store, embedder, config) = sample();
        let items = vec![EvalItem {
            question: "What does the Axiom engine do?".into(),
            expected_keywords: vec!["quantum".into(), "photosynthesis".into()],
        }];
        let report = evaluate(&store, &embedder, &config, &items);
        assert_eq!(report.mean_score, 0.0);
    }

    #[test]
    fn empty_expected_keywords_scores_one() {
        let item = EvalItem {
            question: "q".into(),
            expected_keywords: vec![],
        };
        let s = score_item("anything", &item);
        assert_eq!(s.score, 1.0);
    }
}
