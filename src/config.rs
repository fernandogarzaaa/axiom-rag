//! Retrieval hyperparameters, serializable to `rag_config.json`.
//!
//! The file is the unit of self-improvement: [`crate::improve`] proposes edits
//! to it through the `axiom-agent-task` transaction loop, and the eval verifier
//! gates acceptance.

use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// All tunable knobs of the pipeline in one place.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RagConfig {
    /// Sliding-window size in words.
    pub chunk_size_words: usize,
    /// Overlap between consecutive windows in words.
    pub chunk_overlap_words: usize,
    /// Feature-hashing embedding dimension.
    pub embedding_dim: usize,
    /// Number of chunks handed to generation.
    pub top_k: usize,
    /// Minimum cosine score for the retrieval loop to accept a result set.
    pub min_score: f64,
    /// Maximum query reformulations per question.
    pub max_attempts: usize,
    /// Minimum chunk support for an emitted sentence to survive verification.
    pub claim_threshold: f64,
    /// Cap on sentences in a generated answer.
    pub max_answer_sentences: usize,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            chunk_size_words: 200,
            chunk_overlap_words: 50,
            embedding_dim: 512,
            top_k: 5,
            min_score: 0.08,
            max_attempts: 3,
            claim_threshold: 0.12,
            max_answer_sentences: 6,
        }
    }
}

impl RagConfig {
    /// Read JSON config from `path`.
    pub fn load(path: &Path) -> io::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        serde_json::from_str(&raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Write JSON config to `path` (pretty-printed for human editing).
    pub fn save(&self, path: &Path) -> io::Result<Self> {
        let raw = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, raw)?;
        Ok(self.clone())
    }

    /// Load from `path`, or fall back to defaults when `path` is `None`.
    /// A present-but-unreadable file is an error, never silently ignored.
    pub fn load_optional(path: Option<&Path>) -> io::Result<Self> {
        match path {
            Some(p) => Self::load(p),
            None => Ok(Self::default()),
        }
    }

    /// Ensure a config file exists at `path`, writing defaults if missing.
    /// Returns the effective config.
    pub fn ensure_file(path: &Path) -> io::Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            let cfg = Self::default();
            cfg.save(path)?;
            Ok(cfg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("rag_config.json");
        let cfg = RagConfig::default();
        cfg.save(&p).unwrap();
        let back = RagConfig::load(&p).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn ensure_file_writes_defaults_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("rag_config.json");
        let cfg = RagConfig::ensure_file(&p).unwrap();
        assert_eq!(cfg, RagConfig::default());
        assert!(p.exists());
    }

    #[test]
    fn load_missing_file_errors() {
        let err = RagConfig::load(Path::new("/does/not/exist.json")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }
}
