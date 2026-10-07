//! End-to-end tests: ingest a corpus, query it, verify citations, run eval,
//! and (when the binary is available) run the self-improvement loop.

use std::fs;
use std::path::{Path, PathBuf};

use axiom_rag::chunking::chunk_text;
use axiom_rag::config::RagConfig;
use axiom_rag::embedding::Embedder;
use axiom_rag::ingest::ingest_dir;
use axiom_rag::store::DocStore;
use axiom_rag::verifier::verify_answer;
use axiom_rag::{answer_question, Result};

fn write_doc(dir: &Path, name: &str, content: &str) -> PathBuf {
    let p = dir.join(name);
    fs::write(&p, content).unwrap();
    p
}

fn sample_corpus(dir: &Path) {
    write_doc(
        dir,
        "axiom.md",
        "# Axiom Engine\n\nThe Axiom engine performs test-time training directly on device. \
         It adapts neural networks during inference without any cloud connection. \
         The engine uses a local optimizer named Adam for fast adaptation.",
    );
    write_doc(
        dir,
        "rag.md",
        "# Retrieval Augmented Generation\n\nRetrieval augmented generation combines a retriever with a generator. \
         The retriever finds relevant chunks by cosine similarity over embeddings. \
         The generator then composes an answer with citations to the retrieved chunks.",
    );
    write_doc(
        dir,
        "notes.txt",
        "Deployment notes: the system runs fully offline. No API keys are required. \
         Embeddings are computed locally with feature hashing.",
    );
}

fn build_index(corpus: &Path, index: &Path, config: &RagConfig) {
    let docs = ingest_dir(corpus).unwrap();
    assert!(!docs.is_empty(), "corpus must contain documents");
    let embedder = Embedder::new(config.embedding_dim);
    let mut chunks = Vec::new();
    for doc in &docs {
        chunks.extend(chunk_text(
            &doc.text,
            &doc.path.to_string_lossy(),
            config.chunk_size_words,
            config.chunk_overlap_words,
        ));
    }
    assert!(!chunks.is_empty());
    DocStore::build(chunks, &embedder).save(index).unwrap();
}

#[test]
fn end_to_end_query_returns_cited_verified_answer() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let corpus = tmp.path().join("corpus");
    fs::create_dir(&corpus)?;
    sample_corpus(&corpus);
    let index = tmp.path().join("index.json");
    let config = RagConfig::default();
    build_index(&corpus, &index, &config);

    let store = DocStore::load(&index)?;
    let embedder = Embedder::new(config.embedding_dim);
    let (answer, outcome) =
        answer_question(&store, &embedder, &config, "What does the Axiom engine do?");
    assert!(!answer.abstained, "answer was: {}", answer.text);
    assert!(!answer.sentences.is_empty());
    assert_eq!(answer.sentences.len(), answer.citations.len());
    assert!(!outcome.formulations.is_empty());

    // Every citation must resolve to a real chunk, and the answer must verify.
    let all_chunks: Vec<_> = (0..store.len()).map(|i| store.chunk(i).clone()).collect();
    for c in &answer.citations {
        assert!(
            all_chunks
                .iter()
                .any(|ch| ch.source == c.source && ch.index == c.chunk_index),
            "dangling citation: {c:?}"
        );
    }
    let bad = verify_answer(&answer.sentences, &answer.citations, &embedder, &config);
    assert!(bad.is_empty(), "unsupported sentences: {bad:?}");
    assert!(answer.text.to_lowercase().contains("axiom"));
    Ok(())
}

#[test]
fn query_abstains_when_no_sentences_extractable() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let corpus = tmp.path().join("corpus");
    fs::create_dir(&corpus)?;
    // Only sub-4-token fragments: the sentence splitter yields nothing,
    // so generation must abstain instead of inventing an answer.
    write_doc(&corpus, "frags.txt", "a b c. d e f. g h i! j k l?");
    let index = tmp.path().join("index.json");
    let config = RagConfig::default();
    build_index(&corpus, &index, &config);

    let store = DocStore::load(&index)?;
    let embedder = Embedder::new(config.embedding_dim);
    let (answer, _) = answer_question(&store, &embedder, &config, "What is here?");
    assert!(answer.abstained);
    assert!(answer.citations.is_empty());
    Ok(())
}

#[test]
fn eval_scores_keyword_recall() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let corpus = tmp.path().join("corpus");
    fs::create_dir(&corpus)?;
    sample_corpus(&corpus);
    let index = tmp.path().join("index.json");
    let config = RagConfig::default();
    build_index(&corpus, &index, &config);

    let eval_path = tmp.path().join("eval.json");
    fs::write(
        &eval_path,
        serde_json::json!([
            {"question": "What does the Axiom engine do?",
             "expected_keywords": ["axiom", "test-time training", "device"]},
            {"question": "How does retrieval augmented generation work?",
             "expected_keywords": ["retriever", "citations", "embeddings"]},
        ])
        .to_string(),
    )?;
    let report = axiom_rag::eval::evaluate_from_disk(&index, None, &eval_path)?;
    assert_eq!(report.items.len(), 2);
    assert!(
        (0.0..=1.0).contains(&report.mean_score),
        "mean {}",
        report.mean_score
    );
    assert!(report.mean_score > 0.3, "mean {}", report.mean_score);
    Ok(())
}

#[test]
fn improve_loop_runs_against_real_binary() -> Result<()> {
    let bin = match std::env::var("AXIOM_AGENT_TASK_BIN") {
        Ok(b) => PathBuf::from(b),
        Err(_) => {
            eprintln!(
                "skipping improve_loop_runs_against_real_binary: AXIOM_AGENT_TASK_BIN not set"
            );
            return Ok(());
        }
    };
    assert!(bin.exists(), "binary missing at {}", bin.display());

    let tmp = tempfile::tempdir()?;
    let corpus = tmp.path().join("corpus");
    fs::create_dir(&corpus)?;
    sample_corpus(&corpus);
    let index = tmp.path().join("index.json");
    let config_path = tmp.path().join("rag_config.json");
    let config = RagConfig::default();
    build_index(&corpus, &index, &config);

    let eval_path = tmp.path().join("eval.json");
    fs::write(
        &eval_path,
        serde_json::json!([
            {"question": "What does the Axiom engine do?",
             "expected_keywords": ["axiom", "device"]},
            {"question": "Is the system offline?",
             "expected_keywords": ["offline"]},
        ])
        .to_string(),
    )?;

    let best = axiom_rag::improve::run_improve(&axiom_rag::improve::ImproveOptions {
        agent_task_bin: bin,
        index_path: index,
        eval_set: eval_path,
        config_path: config_path.clone(),
        max_attempts: 6,
        // Under `cargo test` the current exe is the test harness; point the
        // verifier at the real binary instead.
        eval_bin: Some(PathBuf::from(env!("CARGO_BIN_EXE_axiom-rag"))),
    })?;
    // Baseline is already perfect (1.0), so no candidate can strictly improve:
    // the loop must leave the config untouched and return the base config.
    let saved = RagConfig::load(&config_path)?;
    assert_eq!(saved, RagConfig::default());
    assert_eq!(best, RagConfig::default());
    Ok(())
}
