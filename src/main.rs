//! `axiom-rag` command line interface.
//!
//! ```text
//! axiom-rag ingest <dir> --index <index.json> [--config <rag_config.json>]
//! axiom-rag query "<question>" --index <index.json> [--config ...] [--json]
//! axiom-rag eval --index <index.json> --eval-set <eval.json> [--config ...] [--min-score 0.5]
//! axiom-rag improve --index <index.json> --eval-set <eval.json> --config <rag_config.json> [--agent-task-bin ...]
//! axiom-rag serve --index <index.json> [--config ...] [--port 8080]
//! ```
//!
//! Everything works offline. No API keys, no network calls.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use axiom_rag::config::RagConfig;
use axiom_rag::embedding::Embedder;
use axiom_rag::ingest::ingest_dir;
use axiom_rag::store::DocStore;
use axiom_rag::{answer_question, chunking, load_index_and_config, Result};

#[derive(Parser)]
#[command(name = "axiom-rag", version, about = "Agentic RAG, fully offline")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Ingest a directory of .md/.txt/.pdf files into an index.
    Ingest {
        /// Directory to walk recursively.
        dir: PathBuf,
        /// Where to write the index JSON.
        #[arg(long)]
        index: PathBuf,
        /// Optional config (chunking/embedding knobs).
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Ask a question against an index.
    Query {
        /// The question.
        question: String,
        /// Index built by `ingest`.
        #[arg(long)]
        index: PathBuf,
        /// Optional config.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Print machine-readable JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Score the pipeline on an eval set. Exits 1 when mean < --min-score.
    Eval {
        /// Index built by `ingest`.
        #[arg(long)]
        index: PathBuf,
        /// Eval set JSON file.
        #[arg(long)]
        eval_set: PathBuf,
        /// Optional config.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Gate: fail when the mean score is below this.
        #[arg(long, default_value_t = 0.0)]
        min_score: f64,
        /// Suppress the human-readable summary (used by the improve verifier).
        #[arg(long, default_value_t = false)]
        quiet: bool,
    },
    /// Self-improve the retrieval config via the axiom-agent-task loop.
    Improve {
        /// Index built by `ingest`.
        #[arg(long)]
        index: PathBuf,
        /// Eval set JSON file.
        #[arg(long)]
        eval_set: PathBuf,
        /// Config file the loop may edit (created if missing).
        #[arg(long)]
        config: PathBuf,
        /// Path to the `axiom-agent-task` binary.
        #[arg(long, env = "AXIOM_AGENT_TASK_BIN")]
        agent_task_bin: PathBuf,
        /// Max proposals inside the task loop.
        #[arg(long, default_value_t = 8)]
        max_attempts: usize,
    },
    /// Serve the HTTP query API.
    Serve {
        /// Index built by `ingest`.
        #[arg(long)]
        index: PathBuf,
        /// Optional config.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Port to listen on (localhost only).
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Ingest { dir, index, config } => {
            let cfg = RagConfig::load_optional(config.as_deref())?;
            let docs = ingest_dir(&dir)?;
            if docs.is_empty() {
                return Err(format!("no supported documents found in {}", dir.display()).into());
            }
            let embedder = Embedder::new(cfg.embedding_dim);
            let mut chunks = Vec::new();
            for doc in &docs {
                chunks.extend(chunking::chunk_text(
                    &doc.text,
                    &doc.path.to_string_lossy(),
                    cfg.chunk_size_words,
                    cfg.chunk_overlap_words,
                ));
            }
            let store = DocStore::build(chunks, &embedder);
            store.save(&index)?;
            println!(
                "ingested {} documents -> {} chunks (index: {})",
                docs.len(),
                store.len(),
                index.display()
            );
            Ok(())
        }
        Command::Query {
            question,
            index,
            config,
            json,
        } => {
            let (store, cfg) = load_index_and_config(&index, config.as_deref())?;
            let embedder = Embedder::new(cfg.embedding_dim);
            let (answer, outcome) = answer_question(&store, &embedder, &cfg, &question);
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "answer": answer.text,
                        "citations": answer.citations,
                        "attempts": outcome.formulations.len(),
                        "passed": outcome.passed,
                        "abstained": answer.abstained,
                    })
                );
            } else {
                println!("{answer}", answer = answer.text);
                println!();
                for c in &answer.citations {
                    println!("  [{}] {} (chunk {})", c.number, c.source, c.chunk_index);
                }
                println!(
                    "  (retrieval: {} attempt(s), relevance {})",
                    outcome.formulations.len(),
                    if outcome.passed {
                        "passed"
                    } else {
                        "best-effort"
                    }
                );
            }
            Ok(())
        }
        Command::Eval {
            index,
            eval_set,
            config,
            min_score,
            quiet,
        } => {
            let report = axiom_rag::eval::evaluate_from_disk(&index, config.as_deref(), &eval_set)?;
            if !quiet {
                for item in &report.items {
                    println!("  [{:.2}] {}", item.score, item.question);
                    if !item.missing.is_empty() {
                        println!("       missing: {}", item.missing.join(", "));
                    }
                }
                println!("mean score: {:.3}", report.mean_score);
            } else {
                println!("{:.6}", report.mean_score);
            }
            if report.mean_score < min_score {
                std::process::exit(1);
            }
            Ok(())
        }
        Command::Improve {
            index,
            eval_set,
            config,
            agent_task_bin,
            max_attempts,
        } => {
            let best = axiom_rag::improve::run_improve(&axiom_rag::improve::ImproveOptions {
                agent_task_bin,
                index_path: index,
                eval_set,
                config_path: config,
                max_attempts,
                eval_bin: None, // use the current executable as the eval verifier
            })?;
            println!("best config: {best:?}");
            Ok(())
        }
        Command::Serve {
            index,
            config,
            port,
        } => {
            let cfg = RagConfig::load_optional(config.as_deref())?;
            axiom_rag::serve::serve(&index, &cfg, port)
        }
    }
}
