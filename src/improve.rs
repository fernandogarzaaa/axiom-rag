//! Self-improvement through the `axiom-agent-task` transaction loop.
//!
//! When retrieval quality is poor, the agent does not just tweak knobs in
//! memory: it starts a task on the `axiom-agent-task` binary (AXIOM-AETHER,
//! PR #191), proposes candidate `rag_config.json` edits through
//! `task_propose`, and lets the binary apply each candidate transactionally
//! and run the eval verifier. A candidate that regresses the eval score is
//! rolled back byte-for-byte by the binary; a candidate that passes is kept.
//!
//! The binary speaks JSON-RPC 2.0 over stdio:
//!
//! ```text
//! -> {"jsonrpc":"2.0","id":1,"method":"task_start","params":{...}}
//! <- {"jsonrpc":"2.0","id":1,"result":{"task_id":"..."}}
//! ```

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{json, Value};

use crate::config::RagConfig;
use crate::embedding::Embedder;
use crate::eval::{evaluate, load_eval_set};
use crate::store::DocStore;
use crate::Result;

/// Options for one self-improvement run.
pub struct ImproveOptions {
    /// Path to the `axiom-agent-task` binary.
    pub agent_task_bin: PathBuf,
    /// Index built by `axiom-rag ingest`.
    pub index_path: PathBuf,
    /// Eval set JSON file.
    pub eval_set: PathBuf,
    /// Config file the loop is allowed to edit (created if missing).
    pub config_path: PathBuf,
    /// Max proposals the task loop may attempt.
    pub max_attempts: usize,
    /// Binary used for the `eval` verifier command. Defaults to the current
    /// executable; integration tests override this because under `cargo test`
    /// the current executable is the test harness, not `axiom-rag`.
    pub eval_bin: Option<PathBuf>,
}

/// Candidate (top_k, min_score) grid. Small on purpose: every candidate costs
/// a full eval pass plus a verifier run inside the task loop.
fn candidate_grid() -> Vec<(usize, f64)> {
    let mut grid = Vec::new();
    for top_k in [3usize, 5, 8] {
        for min_score in [0.05, 0.08, 0.12] {
            grid.push((top_k, min_score));
        }
    }
    grid
}

/// Quote one shell word with single quotes (POSIX `sh`).
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// A live JSON-RPC session with the agent-task binary.
struct TaskSession {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl TaskSession {
    fn spawn(bin: &Path) -> Result<Self> {
        if !bin.exists() {
            return Err(format!(
                "axiom-agent-task binary not found at {}. Build it with:\n  \
                 git clone https://github.com/fernandogarzaaa/AXIOM-AETHER /tmp/axiom-aether && \
                 cargo build --release --bin axiom-agent-task --manifest-path /tmp/axiom-aether/axiom_engine_rs/Cargo.toml",
                bin.display()
            )
            .into());
        }
        let mut child = Command::new(bin)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("failed to spawn {}: {e}", bin.display()))?;
        let stdin = child.stdin.take().ok_or("no stdin on task binary")?;
        let stdout = child.stdout.take().ok_or("no stdout on task binary")?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        })
    }

    /// Send one request, return the `result` value (or the JSON-RPC error).
    fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let req = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        writeln!(self.stdin, "{}", serde_json::to_string(&req)?)?;
        self.stdin.flush()?;
        let mut line = String::new();
        let n = self.stdout.read_line(&mut line)?;
        if n == 0 {
            return Err("axiom-agent-task closed stdout unexpectedly".into());
        }
        let resp: Value = serde_json::from_str(&line)?;
        if resp.get("id").and_then(|v| v.as_u64()) != Some(id) {
            return Err(format!("mismatched response id: {line}").into());
        }
        if let Some(err) = resp.get("error") {
            return Err(format!("task binary error: {err}").into());
        }
        resp.get("result")
            .cloned()
            .ok_or_else(|| format!("missing result in response: {line}").into())
    }

    fn wait(mut self) -> Result<()> {
        drop(self.stdin);
        let status = self.child.wait()?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("axiom-agent-task exited with {status}").into())
        }
    }
}

/// Run the self-improvement loop. Returns the best config found (which is
/// also written to `config_path` when the task loop accepts it).
pub fn run_improve(opts: &ImproveOptions) -> Result<RagConfig> {
    // The config file must exist: the task binary snapshots `files` at start.
    let base_config = RagConfig::ensure_file(&opts.config_path)?;
    let base_json = serde_json::to_string_pretty(&base_config)?;

    let store = DocStore::load(&opts.index_path)?;
    let items = load_eval_set(&opts.eval_set)?;
    if items.is_empty() {
        return Err("eval set is empty".into());
    }

    // Baseline score with the current config.
    let embedder = Embedder::new(base_config.embedding_dim);
    let baseline = evaluate(&store, &embedder, &base_config, &items).mean_score;
    println!("baseline eval score: {baseline:.3}");

    // Rank candidates locally so the strongest proposals go first.
    let mut ranked: Vec<(f64, RagConfig)> = Vec::new();
    for (top_k, min_score) in candidate_grid() {
        let mut cfg = base_config.clone();
        cfg.top_k = top_k;
        cfg.min_score = min_score;
        let emb = Embedder::new(cfg.embedding_dim);
        let score = evaluate(&store, &emb, &cfg, &items).mean_score;
        let json = serde_json::to_string_pretty(&cfg)?;
        if json == base_json {
            continue; // identical to current: the task loop would dedup it anyway
        }
        ranked.push((score, cfg));
    }
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    println!("{} candidate configs ranked", ranked.len());

    // The verifier: re-run eval with the proposed config; fail on regression.
    let exe = match &opts.eval_bin {
        Some(p) => p.clone(),
        None => std::env::current_exe()?,
    };
    let verify_cmd = format!(
        "{exe} eval --index {index} --eval-set {eval} --config {cfg} --min-score {baseline:.6} --quiet",
        exe = shell_quote(&exe.to_string_lossy()),
        index = shell_quote(&opts.index_path.to_string_lossy()),
        eval = shell_quote(&opts.eval_set.to_string_lossy()),
        cfg = shell_quote(&opts.config_path.to_string_lossy()),
    );

    let cfg_str = opts.config_path.to_string_lossy().to_string();
    let mut session = TaskSession::spawn(&opts.agent_task_bin)?;
    let start = session.call(
        "task_start",
        json!({
            "goal": "improve axiom-rag retrieval config without regressing eval score",
            "verify_cmd": verify_cmd,
            "files": [cfg_str],
            "max_attempts": opts.max_attempts.max(1),
        }),
    )?;
    let task_id = start
        .get("task_id")
        .and_then(|v| v.as_str())
        .ok_or("task_start returned no task_id")?
        .to_string();
    println!("task started: {task_id}");

    // The agent reasons (ranks candidates locally); AXIOM verifies.
    // Propose only the single best candidate: the binary applies it
    // transactionally and the eval gate decides. Proposing more would let a
    // merely-passing-but-worse candidate overwrite a better file state.
    let Some((best_score, best_cfg)) = ranked.into_iter().find(|(score, _)| *score > baseline)
    else {
        let _ = session.call("task_finish", json!({"task_id": task_id, "commit": true}));
        session.wait()?;
        println!("no candidate improves on baseline {baseline:.3}; keeping current config");
        return Ok(base_config);
    };
    println!(
        "proposing best candidate (local score {best_score:.3}): top_k={} min_score={}",
        best_cfg.top_k, best_cfg.min_score
    );
    let content = serde_json::to_string_pretty(&best_cfg)?;
    let outcome = session.call(
        "task_propose",
        json!({
            "task_id": task_id,
            "edits": [{"path": cfg_str, "content": content}],
        }),
    )?;
    let passed = outcome
        .get("passed")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let output = outcome.get("output").and_then(|v| v.as_str()).unwrap_or("");

    let finish = session.call("task_finish", json!({"task_id": task_id, "commit": true}))?;
    let committed = finish
        .get("committed")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    session.wait()?;
    println!("task finished (committed={committed})");
    if passed {
        println!("verifier accepted: best eval score {best_score:.3}");
        Ok(best_cfg)
    } else {
        let reason = output.lines().next().unwrap_or("rejected");
        println!("verifier rejected the best candidate ({reason}); rolled back, keeping baseline");
        Ok(base_config)
    }
}
