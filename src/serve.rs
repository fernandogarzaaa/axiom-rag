//! Minimal HTTP API over `tiny_http`.
//!
//! * `GET /health` -> `{"status":"ok","chunks":N}`
//! * `POST /query` with `{"question": "..."}` ->
//!   `{"answer": "...", "citations": [...], "attempts": N, "passed": bool}`
//!
//! The index is loaded once at startup; every request runs the full agentic
//! pipeline (retrieve -> generate -> verify) on a fresh agent.

use std::path::Path;

use serde_json::json;
use tiny_http::{Method, Response, Server};

use crate::config::RagConfig;
use crate::embedding::Embedder;
use crate::store::DocStore;
use crate::{answer_question, Result};

/// Serve forever on `127.0.0.1:port`.
pub fn serve(index_path: &Path, config: &RagConfig, port: u16) -> Result<()> {
    let store = DocStore::load(index_path)?;
    let embedder = Embedder::new(config.embedding_dim);
    if store.dim() != embedder.dim() {
        return Err(format!(
            "index dim {} != config embedding dim {}",
            store.dim(),
            embedder.dim()
        )
        .into());
    }
    let server = Server::http(format!("127.0.0.1:{port}"))
        .map_err(|e| format!("cannot bind 127.0.0.1:{port}: {e}"))?;
    println!("axiom-rag serving on http://127.0.0.1:{port}/");
    for mut req in server.incoming_requests() {
        let response = match (req.method(), req.url()) {
            (Method::Get, "/health") => {
                Response::from_string(json!({"status": "ok", "chunks": store.len()}).to_string())
                    .with_status_code(200)
            }
            (Method::Post, "/query") => {
                let mut body = String::new();
                let parsed = req
                    .as_reader()
                    .read_to_string(&mut body)
                    .ok()
                    .and_then(|_| serde_json::from_str::<serde_json::Value>(&body).ok());
                match parsed.and_then(|v| {
                    v.get("question")
                        .and_then(|q| q.as_str())
                        .map(str::to_string)
                }) {
                    Some(question) if !question.trim().is_empty() => {
                        let (answer, outcome) =
                            answer_question(&store, &embedder, config, &question);
                        Response::from_string(
                            json!({
                                "answer": answer.text,
                                "citations": answer.citations,
                                "attempts": outcome.formulations.len(),
                                "passed": outcome.passed,
                            })
                            .to_string(),
                        )
                        .with_status_code(200)
                    }
                    _ => Response::from_string(
                        json!({"error": "expected JSON body {\"question\": \"...\"}"}).to_string(),
                    )
                    .with_status_code(400),
                }
            }
            _ => Response::from_string(json!({"error": "not found"}).to_string())
                .with_status_code(404),
        };
        let _ = req.respond(response);
    }
    Ok(())
}
