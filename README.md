# axiom-rag

Agentic retrieval-augmented generation in Rust. Fully offline, zero API keys.

Instead of single-shot RAG, `axiom-rag` runs a **verifier-gated agentic loop**:
each retrieval is checked for relevance and reformulated on failure, and every
generated sentence must be supported by a cited chunk or it is dropped. When
retrieval quality slips, the agent tunes its own configuration through the
[`axiom-agent-task`](https://github.com/fernandogarzaaa/AXIOM-AETHER) transaction
loop: candidate configs are applied, verified by an eval gate, and rolled back
on regression.

## Architecture

```text
                        +------------------+
                        |  ingest <dir>    |
                        |  md / txt / pdf  |
                        +--------+---------+
                                 | chunk (sliding window + overlap)
                                 v
                        +------------------+
                        |  hash embeddings |  deterministic, no model,
                        |  (feature hash)  |  no download, no API key
                        +--------+---------+
                                 | cosine
                                 v
   +----------+   +-------> +-----------+  relevance   +--------------+
   | question +---+         | RETRIEVAL |  verifier    | reformulate  |
   +----------+   |         |   loop    +----(fail)--->+ (pseudo-      |
                  |         +-----+-----+              |  relevance    |
                  |               | (pass: top-k)      |  feedback)    |
                  |               v                    +------+-------+
                  |      +-----------------+                  | max N attempts,
                  |      |   GENERATION    |                  | fingerprints remembered
                  |      | (extractive,    |<-----------------+
                  |      |  cited)         |
                  |      +--------+--------+
                  |               | claim verifier
                  |               v
                  |      +-----------------+
                  +----> |  answer + [n]   |
                         |  citations      |
                         +-----------------+

   self-improvement:  improve --> axiom-agent-task (JSON-RPC/stdio)
                       proposes rag_config.json edits --> eval verifier
                       pass: keep | fail: byte-for-byte rollback
```

## Quickstart

```bash
cargo build --release
./target/release/axiom-rag ingest ./docs --index ./index.json
./target/release/axiom-rag query "What does the Axiom engine do?" --index ./index.json
```

Example output:

```text
The Axiom engine performs test-time training directly on device. [1] It adapts
neural networks during inference without any cloud connection. [1]

  [1] docs/axiom.md (chunk 0)
  [2] docs/axiom.md (chunk 1)
  (retrieval: 1 attempt(s), relevance passed)
```

Evaluate on a question set (keyword recall, deterministic):

```bash
./target/release/axiom-rag eval --index ./index.json \
  --eval-set eval/example_eval.json --min-score 0.5
```

Self-improve the retrieval config (needs the AXIOM binary):

```bash
git clone https://github.com/fernandogarzaaa/AXIOM-AETHER /tmp/axiom-aether
cargo build --release --bin axiom-agent-task \
  --manifest-path /tmp/axiom-aether/axiom_engine_rs/Cargo.toml

./target/release/axiom-rag improve --index ./index.json \
  --eval-set eval/example_eval.json --config ./rag_config.json \
  --agent-task-bin /tmp/axiom-aether/axiom_engine_rs/target/release/axiom-agent-task
```

Serve the HTTP API:

```bash
./target/release/axiom-rag serve --index ./index.json --port 8080
curl -X POST localhost:8080/query -d '{"question":"What does the Axiom engine do?"}'
```

## Design decisions

**Hash embeddings, not model embeddings.** Feature hashing (Weinberger et al.,
2009) maps tokens into a fixed vector with random signs. It is deterministic
across runs and machines, needs no download, and captures lexical overlap
well. The price is coarse semantics on paraphrase; the agentic retrieval loop
exists precisely to compensate: when scores are low, the query is reformulated
with terms harvested from retrieved chunks (pseudo-relevance feedback) and
retried, with failed formulations fingerprinted and never repeated
(AttemptMemory pattern, mirroring AXIOM-AETHER's `agent_task` module).

**Extractive generation.** The answer is composed of verbatim sentences from
retrieved chunks. An extractive answer cannot invent facts; the remaining risk
is an irrelevant pick, which the claim verifier eliminates: each sentence must
be supported by its cited chunk (cosine or token-recall), or it is dropped. If
nothing survives, the system abstains instead of answering. Abstention is a
feature.

**Brute-force search.** Exact cosine over all chunks. Simple, correct, and fast
enough to tens of thousands of chunks; an approximate index would only add
error the verifier then has to reason about.

**AXIOM as the improvement backbone.** `axiom-rag improve` does not tweak knobs
in memory. It opens a task on the `axiom-agent-task` binary, proposes
`rag_config.json` candidates, and the binary applies each one as an
all-or-nothing transaction and runs the eval gate (`axiom-rag eval
--min-score <baseline>`). Regressions are rolled back byte-for-byte;
identical candidates are deduplicated by the binary's AttemptMemory. The RAG
agent reasons and proposes; AXIOM owns verification and rollback.

## Project layout

```text
src/
  main.rs        CLI (ingest / query / eval / improve / serve)
  lib.rs         answer_question pipeline + shared Result
  config.rs      RagConfig (rag_config.json)
  chunking.rs    sliding-window chunker
  embedding.rs   FNV feature-hashing embedder
  ingest.rs      md / txt / pdf loading + recursive walk
  store.rs       vector store + JSON persistence
  retrieval.rs   agentic loop + AttemptMemory + reformulation
  generation.rs  extractive cited answers (+ abstention)
  verifier.rs    claim-support checks
  eval.rs        keyword-recall eval sets
  improve.rs     axiom-agent-task driver (stdio JSON-RPC)
  serve.rs       tiny_http API
tests/
  integration.rs end-to-end: ingest -> query -> verify -> eval -> improve
eval/
  example_eval.json
```

## Guarantees

* No network calls, no API keys, no model downloads. Ever.
* `cargo test` covers chunking, embeddings, retrieval loop, verifier,
  generation, eval, and the full pipeline.
* CI runs build, tests (including the improve loop against a real
  `axiom-agent-task` binary built from AXIOM-AETHER main), clippy with
  `-D warnings`, and `cargo fmt --check`.

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).
