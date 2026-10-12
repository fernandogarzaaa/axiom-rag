# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-10-13

### Fixed
- PDF ingestion: a panic inside `pdf-extract` on a malformed PDF is now caught
  and reported as an `InvalidData` error naming the file, instead of aborting
  the whole process.

### Changed
- CI builds AXIOM-AETHER's `axiom-agent-task` from a pinned commit
  (`87ea1f3`) instead of whatever is on its `main` branch, so this repo's CI
  result no longer changes when another repo changes.

## [0.1.0] - 2026-10-08

- Initial release: offline agentic RAG CLI and HTTP API (md/txt/pdf ingest,
  feature-hash embeddings, verifier-gated retrieval loop, extractive cited
  answers, `improve` loop via AXIOM-AETHER).
