# AGENTS.md

## Scope

This repository contains two codebases:

- the active Rust workspace in `bin/` and `crates/`
- a checked-in Python reference implementation in `mempalace/`

Default to changing the Rust workspace unless the task explicitly asks for Python parity work or reference updates.

## What This Repo Ships Today

The Rust code currently owns:

- the `mempalace-rs` CLI
- the `mempalace-mcp` stdio MCP server
- the SQLite/FTS5 drawer store
- the AAAK compression utilities
- the SQLite knowledge graph library

The Rust CLI surface is intentionally smaller than the Python package. Do not document or imply Python-only commands as if they already exist in Rust.

## Repository Map

- `bin/mempalace-rs/src/main.rs` - CLI command definitions and onboarding flow
- `bin/mempalace-mcp/src/main.rs` - MCP binary entry point
- `crates/core/src/config.rs` - config loading, default paths, and legacy Chroma handoff
- `crates/core/src/project_miner.rs` - file scanning, chunking, room detection, and mining heuristics
- `crates/core/src/knowledge_graph.rs` - temporal fact graph stored in SQLite
- `crates/core/src/aaak.rs` - AAAK dialect and compression helpers
- `crates/store/src/lib.rs` - SQLite storage, exact-cosine/FTS5 search, and embeddings
- `crates/mcp/src/lib.rs` - MCP tool implementations
- `mempalace/` - reference Python implementation, examples, and upstream docs

## Working Rules

- Keep the README honest about current Rust functionality.
- Treat `mempalace/README.md` as product context, not as the source of truth for Rust parity.
- The embedding backend uses model2vec-rs (pure Rust, no C dependencies). Models auto-download from HuggingFace Hub on first run.
- Do not rely on generated or runtime directories such as `target/`, `palace/`, or `palce/` as stable source inputs.
- If you change CLI flags or command behavior, update `README.md`.
- If you change the MCP tool surface, update the docs that describe the tool list.

## Verification

Run the narrowest useful check for the code you touched. Common commands:

```bash
cargo fmt --all
cargo check --workspace
cargo test --workspace
cargo test -p mempalace-core
cargo test -p mempalace-store
cargo run --bin mempalace-rs -- --help
```

For CLI changes, verify the relevant `--help` output. For mining, config, AAAK, or storage changes, prefer the crate-level tests in the crate you modified.

## Implementation Notes

- `init` can run an interactive onboarding flow unless `--no-onboarding` is passed. That flow writes config, people maps, critical facts, and AAAK entity files under `~/.mempalace/`.
- `mine` skips common build and cache directories, skips noisy data files by default, uses the first project path segment as the room, and deduplicates by `source_file`.
- `search` can infer the wing from an explicit scope path or from the current working directory when it looks like a project root.
- Search streams exact cosine comparisons through a bounded top-k heap and fuses them with an independent FTS5 ranking; do not reintroduce a process-local vector index.
- Rust mining uses Tree-sitter structural chunks and stores enriched embedding text separately from verbatim drawer content. Other code uses line-aligned chunks; non-code keeps the text chunker.
- The default embedding model is `minishlab/potion-code-16M-v2`. Stored model metadata must match the configured model; use a full `remine` to change it.
- `remine` recomputes vectors for existing drawers but does not rechunk files. Delete and mine a wing again when adopting new structural chunking behavior.
- `compress` is lossy. Keep the docs and code comments explicit about that.
- The knowledge graph exists in Rust as a library and MCP tool set even though there are no Rust CLI subcommands for it yet.

## When Porting From Python

- Port behavior intentionally, not by copy-pasting feature claims.
- Prefer bringing over tests or adding Rust equivalents when you port logic.
- Keep naming and UX aligned where it makes sense, but do not hide current gaps between the implementations.
