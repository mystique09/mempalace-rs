# mempalace-rs

`mempalace-rs` is the Rust workspace for MemPalace. It currently provides:

- a CLI for initialization, project mining, semantic search, status inspection, and AAAK compression
- a CLI `tool` namespace that mirrors the Rust MCP tool surface for skill and agent fallback
- a SQLite drawer store with model2vec-rs (potion-code-16M-v2) embeddings
- a SQLite knowledge graph for temporal facts
- a stdio MCP server that exposes the MemPalace tool surface

This repository is not yet a full Rust replacement for the Python package. The checked-in [`mempalace/`](mempalace/README.md) tree is the reference implementation and test corpus while the Rust port catches up.

## Status

- Implemented in Rust today: `init`, `status`, `mine`, `search`, `compress`, `migrate`, `remine`, the `tool <MCP_TOOL>` CLI namespace, the SQLite store, the knowledge graph library, and the MCP server.
- Still missing from Rust parity: some Python-only workflows from the reference package.

## Workspace Layout

- `bin/mempalace-rs/` - CLI entry point
- `bin/mempalace-mcp/` - stdio MCP server binary
- `crates/core/` - config, AAAK dialect, project miner, knowledge graph, shared types
- `crates/store/` - SQLite-backed `MemoryStore` implementation with model2vec-rs embeddings
- `crates/mcp/` - MCP tool definitions and server wiring
- `crates/workspace-hack/` - cargo-hakari support crate
- `mempalace/` - reference Python implementation, benchmarks, examples, and upstream docs

## Requirements

- a recent Rust toolchain with Edition 2024 support

## Quick Start

Build the workspace:

```bash
cargo build --workspace
```

Initialize MemPalace state:

```bash
cargo run --bin mempalace-rs -- init
```

Initialize a project and write a local `mempalace.yml` scaffold:

```bash
cargo run --bin mempalace-rs -- init . --force
```

Mine a repository into the palace:

```bash
cargo run --bin mempalace-rs -- mine . --wing mempalace-rs
```

Search stored drawers:

```bash
cargo run --bin mempalace-rs -- search "memory search" .
```

Inspect counts:

```bash
cargo run --bin mempalace-rs -- status
```

Preview AAAK compression output:

```bash
cargo run --bin mempalace-rs -- compress --wing mempalace-rs --dry-run
```

Call an MCP-style tool through the CLI:

```bash
cargo run --bin mempalace-rs -- tool list_wings
```

Run the MCP server over stdio:

```bash
cargo run --bin mempalace-mcp
```

## CLI Surface

| Command | What it does |
| --- | --- |
| `init [DIR]` | Creates global config if needed, optionally writes a project `mempalace.yml`, and runs first-time onboarding in interactive terminals |
| `status` | Prints total drawer count, store path, KG path, and room counts by wing |
| `search <QUERY> [SCOPE]` | Runs hybrid semantic and code search with optional wing or room filters; when no explicit wing is supplied, the CLI can infer one from `SCOPE` or the current project root |
| `mine <DIR>` | Scans text-like project files, skips common binary/media/archive formats, chunks them into drawers, and writes them into a wing |
| `compress` | Reads stored drawers and emits lossy AAAK summaries, either to stdout (`--dry-run`) or to a generated output file |
| `migrate` | Migrates drawers from a legacy ChromaDB SQLite database into the current SQLite store |
| `remine` | Atomically re-embeds existing drawers with the configured model (potion-code-16M-v2 by default) |
| `tool <MCP_TOOL>` | Exposes the Rust MCP tool surface through the CLI and prints MCP-style JSON for automation, skills, and agent fallback |

Useful flags:

- `--palace <PATH>` overrides the configured palace path for any command.
- `--model <REPO_OR_PATH>` selects an embedding model. Changing models requires a full `remine` without `--wing`.
- `init --no-onboarding` skips the interactive bootstrap flow.
- `mine --exclude-data-files` skips noisy `.json`, `.csv`, and `.sql` files in data-heavy folders when you want a cleaner index than the Python default.
- `mine --no-gitignore` disables `.gitignore`-aware scanning.
- `migrate --from <PATH>` imports drawers from a legacy ChromaDB SQLite database.
- `search --all-wings` disables implicit wing narrowing.
- `search --min-score <COSINE>` filters results by their original-query cosine similarity without changing fused ranking.
- `tool --help` lists the mirrored tool names, and `tool <name> --help` shows the tool-specific flags.

## Onboarding And Generated Files

On first run, `init` can ask about work and personal context. That flow writes:

- `~/.mempalace/config.json`
- `~/.mempalace/entity_registry.json`
- `~/.mempalace/people_map.json`
- `~/.mempalace/aaak_entities.md`
- `~/.mempalace/critical_facts.md`

If you pass a project directory to `init`, the CLI also writes a local `mempalace.yml` scaffold and may write a project-local `entities.json` when entity detection finds confirmed people or projects. The scaffold uses the directory name as the default wing and derives room names from top-level folders like `src`, `docs`, `assets`, or `tests`.

## Storage And Configuration

Default runtime files live under `~/.mempalace/`:

- `config.json` - top-level MemPalace config
- `entity_registry.json` - global onboarding registry for people, aliases, and projects
- `people_map.json` - optional nickname to canonical-name map
- `knowledge_graph.sqlite3` - temporal fact storage
- `palace/` - SQLite drawer store by default (`store.sqlite3`)

Environment variables:

- `MEMPALACE_PALACE_PATH`
- `MEMPAL_PALACE_PATH`

If the chosen palace path contains a legacy `chroma.sqlite3`, use `mempalace-rs migrate --from <path>` to import drawers into the current SQLite store.

Search streams exact cosine comparisons through a bounded candidate heap and independently retrieves indexed FTS5/BM25 candidates. A bounded set of code-oriented query views bridges common vocabulary differences such as login, join, connect, and authenticate. Weighted reciprocal-rank fusion combines the dense and lexical channels before identifier-aware reranking. This keeps search independent of the process-local, stale HNSW state that affected the former vectorlite implementation. Results expose fused `relevance` for ordering and the original-query cosine `similarity` for duplicate thresholds.

The source-path and split-identifier embedding representation is versioned in store metadata. Existing stores created with raw-content-only vectors must run one full `mempalace-rs remine` after upgrading; search refuses to mix the two representations silently. That operation cannot reconstruct AST context for old chunks, so code wings must still be deleted and mined again to gain structural chunk boundaries and enclosing-symbol metadata.

## Mining Behavior

The Rust miner keeps source content verbatim while preparing a separate enriched representation for embeddings:

- It mines text-like files by default, including unknown extensions and extensionless source files, while skipping common binary, media, archive, and document formats such as `.swf`, `.fla`, `.png`, `.pdf`, and `.zip`.
- It skips common build and cache directories such as `.git`, `node_modules`, `.venv`, `.next`, `coverage`, and `target`.
- By default it matches the Python miner and includes readable data files in places like `assets/`, `migrations/`, `fixtures/`, and `seed/`.
- `mine --exclude-data-files` restores the Rust-only cleaner-index behavior by skipping noisy `.json`, `.csv`, and `.sql` files in those folders.
- It uses the first path segment under the project root as the drawer room. Root-level files fall back to `general`.
- Rust files use Tree-sitter to isolate bounded structural units such as match arms, with path, language, enclosing-symbol, symbol, and split-identifier context added only to the embedding representation.
- Other recognized code files use line-aligned chunks with file-level semantic context. Non-code text keeps the roughly 800-character, 100-character-overlap chunker.
- It deduplicates by `source_file`, so rerunning `mine` on an unchanged project skips files already in the store.

`remine` recomputes embeddings for existing drawers. To adopt new structural chunk boundaries, delete and mine the affected project wing again so its source files are rechunked.

## MCP Server

The Rust MCP server is implemented in [`crates/mcp`](crates/mcp/src/lib.rs) and launched by [`bin/mempalace-mcp`](bin/mempalace-mcp/src/main.rs). It exposes 19 tools across four areas:

- palace read/write tools such as `mempalace_status`, `mempalace_search`, and `mempalace_add_drawer`
- knowledge-graph tools such as `mempalace_kg_query`, `mempalace_kg_add`, and `mempalace_kg_invalidate`
- navigation tools such as `mempalace_traverse` and `mempalace_find_tunnels`
- diary tools such as `mempalace_diary_write` and `mempalace_diary_read`

The same Rust implementation is also available through the CLI as `mempalace-rs tool <tool_name>`, which prints MCP-style JSON and is intended as a fallback path for environments that can invoke CLI commands but not MCP.

`mempalace_status` also returns the embedded memory protocol and AAAK dialect guidance so an MCP client can learn the expected usage pattern on connect.

## Development

Useful commands while working in this repo:

```bash
cargo fmt --all
cargo check --workspace
cargo test --workspace
cargo run --bin mempalace-rs -- --help
```

Keep these distinctions in mind:

- The Rust workspace lives in `bin/` and `crates/`.
- The Python `mempalace/` tree is not part of the Cargo workspace.
- Python docs are useful for product intent, but they are not an accurate description of current Rust parity.
