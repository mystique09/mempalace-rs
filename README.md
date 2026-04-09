# mempalace-rs

`mempalace-rs` is the Rust workspace for MemPalace. It currently provides:

- a CLI for initialization, project mining, semantic search, status inspection, and AAAK compression
- a LanceDB-backed drawer store with FastEmbed embeddings
- a SQLite knowledge graph for temporal facts
- a stdio MCP server that exposes the MemPalace tool surface

This repository is not yet a full Rust replacement for the Python package. The checked-in [`mempalace/`](mempalace/README.md) tree is the reference implementation and test corpus while the Rust port catches up.

## Status

- Implemented in Rust today: `init`, `status`, `mine`, `search`, `compress`, the LanceDB store, the knowledge graph library, and the MCP server.
- Not exposed through the Rust CLI yet: direct knowledge-graph subcommands and some Python-only workflows from the reference package.
- Runtime bootstrapping is currently Windows-oriented because the embedding setup looks for `onnxruntime.dll`.

## Workspace Layout

- `bin/mempalace-rs/` - CLI entry point
- `bin/mempalace-mcp/` - stdio MCP server binary
- `crates/core/` - config, AAAK dialect, project miner, knowledge graph, shared types
- `crates/store/` - LanceDB-backed `MemoryStore` implementation
- `crates/mcp/` - MCP tool definitions and server wiring
- `crates/workspace-hack/` - cargo-hakari support crate
- `mempalace/` - reference Python implementation, benchmarks, examples, and upstream docs

## Requirements

- a recent Rust toolchain with Edition 2024 support
- a local ONNX Runtime dynamic library available as `onnxruntime.dll`

The repository already includes a bundled candidate at [`.mempalace-bin/onnxruntime.dll`](.mempalace-bin/onnxruntime.dll). On first run the CLI and MCP server will seed `~/.mempalace/onnxruntime.dll` if it does not exist.

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
cargo run --bin mempalace-rs -- search "lancedb index" .
```

Inspect counts:

```bash
cargo run --bin mempalace-rs -- status
```

Preview AAAK compression output:

```bash
cargo run --bin mempalace-rs -- compress --wing mempalace-rs --dry-run
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
| `search <QUERY> [SCOPE]` | Runs semantic search with optional wing or room filters; when no explicit wing is supplied, the CLI can infer one from `SCOPE` or the current project root |
| `mine <DIR>` | Scans readable project files, chunks them into drawers, and writes them into a wing |
| `compress` | Reads stored drawers and emits lossy AAAK summaries, either to stdout (`--dry-run`) or to a generated output file |

Useful flags:

- `--palace <PATH>` overrides the configured palace path for any command.
- `init --no-onboarding` skips the interactive bootstrap flow.
- `mine --exclude-data-files` skips noisy `.json`, `.csv`, and `.sql` files in data-heavy folders when you want a cleaner index than the Python default.
- `mine --no-gitignore` disables `.gitignore`-aware scanning.
- `search --all-wings` disables implicit wing narrowing.

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
- `fastembed/` - embedding model cache
- `onnxruntime.dll` - seeded runtime library
- `palace/` - LanceDB drawer store by default

Environment variables:

- `MEMPALACE_PALACE_PATH`
- `MEMPAL_PALACE_PATH`

If the chosen palace path contains a legacy `chroma.sqlite3`, the Rust code automatically stores LanceDB data under `palace/lancedb` instead of mixing formats in place.

## Mining Behavior

The current Rust miner is intentionally simple and predictable:

- It scans common text and source extensions such as `.md`, `.rs`, `.py`, `.ts`, `.json`, `.sql`, and `.toml`.
- It skips common build and cache directories such as `.git`, `node_modules`, `.venv`, `.next`, `coverage`, and `target`.
- By default it matches the Python miner and includes readable data files in places like `assets/`, `migrations/`, `fixtures/`, and `seed/`.
- `mine --exclude-data-files` restores the Rust-only cleaner-index behavior by skipping noisy `.json`, `.csv`, and `.sql` files in those folders.
- It uses the first path segment under the project root as the drawer room. Root-level files fall back to `general`.
- It chunks content at roughly 800 characters with 100 characters of overlap and ignores chunks shorter than 50 characters.
- It deduplicates by `source_file`, so rerunning `mine` on an unchanged project skips files already in the store.

## MCP Server

The Rust MCP server is implemented in [`crates/mcp`](crates/mcp/src/lib.rs) and launched by [`bin/mempalace-mcp`](bin/mempalace-mcp/src/main.rs). It exposes 19 tools across four areas:

- palace read/write tools such as `mempalace_status`, `mempalace_search`, and `mempalace_add_drawer`
- knowledge-graph tools such as `mempalace_kg_query`, `mempalace_kg_add`, and `mempalace_kg_invalidate`
- navigation tools such as `mempalace_traverse` and `mempalace_find_tunnels`
- diary tools such as `mempalace_diary_write` and `mempalace_diary_read`

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
