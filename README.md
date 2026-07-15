# mempalace-rs

`mempalace-rs` is the Rust workspace for MemPalace. It currently provides:

- a CLI for initialization, project mining, agent-session synchronization, semantic search, status inspection, and AAAK compression
- a CLI `tool` namespace that mirrors the Rust MCP tool surface for skill and agent fallback
- a SQLite drawer store with model2vec-rs (potion-code-16M-v2) embeddings
- a SQLite knowledge graph for temporal facts
- a stdio MCP server that exposes the MemPalace tool surface

This repository is not yet a full Rust replacement for the Python package. The checked-in [`mempalace/`](mempalace/README.md) tree is the reference implementation and test corpus while the Rust port catches up.

## Status

- Implemented in Rust today: `init`, `status`, `mine`, `sync agent-sessions`, `search`, `compress`, `migrate`, `remine`, the `tool <MCP_TOOL>` CLI namespace, the SQLite store, the knowledge graph library, and the MCP server.
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

Synchronize configured Codex and Claude histories:

```bash
cargo run --bin mempalace-rs -- sync agent-sessions --dry-run
cargo run --bin mempalace-rs -- sync agent-sessions
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
| `search <QUERY> [SCOPE]` | Runs content-aware hybrid semantic search with optional wing, room, score, and reference-time filters; when no explicit wing is supplied, the CLI can infer one from `SCOPE` or the current project root |
| `mine <DIR>` | Scans text-like project files, skips common binary/media/archive formats, chunks them into drawers, and writes them into a wing |
| `sync agent-sessions` | Incrementally imports configured Codex conversations, Claude conversations, and Claude durable memory with source-specific filtering and checkpoints |
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
- `sync agent-sessions --source <all|codex|claude>` selects an enabled agent-history source; `--dry-run` reports planned work without writing drawers or checkpoints.
- `migrate --from <PATH>` imports drawers from a legacy ChromaDB SQLite database.
- `search --all-wings` disables implicit wing narrowing.
- `search --min-score <COSINE>` filters results by their original-query cosine similarity without changing fused ranking.
- `search --as-of <TIME>` supplies the reference time for relative phrases such as `10 days ago`; RFC3339, `YYYY-MM-DD`, and LongMemEval date strings are accepted.
- `tool add_drawer --content-kind <KIND>` files an explicit `code`, `conversation`, `documentation`, `diary`, `prose`, or `unknown` kind. Manual additions default to `unknown` for compatibility.
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

### Agent-session synchronization

Native agent history synchronization is disabled unless a source is explicitly
enabled in `config.json`. Merge an `agent_sessions` section like this into the
existing configuration:

```json
{
  "agent_sessions": {
    "sync_on_start": true,
    "interval_seconds": 60,
    "sources": {
      "codex": {
        "enabled": true,
        "path": null,
        "wing": "codex-sessions"
      },
      "claude": {
        "enabled": true,
        "path": null,
        "wing": "claude-sessions",
        "include_memory": true,
        "memory_wing": "claude-memory"
      }
    }
  }
}
```

A null Codex path resolves to `~/.codex/sessions`; a null Claude path resolves
to `~/.claude/projects`. Explicit paths support alternate and cross-platform
installations. Run the manual command once for the visible historical backfill.
The MCP server will not start a surprise first-time backfill; after checkpoints
exist, it starts serving normally and synchronizes only new records in the
background. Set `interval_seconds` to `0` for startup-only synchronization.
Backfill authorization is scoped to the canonical configured root, so changing
that path requires one new explicit manual backfill instead of silently reading
a different history tree in the background.

The adapters stream JSONL and never mine the raw event log. Codex keeps genuine
`user_message` events plus assistant `final_answer` messages and skips subagent
rollouts. Claude follows parent-UUID lineage from genuine prompts to
`end_turn` assistant text and skips sidechains and subagent logs. Both exclude
reasoning/thinking, commentary, tool calls and results, injected instructions,
attachments, system records, and telemetry. Claude Markdown under project
`memory/` directories is indexed separately as durable documentation;
`~/.claude/history.jsonl` is ignored as a duplicate prompt index.

Each source checkpoint stores its last complete-line byte offset and pending
parser state. Unchanged files are skipped by metadata, appended files resume at
that offset, and a partial final line waits for the next pass. Drawer upserts and
checkpoint advancement share one SQLite transaction, while an expiring database
lease prevents multiple MCP processes from performing the same sync concurrently.
Large JSONL files are processed in bounded record/turn batches and embeddings
are generated in bounded batches, so a historical backfill does not accumulate
an entire extracted transcript in RAM.

If the chosen palace path contains a legacy `chroma.sqlite3`, use `mempalace-rs migrate --from <path>` to import drawers into the current SQLite store.

Search streams exact cosine comparisons through a bounded candidate heap and independently retrieves indexed FTS5/BM25 candidates. Every drawer persists a content kind: `code`, `conversation`, `documentation`, `diary`, `prose`, or `unknown`; legacy rows are conservatively backfilled from their wing, room, and source path when possible. Unrecognized and manually filed `unknown` content stays neutral instead of receiving code-only treatment. The original dense query remains shared across all kinds, while bounded code query views apply only to explicit code candidates and scaffold-stripped recall views apply to all non-code candidates. Candidate-aware reranking scopes identifier/path evidence to code, heading/path evidence to documentation, role-labelled retrieval text to conversations, and reliable reference-time evidence to dated memories. Weighted reciprocal-rank fusion combines the channels without a process-local HNSW index. Results expose fused `relevance` for ordering and original-query cosine `similarity` for duplicate thresholds.

The content-aware embedding representation is versioned in store metadata as `content-kind-v3`. Code embeds bounded source, language/symbol, and split-identifier context. Production mining gives conversations stable session and role context, documentation title/heading context, diaries date/agent/topic context, and prose a compact source path, while returned content remains verbatim. Existing stores using an older representation must run one full `mempalace-rs remine`; search refuses to mix incompatible vectors silently. Delete and mine source-backed wings again to generate the new derived retrieval context or structural chunk boundaries; `remine` only recomputes vectors from metadata and retrieval text already stored in each drawer.

## Mining Behavior

The Rust miner keeps source content verbatim while preparing a separate enriched representation for embeddings:

- It mines text-like files by default, including unknown extensions and extensionless source files, while skipping common binary, media, archive, and document formats such as `.swf`, `.fla`, `.png`, `.pdf`, and `.zip`.
- It skips common build and cache directories such as `.git`, `node_modules`, `.venv`, `.next`, `coverage`, and `target`.
- Native Codex and Claude JSONL histories should use `sync agent-sessions`, not the generic project miner; the adapters intentionally remove machine-generated event noise before embedding.
- It always skips the checked-in `multi_domain_queries.json` evaluator corpus so ground-truth queries and labels cannot leak into production retrieval text.
- By default it matches the Python miner and includes readable data files in places like `assets/`, `migrations/`, `fixtures/`, and `seed/`.
- `mine --exclude-data-files` restores the Rust-only cleaner-index behavior by skipping noisy `.json`, `.csv`, and `.sql` files in those folders.
- It uses the first path segment under the project root as the drawer room. Root-level files fall back to `general`.
- Rust files use Tree-sitter to isolate bounded structural units such as match arms, with path, language, enclosing-symbol, symbol, and split-identifier context added only to the embedding representation.
- Other recognized code files use line-aligned chunks with file-level semantic context. Non-code text keeps the roughly 800-character, 100-character-overlap chunker.
- The miner assigns `diary` to diary routes, `conversation` to session/transcript/chat wings, rooms, or paths (including Markdown transcript exports), `code` to recognized source languages, `documentation` to other Markdown/MDX/RST/AsciiDoc and README/PRD/ADR/CHANGELOG files, and `prose` to remaining readable text.
- It deduplicates by `source_file`, so rerunning `mine` on an unchanged project skips files already in the store.

`remine` recomputes embeddings for existing drawers. To adopt new structural chunk boundaries, delete and mine the affected project wing again so its source files are rechunked.

## MCP Server

The Rust MCP server is implemented in [`crates/mcp`](crates/mcp/src/lib.rs) and launched by [`bin/mempalace-mcp`](bin/mempalace-mcp/src/main.rs). It exposes 20 tools across four areas:

- palace read/write tools such as `mempalace_status`, `mempalace_search`, and `mempalace_add_drawer`
- knowledge-graph tools such as `mempalace_kg_query`, `mempalace_kg_add`, and `mempalace_kg_invalidate`
- navigation tools such as `mempalace_traverse` and `mempalace_find_tunnels`
- diary tools such as `mempalace_diary_write` and `mempalace_diary_read`

The same Rust implementation is also available through the CLI as `mempalace-rs tool <tool_name>`, which prints MCP-style JSON and is intended as a fallback path for environments that can invoke CLI commands but not MCP.

`mempalace_search` accepts optional `wing`, `room`, and `as_of` fields and returns each result's `content_kind`. `mempalace_add_drawer` accepts an optional `content_kind`; `mempalace_diary_write` always stores `diary`.

`mempalace_status` also returns the embedded memory protocol and AAAK dialect guidance so an MCP client can learn the expected usage pattern on connect.

When one or more agent-session sources are enabled, the MCP process starts its
incremental sync task only after the stdio service is ready. Synchronization
uses stderr for diagnostics so stdout remains a valid MCP transport.

## Development

Useful commands while working in this repo:

```bash
cargo fmt --all
cargo check --workspace
cargo test --workspace
cargo run --bin mempalace-rs -- --help
```

The Rust LongMemEval adapter exercises the real SQLite store and embedding
model in an isolated temporary palace. It streams the 500-question dataset,
stores one drawer per conversation session using the same user-turn-only raw
mode as the Python benchmark, and can write auditable per-question JSONL:

```bash
curl -fsSL -o /tmp/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
cargo run --release -p mempalace-rs --example longmemeval -- \
  /tmp/longmemeval_s_cleaned.json \
  --output /tmp/mempalace-rs-longmemeval.jsonl
```

Use the checked-in upstream-compatible split for tuning/final separation:

```bash
cargo run --release -p mempalace-rs --example longmemeval -- \
  /tmp/longmemeval_s_cleaned.json \
  --split-file benchmarks/longmemeval_split_50_450.json \
  --subset held-out
```

Run the portable 100-query regression suite across two code projects,
conversations, documentation, diaries, and prose:

```bash
cargo build --release -p mempalace-rs --bin mempalace-rs --example multidomain
target/release/examples/multidomain \
  benchmarks/multi_domain_queries.json --split held-out --check-parity
```

Use `--limit 20` for a smoke run. The benchmark never reads or writes the
configured live palace; it only reuses the configured model cache.
It reports p50/p95 per-question latency. On macOS, build the example first and
wrap `target/release/examples/longmemeval ...` with `/usr/bin/time -l` to
capture peak resident memory without including compilation.
The frozen three-model results and default-model decision are recorded in
[`benchmarks/model_bakeoff.md`](benchmarks/model_bakeoff.md).

Keep these distinctions in mind:

- The Rust workspace lives in `bin/` and `crates/`.
- The Python `mempalace/` tree is not part of the Cargo workspace.
- Python docs are useful for product intent, but they are not an accurate description of current Rust parity.
