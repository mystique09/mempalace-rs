---
name: mempalace-cli-fallback
description: Use when working with MemPalace from a coding agent environment that cannot use MCP tools but can run shell commands or skills. Triggers: MCP is unavailable, only SKILL.md is supported, the agent needs MemPalace status/search/knowledge-graph/diary access through the CLI, or the user asks to use `mempalace-rs tool ...` as a fallback.
---

# MemPalace CLI Fallback

Use this skill when MemPalace MCP tools are unavailable but the agent can run commands in this repository.

The Rust CLI mirrors the MCP tool surface through:

```bash
cargo run --bin mempalace-rs -- tool <tool_name> [flags]
```

Run commands from the repository root.

If the binary is already built and startup latency matters, use:

```bash
target\debug\mempalace-rs.exe tool <tool_name> [flags]
```

## Required Protocol

Follow the same protocol the MCP server expects:

1. On wake-up, call `tool status`.
2. Before responding about any person, project, or past event, call `tool kg_query` or `tool search` first.
3. If unsure about a fact, say "let me check" and run a tool command.
4. After the session, write a diary entry with `tool diary_write`.
5. When facts change, run `tool kg_invalidate` on the old fact and `tool kg_add` for the new fact.

## Read Commands

- `tool status`
- `tool list_wings`
- `tool list_rooms [--wing <wing>]`
- `tool get_taxonomy`
- `tool get_aaak_spec`
- `tool search --query <text> [--limit <n>] [--wing <wing>] [--room <room>]`
- `tool kg_query --entity <name> [--as-of <YYYY-MM-DD>] [--direction outgoing|incoming|both]`
- `tool kg_timeline [--entity <name>]`
- `tool kg_stats`
- `tool diary_read --agent-name <agent> [--last-n <n>]`
- `tool traverse --start-room <room> [--max-hops <n>]`
- `tool find_tunnels [--wing-a <wing>] [--wing-b <wing>]`
- `tool graph_stats`

## Write Commands

- `tool check_duplicate --content <text> [--threshold <float>]`
- `tool add_drawer --wing <wing> --room <room> --content <text> [--source-file <path>] [--added-by <name>]`
- `tool delete_drawer --drawer-id <id>`
- `tool kg_add --subject <name> --predicate <relation> --object <name> [--valid-from <YYYY-MM-DD>] [--source-closet <label>]`
- `tool kg_invalidate --subject <name> --predicate <relation> --object <name> [--ended <YYYY-MM-DD>]`
- `tool diary_write --agent-name <agent> --entry <aaak-or-text> [--topic <topic>]`

## Output Shape

The CLI prints pretty JSON that mirrors the MCP tool responses.

Read the JSON directly instead of scraping prose. Common keys:

- `status`: `total_drawers`, `wings`, `rooms`, `protocol`, `aaak_dialect`
- `search`: `results`
- `kg_query`: `facts`
- `diary_read`: `entries`
- `check_duplicate`: `is_duplicate`, `matches`

## Example Flow

When waking up in a non-MCP environment:

```bash
cargo run --bin mempalace-rs -- tool status
```

When checking a fact before answering:

```bash
cargo run --bin mempalace-rs -- tool kg_query --entity "Riley"
```

When searching broadly instead of querying one entity:

```bash
cargo run --bin mempalace-rs -- tool search --query "auth decision"
```

When recording the session:

```bash
cargo run --bin mempalace-rs -- tool diary_write --agent-name codex --topic general --entry "SESSION:2026-04-17|checked.facts.before.reply|updated.cli.skill.bridge"
```

## Guardrails

- Prefer `kg_query` for entity facts and relationships.
- Prefer `search` for free-text recall.
- Use `check_duplicate` before filing verbatim content with `add_drawer`.
- Keep diary entries compact; AAAK is preferred but not required by the CLI.
- If the user only needs the normal human-facing CLI, do not force the `tool` namespace.
