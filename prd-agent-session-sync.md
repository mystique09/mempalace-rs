# PRD: Incremental agent-session synchronization

**Status:** Implemented locally
**Date:** 2026-07-15
**Target:** Rust core, SQLite store, CLI, MCP startup lifecycle, and local configuration

## Problem Statement

MemPalace can search manually exported Codex conversations, but it does not stay
current with the native session histories produced by coding agents. Codex keeps
its active history as large append-only JSONL files, while Claude keeps
project-scoped JSONL transcripts and separate curated memory Markdown files.
Mining those directories with the generic project miner treats every event as
ordinary text. That indexes reasoning, tool traffic, telemetry, system context,
and duplicate payloads; wastes storage and embedding work; weakens retrieval;
and can make MCP startup consume unnecessary CPU and memory.

The local Codex history is currently about 1.4 GB across more than 500 session
files, while the Claude history contains project transcripts plus a smaller set
of high-signal durable memories. Re-reading or re-embedding every indexed file
on every poll is not acceptable. MemPalace needs source-aware adapters that
extract only useful conversational memory, remember exactly how far each source
has been processed, and safely synchronize new records while the agent is still
writing the session.

## Solution

Add an opt-in agent-session synchronization module with Codex and Claude
adapters. Both adapters present one small synchronization interface to the CLI
and MCP server while hiding their different file layouts and event schemas.

The first historical backfill is run explicitly with visible progress. Once a
source is enabled in configuration, the MCP server becomes available normally
and then runs bounded incremental synchronization in the background on startup
and at a configured interval. A manual command invokes the same synchronization
module for backfills, dry runs, troubleshooting, and immediate refreshes.

Each source file has a durable checkpoint. Unchanged files are skipped after a
metadata check. Appended JSONL files are read from the last committed byte
offset, and only complete new records are parsed. New drawers and the new
checkpoint are committed atomically. Deterministic drawer identifiers and a
database-backed lease make retries and concurrent MCP processes idempotent.

Codex synchronization keeps genuine user messages and assistant final answers.
Claude synchronization keeps genuine user prompts and assistant end-of-turn
text. Both exclude reasoning/thinking, commentary, tool calls and results,
attachments, system/developer context, telemetry, and embedded binary data.
Claude's curated project memory Markdown is synchronized separately from its
transcripts and can be searched in its own wing.

## User Stories

1. As a MemPalace user, I want recent Codex conversations synchronized automatically, so that future searches can refer to decisions from my current work.
2. As a MemPalace user, I want recent Claude conversations synchronized automatically, so that memory is not limited to one coding agent.
3. As a MemPalace user, I want Claude's curated project memories synchronized, so that high-signal durable facts remain searchable independently of transcripts.
4. As a MemPalace user, I want automatic synchronization to be opt-in, so that MemPalace never reads agent histories without explicit configuration.
5. As a MemPalace user, I want missing configuration to keep synchronization disabled, so that upgrades preserve current behavior.
6. As a MemPalace user, I want the MCP server to become responsive before background synchronization begins, so that a large backfill cannot delay tool availability.
7. As a MemPalace user, I want an explicit first-run backfill with progress, so that the initial cost is visible and controllable.
8. As a MemPalace user, I want a manual synchronization command, so that I can refresh immediately without restarting MCP.
9. As a MemPalace user, I want a dry run, so that I can see what would be scanned, skipped, replaced, and added before modifying the palace.
10. As a MemPalace user, I want to select Codex, Claude, or all configured sources, so that I can troubleshoot one adapter independently.
11. As a MemPalace user, I want unchanged session files skipped without parsing or embedding, so that routine polling is cheap.
12. As a MemPalace user, I want appended sessions read only from their previous checkpoint, so that active histories do not require repeated full-file work.
13. As a MemPalace user, I want incomplete trailing JSONL records ignored until complete, so that synchronization is safe while an agent is writing.
14. As a MemPalace user, I want unfinished conversational turns retained as pending state, so that a later final answer completes the correct exchange.
15. As a MemPalace user, I want consecutive user steering messages paired with the eventual final response, so that mid-turn clarifications remain understandable.
16. As a MemPalace user, I want imported exchanges to preserve source, session, project, and time metadata, so that search results retain useful context.
17. As a MemPalace user, I want Codex commentary excluded, so that progress updates do not compete with final decisions.
18. As a MemPalace user, I want Codex reasoning and tool traffic excluded, so that private chain-of-thought-like data and repetitive execution logs are not indexed.
19. As a MemPalace user, I want Claude thinking and tool traffic excluded, so that only conversational outcomes become memory.
20. As a MemPalace user, I want agent system and developer instructions excluded, so that injected context does not dominate semantic retrieval.
21. As a MemPalace user, I want Claude prompt history ignored when the same prompts exist in project transcripts, so that duplicate sources do not distort ranking.
22. As a MemPalace user, I want changed Claude memory files replaced individually, so that editing one memory does not rebuild every Claude source.
23. As a MemPalace user, I want retries to reuse deterministic drawer identifiers, so that crashes cannot create duplicate exchanges.
24. As a MemPalace user, I want checkpoint updates and drawer writes committed together, so that a crash cannot mark uncommitted data as processed.
25. As a MemPalace user, I want concurrent Codex and Claude MCP clients to share a database lease, so that only one process performs synchronization work at a time.
26. As a MemPalace user, I want an expired lease recoverable after a crash, so that background synchronization resumes without manual database repair.
27. As a MemPalace user, I want a parser-version change to trigger targeted reconciliation, so that improved filtering corrects old imported data.
28. As a MemPalace user, I want truncated, replaced, or edited session files reconciled individually, so that unusual file changes do not require a global remine.
29. As a MemPalace user, I want deleted source histories left untouched by default, so that automatic synchronization is non-destructive.
30. As a MemPalace user, I want separate default wings for Codex sessions, Claude sessions, and Claude memory, so that I can scope noisy history when needed.
31. As a MemPalace user, I want verbatim user and final-assistant text returned by search, so that stored memory does not silently rewrite the conversation.
32. As a MemPalace user, I want enriched retrieval text kept separate from verbatim drawer content, so that metadata improves ranking without changing displayed results.
33. As a MemPalace user, I want background failures written only to stderr, so that MCP's stdio protocol remains valid.
34. As a MemPalace user, I want a concise synchronization summary, so that I can see scanned, skipped, appended, reconciled, and written counts.
35. As a maintainer, I want both lifecycle callers to use one synchronization interface, so that CLI and MCP behavior cannot drift.
36. As a maintainer, I want source-specific parsing behind adapters, so that future agents can be added without modifying CLI or MCP orchestration.
37. As a maintainer, I want parser tests built from small synthetic fixtures, so that tests never depend on a developer's private session history.
38. As a maintainer, I want store tests to exercise real SQLite transactions, so that checkpoint atomicity and idempotency are verified at the persistence seam.
39. As a maintainer, I want background-loop tests to use bounded run-once behavior, so that the suite is deterministic and does not sleep indefinitely.
40. As a maintainer, I want existing configuration files to deserialize unchanged, so that the feature is backward compatible.

## Implementation Decisions

- Build one agent-session synchronization module whose external interface runs one bounded synchronization pass and returns a structured report. The CLI and MCP lifecycle are callers of this same interface.
- Define a source-adapter seam because Codex and Claude are two genuinely different formats. Adapters own discovery, event filtering, completion rules, source metadata extraction, and resumable parser state.
- Add source configuration for Codex and Claude under an agent-session section. Each source is disabled by default. Automatic startup synchronization and the polling interval are configurable.
- Resolve default paths to the current user's native Codex sessions directory and Claude projects directory. Explicit configured paths override discovery and support cross-platform installations.
- Use separate default wings for Codex conversations, Claude conversations, and Claude durable memory. The wing names remain configurable.
- Treat the manual command as both the initial-backfill path and an operational escape hatch. It supports source selection and dry-run reporting.
- Store one checkpoint per adapter and canonical source file. The checkpoint records file size, high-resolution modification time, the last committed complete-line byte offset, a fingerprint of bytes immediately before that offset, parser version, serialized pending-turn state, and update time.
- Skip a file without opening it when size, modification time, and parser version match its checkpoint.
- Treat a larger file as an append only when the stored pre-offset fingerprint still matches. Seek directly to the committed offset and parse complete new lines.
- Treat truncation, replacement, same-size modification, fingerprint mismatch, or parser-version mismatch as a targeted full reconciliation of that source.
- Leave an incomplete final JSONL line uncommitted. The next pass resumes at the beginning of that line.
- Assign deterministic drawer identifiers from adapter identity, canonical source identity, and a stable turn anchor. Reprocessing the same turn therefore updates the same drawer.
- Commit drawer upserts, optional source replacement, and checkpoint advancement in one SQLite transaction. Embeddings may be computed before the transaction, but no checkpoint advances unless every database write commits.
- Coordinate automatic and manual synchronizers with a renewable, expiring database lease. A process that cannot acquire the lease reports a skipped pass rather than racing another process.
- Record explicit initialization per adapter and canonical configured root. Existing checkpoints under one root never authorize a surprise historical backfill from a newly configured root.
- Stream source files in bounded record/turn batches, atomically advance the checkpoint after each committed batch, and bound embedding batches. Never load a complete multi-megabyte or gigabyte session into memory.
- Keep full Claude UUID lineage only for unfinished turns. After completion, retain only prompt anchors in a bounded recent window so branch regeneration remains useful without making checkpoint memory grow with total tool history.
- For Codex, prefer the explicit user-message event payload for genuine user text and accept only response messages marked as assistant final answers. Ignore assistant commentary, agent messages, reasoning, tool traffic, telemetry, developer messages, and injected project instructions.
- For Claude, accept non-meta user records that are not tool results and extract only user text content. Accept assistant text blocks only from end-of-turn messages. Ignore thinking blocks, tool-use messages, tool results, attachments, system records, snapshots, mode changes, and title metadata.
- Accumulate consecutive genuine user messages until the adapter observes a final assistant response. Persist unfinished accumulated messages in the checkpoint cursor.
- Represent each completed exchange as verbatim role-labelled content plus separate retrieval text containing trustworthy agent, session, project, and date context.
- Discover Claude durable memory Markdown beneath project memory directories. Chunk and enrich those files as documentation, use file-level replacement on change, and ignore Claude's global prompt-history JSONL.
- Do not automatically delete drawers when a source disappears. Destructive source cleanup requires a future explicit command.
- Version the adapter parsers independently so filtering fixes reconcile only their affected sources.
- Keep background error output off stdout because stdout is reserved for MCP protocol messages.

## Testing Decisions

- Test behavior at the shared synchronization interface wherever possible. Tests should assert reports, stored drawers, checkpoints, and retries rather than private parsing helpers.
- Use synthetic Codex fixtures containing user messages, commentary, final answers, reasoning, tool calls/results, developer messages, consecutive user steering, and a partial final line.
- Use synthetic Claude fixtures containing genuine string and block user content, meta users, tool results, thinking, tool use, end-of-turn text, attachments, and project memory Markdown.
- Verify that only user and final-assistant text appears in stored conversation drawers and that excluded payload text never appears.
- Verify that an unchanged second pass performs no drawer writes and reports the source as skipped.
- Verify that appending a complete turn reads from the checkpoint and adds only the new exchange.
- Verify that appending an incomplete JSON record does not advance the committed offset until the line is completed.
- Verify that a pending user turn survives one pass and is completed by a later appended final answer.
- Verify that a targeted reconciliation removes stale drawers for that source while preserving every other source.
- Verify deterministic identifiers and upserts by replaying a commit and asserting a stable drawer count.
- Verify atomicity with real SQLite by forcing a write failure and asserting that the checkpoint did not advance.
- Verify lease acquisition, contention, expiry, renewal, and release with two store handles using the same database.
- Verify legacy configuration without the new section still loads with both sources disabled.
- Verify configured paths, wings, startup behavior, intervals, and Claude memory inclusion round-trip through configuration.
- Verify CLI help and argument parsing for the new synchronization command.
- Verify the MCP startup path launches synchronization only when a source is enabled and does not block server construction.
- Use the existing deterministic test embedder for persistence tests, matching the store's current testing approach.
- Run crate-level core and store tests regularly, then run formatting, workspace checking, and the complete workspace test suite before completion.

## Out of Scope

- Importing the abandoned Windows `sessions.7z` archive.
- Uploading session data or embeddings to a hosted service.
- Reading or writing agent histories outside configured local roots.
- Indexing tool calls, tool outputs, reasoning, thinking, system prompts, telemetry, attachments, shell snapshots, file history, or Claude task state.
- Automatic deletion when a source file disappears.
- A filesystem watcher; bounded polling is sufficient for the first implementation.
- Cross-machine synchronization or conflict resolution.
- Summarizing conversations with an LLM before storage.
- Adding new embedding models or changing retrieval ranking as part of ingestion.
- Publishing this PRD to GitHub or another issue tracker.

## Further Notes

- The initial backfill can still be substantial because the adapter must stream
  historical JSONL once, but it should embed only a small fraction of the raw
  events. Subsequent passes should normally perform metadata checks plus a small
  append read.
- A source may be actively written while synchronization runs. Complete-line
  offsets and pending cursors are therefore part of the product contract, not
  optional optimizations.
- Local agent histories can contain sensitive user-authored material. The
  feature remains disabled by default, stores data only in the local palace,
  and excludes the largest machine-generated leakage surfaces.
- Implementation verification completed with `cargo fmt --all`,
  `cargo check --workspace`, the CLI help smoke test, and
  `cargo test --workspace` (97 tests passed).
