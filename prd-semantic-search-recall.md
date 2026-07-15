# PRD: Improve synonym-level semantic recall

**Status:** Implemented
**Date:** 2026-07-15
**Target:** Rust miner and SQLite search pipeline

## Problem

After migrating to `minishlab/potion-code-16M-v2`, exact terms search well but natural-language descriptions can miss project-specific identifiers. In Reforged, the query `process a player's very first game login` ranks the intended `XtRequest::FirstJoin` chunk at 929 even though `handle a player's first join request` ranks it first.

The failure is compounded by fixed-width chunks that split code units, a small pre-rerank candidate set, vector-only retrieval despite an existing FTS5 index, and reranking that recognizes whole-query identifier variants but not useful partial terms or code-domain synonyms.

## Goals

- Improve natural-language-to-code recall without restoring a process-local vector index.
- Keep the default model small and CPU-only.
- Preserve verbatim drawer content in search results.
- Make hybrid retrieval a normal indexed channel, not a slow fallback.
- Keep search latency and memory bounded on the current approximately 149,000-drawer palace.

## Proposed behavior

1. Mine code into coherent structural units, prioritizing complete functions and match arms over arbitrary byte windows.
2. Enrich the text used for embeddings with compact path, language, enclosing-symbol, and identifier context while retaining verbatim source as drawer content.
3. Retain a deep bounded dense candidate pool and independently retrieve FTS5/BM25 candidates.
4. Fuse dense, expanded-query, and lexical ranks with weighted reciprocal-rank fusion.
5. Normalize code identifiers, remove stopwords and single-character noise, and use partial identifier evidence during reranking.
6. Apply a small bounded set of general code-domain query expansions for ambiguous vocabulary such as login, join, connect, authenticate, first, and initial.

## Acceptance criteria

- The Reforged query `process a player's very first game login` returns a `FirstJoin` implementation in the top five after a remine.
- Exact `login` searches continue to return direct login-domain code.
- Wing and room filters apply to every retrieval channel.
- `SearchHit.score` remains the exact cosine similarity for the original query; fused ordering is represented by `SearchHit.relevance`.
- Existing duplicate thresholds keep their current semantics.
- Search remains below one second and below 300 MB RSS on the current local palace.
- Core, store, MCP, and workspace tests pass.

## Non-goals

- Reintroducing vectorlite/HNSW or any process-local vector index.
- Calling a hosted LLM on every query.
- Adding a heavyweight transformer reranker to the default path.
- Creating or publishing a GitHub issue.

## Validation

Track Recall@5, MRR@10, p95 latency, and peak RSS on a small checked-in semantic-search regression corpus. The initial corpus must include the Reforged `FirstJoin` case plus exact identifier and exact login controls.

### Initial Reforged result

An isolated full Reforged mine produced 24,119 drawers from 2,509 files in 33.47 seconds and peaked at approximately 208 MB RSS. With the optimized binary:

- `process a player's very first game login` returns `GameCommands::FirstJoin` at rank 1 and the socket `XtRequest::FirstJoin` branch at rank 10, improved from ranks 139 and 929 respectively.
- The five-result query completes in approximately 0.10 seconds with warm filesystem caches; a cold-process run completed in 0.75 seconds. Peak RSS was approximately 236 MB.
- The exact query `login` continues to rank the direct `UserLogin` DTO first.
