# PRD: Multi-domain semantic retrieval quality

**Status:** In progress; core retrieval changes implemented, ship gates remain open
**Date:** 2026-07-15
**Target:** Rust mining, SQLite retrieval, CLI/MCP search, and benchmark tooling

## Summary

Improve `mempalace-rs` semantic retrieval across code, conversations,
documentation, decisions, diaries, and general prose. The solution must retain
the recent code-search gains without treating one Reforged query—or even one
codebase—as the product benchmark.

The current production search is fast and memory-bounded, but its embedding
representation, query expansion, and reranking are biased toward source code.
That bias helped natural-language-to-code retrieval while leaving a measurable
gap on conversational memory. The full LongMemEval run scored 91.6% Recall@5,
5.0 percentage points behind the original MemPalace raw baseline.

## Problem

MemPalace stores heterogeneous information:

- source code and identifiers
- conversation and agent-session transcripts
- README files, PRDs, design documents, and decisions
- diaries, AAAK records, people, and project facts
- miscellaneous prose and configuration

The current Rust search path applies one global representation and ranking
strategy to all of them. It always enriches embeddings with source and
identifier context when available, creates code-oriented query views, and
applies identifier/path-aware reranking. Those signals are useful for code but
are not a complete retrieval strategy for preferences, remembered assistant
responses, temporal questions, or prose whose relevant wording differs from
the query.

This creates two risks:

1. Optimizing only for LongMemEval could improve conversational recall while
   regressing code and documentation search.
2. Continuing to tune only against Reforged could hide failures in every other
   content domain.

## Current baseline

### LongMemEval, 500 questions

| Metric | `mempalace-rs` | Original raw MemPalace |
| --- | ---: | ---: |
| Recall@5 | 91.6% | 96.6% |
| Recall@10 | 95.6% | 98.2% |
| NDCG@10 | 83.0% | 88.9% |
| MRR@10 | 79.8% | Not reported |

Rust performance is already within budget: 12 ms p50, 15 ms p95, roughly
173 MB peak RSS, and 6.9 seconds for the full benchmark in a release build.

The weakest LongMemEval categories are:

| Category | Questions | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| Single-session preference | 30 | 60.0% | 66.7% | 35.1% |
| Single-session assistant | 56 | 82.1% | 87.5% | 75.9% |
| Temporal reasoning | 133 | 89.5% | 96.2% | 71.6% |

Knowledge updates, multi-session memory, and single-session user facts are
already strong. Improvements must target the weak categories without adding
benchmark-specific phrase tables.

## Implementation result

The content-aware retrieval implementation is complete locally with explicit
content kinds, schema migration, source-derived mining assignments, typed query
views, candidate-aware reranking, optional temporal reference time, fixed
LongMemEval splits, and a checked-in 100-query mixed-domain evaluator. Release
validation remains open because the absolute LongMemEval recall gates are not
yet met and the portable corpus still needs authentic repository/session
fixtures before it can be treated as a product ship gate.

On the same 500-question raw LongMemEval protocol, the default
`minishlab/potion-code-16M-v2` configuration now scores:

| Metric | Before | Current | Target |
| --- | ---: | ---: | ---: |
| Recall@5 | 91.6% | 95.8% | 96.6% |
| Recall@10 | 95.6% | 97.2% | 98.2% |
| NDCG@10 | 83.0% | 89.2% | 88.9% |
| MRR@10 | 79.8% | 90.4% | Not specified |

The frozen 450-question held-out split improved from 91.33% to 95.56%
Recall@5 (+4.22 percentage points), 82.61% to 89.2% NDCG@10, and 79.18%
to 90.61% MRR@10. The held-out Recall@5 Wilson intervals are
88.4%-93.6% before and 93.2%-97.1% now. The checked-in multi-domain suite
scores 100% Recall@5
overall and in both code projects, conversations, documentation, diaries, and
prose on its current representative fixtures, and its CLI/MCP top-10 parity
pass succeeds across all six domain labels. Fresh real-repository validation
keeps Reforged `GameCommands::FirstJoin`
at rank 1, the socket `XtRequest::FirstJoin` branch at rank 8, and exact
`login` at rank 1.

Resource budgets remain satisfied for the default: 10 ms p50, 12 ms p95, and
about 173 MB peak RSS. `potion-retrieval-32M` reached the same 95.8% Recall@5
but peaked around 332 MB, so it was rejected. `potion-base-8M` remained near
112 MB but reached only 95.4% Recall@5. The default model therefore remains
unchanged. The complete size, dimension, indexing-time, quality, latency, and
RSS matrix is checked in at `benchmarks/model_bakeoff.md`.

The implementation clears the held-out-improvement, NDCG, latency, memory, and
representative-fixture checks. The absolute full-set Recall@5 and Recall@10
targets remain open. The multi-domain ship gate also remains open until the
representative fixtures are replaced or supplemented with authentic portable
source/session excerpts and a true pre-tuning baseline. Further tuning must use
new development evidence rather than rules based on recorded held-out misses.

### Code search

The existing Reforged control remains important but is only one member of the
regression suite:

- `process a player's very first game login` returns
  `GameCommands::FirstJoin` at rank 1.
- The socket `XtRequest::FirstJoin` branch appears at rank 8.
- Exact `login` continues to rank direct login-domain code first.

The checked-in representative suite now covers Reforged and `mempalace-rs`,
but the authentic multi-project ship corpus and pre-tuning baseline remain open.

## Goals

1. Reach original-MemPalace-level conversational retrieval on the same
   LongMemEval raw protocol.
2. Preserve or improve semantic code retrieval across at least two codebases,
   not only Reforged.
3. Measure documentation, decisions, diaries, and general prose separately
   instead of folding them into a single anecdotal score.
4. Make content-specific retrieval signals explicit and testable while keeping
   the original dense query as a shared retrieval channel.
5. Keep all default search local, CPU-only, deterministic, and free of hosted
   APIs or per-query LLM calls.
6. Preserve verbatim drawer content in results and keep current score semantics.
7. Keep search below existing latency and memory budgets on the real palace.

## Non-goals

- Optimizing only the Reforged `FirstJoin` query.
- Tuning directly against held-out LongMemEval failures.
- Adding a default LLM reranker, hosted embedding API, or cloud dependency.
- Reintroducing vectorlite/HNSW or another process-local vector index.
- Replacing verbatim content with summaries or extracted facts.
- Shipping multiple embedding models in one store before a single-model bakeoff
  proves that a balanced model cannot meet the acceptance criteria.
- Publishing this PRD as a GitHub issue yet.

## Product principles

### Shared retrieval, content-aware evidence

Every search keeps an original-query dense channel. Content-aware behavior may
add candidates or bounded boosts, but it must not hard-route the query to one
domain and exclude relevant results from another.

### Verbatim first

Search may create a separate derived `retrieval_text`, but returned content
remains the exact stored drawer content. Preference or role-aware indexing must
use verbatim spans and labels, not lossy summaries.

### Benchmark before heuristics

First test better balanced embedding models and representation choices. Add
new query expansions or ranking rules only when failure analysis identifies a
general class and the change improves a held-out corpus.

### No benchmark labels in retrieval inputs

Ground-truth IDs, expected paths, and evaluation labels must never enter
embedding text, FTS text, or reranking signals.

## Proposed behavior

### 1. Explicit content kinds

Add a backward-compatible content kind to drawer metadata and SQLite storage:

- `code`
- `conversation`
- `documentation`
- `diary`
- `prose`
- `unknown`

Mining assigns the kind from the source adapter and parser, not from the query.
Legacy rows default to `unknown` and can be backfilled without recomputing
embeddings. A model or embedding-representation change still requires a full
remine.

### 2. Content-aware embedding representations

Use a common representation contract with bounded, kind-specific context:

- **Code:** path, language, enclosing symbol, and split identifiers.
- **Conversation:** stable role labels, conversation/session context when
  available, and both user and assistant verbatim text in production mining.
- **Documentation:** title, heading ancestry, and compact source path.
- **Diary:** date, agent, topic, and verbatim AAAK/body text.
- **Prose/unknown:** content plus only trustworthy source context.

Do not inject source identifiers into benchmark content when those identifiers
are evaluation labels. Version any representation change in store metadata.

### 3. Soft query-intent signals

Derive bounded, deterministic signals from the query:

- code/identifier intent
- preference or recommendation intent
- assistant-response recall
- temporal intent
- general factual/prose intent

These signals adjust query views and candidate scoring; they do not become
exclusive filters. For example, generic recall scaffolding such as “can you
remind me” may be removed from an additional lexical/query view while the
original query remains intact in the primary dense channel.

### 4. Candidate-aware reranking

Apply evidence only where it is meaningful:

- Identifier and path bonuses apply to code candidates.
- Heading/title matches apply to documentation candidates.
- Role-aware matches apply to conversation candidates.
- Preference-intent matches use verbatim preference/concern spans rather than
  synthetic summaries.
- Temporal boosts require both temporal query intent and reliable drawer dates;
  recency is never a universal relevance proxy.
- Generated/build-output penalties remain code-specific.

Dense similarity, FTS5/BM25, and bounded query-view ranks continue to be fused
with weighted reciprocal-rank fusion. Profile weights must be configuration
constants covered by evaluation, not per-query learned state.

### 5. Embedding-model bakeoff

Benchmark the current `minishlab/potion-code-16M-v2` model against small local
general-purpose and code-capable model2vec candidates. Include the original
all-MiniLM-L6-v2 result as a comparison baseline when its runtime is available,
but do not require ONNX in the shipped default path.

Choose a new default only if one model passes the complete quality matrix and
stays within resource budgets. If no single model passes, stop and write an ADR
before designing dual embeddings or per-kind models.

The bakeoff report must include model size, embedding dimensions, mining time,
Recall@5, Recall@10, MRR@10, NDCG@10, p95 latency, and peak RSS.

### 6. Search API compatibility

- `SearchHit.score` remains original-query cosine similarity.
- `SearchHit.relevance` remains the fused user-facing ordering score.
- Existing wing and room filters apply to every retrieval channel.
- CLI and MCP return the same ordering for identical arguments.
- Optional temporal context, if added, is backward compatible and omitted by
  default.

## Evaluation corpus

### Multi-domain checked-in regression suite

Create at least 100 human-labelled queries with relative, portable matchers:

| Domain | Minimum queries | Required coverage |
| --- | ---: | --- |
| Code | 30 | Reforged and `mempalace-rs`; semantic descriptions, exact identifiers, ambiguous vocabulary |
| Conversations | 25 | User facts, assistant facts, preferences, multi-session recall |
| Documentation/decisions | 20 | PRDs, README content, architecture decisions, paraphrased rationale |
| Diaries/AAAK | 15 | Dates, topics, people, changed facts, agent entries |
| General prose and exact controls | 10 | Exact terms, synonyms, distractors, mixed-domain queries |

Each case contains the query, content kind, allowed relevant targets, and target
matcher. Code targets use relative path plus symbol/content matching; prose
targets use stable drawer IDs only in the evaluator, never retrieval text.

Use multiple relevant targets when the query is genuinely ambiguous. Do not
mark one implementation as the only truth simply because it motivated the
test.

### Development and held-out splits

- Reserve a deterministic development subset for tuning.
- Keep at least 70% of the multi-domain corpus held out.
- For LongMemEval, define a deterministic 50-question development set and
  report the remaining 450 separately.
- Do not add rules based on individual held-out misses.
- Publish per-question JSONL results for auditability, but do not commit the
  external LongMemEval dataset.

## Acceptance criteria

### Quality

- LongMemEval full-set Recall@5 is at least 96.6% on the existing raw protocol.
- LongMemEval Recall@10 is at least 98.2% and NDCG@10 is at least 88.9%.
- Freeze the current score on the held-out LongMemEval split before tuning,
  then improve that split by at least 3 percentage points at Recall@5; report
  confidence intervals.
- Multi-domain held-out Recall@5 is at least 90% overall.
- No content domain regresses by more than 2 percentage points from its frozen
  baseline.
- The Reforged `FirstJoin` semantic query remains within the top five.
- Exact `login` remains rank 1 for direct login-domain code.
- The `mempalace-rs` code subset meets the same code Recall@5 threshold as the
  Reforged subset.
- CLI and MCP top-10 ordering is identical on a parity sample from every domain.

### Performance and resources

- Search p95 remains below one second on the current roughly 150,000-drawer
  palace.
- Peak search RSS remains below 300 MB.
- LongMemEval p95 remains below 50 ms in a release build after model warm-up.
- The default path loads at most one embedding model per process.
- No retrieval channel performs an unbounded scan beyond the existing exact
  cosine pass over the filtered corpus.

### Correctness and compatibility

- Verbatim drawer content is unchanged by retrieval enrichment.
- Model and embedding-representation metadata prevent mixed incompatible
  vectors.
- Remine requirements are explicit and covered by CLI errors and README docs.
- Legacy stores open safely and receive a content-kind default or backfill.
- Core, store, CLI, MCP, benchmark, and workspace tests pass.

## Delivery phases

### Phase 0: Freeze evidence

1. Commit the portable multi-domain corpus and evaluator.
2. Record the current model’s scores by domain.
3. Add CLI/MCP parity samples.
4. Record the current real-palace latency and RSS baseline.

### Phase 1: Model and representation bakeoff

1. Add benchmark-only model selection.
2. Evaluate balanced local models without changing production defaults.
3. Test content-kind representations independently from ranking changes.
4. Select one model/representation combination using the complete matrix.

### Phase 2: Content-aware retrieval

1. Add content-kind storage and mining assignments.
2. Scope code-only identifier/path logic to code candidates.
3. Add bounded conversation, documentation, diary, and temporal signals.
4. Tune only on development splits.

### Phase 3: Migration and release validation

1. Version the embedding representation.
2. Implement safe backfill/remine behavior.
3. Run all held-out and full benchmarks once.
4. Update README and MCP/CLI documentation.
5. Record final per-domain quality, latency, memory, mining time, and model size.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| LongMemEval overfitting | Fixed dev/held-out split, per-question audit logs, and no held-out-specific rules |
| Code regression | Multi-project code corpus and frozen rank controls before tuning |
| One model cannot serve every domain | Single-model bakeoff first; ADR gate before dual embeddings |
| Content-kind misclassification | Source-adapter assignment, `unknown` fallback, soft scoring rather than hard routing |
| Temporal boosts create false positives | Require temporal intent and reliable dates; cap contribution |
| New representation forces expensive remine | Version metadata, report migration cost, and make remine explicit |
| Ranking complexity becomes unmaintainable | Typed intent/content signals, bounded constants, and per-domain ablation reports |

## Required deliverables

- Checked-in multi-domain query corpus and evaluator
- LongMemEval dev/held-out reporting
- Model bakeoff report
- Content-kind schema and migration
- Content-aware retrieval implementation and tests
- CLI/MCP parity test suite
- Final quality/performance report
- README and operational migration guidance

## Decision rule

Do not ship a retrieval change because it fixes one query or one benchmark
category. Ship only a configuration that improves the held-out aggregate,
passes every domain floor, preserves exact-search controls, and remains within
the latency and memory budgets.
