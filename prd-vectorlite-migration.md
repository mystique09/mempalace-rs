# PRD: Migrate mempalace-rs from LanceDB to vectorlite (SQLite-native HNSW)

> **Status: superseded.** The vectorlite design was retired after production profiling found that its connection-local HNSW state could be empty, partial, or stale across MCP processes and could reserve excessive memory. The active Rust store uses bounded exact-cosine search plus independent SQLite FTS5 ranking; see `README.md` and `crates/store/src/lib.rs`.

**Status:** Draft  
**Author:** .void22  
**Date:** 2026-04-11  
**Target:** mempalace-rs v0.2.0  
**Supersedes:** Previous sqlite-vec proposal — vectorlite chosen for ANN from day one

---

## 1. Problem Statement

mempalace-rs currently uses LanceDB for vector storage and hybrid search. LanceDB pulls **1,600+ recursive crate dependencies** through Arrow, DataFusion, parquet, object_store, and the lance-* family. This creates real pain:

- **Build times**: 5+ minutes for clean builds, mostly compiling Arrow/DataFusion/lance-*
- **Binary size**: Hundreds of MB linked in for what amounts to a personal search engine
- **Dependency churn**: LanceDB is actively developed; API breaks between minor versions
- **Overkill architecture**: LanceDB is designed for petabyte-scale ML data lakes. MemPalace stores ~29k personal knowledge snippets.
- **Mental overhead**: Understanding a LanceDB failure means understanding the entire Arrow → DataFusion → lance → parquet → object_store stack

Meanwhile, the project already has **rusqlite** in-tree for the knowledge graph. SQLite is the natural backbone — the only thing LanceDB provides that SQLite doesn't is vector ANN search.

The project's promise is **local-first, lightweight, personal**. LanceDB's dependency tree undermines that promise.

---

## 2. Proposed Solution

Replace LanceDB with **vectorlite** — a SQLite extension that adds HNSW-powered vector ANN search — plus FTS5 (built into SQLite) for full-text search, glued together with reciprocal rank fusion (RRF) in application code.

### 2.1 Why vectorlite over sqlite-vec

| Property | sqlite-vec | vectorlite (chosen) |
|---|---|---|
| Search algorithm | Brute force (linear scan) | HNSW (ANN, ~10–40x faster at scale) |
| ANN today? | ❌ (DiskANN in alpha) | ✅ HNSW, production-ready |
| Build dependency | Zero (single .c file) | C++17 + CMake (hnswlib) |
| SQL interface | Virtual table | Virtual table |
| FTS integration | FTS5 (built into SQLite) | FTS5 (built into SQLite) |
| 29k query time | ~10–20ms (brute) | ~1–5ms (HNSW) |
| 500k query time | ~200–400ms (brute) | ~5–15ms (HNSW) |
| Migration path | sqlite-vec → vectorlite if scale demands | Already on ANN — no future migration needed |

**Decision**: vectorlite. The C++ build dep (hnswlib) is a one-time setup cost, and the ANN performance means we never need to worry about scale. The SQLite virtual table interface is identical to sqlite-vec's — it's the same implementation pattern, just with HNSW instead of brute force.

### 2.2 What stays, what goes

| Component | Current (LanceDB) | Proposed (vectorlite + FTS5) |
|---|---|---|
| Vector search | LanceDB IVFFlat ANN | vectorlite HNSW ANN |
| Full-text search | LanceDB FTS (Tantivy-based) | SQLite FTS5 |
| Hybrid search | LanceDB `execute_hybrid()` | RRF in application code |
| Metadata columns | Arrow RecordBatch | SQLite columns |
| Persistence | Lance files in `{palace_path}/` | Single `{palace_path}/store.sqlite3` |
| Embedding backend | model2vec-rs (unchanged) | model2vec-rs (unchanged) |
| `MemoryStore` trait | Implemented by `LanceMemoryStore` | Implemented by `SqliteMemoryStore` |
| Dep tree | ~1,600 crates | ~200 crates |

### 2.3 Dependency impact

```
REMOVED:
  lancedb                    → 1,600+ recursive crates gone
  arrow-array = "57.3.0"     → only used for RecordBatch
  arrow-schema = "57.3.0"    → only used for Schema/ArrowError

ADDED:
  vectorlite = "0.4"         → Rust bindings for vectorlite C extension
  rusqlite = { features = ["bundled", "vtab"] }  → already in-tree for KG

NET: ~1,600 crates removed, ~2 added (vectorlite + its C build infra)
```

---

## 3. Scope & Impact

### 3.1 What changes

| Component | Change | Effort |
|---|---|---|
| `crates/store/Cargo.toml` | Replace `lancedb` + `arrow-*` with `vectorlite` + `rusqlite` | Small |
| `crates/store/src/lib.rs` | Rewrite `LanceMemoryStore` → `SqliteMemoryStore` | Large |
| Schema | Arrow `Schema` → SQL `CREATE TABLE` + FTS5 virtual table + vectorlite virtual table | Medium |
| Vector search | `table.query().nearest_to(vector)` → `SELECT ... FROM vectorlite ORDER BY distance` | Medium |
| FTS | LanceDB `FullTextSearchQuery` → FTS5 `MATCH` | Medium |
| Hybrid | LanceDB `execute_hybrid()` → RRF merge in app code | Medium |
| Index creation | LanceDB `create_index()` → vectorlite `INSERT INTO` + SQLite triggers | Small |
| Batch insert | RecordBatch construction → SQL `INSERT` with rusqlite binding | Medium |
| Migration CLI | `mempalace-rs migrate-store` — reads LanceDB, writes SQLite | Medium |
| Model download | Unchanged (model2vec-rs) | None |
| Mine pipeline | Unchanged (chunk → embed → store) | None |
| Knowledge graph | Unchanged (already SQLite) | None |

### 3.2 Breaking change

**Existing LanceDB data must be migrated.** The migration is non-destructive:

```bash
mempalace-rs migrate-store  # LanceDB → SQLite, one-time
```

Reads all drawers from LanceDB, bulk-inserts into SQLite + FTS5 + vectorlite, verifies counts match. Does not delete the original LanceDB files — user does that manually after confirming success.

### 3.3 What stays the same

- `MemoryStore` trait — identical signature, new implementation
- Palace file format (AAAK dialect)
- Embedding backend (model2vec-rs, already migrated)
- CLI commands: `mine`, `search`, `status`, `compress` — same UX
- Knowledge graph, diary, taxonomy — all unaffected
- MemPalace protocol rules

---

## 4. Technical Design

### 4.1 Schema design

```sql
-- Main drawer table (replaces RecordBatch)
CREATE TABLE drawers (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    wing        TEXT NOT NULL,
    room        TEXT NOT NULL,
    source_file TEXT,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    added_by    TEXT NOT NULL,
    filed_at    TEXT
);

-- FTS5 virtual table (replaces LanceDB FTS index)
CREATE VIRTUAL TABLE drawers_fts USING fts5(
    content,
    content='drawers',
    content_rowid='rowid'
);

-- vectorlite HNSW virtual table (replaces LanceDB IVFFlat)
CREATE VIRTUAL TABLE drawers_vec USING vectorlite(
    embedding float32[384],
    hnsw_max_elements=100000,
    hnsw_dim=384,
    hnsw_M=16,
    hnsw_ef_construction=200
);

-- Triggers keep FTS and vector tables in sync
CREATE TRIGGER drawers_ai AFTER INSERT ON drawers BEGIN
    INSERT INTO drawers_fts(rowid, content) VALUES (new.rowid, new.content);
    INSERT INTO drawers_vec(rowid, embedding) VALUES (new.rowid, ?);
END;

CREATE TRIGGER drawers_ad AFTER DELETE ON drawers BEGIN
    INSERT INTO drawers_fts(drawers_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    DELETE FROM drawers_vec WHERE rowid = old.rowid;
END;
```

### 4.2 Search flow (replaces LanceDB hybrid query)

```
1. Embed query text → Vec<f32> (model2vec-rs, unchanged)
2. Vector ANN: SELECT rowid, distance FROM drawers_vec WHERE embedding MATCH ? ORDER BY distance LIMIT N
3. FTS: SELECT rowid, rank FROM drawers_fts WHERE drawers_fts MATCH ? ORDER BY rank LIMIT N
4. RRF merge: reciprocal_rank_fusion(vector_results, fts_results, k=60)
5. Fetch full rows: SELECT * FROM drawers WHERE rowid IN (?)
6. Apply wing/room filter, rerank, truncate → Vec<SearchHit>
```

### 4.3 RRF (Reciprocal Rank Fusion) implementation

```rust
fn reciprocal_rank_fusion(
    vector_hits: &[(i64, f32)],  // (rowid, distance)
    fts_hits: &[(i64, f32)],     // (rowid, rank)
    k: usize,
) -> Vec<(i64, f32)> {
    let mut scores: HashMap<i64, f32> = HashMap::new();
    for (rank, (rowid, _)) in vector_hits.iter().enumerate() {
        *scores.entry(*rowid).or_default() += 1.0 / (k as f32 + rank as f32 + 1.0);
    }
    for (rank, (rowid, _)) in fts_hits.iter().enumerate() {
        *scores.entry(*rowid).or_default() += 1.0 / (k as f32 + rank as f32 + 1.0);
    }
    let mut merged: Vec<_> = scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}
```

### 4.4 Migration command

```bash
mempalace-rs migrate-store
```

Flow:
1. Open existing LanceDB table
2. Read all drawers (list_drawers + vectors)
3. Create SQLite database at `{palace_path}/store.sqlite3`
4. Run schema DDL (drawers + FTS5 + vectorlite)
5. Bulk INSERT drawers (batched in transactions)
6. Bulk INSERT vectors into vectorlite table
7. Verify: `SELECT COUNT(*) FROM drawers` == LanceDB count
8. Print: "Migrated N drawers to SQLite. Original LanceDB files remain at {path} — delete them manually when ready."

### 4.5 Store path

```
Old: {palace_path}/                    (LanceDB directory with lance files)
     {palace_path}/data/               (LanceDB internal)
     {palace_path}/_indices/           (LanceDB internal)

New: {palace_path}/store.sqlite3       (single file, everything)
```

The legacy Chroma → LanceDB path logic in `config.rs` can be simplified — store is always `{palace_path}/store.sqlite3`.

---

## 5. Success Criteria

- [ ] All existing tests pass with `SqliteMemoryStore` (same test suite, new implementation)
- [ ] `mempalace-rs search "jwt authentication"` returns same-quality results as current LanceDB
- [ ] `mempalace-rs migrate-store` migrates 29k drawers in <30 seconds
- [ ] `cargo build --workspace` completes in <1 minute (vs ~5 min today)
- [ ] `cargo tree | wc -l` drops from ~1,600 to ~200
- [ ] Hybrid search quality (RRF) is subjectively equivalent to LanceDB's built-in hybrid
- [ ] No `lancedb`, `arrow-array`, `arrow-schema` in Cargo.toml
- [ ] SQLite file size ≤ LanceDB directory size (ideally smaller due to no columnar overhead)
- [ ] HNSW index builds in reasonable time on 29k vectors (<5 seconds)

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| vectorlite crate immature / API unstable | Medium | Medium | Pin exact version. API surface is small (virtual table DDL + MATCH). Fallback to sqlite-vec if blocked. |
| HNSW index build memory spike | Low | Low | HNSW on 29k vectors with M=16 is ~10-20MB in memory. Well within bounds. |
| RRF hybrid quality worse than LanceDB built-in | Medium | Medium | Benchmark top-10 results for 20 queries against old LanceDB output before merging. Tune RRF `k` parameter. |
| hnswlib C++ build fails on some platforms | Medium | Medium | Test CI on macos-arm64, macos-x86_64, ubuntu-latest, windows-latest before merging. |
| Migration from LanceDB misses data | Low | High | Verify count. Show diff of any missing drawers. Dry-run mode. |
| SQLite concurrent access from KG + Store | Low | Medium | Use separate connections (KG already uses its own). Enable WAL mode for store. |

---

## 7. Out of Scope

- Changing the `MemoryStore` trait (new implementation, same interface)
- Changing the embedding backend (model2vec-rs stays)
- GPU acceleration
- Knowledge graph or diary format changes
- Changing the palace file format or AAAK dialect
- Supporting both LanceDB and SQLite backends simultaneously (migration is one-way)
- sqlite-vec as an intermediate step (go straight to vectorlite for ANN)

---

## 8. Timeline Estimate

| Phase | Effort |
|---|---|
| Add vectorlite + rusqlite deps, verify builds on all platforms | 1–2 hours |
| Implement `SqliteMemoryStore` — schema, CRUD, status | 4–6 hours |
| Implement vector ANN search via vectorlite virtual table | 2–3 hours |
| Implement FTS5 text search | 1–2 hours |
| Implement RRF hybrid merge | 1–2 hours |
| Implement `migrate-store` CLI command | 2–3 hours |
| Port existing test suite to `SqliteMemoryStore` | 2–4 hours |
| Remove `lancedb`, `arrow-array`, `arrow-schema` from workspace | 1 hour |
| Benchmark search quality vs old LanceDB | 1–2 hours |
| Update README, AGENTS.md | 1 hour |
| **Total** | **~2–3 days** |
