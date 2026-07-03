# PRD: Migrate mempalace-rs from ONNX Runtime to model2vec-rs

**Status:** Draft  
**Author:** .void22  
**Date:** 2026-04-11  
**Target:** mempalace-rs v0.2.0

---

## 1. Problem Statement

mempalace-rs currently depends on ONNX Runtime (`ort` crate) for text embedding inference. This creates significant platform friction:

- **macOS panics on launch**: The binary hardcodes `onnxruntime.dll` (Windows DLL). On macOS, `libonnxruntime.dylib` is required. The `dlopen` call fails with a cryptic panic.
- **Linux users hit the same wall**: Would need `libonnxruntime.so` and the path isn't configured.
- **Every user must manually install ONNX Runtime**: A ~15MB native shared library that isn't managed by Cargo, has platform-specific install steps, and must be placed in a magic path (`~/.mempalace/onnxruntime.dll`).
- **`ort` is an RC crate**: The Rust bindings are at `2.0.0-rc.11` — not stable, API may churn.
- **Zero Rust purity**: The C FFI layer adds build complexity, unsafe surface area, and debugging friction.

The project's offline-first, locally-run promise is undermined by this single platform-dependent C dependency.

---

## 2. Proposed Solution

Replace the ONNX Runtime + `ort` crate stack with **model2vec-rs** — a pure-Rust embedding inference engine with zero C dependencies.

### 2.1 What model2vec-rs is

| Property | ONNX Runtime (current) | model2vec-rs (proposed) |
|---|---|---|
| Language | C/C++ with Rust FFI bindings | Pure Rust |
| Platform deps | `onnxruntime.dll` / `.dylib` / `.so` | None — `cargo build` works everywhere |
| Model format | ONNX `.onnx` file | Potion `.potion` static model |
| Model size | ~80 MB (all-MiniLM-L6-v2) | 7–30 MB (potion-base-32M) |
| Inference speed | ~200 sent/sec (native) | ~340 sent/sec (1.7x faster than Python) |
| Embedding dims | 384 (MiniLM) | Configurable: 64, 128, 256, 384, 512, 768 |
| Quality vs MiniLM | 100% (baseline) | ~95%+ at 384-dim (MTEB avg) |
| Dependency footprint | ~15 MB shared lib + ort crate | Single Rust crate (~1.7 MB) |
| Crate stability | v2.0.0-rc.11 (pre-release) | Stable on crates.io |

### 2.2 Why model2vec-rs specifically

- **True cross-platform**: Works on Windows, macOS (arm64 + x86_64), Linux without any native library install.
- **Single `cargo build`**: Everything compiles from Rust source. No system package manager, no DLL hunting.
- **Embedding quality is sufficient**: potion-base-32M at 384-dim scores ~95% of MiniLM on MTEB benchmarks. For semantic code search and conversation retrieval, this is well within acceptable range.
- **Faster cold start**: Model load time is near-instant (static weights, no graph compilation).
- **Models auto-downloadable**: Can fetch from HuggingFace Hub at first run, same UX as current ONNX model download.
- **Matches crate philosophy**: mempalace-rs already prioritizes local-first, zero-dependency operation. model2vec-rs is the same philosophy applied to embeddings.

---

## 3. Scope & Impact

### 3.1 What changes

| Component | Change | Effort |
|---|---|---|
| `Cargo.toml` | Replace `ort` with `model2vec-rs` | Trivial |
| Embedding module | Rewrite ONNX inference → potion model inference | Medium |
| Model download logic | Change from ONNX `.onnz` fetch to potion `.potion` fetch | Small |
| Model path handling | Remove `~/.mempalace/onnxruntime.dll`; add `~/.mempalace/model.potion` | Small |
| Mine pipeline | Same text chunking → embed → store; only embed backend changes | Small |
| Search pipeline | Same query → embed → cosine similarity → rank; no change | None |
| Storage (drawers) | Drawer format unchanged; vectors are still `Vec<f32>` | None |

### 3.2 What breaks (breaking change)

**All existing drawers must be re-embedded.** The ONNX MiniLM embedding space and the potion embedding space are different. A drawer's stored vector was computed by MiniLM — cosine similarity against a potion query vector would produce garbage results.

**Mitigation**: Ship a migration command `mempalace-rs remine` that re-processes all source files through the new embedding backend. Because the palace stores verbatim source text alongside embeddings, this is a pure recomputation — no data loss.

### 3.3 What stays the same

- Palace file format (AAAK dialect)
- Drawer storage layout
- Semantic search API
- `mine`, `search`, `traverse` commands (same UX)
- Knowledge graph, diary, taxonomy — all unaffected
- MemPalace protocol rules

---

## 4. Technical Design

### 4.1 API migration

**Current (ONNX):**
```rust
use ort::{Session, Value};

let session = Session::builder()?
    .with_model_from_file("path/to/model.onnx")?;

let input = Value::from_array(session.allocator(), &input_ids)?;
let outputs = session.run(vec![input])?;
let embedding: Vec<f32> = outputs[0].extract_tensor()?.into();
```

**Proposed (model2vec-rs):**
```rust
use model2vec::InferenceModel;

let model = InferenceModel::load("path/to/potion-base-32M.potion")?;
let embedding: Vec<f32> = model.embed("some text to embed")?;
```

Lines of code in embedding module: ~80 → ~15.

### 4.2 Model acquisition

**First-run flow (new users):**
```
mempalace-rs mine .
→ No model found at ~/.mempalace/potion-base-32M.potion
→ Download from HuggingFace: minishlab/potion-base-32M (~30 MB)
→ Proceed with mining
```

**Recommended default model:** `minishlab/potion-base-32M` — 32M parameters, 384-dim embeddings, ~30 MB on disk.

**Optional: configurable model.** Advanced users can set `$MEMPALACE_MODEL` or a config field to use higher-quality (potion-retrieval-32M, 768-dim) or smaller (potion-base-8M, 256-dim) models.

### 4.3 Re-embedding migration

```bash
# New command — re-embed all drawers using the new backend
mempalace-rs remine

# Detects stored source text in each drawer →
#   re-embeds with potion model →
#   updates drawer embedding in-place
# Preserves: drawer_id, metadata, source_file, room, wing
```

Implementation: iterate all drawers, read `source_file` content, run `model.embed(content)`, update drawer vector. Show progress bar. Idempotent (can be interrupted and resumed).

### 4.4 Backward-compat detection

On startup, check a palace metadata field `embedding_backend: "potion-base-32M"`. If missing (old palace) or mismatched, print:
```
⚠  Palace embeddings were generated with a different backend.
   Run `mempalace-rs remine` to migrate.
   Search results may be inaccurate until migration completes.
```

---

## 5. Success Criteria

- [ ] `mempalace-rs mine .` works on macOS (arm64), macOS (x86_64), Windows, and Linux with zero native library installs
- [ ] `cargo install mempalace-rs` produces a working binary on all platforms (CI green)
- [ ] `mempalace-rs remine` successfully migrates an existing palace
- [ ] Semantic search quality on migrated palace is subjectively equivalent (same top-3 results for common queries)
- [ ] Model auto-downloads on first run (no manual setup)
- [ ] No `ort`, `onnxruntime`, or `libonnxruntime` in the dependency tree

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| potion embedding quality lower for code search | Medium | Medium | Benchmark against MiniLM on mempalace's own search test suite before committing. If quality is poor, use potion-retrieval-32M (768-dim). |
| Re-embedding migration is slow for large palaces | High | Low | Show progress bar; make migration resumable; it's a one-time cost. |
| model2vec-rs crate API changes before v1.0 | Low | Low | Pin exact version in Cargo.toml; API surface is tiny (2 functions). |
| Users on old mempalace-rs versions can't read new drawers | Medium | Low | Embedding backend version in palace metadata. Old version shows clear error: "Palace created by newer mempalace-rs. Please upgrade." |

---

## 7. Out of Scope

- GPU acceleration (model2vec-rs doesn't support CUDA/Metal yet — but embedding inference is CPU-light anyway)
- Hybrid mode (support both ONNX and potion simultaneously)
- Embedding model fine-tuning or customization
- Changing the palace file format or AAAK dialect
- Knowledge graph or diary format changes

---

## 8. Timeline Estimate

| Phase | Effort |
|---|---|
| Replace `ort` with `model2vec-rs` in Cargo.toml, rewrite embedding module | 2–4 hours |
| Add model auto-download logic | 1–2 hours |
| Implement `remine` migration command | 2–3 hours |
| Test on macOS, Windows, Linux | 2–4 hours |
| Benchmark search quality vs old ONNX backend | 1–2 hours |
| Update README, AGENTS.md, skill files | 1 hour |
| **Total** | **~1.5–2.5 days** |
