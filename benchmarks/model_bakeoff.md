# Local embedding model bakeoff

**Date:** 2026-07-15

**Protocol:** LongMemEval-S cleaned, 500 questions, raw user-turn sessions,
content-aware ranking v3, top 50, release build

**Machine:** Apple M5 (arm64), 24 GiB RAM, macOS 26.5.2

The evaluator deletes and indexes each question's session corpus before its
search. `Indexing` is therefore the cumulative embedding/write time across all
500 independent question corpora; `p95` is the end-to-end per-question latency
including indexing, while `search p95` isolates retrieval after indexing.
Model size is the local Hugging Face cache footprint and RSS is captured with
`/usr/bin/time -l`.

| Model | Cache | Dim | Indexing | R@5 | R@10 | MRR@10 | NDCG@10 | p95 | Search p95 | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `minishlab/potion-code-16M-v2` | 32 MiB | 256 | 3.875 s | **95.8%** | 97.2% | 90.4% | 89.2% | 12 ms | 2 ms | 172.8 MB |
| `minishlab/potion-retrieval-32M` | 125 MiB | 512 | 4.300 s | **95.8%** | 97.4% | **91.1%** | **90.1%** | 14 ms | 2 ms | 332.5 MB |
| `minishlab/potion-base-8M` | 30 MiB | 256 | **3.324 s** | 95.4% | **97.6%** | 90.7% | 89.7% | **11 ms** | 2 ms | **111.5 MB** |

## Decision

Keep `minishlab/potion-code-16M-v2` as the default. The retrieval model does
not improve Recall@5 and exceeds the 300 MB peak-RSS gate. The 8M model saves
memory but loses 0.4 percentage points at Recall@5. No tested model reaches the
96.6% Recall@5 acceptance target, so model replacement alone does not close the
remaining quality gap.

`all-MiniLM-L6-v2` is not included because the shipped pure-Rust model2vec
runtime cannot load it directly. Adding an ONNX runtime solely for a benchmark
would not represent the default product path.

## Reproduction

```bash
cargo build --release -p mempalace-rs --example longmemeval

/usr/bin/time -l target/release/examples/longmemeval \
  /tmp/longmemeval_s_cleaned.json

/usr/bin/time -l target/release/examples/longmemeval \
  /tmp/longmemeval_s_cleaned.json \
  --model minishlab/potion-retrieval-32M

/usr/bin/time -l target/release/examples/longmemeval \
  /tmp/longmemeval_s_cleaned.json \
  --model minishlab/potion-base-8M
```
