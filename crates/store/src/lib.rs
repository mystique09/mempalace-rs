use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use model2vec_rs::model::StaticModel;
use rusqlite::{Connection, params};

use mempalace_core::{
    Drawer, DrawerMetadata, MemoryStore, MempalaceError, Result, RoomStatus, SearchHit,
    SearchQuery, StoreStatus,
};

const DEFAULT_MODEL_REPO: &str = "minishlab/potion-base-32M";
const _EMBEDDING_DIM: usize = 512; // potion-base-32M output dimensionality

/// DDL for the vectorlite HNSW virtual table — created only when the extension loads.
const VECTORLITE_TABLE_DDL: &str = "
CREATE VIRTUAL TABLE IF NOT EXISTS drawers_vec USING vectorlite(
    embedding float32[512] cosine,
    hnsw(max_elements=1000000)
);
";

/// Triggers to keep the vectorlite table in sync with the drawers table.
const VECTORLITE_TRIGGER_DDL: &str = "
CREATE TRIGGER IF NOT EXISTS drawers_vec_ai AFTER INSERT ON drawers BEGIN
    INSERT INTO drawers_vec(rowid, embedding) VALUES (new.rowid, new.embedding);
END;

CREATE TRIGGER IF NOT EXISTS drawers_vec_ad AFTER DELETE ON drawers BEGIN
    DELETE FROM drawers_vec WHERE rowid = old.rowid;
END;
";

/// Schema DDL for the SQLite store.
const SCHEMA_DDL: &str = "
CREATE TABLE IF NOT EXISTS drawers (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    wing        TEXT NOT NULL,
    room        TEXT NOT NULL,
    source_file TEXT,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    added_by    TEXT NOT NULL,
    filed_at    TEXT,
    embedding   BLOB NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS drawers_fts USING fts5(
    content,
    content='drawers',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS drawers_ai AFTER INSERT ON drawers BEGIN
    INSERT INTO drawers_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS drawers_ad AFTER DELETE ON drawers BEGIN
    INSERT INTO drawers_fts(drawers_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
";

fn try_load_vectorlite(conn: &Connection) -> bool {
    let lib_name = if cfg!(target_os = "macos") {
        "vectorlite.dylib"
    } else if cfg!(target_os = "linux") {
        "vectorlite.so"
    } else if cfg!(target_os = "windows") {
        "vectorlite.dll"
    } else {
        return false;
    };

    let mut search_paths: Vec<PathBuf> = vec![
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join(lib_name)))
            .unwrap_or_default(),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".mempalace-bin")
            .join(lib_name),
    ];

    // Also try the build.rs output path (embedded at compile time)
    if let Some(build_path) = option_env!("VECTORLITE_LIB_PATH") {
        search_paths.push(PathBuf::from(build_path));
    }

    search_paths.push(PathBuf::from(lib_name));

    for path in &search_paths {
        if path.exists()
            && unsafe { conn.load_extension_enable().is_ok() }
            && unsafe { conn.load_extension(path, None::<&str>).is_ok() }
        {
            return true;
        }
    }

    false
}

/// Check whether the vectorlite table has an old (too-low) `max_elements` setting
/// and print a one-time warning to stderr. This is intentionally lightweight:
/// it only queries the DDL, does NOT rebuild anything.
fn warn_if_vectorlite_table_needs_resize(conn: &Connection) {
    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='drawers_vec'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|c| c > 0)
        .unwrap_or(false);

    if !table_exists {
        return;
    }

    let sql: rusqlite::Result<String> = conn.query_row(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='drawers_vec'",
        [],
        |row| row.get(0),
    );

    let Ok(sql) = sql else {
        return;
    };

    // Trailing paren distinguishes 100000) from 1000000) — only match the old 100k limit.
    if sql.contains("max_elements=100000)") {
        eprintln!(
            "note: vector index has old max_elements=100000. Run `mempalace-rs resize` to upgrade."
        );
    }
}

#[derive(Clone)]
pub struct SqliteMemoryStore {
    #[allow(dead_code)]
    palace_path: PathBuf,
    embedder: Arc<EmbeddingBackend>,
    conn: Arc<Mutex<Connection>>,
    vectorlite_available: bool,
}

impl SqliteMemoryStore {
    pub fn new(palace_path: impl Into<PathBuf>, model_dir: impl Into<PathBuf>) -> Result<Self> {
        let palace_path = palace_path.into();
        let model_dir = model_dir.into();
        let model_path = model_dir.join("potion-base-32M");
        let repo_or_path = if model_path.join("config.json").exists() {
            model_path.to_string_lossy().to_string()
        } else {
            DEFAULT_MODEL_REPO.to_owned()
        };

        let model = StaticModel::from_pretrained(&repo_or_path, None, None, None)
            .map_err(|err| MempalaceError::Embedding(err.to_string()))?;

        let db_path = palace_path.join("store.sqlite3");

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&db_path)?;

        // Enable WAL mode for better concurrent access
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;

        // Create schema
        conn.execute_batch(SCHEMA_DDL)?;

        // Try to load the vectorlite HNSW extension
        let mut vectorlite_available = try_load_vectorlite(&conn);
        if vectorlite_available {
            // Warn if the vectorlite table has the old 100k limit (but don't rebuild here)
            warn_if_vectorlite_table_needs_resize(&conn);
            // Create vectorlite virtual table and sync triggers;
            // if creation fails (e.g. low-memory CI), fall back to no vectorlite.
            let created = conn.execute_batch(VECTORLITE_TABLE_DDL).is_ok()
                && conn.execute_batch(VECTORLITE_TRIGGER_DDL).is_ok();
            if !created {
                vectorlite_available = false;
            }
        }

        Ok(Self {
            palace_path,
            embedder: Arc::new(EmbeddingBackend::Model2Vec {
                inner: Box::new(Mutex::new(Some(model))),
            }),
            conn: Arc::new(Mutex::new(conn)),
            vectorlite_available,
        })
    }

    #[cfg(test)]
    fn new_for_tests(palace_path: impl Into<PathBuf>) -> Self {
        let palace_path = palace_path.into();
        let db_path = palace_path.join("store.sqlite3");

        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(SCHEMA_DDL).unwrap();

        // Try loading vectorlite in tests too, but tests work fine without it
        let mut vectorlite_available = try_load_vectorlite(&conn);
        if vectorlite_available {
            let created = conn.execute_batch(VECTORLITE_TABLE_DDL).is_ok()
                && conn.execute_batch(VECTORLITE_TRIGGER_DDL).is_ok();
            if !created {
                vectorlite_available = false;
            }
        }

        Self {
            palace_path,
            embedder: Arc::new(EmbeddingBackend::Deterministic { dim: 512 }),
            conn: Arc::new(Mutex::new(conn)),
            vectorlite_available,
        }
    }

    fn with_conn<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        let conn = self
            .conn
            .lock()
            .map_err(|_| MempalaceError::LockPoisoned("sqlite_connection"))?;
        f(&conn)
    }

    fn search_condition(query: &SearchQuery) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(wing) = &query.wing {
            parts.push(format!("wing = '{}'", sql_escape(wing)));
        }
        if let Some(room) = &query.room {
            parts.push(format!("room = '{}'", sql_escape(room)));
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" AND "))
        }
    }

    fn search_filter_clause(query: &SearchQuery) -> String {
        Self::search_condition(query)
            .map(|c| format!("WHERE {c}"))
            .unwrap_or_default()
    }

    fn supports_full_text_search(query: &str) -> bool {
        query.chars().any(char::is_alphanumeric)
    }

    fn retrieval_limit(limit: usize) -> usize {
        (limit.saturating_mul(8)).clamp(limit, 1024)
    }

    pub async fn remine_all(&self, wing: Option<&str>) -> Result<usize> {
        let drawers = self.list_drawers(wing).await?;
        let total = drawers.len();
        if total == 0 {
            return Ok(0);
        }

        // Delete all existing drawers
        self.with_conn(|conn| {
            if let Some(w) = wing {
                conn.execute("DELETE FROM drawers WHERE wing = ?1", params![w])?;
            } else {
                conn.execute("DELETE FROM drawers", [])?;
            }
            Ok(())
        })?;

        // Re-add with fresh embeddings
        self.add_drawers(drawers).await?;
        Ok(total)
    }

    /// Resize the vectorlite HNSW index to a new `max_elements` limit.
    /// Drops and recreates the `drawers_vec` virtual table, then repopulates
    /// from the existing `drawers` embeddings. This is an explicit,
    /// user-initiated operation — it does NOT run automatically during `mine`.
    pub async fn resize_vectorlite_table(&self, max_elements: u64) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| MempalaceError::LockPoisoned("sqlite_connection"))?;

        if !self.vectorlite_available {
            return Err(MempalaceError::Embedding(
                "vectorlite extension is not available".to_owned(),
            ));
        }

        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='drawers_vec'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|c| c > 0)
            .unwrap_or(false);

        if !table_exists {
            return Err(MempalaceError::Embedding(
                "drawers_vec table does not exist yet; run `mine` first to create it".to_owned(),
            ));
        }

        // Build the new DDL with the requested max_elements
        let table_ddl = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS drawers_vec USING vectorlite(\n    embedding float32[512] cosine,\n    hnsw(max_elements={max_elements})\n);\n"
        );

        // Drop triggers first (they depend on the table), then the table
        conn.execute_batch("DROP TRIGGER IF EXISTS drawers_vec_ai;")?;
        conn.execute_batch("DROP TRIGGER IF EXISTS drawers_vec_ad;")?;
        conn.execute_batch("DROP TABLE IF EXISTS drawers_vec;")?;

        // Re-create with the new limit
        conn.execute_batch(&table_ddl)?;
        conn.execute_batch(VECTORLITE_TRIGGER_DDL)?;

        // Repopulate from existing drawer embeddings
        conn.execute_batch(
            "INSERT INTO drawers_vec(rowid, embedding) SELECT rowid, embedding FROM drawers;",
        )?;

        Ok(())
    }
}

#[async_trait]
impl MemoryStore for SqliteMemoryStore {
    async fn add_drawer(&self, drawer: Drawer) -> Result<()> {
        self.add_drawers(vec![drawer]).await
    }

    async fn add_drawers(&self, drawers: Vec<Drawer>) -> Result<()> {
        if drawers.is_empty() {
            return Ok(());
        }

        let texts: Vec<String> = drawers.iter().map(|d| d.content.clone()).collect();
        let vectors = self.embedder.embed_batch(&texts)?;

        self.with_conn(|conn| {
            let tx = conn
                .unchecked_transaction()
                ?;

            for (drawer, vector) in drawers.iter().zip(vectors.iter()) {
                let blob = vector_to_blob(vector);

                tx.execute(
                    "INSERT INTO drawers (id, content, wing, room, source_file, chunk_index, added_by, filed_at, embedding)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    params![
                        drawer.id,
                        drawer.content,
                        drawer.metadata.wing,
                        drawer.metadata.room,
                        drawer.metadata.source_file,
                        drawer.metadata.chunk_index,
                        drawer.metadata.added_by,
                        drawer.metadata.filed_at,
                        blob,
                    ],
                )
                ?;
            }

            tx.commit()
                ?;
            Ok(())
        })
    }

    async fn get_drawer(&self, drawer_id: &str) -> Result<Option<Drawer>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, content, wing, room, source_file, chunk_index, added_by, filed_at
                     FROM drawers WHERE id = ?1",
            )?;

            let rows: Vec<_> = stmt
                .query_map(params![drawer_id], row_to_drawer)?
                .filter_map(|r| r.ok())
                .collect();

            Ok(rows.into_iter().next())
        })
    }

    async fn delete_drawer(&self, drawer_id: &str) -> Result<bool> {
        self.with_conn(|conn| {
            let deleted = conn.execute("DELETE FROM drawers WHERE id = ?1", params![drawer_id])?;
            Ok(deleted > 0)
        })
    }

    async fn delete_source_file(&self, source_file: &str) -> Result<usize> {
        self.with_conn(|conn| {
            let deleted = conn.execute(
                "DELETE FROM drawers WHERE source_file = ?1",
                params![source_file],
            )?;
            Ok(deleted)
        })
    }

    async fn list_drawers(&self, wing: Option<&str>) -> Result<Vec<Drawer>> {
        self.with_conn(|conn| {
            let (sql, params_vec): (String, Vec<String>) = if let Some(w) = wing {
                (
                    "SELECT id, content, wing, room, source_file, chunk_index, added_by, filed_at
                     FROM drawers WHERE wing = ?1"
                        .to_string(),
                    vec![w.to_string()],
                )
            } else {
                (
                    "SELECT id, content, wing, room, source_file, chunk_index, added_by, filed_at
                     FROM drawers"
                        .to_string(),
                    vec![],
                )
            };

            let mut stmt = conn.prepare(&sql)?;

            let param_refs: Vec<&dyn rusqlite::types::ToSql> = params_vec
                .iter()
                .map(|p| p as &dyn rusqlite::types::ToSql)
                .collect();

            let rows: Vec<_> = stmt
                .query_map(param_refs.as_slice(), row_to_drawer)?
                .filter_map(|r| r.ok())
                .collect();

            Ok(rows)
        })
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchHit>> {
        if query.limit == 0 {
            return Ok(Vec::new());
        }

        let retrieval_limit = Self::retrieval_limit(query.limit);
        let query_vector = self.embedder.embed(&query.query)?;
        let use_fts = Self::supports_full_text_search(&query.query);
        let vectorlite_available = self.vectorlite_available;

        self.with_conn(|conn| {
            let filter_where = Self::search_filter_clause(&query);
            let filter_and = Self::search_condition(&query)
                .map(|c| format!("AND {c}"))
                .unwrap_or_default();

            let mut rows: Vec<(Drawer, Vec<f32>)>;
            let vector_hits: Vec<(usize, f32)>;

            if vectorlite_available {
                // --- HNSW ANN via vectorlite ---
                let query_blob = vector_to_blob(&query_vector);

                // Fetch more from ANN to allow for post-filter narrowing
                let knn_limit = retrieval_limit.max(256) as i64;

                let knn_sql = "SELECT rowid, distance FROM drawers_vec WHERE knn_search(embedding, knn_param(?1, ?2, ?3))";
                let mut knn_stmt = conn.prepare(knn_sql)?;

                let knn_results: Vec<(i64, f32)> = knn_stmt
                    .query_map(
                        params![query_blob, knn_limit, 50_i64],
                        |row| {
                            let rowid: i64 = row.get(0)?;
                            let distance: f32 = row.get(1)?;
                            Ok((rowid, distance))
                        },
                    )?
                    .filter_map(|r| r.ok())
                    .collect();

                if knn_results.is_empty() {
                    // KNN returned empty — fall through to brute-force path below
                } else {
                // Build rowid list for fetching full rows
                let rowid_list: Vec<String> = knn_results.iter().map(|(r, _)| r.to_string()).collect();
                let rowid_placeholders = rowid_list.join(",");

                // Fetch drawer rows and map rowid -> index
                let fetch_sql = format!(
                    "SELECT rowid, id, content, wing, room, source_file, chunk_index, added_by, filed_at, embedding
                     FROM drawers WHERE rowid IN ({rowid_placeholders}) {filter_and}"
                );

                let mut fetch_stmt = conn.prepare(&fetch_sql)?;

                let mut rowid_to_idx: HashMap<i64, usize> = HashMap::new();
                rows = Vec::with_capacity(knn_results.len());

                let fetched: Vec<(i64, Drawer, Vec<f32>)> = fetch_stmt
                    .query_map([], |row| {
                        let rowid: i64 = row.get(0)?;
                        let embedding: Vec<u8> = row.get(9)?;
                        let floats = blob_to_vector(&embedding);
                        Ok((rowid, row_to_drawer_offset(row, 1)?, floats))
                    })?
                    .filter_map(|r| r.ok())
                    .collect();

                for (idx, (rowid, drawer, emb)) in fetched.into_iter().enumerate() {
                    rowid_to_idx.insert(rowid, idx);
                    rows.push((drawer, emb));
                }

                if rows.is_empty() {
                    return Ok(Vec::new());
                }

                // Convert cosine distances to scores (distance ∈ [0,2], score = 1 - d/2)
                vector_hits = knn_results
                    .iter()
                    .filter_map(|(rowid, distance)| {
                        rowid_to_idx.get(rowid).map(|&idx| {
                            let score = 1.0 - distance / 2.0;
                            (idx, score)
                        })
                    })
                    .collect();

                // FTS5 search with correct rowid mapping
                let fts_hits: Vec<(usize, f32)> = if use_fts {
                    let fts_sql =
                        "SELECT rowid, rank FROM drawers_fts WHERE drawers_fts MATCH ?1 ORDER BY rank LIMIT ?2";
                    let mut fts_stmt = conn.prepare(fts_sql)?;

                    fts_stmt
                        .query_map(
                            params![query.query, retrieval_limit as i64],
                            |row| {
                                let rowid: i64 = row.get(0)?;
                                let rank: f32 = row.get(1)?;
                                Ok((rowid, rank))
                            },
                        )?
                        .filter_map(|r| r.ok())
                        .filter_map(|(rowid, rank)| {
                            rowid_to_idx.get(&rowid).map(|&idx| (idx, rank))
                        })
                        .collect()
                } else {
                    Vec::new()
                };

                // RRF merge
                let merged = reciprocal_rank_fusion(&vector_hits, &fts_hits, 60);
                let mut sorted: Vec<(usize, f32)> = merged.into_iter().collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

                let mut hits: Vec<SearchHit> = sorted
                    .into_iter()
                    .take(retrieval_limit)
                    .map(|(idx, score)| SearchHit {
                        drawer: rows[idx].0.clone(),
                        score,
                    })
                    .collect();

                rerank_search_hits(&mut hits, &query);
                hits.truncate(query.limit);

                return Ok(hits);
                } // end vectorlite block
            }

            // --- Brute-force fallback ---
            let sql = format!(
                "SELECT id, content, wing, room, source_file, chunk_index, added_by, filed_at, embedding
                 FROM drawers {filter_where}"
            );

            let mut stmt = conn.prepare(&sql)?;

            rows = stmt
                .query_map([], |row| {
                    let embedding: Vec<u8> = row.get(8)?;
                    let floats = blob_to_vector(&embedding);
                    Ok((row_to_drawer(row)?, floats))
                })?
                .filter_map(|r| r.ok())
                .collect();

            if rows.is_empty() {
                return Ok(Vec::new());
            }

            vector_hits = rows
                .iter()
                .enumerate()
                .map(|(idx, (_, emb))| (idx, cosine_similarity(&query_vector, emb)))
                .collect();

            let fts_hits: Vec<(usize, f32)> = if use_fts {
                let fts_sql =
                    "SELECT rowid, rank FROM drawers_fts WHERE drawers_fts MATCH ?1 ORDER BY rank LIMIT ?2";
                let mut fts_stmt = conn.prepare(fts_sql)?;

                let fts_results: Vec<(i64, f32)> = fts_stmt
                    .query_map(
                        params![query.query, retrieval_limit as i64],
                        |row| {
                            let rowid: i64 = row.get(0)?;
                            let rank: f32 = row.get(1)?;
                            Ok((rowid, rank))
                        },
                    )?
                    .filter_map(|r| r.ok())
                    .collect();

                // Build rowid -> index map by querying rowids
                let mut rowid_to_idx: HashMap<i64, usize> = HashMap::new();
                if !fts_results.is_empty() && !rows.is_empty() {
                    // Query rowids for current rows
                    let rowid_sql = format!("SELECT rowid FROM drawers {filter_where}");
                    let mut rid_stmt = conn.prepare(&rowid_sql)?;
                    let all_rowids: Vec<i64> = rid_stmt
                        .query_map([], |row| row.get(0))?
                        .filter_map(|r| r.ok())
                        .collect();
                    for (idx, rowid) in all_rowids.iter().enumerate() {
                        if idx < rows.len() {
                            rowid_to_idx.insert(*rowid, idx);
                        }
                    }
                }

                fts_results
                    .into_iter()
                    .filter_map(|(rowid, rank)| {
                        rowid_to_idx.get(&rowid).map(|&idx| (idx, rank))
                    })
                    .collect()
            } else {
                Vec::new()
            };

            let merged = reciprocal_rank_fusion(&vector_hits, &fts_hits, 60);
            let mut sorted: Vec<(usize, f32)> = merged.into_iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

            let mut hits: Vec<SearchHit> = sorted
                .into_iter()
                .take(retrieval_limit)
                .map(|(idx, score)| SearchHit {
                    drawer: rows[idx].0.clone(),
                    score,
                })
                .collect();

            rerank_search_hits(&mut hits, &query);
            hits.truncate(query.limit);

            Ok(hits)
        })
    }

    async fn status(&self) -> Result<StoreStatus> {
        self.with_conn(|conn| {
            let count: i64 =
                conn.query_row("SELECT COUNT(*) FROM drawers", [], |row| row.get(0))?;
            Ok(StoreStatus {
                total_drawers: count as usize,
            })
        })
    }

    async fn has_source_file(&self, source_file: &str) -> Result<bool> {
        self.with_conn(|conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM drawers WHERE source_file = ?1",
                params![source_file],
                |row| row.get(0),
            )?;
            Ok(count > 0)
        })
    }

    async fn source_files(&self) -> Result<HashSet<String>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT DISTINCT source_file FROM drawers WHERE source_file IS NOT NULL",
            )?;

            let files: HashSet<String> = stmt
                .query_map([], |row| row.get::<_, String>(0))?
                .filter_map(|r| r.ok())
                .collect();

            Ok(files)
        })
    }

    async fn room_counts(&self) -> Result<Vec<RoomStatus>> {
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT wing, room, COUNT(*) as cnt FROM drawers GROUP BY wing, room ORDER BY wing, room",
                )
                ?;

            let rows: Vec<_> = stmt
                .query_map([], |row| {
                    Ok(RoomStatus {
                        wing: row.get(0)?,
                        room: row.get(1)?,
                        total_drawers: row.get::<_, i64>(2)? as usize,
                    })
                })
                ?
                .filter_map(|r| r.ok())
                .collect();

            Ok(rows)
        })
    }
}

// --- Embedding Backend ---

enum EmbeddingBackend {
    Model2Vec {
        inner: Box<Mutex<Option<StaticModel>>>,
    },
    #[cfg(test)]
    Deterministic { dim: usize },
}

impl EmbeddingBackend {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let texts = [text.to_owned()];
        let mut embeddings = self.embed_batch(&texts)?;
        embeddings
            .pop()
            .ok_or_else(|| MempalaceError::Embedding("no embedding returned".to_owned()))
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Model2Vec { inner } => {
                let guard = inner
                    .lock()
                    .map_err(|_| MempalaceError::LockPoisoned("model2vec"))?;
                let model = guard
                    .as_ref()
                    .ok_or_else(|| MempalaceError::Embedding("model not initialized".to_owned()))?;
                let embeddings = model.encode(texts);
                Ok(embeddings.into_iter().map(normalize_vector).collect())
            }
            #[cfg(test)]
            Self::Deterministic { dim } => Ok(texts
                .iter()
                .map(|text| deterministic_embedding(text, *dim))
                .collect()),
        }
    }
}

// --- Vector helpers ---

fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_vector(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(bytes)
        })
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();

    // Split into chunks for potential autovectorization
    let chunks = len / 8;
    let remainder = len % 8;

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for i in 0..chunks {
        let base = i * 8;
        let a0 = a[base];
        let a1 = a[base + 1];
        let a2 = a[base + 2];
        let a3 = a[base + 3];
        let a4 = a[base + 4];
        let a5 = a[base + 5];
        let a6 = a[base + 6];
        let a7 = a[base + 7];

        let b0 = b[base];
        let b1 = b[base + 1];
        let b2 = b[base + 2];
        let b3 = b[base + 3];
        let b4 = b[base + 4];
        let b5 = b[base + 5];
        let b6 = b[base + 6];
        let b7 = b[base + 7];

        dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7;
        norm_a += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7;
        norm_b += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3 + b4 * b4 + b5 * b5 + b6 * b6 + b7 * b7;
    }

    for i in (chunks * 8)..(chunks * 8 + remainder) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn normalize_vector(mut values: Vec<f32>) -> Vec<f32> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut values {
            *value /= norm;
        }
    }
    values
}

// --- RRF (Reciprocal Rank Fusion) ---

fn reciprocal_rank_fusion(
    vector_hits: &[(usize, f32)],
    fts_hits: &[(usize, f32)],
    k: usize,
) -> HashMap<usize, f32> {
    // Sort vector hits by score descending
    let mut sorted_vec = vector_hits.to_vec();
    sorted_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut scores: HashMap<usize, f32> = HashMap::new();

    for (rank, (idx, _)) in sorted_vec.iter().enumerate() {
        *scores.entry(*idx).or_default() += 1.0 / (k as f32 + rank as f32 + 1.0);
    }

    for (rank, (idx, _)) in fts_hits.iter().enumerate() {
        *scores.entry(*idx).or_default() += 1.0 / (k as f32 + rank as f32 + 1.0);
    }

    scores
}

// --- Row mapping ---

fn row_to_drawer(row: &rusqlite::Row<'_>) -> std::result::Result<Drawer, rusqlite::Error> {
    row_to_drawer_offset(row, 0)
}

fn row_to_drawer_offset(
    row: &rusqlite::Row<'_>,
    offset: usize,
) -> std::result::Result<Drawer, rusqlite::Error> {
    Ok(Drawer {
        id: row.get(offset)?,
        content: row.get(offset + 1)?,
        metadata: DrawerMetadata {
            wing: row.get(offset + 2)?,
            room: row.get(offset + 3)?,
            source_file: row.get(offset + 4)?,
            chunk_index: row.get(offset + 5)?,
            added_by: row.get(offset + 6)?,
            filed_at: row.get(offset + 7)?,
        },
    })
}

// --- Utilities ---

fn sql_escape(value: &str) -> String {
    value.replace('\'', "''")
}

// --- Reranking ---

fn rerank_search_hits(hits: &mut [SearchHit], query: &SearchQuery) {
    let tokens = identifier_tokens(&query.query);
    let variants = identifier_variants(&tokens);

    hits.sort_by(|left, right| {
        let left_score = boosted_score(left, &tokens, &variants);
        let right_score = boosted_score(right, &tokens, &variants);
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(Ordering::Equal)
    });
}

fn boosted_score(hit: &SearchHit, tokens: &[String], variants: &[String]) -> f32 {
    let mut score = hit.score;
    let content = hit.drawer.content.to_ascii_lowercase();
    let source_file = hit
        .drawer
        .metadata
        .source_file
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let file_name = source_file
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(source_file.as_str());

    for variant in variants {
        if variant.is_empty() {
            continue;
        }

        if file_name.contains(variant) {
            score += 0.9;
        } else if source_file.contains(variant) {
            score += 0.6;
        }

        if content.contains(variant) {
            score += 0.45;
        }
    }

    if !tokens.is_empty() {
        if tokens.iter().all(|token| file_name.contains(token)) {
            score += 0.45;
        } else if tokens.iter().all(|token| source_file.contains(token)) {
            score += 0.2;
        }

        if tokens.iter().all(|token| content.contains(token)) {
            score += 0.15;
        }
    }

    if source_file.contains("\\generated\\")
        || source_file.contains("/generated/")
        || source_file.contains("\\target\\")
        || source_file.contains("/target/")
    {
        score -= 0.15;
    }

    score
}

fn identifier_tokens(text: &str) -> Vec<String> {
    let mut normalized = String::new();
    let mut previous_kind = CharacterKind::Boundary;

    for ch in text.chars() {
        let current_kind = CharacterKind::classify(ch);
        if current_kind == CharacterKind::Boundary {
            if !normalized.ends_with(' ') {
                normalized.push(' ');
            }
            previous_kind = CharacterKind::Boundary;
            continue;
        }

        if previous_kind == CharacterKind::Lower && current_kind == CharacterKind::Upper {
            normalized.push(' ');
        }

        normalized.push(ch.to_ascii_lowercase());
        previous_kind = current_kind;
    }

    normalized
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>()
}

fn identifier_variants(tokens: &[String]) -> Vec<String> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut variants = Vec::new();
    for variant in [
        tokens.join(" "),
        tokens.join("_"),
        tokens.join("-"),
        tokens.join(""),
    ] {
        if !variant.is_empty() && !variants.contains(&variant) {
            variants.push(variant);
        }
    }

    variants
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum CharacterKind {
    Lower,
    Upper,
    Digit,
    Boundary,
}

impl CharacterKind {
    fn classify(ch: char) -> Self {
        if ch.is_ascii_lowercase() {
            Self::Lower
        } else if ch.is_ascii_uppercase() {
            Self::Upper
        } else if ch.is_ascii_digit() {
            Self::Digit
        } else if ch.is_alphanumeric() {
            Self::Lower
        } else {
            Self::Boundary
        }
    }
}

// --- Test helpers ---

#[cfg(test)]
fn deterministic_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut values = vec![0.0; dim];
    for token in text
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
    {
        let token = token.to_ascii_lowercase();
        let index = stable_hash(&token) % dim;
        values[index] += 1.0;
    }
    normalize_vector(values)
}

#[cfg(test)]
fn stable_hash(value: &str) -> usize {
    let mut hash = 1_469_598_103_934_665_603_u64;
    for byte in value.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash as usize
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::{
        Drawer, DrawerMetadata, MemoryStore, SearchHit, SearchQuery, SqliteMemoryStore,
        identifier_tokens, identifier_variants, rerank_search_hits,
    };

    fn drawer(id: &str, content: &str, wing: &str, room: &str) -> Drawer {
        Drawer {
            id: id.to_owned(),
            content: content.to_owned(),
            metadata: DrawerMetadata {
                wing: wing.to_owned(),
                room: room.to_owned(),
                source_file: Some(format!("{id}.txt")),
                chunk_index: 0,
                added_by: "test".to_owned(),
                filed_at: Some("2026-04-08T00:00:00".to_owned()),
            },
        }
    }

    #[tokio::test]
    async fn add_get_delete_and_status_work() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());

        store
            .add_drawer(drawer(
                "drawer_1",
                "JWT tokens and authentication",
                "project",
                "backend",
            ))
            .await
            .unwrap();

        let status = store.status().await.unwrap();
        assert_eq!(status.total_drawers, 1);

        let fetched = store.get_drawer("drawer_1").await.unwrap().unwrap();
        assert_eq!(fetched.metadata.room, "backend");

        let deleted = store.delete_drawer("drawer_1").await.unwrap();
        assert!(deleted);
        assert_eq!(store.status().await.unwrap().total_drawers, 0);
    }

    #[tokio::test]
    async fn search_respects_filters() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());

        store
            .add_drawer(drawer(
                "drawer_1",
                "JWT tokens and authentication",
                "project",
                "backend",
            ))
            .await
            .unwrap();
        store
            .add_drawer(drawer(
                "drawer_2",
                "planning sprint roadmap",
                "notes",
                "planning",
            ))
            .await
            .unwrap();

        let mut query = SearchQuery::new("jwt authentication");
        query.limit = 2;
        query.wing = Some("project".to_owned());

        let results = store.search(query).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].drawer.id, "drawer_1");
    }

    #[tokio::test]
    async fn list_drawers_respects_wing_filter() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());

        store
            .add_drawer(drawer("drawer_1", "backend auth", "project", "backend"))
            .await
            .unwrap();
        store
            .add_drawer(drawer("drawer_2", "planning notes", "notes", "planning"))
            .await
            .unwrap();

        let project_drawers = store.list_drawers(Some("project")).await.unwrap();
        assert_eq!(project_drawers.len(), 1);
        assert_eq!(project_drawers[0].id, "drawer_1");
    }

    #[tokio::test]
    async fn add_drawers_batches_and_collects_source_files() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());

        store
            .add_drawers(vec![
                drawer("drawer_1", "backend auth", "project", "backend"),
                drawer("drawer_2", "frontend ui", "project", "frontend"),
            ])
            .await
            .unwrap();

        let status = store.status().await.unwrap();
        assert_eq!(status.total_drawers, 2);

        let source_files = store.source_files().await.unwrap();
        assert_eq!(source_files.len(), 2);
        assert!(source_files.contains("drawer_1.txt"));
        assert!(source_files.contains("drawer_2.txt"));
    }

    #[tokio::test]
    async fn search_returns_relevant_results() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());

        store
            .add_drawer(drawer(
                "drawer_1",
                "AAAK is the compressed memory dialect used by MemPalace",
                "project",
                "docs",
            ))
            .await
            .unwrap();

        let mut query = SearchQuery::new("aaak compressed memory");
        query.limit = 5;

        let results = store.search(query).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn identifier_helpers_cover_code_style_variants() {
        let tokens = identifier_tokens("FirstJoin");
        assert_eq!(tokens, vec!["first".to_owned(), "join".to_owned()]);

        let variants = identifier_variants(&tokens);
        assert!(variants.contains(&"first join".to_owned()));
        assert!(variants.contains(&"first_join".to_owned()));
        assert!(variants.contains(&"firstjoin".to_owned()));
    }

    #[test]
    fn reranker_boosts_exact_identifier_hits() {
        let query = SearchQuery::new("first join");
        let mut hits = vec![
            SearchHit {
                drawer: Drawer {
                    id: "sql".to_owned(),
                    content: "SELECT first_name FROM users JOIN guilds ON true".to_owned(),
                    metadata: DrawerMetadata {
                        wing: "reforged".to_owned(),
                        room: "crates".to_owned(),
                        source_file: Some(
                            r"F:\Dev\reforged\crates\infrastructure\src\repositories\postgres\user\read.rs"
                                .to_owned(),
                        ),
                        chunk_index: 0,
                        added_by: "test".to_owned(),
                        filed_at: None,
                    },
                },
                score: 1.0,
            },
            SearchHit {
                drawer: Drawer {
                    id: "code".to_owned(),
                    content:
                        "XtRequest::FirstJoin(_) => { let first_join_request = FirstJoinRequest { socket_id: pid }; }"
                            .to_owned(),
                    metadata: DrawerMetadata {
                        wing: "reforged".to_owned(),
                        room: "crates".to_owned(),
                        source_file: Some(
                            r"F:\Dev\reforged\crates\socket\src\connection.rs".to_owned(),
                        ),
                        chunk_index: 0,
                        added_by: "test".to_owned(),
                        filed_at: None,
                    },
                },
                score: 0.95,
            },
        ];

        rerank_search_hits(&mut hits, &query);
        assert_eq!(hits[0].drawer.id, "code");
    }

    #[tokio::test]
    async fn resize_vectorlite_no_table_returns_err() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());

        // Drop drawers_vec to simulate no table
        store
            .with_conn(|conn| {
                conn.execute_batch("DROP TABLE IF EXISTS drawers_vec;")
                    .unwrap();
                Ok(())
            })
            .unwrap();

        // No drawers_vec table → error (user should mine first)
        let result = store.resize_vectorlite_table(500_000).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn resize_vectorlite_old_limit_rebuilds_with_new_limit() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());

        // Must have vectorlite for this test
        if !store.vectorlite_available {
            eprintln!("vectorlite not available, skipping resize rebuild test");
            return;
        }

        // Manually set up the old table with the too-low max_elements
        store.with_conn(|conn| {
            let old_table_ddl = "
                CREATE VIRTUAL TABLE IF NOT EXISTS drawers_vec USING vectorlite(
                    embedding float32[512] cosine,
                    hnsw(max_elements=100000)
                );
            ";
            conn.execute_batch(old_table_ddl).unwrap();
            conn.execute_batch(super::VECTORLITE_TRIGGER_DDL).unwrap();

            // Insert a drawer with a zero embedding
            let zero_embedding: Vec<f32> = vec![0.0f32; 512];
            let emb_bytes: Vec<u8> = zero_embedding
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            conn.execute(
                "INSERT INTO drawers (id, content, wing, room, added_by, embedding) VALUES (?1,?2,?3,?4,?5,?6)",
                rusqlite::params!["test", "test content", "test", "test", "test", emb_bytes],
            ).unwrap();
            Ok(())
        }).unwrap();

        // Run resize with custom max_elements
        store.resize_vectorlite_table(500_000).await.unwrap();

        // Verify the new table has the requested max_elements
        store
            .with_conn(|conn| {
                let sql: String = conn
                    .query_row(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND name='drawers_vec'",
                        [],
                        |row| row.get(0),
                    )
                    .unwrap();
                assert!(
                    sql.contains("max_elements=500000"),
                    "expected max_elements=500000 in DDL, got: {sql}"
                );

                // Verify source data still exists
                let count: i64 = conn
                    .query_row("SELECT COUNT(*) FROM drawers", [], |row| row.get(0))
                    .unwrap();
                assert_eq!(count, 1, "source data should survive resize");
                Ok(())
            })
            .unwrap();
    }
}
