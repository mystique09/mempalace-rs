use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
};

use async_trait::async_trait;
use model2vec_rs::model::StaticModel;
use rusqlite::{Connection, params};

use mempalace_core::{
    Drawer, DrawerMetadata, MemoryStore, MempalaceError, Result, RoomStatus, SearchHit,
    SearchQuery, StoreStatus,
};

const DEFAULT_MODEL_REPO: &str = "minishlab/potion-code-16M-v2";
const LEGACY_DEFAULT_MODEL_REPO: &str = "minishlab/potion-base-32M";
const EMBEDDING_MODEL_KEY: &str = "embedding_model";
const EMBEDDING_REPRESENTATION_KEY: &str = "embedding_representation";
const CURRENT_EMBEDDING_REPRESENTATION: &str = "source-identifiers-v1";
const RRF_K: usize = 60;

/// Schema DDL for the SQLite store.
const SCHEMA_DDL: &str = "
CREATE TABLE IF NOT EXISTS drawers (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    retrieval_text TEXT,
    wing        TEXT NOT NULL,
    room        TEXT NOT NULL,
    source_file TEXT,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    added_by    TEXT NOT NULL,
    filed_at    TEXT,
    embedding   BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS store_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
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

CREATE TRIGGER IF NOT EXISTS drawers_au AFTER UPDATE OF content ON drawers BEGIN
    INSERT INTO drawers_fts(drawers_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO drawers_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE INDEX IF NOT EXISTS idx_drawers_wing_room ON drawers(wing, room);
";

/// Derive the local cache directory basename for a model repo slug.
fn model_basename(repo: &str) -> &str {
    repo.rsplit('/').next().unwrap_or(repo)
}

fn stored_embedding_model(conn: &Connection) -> Result<Option<String>> {
    match conn.query_row(
        "SELECT value FROM store_metadata WHERE key = ?1",
        params![EMBEDDING_MODEL_KEY],
        |row| row.get(0),
    ) {
        Ok(model) => Ok(Some(model)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn stored_embedding_representation(conn: &Connection) -> Result<Option<String>> {
    match conn.query_row(
        "SELECT value FROM store_metadata WHERE key = ?1",
        params![EMBEDDING_REPRESENTATION_KEY],
        |row| row.get(0),
    ) {
        Ok(version) => Ok(Some(version)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn migrate_schema(conn: &Connection) -> Result<()> {
    let has_retrieval_text = {
        let mut statement = conn.prepare("PRAGMA table_info(drawers)")?;
        statement
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<std::result::Result<Vec<_>, _>>()?
            .iter()
            .any(|column| column == "retrieval_text")
    };
    if !has_retrieval_text {
        conn.execute("ALTER TABLE drawers ADD COLUMN retrieval_text TEXT", [])?;
    }
    Ok(())
}

fn initialize_legacy_embedding_model(conn: &Connection) -> Result<()> {
    if stored_embedding_model(conn)?.is_some() {
        return Ok(());
    }

    let drawer_count: i64 = conn.query_row("SELECT COUNT(*) FROM drawers", [], |row| row.get(0))?;
    if drawer_count > 0 {
        conn.execute(
            "INSERT INTO store_metadata(key, value) VALUES (?1, ?2)",
            params![EMBEDDING_MODEL_KEY, LEGACY_DEFAULT_MODEL_REPO],
        )?;
    }

    Ok(())
}

fn initialize_embedding_representation(conn: &Connection) -> Result<()> {
    if stored_embedding_representation(conn)?.is_some() {
        return Ok(());
    }

    let drawer_count: i64 = conn.query_row("SELECT COUNT(*) FROM drawers", [], |row| row.get(0))?;
    if drawer_count == 0 {
        conn.execute(
            "INSERT INTO store_metadata(key, value) VALUES (?1, ?2)",
            params![
                EMBEDDING_REPRESENTATION_KEY,
                CURRENT_EMBEDDING_REPRESENTATION
            ],
        )?;
    }
    Ok(())
}

#[derive(Clone)]
pub struct SqliteMemoryStore {
    embedder: Arc<EmbeddingBackend>,
    conn: Arc<Mutex<Connection>>,
    embedding_model: String,
}

impl SqliteMemoryStore {
    pub fn embedding_dim(&self) -> Option<usize> {
        self.embedder.embedding_dim()
    }

    pub fn new(
        palace_path: impl Into<PathBuf>,
        model_dir: impl Into<PathBuf>,
        model_repo: Option<&str>,
    ) -> Result<Self> {
        let palace_path = palace_path.into();
        let model_dir = model_dir.into();
        let embedding_model = model_repo.unwrap_or(DEFAULT_MODEL_REPO).to_owned();

        let repo_or_path = model_repo
            .map(|r| r.to_owned())
            .or_else(|| {
                // Derive local cache dir from the default model name
                let local = model_dir.join(model_basename(DEFAULT_MODEL_REPO));
                local
                    .join("config.json")
                    .exists()
                    .then(|| local.to_string_lossy().to_string())
            })
            .unwrap_or_else(|| DEFAULT_MODEL_REPO.to_owned());

        let db_path = palace_path.join("store.sqlite3");

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&db_path)?;

        // Enable WAL mode for better concurrent access
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;

        // Keep each MCP process bounded. Exact search streams embedding blobs,
        // so a huge per-process SQLite cache/mmap provides little benefit.
        conn.execute_batch("PRAGMA mmap_size = 67108864;")?; // 64 MiB
        conn.execute_batch("PRAGMA cache_size = -16384;")?; // 16 MiB

        // Create schema
        conn.execute_batch(SCHEMA_DDL)?;
        migrate_schema(&conn)?;
        initialize_legacy_embedding_model(&conn)?;
        initialize_embedding_representation(&conn)?;
        // Legacy vectorlite triggers point at a connection-local index and make
        // concurrent MCP processes return incomplete results. Exact search uses
        // the canonical embeddings in `drawers`, so retire those triggers.
        conn.execute_batch(
            "DROP TRIGGER IF EXISTS drawers_vec_ai; DROP TRIGGER IF EXISTS drawers_vec_ad;",
        )?;

        let stored_dimension = match conn.query_row(
            "SELECT length(embedding) / 4 FROM drawers LIMIT 1",
            [],
            |row| row.get::<_, i64>(0),
        ) {
            Ok(dimension) => dimension as usize,
            Err(rusqlite::Error::QueryReturnedNoRows) => 0,
            Err(error) => return Err(error.into()),
        };

        Ok(Self {
            embedder: Arc::new(EmbeddingBackend::Model2Vec {
                repo_or_path,
                inner: Box::new(Mutex::new(None)),
                dimension: AtomicUsize::new(stored_dimension),
            }),
            conn: Arc::new(Mutex::new(conn)),
            embedding_model,
        })
    }

    #[cfg(test)]
    fn new_for_tests(palace_path: impl Into<PathBuf>) -> Self {
        Self::new_for_tests_with_model(palace_path, DEFAULT_MODEL_REPO)
    }

    #[cfg(test)]
    fn new_for_tests_with_model(
        palace_path: impl Into<PathBuf>,
        embedding_model: impl Into<String>,
    ) -> Self {
        let palace_path = palace_path.into();
        let db_path = palace_path.join("store.sqlite3");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(SCHEMA_DDL).unwrap();
        migrate_schema(&conn).unwrap();
        initialize_legacy_embedding_model(&conn).unwrap();
        initialize_embedding_representation(&conn).unwrap();

        Self {
            embedder: Arc::new(EmbeddingBackend::Deterministic { dim: 512 }),
            conn: Arc::new(Mutex::new(conn)),
            embedding_model: embedding_model.into(),
        }
    }

    fn ensure_embedding_metadata(&self, conn: &Connection, initialize: bool) -> Result<()> {
        let stored = stored_embedding_model(conn)?;
        match stored {
            Some(stored) if stored != self.embedding_model => {
                return Err(MempalaceError::Embedding(format!(
                    "store embeddings use model '{stored}', but this process is configured for '{}'; reopen with --model {stored} or run a full `mempalace-rs --model {} remine`",
                    self.embedding_model, self.embedding_model
                )));
            }
            Some(_) => {}
            None if initialize => {
                conn.execute(
                    "INSERT INTO store_metadata(key, value) VALUES (?1, ?2)",
                    params![EMBEDDING_MODEL_KEY, self.embedding_model],
                )?;
            }
            None => {}
        }

        let stored_representation = stored_embedding_representation(conn)?;
        match stored_representation.as_deref() {
            Some(CURRENT_EMBEDDING_REPRESENTATION) => Ok(()),
            Some(stored) => Err(MempalaceError::Embedding(format!(
                "store embeddings use representation '{stored}', but this version requires '{CURRENT_EMBEDDING_REPRESENTATION}'; run a full `mempalace-rs remine`"
            ))),
            None if initialize => {
                let drawer_count: i64 =
                    conn.query_row("SELECT COUNT(*) FROM drawers", [], |row| row.get(0))?;
                if drawer_count > 0 {
                    return Err(MempalaceError::Embedding(
                        "store embeddings use the legacy raw-content representation; run a full `mempalace-rs remine` before adding drawers"
                            .to_owned(),
                    ));
                }
                conn.execute(
                    "INSERT INTO store_metadata(key, value) VALUES (?1, ?2)",
                    params![
                        EMBEDDING_REPRESENTATION_KEY,
                        CURRENT_EMBEDDING_REPRESENTATION
                    ],
                )?;
                Ok(())
            }
            None => {
                let drawer_count: i64 =
                    conn.query_row("SELECT COUNT(*) FROM drawers", [], |row| row.get(0))?;
                if drawer_count == 0 {
                    Ok(())
                } else {
                    Err(MempalaceError::Embedding(
                        "store embeddings use the legacy raw-content representation; run a full `mempalace-rs remine`"
                            .to_owned(),
                    ))
                }
            }
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
        Self::search_condition_for(query, "")
    }

    fn search_condition_for(query: &SearchQuery, prefix: &str) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(wing) = &query.wing {
            parts.push(format!("{prefix}wing = '{}'", sql_escape(wing)));
        }
        if let Some(room) = &query.room {
            parts.push(format!("{prefix}room = '{}'", sql_escape(room)));
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

    fn retrieval_limit(limit: usize) -> usize {
        limit.min(1024).saturating_mul(8).clamp(1024, 8192)
    }

    /// Delete all drawers in a wing. Cascades to FTS via triggers.
    /// Returns the number of deleted drawers.
    pub async fn delete_wing(&self, wing: &str) -> Result<usize> {
        self.with_conn(|conn| {
            let deleted = conn.execute("DELETE FROM drawers WHERE wing = ?1", params![wing])?;
            Ok(deleted)
        })
    }

    pub async fn remine_all(&self, wing: Option<&str>) -> Result<usize> {
        const BATCH_SIZE: i64 = 256;

        let initial_model = self.with_conn(stored_embedding_model)?;
        let initial_representation = self.with_conn(stored_embedding_representation)?;
        if wing.is_some()
            && (initial_model
                .as_deref()
                .is_some_and(|stored| stored != self.embedding_model)
                || initial_representation.as_deref() != Some(CURRENT_EMBEDDING_REPRESENTATION))
        {
            return Err(MempalaceError::Embedding(
                "changing the embedding model or representation requires a full remine without --wing"
                    .to_owned(),
            ));
        }

        self.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TEMP TABLE IF NOT EXISTS embedding_migration (
                    id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                 );
                 DELETE FROM embedding_migration;",
            )?;
            Ok(())
        })?;

        let mut last_id: Option<String> = None;
        let mut total = 0_usize;
        loop {
            let batch = self.with_conn(|conn| {
                let (sql, wing_param): (&str, Option<&str>) = if wing.is_some() {
                    (
                        "SELECT id, content, retrieval_text, source_file FROM drawers
                         WHERE wing = ?1 AND (?2 IS NULL OR id > ?2)
                         ORDER BY id LIMIT ?3",
                        wing,
                    )
                } else {
                    (
                        "SELECT id, content, retrieval_text, source_file FROM drawers
                         WHERE (?2 IS NULL OR id > ?2)
                         ORDER BY id LIMIT ?3",
                        None,
                    )
                };
                let mut statement = conn.prepare(sql)?;
                let rows = statement
                    .query_map(params![wing_param, last_id.as_deref(), BATCH_SIZE], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, Option<String>>(2)?,
                            row.get::<_, Option<String>>(3)?,
                        ))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })?;

            if batch.is_empty() {
                break;
            }

            let texts = batch
                .iter()
                .map(|(_, content, retrieval_text, source_file)| {
                    embedding_input(content, retrieval_text.as_deref(), source_file.as_deref())
                })
                .collect::<Vec<_>>();
            let vectors = self.embedder.embed_batch(&texts)?;
            self.with_conn(|conn| {
                let transaction = conn.unchecked_transaction()?;
                for ((id, _, _, _), vector) in batch.iter().zip(&vectors) {
                    transaction.execute(
                        "INSERT INTO embedding_migration(id, embedding) VALUES (?1, ?2)",
                        params![id, vector_to_blob(vector)],
                    )?;
                }
                transaction.commit()?;
                Ok(())
            })?;

            total += batch.len();
            last_id = batch.last().map(|(id, _, _, _)| id.clone());
        }

        let result = self.with_conn(|conn| {
            let transaction = conn.unchecked_transaction()?;
            if stored_embedding_model(&transaction)? != initial_model {
                return Err(MempalaceError::Embedding(
                    "the embedding model changed during remine; retry the operation".to_owned(),
                ));
            }
            if stored_embedding_representation(&transaction)? != initial_representation {
                return Err(MempalaceError::Embedding(
                    "the embedding representation changed during remine; retry the operation"
                        .to_owned(),
                ));
            }

            let current_count: i64 = if let Some(wing) = wing {
                transaction.query_row(
                    "SELECT COUNT(*) FROM drawers WHERE wing = ?1",
                    params![wing],
                    |row| row.get(0),
                )?
            } else {
                transaction.query_row("SELECT COUNT(*) FROM drawers", [], |row| row.get(0))?
            };
            let matched_count: i64 = if let Some(wing) = wing {
                transaction.query_row(
                    "SELECT COUNT(*) FROM drawers AS d
                     JOIN embedding_migration AS m ON m.id = d.id
                     WHERE d.wing = ?1",
                    params![wing],
                    |row| row.get(0),
                )?
            } else {
                transaction.query_row(
                    "SELECT COUNT(*) FROM drawers AS d
                     JOIN embedding_migration AS m ON m.id = d.id",
                    [],
                    |row| row.get(0),
                )?
            };
            if current_count != total as i64 || matched_count != total as i64 {
                return Err(MempalaceError::Embedding(
                    "drawers changed during remine; no embeddings were replaced, retry the operation"
                        .to_owned(),
                ));
            }

            if let Some(wing) = wing {
                transaction.execute(
                    "UPDATE drawers SET embedding = (
                        SELECT embedding FROM embedding_migration WHERE id = drawers.id
                     ) WHERE wing = ?1",
                    params![wing],
                )?;
            } else {
                transaction.execute(
                    "UPDATE drawers SET embedding = (
                        SELECT embedding FROM embedding_migration WHERE id = drawers.id
                     )",
                    [],
                )?;
                transaction.execute(
                    "INSERT INTO store_metadata(key, value) VALUES (?1, ?2)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    params![EMBEDDING_MODEL_KEY, self.embedding_model],
                )?;
                transaction.execute(
                    "INSERT INTO store_metadata(key, value) VALUES (?1, ?2)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    params![
                        EMBEDDING_REPRESENTATION_KEY,
                        CURRENT_EMBEDDING_REPRESENTATION
                    ],
                )?;
            }

            transaction.commit()?;
            Ok(())
        });

        let _ = self.with_conn(|conn| {
            conn.execute_batch("DROP TABLE IF EXISTS temp.embedding_migration;")?;
            Ok(())
        });
        result?;
        Ok(total)
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

        self.with_conn(|conn| self.ensure_embedding_metadata(conn, true))?;

        let texts: Vec<String> = drawers
            .iter()
            .map(|drawer| {
                embedding_input(
                    &drawer.content,
                    drawer.retrieval_text.as_deref(),
                    drawer.metadata.source_file.as_deref(),
                )
            })
            .collect();
        let vectors = self.embedder.embed_batch(&texts)?;

        self.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;

            for (drawer, vector) in drawers.iter().zip(vectors.iter()) {
                let blob = vector_to_blob(vector);

                tx.execute(
                    "INSERT INTO drawers (id, content, retrieval_text, wing, room, source_file, chunk_index, added_by, filed_at, embedding)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    params![
                        drawer.id,
                        drawer.content,
                        drawer.retrieval_text,
                        drawer.metadata.wing,
                        drawer.metadata.room,
                        drawer.metadata.source_file,
                        drawer.metadata.chunk_index,
                        drawer.metadata.added_by,
                        drawer.metadata.filed_at,
                        blob,
                    ],
                )?;
            }

            tx.commit()?;
            Ok(())
        })
    }

    async fn get_drawer(&self, drawer_id: &str) -> Result<Option<Drawer>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, content, retrieval_text, wing, room, source_file, chunk_index, added_by, filed_at
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
                    "SELECT id, content, retrieval_text, wing, room, source_file, chunk_index, added_by, filed_at
                     FROM drawers WHERE wing = ?1"
                        .to_string(),
                    vec![w.to_string()],
                )
            } else {
                (
                    "SELECT id, content, retrieval_text, wing, room, source_file, chunk_index, added_by, filed_at
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

        self.with_conn(|conn| self.ensure_embedding_metadata(conn, false))?;

        let result_limit = query.limit.min(1024);
        let retrieval_limit = Self::retrieval_limit(query.limit);
        let query_views = query_embedding_views(&query.query);
        let query_vectors = self.embedder.embed_batch(&query_views)?;
        let original_query_vector = query_vectors
            .first()
            .ok_or_else(|| MempalaceError::Embedding("no query embedding returned".to_owned()))?;

        self.with_conn(|conn| {
            let filter_where = Self::search_filter_clause(&query);
            let vector_sql = format!("SELECT rowid, embedding FROM drawers {filter_where}");
            let mut vector_stmt = conn.prepare(&vector_sql)?;
            let mut vector_rows = vector_stmt.query([])?;
            let mut vector_heaps = query_vectors
                .iter()
                .map(|_| BinaryHeap::with_capacity(retrieval_limit.saturating_add(1)))
                .collect::<Vec<BinaryHeap<Reverse<VectorCandidate>>>>();

            while let Some(row) = vector_rows.next()? {
                let rowid: i64 = row.get(0)?;
                let embedding: Vec<u8> = row.get(1)?;
                for (query_vector, heap) in query_vectors.iter().zip(&mut vector_heaps) {
                    let similarity = cosine_similarity_blob(query_vector, &embedding)?;
                    retain_top_candidate(
                        heap,
                        VectorCandidate { rowid, similarity },
                        retrieval_limit,
                    );
                }
            }
            let mut fused_scores = HashMap::<i64, f32>::new();
            for (view_index, heap) in vector_heaps.into_iter().enumerate() {
                let mut candidates = heap
                    .into_iter()
                    .map(|Reverse(candidate)| candidate)
                    .collect::<Vec<_>>();
                candidates.sort_by(|left, right| right.cmp(left));
                let weight = embedding_view_weight(view_index);
                for (rank, candidate) in candidates.iter().enumerate() {
                    *fused_scores.entry(candidate.rowid).or_default() +=
                        weight * reciprocal_rank_score(rank, RRF_K);
                }
            }

            for (view_index, ranked_rowids) in
                fts_ranked_candidates(conn, &query, retrieval_limit)?
                    .into_iter()
                    .enumerate()
            {
                let weight = lexical_view_weight(view_index);
                for (rank, rowid) in ranked_rowids.into_iter().enumerate() {
                    *fused_scores.entry(rowid).or_default() +=
                        weight * reciprocal_rank_score(rank, RRF_K);
                }
            }
            if fused_scores.is_empty() {
                return Ok(Vec::new());
            }

            let mut ordered_rowids = fused_scores.keys().copied().collect::<Vec<_>>();
            ordered_rowids.sort_unstable();
            let rowid_list = ordered_rowids
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let fetch_sql = format!(
                "SELECT rowid, id, content, retrieval_text, wing, room, source_file, chunk_index, added_by, filed_at, embedding
                 FROM drawers WHERE rowid IN ({rowid_list})"
            );
            let mut fetch_stmt = conn.prepare(&fetch_sql)?;
            let mut fetched_rows = fetch_stmt.query([])?;
            let mut hits = Vec::with_capacity(ordered_rowids.len());

            while let Some(row) = fetched_rows.next()? {
                let rowid: i64 = row.get(0)?;
                let embedding: Vec<u8> = row.get(10)?;
                let score = cosine_similarity_blob(original_query_vector, &embedding)?;
                let relevance = fused_scores[&rowid];
                hits.push(SearchHit {
                    drawer: row_to_drawer_offset(row, 1)?,
                    score,
                    relevance,
                });
            }

            normalize_relevance_scores(&mut hits);
            rerank_search_hits(&mut hits, &query);
            hits.retain(|h| {
                query
                    .min_score
                    .is_none_or(|threshold| h.score >= threshold)
            });
            hits.truncate(result_limit);

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
        repo_or_path: String,
        inner: Box<Mutex<Option<StaticModel>>>,
        dimension: AtomicUsize,
    },
    #[cfg(test)]
    Deterministic { dim: usize },
}

impl EmbeddingBackend {
    fn embedding_dim(&self) -> Option<usize> {
        match self {
            Self::Model2Vec { dimension, .. } => {
                let dimension = dimension.load(AtomicOrdering::Relaxed);
                (dimension > 0).then_some(dimension)
            }
            #[cfg(test)]
            Self::Deterministic { dim } => Some(*dim),
        }
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Model2Vec {
                repo_or_path,
                inner,
                dimension,
            } => {
                let mut guard = inner
                    .lock()
                    .map_err(|_| MempalaceError::LockPoisoned("model2vec"))?;
                if guard.is_none() {
                    *guard = Some(
                        StaticModel::from_pretrained(repo_or_path, None, None, None)
                            .map_err(|err| MempalaceError::Embedding(err.to_string()))?,
                    );
                }
                let model = guard.as_ref().expect("model initialized above");
                let embeddings = model.encode(texts);
                if let Some(first) = embeddings.first() {
                    dimension.store(first.len(), AtomicOrdering::Relaxed);
                }
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

#[derive(Debug, Copy, Clone)]
struct VectorCandidate {
    rowid: i64,
    similarity: f32,
}

impl PartialEq for VectorCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.rowid == other.rowid && self.similarity.to_bits() == other.similarity.to_bits()
    }
}

impl Eq for VectorCandidate {}

impl PartialOrd for VectorCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VectorCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.similarity
            .total_cmp(&other.similarity)
            .then_with(|| self.rowid.cmp(&other.rowid))
    }
}

fn retain_top_candidate(
    heap: &mut BinaryHeap<Reverse<VectorCandidate>>,
    candidate: VectorCandidate,
    limit: usize,
) {
    if limit == 0 || !candidate.similarity.is_finite() {
        return;
    }

    if heap.len() < limit {
        heap.push(Reverse(candidate));
        return;
    }

    if heap
        .peek()
        .is_some_and(|Reverse(lowest)| candidate > *lowest)
    {
        heap.pop();
        heap.push(Reverse(candidate));
    }
}

fn cosine_similarity_blob(query: &[f32], blob: &[u8]) -> Result<f32> {
    let expected_bytes = query.len().saturating_mul(std::mem::size_of::<f32>());
    if blob.len() != expected_bytes {
        return Err(MempalaceError::Embedding(format!(
            "stored embedding has {} dimensions, but the configured model produced {}; run a full `mempalace-rs remine`",
            blob.len() / std::mem::size_of::<f32>(),
            query.len()
        )));
    }

    let mut dot = 0.0_f32;
    let mut query_norm = 0.0_f32;
    let mut stored_norm = 0.0_f32;
    for (query_value, chunk) in query.iter().zip(blob.chunks_exact(4)) {
        let stored_value = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
        dot += query_value * stored_value;
        query_norm += query_value * query_value;
        stored_norm += stored_value * stored_value;
    }

    if query_norm == 0.0 || stored_norm == 0.0 {
        return Ok(0.0);
    }

    Ok((dot / (query_norm.sqrt() * stored_norm.sqrt())).clamp(-1.0, 1.0))
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

// --- Rank scoring ---

fn reciprocal_rank_score(rank: usize, k: usize) -> f32 {
    1.0 / (k as f32 + rank as f32 + 1.0)
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
        retrieval_text: row.get(offset + 2)?,
        metadata: DrawerMetadata {
            wing: row.get(offset + 3)?,
            room: row.get(offset + 4)?,
            source_file: row.get(offset + 5)?,
            chunk_index: row.get(offset + 6)?,
            added_by: row.get(offset + 7)?,
            filed_at: row.get(offset + 8)?,
        },
    })
}

// --- Utilities ---

fn sql_escape(value: &str) -> String {
    value.replace('\'', "''")
}

fn embedding_input(
    content: &str,
    retrieval_text: Option<&str>,
    source_file: Option<&str>,
) -> String {
    const MAX_IDENTIFIER_TOKENS: usize = 48;

    let body = retrieval_text.unwrap_or(content);
    let source = source_file.map(compact_source_path).unwrap_or_default();
    let mut seen = HashSet::new();
    let mut identifiers = Vec::new();
    for token in source_file
        .into_iter()
        .chain(std::iter::once(body))
        .flat_map(identifier_tokens)
    {
        if seen.insert(token.clone()) {
            identifiers.push(token);
            if identifiers.len() == MAX_IDENTIFIER_TOKENS {
                break;
            }
        }
    }

    let mut input = String::with_capacity(body.len().saturating_add(384));
    if !source.is_empty() {
        input.push_str("source: ");
        input.push_str(&source);
        input.push('\n');
    }
    if !identifiers.is_empty() {
        input.push_str("identifiers: ");
        input.push_str(&identifiers.join(" "));
        input.push('\n');
    }
    input.push_str(body);
    input
}

fn compact_source_path(source_file: &str) -> String {
    let mut components = source_file
        .split(['/', '\\'])
        .filter(|component| !component.is_empty())
        .rev()
        .take(6)
        .collect::<Vec<_>>();
    components.reverse();
    components.join("/")
}

fn query_embedding_views(query: &str) -> Vec<String> {
    const MAX_VIEWS: usize = 3;

    let original_tokens = identifier_tokens(query);
    let mut views = vec![query.to_owned()];
    if original_tokens.len() < 2 {
        return views;
    }

    let canonical_tokens = canonical_query_tokens(&original_tokens);
    let concepts = semantic_concept_tokens(&original_tokens);
    push_unique_nonempty(&mut views, canonical_tokens.join(" "));

    if !concepts.is_empty() {
        let mut concept_terms = concepts.clone();
        concept_terms.extend(
            concepts
                .windows(2)
                .map(|pair| format!("{}{}", pair[0], pair[1])),
        );
        push_unique_nonempty(&mut views, concept_terms.join(" "));
    }

    views.truncate(MAX_VIEWS);
    views
}

fn push_unique_nonempty(values: &mut Vec<String>, value: String) {
    if !value.trim().is_empty() && !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

fn embedding_view_weight(view_index: usize) -> f32 {
    match view_index {
        0 => 1.0,
        1 => 0.65,
        _ => 0.8,
    }
}

fn lexical_view_weight(view_index: usize) -> f32 {
    if view_index == 0 { 0.9 } else { 1.2 }
}

fn fts_ranked_candidates(
    conn: &Connection,
    query: &SearchQuery,
    limit: usize,
) -> Result<Vec<Vec<i64>>> {
    let fts_exists: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'drawers_fts')",
        [],
        |row| row.get(0),
    )?;
    if !fts_exists {
        return Ok(Vec::new());
    }

    let allowed_rowids = if SqliteMemoryStore::search_condition(query).is_some() {
        let sql = format!(
            "SELECT rowid FROM drawers {}",
            SqliteMemoryStore::search_filter_clause(query)
        );
        let mut statement = conn.prepare(&sql)?;
        Some(
            statement
                .query_map([], |row| row.get::<_, i64>(0))?
                .collect::<std::result::Result<HashSet<_>, _>>()?,
        )
    } else {
        None
    };

    // Keep FTS5's optimized rank traversal independent from the metadata
    // lookup. Joining before ordering makes broad prefix queries pathological,
    // while truncating before filtering can hide every result from a small
    // wing. Streaming ranked rowids and filtering them in memory preserves
    // both the lexical order and the filter contract.
    let mut statement = conn.prepare(
        "SELECT rowid
         FROM drawers_fts
         WHERE drawers_fts MATCH ?1
         ORDER BY rank",
    )?;
    let mut ranked_views = Vec::new();
    for fts_query in fts_query_views(&query.query) {
        let mut rows = statement.query(params![fts_query])?;
        let mut rowids = Vec::with_capacity(limit);
        while let Some(row) = rows.next()? {
            let rowid = row.get::<_, i64>(0)?;
            if allowed_rowids
                .as_ref()
                .is_none_or(|allowed| allowed.contains(&rowid))
            {
                rowids.push(rowid);
                if rowids.len() == limit {
                    break;
                }
            }
        }
        ranked_views.push(rowids);
    }
    Ok(ranked_views)
}

fn fts_query_views(query: &str) -> Vec<String> {
    let tokens = identifier_tokens(query);
    let mut views = Vec::new();
    if !tokens.is_empty() {
        views.push(
            tokens
                .iter()
                .map(|token| format!("{token}*"))
                .collect::<Vec<_>>()
                .join(" OR "),
        );
    }

    if tokens.len() < 2 {
        return views;
    }

    let concepts = semantic_concept_tokens(&tokens);
    if !concepts.is_empty() {
        let mut terms = concepts
            .iter()
            .map(|token| format!("{token}*"))
            .collect::<Vec<_>>();
        for pair in concepts.windows(2) {
            terms.push(format!("{}{}*", pair[0], pair[1]));
            terms.push(format!("\"{} {}\"", pair[0], pair[1]));
        }
        views.push(terms.join(" OR "));
    }
    views
}

fn canonical_query_tokens(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .map(|token| synonym_canonical(token).unwrap_or(token).to_owned())
        .collect()
}

fn semantic_concept_tokens(tokens: &[String]) -> Vec<String> {
    let mut concepts = Vec::new();
    for token in tokens {
        if let Some(canonical) = synonym_canonical(token)
            && !concepts.iter().any(|concept| concept == canonical)
        {
            concepts.push(canonical.to_owned());
        }
    }
    concepts
}

fn synonym_canonical(token: &str) -> Option<&'static str> {
    const GROUPS: &[(&str, &[&str])] = &[
        (
            "handle",
            &["handle", "process", "execute", "dispatch", "perform", "run"],
        ),
        ("player", &["player", "user", "client", "gamer", "account"]),
        (
            "first",
            &["first", "initial", "initially", "earliest", "new"],
        ),
        (
            "join",
            &[
                "join",
                "login",
                "logon",
                "signin",
                "signon",
                "authenticate",
                "authentication",
                "enter",
                "connect",
                "connection",
            ],
        ),
        ("create", &["create", "add", "insert", "build", "make"]),
        ("delete", &["delete", "remove", "erase", "destroy"]),
        ("fetch", &["fetch", "get", "retrieve", "load", "read"]),
        ("update", &["update", "modify", "change", "edit", "write"]),
        ("error", &["error", "failure", "exception", "fault"]),
        (
            "config",
            &["config", "configuration", "settings", "preferences"],
        ),
    ];

    GROUPS
        .iter()
        .find_map(|(canonical, aliases)| aliases.contains(&token).then_some(*canonical))
}

// --- Reranking ---

fn rerank_search_hits(hits: &mut [SearchHit], query: &SearchQuery) {
    let mut tokens = identifier_tokens(&query.query);
    let concepts = if tokens.len() >= 2 {
        semantic_concept_tokens(&tokens)
    } else {
        Vec::new()
    };
    for concept in &concepts {
        if !tokens.contains(concept) {
            tokens.push(concept.clone());
        }
    }
    let mut variants = identifier_variants(&identifier_tokens(&query.query));
    for variant in identifier_variants(&concepts) {
        if !variants.contains(&variant) {
            variants.push(variant);
        }
    }

    for hit in hits.iter_mut() {
        hit.relevance = boosted_score(hit, &tokens, &variants);
    }
    normalize_relevance_scores(hits);

    hits.sort_by(|left, right| {
        right
            .relevance
            .partial_cmp(&left.relevance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| right.score.total_cmp(&left.score))
            .then_with(|| left.drawer.id.cmp(&right.drawer.id))
    });
}

fn normalize_relevance_scores(hits: &mut [SearchHit]) {
    let max_score = hits
        .iter()
        .map(|hit| hit.relevance)
        .filter(|score| score.is_finite())
        .fold(0.0_f32, f32::max);

    if max_score <= 0.0 {
        return;
    }

    for hit in hits {
        hit.relevance = (hit.relevance / max_score).clamp(0.0, 1.0);
    }
}

fn boosted_score(hit: &SearchHit, tokens: &[String], variants: &[String]) -> f32 {
    let mut score = hit.relevance;
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

    if variants.iter().any(|variant| file_name.contains(variant)) {
        score += 0.9;
    } else if variants.iter().any(|variant| source_file.contains(variant)) {
        score += 0.6;
    }

    if variants.iter().any(|variant| content.contains(variant)) {
        score += 0.45;
    }
    if variants.iter().any(|variant| {
        !variant.chars().any(|ch| matches!(ch, ' ' | '_' | '-'))
            && variant.len() >= 6
            && content.contains(variant)
    }) {
        score += 0.2;
    }

    let content_tokens = identifier_tokens(&hit.drawer.content)
        .into_iter()
        .collect::<HashSet<_>>();
    let path_tokens = identifier_tokens(&source_file)
        .into_iter()
        .collect::<HashSet<_>>();
    let content_matches = tokens
        .iter()
        .filter(|token| content_tokens.contains(*token))
        .count();
    let path_matches = tokens
        .iter()
        .filter(|token| path_tokens.contains(*token))
        .count();
    if !tokens.is_empty() {
        score += 0.04 * content_matches.min(6) as f32;
        score += 0.05 * path_matches.min(4) as f32;
        if content_matches >= 2 {
            score += 0.1 * (content_matches as f32 / tokens.len() as f32);
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
        .filter(|token| token.len() > 1 && !is_stopword(token))
        .map(str::to_owned)
        .collect::<Vec<_>>()
}

fn identifier_variants(tokens: &[String]) -> Vec<String> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut variants = Vec::new();
    let max_width = tokens.len().min(3);
    let min_width = if tokens.len() == 1 { 1 } else { 2 };
    for width in min_width..=max_width {
        for window in tokens.windows(width) {
            for variant in [
                window.join(" "),
                window.join("_"),
                window.join("-"),
                window.join(""),
            ] {
                if !variant.is_empty() && !variants.contains(&variant) {
                    variants.push(variant);
                }
            }
        }
    }

    variants
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "a" | "an"
            | "the"
            | "and"
            | "or"
            | "of"
            | "to"
            | "for"
            | "from"
            | "in"
            | "on"
            | "at"
            | "by"
            | "with"
            | "without"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "this"
            | "that"
            | "these"
            | "those"
            | "very"
            | "how"
            | "what"
            | "when"
            | "where"
            | "why"
            | "who"
            | "which"
            | "let"
            | "mut"
            | "pub"
            | "fn"
            | "impl"
            | "self"
            | "super"
            | "crate"
            | "use"
            | "mod"
            | "async"
            | "await"
            | "return"
            | "match"
            | "if"
            | "else"
            | "while"
            | "loop"
            | "true"
            | "false"
            | "some"
            | "none"
            | "ok"
            | "err"
    )
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
    use rusqlite::params;
    use tempfile::tempdir;

    use super::{
        CURRENT_EMBEDDING_REPRESENTATION, Drawer, DrawerMetadata, EMBEDDING_REPRESENTATION_KEY,
        MemoryStore, SearchHit, SearchQuery, SqliteMemoryStore, fts_ranked_candidates,
        identifier_tokens, identifier_variants, rerank_search_hits, stored_embedding_model,
        stored_embedding_representation,
    };

    fn drawer(id: &str, content: &str, wing: &str, room: &str) -> Drawer {
        Drawer {
            id: id.to_owned(),
            content: content.to_owned(),
            retrieval_text: None,
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

    #[tokio::test]
    async fn search_bridges_natural_language_login_to_first_join_code() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());

        store
            .add_drawers(vec![
                drawer(
                    "first_join_handler",
                    "XtRequest::FirstJoin(_) => emulator_client.first_join(FirstJoinRequest { socket_id: pid }).await",
                    "reforged",
                    "crates",
                ),
                drawer(
                    "player_loop",
                    "for player in players { process player game position and stamina updates }",
                    "reforged",
                    "crates",
                ),
                drawer(
                    "login_mapper",
                    "map the user login location and login status into a database model",
                    "reforged",
                    "crates",
                ),
                drawer(
                    "other_wing_first_join",
                    "XtRequest::FirstJoin(_) => process the initial player connection",
                    "other",
                    "crates",
                ),
            ])
            .await
            .unwrap();

        let mut query = SearchQuery::new("process a player's very first game login");
        query.limit = 1;
        query.wing = Some("reforged".to_owned());
        query.room = Some("crates".to_owned());

        let results = store.search(query).await.unwrap();

        assert_eq!(results[0].drawer.id, "first_join_handler");
    }

    #[test]
    fn fts_filter_does_not_lose_small_wing_after_global_matches() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());

        store
            .with_conn(|conn| {
                let transaction = conn.unchecked_transaction()?;
                {
                    let mut insert = transaction.prepare(
                        "INSERT INTO drawers
                         (id, content, wing, room, added_by, embedding)
                         VALUES (?1, 'first join handler', ?2, 'crates', 'test', ?3)",
                    )?;
                    for index in 0..8_200 {
                        insert.execute(params![
                            format!("other_{index}"),
                            "other",
                            vec![0_u8; 4]
                        ])?;
                    }
                    insert.execute(params!["target", "reforged", vec![0_u8; 4]])?;
                }
                transaction.commit()?;

                let target_rowid: i64 =
                    conn.query_row("SELECT rowid FROM drawers WHERE id = 'target'", [], |row| {
                        row.get(0)
                    })?;
                let mut query = SearchQuery::new("first join");
                query.wing = Some("reforged".to_owned());
                let ranked_views = fts_ranked_candidates(conn, &query, 1)?;

                assert!(!ranked_views.is_empty());
                assert!(
                    ranked_views
                        .iter()
                        .all(|ranked| ranked.as_slice() == [target_rowid])
                );
                Ok(())
            })
            .unwrap();
    }

    #[tokio::test]
    async fn exact_login_query_still_prefers_literal_login_code() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());
        store
            .add_drawers(vec![
                drawer(
                    "first_join_handler",
                    "XtRequest::FirstJoin(_) => emulator_client.first_join(request).await",
                    "reforged",
                    "crates",
                ),
                drawer(
                    "login_mapper",
                    "map the user login location and login status into a database model",
                    "reforged",
                    "crates",
                ),
            ])
            .await
            .unwrap();

        let mut query = SearchQuery::new("login");
        query.limit = 1;
        query.wing = Some("reforged".to_owned());

        let results = store.search(query).await.unwrap();

        assert_eq!(results[0].drawer.id, "login_mapper");
    }

    #[tokio::test]
    async fn search_embeds_retrieval_text_but_returns_verbatim_content() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());
        let mut target = drawer(
            "semantic_code",
            "XtRequest::FirstJoin(_) => dispatch(request)",
            "reforged",
            "crates",
        );
        target.retrieval_text = Some("nebula quasar semantic bridge".to_owned());
        store
            .add_drawers(vec![
                target,
                drawer(
                    "distractor",
                    "update every player position during the game loop",
                    "reforged",
                    "crates",
                ),
            ])
            .await
            .unwrap();

        let results = store
            .search(SearchQuery::new("nebula quasar semantic bridge"))
            .await
            .unwrap();

        assert_eq!(results[0].drawer.id, "semantic_code");
        assert_eq!(
            results[0].drawer.content,
            "XtRequest::FirstJoin(_) => dispatch(request)"
        );
        assert_eq!(
            store
                .get_drawer("semantic_code")
                .await
                .unwrap()
                .unwrap()
                .retrieval_text
                .as_deref(),
            Some("nebula quasar semantic bridge")
        );
    }

    #[tokio::test]
    async fn semantic_search_reports_lancedb_style_normalized_relevance() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());

        store
            .add_drawers(vec![
                drawer(
                    "relevant",
                    "AAAK is the compressed memory dialect used by MemPalace",
                    "project",
                    "docs",
                ),
                drawer(
                    "distractor",
                    "SQLite transaction tuning and WAL checkpoint behavior",
                    "project",
                    "storage",
                ),
            ])
            .await
            .unwrap();

        let results = store
            .search(SearchQuery::new("aaak compressed memory"))
            .await
            .unwrap();

        assert_eq!(results[0].drawer.id, "relevant");
        assert!(
            (results[0].relevance - 1.0).abs() < f32::EPSILON,
            "top semantic result should be normalized to 1.0, got {}",
            results[0].relevance
        );
        assert!(
            results
                .iter()
                .all(|hit| (0.0..=1.0).contains(&hit.relevance))
        );
        assert!(results.iter().all(|hit| (-1.0..=1.0).contains(&hit.score)));
    }

    #[tokio::test]
    async fn semantic_search_does_not_depend_on_fts() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());
        store
            .add_drawers(vec![
                drawer(
                    "semantic_target",
                    "A player entering the world for the first time",
                    "project",
                    "gameplay",
                ),
                drawer(
                    "distractor",
                    "SQLite transaction tuning and WAL checkpoints",
                    "project",
                    "storage",
                ),
            ])
            .await
            .unwrap();
        store
            .with_conn(|conn| {
                conn.execute_batch(
                    "DROP TRIGGER IF EXISTS drawers_ai;
                     DROP TRIGGER IF EXISTS drawers_ad;
                     DROP TABLE drawers_fts;",
                )?;
                Ok(())
            })
            .unwrap();

        let results = store
            .search(SearchQuery::new("a player's initial game login"))
            .await
            .unwrap();

        assert_eq!(results[0].drawer.id, "semantic_target");
        assert!(results[0].score > 0.0);
        assert_eq!(results[0].relevance, 1.0);
    }

    #[tokio::test]
    async fn search_rejects_embeddings_from_a_different_model() {
        let tmp = tempdir().unwrap();
        let original =
            SqliteMemoryStore::new_for_tests_with_model(tmp.path(), "minishlab/potion-base-32M");
        original
            .add_drawer(drawer(
                "drawer_1",
                "AAAK compressed memory",
                "project",
                "docs",
            ))
            .await
            .unwrap();
        drop(original);

        let incompatible = SqliteMemoryStore::new_for_tests_with_model(
            tmp.path(),
            "minishlab/potion-retrieval-32M",
        );
        let error = incompatible
            .search(SearchQuery::new("aaak memory"))
            .await
            .expect_err("mixed-model search must fail instead of returning degraded results");

        let message = error.to_string();
        assert!(message.contains("potion-base-32M"));
        assert!(message.contains("remine"));

        assert_eq!(incompatible.remine_all(None).await.unwrap(), 1);
        let results = incompatible
            .search(SearchQuery::new("aaak memory"))
            .await
            .unwrap();
        assert_eq!(results[0].drawer.id, "drawer_1");
        assert_eq!(
            incompatible.with_conn(stored_embedding_model).unwrap(),
            Some("minishlab/potion-retrieval-32M".to_owned())
        );
    }

    #[tokio::test]
    async fn legacy_embedding_representation_requires_full_remine() {
        let tmp = tempdir().unwrap();
        let store = SqliteMemoryStore::new_for_tests(tmp.path());
        store
            .add_drawer(drawer(
                "legacy_raw",
                "first join handler",
                "reforged",
                "crates",
            ))
            .await
            .unwrap();
        store
            .with_conn(|conn| {
                conn.execute(
                    "DELETE FROM store_metadata WHERE key = ?1",
                    params![EMBEDDING_REPRESENTATION_KEY],
                )?;
                Ok(())
            })
            .unwrap();

        let error = store
            .search(SearchQuery::new("first join"))
            .await
            .expect_err("raw-content vectors must not mix with enriched vectors");
        assert!(error.to_string().contains("full `mempalace-rs remine`"));

        assert_eq!(store.remine_all(None).await.unwrap(), 1);
        assert_eq!(
            store
                .with_conn(stored_embedding_representation)
                .unwrap()
                .as_deref(),
            Some(CURRENT_EMBEDDING_REPRESENTATION)
        );
        assert_eq!(
            store.search(SearchQuery::new("first join")).await.unwrap()[0]
                .drawer
                .id,
            "legacy_raw"
        );
    }

    #[tokio::test]
    async fn metadata_less_legacy_store_requires_full_remine_for_new_default() {
        let tmp = tempdir().unwrap();
        let conn = rusqlite::Connection::open(tmp.path().join("store.sqlite3")).unwrap();
        conn.execute_batch(super::SCHEMA_DDL).unwrap();
        conn.execute(
            "INSERT INTO drawers (id, content, wing, room, added_by, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                "legacy_drawer",
                "legacy MiniLM-compatible content",
                "project",
                "docs",
                "test",
                super::vector_to_blob(&vec![0.0; 512]),
            ],
        )
        .unwrap();
        drop(conn);

        let store = SqliteMemoryStore::new(
            tmp.path(),
            tmp.path().join("models"),
            Some("minishlab/potion-code-16M-v2"),
        )
        .unwrap();
        let error = store
            .search(SearchQuery::new("legacy content"))
            .await
            .expect_err("legacy vectors must not be labeled as the new default model");

        let message = error.to_string();
        assert!(message.contains("potion-base-32M"));
        assert!(message.contains("remine"));
    }

    #[tokio::test]
    async fn concurrent_store_search_sees_drawers_written_by_other_connections() {
        let tmp = tempdir().unwrap();
        let writer_a = SqliteMemoryStore::new_for_tests(tmp.path());
        let writer_b = SqliteMemoryStore::new_for_tests(tmp.path());

        writer_a
            .add_drawer(drawer(
                "shared_target",
                "nebula protocol ownership decision",
                "project",
                "decisions",
            ))
            .await
            .unwrap();
        writer_b
            .add_drawer(drawer(
                "local_distractor",
                "frontend button colors and spacing",
                "project",
                "design",
            ))
            .await
            .unwrap();

        let results = writer_b
            .search(SearchQuery::new("nebula protocol ownership"))
            .await
            .unwrap();

        assert_eq!(results[0].drawer.id, "shared_target");
    }

    #[tokio::test]
    async fn production_startup_is_lazy_and_uses_bounded_sqlite_memory() {
        let tmp = tempdir().unwrap();
        let missing_model = tmp.path().join("missing-model");
        let store = SqliteMemoryStore::new(
            tmp.path(),
            tmp.path().join("models"),
            Some(missing_model.to_str().unwrap()),
        )
        .expect("opening the store must not load the embedding model");

        assert_eq!(store.status().await.unwrap().total_drawers, 0);
        assert_eq!(store.embedding_dim(), None);
        store
            .with_conn(|conn| {
                let mmap_size: i64 = conn.query_row("PRAGMA mmap_size", [], |row| row.get(0))?;
                let cache_size: i64 = conn.query_row("PRAGMA cache_size", [], |row| row.get(0))?;
                assert!(mmap_size <= 67_108_864);
                assert_eq!(cache_size, -16_384);
                Ok(())
            })
            .unwrap();

        let error = store
            .search(SearchQuery::new("load the model now"))
            .await
            .expect_err("the first embedding operation should load the missing model");
        assert!(error.to_string().contains("embedding error"));
    }

    #[test]
    fn identifier_helpers_cover_code_style_variants() {
        let tokens = identifier_tokens("FirstJoin");
        assert_eq!(tokens, vec!["first".to_owned(), "join".to_owned()]);

        assert_eq!(
            identifier_tokens("process a player's very first game login"),
            vec!["process", "player", "first", "game", "login"]
        );

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
                    retrieval_text: None,
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
                relevance: 1.0,
            },
            SearchHit {
                drawer: Drawer {
                    id: "code".to_owned(),
                    content:
                        "XtRequest::FirstJoin(_) => { let first_join_request = FirstJoinRequest { socket_id: pid }; }"
                            .to_owned(),
                    retrieval_text: None,
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
                relevance: 0.95,
            },
        ];

        rerank_search_hits(&mut hits, &query);
        assert_eq!(hits[0].drawer.id, "code");
    }
}
