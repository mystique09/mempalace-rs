use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
    types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::{
    DistanceType,
    index::{
        Index,
        scalar::{FtsIndexBuilder, FullTextSearchQuery},
        vector::IvfFlatIndexBuilder,
    },
    query::{ExecutableQuery, QueryBase, QueryExecutionOptions, Select},
};
use model2vec_rs::model::StaticModel;

use mempalace_core::{
    Drawer, DrawerMetadata, MemoryStore, MempalaceError, Result, RoomStatus, SearchHit,
    SearchQuery, StoreStatus,
};

const DEFAULT_MODEL_REPO: &str = "minishlab/potion-base-32M";
const RESULT_COLUMNS: &[&str] = &[
    "id",
    "content",
    "wing",
    "room",
    "source_file",
    "chunk_index",
    "added_by",
    "filed_at",
];
const HYBRID_SCORE_COLUMN: &str = "_relevance_score";
const VECTOR_DISTANCE_COLUMN: &str = "_distance";
const FTS_SCORE_COLUMN: &str = "_score";

#[derive(Clone)]
pub struct LanceMemoryStore {
    palace_path: PathBuf,
    table_name: String,
    embedder: Arc<EmbeddingBackend>,
    connection: Arc<Mutex<Option<lancedb::Connection>>>,
    table: Arc<Mutex<Option<lancedb::Table>>>,
    indices_ready: Arc<Mutex<bool>>,
}

impl LanceMemoryStore {
    pub fn new(
        palace_path: impl Into<PathBuf>,
        table_name: impl Into<String>,
        model_dir: impl Into<PathBuf>,
    ) -> Result<Self> {
        let model_dir = model_dir.into();
        let model_path = model_dir.join("potion-base-32M");
        let repo_or_path = if model_path.join("config.json").exists() {
            model_path.to_string_lossy().to_string()
        } else {
            DEFAULT_MODEL_REPO.to_owned()
        };

        let model = StaticModel::from_pretrained(&repo_or_path, None, None, None)
            .map_err(|err| MempalaceError::Embedding(err.to_string()))?;
        let dim = model
            .encode(&["dim".to_string()])
            .first()
            .map(|vec| vec.len())
            .unwrap_or(384);

        Ok(Self {
            palace_path: palace_path.into(),
            table_name: table_name.into(),
            embedder: Arc::new(EmbeddingBackend::Model2Vec {
                dim,
                inner: Box::new(Mutex::new(Some(model))),
            }),
            connection: Arc::new(Mutex::new(None)),
            table: Arc::new(Mutex::new(None)),
            indices_ready: Arc::new(Mutex::new(false)),
        })
    }

    #[cfg(test)]
    fn new_for_tests(palace_path: impl Into<PathBuf>, table_name: impl Into<String>) -> Self {
        Self {
            palace_path: palace_path.into(),
            table_name: table_name.into(),
            embedder: Arc::new(EmbeddingBackend::Deterministic { dim: 64 }),
            connection: Arc::new(Mutex::new(None)),
            table: Arc::new(Mutex::new(None)),
            indices_ready: Arc::new(Mutex::new(false)),
        }
    }

    pub fn palace_path(&self) -> &Path {
        &self.palace_path
    }

    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    pub async fn has_source_file(&self, source_file: &str) -> Result<bool> {
        let Some(table) = self.open_table().await? else {
            return Ok(false);
        };

        let batches = table
            .query()
            .only_if(format!("source_file = '{}'", sql_escape(source_file)))
            .limit(1)
            .select(Select::columns(&["id"]))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        Ok(batches.iter().any(|batch| batch.num_rows() > 0))
    }

    pub async fn delete_source_file(&self, source_file: &str) -> Result<usize> {
        let Some(table) = self.open_table().await? else {
            return Ok(0);
        };

        let deleted = table
            .delete(&format!("source_file = '{}'", sql_escape(source_file)))
            .await?;
        Ok(deleted.num_deleted_rows as usize)
    }

    pub async fn remine_all(&self, wing: Option<&str>) -> Result<usize> {
        let drawers = self.list_drawers(wing).await?;
        let total = drawers.len();
        if total == 0 {
            return Ok(0);
        }

        let conn = self.connect().await?;
        if conn.open_table(&self.table_name).execute().await.is_ok() {
            conn.drop_table(&self.table_name, &[]).await?;
            *self
                .table
                .lock()
                .map_err(|_| MempalaceError::LockPoisoned("lancedb_table"))? = None;
            *self
                .indices_ready
                .lock()
                .map_err(|_| MempalaceError::LockPoisoned("lancedb_indices_ready"))? = false;
        }

        self.add_drawers(drawers).await?;
        Ok(total)
    }

    pub async fn room_counts(&self) -> Result<Vec<RoomStatus>> {
        let Some(table) = self.open_table().await? else {
            return Ok(Vec::new());
        };

        let batches = table
            .query()
            .select(Select::columns(&["wing", "room"]))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut counts: BTreeMap<(String, String), usize> = BTreeMap::new();
        for batch in batches {
            for row in 0..batch.num_rows() {
                let wing = string_value(&batch, "wing", row)?;
                let room = string_value(&batch, "room", row)?;
                *counts.entry((wing, room)).or_insert(0) += 1;
            }
        }

        Ok(counts
            .into_iter()
            .map(|((wing, room), total_drawers)| RoomStatus {
                wing,
                room,
                total_drawers,
            })
            .collect())
    }

    async fn connect(&self) -> Result<lancedb::Connection> {
        if let Some(conn) = self
            .connection
            .lock()
            .map_err(|_| MempalaceError::LockPoisoned("lancedb_connection"))?
            .clone()
        {
            return Ok(conn);
        }

        let conn = lancedb::connect(self.palace_path.to_string_lossy().as_ref())
            .execute()
            .await?;
        *self
            .connection
            .lock()
            .map_err(|_| MempalaceError::LockPoisoned("lancedb_connection"))? = Some(conn.clone());
        Ok(conn)
    }

    async fn open_table(&self) -> Result<Option<lancedb::Table>> {
        if let Some(table) = self
            .table
            .lock()
            .map_err(|_| MempalaceError::LockPoisoned("lancedb_table"))?
            .clone()
        {
            return Ok(Some(table));
        }

        let conn = self.connect().await?;
        match conn.open_table(&self.table_name).execute().await {
            Ok(table) => {
                *self
                    .table
                    .lock()
                    .map_err(|_| MempalaceError::LockPoisoned("lancedb_table"))? =
                    Some(table.clone());
                Ok(Some(table))
            }
            Err(lancedb::Error::TableNotFound { .. }) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    async fn ensure_table(&self) -> Result<lancedb::Table> {
        if let Some(table) = self.open_table().await? {
            return Ok(table);
        }

        let conn = self.connect().await?;
        let table = conn
            .create_empty_table(&self.table_name, self.schema())
            .execute()
            .await?;
        *self
            .table
            .lock()
            .map_err(|_| MempalaceError::LockPoisoned("lancedb_table"))? = Some(table.clone());
        Ok(table)
    }

    async fn ensure_indices(&self, table: &lancedb::Table) -> Result<()> {
        if *self
            .indices_ready
            .lock()
            .map_err(|_| MempalaceError::LockPoisoned("lancedb_indices_ready"))?
        {
            return Ok(());
        }

        let existing = table.list_indices().await?;
        let has_column_index = |column: &str| {
            existing
                .iter()
                .any(|index| index.columns.iter().any(|indexed| indexed == column))
        };

        if !has_column_index("content") {
            table
                .create_index(&["content"], Index::FTS(FtsIndexBuilder::default()))
                .execute()
                .await?;
        }

        if !has_column_index("vector") {
            table
                .create_index(
                    &["vector"],
                    Index::IvfFlat(
                        IvfFlatIndexBuilder::default().distance_type(DistanceType::Cosine),
                    ),
                )
                .execute()
                .await?;
        }

        for column in ["wing", "room", "source_file"] {
            if !has_column_index(column) {
                table.create_index(&[column], Index::Auto).execute().await?;
            }
        }

        *self
            .indices_ready
            .lock()
            .map_err(|_| MempalaceError::LockPoisoned("lancedb_indices_ready"))? = true;
        Ok(())
    }

    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("wing", DataType::Utf8, false),
            Field::new("room", DataType::Utf8, false),
            Field::new("source_file", DataType::Utf8, true),
            Field::new("chunk_index", DataType::Int64, false),
            Field::new("added_by", DataType::Utf8, false),
            Field::new("filed_at", DataType::Utf8, true),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.embedder.dimension() as i32,
                ),
                false,
            ),
        ]))
    }

    fn drawers_batch(&self, drawers: &[Drawer], vectors: Vec<Vec<f32>>) -> Result<RecordBatch> {
        if drawers.len() != vectors.len() {
            return Err(MempalaceError::Embedding(
                "drawer/vector batch length mismatch".to_owned(),
            ));
        }

        let schema = self.schema();
        let vector_values = vectors
            .into_iter()
            .map(|vector| Some(vector.into_iter().map(Some).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vector_values,
            self.embedder.dimension() as i32,
        );

        Ok(RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(
                    drawers
                        .iter()
                        .map(|drawer| drawer.id.clone())
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
                Arc::new(StringArray::from(
                    drawers
                        .iter()
                        .map(|drawer| drawer.content.clone())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    drawers
                        .iter()
                        .map(|drawer| drawer.metadata.wing.clone())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    drawers
                        .iter()
                        .map(|drawer| drawer.metadata.room.clone())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    drawers
                        .iter()
                        .map(|drawer| drawer.metadata.source_file.clone())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(
                    drawers
                        .iter()
                        .map(|drawer| drawer.metadata.chunk_index)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    drawers
                        .iter()
                        .map(|drawer| drawer.metadata.added_by.clone())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    drawers
                        .iter()
                        .map(|drawer| drawer.metadata.filed_at.clone())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(vector_array),
            ],
        )?)
    }

    fn read_drawers(&self, batches: Vec<RecordBatch>) -> Result<Vec<Drawer>> {
        let mut drawers = Vec::new();

        for batch in batches {
            for row in 0..batch.num_rows() {
                drawers.push(Drawer {
                    id: string_value(&batch, "id", row)?,
                    content: string_value(&batch, "content", row)?,
                    metadata: DrawerMetadata {
                        wing: string_value(&batch, "wing", row)?,
                        room: string_value(&batch, "room", row)?,
                        source_file: optional_string_value(&batch, "source_file", row)?,
                        chunk_index: int64_value(&batch, "chunk_index", row)?,
                        added_by: string_value(&batch, "added_by", row)?,
                        filed_at: optional_string_value(&batch, "filed_at", row)?,
                    },
                });
            }
        }

        Ok(drawers)
    }

    fn filter_clause(drawer_id: &str) -> String {
        format!("id = '{}'", sql_escape(drawer_id))
    }

    fn wing_filter(wing: Option<&str>) -> Option<String> {
        wing.map(|wing| format!("wing = '{}'", sql_escape(wing)))
    }

    fn search_filter(query: &SearchQuery) -> Option<String> {
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

    fn supports_full_text_search(query: &str) -> bool {
        query.chars().any(char::is_alphanumeric)
    }

    fn retrieval_limit(limit: usize) -> usize {
        (limit.saturating_mul(8)).clamp(limit, 64)
    }

    async fn vector_batches(
        &self,
        table: &lancedb::Table,
        query: &SearchQuery,
        vector: Vec<f32>,
    ) -> Result<Vec<RecordBatch>> {
        let mut columns = RESULT_COLUMNS.to_vec();
        columns.push(VECTOR_DISTANCE_COLUMN);

        let mut builder = table
            .query()
            .nearest_to(vector)?
            .distance_type(DistanceType::Cosine)
            .limit(query.limit)
            .select(Select::columns(&columns));

        if let Some(filter) = Self::search_filter(query) {
            builder = builder.only_if(filter);
        }

        builder
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await
            .map_err(Into::into)
    }

    async fn hybrid_batches(
        &self,
        table: &lancedb::Table,
        query: &SearchQuery,
        vector: Vec<f32>,
    ) -> Result<Vec<RecordBatch>> {
        let mut builder = table
            .query()
            .full_text_search(FullTextSearchQuery::new(query.query.clone()))
            .nearest_to(vector)?
            .distance_type(DistanceType::Cosine)
            .limit(query.limit);

        if let Some(filter) = Self::search_filter(query) {
            builder = builder.only_if(filter);
        }

        builder
            .execute_hybrid(QueryExecutionOptions::default())
            .await?
            .try_collect::<Vec<_>>()
            .await
            .map_err(Into::into)
    }

    fn read_scores(&self, batches: &[RecordBatch]) -> Result<Vec<f32>> {
        let mut scores = Vec::new();
        let mut normalize = false;

        for batch in batches {
            if let Some(column) = batch.column_by_name(HYBRID_SCORE_COLUMN) {
                let values = column
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        MempalaceError::Embedding("hybrid score column type mismatch".to_owned())
                    })?;
                scores.extend((0..batch.num_rows()).map(|row| values.value(row)));
                normalize = true;
                continue;
            }

            if let Some(column) = batch.column_by_name(VECTOR_DISTANCE_COLUMN) {
                let values = column
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        MempalaceError::Embedding("distance column type mismatch".to_owned())
                    })?;
                scores.extend((0..batch.num_rows()).map(|row| 1.0 - values.value(row)));
                continue;
            }

            if let Some(column) = batch.column_by_name(FTS_SCORE_COLUMN) {
                let values = column
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        MempalaceError::Embedding("fts score column type mismatch".to_owned())
                    })?;
                scores.extend((0..batch.num_rows()).map(|row| values.value(row)));
                normalize = true;
                continue;
            }

            scores.extend((0..batch.num_rows()).map(|_| 0.0));
        }

        if normalize {
            let max_score = scores.iter().copied().fold(0.0_f32, f32::max);
            if max_score > 0.0 {
                for score in &mut scores {
                    *score /= max_score;
                }
            }
        }

        Ok(scores)
    }
}

#[async_trait]
impl MemoryStore for LanceMemoryStore {
    async fn add_drawer(&self, drawer: Drawer) -> Result<()> {
        self.add_drawers(vec![drawer]).await
    }

    async fn add_drawers(&self, drawers: Vec<Drawer>) -> Result<()> {
        if drawers.is_empty() {
            return Ok(());
        }

        let texts = drawers
            .iter()
            .map(|drawer| drawer.content.clone())
            .collect::<Vec<_>>();
        let vectors = self.embedder.embed_batch(&texts)?;
        let batch = self.drawers_batch(&drawers, vectors)?;
        let table = self.ensure_table().await?;
        table.add(batch).execute().await?;
        self.ensure_indices(&table).await?;
        Ok(())
    }

    async fn get_drawer(&self, drawer_id: &str) -> Result<Option<Drawer>> {
        let Some(table) = self.open_table().await? else {
            return Ok(None);
        };

        let batches = table
            .query()
            .only_if(Self::filter_clause(drawer_id))
            .limit(1)
            .select(Select::columns(&[
                "id",
                "content",
                "wing",
                "room",
                "source_file",
                "chunk_index",
                "added_by",
                "filed_at",
            ]))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        Ok(self.read_drawers(batches)?.into_iter().next())
    }

    async fn delete_drawer(&self, drawer_id: &str) -> Result<bool> {
        let Some(table) = self.open_table().await? else {
            return Ok(false);
        };

        let deleted = table.delete(&Self::filter_clause(drawer_id)).await?;
        Ok(deleted.num_deleted_rows > 0)
    }

    async fn delete_source_file(&self, source_file: &str) -> Result<usize> {
        LanceMemoryStore::delete_source_file(self, source_file).await
    }

    async fn list_drawers(&self, wing: Option<&str>) -> Result<Vec<Drawer>> {
        let Some(table) = self.open_table().await? else {
            return Ok(Vec::new());
        };

        let mut builder = table.query().select(Select::columns(&[
            "id",
            "content",
            "wing",
            "room",
            "source_file",
            "chunk_index",
            "added_by",
            "filed_at",
        ]));

        if let Some(filter) = Self::wing_filter(wing) {
            builder = builder.only_if(filter);
        }

        let batches = builder.execute().await?.try_collect::<Vec<_>>().await?;
        self.read_drawers(batches)
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchHit>> {
        if query.limit == 0 {
            return Ok(Vec::new());
        }

        let Some(table) = self.open_table().await? else {
            return Ok(Vec::new());
        };

        self.ensure_indices(&table).await?;
        let retrieval_query = SearchQuery {
            limit: Self::retrieval_limit(query.limit),
            ..query.clone()
        };
        let vector = self.embedder.embed(&query.query)?;
        let batches = if Self::supports_full_text_search(&query.query) {
            match self
                .hybrid_batches(&table, &retrieval_query, vector.clone())
                .await
            {
                Ok(batches) => batches,
                Err(_) => {
                    self.vector_batches(&table, &retrieval_query, vector)
                        .await?
                }
            }
        } else {
            self.vector_batches(&table, &retrieval_query, vector)
                .await?
        };
        let drawers = self.read_drawers(batches.clone())?;
        let scores = self.read_scores(&batches)?;
        let mut hits = drawers
            .into_iter()
            .enumerate()
            .map(|(index, drawer)| SearchHit {
                drawer,
                score: scores.get(index).copied().unwrap_or(0.0),
            })
            .collect::<Vec<_>>();
        rerank_search_hits(&mut hits, &query);
        hits.truncate(query.limit);
        Ok(hits)
    }

    async fn status(&self) -> Result<StoreStatus> {
        let Some(table) = self.open_table().await? else {
            return Ok(StoreStatus::default());
        };

        Ok(StoreStatus {
            total_drawers: table.count_rows(None).await?,
        })
    }

    async fn has_source_file(&self, source_file: &str) -> Result<bool> {
        LanceMemoryStore::has_source_file(self, source_file).await
    }

    async fn source_files(&self) -> Result<HashSet<String>> {
        let Some(table) = self.open_table().await? else {
            return Ok(HashSet::new());
        };

        let batches = table
            .query()
            .select(Select::columns(&["source_file"]))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut files = HashSet::new();
        for batch in batches {
            let array = batch
                .column_by_name("source_file")
                .ok_or_else(|| MempalaceError::Embedding("missing column: source_file".to_owned()))?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    MempalaceError::Embedding("column type mismatch: source_file".to_owned())
                })?;

            for row in 0..batch.num_rows() {
                if !array.is_null(row) {
                    files.insert(array.value(row).to_owned());
                }
            }
        }

        Ok(files)
    }

    async fn room_counts(&self) -> Result<Vec<RoomStatus>> {
        LanceMemoryStore::room_counts(self).await
    }
}

enum EmbeddingBackend {
    Model2Vec {
        dim: usize,
        inner: Box<Mutex<Option<StaticModel>>>,
    },
    #[cfg(test)]
    Deterministic { dim: usize },
}

impl EmbeddingBackend {
    fn dimension(&self) -> usize {
        match self {
            Self::Model2Vec { dim, .. } => *dim,
            #[cfg(test)]
            Self::Deterministic { dim } => *dim,
        }
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let texts = [text.to_owned()];
        let mut embeddings = self.embed_batch(&texts)?;
        embeddings
            .pop()
            .ok_or_else(|| MempalaceError::Embedding("no embedding returned".to_owned()))
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Model2Vec { inner, .. } => {
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

fn sql_escape(value: &str) -> String {
    value.replace('\'', "''")
}

fn string_value(batch: &RecordBatch, column: &str, row: usize) -> Result<String> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| MempalaceError::Embedding(format!("missing column: {column}")))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| MempalaceError::Embedding(format!("column type mismatch: {column}")))?;
    Ok(array.value(row).to_owned())
}

fn optional_string_value(batch: &RecordBatch, column: &str, row: usize) -> Result<Option<String>> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| MempalaceError::Embedding(format!("missing column: {column}")))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| MempalaceError::Embedding(format!("column type mismatch: {column}")))?;

    if array.is_null(row) {
        Ok(None)
    } else {
        Ok(Some(array.value(row).to_owned()))
    }
}

fn int64_value(batch: &RecordBatch, column: &str, row: usize) -> Result<i64> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| MempalaceError::Embedding(format!("missing column: {column}")))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| MempalaceError::Embedding(format!("column type mismatch: {column}")))?;
    Ok(array.value(row))
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

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::{
        Drawer, DrawerMetadata, HYBRID_SCORE_COLUMN, LanceMemoryStore, MemoryStore, SearchHit,
        SearchQuery, identifier_tokens, identifier_variants, rerank_search_hits,
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
        let store = LanceMemoryStore::new_for_tests(tmp.path(), "drawers");

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
        let store = LanceMemoryStore::new_for_tests(tmp.path(), "drawers");

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
        let store = LanceMemoryStore::new_for_tests(tmp.path(), "drawers");

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
        let store = LanceMemoryStore::new_for_tests(tmp.path(), "drawers");

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
    async fn add_drawers_creates_search_indices() {
        let tmp = tempdir().unwrap();
        let store = LanceMemoryStore::new_for_tests(tmp.path(), "drawers");

        store
            .add_drawer(drawer(
                "drawer_1",
                "AAAK compression dialect",
                "project",
                "docs",
            ))
            .await
            .unwrap();

        let table = store.ensure_table().await.unwrap();
        let indices = table.list_indices().await.unwrap();

        assert!(
            indices
                .iter()
                .any(|index| index.columns == vec!["content".to_owned()])
        );
        assert!(
            indices
                .iter()
                .any(|index| index.columns == vec!["vector".to_owned()])
        );
        assert!(
            indices
                .iter()
                .any(|index| index.columns == vec!["wing".to_owned()])
        );
        assert!(
            indices
                .iter()
                .any(|index| index.columns == vec!["room".to_owned()])
        );
        assert!(
            indices
                .iter()
                .any(|index| index.columns == vec!["source_file".to_owned()])
        );
    }

    #[tokio::test]
    async fn hybrid_search_executes_without_fallback() {
        let tmp = tempdir().unwrap();
        let store = LanceMemoryStore::new_for_tests(tmp.path(), "drawers");

        store
            .add_drawer(drawer(
                "drawer_1",
                "AAAK is the compressed memory dialect used by MemPalace",
                "project",
                "docs",
            ))
            .await
            .unwrap();

        let table = store.ensure_table().await.unwrap();
        store.ensure_indices(&table).await.unwrap();

        let mut query = SearchQuery::new("aaak");
        query.limit = 5;
        let vector = store.embedder.embed(&query.query).unwrap();
        let batches = store.hybrid_batches(&table, &query, vector).await.unwrap();
        let scores = store.read_scores(&batches).unwrap();

        assert!(!batches.is_empty());
        assert!(
            batches
                .iter()
                .any(|batch| batch.column_by_name(HYBRID_SCORE_COLUMN).is_some())
        );
        assert!(scores.iter().any(|score| *score > 0.0));
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
}
