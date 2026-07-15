use std::collections::HashSet;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DrawerMetadata {
    pub wing: String,
    pub room: String,
    pub source_file: Option<String>,
    pub chunk_index: i64,
    pub added_by: String,
    pub filed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Drawer {
    pub id: String,
    pub content: String,
    /// Optional enriched representation used only for embedding and retrieval.
    /// User-facing APIs must continue to return `content` as the verbatim source.
    #[serde(default, skip_serializing)]
    pub retrieval_text: Option<String>,
    pub metadata: DrawerMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchHit {
    pub drawer: Drawer,
    /// Exact cosine similarity used by duplicate checks and score thresholds.
    pub score: f32,
    /// Normalized hybrid ranking relevance used to order user-facing results.
    #[serde(default)]
    pub relevance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchQuery {
    pub query: String,
    pub limit: usize,
    pub wing: Option<String>,
    pub room: Option<String>,
    pub min_score: Option<f32>,
}

impl SearchQuery {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            limit: 5,
            wing: None,
            room: None,
            min_score: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoreStatus {
    pub total_drawers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoomStatus {
    pub wing: String,
    pub room: String,
    pub total_drawers: usize,
}

#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn add_drawer(&self, drawer: Drawer) -> Result<()>;
    async fn add_drawers(&self, drawers: Vec<Drawer>) -> Result<()> {
        for drawer in drawers {
            self.add_drawer(drawer).await?;
        }
        Ok(())
    }
    async fn get_drawer(&self, drawer_id: &str) -> Result<Option<Drawer>>;
    async fn delete_drawer(&self, drawer_id: &str) -> Result<bool>;
    async fn delete_source_file(&self, source_file: &str) -> Result<usize> {
        let mut deleted = 0usize;
        for drawer in self.list_drawers(None).await? {
            if drawer.metadata.source_file.as_deref() == Some(source_file)
                && self.delete_drawer(&drawer.id).await?
            {
                deleted += 1;
            }
        }
        Ok(deleted)
    }
    async fn list_drawers(&self, wing: Option<&str>) -> Result<Vec<Drawer>>;
    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchHit>>;
    async fn status(&self) -> Result<StoreStatus>;
    async fn has_source_file(&self, source_file: &str) -> Result<bool>;
    async fn source_files(&self) -> Result<HashSet<String>> {
        let mut files = HashSet::new();
        for drawer in self.list_drawers(None).await? {
            if let Some(source_file) = drawer.metadata.source_file {
                files.insert(source_file);
            }
        }
        Ok(files)
    }
    async fn room_counts(&self) -> Result<Vec<RoomStatus>>;
}
