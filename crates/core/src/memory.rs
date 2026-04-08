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
    pub metadata: DrawerMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchHit {
    pub drawer: Drawer,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchQuery {
    pub query: String,
    pub limit: usize,
    pub wing: Option<String>,
    pub room: Option<String>,
}

impl SearchQuery {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            limit: 5,
            wing: None,
            room: None,
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
