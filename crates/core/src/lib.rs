pub mod aaak;
pub mod config;
pub mod entity_detector;
pub mod error;
pub mod knowledge_graph;
pub mod memory;
pub mod project_miner;

pub use aaak::{
    AaakCompressionStats, AaakDecoded, AaakDialect, AaakFile, AaakHeader, AaakTunnel, AaakZettel,
};
pub use config::{DEFAULT_COLLECTION_NAME, DEFAULT_PALACE_PATH_SUFFIX, MempalaceConfig};
pub use entity_detector::{
    DetectedEntities, DetectedEntity, DetectedEntityKind, detect_entities, scan_for_detection,
};
pub use error::{MempalaceError, Result};
pub use knowledge_graph::{
    EntityRecord, FactRecord, KnowledgeGraph, KnowledgeGraphStats, RelationshipRecord,
};
pub use memory::{
    ContentKind, Drawer, DrawerMetadata, MemoryStore, RoomStatus, SearchHit, SearchQuery,
    StoreStatus,
};
pub use project_miner::{
    MineOptions, MineSummary, RetrievalContext, mine_project, retrieval_text_for_content,
    scan_project,
};
