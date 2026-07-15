use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MempalaceError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("regex error: {0}")]
    Regex(#[from] regex::Error),

    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("home directory not found")]
    MissingHomeDirectory,

    #[error("not implemented yet: {0}")]
    NotImplemented(&'static str),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("agent-session sync error: {0}")]
    AgentSessionSync(String),

    #[error("lock poisoned: {0}")]
    LockPoisoned(&'static str),

    #[error("path does not have a parent: {0}")]
    MissingParent(PathBuf),
}

pub type Result<T> = std::result::Result<T, MempalaceError>;
