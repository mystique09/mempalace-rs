use std::{
    fs,
    path::{Path, PathBuf},
};

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{MempalaceError, Result};

pub const DEFAULT_KG_FILENAME: &str = "knowledge_graph.sqlite3";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EntityRecord {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub properties_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RelationshipRecord {
    pub direction: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub confidence: f64,
    pub source_closet: Option<String>,
    pub current: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FactRecord {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub current: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KnowledgeGraphStats {
    pub entities: usize,
    pub triples: usize,
    pub current_facts: usize,
    pub expired_facts: usize,
    pub relationship_types: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct KnowledgeGraph {
    db_path: PathBuf,
}

impl KnowledgeGraph {
    pub fn new(db_path: impl Into<PathBuf>) -> Result<Self> {
        let db_path = db_path.into();
        let Some(parent) = db_path.parent() else {
            return Err(MempalaceError::MissingParent(db_path));
        };

        fs::create_dir_all(parent)?;

        let graph = Self { db_path };
        graph.init_db()?;
        Ok(graph)
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn add_entity(
        &self,
        name: &str,
        entity_type: &str,
        properties_json: Option<&str>,
    ) -> Result<String> {
        let entity_id = Self::entity_id(name);
        let properties_json = properties_json.unwrap_or("{}");
        let conn = self.connection()?;

        conn.execute(
            "INSERT OR REPLACE INTO entities (id, name, type, properties) VALUES (?1, ?2, ?3, ?4)",
            params![entity_id, name, entity_type, properties_json],
        )?;

        Ok(entity_id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        valid_from: Option<&str>,
        valid_to: Option<&str>,
        confidence: f64,
        source_closet: Option<&str>,
        source_file: Option<&str>,
    ) -> Result<String> {
        let subject_id = Self::entity_id(subject);
        let object_id = Self::entity_id(object);
        let predicate = Self::normalize_predicate(predicate);

        let conn = self.connection()?;
        conn.execute(
            "INSERT OR IGNORE INTO entities (id, name) VALUES (?1, ?2)",
            params![subject_id, subject],
        )?;
        conn.execute(
            "INSERT OR IGNORE INTO entities (id, name) VALUES (?1, ?2)",
            params![object_id, object],
        )?;

        let existing = conn
            .query_row(
                "SELECT id FROM triples WHERE subject = ?1 AND predicate = ?2 AND object = ?3 AND valid_to IS NULL",
                params![subject_id, predicate, object_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;

        if let Some(existing_id) = existing {
            return Ok(existing_id);
        }

        let triple_id = format!(
            "t_{}_{}_{}_{}",
            subject_id,
            predicate,
            object_id,
            Uuid::now_v7().simple()
        );

        conn.execute(
            "INSERT INTO triples (id, subject, predicate, object, valid_from, valid_to, confidence, source_closet, source_file) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                triple_id,
                subject_id,
                predicate,
                object_id,
                valid_from,
                valid_to,
                confidence,
                source_closet,
                source_file
            ],
        )?;

        Ok(triple_id)
    }

    pub fn invalidate(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        ended: Option<&str>,
    ) -> Result<()> {
        let ended = ended
            .map(str::to_owned)
            .unwrap_or_else(|| Utc::now().date_naive().to_string());

        let conn = self.connection()?;
        conn.execute(
            "UPDATE triples SET valid_to = ?1 WHERE subject = ?2 AND predicate = ?3 AND object = ?4 AND valid_to IS NULL",
            params![
                ended,
                Self::entity_id(subject),
                Self::normalize_predicate(predicate),
                Self::entity_id(object)
            ],
        )?;

        Ok(())
    }

    pub fn query_entity(
        &self,
        name: &str,
        as_of: Option<&str>,
        direction: &str,
    ) -> Result<Vec<RelationshipRecord>> {
        let entity_id = Self::entity_id(name);
        let conn = self.connection()?;
        let mut results = Vec::new();

        if matches!(direction, "outgoing" | "both") {
            let base = "SELECT t.predicate, o.name, t.valid_from, t.valid_to, t.confidence, t.source_closet FROM triples t JOIN entities o ON t.object = o.id WHERE t.subject = ?1";
            let sql = if as_of.is_some() {
                format!(
                    "{base} AND (t.valid_from IS NULL OR t.valid_from <= ?2) AND (t.valid_to IS NULL OR t.valid_to >= ?2)"
                )
            } else {
                base.to_owned()
            };

            let mut stmt = conn.prepare(&sql)?;
            if let Some(as_of) = as_of {
                let mut rows = stmt.query(params![entity_id, as_of])?;
                while let Some(row) = rows.next()? {
                    let valid_to: Option<String> = row.get(3)?;
                    results.push(RelationshipRecord {
                        direction: "outgoing".to_owned(),
                        subject: name.to_owned(),
                        predicate: row.get(0)?,
                        object: row.get(1)?,
                        valid_from: row.get(2)?,
                        valid_to: valid_to.clone(),
                        confidence: row.get(4)?,
                        source_closet: row.get(5)?,
                        current: valid_to.is_none(),
                    });
                }
            } else {
                let mut rows = stmt.query(params![entity_id])?;
                while let Some(row) = rows.next()? {
                    let valid_to: Option<String> = row.get(3)?;
                    results.push(RelationshipRecord {
                        direction: "outgoing".to_owned(),
                        subject: name.to_owned(),
                        predicate: row.get(0)?,
                        object: row.get(1)?,
                        valid_from: row.get(2)?,
                        valid_to: valid_to.clone(),
                        confidence: row.get(4)?,
                        source_closet: row.get(5)?,
                        current: valid_to.is_none(),
                    });
                }
            }
        }

        if matches!(direction, "incoming" | "both") {
            let base = "SELECT t.predicate, s.name, t.valid_from, t.valid_to, t.confidence, t.source_closet FROM triples t JOIN entities s ON t.subject = s.id WHERE t.object = ?1";
            let sql = if as_of.is_some() {
                format!(
                    "{base} AND (t.valid_from IS NULL OR t.valid_from <= ?2) AND (t.valid_to IS NULL OR t.valid_to >= ?2)"
                )
            } else {
                base.to_owned()
            };

            let mut stmt = conn.prepare(&sql)?;
            if let Some(as_of) = as_of {
                let mut rows = stmt.query(params![entity_id, as_of])?;
                while let Some(row) = rows.next()? {
                    let valid_to: Option<String> = row.get(3)?;
                    results.push(RelationshipRecord {
                        direction: "incoming".to_owned(),
                        subject: row.get(1)?,
                        predicate: row.get(0)?,
                        object: name.to_owned(),
                        valid_from: row.get(2)?,
                        valid_to: valid_to.clone(),
                        confidence: row.get(4)?,
                        source_closet: row.get(5)?,
                        current: valid_to.is_none(),
                    });
                }
            } else {
                let mut rows = stmt.query(params![entity_id])?;
                while let Some(row) = rows.next()? {
                    let valid_to: Option<String> = row.get(3)?;
                    results.push(RelationshipRecord {
                        direction: "incoming".to_owned(),
                        subject: row.get(1)?,
                        predicate: row.get(0)?,
                        object: name.to_owned(),
                        valid_from: row.get(2)?,
                        valid_to: valid_to.clone(),
                        confidence: row.get(4)?,
                        source_closet: row.get(5)?,
                        current: valid_to.is_none(),
                    });
                }
            }
        }

        Ok(results)
    }

    pub fn query_relationship(
        &self,
        predicate: &str,
        as_of: Option<&str>,
    ) -> Result<Vec<FactRecord>> {
        let conn = self.connection()?;
        let predicate = Self::normalize_predicate(predicate);
        let base = "SELECT s.name, o.name, t.valid_from, t.valid_to FROM triples t JOIN entities s ON t.subject = s.id JOIN entities o ON t.object = o.id WHERE t.predicate = ?1";
        let sql = if as_of.is_some() {
            format!(
                "{base} AND (t.valid_from IS NULL OR t.valid_from <= ?2) AND (t.valid_to IS NULL OR t.valid_to >= ?2)"
            )
        } else {
            base.to_owned()
        };

        let mut facts = Vec::new();
        let mut stmt = conn.prepare(&sql)?;
        if let Some(as_of) = as_of {
            let mut rows = stmt.query(params![predicate, as_of])?;
            while let Some(row) = rows.next()? {
                let valid_to: Option<String> = row.get(3)?;
                facts.push(FactRecord {
                    subject: row.get(0)?,
                    predicate: predicate.clone(),
                    object: row.get(1)?,
                    valid_from: row.get(2)?,
                    valid_to: valid_to.clone(),
                    current: valid_to.is_none(),
                });
            }
        } else {
            let mut rows = stmt.query(params![predicate])?;
            while let Some(row) = rows.next()? {
                let valid_to: Option<String> = row.get(3)?;
                facts.push(FactRecord {
                    subject: row.get(0)?,
                    predicate: predicate.clone(),
                    object: row.get(1)?,
                    valid_from: row.get(2)?,
                    valid_to: valid_to.clone(),
                    current: valid_to.is_none(),
                });
            }
        }

        Ok(facts)
    }

    pub fn timeline(&self, entity_name: Option<&str>) -> Result<Vec<FactRecord>> {
        let conn = self.connection()?;
        let (sql, params_vec): (String, Vec<String>) = if let Some(entity_name) = entity_name {
            let entity_id = Self::entity_id(entity_name);
            (
                "SELECT s.name, t.predicate, o.name, t.valid_from, t.valid_to FROM triples t JOIN entities s ON t.subject = s.id JOIN entities o ON t.object = o.id WHERE t.subject = ?1 OR t.object = ?1 ORDER BY t.valid_from IS NULL, t.valid_from ASC LIMIT 100".to_owned(),
                vec![entity_id],
            )
        } else {
            (
                "SELECT s.name, t.predicate, o.name, t.valid_from, t.valid_to FROM triples t JOIN entities s ON t.subject = s.id JOIN entities o ON t.object = o.id ORDER BY t.valid_from IS NULL, t.valid_from ASC LIMIT 100".to_owned(),
                Vec::new(),
            )
        };

        let mut facts = Vec::new();
        let mut stmt = conn.prepare(&sql)?;
        if params_vec.is_empty() {
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let valid_to: Option<String> = row.get(4)?;
                facts.push(FactRecord {
                    subject: row.get(0)?,
                    predicate: row.get(1)?,
                    object: row.get(2)?,
                    valid_from: row.get(3)?,
                    valid_to: valid_to.clone(),
                    current: valid_to.is_none(),
                });
            }
        } else {
            let mut rows = stmt.query(params![params_vec[0]])?;
            while let Some(row) = rows.next()? {
                let valid_to: Option<String> = row.get(4)?;
                facts.push(FactRecord {
                    subject: row.get(0)?,
                    predicate: row.get(1)?,
                    object: row.get(2)?,
                    valid_from: row.get(3)?,
                    valid_to: valid_to.clone(),
                    current: valid_to.is_none(),
                });
            }
        }

        Ok(facts)
    }

    pub fn stats(&self) -> Result<KnowledgeGraphStats> {
        let conn = self.connection()?;

        let entities = conn.query_row("SELECT COUNT(*) FROM entities", [], |row| {
            row.get::<_, i64>(0)
        })? as usize;
        let triples = conn.query_row("SELECT COUNT(*) FROM triples", [], |row| {
            row.get::<_, i64>(0)
        })? as usize;
        let current_facts = conn.query_row(
            "SELECT COUNT(*) FROM triples WHERE valid_to IS NULL",
            [],
            |row| row.get::<_, i64>(0),
        )? as usize;

        let relationship_types = {
            let mut stmt =
                conn.prepare("SELECT DISTINCT predicate FROM triples ORDER BY predicate")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            let mut predicates = Vec::new();
            for row in rows {
                predicates.push(row?);
            }
            predicates
        };

        Ok(KnowledgeGraphStats {
            entities,
            triples,
            current_facts,
            expired_facts: triples.saturating_sub(current_facts),
            relationship_types,
        })
    }

    fn init_db(&self) -> Result<()> {
        let conn = self.connection()?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT DEFAULT 'unknown',
                properties TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS triples (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                confidence REAL DEFAULT 1.0,
                source_closet TEXT,
                source_file TEXT,
                extracted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject) REFERENCES entities(id),
                FOREIGN KEY (object) REFERENCES entities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
            CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
            CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
            CREATE INDEX IF NOT EXISTS idx_triples_valid ON triples(valid_from, valid_to);
            ",
        )?;
        Ok(())
    }

    fn connection(&self) -> Result<Connection> {
        let conn = Connection::open(&self.db_path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        Ok(conn)
    }

    pub fn entity_id(name: &str) -> String {
        name.to_lowercase().replace(' ', "_").replace('\'', "")
    }

    fn normalize_predicate(predicate: &str) -> String {
        predicate.to_lowercase().replace(' ', "_")
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use uuid::Uuid;

    use super::KnowledgeGraph;

    fn seeded_graph() -> KnowledgeGraph {
        let graph = KnowledgeGraph::new(
            std::env::temp_dir().join(format!("mempalace-test-{}.sqlite3", Uuid::now_v7())),
        )
        .unwrap();
        graph
            .add_triple(
                "Alice",
                "parent_of",
                "Max",
                Some("2015-04-01"),
                None,
                1.0,
                None,
                None,
            )
            .unwrap();
        graph
            .add_triple(
                "Max",
                "does",
                "swimming",
                Some("2025-01-01"),
                None,
                1.0,
                None,
                None,
            )
            .unwrap();
        graph
            .add_triple(
                "Max",
                "does",
                "chess",
                Some("2024-06-01"),
                None,
                1.0,
                None,
                None,
            )
            .unwrap();
        graph
            .add_triple(
                "Alice",
                "works_at",
                "Acme Corp",
                Some("2020-01-01"),
                Some("2024-12-31"),
                1.0,
                None,
                None,
            )
            .unwrap();
        graph
            .add_triple(
                "Alice",
                "works_at",
                "NewCo",
                Some("2025-01-01"),
                None,
                1.0,
                None,
                None,
            )
            .unwrap();
        graph
    }

    #[test]
    fn add_entity_normalizes_id() {
        let tmp = tempdir().unwrap();
        let graph = KnowledgeGraph::new(tmp.path().join("kg.sqlite3")).unwrap();
        let id = graph.add_entity("Dr. Chen", "person", None).unwrap();
        assert_eq!(id, "dr._chen");
    }

    #[test]
    fn duplicate_triple_returns_existing_id() {
        let tmp = tempdir().unwrap();
        let graph = KnowledgeGraph::new(tmp.path().join("kg.sqlite3")).unwrap();

        let first = graph
            .add_triple("Alice", "knows", "Bob", None, None, 1.0, None, None)
            .unwrap();
        let second = graph
            .add_triple("Alice", "knows", "Bob", None, None, 1.0, None, None)
            .unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn query_and_invalidate_work() {
        let graph = seeded_graph();
        let current = graph
            .query_entity("Alice", Some("2025-06-01"), "outgoing")
            .unwrap();
        let employers: Vec<_> = current
            .iter()
            .filter(|fact| fact.predicate == "works_at")
            .map(|fact| fact.object.clone())
            .collect();

        assert_eq!(employers, vec!["NewCo"]);

        graph
            .invalidate("Max", "does", "chess", Some("2026-01-01"))
            .unwrap();
        let max_facts = graph.query_entity("Max", None, "outgoing").unwrap();
        let chess = max_facts
            .iter()
            .find(|fact| fact.object == "chess")
            .expect("chess fact should exist");

        assert_eq!(chess.valid_to.as_deref(), Some("2026-01-01"));
        assert!(!chess.current);
    }

    #[test]
    fn timeline_is_limited() {
        let tmp = tempdir().unwrap();
        let graph = KnowledgeGraph::new(tmp.path().join("kg.sqlite3")).unwrap();

        for i in 0..105 {
            graph
                .add_triple(
                    "hub",
                    "connects_to",
                    &format!("spoke_{i}"),
                    Some("2025-01-01"),
                    None,
                    1.0,
                    None,
                    None,
                )
                .unwrap();
        }

        let timeline = graph.timeline(Some("hub")).unwrap();
        assert_eq!(timeline.len(), 100);
    }
}
