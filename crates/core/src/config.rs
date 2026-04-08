use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    error::{MempalaceError, Result},
    knowledge_graph::DEFAULT_KG_FILENAME,
};

pub const DEFAULT_COLLECTION_NAME: &str = "mempalace_drawers";
pub const DEFAULT_PALACE_PATH_SUFFIX: &str = ".mempalace/palace";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct FileConfig {
    #[serde(default)]
    palace_path: Option<PathBuf>,
    #[serde(default)]
    collection_name: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    topic_wings: Vec<String>,
    #[serde(default)]
    projects: Vec<String>,
    #[serde(default)]
    people_map: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct MempalaceConfig {
    config_dir: PathBuf,
    file_config: FileConfig,
}

impl MempalaceConfig {
    pub fn load() -> Result<Self> {
        Self::load_with_dir(Self::default_config_dir()?)
    }

    pub fn load_with_dir(config_dir: impl Into<PathBuf>) -> Result<Self> {
        let config_dir = config_dir.into();
        let config_path = config_dir.join("config.json");
        let mut file_config = if config_path.exists() {
            let raw = fs::read_to_string(&config_path)?;
            serde_json::from_str(&raw)?
        } else {
            FileConfig::default()
        };

        let people_map_path = config_dir.join("people_map.json");
        if people_map_path.exists() {
            let raw = fs::read_to_string(&people_map_path)?;
            file_config.people_map = serde_json::from_str(&raw)?;
        }

        Ok(Self {
            config_dir,
            file_config,
        })
    }

    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    pub fn config_path(&self) -> PathBuf {
        self.config_dir.join("config.json")
    }

    pub fn people_map_path(&self) -> PathBuf {
        self.config_dir.join("people_map.json")
    }

    pub fn fastembed_cache_path(&self) -> PathBuf {
        self.config_dir.join("fastembed")
    }

    pub fn onnxruntime_dylib_path(&self) -> PathBuf {
        self.config_dir.join("onnxruntime.dll")
    }

    pub fn palace_path(&self) -> PathBuf {
        std::env::var_os("MEMPALACE_PALACE_PATH")
            .or_else(|| std::env::var_os("MEMPAL_PALACE_PATH"))
            .map(PathBuf::from)
            .or_else(|| self.file_config.palace_path.clone())
            .unwrap_or_else(|| {
                Self::default_config_dir()
                    .expect("home directory should exist")
                    .join("palace")
            })
    }

    pub fn store_path(&self) -> PathBuf {
        Self::resolve_store_path(self.palace_path())
    }

    pub fn collection_name(&self) -> &str {
        self.file_config
            .collection_name
            .as_deref()
            .unwrap_or(DEFAULT_COLLECTION_NAME)
    }

    pub fn mode(&self) -> Option<&str> {
        self.file_config.mode.as_deref()
    }

    pub fn topic_wings(&self) -> &[String] {
        &self.file_config.topic_wings
    }

    pub fn projects(&self) -> &[String] {
        &self.file_config.projects
    }

    pub fn people_map(&self) -> &BTreeMap<String, String> {
        &self.file_config.people_map
    }

    pub fn knowledge_graph_path(&self) -> PathBuf {
        self.config_dir.join(DEFAULT_KG_FILENAME)
    }

    pub fn aaak_entities_path(&self) -> PathBuf {
        self.config_dir.join("aaak_entities.md")
    }

    pub fn critical_facts_path(&self) -> PathBuf {
        self.config_dir.join("critical_facts.md")
    }

    pub fn init(&self) -> Result<PathBuf> {
        fs::create_dir_all(&self.config_dir)?;
        fs::create_dir_all(self.palace_path())?;
        fs::create_dir_all(self.store_path())?;
        fs::create_dir_all(self.fastembed_cache_path())?;

        let config_path = self.config_path();
        if !config_path.exists() {
            let default_config = FileConfig {
                palace_path: Some(self.palace_path()),
                collection_name: Some(self.collection_name().to_owned()),
                mode: None,
                topic_wings: Vec::new(),
                projects: Vec::new(),
                people_map: BTreeMap::new(),
            };

            let raw = serde_json::to_string_pretty(&default_config)?;
            fs::write(&config_path, raw)?;
        }

        Ok(config_path)
    }

    pub fn save_onboarding(
        &mut self,
        mode: String,
        topic_wings: Vec<String>,
        projects: Vec<String>,
    ) -> Result<PathBuf> {
        self.file_config.mode = Some(mode);
        self.file_config.topic_wings = topic_wings;
        self.file_config.projects = projects;

        fs::create_dir_all(&self.config_dir)?;
        let config_path = self.config_path();
        let raw = serde_json::to_string_pretty(&self.file_config)?;
        fs::write(&config_path, raw)?;
        Ok(config_path)
    }

    pub fn save_people_map(&mut self, people_map: BTreeMap<String, String>) -> Result<PathBuf> {
        self.file_config.people_map = people_map;
        fs::create_dir_all(&self.config_dir)?;
        let people_map_path = self.people_map_path();
        let raw = serde_json::to_string_pretty(&self.file_config.people_map)?;
        fs::write(&people_map_path, raw)?;
        Ok(people_map_path)
    }

    fn default_config_dir() -> Result<PathBuf> {
        dirs::home_dir()
            .map(|home| home.join(".mempalace"))
            .ok_or(MempalaceError::MissingHomeDirectory)
    }

    pub fn resolve_store_path(palace_path: impl Into<PathBuf>) -> PathBuf {
        let palace_path = palace_path.into();
        if palace_path.join("chroma.sqlite3").exists() {
            palace_path.join("lancedb")
        } else {
            palace_path
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use tempfile::tempdir;

    use super::MempalaceConfig;

    #[test]
    fn init_writes_default_config() {
        let tmp = tempdir().unwrap();
        let config_dir = tmp.path().join(".mempalace");
        let config = MempalaceConfig::load_with_dir(&config_dir).unwrap();

        let path = config.init().unwrap();

        assert!(path.exists());
        assert!(config.palace_path().exists());
        assert_eq!(
            config.onnxruntime_dylib_path(),
            config_dir.join("onnxruntime.dll")
        );
    }

    #[test]
    fn env_override_wins() {
        let tmp = tempdir().unwrap();
        let env_path = tmp.path().join("from_env");
        unsafe {
            std::env::set_var("MEMPALACE_PALACE_PATH", &env_path);
        }

        let config = MempalaceConfig::load_with_dir(tmp.path().join(".mempalace")).unwrap();
        assert_eq!(config.palace_path(), env_path);

        unsafe {
            std::env::remove_var("MEMPALACE_PALACE_PATH");
        }
    }

    #[test]
    fn save_people_map_round_trips() {
        let tmp = tempdir().unwrap();
        let config_dir = tmp.path().join(".mempalace");
        let mut config = MempalaceConfig::load_with_dir(&config_dir).unwrap();

        let mut people_map = BTreeMap::new();
        people_map.insert("benji".to_owned(), "Benji".to_owned());
        config.save_people_map(people_map.clone()).unwrap();

        let loaded = MempalaceConfig::load_with_dir(config_dir).unwrap();
        assert_eq!(loaded.people_map(), &people_map);
        assert!(loaded.people_map_path().exists());
    }

    #[test]
    fn load_reads_legacy_people_map_file() {
        let tmp = tempdir().unwrap();
        let config_dir = tmp.path().join(".mempalace");
        std::fs::create_dir_all(&config_dir).unwrap();
        std::fs::write(
            config_dir.join("config.json"),
            r#"{"palace_path":"C:\\Users\\benji\\.mempalace\\palace","collection_name":"mempalace_drawers","topic_wings":["legacy"]}"#,
        )
        .unwrap();
        std::fs::write(config_dir.join("people_map.json"), r#"{"benji":"Benji"}"#).unwrap();

        let loaded = MempalaceConfig::load_with_dir(config_dir).unwrap();
        assert_eq!(loaded.people_map().get("benji"), Some(&"Benji".to_owned()));
    }

    #[test]
    fn store_path_moves_under_lancedb_when_chroma_is_present() {
        let tmp = tempdir().unwrap();
        let palace_path = tmp.path().join("palace");
        std::fs::create_dir_all(&palace_path).unwrap();
        std::fs::write(palace_path.join("chroma.sqlite3"), b"legacy").unwrap();

        assert_eq!(
            MempalaceConfig::resolve_store_path(&palace_path),
            palace_path.join("lancedb")
        );
    }

    #[test]
    fn fastembed_cache_lives_under_config_dir() {
        let tmp = tempdir().unwrap();
        let config_dir = tmp.path().join(".mempalace");
        let config = MempalaceConfig::load_with_dir(&config_dir).unwrap();

        assert_eq!(config.fastembed_cache_path(), config_dir.join("fastembed"));
    }

    #[test]
    fn save_onboarding_round_trips() {
        let tmp = tempdir().unwrap();
        let config_dir = tmp.path().join(".mempalace");
        let mut config = MempalaceConfig::load_with_dir(&config_dir).unwrap();

        config
            .save_onboarding(
                "combo".to_owned(),
                vec!["family".to_owned(), "projects".to_owned()],
                vec!["mempalace-rs".to_owned()],
            )
            .unwrap();

        let loaded = MempalaceConfig::load_with_dir(config_dir).unwrap();
        assert_eq!(loaded.mode(), Some("combo"));
        assert_eq!(loaded.topic_wings(), ["family", "projects"]);
        assert_eq!(loaded.projects(), ["mempalace-rs"]);
    }
}
