use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    fs,
    path::{Component, Path, PathBuf},
};

use chrono::Utc;
use ignore::WalkBuilder;
use serde::Deserialize;
use uuid::Uuid;

use crate::{Drawer, DrawerMetadata, MemoryStore, Result};

const CHUNK_SIZE: usize = 800;
const CHUNK_OVERLAP: usize = 100;
const MIN_CHUNK_SIZE: usize = 50;
const STORE_WRITE_BATCH_SIZE: usize = 256;

const READABLE_EXTENSIONS: &[&str] = &[
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml", ".html", ".css",
    ".java", ".go", ".rs", ".rb", ".sh", ".csv", ".sql", ".toml",
];
const NOISY_DATA_EXTENSIONS: &[&str] = &[".json", ".csv", ".sql"];
const NOISY_DATA_DIRS: &[&str] = &[
    "assets",
    "migrations",
    "fixtures",
    "generated",
    "seed",
    "seeds",
];
const MAX_DEFAULT_DATA_FILE_BYTES: u64 = 256 * 1024;

const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".mempalace",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    ".eggs",
    "htmlcov",
    "target",
];

const SKIP_FILENAMES: &[&str] = &[
    "mempalace.yaml",
    "mempalace.yml",
    "mempal.yaml",
    "mempal.yml",
    ".gitignore",
    "package-lock.json",
];

#[derive(Debug, Clone)]
pub struct MineOptions {
    pub wing: Option<String>,
    pub agent: String,
    pub limit: usize,
    pub dry_run: bool,
    pub skip_existing: bool,
    pub exclude_data_files: bool,
    pub respect_gitignore: bool,
    pub log_progress: bool,
}

impl Default for MineOptions {
    fn default() -> Self {
        Self {
            wing: None,
            agent: "mempalace".to_owned(),
            limit: 0,
            dry_run: false,
            skip_existing: false,
            exclude_data_files: false,
            respect_gitignore: true,
            log_progress: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MineSummary {
    pub wing: String,
    pub files_scanned: usize,
    pub files_processed: usize,
    pub files_skipped: usize,
    pub files_replaced: usize,
    pub total_drawers: usize,
    pub room_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ProjectConfig {
    #[serde(default)]
    wing: Option<String>,
    #[serde(default)]
    rooms: Vec<ProjectRoomConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct ProjectRoomConfig {
    name: String,
    #[serde(default)]
    keywords: Vec<String>,
}

#[derive(Debug, Clone)]
struct ProjectRoutingConfig {
    root: PathBuf,
    config: ProjectConfig,
}

pub async fn mine_project<S: MemoryStore + ?Sized>(
    store: &S,
    project_dir: impl AsRef<Path>,
    options: &MineOptions,
) -> Result<MineSummary> {
    let project_dir = project_dir.as_ref();
    let project_path = project_dir.canonicalize()?;
    let routing = load_project_config(&project_path)?;
    let wing = options
        .wing
        .clone()
        .or_else(|| {
            routing
                .as_ref()
                .and_then(|routing| routing.config.wing.clone())
        })
        .unwrap_or_else(|| {
            project_name(
                routing
                    .as_ref()
                    .map(|routing| routing.root.as_path())
                    .unwrap_or(project_path.as_path()),
            )
        });

    let files = scan_project(
        &project_path,
        options.respect_gitignore,
        options.exclude_data_files,
        options.limit,
    )?;
    let mut summary = MineSummary {
        wing: wing.clone(),
        files_scanned: files.len(),
        ..MineSummary::default()
    };
    let mut pending_drawers = VecDeque::new();
    let mut existing_sources = if options.dry_run {
        HashSet::new()
    } else {
        store.source_files().await?
    };

    for (file_index, file) in files.into_iter().enumerate() {
        let progress = file_index + 1;
        let display_name = file
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| file.display().to_string());
        let source_file = file.to_string_lossy().to_string();
        let already_exists = !options.dry_run && existing_sources.contains(&source_file);
        if already_exists && options.skip_existing {
            summary.files_skipped += 1;
            if options.log_progress {
                print_skipped_file(
                    progress,
                    summary.files_scanned,
                    &display_name,
                    "already indexed",
                );
            }
            continue;
        }

        let raw = fs::read(&file)?;
        if already_exists {
            store.delete_source_file(&source_file).await?;
            summary.files_replaced += 1;
        }
        let content = String::from_utf8_lossy(&raw);
        let content = content.trim();
        if content.len() < MIN_CHUNK_SIZE {
            summary.files_skipped += 1;
            if options.log_progress {
                print_skipped_file(progress, summary.files_scanned, &display_name, "too short");
            }
            continue;
        }

        let room = detect_room(&file, content, routing.as_ref(), &project_path);
        let chunks = chunk_text(content);
        if chunks.is_empty() {
            summary.files_skipped += 1;
            if options.log_progress {
                print_skipped_file(progress, summary.files_scanned, &display_name, "no chunks");
            }
            continue;
        }

        let drawer_count = chunks.len();
        summary.files_processed += 1;
        *summary.room_counts.entry(room.clone()).or_insert(0) += 1;
        summary.total_drawers += drawer_count;

        if options.log_progress && options.dry_run {
            if options.dry_run {
                println!(
                    "    [DRY RUN] {} -> room:{} ({} drawers)",
                    display_name, room, drawer_count
                );
            } else {
                println!(
                    "  ✓ [{:4}/{}] {:50} +{}",
                    summary.files_processed + summary.files_skipped,
                    summary.files_scanned,
                    display_name.chars().take(50).collect::<String>(),
                    drawer_count
                );
            }
        }

        if options.dry_run {
            continue;
        }

        let filed_at = Utc::now().to_rfc3339();
        let drawers = chunks
            .into_iter()
            .enumerate()
            .map(|(chunk_index, chunk)| Drawer {
                id: format!("drawer_{}", Uuid::now_v7().simple()),
                content: chunk,
                metadata: DrawerMetadata {
                    wing: wing.clone(),
                    room: room.clone(),
                    source_file: Some(source_file.clone()),
                    chunk_index: chunk_index as i64,
                    added_by: options.agent.clone(),
                    filed_at: Some(filed_at.clone()),
                },
            })
            .collect::<Vec<_>>();
        if drawer_count > STORE_WRITE_BATCH_SIZE {
            let flushed = flush_remaining_drawers(store, &mut pending_drawers).await?;
            if options.log_progress && flushed > 0 {
                println!("      flushed {flushed} queued drawers");
            }
            add_large_file_drawers(
                store,
                drawers,
                progress,
                summary.files_scanned,
                &display_name,
                options.log_progress,
            )
            .await?;
        } else {
            pending_drawers.extend(drawers);
            flush_full_drawer_batches(store, &mut pending_drawers).await?;
            if options.log_progress {
                print_processed_file(progress, summary.files_scanned, &display_name, drawer_count);
            }
        }
        existing_sources.insert(source_file);
    }

    let flushed = flush_remaining_drawers(store, &mut pending_drawers).await?;
    if options.log_progress && flushed > 0 {
        println!("      flushed {flushed} queued drawers");
    }

    Ok(summary)
}

async fn flush_full_drawer_batches<S: MemoryStore + ?Sized>(
    store: &S,
    pending_drawers: &mut VecDeque<Drawer>,
) -> Result<()> {
    while pending_drawers.len() >= STORE_WRITE_BATCH_SIZE {
        let batch = pending_drawers
            .drain(..STORE_WRITE_BATCH_SIZE)
            .collect::<Vec<_>>();
        store.add_drawers(batch).await?;
    }

    Ok(())
}

async fn add_large_file_drawers<S: MemoryStore + ?Sized>(
    store: &S,
    drawers: Vec<Drawer>,
    progress: usize,
    total_files: usize,
    display_name: &str,
    log_progress: bool,
) -> Result<()> {
    let total_drawers = drawers.len();
    let total_batches = total_drawers.div_ceil(STORE_WRITE_BATCH_SIZE);
    let mut processed_drawers = 0usize;
    let mut pending = VecDeque::from(drawers);

    if log_progress {
        print_working_file(progress, total_files, display_name, total_drawers);
    }

    for batch_index in 0..total_batches {
        let batch = pending
            .drain(..pending.len().min(STORE_WRITE_BATCH_SIZE))
            .collect::<Vec<_>>();
        processed_drawers += batch.len();
        store.add_drawers(batch).await?;
        if log_progress {
            println!(
                "      batch {}/{} {}/{} drawers",
                batch_index + 1,
                total_batches,
                processed_drawers,
                total_drawers
            );
        }
    }

    if log_progress {
        print_processed_file(progress, total_files, display_name, total_drawers);
    }

    Ok(())
}

async fn flush_remaining_drawers<S: MemoryStore + ?Sized>(
    store: &S,
    pending_drawers: &mut VecDeque<Drawer>,
) -> Result<usize> {
    if pending_drawers.is_empty() {
        return Ok(0);
    }

    let batch = pending_drawers.drain(..).collect::<Vec<_>>();
    let flushed = batch.len();
    store.add_drawers(batch).await?;
    Ok(flushed)
}

fn print_processed_file(
    progress: usize,
    total_files: usize,
    display_name: &str,
    drawer_count: usize,
) {
    println!(
        "  OK  [{:4}/{}] {:50} +{}",
        progress,
        total_files,
        display_name.chars().take(50).collect::<String>(),
        drawer_count
    );
}

fn print_skipped_file(progress: usize, total_files: usize, display_name: &str, reason: &str) {
    println!(
        "  SKIP[{:4}/{}] {:50} ({})",
        progress,
        total_files,
        display_name.chars().take(50).collect::<String>(),
        reason
    );
}

fn print_working_file(
    progress: usize,
    total_files: usize,
    display_name: &str,
    drawer_count: usize,
) {
    println!(
        "  WORK[{:4}/{}] {:50} +{}",
        progress,
        total_files,
        display_name.chars().take(50).collect::<String>(),
        drawer_count
    );
}

pub fn scan_project(
    project_dir: impl AsRef<Path>,
    respect_gitignore: bool,
    exclude_data_files: bool,
    limit: usize,
) -> Result<Vec<PathBuf>> {
    let project_dir = project_dir.as_ref();
    let mut builder = WalkBuilder::new(project_dir);
    builder.hidden(false);
    builder.require_git(false);
    builder.parents(respect_gitignore);
    builder.ignore(respect_gitignore);
    builder.git_ignore(respect_gitignore);
    builder.git_global(respect_gitignore);
    builder.git_exclude(respect_gitignore);
    builder.sort_by_file_path(|a, b| a.cmp(b));
    builder.filter_entry(|entry| {
        if !entry
            .file_type()
            .map(|file_type| file_type.is_dir())
            .unwrap_or(false)
        {
            return true;
        }
        !should_skip_dir(entry.path())
    });

    let mut files = Vec::new();
    for result in builder.build() {
        let Ok(entry) = result else {
            continue;
        };

        let Some(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }
        if !is_readable_file(entry.path()) {
            continue;
        }
        if exclude_data_files && is_probably_noisy_data_file(entry.path()) {
            continue;
        }

        files.push(entry.into_path());
        if limit > 0 && files.len() >= limit {
            break;
        }
    }

    Ok(files)
}

fn project_name(project_path: &Path) -> String {
    project_path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "project".to_owned())
}

fn detect_room(
    filepath: &Path,
    content: &str,
    routing: Option<&ProjectRoutingConfig>,
    project_path: &Path,
) -> String {
    if let Some(routing) = routing {
        if let Some(room) = detect_room_from_config(filepath, content, routing) {
            return room;
        }
    }

    detect_room_from_path(filepath, project_path)
}

fn detect_room_from_path(filepath: &Path, project_path: &Path) -> String {
    let Ok(relative) = filepath.strip_prefix(project_path) else {
        return "general".to_owned();
    };

    let mut parts = relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().to_string()),
            _ => None,
        })
        .collect::<Vec<_>>();

    if parts.len() > 1 {
        sanitize_room(&parts.remove(0))
    } else {
        "general".to_owned()
    }
}

fn detect_room_from_config(
    filepath: &Path,
    content: &str,
    routing: &ProjectRoutingConfig,
) -> Option<String> {
    if routing.config.rooms.is_empty() {
        return None;
    }

    let relative = filepath
        .strip_prefix(&routing.root)
        .ok()?
        .to_string_lossy()
        .replace('\\', "/")
        .to_lowercase();
    let filename = filepath.file_stem()?.to_string_lossy().to_lowercase();
    let content_lower = content
        .chars()
        .take(2_000)
        .collect::<String>()
        .to_lowercase();
    let path_parts = relative.split('/').collect::<Vec<_>>();

    for part in path_parts.iter().take(path_parts.len().saturating_sub(1)) {
        for room in &routing.config.rooms {
            let candidates = std::iter::once(room.name.as_str())
                .chain(room.keywords.iter().map(String::as_str))
                .map(|candidate| candidate.to_lowercase())
                .collect::<Vec<_>>();
            if candidates.iter().any(|candidate| {
                *part == candidate || part.contains(candidate.as_str()) || candidate.contains(*part)
            }) {
                return Some(room.name.clone());
            }
        }
    }

    for room in &routing.config.rooms {
        let room_name = room.name.to_lowercase();
        if room_name.contains(&filename) || filename.contains(&room_name) {
            return Some(room.name.clone());
        }
    }

    let mut best_room = None;
    let mut best_score = 0usize;
    for room in &routing.config.rooms {
        let score = room
            .keywords
            .iter()
            .chain(std::iter::once(&room.name))
            .map(|keyword| content_lower.matches(&keyword.to_lowercase()).count())
            .sum::<usize>();
        if score > best_score {
            best_score = score;
            best_room = Some(room.name.clone());
        }
    }

    if best_score > 0 {
        best_room
    } else {
        Some("general".to_owned())
    }
}

fn sanitize_room(room: &str) -> String {
    room.trim().replace(' ', "_").to_lowercase()
}

fn chunk_text(content: &str) -> Vec<String> {
    let content = content.trim();
    if content.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < content.len() {
        let mut end = floor_char_boundary(content, (start + CHUNK_SIZE).min(content.len()));

        if end < content.len() {
            if let Some(index) = content[start..end]
                .rfind("\n\n")
                .filter(|index| *index > CHUNK_SIZE / 2)
            {
                end = start + index;
            } else if let Some(index) = content[start..end]
                .rfind('\n')
                .filter(|index| *index > CHUNK_SIZE / 2)
            {
                end = start + index;
            }
        }

        let chunk = content[start..end].trim();
        if chunk.len() >= MIN_CHUNK_SIZE {
            chunks.push(chunk.to_owned());
        }

        if end >= content.len() {
            break;
        }

        start = floor_char_boundary(content, end.saturating_sub(CHUNK_OVERLAP));
    }

    chunks
}

fn floor_char_boundary(content: &str, mut index: usize) -> usize {
    while index > 0 && !content.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn should_skip_dir(path: &Path) -> bool {
    let Some(name) = path.file_name().map(|name| name.to_string_lossy()) else {
        return false;
    };
    SKIP_DIRS.contains(&name.as_ref()) || name.ends_with(".egg-info")
}

fn is_readable_file(path: &Path) -> bool {
    let Some(file_name) = path.file_name().map(|name| name.to_string_lossy()) else {
        return false;
    };
    if SKIP_FILENAMES.contains(&file_name.as_ref()) {
        return false;
    }

    let Some(extension) = path
        .extension()
        .map(|ext| format!(".{}", ext.to_string_lossy().to_lowercase()))
    else {
        return false;
    };
    READABLE_EXTENSIONS.contains(&extension.as_str())
}

fn is_probably_noisy_data_file(path: &Path) -> bool {
    let Some(extension) = path
        .extension()
        .map(|ext| format!(".{}", ext.to_string_lossy().to_lowercase()))
    else {
        return false;
    };

    if !NOISY_DATA_EXTENSIONS.contains(&extension.as_str()) {
        return false;
    }

    let in_noisy_dir = path.components().any(|component| match component {
        Component::Normal(part) => {
            let part = part.to_string_lossy().to_lowercase();
            NOISY_DATA_DIRS.contains(&part.as_str())
        }
        _ => false,
    });

    if in_noisy_dir {
        return true;
    }

    fs::metadata(path)
        .map(|metadata| metadata.len() > MAX_DEFAULT_DATA_FILE_BYTES)
        .unwrap_or(false)
}

fn load_project_config(project_path: &Path) -> Result<Option<ProjectRoutingConfig>> {
    for ancestor in project_path.ancestors() {
        if let Some(config_path) = project_config_path(ancestor) {
            let raw = fs::read_to_string(&config_path)?;
            let config = serde_yaml::from_str::<ProjectConfig>(&raw)?;
            return Ok(Some(ProjectRoutingConfig {
                root: ancestor.to_path_buf(),
                config,
            }));
        }
    }

    Ok(None)
}

fn project_config_path(path: &Path) -> Option<PathBuf> {
    [
        "mempalace.yaml",
        "mempalace.yml",
        "mempal.yaml",
        "mempal.yml",
    ]
    .into_iter()
    .map(|name| path.join(name))
    .find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashSet, VecDeque},
        fs,
        path::{Path, PathBuf},
        sync::Mutex,
    };

    use async_trait::async_trait;
    use tempfile::tempdir;

    use crate::{Drawer, DrawerMetadata, MemoryStore, Result, SearchHit, SearchQuery, StoreStatus};

    use super::{
        MineOptions, STORE_WRITE_BATCH_SIZE, chunk_text, detect_room, flush_full_drawer_batches,
        load_project_config, mine_project, scan_project,
    };

    #[derive(Default)]
    struct MockStore {
        drawers: Mutex<Vec<Drawer>>,
        add_drawers_calls: Mutex<usize>,
    }

    #[async_trait]
    impl MemoryStore for MockStore {
        async fn add_drawer(&self, drawer: Drawer) -> Result<()> {
            self.drawers.lock().unwrap().push(drawer);
            Ok(())
        }

        async fn add_drawers(&self, drawers: Vec<Drawer>) -> Result<()> {
            *self.add_drawers_calls.lock().unwrap() += 1;
            self.drawers.lock().unwrap().extend(drawers);
            Ok(())
        }

        async fn get_drawer(&self, drawer_id: &str) -> Result<Option<Drawer>> {
            Ok(self
                .drawers
                .lock()
                .unwrap()
                .iter()
                .find(|drawer| drawer.id == drawer_id)
                .cloned())
        }

        async fn delete_drawer(&self, drawer_id: &str) -> Result<bool> {
            let mut drawers = self.drawers.lock().unwrap();
            let before = drawers.len();
            drawers.retain(|drawer| drawer.id != drawer_id);
            Ok(drawers.len() != before)
        }

        async fn delete_source_file(&self, source_file: &str) -> Result<usize> {
            let mut drawers = self.drawers.lock().unwrap();
            let before = drawers.len();
            drawers.retain(|drawer| drawer.metadata.source_file.as_deref() != Some(source_file));
            Ok(before - drawers.len())
        }

        async fn list_drawers(&self, wing: Option<&str>) -> Result<Vec<Drawer>> {
            Ok(self
                .drawers
                .lock()
                .unwrap()
                .iter()
                .filter(|drawer| wing.is_none_or(|wing| drawer.metadata.wing == wing))
                .cloned()
                .collect())
        }

        async fn search(&self, _query: SearchQuery) -> Result<Vec<SearchHit>> {
            Ok(Vec::new())
        }

        async fn status(&self) -> Result<StoreStatus> {
            Ok(StoreStatus {
                total_drawers: self.drawers.lock().unwrap().len(),
            })
        }

        async fn has_source_file(&self, source_file: &str) -> Result<bool> {
            Ok(self
                .drawers
                .lock()
                .unwrap()
                .iter()
                .any(|drawer| drawer.metadata.source_file.as_deref() == Some(source_file)))
        }

        async fn source_files(&self) -> Result<HashSet<String>> {
            Ok(self
                .drawers
                .lock()
                .unwrap()
                .iter()
                .filter_map(|drawer| drawer.metadata.source_file.clone())
                .collect())
        }

        async fn room_counts(&self) -> Result<Vec<crate::RoomStatus>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn chunk_text_splits_long_content() {
        let input = format!("{}\n\n{}", "a".repeat(900), "b".repeat(900));
        let chunks = chunk_text(&input);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|chunk| chunk.len() >= 50));
    }

    #[test]
    fn chunk_text_handles_unicode_boundaries() {
        let input = format!("{}\n\n{}", "é".repeat(500), "漢".repeat(500));
        let chunks = chunk_text(&input);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|chunk| !chunk.is_empty()));
    }

    #[test]
    fn detect_room_uses_first_path_segment() {
        let project = PathBuf::from("F:/Dev/example");
        let file = project.join("src").join("lib.rs");
        assert_eq!(detect_room(&file, "fn main() {}", None, &project), "src");

        let root_file = project.join("README.md");
        assert_eq!(detect_room(&root_file, "# hi", None, &project), "general");
    }

    #[test]
    fn detect_room_uses_project_config_keywords() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        fs::write(
            root.join("mempalace.yml"),
            "wing: demo\nrooms:\n  - name: crates\n    keywords:\n      - rust\n",
        )
        .unwrap();
        fs::create_dir_all(root.join("nested").join("pkg")).unwrap();
        let file = root.join("nested").join("pkg").join("lib.rs");
        let routing = load_project_config(&file.parent().unwrap().to_path_buf())
            .unwrap()
            .unwrap();

        assert_eq!(
            detect_room(
                &file,
                "this rust crate exists",
                Some(&routing),
                file.parent().unwrap()
            ),
            "crates"
        );
    }

    #[test]
    fn scan_project_skips_common_dirs_and_respects_gitignore() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("node_modules")).unwrap();
        fs::create_dir_all(root.join("docs")).unwrap();
        fs::write(root.join(".gitignore"), "docs/\n").unwrap();
        fs::write(root.join("src").join("main.rs"), "fn main() {}\n").unwrap();
        fs::write(
            root.join("node_modules").join("lib.js"),
            "console.log('x');\n",
        )
        .unwrap();
        fs::write(root.join("docs").join("guide.md"), "# hidden\n").unwrap();

        let files = scan_project(root, true, false, 0).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with(Path::new("src").join("main.rs")));
    }

    #[test]
    fn scan_project_includes_noisy_data_files_by_default() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("assets").join("migrations")).unwrap();
        fs::write(root.join("Cargo.toml"), "[package]\nname='demo'\n").unwrap();
        fs::write(
            root.join("assets").join("migrations").join("quests.json"),
            "{\n  \"quest\": \"doomkitten\"\n}\n",
        )
        .unwrap();

        let files = scan_project(root, true, false, 0).unwrap();
        assert!(files.iter().any(|path| path.ends_with("quests.json")));

        let files = scan_project(root, true, true, 0).unwrap();
        assert!(files.iter().all(|path| !path.ends_with("quests.json")));
    }

    #[tokio::test]
    async fn mine_project_replaces_existing_source_files_by_default() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        fs::write(root.join("Cargo.toml"), "[package]\nname='demo'\n").unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        let file = root.join("src").join("lib.rs");
        fs::write(&file, "fn alpha() {}\n".repeat(20)).unwrap();

        let store = MockStore::default();
        let first = mine_project(&store, root, &MineOptions::default())
            .await
            .unwrap();
        assert!(first.files_processed >= 1);

        fs::write(&file, "fn beta() {}\n".repeat(20)).unwrap();
        let second = mine_project(&store, root, &MineOptions::default())
            .await
            .unwrap();

        assert!(second.files_replaced >= 1);
        let drawers = store.list_drawers(Some(&second.wing)).await.unwrap();
        assert!(drawers.iter().any(|drawer| drawer.content.contains("beta")));
        assert!(
            !drawers
                .iter()
                .any(|drawer| drawer.content.contains("alpha"))
        );
    }

    #[tokio::test]
    async fn mine_project_batches_store_writes_across_files() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("docs")).unwrap();
        fs::write(
            root.join("src").join("lib.rs"),
            "fn alpha() {}\n".repeat(20),
        )
        .unwrap();
        fs::write(root.join("docs").join("guide.md"), "# Guide\n".repeat(20)).unwrap();

        let store = MockStore::default();
        let summary = mine_project(&store, root, &MineOptions::default())
            .await
            .unwrap();

        assert_eq!(summary.files_processed, 2);
        assert_eq!(*store.add_drawers_calls.lock().unwrap(), 1);
        assert_eq!(store.drawers.lock().unwrap().len(), summary.total_drawers);
    }

    #[tokio::test]
    async fn flush_full_drawer_batches_splits_large_pending_sets() {
        let store = MockStore::default();
        let mut pending = VecDeque::from(
            (0..(STORE_WRITE_BATCH_SIZE + 10))
                .map(|index| Drawer {
                    id: format!("drawer_{index}"),
                    content: "chunk".to_owned(),
                    metadata: DrawerMetadata {
                        wing: "project".to_owned(),
                        room: "src".to_owned(),
                        source_file: Some(format!("drawer_{index}.txt")),
                        chunk_index: 0,
                        added_by: "test".to_owned(),
                        filed_at: Some("2026-04-08T00:00:00".to_owned()),
                    },
                })
                .collect::<Vec<_>>(),
        );

        flush_full_drawer_batches(&store, &mut pending)
            .await
            .unwrap();

        assert_eq!(*store.add_drawers_calls.lock().unwrap(), 1);
        assert_eq!(store.drawers.lock().unwrap().len(), STORE_WRITE_BATCH_SIZE);
        assert_eq!(pending.len(), 10);
    }
}
