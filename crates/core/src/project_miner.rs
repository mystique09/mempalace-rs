use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Component, Path, PathBuf},
};

use chrono::Utc;
use ignore::WalkBuilder;
use uuid::Uuid;

use crate::{Drawer, DrawerMetadata, MemoryStore, Result};

const CHUNK_SIZE: usize = 800;
const CHUNK_OVERLAP: usize = 100;
const MIN_CHUNK_SIZE: usize = 50;

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
    pub include_data_files: bool,
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
            include_data_files: false,
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
    pub total_drawers: usize,
    pub room_counts: BTreeMap<String, usize>,
}

pub async fn mine_project<S: MemoryStore + ?Sized>(
    store: &S,
    project_dir: impl AsRef<Path>,
    options: &MineOptions,
) -> Result<MineSummary> {
    let project_dir = project_dir.as_ref();
    let project_path = project_dir.canonicalize()?;
    let wing = options
        .wing
        .clone()
        .unwrap_or_else(|| project_name(&project_path));

    let files = scan_project(
        &project_path,
        options.respect_gitignore,
        options.include_data_files,
        options.limit,
    )?;
    let mut summary = MineSummary {
        wing: wing.clone(),
        files_scanned: files.len(),
        ..MineSummary::default()
    };
    let mut existing_sources = if options.dry_run {
        HashSet::new()
    } else {
        store.source_files().await?
    };

    for file in files {
        let source_file = file.to_string_lossy().to_string();
        if !options.dry_run && existing_sources.contains(&source_file) {
            summary.files_skipped += 1;
            continue;
        }

        let raw = fs::read(&file)?;
        let content = String::from_utf8_lossy(&raw);
        let content = content.trim();
        if content.len() < MIN_CHUNK_SIZE {
            summary.files_skipped += 1;
            continue;
        }

        let room = detect_room(&file, &project_path);
        let chunks = chunk_text(content);
        if chunks.is_empty() {
            summary.files_skipped += 1;
            continue;
        }

        let drawer_count = chunks.len();
        summary.files_processed += 1;
        *summary.room_counts.entry(room.clone()).or_insert(0) += 1;
        summary.total_drawers += drawer_count;

        if options.log_progress {
            let display_name = file
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| file.display().to_string());
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
        store.add_drawers(drawers).await?;
        existing_sources.insert(source_file);
    }

    Ok(summary)
}

pub fn scan_project(
    project_dir: impl AsRef<Path>,
    respect_gitignore: bool,
    include_data_files: bool,
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
        if !include_data_files && is_probably_noisy_data_file(entry.path()) {
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

fn detect_room(filepath: &Path, project_path: &Path) -> String {
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

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use tempfile::tempdir;

    use super::{chunk_text, detect_room, scan_project};

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
        assert_eq!(detect_room(&file, &project), "src");

        let root_file = project.join("README.md");
        assert_eq!(detect_room(&root_file, &project), "general");
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
    fn scan_project_skips_noisy_data_files_by_default() {
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
        assert!(files.iter().all(|path| !path.ends_with("quests.json")));

        let files = scan_project(root, true, true, 0).unwrap();
        assert!(files.iter().any(|path| path.ends_with("quests.json")));
    }
}
