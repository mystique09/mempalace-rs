use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    fs,
    io::Read,
    ops::Range,
    path::{Component, Path, PathBuf},
};

use chrono::Utc;
use ignore::WalkBuilder;
use serde::Deserialize;
use uuid::Uuid;

use crate::{ContentKind, Drawer, DrawerMetadata, MemoryStore, Result};

const CHUNK_SIZE: usize = 800;
const CHUNK_OVERLAP: usize = 100;
const MIN_CHUNK_SIZE: usize = 50;
const MAX_STRUCTURAL_CODE_UNIT_SIZE: usize = CHUNK_SIZE * 2;
const STORE_WRITE_BATCH_SIZE: usize = 256;
const TEXT_SNIFF_BYTES: usize = 8 * 1024;
const MAX_BINARY_CONTROL_RATIO_PERCENT: usize = 1;
const MAX_REPLACEMENT_RATIO_PERCENT: usize = 10;

const BINARY_EXTENSIONS: &[&str] = &[
    ".7z", ".a", ".avi", ".avif", ".bin", ".bmp", ".bz2", ".class", ".cur", ".db", ".dib", ".dll",
    ".doc", ".docx", ".dylib", ".eot", ".exe", ".fla", ".flac", ".gif", ".gz", ".ico", ".jar",
    ".jpeg", ".jpg", ".lib", ".m4a", ".mkv", ".mov", ".mp3", ".mp4", ".o", ".obj", ".ogg", ".otf",
    ".pdf", ".png", ".ppt", ".pptx", ".pyc", ".pyd", ".so", ".sqlite", ".sqlite3", ".svgz", ".swf",
    ".tar", ".ttf", ".wav", ".webm", ".webp", ".woff", ".woff2", ".xls", ".xlsx", ".xz", ".zip",
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
    // Evaluation queries and labels must never become production retrieval text.
    "multi_domain_queries.json",
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

#[derive(Debug, Clone)]
struct MinedChunk {
    content: String,
    retrieval_text: Option<String>,
}

#[derive(Debug, Clone)]
struct RustCodeUnit {
    range: Range<usize>,
    symbol: Option<String>,
    enclosing: Option<String>,
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
        let content_kind = content_kind_for_path(&file, &wing, &room);
        let relative_path = file
            .strip_prefix(&project_path)
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");
        let chunks = chunk_file(&file, &project_path, content)
            .into_iter()
            .map(|mut chunk| {
                if chunk.retrieval_text.is_none() {
                    chunk.retrieval_text = retrieval_text_for_content(
                        content_kind,
                        RetrievalContext {
                            path: Some(&relative_path),
                            wing: &wing,
                            room: &room,
                            agent: &options.agent,
                            filed_at: None,
                        },
                        content,
                        &chunk.content,
                    );
                }
                chunk
            })
            .collect::<Vec<_>>();
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
                content: chunk.content,
                retrieval_text: chunk.retrieval_text,
                metadata: DrawerMetadata {
                    content_kind,
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
    if let Some(routing) = routing
        && let Some(room) = detect_room_from_config(filepath, content, routing)
    {
        return room;
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

fn chunk_file(path: &Path, project_path: &Path, content: &str) -> Vec<MinedChunk> {
    let relative_path = path
        .strip_prefix(project_path)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");

    if normalized_extension(path).as_deref() == Some(".rs") {
        return chunk_rust(content, &relative_path);
    }

    if let Some(language) = code_language(path) {
        return chunk_code_line_ranges(content)
            .into_iter()
            .map(|range| {
                let code = content[range].to_owned();
                let symbol = path.file_stem().and_then(|stem| stem.to_str());
                MinedChunk {
                    retrieval_text: Some(code_retrieval_text(
                        &relative_path,
                        language,
                        Some("file"),
                        symbol,
                        &code,
                    )),
                    content: code,
                }
            })
            .collect();
    }

    chunk_text(content)
        .into_iter()
        .map(|content| MinedChunk {
            content,
            retrieval_text: None,
        })
        .collect()
}

fn chunk_rust(content: &str, relative_path: &str) -> Vec<MinedChunk> {
    let content = content.trim();
    if content.is_empty() {
        return Vec::new();
    }

    let parsed = parse_rust(content);
    let structural_units = parsed
        .as_ref()
        .map(|tree| rust_structural_units(tree.root_node(), content))
        .unwrap_or_default();

    let mut ranges = Vec::new();
    let mut cursor = 0usize;
    for code_unit in &structural_units {
        if cursor < code_unit.range.start {
            ranges.extend(
                chunk_code_line_ranges(&content[cursor..code_unit.range.start])
                    .into_iter()
                    .map(|range| (range.start + cursor)..(range.end + cursor)),
            );
        }
        ranges.push(code_unit.range.clone());
        cursor = code_unit.range.end;
    }
    if cursor < content.len() {
        ranges.extend(
            chunk_code_line_ranges(&content[cursor..])
                .into_iter()
                .map(|range| (range.start + cursor)..(range.end + cursor)),
        );
    }

    ranges
        .into_iter()
        .map(|range| {
            let code = content[range.clone()].to_owned();
            let explicit_context = structural_units.iter().find(|unit| unit.range == range);
            let inferred_context = parsed
                .as_ref()
                .and_then(|tree| rust_context_for_range(tree.root_node(), range.clone(), content));
            let symbol = explicit_context
                .and_then(|unit| unit.symbol.as_deref())
                .or_else(|| {
                    inferred_context
                        .as_ref()
                        .and_then(|context| context.0.as_deref())
                });
            let enclosing = explicit_context
                .and_then(|unit| unit.enclosing.as_deref())
                .or_else(|| {
                    inferred_context
                        .as_ref()
                        .and_then(|context| context.1.as_deref())
                });

            MinedChunk {
                retrieval_text: Some(code_retrieval_text(
                    relative_path,
                    "rust",
                    enclosing,
                    symbol,
                    &code,
                )),
                content: code,
            }
        })
        .collect()
}

fn parse_rust(content: &str) -> Option<tree_sitter::Tree> {
    let mut parser = tree_sitter::Parser::new();
    let language = tree_sitter_rust::LANGUAGE.into();
    parser.set_language(&language).ok()?;
    parser.parse(content, None)
}

fn rust_structural_units(root: tree_sitter::Node<'_>, content: &str) -> Vec<RustCodeUnit> {
    let mut units = Vec::new();
    collect_rust_structural_units(root, content, &mut units);
    units.sort_by_key(|unit| unit.range.start);

    let mut non_overlapping = Vec::new();
    for unit in units {
        if non_overlapping
            .last()
            .is_none_or(|previous: &RustCodeUnit| unit.range.start >= previous.range.end)
        {
            non_overlapping.push(unit);
        }
    }
    non_overlapping
}

fn collect_rust_structural_units(
    node: tree_sitter::Node<'_>,
    content: &str,
    units: &mut Vec<RustCodeUnit>,
) {
    if node.kind() == "function_item" {
        let range = trimmed_range(content, node.byte_range());
        let length = range
            .as_ref()
            .map_or(0, |range| range.end.saturating_sub(range.start));
        if (MIN_CHUNK_SIZE..=MAX_STRUCTURAL_CODE_UNIT_SIZE).contains(&length)
            && !rust_node_contains_kind(node, "match_arm")
        {
            units.push(RustCodeUnit {
                range: range.expect("length came from this range"),
                symbol: rust_node_symbol(node, content),
                enclosing: rust_enclosing_symbol(node, content),
            });
            return;
        }
    }

    if node.kind() == "match_arm" {
        let Some(range) = trimmed_range(
            content,
            extend_rust_match_arm_range(node.byte_range(), content),
        ) else {
            return;
        };
        let length = range.end.saturating_sub(range.start);
        if (MIN_CHUNK_SIZE..=MAX_STRUCTURAL_CODE_UNIT_SIZE).contains(&length) {
            let symbol = node
                .child_by_field_name("pattern")
                .and_then(|pattern| pattern.utf8_text(content.as_bytes()).ok())
                .and_then(rust_match_pattern_symbol);
            units.push(RustCodeUnit {
                range,
                symbol,
                enclosing: rust_enclosing_symbol(node, content),
            });
            return;
        }
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_rust_structural_units(child, content, units);
    }
}

fn rust_node_contains_kind(node: tree_sitter::Node<'_>, kind: &str) -> bool {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .any(|child| child.kind() == kind || rust_node_contains_kind(child, kind))
}

fn extend_rust_match_arm_range(mut range: Range<usize>, content: &str) -> Range<usize> {
    let line_start = content[..range.start]
        .rfind('\n')
        .map_or(0, |index| index + 1);
    if content[line_start..range.start]
        .chars()
        .all(char::is_whitespace)
    {
        range.start = line_start;
    }

    let remainder = &content[range.end..];
    let horizontal_whitespace = remainder
        .char_indices()
        .take_while(|(_, ch)| matches!(ch, ' ' | '\t'))
        .map(|(index, ch)| index + ch.len_utf8())
        .last()
        .unwrap_or(0);
    let comma_index = range.end + horizontal_whitespace;
    if content[comma_index..].starts_with(',') {
        range.end = comma_index + 1;
    }

    range
}

fn chunk_code_line_ranges(content: &str) -> Vec<Range<usize>> {
    let Some(content_range) = trimmed_range(content, 0..content.len()) else {
        return Vec::new();
    };

    let mut chunks = Vec::new();
    let mut start = content_range.start;
    while start < content_range.end {
        let desired_end = floor_char_boundary(content, (start + CHUNK_SIZE).min(content_range.end));
        let end = if desired_end >= content_range.end {
            content_range.end
        } else if let Some(newline) = content[start..desired_end]
            .rfind('\n')
            .filter(|newline| *newline >= CHUNK_SIZE / 2)
        {
            start + newline + 1
        } else {
            content[desired_end..content_range.end]
                .find('\n')
                .map_or(content_range.end, |newline| desired_end + newline + 1)
        };

        if let Some(range) = trimmed_range(content, start..end)
            && range.end.saturating_sub(range.start) >= MIN_CHUNK_SIZE
        {
            chunks.push(range);
        }
        start = end;
    }

    chunks
}

fn trimmed_range(content: &str, range: Range<usize>) -> Option<Range<usize>> {
    let value = &content[range.clone()];
    let trimmed_start = value.len() - value.trim_start().len();
    let trimmed_end = value.trim_end().len();
    (trimmed_start < trimmed_end)
        .then_some((range.start + trimmed_start)..(range.start + trimmed_end))
}

fn rust_context_for_range(
    root: tree_sitter::Node<'_>,
    range: Range<usize>,
    content: &str,
) -> Option<(Option<String>, Option<String>)> {
    let mut node = root.named_descendant_for_byte_range(range.start, range.start)?;
    let mut symbols = Vec::new();
    loop {
        if let Some(symbol) = rust_node_symbol(node, content) {
            symbols.push(symbol);
        }
        let Some(parent) = node.parent() else {
            break;
        };
        node = parent;
    }

    let symbol = symbols.first().cloned();
    let enclosing = symbols.get(1).cloned();
    Some((symbol, enclosing))
}

fn rust_enclosing_symbol(node: tree_sitter::Node<'_>, content: &str) -> Option<String> {
    let mut ancestor = node.parent();
    while let Some(node) = ancestor {
        if let Some(symbol) = rust_node_symbol(node, content) {
            return Some(symbol);
        }
        ancestor = node.parent();
    }
    None
}

fn rust_node_symbol(node: tree_sitter::Node<'_>, content: &str) -> Option<String> {
    match node.kind() {
        "function_item" | "struct_item" | "enum_item" | "trait_item" | "mod_item" | "type_item"
        | "const_item" | "static_item" => node
            .child_by_field_name("name")
            .and_then(|name| name.utf8_text(content.as_bytes()).ok())
            .map(str::to_owned),
        "impl_item" => node
            .child_by_field_name("type")
            .and_then(|name| name.utf8_text(content.as_bytes()).ok())
            .map(str::to_owned),
        _ => None,
    }
}

fn rust_match_pattern_symbol(pattern: &str) -> Option<String> {
    let pattern = pattern.trim().trim_start_matches('|').trim();
    let symbol = pattern
        .split(|ch: char| ch.is_whitespace() || matches!(ch, '(' | '{' | '@' | '|'))
        .next()
        .unwrap_or_default()
        .trim_matches('&');
    (!symbol.is_empty() && symbol != "_").then(|| symbol.to_owned())
}

fn code_retrieval_text(
    path: &str,
    language: &str,
    enclosing: Option<&str>,
    symbol: Option<&str>,
    code: &str,
) -> String {
    let enclosing = enclosing.unwrap_or("file");
    let symbol = symbol.unwrap_or("file");
    let identifier_words = [symbol, enclosing]
        .into_iter()
        .flat_map(split_identifier_words)
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "path: {path}\nlanguage: {language}\nenclosing: {enclosing}\nsymbol: {symbol}\nidentifier words: {identifier_words}\ncode:\n{code}"
    )
}

#[derive(Debug, Clone, Copy)]
pub struct RetrievalContext<'a> {
    pub path: Option<&'a str>,
    pub wing: &'a str,
    pub room: &'a str,
    pub agent: &'a str,
    pub filed_at: Option<&'a str>,
}

pub fn retrieval_text_for_content(
    content_kind: ContentKind,
    context: RetrievalContext<'_>,
    document: &str,
    chunk: &str,
) -> Option<String> {
    let RetrievalContext {
        path,
        wing,
        room,
        agent,
        filed_at,
    } = context;
    Some(match content_kind {
        ContentKind::Conversation => conversation_retrieval_text(path, wing, room, chunk),
        ContentKind::Documentation => documentation_retrieval_text(path, document, chunk),
        ContentKind::Diary => {
            let date = filed_at
                .and_then(valid_date_prefix)
                .or_else(|| path.and_then(path_date))
                .unwrap_or("unknown");
            format!(
                "kind: diary\ndate: {date}\nagent: {agent}\ntopic: {room}{}\nverbatim:\n{chunk}",
                source_context(path)
            )
        }
        ContentKind::Prose | ContentKind::Unknown => path.map_or_else(
            || chunk.to_owned(),
            |path| format!("kind: prose\nsource: {path}\nverbatim:\n{chunk}"),
        ),
        ContentKind::Code => return None,
    })
}

fn source_context(path: Option<&str>) -> String {
    path.map_or_else(String::new, |path| format!("\nsource: {path}"))
}

fn conversation_retrieval_text(path: Option<&str>, wing: &str, room: &str, chunk: &str) -> String {
    let mut turns = Vec::new();
    for line in chunk.lines() {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
            collect_conversation_turns(&value, &mut turns);
        }
    }
    turns.dedup();

    let role_context = if turns.is_empty() {
        String::new()
    } else {
        format!("\nroles:\n{}", turns.join("\n"))
    };
    format!(
        "kind: conversation\nsession: {wing}/{room}{}{role_context}\nverbatim:\n{chunk}",
        source_context(path)
    )
}

fn collect_conversation_turns(value: &serde_json::Value, turns: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(object) => {
            if let Some(role) = object.get("role").and_then(serde_json::Value::as_str)
                && matches!(role, "user" | "assistant" | "system" | "developer" | "tool")
                && let Some(content) = object.get("content")
                && let Some(text) = conversation_content_text(content)
            {
                turns.push(format!("{role}: {text}"));
            }
            for nested in object.values() {
                collect_conversation_turns(nested, turns);
            }
        }
        serde_json::Value::Array(values) => {
            for nested in values {
                collect_conversation_turns(nested, turns);
            }
        }
        _ => {}
    }
}

fn conversation_content_text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => {
            Some(text.split_whitespace().collect::<Vec<_>>().join(" "))
        }
        serde_json::Value::Array(items) => {
            let text = items
                .iter()
                .filter_map(|item| {
                    item.get("text")
                        .and_then(serde_json::Value::as_str)
                        .or_else(|| item.get("content").and_then(serde_json::Value::as_str))
                })
                .collect::<Vec<_>>()
                .join(" ");
            (!text.is_empty()).then_some(text)
        }
        serde_json::Value::Object(object) => object
            .get("text")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

fn documentation_retrieval_text(path: Option<&str>, document: &str, chunk: &str) -> String {
    let title = document
        .lines()
        .find_map(|line| line.trim().strip_prefix("# ").map(str::trim))
        .or_else(|| {
            path.and_then(|path| Path::new(path).file_stem().and_then(|stem| stem.to_str()))
        })
        .unwrap_or("document");
    let chunk_offset = document.find(chunk).unwrap_or(document.len());
    let headings = document[..chunk_offset]
        .lines()
        .rev()
        .filter_map(|line| {
            let line = line.trim();
            line.starts_with('#')
                .then(|| line.trim_start_matches('#').trim())
                .filter(|heading| !heading.is_empty())
        })
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join(" > ");
    let heading_context = if headings.is_empty() {
        String::new()
    } else {
        format!("\nheadings: {headings}")
    };
    format!(
        "kind: documentation{}\ntitle: {title}{heading_context}\nverbatim:\n{chunk}",
        source_context(path)
    )
}

fn path_date(path: &str) -> Option<&str> {
    path.split(['/', '_']).find_map(valid_date_prefix)
}

fn valid_date_prefix(value: &str) -> Option<&str> {
    value
        .get(..10)
        .filter(|date| chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").is_ok())
}

fn split_identifier_words(identifier: &str) -> Vec<String> {
    let chars = identifier.chars().collect::<Vec<_>>();
    let mut words = Vec::new();
    let mut current = String::new();

    for (index, ch) in chars.iter().copied().enumerate() {
        if !ch.is_alphanumeric() {
            if !current.is_empty() {
                words.push(current.to_lowercase());
                current.clear();
            }
            continue;
        }

        let previous = index.checked_sub(1).and_then(|index| chars.get(index));
        let next = chars.get(index + 1);
        let starts_word = !current.is_empty()
            && ch.is_uppercase()
            && (previous.is_some_and(|previous| previous.is_lowercase() || previous.is_numeric())
                || (previous.is_some_and(|previous| previous.is_uppercase())
                    && next.is_some_and(|next| next.is_lowercase())));
        if starts_word {
            words.push(current.to_lowercase());
            current.clear();
        }
        current.push(ch);
    }

    if !current.is_empty() {
        words.push(current.to_lowercase());
    }
    words
}

fn code_language(path: &Path) -> Option<&'static str> {
    match normalized_extension(path).as_deref()? {
        ".c" | ".h" => Some("c"),
        ".cc" | ".cpp" | ".cxx" | ".hpp" => Some("cpp"),
        ".cs" => Some("csharp"),
        ".go" => Some("go"),
        ".java" => Some("java"),
        ".js" | ".jsx" | ".mjs" | ".cjs" => Some("javascript"),
        ".kt" | ".kts" => Some("kotlin"),
        ".php" => Some("php"),
        ".py" => Some("python"),
        ".rb" => Some("ruby"),
        ".rs" => Some("rust"),
        ".sh" | ".bash" | ".zsh" => Some("shell"),
        ".swift" => Some("swift"),
        ".ts" | ".tsx" => Some("typescript"),
        _ => None,
    }
}

fn content_kind_for_path(path: &Path, wing: &str, room: &str) -> ContentKind {
    let wing = wing.to_ascii_lowercase();
    let room = room.to_ascii_lowercase();
    let path_text = path.to_string_lossy().to_ascii_lowercase();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if room == "diary" || path_text.split(['/', '\\']).any(|part| part == "diary") {
        return ContentKind::Diary;
    }
    if code_language(path).is_some() {
        return ContentKind::Code;
    }
    if ["session", "conversation", "transcript", "chat"]
        .iter()
        .any(|marker| wing.contains(marker) || room.contains(marker) || path_text.contains(marker))
    {
        return ContentKind::Conversation;
    }
    let extension = normalized_extension(path).unwrap_or_default();
    if matches!(
        extension.as_str(),
        ".md" | ".mdx" | ".rst" | ".adoc" | ".asciidoc"
    ) || ["readme", "prd", "adr", "changelog"]
        .iter()
        .any(|prefix| file_name.starts_with(prefix))
    {
        return ContentKind::Documentation;
    }

    ContentKind::Prose
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

    if has_binary_extension(path) {
        return false;
    }

    is_probably_text_file(path)
}

fn is_probably_noisy_data_file(path: &Path) -> bool {
    let Some(extension) = normalized_extension(path) else {
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

fn has_binary_extension(path: &Path) -> bool {
    normalized_extension(path)
        .map(|extension| BINARY_EXTENSIONS.contains(&extension.as_str()))
        .unwrap_or(false)
}

fn normalized_extension(path: &Path) -> Option<String> {
    path.extension()
        .map(|ext| format!(".{}", ext.to_string_lossy().to_lowercase()))
}

fn is_probably_text_file(path: &Path) -> bool {
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };

    let mut sample = [0u8; TEXT_SNIFF_BYTES];
    let Ok(read) = file.read(&mut sample) else {
        return false;
    };
    let sample = &sample[..read];

    if sample.is_empty() {
        return true;
    }
    if sample.contains(&0) {
        return false;
    }

    let control_bytes = sample
        .iter()
        .filter(|byte| matches!(byte, 0x01..=0x08 | 0x0B | 0x0C | 0x0E..=0x1F))
        .count();
    if control_bytes * 100 > sample.len() * MAX_BINARY_CONTROL_RATIO_PERCENT {
        return false;
    }

    let decoded = String::from_utf8_lossy(sample);
    let decoded_chars = decoded.chars().count();
    if decoded_chars == 0 {
        return false;
    }

    let replacement_chars = decoded.chars().filter(|&ch| ch == '\u{FFFD}').count();
    replacement_chars * 100 <= decoded_chars * MAX_REPLACEMENT_RATIO_PERCENT
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

    use crate::{
        ContentKind, Drawer, DrawerMetadata, MemoryStore, Result, SearchHit, SearchQuery,
        StoreStatus,
    };

    use super::{
        MineOptions, RetrievalContext, STORE_WRITE_BATCH_SIZE, chunk_text, content_kind_for_path,
        detect_room, flush_full_drawer_batches, load_project_config, mine_project,
        retrieval_text_for_content, scan_project,
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
    fn content_kind_classifies_project_sources_conservatively() {
        assert_eq!(
            content_kind_for_path(Path::new("src/lib.rs"), "project", "src"),
            ContentKind::Code
        );
        assert_eq!(
            content_kind_for_path(Path::new("web/app.tsx"), "project", "web"),
            ContentKind::Code
        );
        assert_eq!(
            content_kind_for_path(Path::new("src/main.rs"), "chat-server", "session-runtime"),
            ContentKind::Code
        );
        assert_eq!(
            content_kind_for_path(Path::new("README.md"), "project", "general"),
            ContentKind::Documentation
        );
        assert_eq!(
            content_kind_for_path(Path::new("2026/turn.jsonl"), "codex-sessions", "2026"),
            ContentKind::Conversation
        );
        assert_eq!(
            content_kind_for_path(Path::new("2026/transcript.md"), "codex-sessions", "2026"),
            ContentKind::Conversation
        );
        assert_eq!(
            content_kind_for_path(Path::new("notes/today.txt"), "project", "notes"),
            ContentKind::Prose
        );
        assert_eq!(
            content_kind_for_path(Path::new("diary/today.txt"), "agent", "diary"),
            ContentKind::Diary
        );
    }

    #[test]
    fn non_code_retrieval_adapters_add_trustworthy_context_and_keep_verbatim_text() {
        let conversation = r#"{"role":"user","content":"I prefer compact updates."}
{"role":"assistant","content":"I will keep them concise."}"#;
        let conversation_text = retrieval_text_for_content(
            ContentKind::Conversation,
            RetrievalContext {
                path: Some("2026/turn.jsonl"),
                wing: "codex-sessions",
                room: "2026",
                agent: "codex",
                filed_at: None,
            },
            conversation,
            conversation,
        )
        .unwrap();
        assert!(conversation_text.contains("user: I prefer compact updates."));
        assert!(conversation_text.contains("assistant: I will keep them concise."));
        assert!(conversation_text.ends_with(conversation));

        let document = "# Search design\n\n## Memory budget\n\nKeep RSS below 300 MB.";
        let documentation_text = retrieval_text_for_content(
            ContentKind::Documentation,
            RetrievalContext {
                path: Some("docs/search.md"),
                wing: "project",
                room: "docs",
                agent: "codex",
                filed_at: None,
            },
            document,
            "Keep RSS below 300 MB.",
        )
        .unwrap();
        assert!(documentation_text.contains("title: Search design"));
        assert!(documentation_text.contains("headings: Search design > Memory budget"));
        assert!(documentation_text.ends_with("Keep RSS below 300 MB."));

        let diary = "AAAK: search ranking improved after the model bakeoff.";
        let diary_text = retrieval_text_for_content(
            ContentKind::Diary,
            RetrievalContext {
                path: Some("diary/2026-07-15_search.md"),
                wing: "agent",
                room: "search-quality",
                agent: "codex",
                filed_at: None,
            },
            diary,
            diary,
        )
        .unwrap();
        assert!(diary_text.contains("date: 2026-07-15"));
        assert!(diary_text.contains("agent: codex"));
        assert!(diary_text.contains("topic: search-quality"));
        assert!(diary_text.ends_with(diary));

        let manual_diary = retrieval_text_for_content(
            ContentKind::Diary,
            RetrievalContext {
                path: Some("diary/retrieval-quality"),
                wing: "wing_codex",
                room: "retrieval-quality",
                agent: "codex",
                filed_at: Some("2026-07-15T14:00:00+08:00"),
            },
            diary,
            diary,
        )
        .unwrap();
        assert!(manual_diary.contains("date: 2026-07-15"));
        assert!(manual_diary.contains("topic: retrieval-quality"));

        assert_eq!(
            retrieval_text_for_content(
                ContentKind::Unknown,
                RetrievalContext {
                    path: None,
                    wing: "legacy",
                    room: "general",
                    agent: "mcp",
                    filed_at: None,
                },
                "Neutral manual memory.",
                "Neutral manual memory.",
            )
            .unwrap(),
            "Neutral manual memory."
        );
    }

    #[tokio::test]
    async fn mine_project_enriches_markdown_session_exports_as_conversations() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        fs::write(root.join("mempalace.yml"), "wing: codex-sessions\n").unwrap();
        fs::create_dir_all(root.join("2026")).unwrap();
        let transcript = "user: I prefer concise progress updates.\nassistant: I will keep status messages brief and concrete.\n";
        fs::write(root.join("2026").join("transcript.md"), transcript).unwrap();

        let store = MockStore::default();
        mine_project(&store, root, &MineOptions::default())
            .await
            .unwrap();

        let drawers = store.list_drawers(None).await.unwrap();
        assert_eq!(drawers.len(), 1);
        assert_eq!(drawers[0].metadata.content_kind, ContentKind::Conversation);
        assert_eq!(drawers[0].content, transcript.trim());
        let retrieval_text = drawers[0].retrieval_text.as_deref().unwrap();
        assert!(retrieval_text.contains("kind: conversation"));
        assert!(retrieval_text.contains("session: codex-sessions/2026"));
        assert!(retrieval_text.ends_with(transcript.trim()));
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
        let routing = load_project_config(file.parent().unwrap())
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
            root.join("multi_domain_queries.json"),
            r#"{"benchmark_labels":["must not be mined"]}"#,
        )
        .unwrap();
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

    #[test]
    fn scan_project_includes_text_like_files_and_skips_known_binaries() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src").join("Avatar.as"),
            "package {\n  public class Avatar {}\n}\n",
        )
        .unwrap();
        fs::write(root.join("Dockerfile"), "FROM rust:1.87\n").unwrap();
        fs::write(root.join("notes.custom"), "quest=doomkitten\n").unwrap();
        fs::write(root.join("movie.swf"), b"FWS\x09\x00\x00\x00\x00\x00\x00").unwrap();
        fs::write(root.join("project.fla"), b"PK\x03\x04\x14\x00\x00\x00").unwrap();
        fs::write(root.join("blob.dat"), [0, 159, 146, 150, 0, 1, 2, 3]).unwrap();

        let files = scan_project(root, true, false, 0).unwrap();

        assert!(
            files
                .iter()
                .any(|path| path.ends_with(Path::new("src").join("Avatar.as")))
        );
        assert!(files.iter().any(|path| path.ends_with("Dockerfile")));
        assert!(files.iter().any(|path| path.ends_with("notes.custom")));
        assert!(files.iter().all(|path| !path.ends_with("movie.swf")));
        assert!(files.iter().all(|path| !path.ends_with("project.fla")));
        assert!(files.iter().all(|path| !path.ends_with("blob.dat")));
    }

    #[tokio::test]
    async fn mine_project_keeps_rust_match_arms_as_verbatim_code_units() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        fs::write(root.join("Cargo.toml"), "[package]\nname='demo'\n").unwrap();
        fs::create_dir_all(root.join("src")).unwrap();

        let first_join_arm = r#"XtRequest::FirstJoin(_) => {
                let first_join_request = FirstJoinRequest { socket_id: pid };
                emulator_client.first_join(first_join_request).await?;
            }"#;
        let source = format!(
            r#"async fn process_packet_cmd(xt_request: XtRequest) -> Result<(), XtRequestError> {{
    match xt_request {{
        XtRequest::ExecCommand(_) => {{
{}
        }},
        {}
        XtRequest::RetrieveAllUserData(_) => {{
            emulator_client.retrieve_all_user_data(pid).await?;
        }},
    }}
}}
"#,
            "            trace_packet_processing(pid);\n".repeat(40),
            first_join_arm
        );
        let file = root.join("src").join("connection.rs");
        fs::write(&file, &source).unwrap();

        let store = MockStore::default();
        mine_project(&store, root, &MineOptions::default())
            .await
            .unwrap();

        let drawers = store.list_drawers(None).await.unwrap();
        let first_join_drawer = drawers
            .iter()
            .find(|drawer| drawer.content.contains("XtRequest::FirstJoin"))
            .expect("the FirstJoin match arm should be mined");
        assert_eq!(first_join_drawer.content, first_join_arm);
        assert!(source.contains(&first_join_drawer.content));
    }

    #[tokio::test]
    async fn mine_project_adds_rust_symbol_context_to_embedding_text() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        fs::write(root.join("Cargo.toml"), "[package]\nname='demo'\n").unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        let source = r#"async fn process_packet_cmd(xt_request: XtRequest) -> Result<(), XtRequestError> {
    match xt_request {
        XtRequest::FirstJoin(_) => {
            emulator_client.first_join(FirstJoinRequest { socket_id: pid }).await?;
        }
    }
}"#;
        fs::write(root.join("src").join("connection.rs"), source).unwrap();

        let store = MockStore::default();
        mine_project(&store, root, &MineOptions::default())
            .await
            .unwrap();

        let drawers = store.list_drawers(None).await.unwrap();
        let drawer = drawers
            .iter()
            .find(|drawer| drawer.content.contains("XtRequest::FirstJoin"))
            .expect("the FirstJoin match arm should be mined");
        let retrieval_text = drawer
            .retrieval_text
            .as_deref()
            .expect("Rust chunks should carry enriched embedding text");
        assert!(retrieval_text.contains("path: src/connection.rs"));
        assert!(retrieval_text.contains("language: rust"));
        assert!(retrieval_text.contains("enclosing: process_packet_cmd"));
        assert!(retrieval_text.contains("symbol: XtRequest::FirstJoin"));
        assert!(
            retrieval_text.contains("identifier words: xt request first join process packet cmd")
        );
        assert!(retrieval_text.ends_with(&drawer.content));
    }

    #[tokio::test]
    async fn mine_project_keeps_bounded_rust_functions_as_verbatim_code_units() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        fs::write(root.join("Cargo.toml"), "[package]\nname='demo'\n").unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        let source = r#"pub fn authenticate_player(username: &str, password: &str) -> bool {
    let credentials_present = !username.is_empty() && !password.is_empty();
    credentials_present
}"#;
        fs::write(root.join("src").join("auth.rs"), source).unwrap();

        let store = MockStore::default();
        mine_project(&store, root, &MineOptions::default())
            .await
            .unwrap();

        let drawers = store.list_drawers(None).await.unwrap();
        let function = drawers
            .iter()
            .find(|drawer| drawer.content.contains("fn authenticate_player"))
            .expect("the bounded function should be mined");
        assert_eq!(function.content, source);
        let retrieval_text = function.retrieval_text.as_deref().unwrap();
        assert!(retrieval_text.contains("symbol: authenticate_player"));
        assert!(retrieval_text.contains("identifier words: authenticate player"));
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
                    retrieval_text: None,
                    metadata: DrawerMetadata {
                        content_kind: ContentKind::Prose,
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
