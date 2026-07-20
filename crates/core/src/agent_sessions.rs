use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs::{self, File},
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use async_trait::async_trait;
use chrono::Utc;
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::{
    AgentSessionsConfig, ContentKind, Drawer, DrawerMetadata, MempalaceError, Result,
    RetrievalContext, retrieval_text_for_content,
};

const CODEX_ADAPTER: &str = "codex";
const CODEX_PARSER_VERSION: u32 = 1;
const CLAUDE_ADAPTER: &str = "claude";
const CLAUDE_PARSER_VERSION: u32 = 1;
const CLAUDE_MEMORY_ADAPTER: &str = "claude-memory";
const CLAUDE_MEMORY_PARSER_VERSION: u32 = 1;
const SYNC_LEASE_TTL_MS: i64 = 5 * 60 * 1_000;
const FINGERPRINT_BYTES: u64 = 4 * 1024;
const SYNC_DRAWER_BATCH_SIZE: usize = 64;
const SYNC_RECORD_BATCH_SIZE: usize = 10_000;
const CLAUDE_COMPLETED_ROOT_LIMIT: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentSessionSelection {
    All,
    Codex,
    Claude,
}

impl AgentSessionSelection {
    fn includes_codex(self) -> bool {
        matches!(self, Self::All | Self::Codex)
    }

    fn includes_claude(self) -> bool {
        matches!(self, Self::All | Self::Claude)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AgentSessionSyncOptions {
    pub selection: AgentSessionSelection,
    pub dry_run: bool,
    pub allow_initial_backfill: bool,
}

impl Default for AgentSessionSyncOptions {
    fn default() -> Self {
        Self {
            selection: AgentSessionSelection::All,
            dry_run: false,
            allow_initial_backfill: false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AgentSessionSyncReport {
    pub lease_contended: bool,
    pub sources_missing: usize,
    pub sources_needing_backfill: usize,
    pub files_seen: usize,
    pub files_skipped: usize,
    pub files_appended: usize,
    pub files_reconciled: usize,
    pub malformed_records: usize,
    pub drawers_written: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentSessionCheckpoint {
    pub adapter: String,
    pub source_key: String,
    pub source_path: String,
    pub file_size: u64,
    pub modified_ns: i64,
    pub byte_offset: u64,
    pub fingerprint: String,
    pub parser_version: u32,
    pub cursor_json: String,
    pub updated_at: String,
}

#[derive(Debug, Clone)]
pub struct AgentSessionCommit {
    pub checkpoint: AgentSessionCheckpoint,
    pub drawers: Vec<Drawer>,
    pub replace_source: bool,
}

#[async_trait]
pub trait AgentSessionStore: Send + Sync {
    async fn agent_session_checkpoint(
        &self,
        adapter: &str,
        source_key: &str,
    ) -> Result<Option<AgentSessionCheckpoint>>;

    async fn agent_session_root_initialized(&self, adapter: &str, root: &str) -> Result<bool>;

    async fn mark_agent_session_root_initialized(&self, adapter: &str, root: &str) -> Result<()>;

    async fn commit_agent_session_sync(&self, commit: AgentSessionCommit) -> Result<usize>;

    async fn try_acquire_agent_session_lease(
        &self,
        owner: &str,
        now_ms: i64,
        expires_at_ms: i64,
    ) -> Result<bool>;

    async fn renew_agent_session_lease(&self, owner: &str, expires_at_ms: i64) -> Result<bool>;

    async fn release_agent_session_lease(&self, owner: &str) -> Result<()>;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PendingPrompt {
    text: String,
    anchor: String,
    timestamp: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SessionCursor {
    session_id: Option<String>,
    project: Option<String>,
    skip_source: bool,
    pending_codex: Vec<PendingPrompt>,
    #[serde(default)]
    claude_lineage: HashMap<String, String>,
    #[serde(default)]
    claude_prompts: HashMap<String, Vec<PendingPrompt>>,
    #[serde(default)]
    claude_completed_prompts: HashSet<String>,
    #[serde(default)]
    claude_completed_order: VecDeque<String>,
}

#[derive(Debug)]
struct ParsedExchange {
    anchor: String,
    users: Vec<String>,
    assistant: String,
    timestamp: Option<String>,
    session_id: Option<String>,
    project: Option<String>,
}

#[derive(Debug)]
struct ParseOutcome {
    cursor: SessionCursor,
    exchanges: Vec<ParsedExchange>,
    byte_offset: u64,
    malformed_records: usize,
    batch_full: bool,
}

pub async fn sync_agent_sessions<S: AgentSessionStore + ?Sized>(
    store: &S,
    config: &AgentSessionsConfig,
    options: AgentSessionSyncOptions,
) -> Result<AgentSessionSyncReport> {
    sync_agent_sessions_inner(store, config, options, false).await
}

pub async fn sync_agent_sessions_with_progress<S: AgentSessionStore + ?Sized>(
    store: &S,
    config: &AgentSessionsConfig,
    options: AgentSessionSyncOptions,
) -> Result<AgentSessionSyncReport> {
    sync_agent_sessions_inner(store, config, options, true).await
}

async fn sync_agent_sessions_inner<S: AgentSessionStore + ?Sized>(
    store: &S,
    config: &AgentSessionsConfig,
    options: AgentSessionSyncOptions,
    log_progress: bool,
) -> Result<AgentSessionSyncReport> {
    let has_codex = options.selection.includes_codex() && config.sources.codex.enabled;
    let has_claude = options.selection.includes_claude() && config.sources.claude.enabled;
    if !has_codex && !has_claude {
        return Ok(AgentSessionSyncReport::default());
    }

    let owner = Uuid::now_v7().simple().to_string();
    if !options.dry_run {
        let now_ms = Utc::now().timestamp_millis();
        if !store
            .try_acquire_agent_session_lease(&owner, now_ms, now_ms + SYNC_LEASE_TTL_MS)
            .await?
        {
            return Ok(AgentSessionSyncReport {
                lease_contended: true,
                ..AgentSessionSyncReport::default()
            });
        }
    }

    let result = sync_enabled_sources(store, config, options, &owner, log_progress).await;
    if !options.dry_run {
        let release = store.release_agent_session_lease(&owner).await;
        if result.is_ok() {
            release?;
        }
    }
    result
}

async fn sync_enabled_sources<S: AgentSessionStore + ?Sized>(
    store: &S,
    config: &AgentSessionsConfig,
    options: AgentSessionSyncOptions,
    lease_owner: &str,
    log_progress: bool,
) -> Result<AgentSessionSyncReport> {
    let mut report = AgentSessionSyncReport::default();

    if options.selection.includes_codex() && config.sources.codex.enabled {
        let root = resolve_source_root(
            config.sources.codex.path.as_deref(),
            &[".codex", "sessions"],
        )?;
        if root.is_dir() {
            let root_key = canonical_path_string(&root)?;
            let initialized = options.allow_initial_backfill
                || store
                    .agent_session_root_initialized(CODEX_ADAPTER, &root_key)
                    .await?;
            if !initialized {
                report.sources_needing_backfill += 1;
            } else {
                let files = discover_jsonl_files(&root);
                let total = files.len();
                for (index, path) in files.into_iter().enumerate() {
                    if log_progress {
                        eprintln!(
                            "mempalace: codex session {}/{}: {}",
                            index + 1,
                            total,
                            path.display()
                        );
                    }
                    sync_codex_file(
                        store,
                        &path,
                        &config.sources.codex.wing,
                        options.dry_run,
                        lease_owner,
                        log_progress,
                        &mut report,
                    )
                    .await?;
                }
                if options.allow_initial_backfill && !options.dry_run {
                    renew_sync_lease(store, lease_owner).await?;
                    store
                        .mark_agent_session_root_initialized(CODEX_ADAPTER, &root_key)
                        .await?;
                }
            }
        } else if !root.is_dir() {
            report.sources_missing += 1;
        }
    }

    if options.selection.includes_claude() && config.sources.claude.enabled {
        let root = resolve_source_root(
            config.sources.claude.path.as_deref(),
            &[".claude", "projects"],
        )?;
        if root.is_dir() {
            let root_key = canonical_path_string(&root)?;
            let sessions_initialized = options.allow_initial_backfill
                || store
                    .agent_session_root_initialized(CLAUDE_ADAPTER, &root_key)
                    .await?;
            if sessions_initialized {
                let files = discover_claude_session_files(&root)?;
                let total = files.len();
                for (index, path) in files.into_iter().enumerate() {
                    if log_progress {
                        eprintln!(
                            "mempalace: claude session {}/{}: {}",
                            index + 1,
                            total,
                            path.display()
                        );
                    }
                    sync_claude_file(
                        store,
                        &path,
                        &config.sources.claude.wing,
                        options.dry_run,
                        lease_owner,
                        log_progress,
                        &mut report,
                    )
                    .await?;
                }
                if options.allow_initial_backfill && !options.dry_run {
                    renew_sync_lease(store, lease_owner).await?;
                    store
                        .mark_agent_session_root_initialized(CLAUDE_ADAPTER, &root_key)
                        .await?;
                }
            } else {
                report.sources_needing_backfill += 1;
            }
            if config.sources.claude.include_memory
                && (options.allow_initial_backfill
                    || store
                        .agent_session_root_initialized(CLAUDE_MEMORY_ADAPTER, &root_key)
                        .await?)
            {
                let files = discover_claude_memory_files(&root);
                let total = files.len();
                for (index, path) in files.into_iter().enumerate() {
                    if log_progress {
                        eprintln!(
                            "mempalace: claude memory {}/{}: {}",
                            index + 1,
                            total,
                            path.display()
                        );
                    }
                    renew_sync_lease_unless_dry(store, lease_owner, options.dry_run).await?;
                    sync_claude_memory_file(
                        store,
                        &root,
                        &path,
                        &config.sources.claude.memory_wing,
                        options.dry_run,
                        &mut report,
                    )
                    .await?;
                }
                if options.allow_initial_backfill && !options.dry_run {
                    renew_sync_lease(store, lease_owner).await?;
                    store
                        .mark_agent_session_root_initialized(CLAUDE_MEMORY_ADAPTER, &root_key)
                        .await?;
                }
            } else if config.sources.claude.include_memory {
                report.sources_needing_backfill += 1;
            }
        } else {
            report.sources_missing += 1;
        }
    }

    Ok(report)
}

async fn renew_sync_lease<S: AgentSessionStore + ?Sized>(store: &S, owner: &str) -> Result<()> {
    let expires_at = Utc::now().timestamp_millis() + SYNC_LEASE_TTL_MS;
    if store.renew_agent_session_lease(owner, expires_at).await? {
        Ok(())
    } else {
        Err(MempalaceError::AgentSessionSync(
            "agent-session synchronization lease was lost".to_owned(),
        ))
    }
}

async fn renew_sync_lease_unless_dry<S: AgentSessionStore + ?Sized>(
    store: &S,
    owner: &str,
    dry_run: bool,
) -> Result<()> {
    if dry_run {
        Ok(())
    } else {
        renew_sync_lease(store, owner).await
    }
}

async fn sync_claude_memory_file<S: AgentSessionStore + ?Sized>(
    store: &S,
    root: &Path,
    path: &Path,
    wing: &str,
    dry_run: bool,
    report: &mut AgentSessionSyncReport,
) -> Result<()> {
    report.files_seen += 1;
    let canonical = path.canonicalize()?;
    let source_path = canonical.to_string_lossy().to_string();
    let relative = canonical
        .strip_prefix(root.canonicalize()?)
        .unwrap_or(&canonical)
        .to_string_lossy()
        .replace('\\', "/");
    let source_key = format!("claude/memory/{source_path}");
    let metadata = fs::metadata(&canonical)?;
    let file_size = metadata.len();
    let modified_ns = modified_ns(&metadata);
    let checkpoint = store
        .agent_session_checkpoint(CLAUDE_MEMORY_ADAPTER, &source_key)
        .await?;
    if checkpoint.as_ref().is_some_and(|state| {
        state.parser_version == CLAUDE_MEMORY_PARSER_VERSION
            && state.source_path == source_path
            && state.file_size == file_size
            && state.modified_ns == modified_ns
    }) {
        report.files_skipped += 1;
        return Ok(());
    }

    report.files_reconciled += 1;
    let raw = fs::read_to_string(&canonical)?;
    let body = strip_markdown_frontmatter(&raw).trim();
    let mut chunks = crate::project_miner::chunk_text(body);
    if chunks.is_empty() && !body.is_empty() {
        chunks.push(body.to_owned());
    }
    let room = relative
        .split('/')
        .next()
        .map(slugify)
        .filter(|room| !room.is_empty())
        .unwrap_or_else(|| "general".to_owned());
    let drawers = chunks
        .into_iter()
        .enumerate()
        .map(|(index, content)| {
            let stable_name = format!("{CLAUDE_MEMORY_ADAPTER}:{source_key}:{index}");
            let retrieval_text = retrieval_text_for_content(
                ContentKind::Documentation,
                RetrievalContext {
                    path: Some(&relative),
                    wing,
                    room: &room,
                    agent: CLAUDE_ADAPTER,
                    filed_at: None,
                },
                body,
                &content,
            );
            Drawer {
                id: format!(
                    "drawer_session_{}",
                    Uuid::new_v5(&Uuid::NAMESPACE_URL, stable_name.as_bytes()).simple()
                ),
                content,
                retrieval_text,
                metadata: DrawerMetadata {
                    content_kind: ContentKind::Documentation,
                    wing: wing.to_owned(),
                    room: room.clone(),
                    source_file: Some(source_path.clone()),
                    chunk_index: index as i64,
                    added_by: CLAUDE_ADAPTER.to_owned(),
                    filed_at: None,
                },
            }
        })
        .collect::<Vec<_>>();
    report.drawers_written += drawers.len();
    if dry_run {
        return Ok(());
    }

    store
        .commit_agent_session_sync(AgentSessionCommit {
            checkpoint: AgentSessionCheckpoint {
                adapter: CLAUDE_MEMORY_ADAPTER.to_owned(),
                source_key,
                source_path,
                file_size,
                modified_ns,
                byte_offset: file_size,
                fingerprint: fingerprint_before(&canonical, file_size)?,
                parser_version: CLAUDE_MEMORY_PARSER_VERSION,
                cursor_json: "{}".to_owned(),
                updated_at: Utc::now().to_rfc3339(),
            },
            drawers,
            replace_source: true,
        })
        .await?;
    Ok(())
}

async fn sync_claude_file<S: AgentSessionStore + ?Sized>(
    store: &S,
    path: &Path,
    wing: &str,
    dry_run: bool,
    lease_owner: &str,
    log_progress: bool,
    report: &mut AgentSessionSyncReport,
) -> Result<()> {
    report.files_seen += 1;
    let canonical = path.canonicalize()?;
    let source_path = canonical.to_string_lossy().to_string();
    let source_key = format!("claude/session/{source_path}");
    let metadata = fs::metadata(&canonical)?;
    let file_size = metadata.len();
    let modified_ns = modified_ns(&metadata);
    let checkpoint = store
        .agent_session_checkpoint(CLAUDE_ADAPTER, &source_key)
        .await?;

    if checkpoint.as_ref().is_some_and(|state| {
        state.parser_version == CLAUDE_PARSER_VERSION
            && state.source_path == source_path
            && state.file_size == file_size
            && state.modified_ns == modified_ns
    }) {
        report.files_skipped += 1;
        return Ok(());
    }

    let append_state = checkpoint.as_ref().filter(|state| {
        state.parser_version == CLAUDE_PARSER_VERSION
            && state.source_path == source_path
            && file_size > state.file_size
            && state.byte_offset <= file_size
            && fingerprint_before(&canonical, state.byte_offset)
                .is_ok_and(|fingerprint| fingerprint == state.fingerprint)
    });
    let (start_offset, cursor, replace_source) = if let Some(state) = append_state {
        let cursor = serde_json::from_str(&state.cursor_json).unwrap_or_default();
        report.files_appended += 1;
        (state.byte_offset, cursor, false)
    } else {
        report.files_reconciled += 1;
        (0, SessionCursor::default(), true)
    };

    let mut next_offset = start_offset;
    let mut next_cursor = cursor;
    let mut replace_batch_source = replace_source;
    loop {
        renew_sync_lease_unless_dry(store, lease_owner, dry_run).await?;
        let outcome = parse_claude_file(&canonical, next_offset, next_cursor)?;
        report.malformed_records += outcome.malformed_records;
        let batch_full = outcome.batch_full;
        let byte_offset = outcome.byte_offset;
        let cursor_json = serde_json::to_string(&outcome.cursor)?;
        let drawers = outcome
            .exchanges
            .into_iter()
            .map(|exchange| exchange_drawer(CLAUDE_ADAPTER, wing, &source_path, exchange))
            .collect::<Vec<_>>();
        report.drawers_written += drawers.len();
        if log_progress && batch_full {
            eprintln!("mempalace: claude session progress: {byte_offset}/{file_size} bytes");
        }

        if !dry_run {
            renew_sync_lease(store, lease_owner).await?;
            store
                .commit_agent_session_sync(AgentSessionCommit {
                    checkpoint: AgentSessionCheckpoint {
                        adapter: CLAUDE_ADAPTER.to_owned(),
                        source_key: source_key.clone(),
                        source_path: source_path.clone(),
                        file_size: if batch_full { byte_offset } else { file_size },
                        modified_ns,
                        byte_offset,
                        fingerprint: fingerprint_before(&canonical, byte_offset)?,
                        parser_version: CLAUDE_PARSER_VERSION,
                        cursor_json,
                        updated_at: Utc::now().to_rfc3339(),
                    },
                    drawers,
                    replace_source: replace_batch_source,
                })
                .await?;
        }
        if !batch_full {
            break;
        }
        next_offset = byte_offset;
        next_cursor = outcome.cursor;
        replace_batch_source = false;
    }
    Ok(())
}

async fn sync_codex_file<S: AgentSessionStore + ?Sized>(
    store: &S,
    path: &Path,
    wing: &str,
    dry_run: bool,
    lease_owner: &str,
    log_progress: bool,
    report: &mut AgentSessionSyncReport,
) -> Result<()> {
    report.files_seen += 1;
    let canonical = path.canonicalize()?;
    let source_path = canonical.to_string_lossy().to_string();
    let source_key = source_path.clone();
    let metadata = fs::metadata(&canonical)?;
    let file_size = metadata.len();
    let modified_ns = modified_ns(&metadata);
    let checkpoint = store
        .agent_session_checkpoint(CODEX_ADAPTER, &source_key)
        .await?;

    if checkpoint.as_ref().is_some_and(|state| {
        state.parser_version == CODEX_PARSER_VERSION
            && state.source_path == source_path
            && state.file_size == file_size
            && state.modified_ns == modified_ns
    }) {
        report.files_skipped += 1;
        return Ok(());
    }

    let append_state = checkpoint.as_ref().filter(|state| {
        state.parser_version == CODEX_PARSER_VERSION
            && state.source_path == source_path
            && file_size > state.file_size
            && state.byte_offset <= file_size
            && fingerprint_before(&canonical, state.byte_offset)
                .is_ok_and(|fingerprint| fingerprint == state.fingerprint)
    });
    let (start_offset, cursor, replace_source) = if let Some(state) = append_state {
        let cursor = serde_json::from_str(&state.cursor_json).unwrap_or_default();
        report.files_appended += 1;
        (state.byte_offset, cursor, false)
    } else {
        report.files_reconciled += 1;
        (0, SessionCursor::default(), true)
    };

    let mut next_offset = start_offset;
    let mut next_cursor = cursor;
    let mut replace_batch_source = replace_source;
    loop {
        renew_sync_lease_unless_dry(store, lease_owner, dry_run).await?;
        let outcome = parse_codex_file(&canonical, next_offset, next_cursor)?;
        report.malformed_records += outcome.malformed_records;
        let batch_full = outcome.batch_full;
        let byte_offset = outcome.byte_offset;
        let cursor_json = serde_json::to_string(&outcome.cursor)?;
        let drawers = if outcome.cursor.skip_source {
            Vec::new()
        } else {
            outcome
                .exchanges
                .into_iter()
                .map(|exchange| exchange_drawer(CODEX_ADAPTER, wing, &source_path, exchange))
                .collect::<Vec<_>>()
        };
        report.drawers_written += drawers.len();
        if log_progress && batch_full {
            eprintln!("mempalace: codex session progress: {byte_offset}/{file_size} bytes");
        }

        if !dry_run {
            renew_sync_lease(store, lease_owner).await?;
            store
                .commit_agent_session_sync(AgentSessionCommit {
                    checkpoint: AgentSessionCheckpoint {
                        adapter: CODEX_ADAPTER.to_owned(),
                        source_key: source_key.clone(),
                        source_path: source_path.clone(),
                        file_size: if batch_full { byte_offset } else { file_size },
                        modified_ns,
                        byte_offset,
                        fingerprint: fingerprint_before(&canonical, byte_offset)?,
                        parser_version: CODEX_PARSER_VERSION,
                        cursor_json,
                        updated_at: Utc::now().to_rfc3339(),
                    },
                    drawers,
                    replace_source: replace_batch_source,
                })
                .await?;
        }
        if !batch_full {
            break;
        }
        next_offset = byte_offset;
        next_cursor = outcome.cursor;
        replace_batch_source = false;
    }
    Ok(())
}

fn parse_codex_file(
    path: &Path,
    start_offset: u64,
    mut cursor: SessionCursor,
) -> Result<ParseOutcome> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(start_offset))?;
    let mut reader = BufReader::new(file);
    let mut offset = start_offset;
    let mut exchanges = Vec::new();
    let mut malformed_records = 0;
    let mut records_read = 0;
    let mut batch_full = false;

    loop {
        let record_offset = offset;
        let mut line = Vec::new();
        let read = reader.read_until(b'\n', &mut line)?;
        if read == 0 {
            break;
        }
        if line.last() != Some(&b'\n') {
            break;
        }
        offset += read as u64;
        while matches!(line.last(), Some(b'\n' | b'\r')) {
            line.pop();
        }
        if line.is_empty() {
            continue;
        }
        records_read += 1;
        let record: Value = match serde_json::from_slice(&line) {
            Ok(record) => record,
            Err(_) => {
                malformed_records += 1;
                if records_read >= SYNC_RECORD_BATCH_SIZE {
                    batch_full = true;
                    break;
                }
                continue;
            }
        };
        ingest_codex_record(&record, record_offset, &mut cursor, &mut exchanges);
        if exchanges.len() >= SYNC_DRAWER_BATCH_SIZE || records_read >= SYNC_RECORD_BATCH_SIZE {
            batch_full = true;
            break;
        }
    }

    Ok(ParseOutcome {
        cursor,
        exchanges,
        byte_offset: offset,
        malformed_records,
        batch_full,
    })
}

fn parse_claude_file(
    path: &Path,
    start_offset: u64,
    mut cursor: SessionCursor,
) -> Result<ParseOutcome> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(start_offset))?;
    let mut reader = BufReader::new(file);
    let mut offset = start_offset;
    let mut exchanges = Vec::new();
    let mut malformed_records = 0;
    let mut records_read = 0;
    let mut batch_full = false;

    loop {
        let record_offset = offset;
        let mut line = Vec::new();
        let read = reader.read_until(b'\n', &mut line)?;
        if read == 0 {
            break;
        }
        if line.last() != Some(&b'\n') {
            break;
        }
        offset += read as u64;
        while matches!(line.last(), Some(b'\n' | b'\r')) {
            line.pop();
        }
        if line.is_empty() {
            continue;
        }
        records_read += 1;
        let record: Value = match serde_json::from_slice(&line) {
            Ok(record) => record,
            Err(_) => {
                malformed_records += 1;
                if records_read >= SYNC_RECORD_BATCH_SIZE {
                    batch_full = true;
                    break;
                }
                continue;
            }
        };
        ingest_claude_record(&record, record_offset, &mut cursor, &mut exchanges);
        if exchanges.len() >= SYNC_DRAWER_BATCH_SIZE || records_read >= SYNC_RECORD_BATCH_SIZE {
            batch_full = true;
            break;
        }
    }

    Ok(ParseOutcome {
        cursor,
        exchanges,
        byte_offset: offset,
        malformed_records,
        batch_full,
    })
}

fn ingest_codex_record(
    record: &Value,
    record_offset: u64,
    cursor: &mut SessionCursor,
    exchanges: &mut Vec<ParsedExchange>,
) {
    let record_type = record.get("type").and_then(Value::as_str);
    let payload = record.get("payload").unwrap_or(&Value::Null);
    let timestamp = record
        .get("timestamp")
        .and_then(Value::as_str)
        .map(str::to_owned);

    if record_type == Some("session_meta") && cursor.session_id.is_none() {
        cursor.session_id = payload.get("id").and_then(Value::as_str).map(str::to_owned);
        cursor.project = payload
            .get("cwd")
            .and_then(Value::as_str)
            .map(str::to_owned);
        cursor.skip_source = payload.get("thread_source").and_then(Value::as_str)
            == Some("subagent")
            || payload
                .get("source")
                .and_then(|source| source.get("subagent"))
                .is_some_and(|subagent| !subagent.is_null());
        return;
    }
    if cursor.skip_source {
        return;
    }

    if record_type == Some("event_msg")
        && payload.get("type").and_then(Value::as_str) == Some("user_message")
    {
        if let Some(text) = payload
            .get("message")
            .and_then(Value::as_str)
            .and_then(normalize_text)
        {
            let anchor = payload
                .get("client_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| record_offset.to_string());
            cursor.pending_codex.push(PendingPrompt {
                text,
                anchor,
                timestamp,
            });
        }
        return;
    }

    let is_final = record_type == Some("response_item")
        && payload.get("type").and_then(Value::as_str) == Some("message")
        && payload.get("role").and_then(Value::as_str) == Some("assistant")
        && payload.get("phase").and_then(Value::as_str) == Some("final_answer");
    if !is_final || cursor.pending_codex.is_empty() {
        return;
    }

    let assistant = payload
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .filter_map(normalize_text)
        .collect::<Vec<_>>()
        .join("\n");
    if assistant.is_empty() {
        return;
    }

    let pending = std::mem::take(&mut cursor.pending_codex);
    let anchor = pending
        .first()
        .map(|prompt| prompt.anchor.clone())
        .unwrap_or_else(|| record_offset.to_string());
    let user_timestamp = pending.first().and_then(|prompt| prompt.timestamp.clone());
    exchanges.push(ParsedExchange {
        anchor,
        users: pending.into_iter().map(|prompt| prompt.text).collect(),
        assistant,
        timestamp: timestamp.or(user_timestamp),
        session_id: cursor.session_id.clone(),
        project: cursor.project.clone(),
    });
}

fn ingest_claude_record(
    record: &Value,
    record_offset: u64,
    cursor: &mut SessionCursor,
    exchanges: &mut Vec<ParsedExchange>,
) {
    if record.get("isSidechain").and_then(Value::as_bool) == Some(true)
        || record.get("agentId").is_some_and(|agent| !agent.is_null())
    {
        return;
    }

    if cursor.session_id.is_none() {
        cursor.session_id = record
            .get("sessionId")
            .or_else(|| record.get("session_id"))
            .and_then(Value::as_str)
            .map(str::to_owned);
    }
    if cursor.project.is_none() {
        cursor.project = record.get("cwd").and_then(Value::as_str).map(str::to_owned);
    }

    let record_type = record.get("type").and_then(Value::as_str);
    let uuid = record.get("uuid").and_then(Value::as_str);
    let parent_uuid = record.get("parentUuid").and_then(Value::as_str);
    let inherited_user = parent_uuid
        .and_then(|parent| cursor.claude_lineage.get(parent))
        .cloned();
    let message = record.get("message").unwrap_or(&Value::Null);
    let genuine_user = record_type == Some("user")
        && message.get("role").and_then(Value::as_str) == Some("user")
        && record.get("isMeta").and_then(Value::as_bool) != Some(true)
        && record
            .get("toolUseResult")
            .is_none_or(|result| result.is_null())
        && record
            .get("sourceToolAssistantUUID")
            .is_none_or(|source| source.is_null());

    if genuine_user {
        if let (Some(uuid), Some(text)) = (uuid, claude_message_text(message)) {
            let prompt_root = inherited_user
                .as_ref()
                .filter(|root| !cursor.claude_completed_prompts.contains(*root))
                .cloned()
                .unwrap_or_else(|| uuid.to_owned());
            cursor
                .claude_lineage
                .insert(uuid.to_owned(), prompt_root.clone());
            cursor
                .claude_prompts
                .entry(prompt_root)
                .or_default()
                .push(PendingPrompt {
                    text,
                    anchor: uuid.to_owned(),
                    timestamp: record
                        .get("timestamp")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                });
        }
        return;
    }

    if let (Some(uuid), Some(user_uuid)) = (uuid, inherited_user.as_deref()) {
        cursor
            .claude_lineage
            .insert(uuid.to_owned(), user_uuid.to_owned());
    }

    let is_final = record_type == Some("assistant")
        && message.get("role").and_then(Value::as_str) == Some("assistant")
        && message.get("stop_reason").and_then(Value::as_str) == Some("end_turn");
    if !is_final {
        return;
    }
    let Some(assistant) = claude_message_text(message) else {
        return;
    };
    let Some(user_uuid) = inherited_user else {
        return;
    };
    let Some(prompts) = cursor.claude_prompts.get(&user_uuid).cloned() else {
        return;
    };
    let message_id = message
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("offset-{record_offset}"));

    let anchor = prompts
        .first()
        .map(|prompt| prompt.anchor.clone())
        .unwrap_or_else(|| record_offset.to_string());
    let prompt_timestamp = prompts.first().and_then(|prompt| prompt.timestamp.clone());

    exchanges.push(ParsedExchange {
        anchor: format!("{anchor}:{message_id}"),
        users: prompts.into_iter().map(|prompt| prompt.text).collect(),
        assistant,
        timestamp: record
            .get("timestamp")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or(prompt_timestamp),
        session_id: cursor.session_id.clone(),
        project: cursor.project.clone(),
    });
    mark_claude_prompt_completed(cursor, &user_uuid);
}

fn mark_claude_prompt_completed(cursor: &mut SessionCursor, root: &str) {
    if cursor.claude_completed_prompts.insert(root.to_owned()) {
        cursor.claude_completed_order.push_back(root.to_owned());
    }

    let prompt_anchors = cursor
        .claude_prompts
        .get(root)
        .into_iter()
        .flatten()
        .map(|prompt| prompt.anchor.clone())
        .collect::<HashSet<_>>();
    cursor
        .claude_lineage
        .retain(|uuid, mapped_root| mapped_root != root || prompt_anchors.contains(uuid));

    while cursor.claude_completed_order.len() > CLAUDE_COMPLETED_ROOT_LIMIT {
        let Some(expired_root) = cursor.claude_completed_order.pop_front() else {
            break;
        };
        cursor.claude_completed_prompts.remove(&expired_root);
        cursor.claude_prompts.remove(&expired_root);
        cursor
            .claude_lineage
            .retain(|_, mapped_root| mapped_root != &expired_root);
    }
}

fn claude_message_text(message: &Value) -> Option<String> {
    let content = message.get("content")?;
    let text = if let Some(text) = content.as_str() {
        text.to_owned()
    } else {
        content
            .as_array()?
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .filter_map(normalize_text)
            .collect::<Vec<_>>()
            .join("\n")
    };
    normalize_text(&text)
}

fn exchange_drawer(
    adapter: &str,
    wing: &str,
    source_path: &str,
    exchange: ParsedExchange,
) -> Drawer {
    let session_id = exchange
        .session_id
        .clone()
        .unwrap_or_else(|| "unknown-session".to_owned());
    let stable_name = format!("{adapter}:{source_path}:{session_id}:{}", exchange.anchor);
    let id = format!(
        "drawer_session_{}",
        Uuid::new_v5(&Uuid::NAMESPACE_URL, stable_name.as_bytes()).simple()
    );
    let mut content_lines = exchange
        .users
        .iter()
        .map(|user| format!("user: {user}"))
        .collect::<Vec<_>>();
    content_lines.push(format!("assistant: {}", exchange.assistant));
    let content = content_lines.join("\n");
    let room = exchange
        .project
        .as_deref()
        .and_then(project_name)
        .map(slugify)
        .filter(|room| !room.is_empty())
        .unwrap_or_else(|| "general".to_owned());
    let date = exchange
        .timestamp
        .as_deref()
        .and_then(|timestamp| timestamp.get(..10));
    let mut context = vec![
        "kind: conversation".to_owned(),
        format!("agent: {adapter}"),
        format!("session: {session_id}"),
    ];
    if let Some(project) = exchange.project.as_deref() {
        context.push(format!("project: {project}"));
    }
    if let Some(date) = date {
        context.push(format!("date: {date}"));
    }
    context.push(content.clone());

    Drawer {
        id,
        content,
        retrieval_text: Some(context.join("\n")),
        metadata: DrawerMetadata {
            content_kind: ContentKind::Conversation,
            wing: wing.to_owned(),
            room,
            source_file: Some(source_path.to_owned()),
            chunk_index: 0,
            added_by: adapter.to_owned(),
            filed_at: exchange.timestamp,
        },
    }
}

fn normalize_text(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

fn project_name(path: &str) -> Option<&str> {
    path.rsplit(['/', '\\']).find(|part| !part.is_empty())
}

fn slugify(value: &str) -> String {
    let mut slug = String::new();
    let mut separated = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
            separated = false;
        } else if !separated && !slug.is_empty() {
            slug.push('-');
            separated = true;
        }
    }
    slug.trim_end_matches('-').to_owned()
}

fn resolve_source_root(configured: Option<&Path>, default_parts: &[&str]) -> Result<PathBuf> {
    if let Some(configured) = configured {
        let text = configured.to_string_lossy();
        if text == "~" || text.starts_with("~/") || text.starts_with("~\\") {
            let home = dirs::home_dir().ok_or(MempalaceError::MissingHomeDirectory)?;
            let suffix = text.trim_start_matches('~').trim_start_matches(['/', '\\']);
            return Ok(home.join(suffix));
        }
        return Ok(configured.to_path_buf());
    }

    let mut path = dirs::home_dir().ok_or(MempalaceError::MissingHomeDirectory)?;
    for part in default_parts {
        path.push(part);
    }
    Ok(path)
}

fn canonical_path_string(path: &Path) -> Result<String> {
    Ok(path.canonicalize()?.to_string_lossy().to_string())
}

fn discover_jsonl_files(root: &Path) -> Vec<PathBuf> {
    let mut files = WalkBuilder::new(root)
        .hidden(false)
        .follow_links(false)
        .build()
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_some_and(|kind| kind.is_file()))
        .map(|entry| entry.into_path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("jsonl"))
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn discover_claude_session_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for project in fs::read_dir(root)? {
        let project = project?;
        if !project.file_type()?.is_dir() {
            continue;
        }
        for entry in fs::read_dir(project.path())? {
            let entry = entry?;
            if entry.file_type()?.is_file()
                && entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("jsonl"))
            {
                files.push(entry.path());
            }
        }
    }
    files.sort();
    Ok(files)
}

fn discover_claude_memory_files(root: &Path) -> Vec<PathBuf> {
    let mut files = WalkBuilder::new(root)
        .hidden(false)
        .follow_links(false)
        .build()
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_some_and(|kind| kind.is_file()))
        .map(|entry| entry.into_path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
                && path
                    .parent()
                    .and_then(Path::file_name)
                    .is_some_and(|name| name == "memory")
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn strip_markdown_frontmatter(content: &str) -> &str {
    if let Some(rest) = content.strip_prefix("---\n") {
        return rest
            .find("\n---\n")
            .map(|end| &rest[end + "\n---\n".len()..])
            .unwrap_or(content);
    }
    if let Some(rest) = content.strip_prefix("---\r\n") {
        return rest
            .find("\r\n---\r\n")
            .map(|end| &rest[end + "\r\n---\r\n".len()..])
            .unwrap_or(content);
    }
    content
}

fn modified_ns(metadata: &fs::Metadata) -> i64 {
    metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos().min(i64::MAX as u128) as i64)
        .unwrap_or_default()
}

fn fingerprint_before(path: &Path, offset: u64) -> Result<String> {
    let mut file = File::open(path)?;
    let start = offset.saturating_sub(FINGERPRINT_BYTES);
    file.seek(SeekFrom::Start(start))?;
    let mut remaining = (offset - start) as usize;
    let mut buffer = vec![0_u8; remaining];
    let mut read = 0;
    while remaining > 0 {
        let count = file.read(&mut buffer[read..])?;
        if count == 0 {
            break;
        }
        read += count;
        remaining -= count;
    }
    buffer.truncate(read);
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in buffer {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(format!("{hash:016x}"))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        fs,
        io::Write,
        sync::{Arc, Mutex},
    };

    use async_trait::async_trait;
    use tempfile::tempdir;

    use super::{
        AgentSessionCheckpoint, AgentSessionCommit, AgentSessionSelection, AgentSessionStore,
        AgentSessionSyncOptions, sync_agent_sessions,
    };
    use crate::{AgentSessionsConfig, Drawer, Result};

    #[derive(Clone, Default)]
    struct TestStore {
        checkpoints: Arc<Mutex<HashMap<(String, String), AgentSessionCheckpoint>>>,
        drawers: Arc<Mutex<Vec<Drawer>>>,
        roots: Arc<Mutex<HashSet<(String, String)>>>,
        commits: Arc<Mutex<usize>>,
    }

    #[async_trait]
    impl AgentSessionStore for TestStore {
        async fn agent_session_checkpoint(
            &self,
            adapter: &str,
            source_key: &str,
        ) -> Result<Option<AgentSessionCheckpoint>> {
            Ok(self
                .checkpoints
                .lock()
                .unwrap()
                .get(&(adapter.to_owned(), source_key.to_owned()))
                .cloned())
        }

        async fn agent_session_root_initialized(&self, adapter: &str, root: &str) -> Result<bool> {
            Ok(self
                .roots
                .lock()
                .unwrap()
                .contains(&(adapter.to_owned(), root.to_owned())))
        }

        async fn mark_agent_session_root_initialized(
            &self,
            adapter: &str,
            root: &str,
        ) -> Result<()> {
            self.roots
                .lock()
                .unwrap()
                .insert((adapter.to_owned(), root.to_owned()));
            Ok(())
        }

        async fn commit_agent_session_sync(&self, commit: AgentSessionCommit) -> Result<usize> {
            *self.commits.lock().unwrap() += 1;
            let written = commit.drawers.len();
            let mut drawers = self.drawers.lock().unwrap();
            if commit.replace_source {
                drawers.retain(|drawer| {
                    drawer.metadata.source_file.as_deref()
                        != Some(commit.checkpoint.source_path.as_str())
                });
            }
            for incoming in commit.drawers {
                if let Some(existing) = drawers.iter_mut().find(|drawer| drawer.id == incoming.id) {
                    *existing = incoming;
                } else {
                    drawers.push(incoming);
                }
            }
            self.checkpoints.lock().unwrap().insert(
                (
                    commit.checkpoint.adapter.clone(),
                    commit.checkpoint.source_key.clone(),
                ),
                commit.checkpoint,
            );
            Ok(written)
        }

        async fn try_acquire_agent_session_lease(
            &self,
            _owner: &str,
            _now_ms: i64,
            _expires_at_ms: i64,
        ) -> Result<bool> {
            Ok(true)
        }

        async fn renew_agent_session_lease(
            &self,
            _owner: &str,
            _expires_at_ms: i64,
        ) -> Result<bool> {
            Ok(true)
        }

        async fn release_agent_session_lease(&self, _owner: &str) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn codex_sync_stores_only_user_and_final_answer() {
        let tmp = tempdir().unwrap();
        let sessions = tmp.path().join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let session_file = sessions.join("session.jsonl");
        fs::write(
            &session_file,
            concat!(
                "{\"timestamp\":\"2026-07-15T01:00:00Z\",\"type\":\"session_meta\",\"payload\":{\"id\":\"session-1\",\"cwd\":\"/workspace/demo\"}}\n",
                "{\"timestamp\":\"2026-07-15T01:01:00Z\",\"type\":\"event_msg\",\"payload\":{\"type\":\"user_message\",\"message\":\"How should sync resume?\"}}\n",
                "{\"timestamp\":\"2026-07-15T01:01:01Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"reasoning\",\"text\":\"private reasoning\"}}\n",
                "{\"timestamp\":\"2026-07-15T01:01:02Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"phase\":\"commentary\",\"content\":[{\"type\":\"output_text\",\"text\":\"still working\"}]}}\n",
                "{\"timestamp\":\"2026-07-15T01:01:03Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"function_call_output\",\"output\":\"tool secret\"}}\n",
                "{\"timestamp\":\"2026-07-15T01:01:04Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"developer\",\"content\":[{\"type\":\"input_text\",\"text\":\"developer secret\"}]}}\n",
                "{\"timestamp\":\"2026-07-15T01:01:05Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"injected project instructions\"}]}}\n",
                "{\"timestamp\":\"2026-07-15T01:02:00Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"phase\":\"final_answer\",\"content\":[{\"type\":\"output_text\",\"text\":\"Resume from a committed byte offset.\"}]}}\n"
            ),
        )
        .unwrap();

        let mut config = AgentSessionsConfig::default();
        config.sources.codex.enabled = true;
        config.sources.codex.path = Some(sessions);
        let store = TestStore::default();

        let report = sync_agent_sessions(
            &store,
            &config,
            AgentSessionSyncOptions {
                selection: AgentSessionSelection::Codex,
                dry_run: false,
                allow_initial_backfill: true,
            },
        )
        .await
        .unwrap();

        assert_eq!(report.drawers_written, 1);
        let drawers = store.drawers.lock().unwrap();
        assert_eq!(drawers.len(), 1);
        assert_eq!(
            drawers[0].content,
            "user: How should sync resume?\nassistant: Resume from a committed byte offset."
        );
        assert!(!drawers[0].content.contains("private reasoning"));
        assert!(!drawers[0].content.contains("still working"));
        assert!(!drawers[0].content.contains("tool secret"));
        assert!(!drawers[0].content.contains("developer secret"));
        assert!(!drawers[0].content.contains("injected project instructions"));
    }

    #[tokio::test]
    async fn codex_sync_skips_unchanged_files_and_resumes_pending_turns() {
        let tmp = tempdir().unwrap();
        let sessions = tmp.path().join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let session_file = sessions.join("session.jsonl");
        fs::write(
            &session_file,
            concat!(
                "{\"timestamp\":\"2026-07-15T01:00:00Z\",\"type\":\"session_meta\",\"payload\":{\"id\":\"session-2\",\"cwd\":\"/workspace/demo\"}}\n",
                "{\"timestamp\":\"2026-07-15T01:01:00Z\",\"type\":\"event_msg\",\"payload\":{\"type\":\"user_message\",\"client_id\":\"prompt-1\",\"message\":\"Will this resume later?\"}}\n"
            ),
        )
        .unwrap();

        let mut config = AgentSessionsConfig::default();
        config.sources.codex.enabled = true;
        config.sources.codex.path = Some(sessions);
        let store = TestStore::default();
        let options = AgentSessionSyncOptions {
            selection: AgentSessionSelection::Codex,
            dry_run: false,
            allow_initial_backfill: true,
        };

        let first = sync_agent_sessions(&store, &config, options).await.unwrap();
        assert_eq!(first.drawers_written, 0);
        assert!(store.drawers.lock().unwrap().is_empty());

        let unchanged = sync_agent_sessions(&store, &config, options).await.unwrap();
        assert_eq!(unchanged.files_skipped, 1);
        assert_eq!(unchanged.drawers_written, 0);

        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&session_file)
            .unwrap();
        file.write_all(
            concat!(
                r#"{"timestamp":"2026-07-15T01:02:00Z","type":"response_item","payload":{"type":"message","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"Yes, from the committed cursor."}]}}"#,
                "\n"
            )
            .as_bytes(),
        )
        .unwrap();
        file.flush().unwrap();

        let appended = sync_agent_sessions(&store, &config, options).await.unwrap();
        assert_eq!(appended.files_appended, 1);
        assert_eq!(appended.drawers_written, 1);
        let drawers = store.drawers.lock().unwrap();
        assert_eq!(drawers.len(), 1);
        assert_eq!(
            drawers[0].content,
            "user: Will this resume later?\nassistant: Yes, from the committed cursor."
        );
    }

    #[tokio::test]
    async fn background_mode_requires_a_manual_initial_backfill() {
        let tmp = tempdir().unwrap();
        let sessions = tmp.path().join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        fs::write(sessions.join("session.jsonl"), "{}\n").unwrap();
        let mut config = AgentSessionsConfig::default();
        config.sources.codex.enabled = true;
        config.sources.codex.path = Some(sessions);
        let store = TestStore::default();

        let report = sync_agent_sessions(
            &store,
            &config,
            AgentSessionSyncOptions {
                selection: AgentSessionSelection::Codex,
                dry_run: false,
                allow_initial_backfill: false,
            },
        )
        .await
        .unwrap();

        assert_eq!(report.sources_needing_backfill, 1);
        assert_eq!(report.files_seen, 0);
        assert!(store.checkpoints.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn codex_sync_waits_for_a_complete_trailing_json_line() {
        let tmp = tempdir().unwrap();
        let sessions = tmp.path().join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let session_file = sessions.join("session.jsonl");
        let prefix = concat!(
            "{\"timestamp\":\"2026-07-15T01:00:00Z\",\"type\":\"session_meta\",\"payload\":{\"id\":\"session-partial\",\"cwd\":\"/workspace/demo\"}}\n",
            "{\"timestamp\":\"2026-07-15T01:01:00Z\",\"type\":\"event_msg\",\"payload\":{\"type\":\"user_message\",\"message\":\"Is the record complete?\"}}\n"
        );
        let final_line = concat!(
            "{\"timestamp\":\"2026-07-15T01:02:00Z\",\"type\":\"response_item\",",
            "\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"phase\":\"final_answer\",",
            "\"content\":[{\"type\":\"output_text\",\"text\":\"It is complete now.\"}]}}\n"
        );
        let split = final_line.len() / 2;
        fs::write(&session_file, format!("{prefix}{}", &final_line[..split])).unwrap();

        let mut config = AgentSessionsConfig::default();
        config.sources.codex.enabled = true;
        config.sources.codex.path = Some(sessions);
        let store = TestStore::default();
        let options = AgentSessionSyncOptions {
            selection: AgentSessionSelection::Codex,
            dry_run: false,
            allow_initial_backfill: true,
        };

        let partial = sync_agent_sessions(&store, &config, options).await.unwrap();
        assert_eq!(partial.drawers_written, 0);
        let checkpoint = store
            .checkpoints
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .clone();
        assert_eq!(checkpoint.byte_offset, prefix.len() as u64);

        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&session_file)
            .unwrap();
        file.write_all(&final_line.as_bytes()[split..]).unwrap();
        file.flush().unwrap();

        let completed = sync_agent_sessions(&store, &config, options).await.unwrap();
        assert_eq!(completed.files_appended, 1);
        assert_eq!(completed.drawers_written, 1);
        assert_eq!(store.drawers.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn codex_sync_commits_large_sources_in_bounded_batches() {
        let tmp = tempdir().unwrap();
        let sessions = tmp.path().join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let mut transcript = String::from(
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"batched\",\"cwd\":\"/workspace/demo\"}}\n",
        );
        for index in 0..65 {
            transcript.push_str(&format!(
                "{{\"type\":\"event_msg\",\"payload\":{{\"type\":\"user_message\",\"client_id\":\"prompt-{index}\",\"message\":\"question {index}\"}}}}\n"
            ));
            transcript.push_str(&format!(
                "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"assistant\",\"phase\":\"final_answer\",\"content\":[{{\"type\":\"output_text\",\"text\":\"answer {index}\"}}]}}}}\n"
            ));
        }
        fs::write(sessions.join("session.jsonl"), transcript).unwrap();

        let mut config = AgentSessionsConfig::default();
        config.sources.codex.enabled = true;
        config.sources.codex.path = Some(sessions);
        let store = TestStore::default();

        let report = sync_agent_sessions(
            &store,
            &config,
            AgentSessionSyncOptions {
                selection: AgentSessionSelection::Codex,
                dry_run: false,
                allow_initial_backfill: true,
            },
        )
        .await
        .unwrap();

        assert_eq!(report.drawers_written, 65);
        assert_eq!(store.drawers.lock().unwrap().len(), 65);
        assert_eq!(*store.commits.lock().unwrap(), 2);
    }

    #[tokio::test]
    async fn background_authorization_is_scoped_to_the_configured_root() {
        let tmp = tempdir().unwrap();
        let first_root = tmp.path().join("first");
        let second_root = tmp.path().join("second");
        fs::create_dir_all(&first_root).unwrap();
        fs::create_dir_all(&second_root).unwrap();
        fs::write(first_root.join("session.jsonl"), "{}\n").unwrap();
        fs::write(second_root.join("session.jsonl"), "{}\n").unwrap();
        let mut config = AgentSessionsConfig::default();
        config.sources.codex.enabled = true;
        config.sources.codex.path = Some(first_root);
        let store = TestStore::default();

        sync_agent_sessions(
            &store,
            &config,
            AgentSessionSyncOptions {
                selection: AgentSessionSelection::Codex,
                dry_run: false,
                allow_initial_backfill: true,
            },
        )
        .await
        .unwrap();

        config.sources.codex.path = Some(second_root);
        let report = sync_agent_sessions(
            &store,
            &config,
            AgentSessionSyncOptions {
                selection: AgentSessionSelection::Codex,
                dry_run: false,
                allow_initial_backfill: false,
            },
        )
        .await
        .unwrap();

        assert_eq!(report.sources_needing_backfill, 1);
        assert_eq!(report.files_seen, 0);
    }

    #[tokio::test]
    async fn codex_reconciliation_replaces_only_the_changed_source() {
        let tmp = tempdir().unwrap();
        let sessions = tmp.path().join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let first = sessions.join("first.jsonl");
        let second = sessions.join("second.jsonl");
        let transcript = |session: &str, answer: &str| {
            format!(
                "{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"{session}\",\"cwd\":\"/workspace/demo\"}}}}\n{{\"type\":\"event_msg\",\"payload\":{{\"type\":\"user_message\",\"message\":\"question for {session}\"}}}}\n{{\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"assistant\",\"phase\":\"final_answer\",\"content\":[{{\"type\":\"output_text\",\"text\":\"{answer}\"}}]}}}}\n"
            )
        };
        fs::write(&first, transcript("first", "old first answer")).unwrap();
        fs::write(&second, transcript("second", "stable second answer")).unwrap();
        let mut config = AgentSessionsConfig::default();
        config.sources.codex.enabled = true;
        config.sources.codex.path = Some(sessions);
        let store = TestStore::default();
        let options = AgentSessionSyncOptions {
            selection: AgentSessionSelection::Codex,
            dry_run: false,
            allow_initial_backfill: true,
        };

        sync_agent_sessions(&store, &config, options).await.unwrap();
        fs::write(
            &first,
            transcript("first", "new and substantially longer first answer"),
        )
        .unwrap();
        sync_agent_sessions(&store, &config, options).await.unwrap();

        let drawers = store.drawers.lock().unwrap();
        assert_eq!(drawers.len(), 2);
        assert!(
            drawers
                .iter()
                .any(|drawer| drawer.content.contains("new and substantially longer"))
        );
        assert!(
            drawers
                .iter()
                .any(|drawer| drawer.content.contains("stable second answer"))
        );
        assert!(
            drawers
                .iter()
                .all(|drawer| !drawer.content.contains("old first answer"))
        );
    }

    #[tokio::test]
    async fn claude_sync_follows_lineage_and_keeps_only_end_turn_text() {
        let tmp = tempdir().unwrap();
        let projects = tmp.path().join("projects");
        let project = projects.join("-workspace-demo");
        fs::create_dir_all(&project).unwrap();
        fs::write(
            project.join("claude-session.jsonl"),
            concat!(
                "{\"type\":\"user\",\"uuid\":\"user-1\",\"parentUuid\":null,\"sessionId\":\"claude-session\",\"cwd\":\"/workspace/demo\",\"timestamp\":\"2026-07-15T02:00:00Z\",\"isMeta\":false,\"message\":{\"role\":\"user\",\"content\":\"How should Claude sessions sync?\"}}\n",
                "{\"type\":\"user\",\"uuid\":\"user-2\",\"parentUuid\":\"user-1\",\"sessionId\":\"claude-session\",\"cwd\":\"/workspace/demo\",\"timestamp\":\"2026-07-15T02:00:00Z\",\"isMeta\":false,\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Also preserve steering messages.\"},{\"type\":\"image\",\"source\":{\"data\":\"binary attachment secret\"}}]}}\n",
                "{\"type\":\"user\",\"uuid\":\"meta-1\",\"parentUuid\":\"user-2\",\"sessionId\":\"claude-session\",\"cwd\":\"/workspace/demo\",\"timestamp\":\"2026-07-15T02:00:00Z\",\"isMeta\":true,\"message\":{\"role\":\"user\",\"content\":\"meta system secret\"}}\n",
                "{\"type\":\"assistant\",\"uuid\":\"assistant-tool\",\"parentUuid\":\"meta-1\",\"sessionId\":\"claude-session\",\"cwd\":\"/workspace/demo\",\"timestamp\":\"2026-07-15T02:00:01Z\",\"message\":{\"id\":\"message-tool\",\"role\":\"assistant\",\"stop_reason\":\"tool_use\",\"content\":[{\"type\":\"thinking\",\"thinking\":\"private thought\"},{\"type\":\"text\",\"text\":\"intermediate text\"},{\"type\":\"tool_use\",\"name\":\"Read\"}]}}\n",
                "{\"type\":\"user\",\"uuid\":\"tool-result\",\"parentUuid\":\"assistant-tool\",\"sessionId\":\"claude-session\",\"cwd\":\"/workspace/demo\",\"timestamp\":\"2026-07-15T02:00:02Z\",\"toolUseResult\":{\"stdout\":\"tool secret\"},\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"content\":\"tool secret\"}]}}\n",
                "{\"type\":\"assistant\",\"uuid\":\"assistant-final\",\"parentUuid\":\"tool-result\",\"sessionId\":\"claude-session\",\"cwd\":\"/workspace/demo\",\"timestamp\":\"2026-07-15T02:00:03Z\",\"message\":{\"id\":\"message-final\",\"role\":\"assistant\",\"stop_reason\":\"end_turn\",\"content\":[{\"type\":\"text\",\"text\":\"Follow the parent UUID lineage.\"}]}}\n"
            ),
        )
        .unwrap();

        let mut config = AgentSessionsConfig::default();
        config.sources.claude.enabled = true;
        config.sources.claude.path = Some(projects);
        config.sources.claude.include_memory = false;
        let store = TestStore::default();

        let report = sync_agent_sessions(
            &store,
            &config,
            AgentSessionSyncOptions {
                selection: AgentSessionSelection::Claude,
                dry_run: false,
                allow_initial_backfill: true,
            },
        )
        .await
        .unwrap();

        assert_eq!(report.drawers_written, 1);
        let drawers = store.drawers.lock().unwrap();
        assert_eq!(drawers.len(), 1);
        assert_eq!(
            drawers[0].content,
            "user: How should Claude sessions sync?\nuser: Also preserve steering messages.\nassistant: Follow the parent UUID lineage."
        );
        assert!(!drawers[0].content.contains("private thought"));
        assert!(!drawers[0].content.contains("intermediate text"));
        assert!(!drawers[0].content.contains("tool secret"));
        assert!(!drawers[0].content.contains("meta system secret"));
        assert!(!drawers[0].content.contains("binary attachment secret"));
    }

    #[tokio::test]
    async fn claude_checkpoints_are_unique_across_project_directories() {
        let tmp = tempdir().unwrap();
        let projects = tmp.path().join("projects");
        for (project, session, prompt) in [
            ("project-a", "session-a", "alpha prompt"),
            ("project-b", "session-b", "beta prompt"),
        ] {
            let directory = projects.join(project);
            fs::create_dir_all(&directory).unwrap();
            fs::write(
                directory.join("same-name.jsonl"),
                format!(
                    "{{\"type\":\"user\",\"uuid\":\"{session}-user\",\"parentUuid\":null,\"sessionId\":\"{session}\",\"cwd\":\"/workspace/{project}\",\"isMeta\":false,\"message\":{{\"role\":\"user\",\"content\":\"{prompt}\"}}}}\n{{\"type\":\"assistant\",\"uuid\":\"{session}-answer\",\"parentUuid\":\"{session}-user\",\"sessionId\":\"{session}\",\"cwd\":\"/workspace/{project}\",\"message\":{{\"id\":\"{session}-message\",\"role\":\"assistant\",\"stop_reason\":\"end_turn\",\"content\":[{{\"type\":\"text\",\"text\":\"final answer\"}}]}}}}\n"
                ),
            )
            .unwrap();
        }

        let mut config = AgentSessionsConfig::default();
        config.sources.claude.enabled = true;
        config.sources.claude.path = Some(projects);
        config.sources.claude.include_memory = false;
        let store = TestStore::default();

        let report = sync_agent_sessions(
            &store,
            &config,
            AgentSessionSyncOptions {
                selection: AgentSessionSelection::Claude,
                dry_run: false,
                allow_initial_backfill: true,
            },
        )
        .await
        .unwrap();

        assert_eq!(report.drawers_written, 2);
        assert_eq!(store.drawers.lock().unwrap().len(), 2);
        assert_eq!(
            store
                .checkpoints
                .lock()
                .unwrap()
                .keys()
                .filter(|(adapter, _)| adapter == "claude")
                .count(),
            2
        );
    }

    #[tokio::test]
    async fn claude_checkpoint_prunes_completed_lineage_history() {
        let tmp = tempdir().unwrap();
        let projects = tmp.path().join("projects");
        let project = projects.join("project-a");
        fs::create_dir_all(&project).unwrap();
        let mut transcript = String::new();
        for index in 0..300 {
            transcript.push_str(&format!(
                "{{\"type\":\"user\",\"uuid\":\"user-{index}\",\"parentUuid\":null,\"sessionId\":\"bounded\",\"cwd\":\"/workspace/project-a\",\"isMeta\":false,\"message\":{{\"role\":\"user\",\"content\":\"prompt {index}\"}}}}\n"
            ));
            transcript.push_str(&format!(
                "{{\"type\":\"assistant\",\"uuid\":\"assistant-{index}\",\"parentUuid\":\"user-{index}\",\"sessionId\":\"bounded\",\"cwd\":\"/workspace/project-a\",\"message\":{{\"id\":\"message-{index}\",\"role\":\"assistant\",\"stop_reason\":\"end_turn\",\"content\":[{{\"type\":\"text\",\"text\":\"answer {index}\"}}]}}}}\n"
            ));
        }
        fs::write(project.join("bounded.jsonl"), transcript).unwrap();
        let mut config = AgentSessionsConfig::default();
        config.sources.claude.enabled = true;
        config.sources.claude.path = Some(projects);
        config.sources.claude.include_memory = false;
        let store = TestStore::default();

        let report = sync_agent_sessions(
            &store,
            &config,
            AgentSessionSyncOptions {
                selection: AgentSessionSelection::Claude,
                dry_run: false,
                allow_initial_backfill: true,
            },
        )
        .await
        .unwrap();

        assert_eq!(report.drawers_written, 300);
        let checkpoint = store
            .checkpoints
            .lock()
            .unwrap()
            .values()
            .find(|checkpoint| checkpoint.adapter == "claude")
            .unwrap()
            .clone();
        let cursor: super::SessionCursor = serde_json::from_str(&checkpoint.cursor_json).unwrap();
        assert_eq!(
            cursor.claude_completed_order.len(),
            super::CLAUDE_COMPLETED_ROOT_LIMIT
        );
        assert_eq!(
            cursor.claude_completed_prompts.len(),
            super::CLAUDE_COMPLETED_ROOT_LIMIT
        );
        assert_eq!(
            cursor.claude_prompts.len(),
            super::CLAUDE_COMPLETED_ROOT_LIMIT
        );
        assert_eq!(
            cursor.claude_lineage.len(),
            super::CLAUDE_COMPLETED_ROOT_LIMIT
        );
    }

    #[tokio::test]
    async fn claude_memory_sync_indexes_body_and_replaces_only_changed_file() {
        let tmp = tempdir().unwrap();
        let projects = tmp.path().join("projects");
        let memory = projects.join("-workspace-demo").join("memory");
        fs::create_dir_all(&memory).unwrap();
        let memory_file = memory.join("testing.md");
        fs::write(
            &memory_file,
            "---\nname: testing\ndescription: frontmatter only\n---\n# Testing\nRun targeted tests first.\n",
        )
        .unwrap();

        let mut config = AgentSessionsConfig::default();
        config.sources.claude.enabled = true;
        config.sources.claude.path = Some(projects);
        config.sources.claude.include_memory = true;
        let store = TestStore::default();
        let options = AgentSessionSyncOptions {
            selection: AgentSessionSelection::Claude,
            dry_run: false,
            allow_initial_backfill: true,
        };

        let first = sync_agent_sessions(&store, &config, options).await.unwrap();
        assert_eq!(first.drawers_written, 1);
        {
            let drawers = store.drawers.lock().unwrap();
            assert_eq!(drawers.len(), 1);
            assert_eq!(drawers[0].metadata.wing, "claude-memory");
            assert!(drawers[0].content.contains("Run targeted tests first."));
            assert!(!drawers[0].content.contains("frontmatter only"));
        }

        fs::write(
            &memory_file,
            "---\nname: testing\ndescription: frontmatter only\n---\n# Testing\nRun the full workspace suite last.\n",
        )
        .unwrap();
        let changed = sync_agent_sessions(&store, &config, options).await.unwrap();
        assert_eq!(changed.drawers_written, 1);
        let drawers = store.drawers.lock().unwrap();
        assert_eq!(drawers.len(), 1);
        assert!(drawers[0].content.contains("full workspace suite last"));
        assert!(!drawers[0].content.contains("targeted tests first"));
    }
}
