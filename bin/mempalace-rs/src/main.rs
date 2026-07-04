use std::{
    collections::BTreeMap,
    env, fs,
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
};

use clap::{Parser, Subcommand};
use mempalace_core::{
    AaakDialect, DetectedEntities, Drawer, DrawerMetadata, KnowledgeGraph, MemoryStore,
    MempalaceConfig, MineOptions, SearchQuery, detect_entities, mine_project, scan_for_detection,
};
use mempalace_mcp::McpServer;
use mempalace_store::SqliteMemoryStore;
use rusqlite::{Connection, params};
use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(name = "mempalace-rs", about = "MemPalace in Rust", version)]
struct Cli {
    #[arg(long, global = true)]
    palace: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Init {
        dir: Option<PathBuf>,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        no_onboarding: bool,
        #[arg(long)]
        no_entity_scan: bool,
    },
    Status,
    Search {
        query: String,
        scope: Option<PathBuf>,
        #[arg(long)]
        wing: Option<String>,
        #[arg(long)]
        all_wings: bool,
        #[arg(long)]
        room: Option<String>,
        #[arg(long, default_value_t = 5)]
        results: usize,
    },
    Compress {
        #[arg(long)]
        wing: Option<String>,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        dry_run: bool,
    },
    Mine {
        dir: PathBuf,
        #[arg(long)]
        wing: Option<String>,
        #[arg(long, default_value = "mempalace")]
        agent: String,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        skip_existing: bool,
        #[arg(long)]
        exclude_data_files: bool,
        #[arg(long)]
        no_gitignore: bool,
    },
    Remine {
        #[arg(long)]
        wing: Option<String>,
        #[arg(long)]
        dry_run: bool,
    },
    /// Resize the vectorlite HNSW index max_elements. Uses vectorlite's
    /// save/load operations for O(1) reallocation when supported, falling back
    /// to a full rebuild from existing embeddings on older vectorlite builds.
    Resize {
        /// New max_elements for the HNSW index (default: 1,000,000).
        #[arg(long, default_value_t = 1_000_000)]
        max_elements: u64,
    },
    Migrate {
        #[arg(long = "from")]
        from_path: PathBuf,
        #[arg(long)]
        dry_run: bool,
    },
    Tool {
        #[command(subcommand)]
        command: ToolCommand,
    },
}

#[derive(Debug, Subcommand)]
#[command(rename_all = "snake_case")]
enum ToolCommand {
    Status,
    ListWings,
    ListRooms {
        #[arg(long)]
        wing: Option<String>,
    },
    GetTaxonomy,
    GetAaakSpec,
    Search {
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long)]
        wing: Option<String>,
        #[arg(long)]
        room: Option<String>,
    },
    CheckDuplicate {
        #[arg(long)]
        content: String,
        #[arg(long, default_value_t = 0.9)]
        threshold: f32,
    },
    AddDrawer {
        #[arg(long)]
        wing: String,
        #[arg(long)]
        room: String,
        #[arg(long)]
        content: String,
        #[arg(long = "source-file")]
        source_file: Option<String>,
        #[arg(long = "added-by", default_value = "cli")]
        added_by: String,
    },
    DeleteDrawer {
        #[arg(long = "drawer-id")]
        drawer_id: String,
    },
    KgQuery {
        #[arg(long)]
        entity: String,
        #[arg(long = "as-of")]
        as_of: Option<String>,
        #[arg(long, default_value = "both")]
        direction: String,
    },
    KgAdd {
        #[arg(long)]
        subject: String,
        #[arg(long)]
        predicate: String,
        #[arg(long)]
        object: String,
        #[arg(long = "valid-from")]
        valid_from: Option<String>,
        #[arg(long = "source-closet")]
        source_closet: Option<String>,
    },
    KgInvalidate {
        #[arg(long)]
        subject: String,
        #[arg(long)]
        predicate: String,
        #[arg(long)]
        object: String,
        #[arg(long = "ended")]
        ended: Option<String>,
    },
    KgTimeline {
        #[arg(long)]
        entity: Option<String>,
    },
    KgStats,
    DiaryWrite {
        #[arg(long = "agent-name")]
        agent_name: String,
        #[arg(long)]
        entry: String,
        #[arg(long, default_value = "general")]
        topic: String,
    },
    DiaryRead {
        #[arg(long = "agent-name")]
        agent_name: String,
        #[arg(long = "last-n", default_value_t = 10)]
        last_n: usize,
    },
    Traverse {
        #[arg(long = "start-room")]
        start_room: String,
        #[arg(long = "max-hops", default_value_t = 2)]
        max_hops: usize,
    },
    FindTunnels {
        #[arg(long = "wing-a")]
        wing_a: Option<String>,
        #[arg(long = "wing-b")]
        wing_b: Option<String>,
    },
    GraphStats,
}

struct AppContext {
    config: MempalaceConfig,
    palace_root: PathBuf,
    store_path: PathBuf,
    store: SqliteMemoryStore,
    graph: KnowledgeGraph,
}

const INIT_ROOM_MAP: &[(&str, &str)] = &[
    ("frontend", "frontend"),
    ("front_end", "frontend"),
    ("client", "frontend"),
    ("ui", "frontend"),
    ("views", "frontend"),
    ("components", "frontend"),
    ("pages", "frontend"),
    ("backend", "backend"),
    ("back_end", "backend"),
    ("server", "backend"),
    ("api", "backend"),
    ("routes", "backend"),
    ("services", "backend"),
    ("controllers", "backend"),
    ("models", "backend"),
    ("database", "backend"),
    ("db", "backend"),
    ("docs", "documentation"),
    ("doc", "documentation"),
    ("documentation", "documentation"),
    ("wiki", "documentation"),
    ("readme", "documentation"),
    ("notes", "documentation"),
    ("design", "design"),
    ("designs", "design"),
    ("mockups", "design"),
    ("wireframes", "design"),
    ("assets", "design"),
    ("storyboard", "design"),
    ("costs", "costs"),
    ("cost", "costs"),
    ("budget", "costs"),
    ("finance", "costs"),
    ("financial", "costs"),
    ("pricing", "costs"),
    ("invoices", "costs"),
    ("accounting", "costs"),
    ("meetings", "meetings"),
    ("meeting", "meetings"),
    ("calls", "meetings"),
    ("meeting_notes", "meetings"),
    ("standup", "meetings"),
    ("minutes", "meetings"),
    ("team", "team"),
    ("staff", "team"),
    ("hr", "team"),
    ("hiring", "team"),
    ("employees", "team"),
    ("people", "team"),
    ("research", "research"),
    ("references", "research"),
    ("reading", "research"),
    ("papers", "research"),
    ("planning", "planning"),
    ("roadmap", "planning"),
    ("strategy", "planning"),
    ("specs", "planning"),
    ("requirements", "planning"),
    ("tests", "testing"),
    ("test", "testing"),
    ("testing", "testing"),
    ("qa", "testing"),
    ("scripts", "scripts"),
    ("tools", "scripts"),
    ("utils", "scripts"),
    ("config", "configuration"),
    ("configs", "configuration"),
    ("settings", "configuration"),
    ("infrastructure", "configuration"),
    ("infra", "configuration"),
    ("deploy", "configuration"),
];

const WORK_WINGS: &[&str] = &["projects", "clients", "team", "decisions", "research"];
const PERSONAL_WINGS: &[&str] = &[
    "family",
    "health",
    "creative",
    "reflections",
    "relationships",
];
const COMBO_WINGS: &[&str] = &[
    "family",
    "work",
    "health",
    "creative",
    "projects",
    "reflections",
];

#[derive(Debug, Clone)]
struct OnboardingPerson {
    name: String,
    relationship: String,
    context: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Init {
            dir,
            force,
            no_onboarding,
            no_entity_scan,
        } => {
            let app = open_context(cli.palace).await?;
            run_init(&app, dir, force, no_onboarding, no_entity_scan)?;
        }
        Command::Status => {
            let app = open_context(cli.palace).await?;
            print_status(&app).await?;
        }
        Command::Search {
            query,
            scope,
            wing,
            all_wings,
            room,
            results,
        } => {
            let app = open_context(cli.palace).await?;
            run_search(&app, query, scope, wing, all_wings, room, results).await?;
        }
        Command::Compress {
            wing,
            config,
            output,
            dry_run,
        } => {
            let app = open_context(cli.palace).await?;
            run_compress(&app, wing, config, output, dry_run).await?;
        }
        Command::Mine {
            dir,
            wing,
            agent,
            limit,
            dry_run,
            skip_existing,
            exclude_data_files,
            no_gitignore,
        } => {
            let app = open_context(cli.palace).await?;
            run_mine(
                &app,
                dir,
                MineOptions {
                    wing,
                    agent,
                    limit,
                    dry_run,
                    skip_existing,
                    exclude_data_files,
                    respect_gitignore: !no_gitignore,
                    log_progress: true,
                },
            )
            .await?;
        }
        Command::Remine { wing, dry_run } => {
            let app = open_context(cli.palace).await?;
            run_remine(&app, wing, dry_run).await?;
        }
        Command::Resize { max_elements } => {
            let app = open_context(cli.palace).await?;
            run_resize(&app, max_elements).await?;
        }
        Command::Migrate { from_path, dry_run } => {
            let app = open_context(cli.palace).await?;
            run_migrate(&app, from_path, dry_run).await?;
        }
        Command::Tool { command } => {
            let server = McpServer::open_with_palace(cli.palace).await?;
            run_tool_command(&server, command).await?;
        }
    }

    Ok(())
}

async fn run_migrate(
    app: &AppContext,
    chroma_db: PathBuf,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let chroma_db = chroma_db.canonicalize().unwrap_or(chroma_db);
    if !chroma_db.exists() {
        return Err(format!("chroma database not found: {}", chroma_db.display()).into());
    }

    println!();
    println!("{:=<55}", "");
    println!("  MemPalace Migrate");
    println!("{:=<55}", "");
    println!("  Source: {}", chroma_db.display());
    println!("  Target: {}", app.store_path.display());
    if dry_run {
        println!("  DRY RUN: true");
    }
    println!("{:-<55}", "");

    // Open the old ChromaDB SQLite
    let drawers = extract_chroma_drawers(&chroma_db)?;
    println!("  Found {} drawers in ChromaDB", drawers.len());

    if drawers.is_empty() {
        println!("  Nothing to migrate.");
        return Ok(());
    }

    // Show wing/room distribution
    let mut wing_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut room_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for d in &drawers {
        *wing_counts.entry(d.metadata.wing.clone()).or_default() += 1;
        *room_counts
            .entry(format!("{}/{}", d.metadata.wing, d.metadata.room))
            .or_default() += 1;
    }

    println!();
    println!("  Distribution:");
    let mut wings: Vec<_> = wing_counts.iter().collect();
    wings.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (wing, count) in wings {
        println!("    {wing:30} {count} drawers");
    }

    if dry_run {
        println!();
        println!("  Dry run complete. No data written.");
        return Ok(());
    }

    // Migrate in batches to avoid memory pressure with very large palaces
    let batch_size = 256;
    let total_batches = drawers.len().div_ceil(batch_size);

    for (batch_num, chunk) in drawers.chunks(batch_size).enumerate() {
        let batch_display = batch_num + 1;
        println!(
            "  Migrating batch {batch_display}/{total_batches} ({} drawers)...",
            chunk.len()
        );
        app.store.add_drawers(chunk.to_vec()).await?;
    }

    let status = app.store.status().await?;
    println!();
    println!("{:=<55}", "");
    println!("  Migration complete.");
    println!("  Total drawers: {}", status.total_drawers);
    println!("  Original chroma.sqlite3 was left untouched.");
    println!("{:=<55}", "");

    Ok(())
}

/// Extract all drawers from a ChromaDB SQLite database, mirroring the Python
/// `extract_drawers_from_sqlite` function in the original mempalace.
fn extract_chroma_drawers(db_path: &PathBuf) -> Result<Vec<Drawer>, Box<dyn std::error::Error>> {
    let conn = Connection::open(db_path)?;

    // Get all embedding IDs and their documents
    let mut stmt = conn.prepare(
        "SELECT e.embedding_id,
                MAX(CASE WHEN em.key = 'chroma:document' THEN em.string_value END) as document
         FROM embeddings e
         JOIN embedding_metadata em ON em.id = e.id
         GROUP BY e.embedding_id",
    )?;

    struct ChromaRow {
        embedding_id: String,
        document: Option<String>,
    }

    let rows: Vec<ChromaRow> = stmt
        .query_map([], |row| {
            Ok(ChromaRow {
                embedding_id: row.get(0)?,
                document: row.get(1)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    // Prepare metadata query
    let mut meta_stmt = conn.prepare(
        "SELECT em.key, em.string_value, em.int_value, em.float_value, em.bool_value
         FROM embedding_metadata em
         JOIN embeddings e ON e.id = em.id
         WHERE e.embedding_id = ?1
           AND em.key NOT LIKE 'chroma:%'",
    )?;

    let mut drawers = Vec::with_capacity(rows.len());

    for row in rows {
        let document = match row.document {
            Some(doc) if !doc.trim().is_empty() => doc,
            _ => continue,
        };

        let meta_rows: Vec<_> = meta_stmt
            .query_map(params![row.embedding_id], |mr| {
                Ok((
                    mr.get::<_, String>(0)?,
                    mr.get::<_, Option<String>>(1)?,
                    mr.get::<_, Option<i64>>(2)?,
                    mr.get::<_, Option<f64>>(3)?,
                    mr.get::<_, Option<bool>>(4)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let mut metadata: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for (key, string_val, int_val, float_val, bool_val) in meta_rows {
            let value = if let Some(s) = string_val {
                s
            } else if let Some(i) = int_val {
                i.to_string()
            } else if let Some(f) = float_val {
                f.to_string()
            } else if let Some(b) = bool_val {
                b.to_string()
            } else {
                continue;
            };
            metadata.insert(key, value);
        }

        let wing = metadata
            .get("wing")
            .cloned()
            .unwrap_or_else(|| "chroma-legacy".to_owned());
        let room = metadata
            .get("room")
            .cloned()
            .unwrap_or_else(|| "general".to_owned());
        let source_file = metadata.get("source_file").cloned();
        let added_by = metadata
            .get("added_by")
            .cloned()
            .unwrap_or_else(|| "migration".to_owned());
        let filed_at = metadata.get("filed_at").cloned();

        drawers.push(Drawer {
            id: Uuid::now_v7().to_string(),
            content: document,
            metadata: DrawerMetadata {
                wing,
                room,
                source_file,
                chunk_index: 0,
                added_by,
                filed_at,
            },
        });
    }

    Ok(drawers)
}

async fn run_tool_command(
    server: &McpServer,
    command: ToolCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    let value = match command {
        ToolCommand::Status => server.tool_status().await.map_err(io::Error::other)?,
        ToolCommand::ListWings => server.tool_list_wings().await.map_err(io::Error::other)?,
        ToolCommand::ListRooms { wing } => server
            .tool_list_rooms(wing)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::GetTaxonomy => server.tool_get_taxonomy().await.map_err(io::Error::other)?,
        ToolCommand::GetAaakSpec => server.tool_get_aaak_spec(),
        ToolCommand::Search {
            query,
            limit,
            wing,
            room,
        } => server
            .tool_search(query, limit, wing, room)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::CheckDuplicate { content, threshold } => server
            .tool_check_duplicate(content, threshold)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::AddDrawer {
            wing,
            room,
            content,
            source_file,
            added_by,
        } => server
            .tool_add_drawer(wing, room, content, source_file, added_by)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::DeleteDrawer { drawer_id } => server
            .tool_delete_drawer(drawer_id)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::KgQuery {
            entity,
            as_of,
            direction,
        } => server
            .tool_kg_query(entity, as_of, direction)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::KgAdd {
            subject,
            predicate,
            object,
            valid_from,
            source_closet,
        } => server
            .tool_kg_add(subject, predicate, object, valid_from, source_closet)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::KgInvalidate {
            subject,
            predicate,
            object,
            ended,
        } => server
            .tool_kg_invalidate(subject, predicate, object, ended)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::KgTimeline { entity } => server
            .tool_kg_timeline(entity)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::KgStats => server.tool_kg_stats().await.map_err(io::Error::other)?,
        ToolCommand::DiaryWrite {
            agent_name,
            entry,
            topic,
        } => server
            .tool_diary_write(agent_name, entry, topic)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::DiaryRead { agent_name, last_n } => server
            .tool_diary_read(agent_name, last_n)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::Traverse {
            start_room,
            max_hops,
        } => server
            .tool_traverse(start_room, max_hops)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::FindTunnels { wing_a, wing_b } => server
            .tool_find_tunnels(wing_a, wing_b)
            .await
            .map_err(io::Error::other)?,
        ToolCommand::GraphStats => server.tool_graph_stats().await.map_err(io::Error::other)?,
    };

    print_json(&value)?;
    Ok(())
}

fn print_json(value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

async fn run_compress(
    app: &AppContext,
    wing: Option<String>,
    config_path: Option<PathBuf>,
    output_path: Option<PathBuf>,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let dialect = load_dialect(app, config_path)?;
    let drawers = app.store.list_drawers(wing.as_deref()).await?;

    if drawers.is_empty() {
        if let Some(wing) = wing {
            println!("no drawers found in wing '{wing}'");
        } else {
            println!("no drawers found");
        }
        return Ok(());
    }

    println!();
    println!("{:=<55}", "");
    println!("  AAAK Compress");
    println!("{:=<55}", "");
    println!("  Drawers: {}", drawers.len());
    if let Some(wing) = &wing {
        println!("  Wing:    {wing}");
    }
    println!();

    let mut entries = Vec::with_capacity(drawers.len());
    let mut original_chars = 0usize;
    let mut compressed_chars = 0usize;
    let mut original_tokens_est = 0usize;
    let mut summary_tokens_est = 0usize;

    for drawer in drawers {
        let compressed = dialect.compress(&drawer.content, Some(&drawer.metadata));
        let stats = dialect.compression_stats(&drawer.content, &compressed);
        original_chars += stats.original_chars;
        compressed_chars += stats.summary_chars;
        original_tokens_est += stats.original_tokens_est;
        summary_tokens_est += stats.summary_tokens_est;

        if dry_run {
            let source = drawer
                .metadata
                .source_file
                .as_deref()
                .and_then(|path| Path::new(path).file_name())
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| drawer.id.clone());
            println!(
                "  [{}/{}] {}",
                drawer.metadata.wing, drawer.metadata.room, source
            );
            println!(
                "    {}t -> {}t (~{:.1}x)",
                stats.original_tokens_est, stats.summary_tokens_est, stats.size_ratio
            );
            println!("    {compressed}");
            println!();
        }

        entries.push(compressed);
    }

    let output = entries.join("\n\n");
    if !dry_run {
        let output_path =
            output_path.unwrap_or_else(|| default_aaak_output_path(app, wing.as_deref()));
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&output_path, output)?;
        println!("  Wrote: {}", output_path.display());
    }

    let ratio = original_tokens_est as f32 / summary_tokens_est.max(1) as f32;
    println!(
        "  Total: {}t -> {}t (~{:.1}x, {} -> {} chars)",
        original_tokens_est, summary_tokens_est, ratio, original_chars, compressed_chars
    );
    if dry_run {
        println!("  Dry run: nothing written");
    }
    println!("{:=<55}", "");

    Ok(())
}

fn load_dialect(
    app: &AppContext,
    config_path: Option<PathBuf>,
) -> Result<AaakDialect, Box<dyn std::error::Error>> {
    if let Some(path) = config_path {
        return Ok(AaakDialect::from_config_path(path)?);
    }

    let local = PathBuf::from("entities.json");
    if local.is_file() {
        return Ok(AaakDialect::from_config_path(local)?);
    }

    let palace_entities = app.palace_root.join("entities.json");
    if palace_entities.is_file() {
        return Ok(AaakDialect::from_config_path(palace_entities)?);
    }

    Ok(AaakDialect::default())
}

fn default_aaak_output_path(app: &AppContext, wing: Option<&str>) -> PathBuf {
    let label = wing.unwrap_or("all").replace(['\\', '/', ':'], "_");
    app.config
        .config_dir()
        .join("aaak")
        .join(format!("{label}.aaak"))
}

fn run_init(
    app: &AppContext,
    dir: Option<PathBuf>,
    force: bool,
    no_onboarding: bool,
    no_entity_scan: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    print_init(app);

    let project_dir = dir.map(|path| path.canonicalize()).transpose()?;

    if let Some(dir) = project_dir.as_deref() {
        if !no_entity_scan {
            run_project_entity_detection(dir, force)?;
        }

        let config_path = write_project_config_scaffold(dir.to_path_buf(), force)?;
        println!("project config: {}", config_path.display());
    }

    if !no_onboarding && should_run_onboarding(&app.config) {
        if io::stdin().is_terminal() {
            println!();
            run_onboarding(app, project_dir.as_deref())?;
        } else {
            println!("note: onboarding skipped because stdin is not interactive.");
        }
    }

    Ok(())
}

fn resolve_search_wing(
    explicit_wing: Option<String>,
    all_wings: bool,
    scope: Option<PathBuf>,
    cwd: Option<PathBuf>,
) -> Option<String> {
    if explicit_wing.is_some() || all_wings {
        return explicit_wing;
    }

    if let Some(scope) = scope {
        return infer_wing_from_scope(&scope);
    }

    cwd.filter(|path| looks_like_project_root(path))
        .and_then(|path| infer_wing_from_scope(&path))
}

fn infer_wing_from_scope(scope: &Path) -> Option<String> {
    let canonical = scope.canonicalize().ok()?;
    let base = if canonical.is_file() {
        canonical.parent()?.to_path_buf()
    } else {
        canonical
    };

    base.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .filter(|name| !name.is_empty())
}

fn looks_like_project_root(path: &Path) -> bool {
    [
        ".git",
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "go.mod",
        "tsconfig.json",
    ]
    .into_iter()
    .any(|marker| path.join(marker).exists())
}

async fn open_context(
    palace_override: Option<PathBuf>,
) -> Result<AppContext, Box<dyn std::error::Error>> {
    let config = MempalaceConfig::load()?;
    config.init()?;

    let knowledge_graph_path = palace_override
        .as_deref()
        .map(MempalaceConfig::knowledge_graph_path_for_palace)
        .unwrap_or_else(|| config.knowledge_graph_path());
    let palace_root = palace_override.unwrap_or_else(|| config.palace_path());
    let store_path = palace_root.join("store.sqlite3");
    let model_cache_path = config.model_cache_path();
    fs::create_dir_all(&palace_root)?;
    fs::create_dir_all(&model_cache_path)?;

    let store = SqliteMemoryStore::new(&palace_root, &model_cache_path)?;
    let graph = KnowledgeGraph::new(knowledge_graph_path)?;

    Ok(AppContext {
        config,
        palace_root,
        store_path,
        store,
        graph,
    })
}

fn print_init(app: &AppContext) {
    println!("mempalace-rs initialized");
    println!("config: {}", app.config.config_path().display());
    println!("palace root: {}", app.palace_root.display());
    println!("store: {}", app.store_path.display());
    println!("kg: {}", app.graph.db_path().display());

    if legacy_chroma_detected(&app.palace_root, &app.store_path) {
        println!();
        println!("note: legacy Chroma data detected.");
        println!(
            "rust drawers will be stored in: {}",
            app.store_path.display()
        );
    }
}

fn should_run_onboarding(config: &MempalaceConfig) -> bool {
    config.mode().is_none()
        || config.topic_wings().is_empty()
        || !config.entity_registry_path().exists()
        || !config.aaak_entities_path().exists()
        || !config.critical_facts_path().exists()
}

fn run_onboarding(
    app: &AppContext,
    directory: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mode = ask_mode()?;
    let (mut people, aliases) = ask_people(&mode)?;
    let projects = ask_projects(&mode)?;
    let wings = ask_wings(&mode)?;

    let default_directory = directory.unwrap_or_else(|| Path::new("."));
    if prompt_yes_no(
        "  Scan your files for additional names we might have missed?",
        true,
    )? {
        let scan_dir = prompt(
            "  Directory to scan",
            Some(default_directory.to_string_lossy().as_ref()),
        )?;
        let detected = detect_additional_people(Path::new(&scan_dir), &people)?;
        if !detected.is_empty() {
            println!();
            println!("  Found additional candidates:");
            for candidate in &detected {
                println!(
                    "    {:20} confidence={:.0}% {}",
                    candidate.name,
                    candidate.confidence * 100.0,
                    candidate.signals.first().cloned().unwrap_or_default()
                );
            }

            if prompt_yes_no("  Add any of these to your registry?", true)? {
                for candidate in detected {
                    let answer = prompt(
                        &format!("    {} - add as person? [y/N]", candidate.name),
                        Some("n"),
                    )?;
                    if !matches!(answer.to_lowercase().as_str(), "y" | "yes") {
                        continue;
                    }
                    let relationship = prompt(
                        &format!("    Relationship/role for {}", candidate.name),
                        None,
                    )?;
                    let context = if mode == "combo" {
                        let choice = prompt("    Context [personal/work]", Some("personal"))?;
                        if choice.to_lowercase().starts_with('w') {
                            "work".to_owned()
                        } else {
                            "personal".to_owned()
                        }
                    } else if mode == "work" {
                        "work".to_owned()
                    } else {
                        "personal".to_owned()
                    };
                    people.push(OnboardingPerson {
                        name: candidate.name,
                        relationship,
                        context,
                    });
                }
            }
        }
    }

    let mut config = app.config.clone();
    config.save_onboarding(mode.clone(), wings.clone(), projects.clone())?;
    config.save_people_map(aliases)?;

    write_aaak_bootstrap(&config, &people, &projects, &wings, &mode)?;
    write_entity_registry(&config, &people, &projects)?;

    println!();
    println!("{:=<58}", "");
    println!("  Setup Complete");
    println!("{:=<58}", "");
    println!("  Mode: {}", mode);
    println!("  People: {}", people.len());
    println!("  Projects: {}", projects.len());
    println!("  Wings: {}", wings.join(", "));
    println!(
        "  AAAK entity registry: {}",
        config.aaak_entities_path().display()
    );
    println!(
        "  Critical facts: {}",
        config.critical_facts_path().display()
    );
    println!(
        "  Registry saved to: {}",
        config.entity_registry_path().display()
    );
    if let Some(directory) = directory {
        println!("  Project dir: {}", directory.display());
    }
    println!("{:=<58}", "");

    Ok(())
}

fn ask_mode() -> Result<String, Box<dyn std::error::Error>> {
    println!("{:=<58}", "");
    println!("  Welcome to MemPalace");
    println!("{:=<58}", "");
    println!("  MemPalace works better if it knows your world first.");
    println!("  Pick a mode:");
    println!("    [1] Work");
    println!("    [2] Personal");
    println!("    [3] Both");

    loop {
        match prompt("  Your choice [1/2/3]", None)?.as_str() {
            "1" => return Ok("work".to_owned()),
            "2" => return Ok("personal".to_owned()),
            "3" => return Ok("combo".to_owned()),
            _ => println!("  Please enter 1, 2, or 3."),
        }
    }
}

type PeopleResult = (Vec<OnboardingPerson>, BTreeMap<String, String>);

fn ask_people(mode: &str) -> Result<PeopleResult, Box<dyn std::error::Error>> {
    let mut people = Vec::new();
    let mut aliases = BTreeMap::new();

    if matches!(mode, "personal" | "combo") {
        println!();
        println!("  Personal world: enter important people as `name, relationship`.");
        println!("  Press enter on an empty line when finished.");
        loop {
            let entry = prompt("  Person", None)?;
            if entry.is_empty() {
                break;
            }

            let (name, relationship) = split_name_and_label(&entry);
            if name.is_empty() {
                continue;
            }

            let nickname = prompt(&format!("  Nickname for {name}"), None)?;
            if !nickname.is_empty() {
                aliases.insert(nickname, name.clone());
            }

            people.push(OnboardingPerson {
                name,
                relationship,
                context: "personal".to_owned(),
            });
        }
    }

    if matches!(mode, "work" | "combo") {
        println!();
        println!("  Work world: enter colleagues, clients, or collaborators.");
        println!("  Use `name, role` and press enter on an empty line when finished.");
        loop {
            let entry = prompt("  Person", None)?;
            if entry.is_empty() {
                break;
            }

            let (name, relationship) = split_name_and_label(&entry);
            if name.is_empty() {
                continue;
            }

            people.push(OnboardingPerson {
                name,
                relationship,
                context: "work".to_owned(),
            });
        }
    }

    Ok((people, aliases))
}

fn ask_projects(mode: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if mode == "personal" {
        return Ok(Vec::new());
    }

    let mut projects = Vec::new();
    println!();
    println!("  Main projects: enter one per line. Press enter when finished.");
    loop {
        let project = prompt("  Project", None)?;
        if project.is_empty() {
            break;
        }
        projects.push(project);
    }

    Ok(projects)
}

fn ask_wings(mode: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let defaults = match mode {
        "work" => WORK_WINGS,
        "personal" => PERSONAL_WINGS,
        _ => COMBO_WINGS,
    };

    println!();
    println!("  Suggested wings: {}", defaults.join(", "));
    let custom = prompt("  Wings (comma-separated, enter to keep defaults)", None)?;
    if custom.trim().is_empty() {
        return Ok(defaults.iter().map(|wing| (*wing).to_owned()).collect());
    }

    let wings = custom
        .split(',')
        .map(str::trim)
        .filter(|wing| !wing.is_empty())
        .map(|wing| wing.to_owned())
        .collect::<Vec<_>>();

    if wings.is_empty() {
        return Ok(defaults.iter().map(|wing| (*wing).to_owned()).collect());
    }

    Ok(wings)
}

fn prompt(label: &str, default: Option<&str>) -> Result<String, Box<dyn std::error::Error>> {
    match default {
        Some(default) => print!("{label} [{default}]: "),
        None => print!("{label}: "),
    }
    io::stdout().flush()?;

    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let line = line.trim().to_owned();
    if line.is_empty() {
        Ok(default.unwrap_or_default().to_owned())
    } else {
        Ok(line)
    }
}

fn prompt_yes_no(label: &str, default_yes: bool) -> Result<bool, Box<dyn std::error::Error>> {
    let default = if default_yes { "y" } else { "n" };
    let answer = prompt(label, Some(default))?;
    if answer.trim().is_empty() {
        return Ok(default_yes);
    }

    Ok(matches!(answer.to_lowercase().as_str(), "y" | "yes"))
}

fn split_name_and_label(input: &str) -> (String, String) {
    let mut parts = input.splitn(2, ',').map(str::trim);
    let name = parts.next().unwrap_or_default().to_owned();
    let label = parts.next().unwrap_or_default().to_owned();
    (name, label)
}

fn run_project_entity_detection(
    project_dir: &Path,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let files = scan_for_detection(project_dir, 10)?;
    if files.is_empty() {
        return Ok(());
    }

    println!();
    println!("  Scanning for entities in: {}", project_dir.display());
    println!("  Reading {} files...", files.len());
    let detected = detect_entities(&files, 10)?;
    if detected.people.is_empty() && detected.projects.is_empty() && detected.uncertain.is_empty() {
        println!("  No entity candidates detected.");
        return Ok(());
    }

    let (people, projects) = confirm_detected_entities(&detected, io::stdin().is_terminal())?;
    if people.is_empty() && projects.is_empty() {
        return Ok(());
    }

    let entity_path = project_dir.join("entities.json");
    if entity_path.exists() && !force {
        println!("  Entity config already exists: {}", entity_path.display());
        return Ok(());
    }

    write_entities_config(&entity_path, &people, &projects)?;
    println!("  Entities saved: {}", entity_path.display());
    Ok(())
}

fn confirm_detected_entities(
    detected: &DetectedEntities,
    interactive: bool,
) -> Result<(Vec<OnboardingPerson>, Vec<String>), Box<dyn std::error::Error>> {
    println!();
    println!("{:=<58}", "");
    println!("  MemPalace - Entity Detection");
    println!("{:=<58}", "");
    print_detected_entities("PEOPLE", &detected.people);
    print_detected_entities("PROJECTS", &detected.projects);
    if !detected.uncertain.is_empty() {
        print_detected_entities("UNCERTAIN", &detected.uncertain);
    }

    let mut confirmed_people = detected
        .people
        .iter()
        .map(|entity| OnboardingPerson {
            name: entity.name.clone(),
            relationship: String::new(),
            context: "work".to_owned(),
        })
        .collect::<Vec<_>>();
    let mut confirmed_projects = detected
        .projects
        .iter()
        .map(|entity| entity.name.clone())
        .collect::<Vec<_>>();

    if !interactive {
        return Ok((confirmed_people, confirmed_projects));
    }

    println!("  Options:");
    println!("    [enter] Accept detected");
    println!("    [edit]  Review uncertain and add/remove entries");
    println!("    [add]   Add missing entries manually");
    let choice = prompt("  Your choice [enter/edit/add]", None)?.to_lowercase();

    if choice == "edit" {
        for entity in &detected.uncertain {
            let answer = prompt(
                &format!(
                    "  {} - classify as (p)erson, p(r)oject, or (s)kip",
                    entity.name
                ),
                Some("s"),
            )?;
            match answer.to_lowercase().as_str() {
                "p" => confirmed_people.push(OnboardingPerson {
                    name: entity.name.clone(),
                    relationship: String::new(),
                    context: "work".to_owned(),
                }),
                "r" => confirmed_projects.push(entity.name.clone()),
                _ => {}
            }
        }

        confirmed_people = remove_people(confirmed_people)?;
        confirmed_projects = remove_projects(confirmed_projects)?;
    }

    if choice == "add" || prompt_yes_no("  Add any missing entities?", false)? {
        loop {
            let name = prompt("  Name (enter to stop)", None)?;
            if name.is_empty() {
                break;
            }
            let kind = prompt("  Is it a (p)erson or p(r)oject", Some("p"))?;
            if kind.to_lowercase() == "r" {
                confirmed_projects.push(name);
            } else {
                confirmed_people.push(OnboardingPerson {
                    name,
                    relationship: String::new(),
                    context: "work".to_owned(),
                });
            }
        }
    }

    confirmed_people.sort_by(|left, right| left.name.cmp(&right.name));
    confirmed_people.dedup_by(|left, right| left.name.eq_ignore_ascii_case(&right.name));
    confirmed_projects.sort();
    confirmed_projects.dedup_by(|left, right| left.eq_ignore_ascii_case(right));
    Ok((confirmed_people, confirmed_projects))
}

fn print_detected_entities(label: &str, entities: &[mempalace_core::DetectedEntity]) {
    println!();
    println!("  {label}:");
    if entities.is_empty() {
        println!("    (none detected)");
        return;
    }

    for (index, entity) in entities.iter().enumerate() {
        let filled = (entity.confidence * 5.0).floor() as usize;
        let bar = format!("{}{}", "o".repeat(filled), ".".repeat(5 - filled.min(5)));
        let signals = entity.signals.iter().take(2).cloned().collect::<Vec<_>>();
        println!(
            "    {:2}. {:20} [{}] {}",
            index + 1,
            entity.name,
            bar,
            signals.join(", ")
        );
    }
}

fn remove_people(
    people: Vec<OnboardingPerson>,
) -> Result<Vec<OnboardingPerson>, Box<dyn std::error::Error>> {
    if people.is_empty() {
        return Ok(people);
    }

    println!(
        "  Current people: {}",
        people
            .iter()
            .enumerate()
            .map(|(index, person)| format!("{}:{}", index + 1, person.name))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let remove = prompt("  Numbers to remove from people (comma-separated)", None)?;
    if remove.trim().is_empty() {
        return Ok(people);
    }

    let remove = parse_indexes(&remove);
    Ok(people
        .into_iter()
        .enumerate()
        .filter_map(|(index, person)| (!remove.contains(&(index + 1))).then_some(person))
        .collect())
}

fn remove_projects(projects: Vec<String>) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if projects.is_empty() {
        return Ok(projects);
    }

    println!(
        "  Current projects: {}",
        projects
            .iter()
            .enumerate()
            .map(|(index, project)| format!("{}:{}", index + 1, project))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let remove = prompt("  Numbers to remove from projects (comma-separated)", None)?;
    if remove.trim().is_empty() {
        return Ok(projects);
    }

    let remove = parse_indexes(&remove);
    Ok(projects
        .into_iter()
        .enumerate()
        .filter_map(|(index, project)| (!remove.contains(&(index + 1))).then_some(project))
        .collect())
}

fn parse_indexes(input: &str) -> Vec<usize> {
    input
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect()
}

fn detect_additional_people(
    directory: &Path,
    known_people: &[OnboardingPerson],
) -> Result<Vec<mempalace_core::DetectedEntity>, Box<dyn std::error::Error>> {
    let files = scan_for_detection(directory, 10)?;
    if files.is_empty() {
        return Ok(Vec::new());
    }

    let known_names = known_people
        .iter()
        .map(|person| person.name.to_lowercase())
        .collect::<Vec<_>>();
    let detected = detect_entities(&files, 10)?;
    Ok(detected
        .people
        .into_iter()
        .filter(|entity| entity.confidence >= 0.7)
        .filter(|entity| {
            !known_names
                .iter()
                .any(|known| known == &entity.name.to_lowercase())
        })
        .collect())
}

fn write_entities_config(
    path: &Path,
    people: &[OnboardingPerson],
    projects: &[String],
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut entities = BTreeMap::new();
    for person in people {
        entities.insert(
            person.name.clone(),
            next_entity_code(&person.name, &entities),
        );
    }
    for project in projects {
        entities.insert(project.clone(), next_entity_code(project, &entities));
    }

    let dialect = AaakDialect::new(entities, Vec::new());
    Ok(dialect.save_config(path.to_path_buf())?)
}

#[derive(Debug, Serialize)]
struct EntityRegistryFile {
    version: u8,
    mode: String,
    people: BTreeMap<String, EntityRegistryPerson>,
    projects: Vec<String>,
    ambiguous_flags: Vec<String>,
    wiki_cache: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct EntityRegistryPerson {
    source: &'static str,
    contexts: Vec<String>,
    aliases: Vec<String>,
    relationship: String,
    confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    canonical: Option<String>,
}

fn write_entity_registry(
    config: &MempalaceConfig,
    people: &[OnboardingPerson],
    projects: &[String],
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let aliases = config.people_map();
    let reverse_aliases = aliases
        .iter()
        .map(|(alias, canonical)| (canonical.as_str(), alias.as_str()))
        .collect::<BTreeMap<_, _>>();

    let mut registry_people = BTreeMap::new();
    for person in people {
        let alias = reverse_aliases.get(person.name.as_str()).copied();
        registry_people.insert(
            person.name.clone(),
            EntityRegistryPerson {
                source: "onboarding",
                contexts: vec![person.context.clone()],
                aliases: alias.into_iter().map(str::to_owned).collect(),
                relationship: person.relationship.clone(),
                confidence: 1.0,
                canonical: None,
            },
        );

        if let Some(alias) = alias {
            registry_people.insert(
                alias.to_owned(),
                EntityRegistryPerson {
                    source: "onboarding",
                    contexts: vec![person.context.clone()],
                    aliases: vec![person.name.clone()],
                    relationship: person.relationship.clone(),
                    confidence: 1.0,
                    canonical: Some(person.name.clone()),
                },
            );
        }
    }

    let ambiguous_flags = registry_people
        .keys()
        .filter(|&name| is_ambiguous_registry_name(name))
        .map(|name| name.to_lowercase())
        .collect::<Vec<_>>();

    let raw = serde_json::to_string_pretty(&EntityRegistryFile {
        version: 1,
        mode: config.mode().unwrap_or("personal").to_owned(),
        people: registry_people,
        projects: projects.to_vec(),
        ambiguous_flags,
        wiki_cache: BTreeMap::new(),
    })?;

    let path = config.entity_registry_path();
    fs::write(&path, raw)?;
    Ok(path)
}

fn is_ambiguous_registry_name(name: &str) -> bool {
    const COMMON_ENGLISH_WORDS: &[&str] = &[
        "ever", "grace", "will", "bill", "mark", "april", "may", "june", "joy", "hope", "faith",
        "chance", "chase", "hunter", "dash", "flash", "star", "sky", "river", "brook", "lane",
        "art", "clay", "gil", "nat", "max", "rex", "ray",
    ];

    COMMON_ENGLISH_WORDS
        .iter()
        .any(|word| word.eq_ignore_ascii_case(name))
}

fn next_entity_code(name: &str, existing: &BTreeMap<String, String>) -> String {
    let letters = name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_uppercase();
    let letters = if letters.is_empty() {
        "ENT".to_owned()
    } else {
        letters
    };

    for len in 3..=letters.len().max(3) {
        let candidate = letters.chars().take(len).collect::<String>();
        if candidate.len() < 3 {
            continue;
        }
        if !existing.values().any(|value| value == &candidate) {
            return candidate;
        }
    }

    let mut suffix = existing.len() + 1;
    loop {
        let candidate = format!("{}{}", letters.chars().take(3).collect::<String>(), suffix);
        if !existing.values().any(|value| value == &candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

fn write_aaak_bootstrap(
    config: &MempalaceConfig,
    people: &[OnboardingPerson],
    projects: &[String],
    wings: &[String],
    mode: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut entity_codes = BTreeMap::new();
    for person in people {
        entity_codes.insert(
            person.name.clone(),
            next_entity_code(&person.name, &entity_codes),
        );
    }
    for project in projects {
        entity_codes.insert(project.clone(), next_entity_code(project, &entity_codes));
    }

    let mut registry = vec![
        "# AAAK Entity Registry".to_owned(),
        "# Auto-generated by mempalace-rs init.".to_owned(),
        String::new(),
        "## People".to_owned(),
    ];

    for person in people {
        let code = entity_codes
            .get(&person.name)
            .cloned()
            .unwrap_or_else(|| next_entity_code(&person.name, &entity_codes));
        if person.relationship.is_empty() {
            registry.push(format!("  {code}={}", person.name));
        } else {
            registry.push(format!(
                "  {code}={} ({})",
                person.name, person.relationship
            ));
        }
    }

    if !projects.is_empty() {
        registry.push(String::new());
        registry.push("## Projects".to_owned());
        for project in projects {
            let code = entity_codes
                .get(project)
                .cloned()
                .unwrap_or_else(|| next_entity_code(project, &entity_codes));
            registry.push(format!("  {code}={project}"));
        }
    }

    let mut facts = vec![
        "# Critical Facts".to_owned(),
        String::new(),
        format!("Mode: {mode}"),
        format!("Wings: {}", wings.join(", ")),
        String::new(),
    ];

    let personal = people
        .iter()
        .filter(|person| person.context == "personal")
        .collect::<Vec<_>>();
    if !personal.is_empty() {
        facts.push("## People (personal)".to_owned());
        for person in personal {
            let code = entity_codes
                .get(&person.name)
                .cloned()
                .unwrap_or_else(|| next_entity_code(&person.name, &entity_codes));
            if person.relationship.is_empty() {
                facts.push(format!("- {} ({code})", person.name));
            } else {
                facts.push(format!(
                    "- {} ({code}) - {}",
                    person.name, person.relationship
                ));
            }
        }
        facts.push(String::new());
    }

    let work = people
        .iter()
        .filter(|person| person.context == "work")
        .collect::<Vec<_>>();
    if !work.is_empty() {
        facts.push("## People (work)".to_owned());
        for person in work {
            let code = entity_codes
                .get(&person.name)
                .cloned()
                .unwrap_or_else(|| next_entity_code(&person.name, &entity_codes));
            if person.relationship.is_empty() {
                facts.push(format!("- {} ({code})", person.name));
            } else {
                facts.push(format!(
                    "- {} ({code}) - {}",
                    person.name, person.relationship
                ));
            }
        }
        facts.push(String::new());
    }

    if !projects.is_empty() {
        facts.push("## Projects".to_owned());
        for project in projects {
            facts.push(format!("- {project}"));
        }
        facts.push(String::new());
    }

    fs::write(config.aaak_entities_path(), registry.join("\n"))?;
    fs::write(config.critical_facts_path(), facts.join("\n"))?;
    Ok(())
}

fn write_project_config_scaffold(
    dir: PathBuf,
    force: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let project_dir = dir.canonicalize()?;
    if !project_dir.is_dir() {
        return Err(format!("not a directory: {}", project_dir.display()).into());
    }

    let config_path = preferred_project_config_path(&project_dir);
    if config_path.exists() && !force {
        return Ok(config_path);
    }

    let wing = infer_wing_from_scope(&project_dir)
        .filter(|wing| !wing.is_empty())
        .unwrap_or_else(|| "project".to_owned());
    let rooms = detect_init_rooms(&project_dir)?;
    let contents = render_project_config(&wing, &rooms);
    fs::write(&config_path, contents)?;

    Ok(config_path)
}

fn preferred_project_config_path(project_dir: &Path) -> PathBuf {
    let yaml = project_dir.join("mempalace.yaml");
    if yaml.exists() {
        return yaml;
    }

    let yml = project_dir.join("mempalace.yml");
    if yml.exists() {
        return yml;
    }

    project_dir.join("mempalace.yml")
}

type InitRooms = Vec<(String, String, Vec<String>)>;

fn detect_init_rooms(project_dir: &Path) -> Result<InitRooms, Box<dyn std::error::Error>> {
    let mut rooms = Vec::new();

    for entry in fs::read_dir(project_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() || should_skip_init_dir(&path) {
            continue;
        }

        let original = entry.file_name().to_string_lossy().to_string();
        let normalized = normalize_init_name(&original);
        if normalized.len() <= 2 {
            continue;
        }

        let room = INIT_ROOM_MAP
            .iter()
            .find_map(|(key, mapped)| (*key == normalized).then_some((*mapped).to_owned()))
            .unwrap_or_else(|| normalized.clone());

        if rooms.iter().any(|(name, _, _)| name == &room) {
            continue;
        }

        rooms.push((
            room.clone(),
            format!("Files from {original}/"),
            vec![room, original.to_lowercase()],
        ));
    }

    if !rooms.iter().any(|(name, _, _)| name == "general") {
        rooms.push((
            "general".to_owned(),
            "Files that don't fit other rooms".to_owned(),
            Vec::new(),
        ));
    }

    rooms.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(rooms)
}

fn should_skip_init_dir(path: &Path) -> bool {
    let Some(name) = path.file_name().map(|name| name.to_string_lossy()) else {
        return true;
    };

    matches!(
        name.as_ref(),
        ".git"
            | "node_modules"
            | "__pycache__"
            | ".venv"
            | "venv"
            | "env"
            | "dist"
            | "build"
            | ".next"
            | "coverage"
            | ".mempalace"
            | "target"
    )
}

fn normalize_init_name(name: &str) -> String {
    name.trim().replace(['-', ' '], "_").to_lowercase()
}

fn render_project_config(wing: &str, rooms: &[(String, String, Vec<String>)]) -> String {
    let mut output = String::new();
    output.push_str(&format!("wing: {wing}\n"));
    output.push_str("rooms:\n");

    for (name, description, keywords) in rooms {
        output.push_str(&format!("  - name: {name}\n"));
        output.push_str(&format!("    description: {}\n", yaml_quote(description)));
        if keywords.is_empty() {
            output.push_str("    keywords: []\n");
        } else {
            output.push_str("    keywords:\n");
            for keyword in keywords {
                output.push_str(&format!("      - {}\n", yaml_quote(keyword)));
            }
        }
    }

    output
}

fn yaml_quote(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

async fn print_status(app: &AppContext) -> Result<(), Box<dyn std::error::Error>> {
    let status = app.store.status().await?;
    let room_counts = app.store.room_counts().await?;

    println!("drawers: {}", status.total_drawers);
    println!("store: {}", app.store_path.display());
    println!("kg: {}", app.graph.db_path().display());

    if room_counts.is_empty() {
        println!("rooms: none yet");
    } else {
        let mut current_wing = String::new();
        for room in room_counts {
            if room.wing != current_wing {
                current_wing = room.wing.clone();
                println!();
                println!("wing: {}", current_wing);
            }
            println!("  {:20} {}", room.room, room.total_drawers);
        }
    }

    if legacy_chroma_detected(&app.palace_root, &app.store_path) {
        println!();
        println!(
            "note: legacy Chroma drawers are still in {}",
            app.palace_root.display()
        );
        println!("note: migrate them before expecting old search results here.");
    }

    Ok(())
}

async fn run_search(
    app: &AppContext,
    query: String,
    scope: Option<PathBuf>,
    wing: Option<String>,
    all_wings: bool,
    room: Option<String>,
    results: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let wing = resolve_search_wing(wing, all_wings, scope, env::current_dir().ok());
    let mut search = SearchQuery::new(query);
    search.limit = results;
    search.wing = wing;
    search.room = room;

    let hits = app.store.search(search).await?;
    if hits.is_empty() {
        println!("no results");
        if legacy_chroma_detected(&app.palace_root, &app.store_path) {
            println!(
                "note: search only covers the SQLite store at {}",
                app.store_path.display()
            );
        }
        return Ok(());
    }

    for hit in hits {
        let meta = &hit.drawer.metadata;
        println!("[{:.3}] {}/{}", hit.score, meta.wing, meta.room);
        if let Some(source_file) = &meta.source_file {
            println!("{}", source_file);
        }
        println!("{}", excerpt(&hit.drawer.content, 220));
        println!();
    }

    Ok(())
}

async fn run_mine(
    app: &AppContext,
    dir: PathBuf,
    options: MineOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("{:=<55}", "");
    println!("  MemPalace Mine");
    println!("{:=<55}", "");
    println!("  Mining:  {}", dir.display());
    println!("  Store:   {}", app.store_path.display());
    if options.dry_run {
        println!("  DRY RUN: true");
    }
    if options.skip_existing {
        println!("  Existing files: SKIP");
    }
    if options.exclude_data_files {
        println!("  Data files: EXCLUDED");
    }
    if !options.respect_gitignore {
        println!("  .gitignore: DISABLED");
    }
    println!("{:-<55}", "");

    let summary = mine_project(&app.store, dir, &options).await?;

    println!();
    println!("{:=<55}", "");
    println!("  Done.");
    println!("  Wing: {}", summary.wing);
    println!("  Files scanned: {}", summary.files_scanned);
    println!("  Files processed: {}", summary.files_processed);
    println!("  Files skipped: {}", summary.files_skipped);
    println!("  Files replaced: {}", summary.files_replaced);
    println!("  Drawers filed: {}", summary.total_drawers);

    if !summary.room_counts.is_empty() {
        println!();
        println!("  By room:");
        for (room, count) in summary.room_counts {
            println!("    {:20} {} files", room, count);
        }
    }

    if legacy_chroma_detected(&app.palace_root, &app.store_path) {
        println!();
        println!("  Note: legacy Chroma data was left untouched at:");
        println!("        {}", app.palace_root.display());
    }

    println!("{:=<55}", "");

    Ok(())
}

async fn run_remine(
    app: &AppContext,
    wing: Option<String>,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("{:=<55}", "");
    println!("  MemPalace Remine");
    println!("{:=<55}", "");
    println!("  Model: potion-base-32M (model2vec-rs)");
    if dry_run {
        println!("  DRY RUN: true");
    }

    let total = app.store.remine_all(wing.as_deref()).await?;
    if total == 0 {
        println!("  No drawers to remine.");
        return Ok(());
    }

    println!();
    println!("{:=<55}", "");
    println!("  Done.");
    println!("  Drawers re-embedded: {total}");
    println!("{:=<55}", "");

    Ok(())
}

async fn run_resize(app: &AppContext, max_elements: u64) -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("{:=<55}", "");
    println!("  MemPalace Resize");
    println!("{:=<55}", "");
    println!("  New max_elements: {max_elements}");
    println!("  Store: {}", app.store_path.display());
    println!("{:-<55}", "");

    let status_before = app.store.status().await?;
    println!("  Drawers before: {}", status_before.total_drawers);
    println!();

    app.store.resize_vectorlite_table(max_elements).await?;

    let status_after = app.store.status().await?;
    println!();
    println!("{:=<55}", "");
    println!("  Done.");
    println!("  Drawers after: {}", status_after.total_drawers);
    println!("  max_elements: {max_elements}");
    println!("{:=<55}", "");

    Ok(())
}

fn legacy_chroma_detected(palace_root: &Path, store_path: &Path) -> bool {
    palace_root != store_path && palace_root.join("chroma.sqlite3").exists()
}

fn excerpt(content: &str, max_chars: usize) -> String {
    let compact = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        compact
    } else {
        format!("{}...", compact.chars().take(max_chars).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use mempalace_core::MempalaceConfig;
    use tempfile::tempdir;

    use super::{
        Cli, Command, OnboardingPerson, ToolCommand, infer_wing_from_scope,
        looks_like_project_root, resolve_search_wing, run_project_entity_detection,
        write_aaak_bootstrap, write_entity_registry, write_project_config_scaffold,
    };
    use clap::Parser;

    #[test]
    fn search_accepts_scope_positional_argument() {
        let cli = Cli::try_parse_from(["mempalace-rs", "search", "aaak", "."]).unwrap();
        match cli.command {
            Command::Search { scope, .. } => assert_eq!(scope, Some(".".into())),
            command => panic!("unexpected command parsed: {command:?}"),
        }
    }

    #[test]
    fn init_accepts_project_dir() {
        let cli = Cli::try_parse_from(["mempalace-rs", "init", "."]).unwrap();
        match cli.command {
            Command::Init { dir, .. } => assert_eq!(dir, Some(".".into())),
            command => panic!("unexpected command parsed: {command:?}"),
        }
    }

    #[test]
    fn tool_namespace_accepts_mcp_style_subcommands() {
        let cli = Cli::try_parse_from(["mempalace-rs", "tool", "list_wings"]).unwrap();
        match cli.command {
            Command::Tool {
                command: ToolCommand::ListWings,
            } => {}
            command => panic!("unexpected command parsed: {command:?}"),
        }

        let cli = Cli::try_parse_from([
            "mempalace-rs",
            "tool",
            "kg_query",
            "--entity",
            "Riley",
            "--direction",
            "incoming",
        ])
        .unwrap();
        match cli.command {
            Command::Tool {
                command:
                    ToolCommand::KgQuery {
                        entity,
                        as_of,
                        direction,
                    },
            } => {
                assert_eq!(entity, "Riley");
                assert_eq!(as_of, None);
                assert_eq!(direction, "incoming");
            }
            command => panic!("unexpected command parsed: {command:?}"),
        }

        let cli = Cli::try_parse_from([
            "mempalace-rs",
            "tool",
            "diary_read",
            "--agent-name",
            "codex",
        ])
        .unwrap();
        match cli.command {
            Command::Tool {
                command: ToolCommand::DiaryRead { agent_name, last_n },
            } => {
                assert_eq!(agent_name, "codex");
                assert_eq!(last_n, 10);
            }
            command => panic!("unexpected command parsed: {command:?}"),
        }
    }

    #[test]
    fn resolve_search_wing_prefers_scope_then_cwd_project() {
        let scoped = tempdir().unwrap();
        std::fs::write(
            scoped.path().join("Cargo.toml"),
            "[package]\nname='scoped'\n",
        )
        .unwrap();
        let cwd = tempdir().unwrap();
        std::fs::write(cwd.path().join("Cargo.toml"), "[package]\nname='cwd'\n").unwrap();

        let wing = resolve_search_wing(
            None,
            false,
            Some(scoped.path().to_path_buf()),
            Some(cwd.path().to_path_buf()),
        );
        assert_eq!(wing, infer_wing_from_scope(scoped.path()));

        let wing = resolve_search_wing(None, false, None, Some(cwd.path().to_path_buf()));
        assert_eq!(wing, infer_wing_from_scope(cwd.path()));
        assert!(looks_like_project_root(cwd.path()));
    }

    #[test]
    fn init_writes_project_config_scaffold() {
        let tmp = tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::create_dir_all(tmp.path().join("docs")).unwrap();

        let config_path = write_project_config_scaffold(tmp.path().to_path_buf(), false).unwrap();
        let raw = std::fs::read_to_string(config_path).unwrap();

        assert!(raw.contains("wing:"));
        assert!(raw.contains("name: src"));
        assert!(raw.contains("name: documentation"));
        assert!(raw.contains("name: general"));
    }

    #[test]
    fn onboarding_bootstrap_writes_files() {
        let tmp = tempdir().unwrap();
        let config_dir = tmp.path().join(".mempalace");
        let mut config = MempalaceConfig::load_with_dir(&config_dir).unwrap();
        config.init().unwrap();
        config
            .save_onboarding(
                "combo".to_owned(),
                vec!["family".to_owned(), "projects".to_owned()],
                vec!["mempalace-rs".to_owned()],
            )
            .unwrap();

        let people = vec![OnboardingPerson {
            name: "Riley".to_owned(),
            relationship: "daughter".to_owned(),
            context: "personal".to_owned(),
        }];
        let projects = vec!["mempalace-rs".to_owned()];
        let wings = vec!["family".to_owned(), "projects".to_owned()];
        let mut aliases = BTreeMap::new();
        aliases.insert("Rye".to_owned(), "Riley".to_owned());
        config.save_people_map(aliases).unwrap();

        write_aaak_bootstrap(&config, &people, &projects, &wings, "combo").unwrap();
        write_entity_registry(&config, &people, &projects).unwrap();

        let entities = std::fs::read_to_string(config.aaak_entities_path()).unwrap();
        let facts = std::fs::read_to_string(config.critical_facts_path()).unwrap();
        let registry = std::fs::read_to_string(config.entity_registry_path()).unwrap();
        assert!(entities.contains("RIL=Riley"));
        assert!(entities.contains("MEM=mempalace-rs"));
        assert!(facts.contains("Mode: combo"));
        assert!(facts.contains("Wings: family, projects"));
        assert!(registry.contains("\"projects\": ["));
        assert!(registry.contains("\"Riley\""));
        assert!(registry.contains("\"Rye\""));
        assert_eq!(
            config
                .entity_registry_path()
                .file_name()
                .and_then(|name| name.to_str()),
            Some("entity_registry.json")
        );
    }

    #[test]
    fn init_entity_detection_writes_project_entities() {
        let tmp = tempdir().unwrap();
        let notes = [
            "Riley said the Lantern roadmap is ready.",
            "Riley smiled. Riley asked about Lantern v2.",
            "Thanks Riley for shipping Lantern.",
            "We are building Lantern and the Lantern architecture is changing.",
            "pip install Lantern",
        ]
        .join("\n");
        std::fs::write(tmp.path().join("notes.md"), notes).unwrap();

        run_project_entity_detection(tmp.path(), true).unwrap();

        let entities = std::fs::read_to_string(tmp.path().join("entities.json")).unwrap();
        assert!(entities.contains("Riley"));
        assert!(entities.contains("Lantern"));
    }
}
