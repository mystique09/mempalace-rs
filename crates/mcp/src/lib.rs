use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    path::{Path, PathBuf},
};

use chrono::Utc;
use mempalace_core::{
    Drawer, DrawerMetadata, KnowledgeGraph, MemoryStore, MempalaceConfig, SearchQuery,
};
use mempalace_store::LanceMemoryStore;
use rmcp::{
    ErrorData, Json, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::stdio,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

const PALACE_PROTOCOL: &str = r#"IMPORTANT - MemPalace Memory Protocol:
1. ON WAKE-UP: Call mempalace_status to load palace overview + AAAK spec.
2. BEFORE RESPONDING about any person, project, or past event: call mempalace_kg_query or mempalace_search FIRST. Never guess - verify.
3. IF UNSURE about a fact (name, gender, age, relationship): say "let me check" and query the palace. Wrong is worse than slow.
4. AFTER EACH SESSION: call mempalace_diary_write to record what happened, what you learned, what matters.
5. WHEN FACTS CHANGE: call mempalace_kg_invalidate on the old fact, mempalace_kg_add for the new one.

This protocol ensures the AI KNOWS before it speaks. Storage is not memory - but storage + this protocol = memory."#;

const AAAK_SPEC: &str = r#"AAAK is a compressed memory dialect that MemPalace uses for efficient storage.
It is designed to be readable by both humans and LLMs without decoding.

FORMAT:
  ENTITIES: 3-letter uppercase codes. ALC=Alice, JOR=Jordan, RIL=Riley, MAX=Max, BEN=Ben.
  EMOTIONS: *action markers* before/during text. *warm*=joy, *fierce*=determined, *raw*=vulnerable, *bloom*=tenderness.
  STRUCTURE: Pipe-separated fields. FAM: family | PROJ: projects | WARN: warnings/reminders.
  DATES: ISO format (2026-03-31). COUNTS: Nx = N mentions (e.g. 570x).
  IMPORTANCE: 1-5 stars.
  HALLS: hall_facts, hall_events, hall_discoveries, hall_preferences, hall_advice.
  WINGS: wing_user, wing_agent, wing_team, wing_code, wing_myproject, wing_hardware, wing_ue5, wing_ai_research.
  ROOMS: Hyphenated slugs representing named ideas (e.g. chromadb-setup, gpu-pricing).

EXAMPLE:
  FAM: ALC->JOR | 2D(kids): RIL(18,sports) MAX(11,chess+swimming) | BEN(contributor)

Read AAAK naturally - expand codes mentally, treat *markers* as emotional context.
When WRITING AAAK: use entity codes, mark emotions, keep structure tight."#;

type McpResult = Result<Json<Value>, ErrorData>;

#[derive(Clone)]
struct AppContext {
    palace_root: PathBuf,
    store: LanceMemoryStore,
    graph: KnowledgeGraph,
}

#[derive(Default, Clone)]
struct GraphNode {
    wings: BTreeSet<String>,
    halls: BTreeSet<String>,
    dates: BTreeSet<String>,
    count: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ListRoomsRequest {
    wing: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchRequest {
    query: String,
    limit: Option<usize>,
    wing: Option<String>,
    room: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CheckDuplicateRequest {
    content: String,
    threshold: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AddDrawerRequest {
    wing: String,
    room: String,
    content: String,
    source_file: Option<String>,
    added_by: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DeleteDrawerRequest {
    drawer_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct KgQueryRequest {
    entity: String,
    as_of: Option<String>,
    direction: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct KgAddRequest {
    subject: String,
    predicate: String,
    object: String,
    valid_from: Option<String>,
    source_closet: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct KgInvalidateRequest {
    subject: String,
    predicate: String,
    object: String,
    ended: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct KgTimelineRequest {
    entity: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DiaryWriteRequest {
    agent_name: String,
    entry: String,
    topic: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DiaryReadRequest {
    agent_name: String,
    last_n: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct TraverseRequest {
    start_room: String,
    max_hops: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FindTunnelsRequest {
    wing_a: Option<String>,
    wing_b: Option<String>,
}

#[derive(Clone)]
pub struct McpServer {
    app: AppContext,
    tool_router: ToolRouter<Self>,
}

impl McpServer {
    pub async fn open() -> Result<Self, Box<dyn std::error::Error>> {
        let config = MempalaceConfig::load()?;
        config.init()?;

        let palace_root = config.palace_path();
        let store_path = MempalaceConfig::resolve_store_path(&palace_root);
        let fastembed_cache_path = config.fastembed_cache_path();
        let onnxruntime_dylib_path = config.onnxruntime_dylib_path();

        fs::create_dir_all(&palace_root)?;
        fs::create_dir_all(&store_path)?;
        fs::create_dir_all(&fastembed_cache_path)?;
        seed_onnxruntime_dylib(&onnxruntime_dylib_path)?;

        let store =
            LanceMemoryStore::new(&store_path, config.collection_name(), &fastembed_cache_path)?;
        let graph = KnowledgeGraph::new(config.knowledge_graph_path())?;

        Ok(Self {
            app: AppContext {
                palace_root,
                store,
                graph,
            },
            tool_router: Self::tool_router(),
        })
    }

    pub async fn run_stdio() -> Result<(), Box<dyn std::error::Error>> {
        let server = Self::open().await?;
        let service = server.serve(stdio()).await?;
        service.waiting().await?;
        Ok(())
    }

    async fn tool_status(&self) -> Result<Value, String> {
        let status = self
            .app
            .store
            .status()
            .await
            .map_err(|error| error.to_string())?;
        let room_counts = self
            .app
            .store
            .room_counts()
            .await
            .map_err(|error| error.to_string())?;

        let mut wings = BTreeMap::<String, usize>::new();
        let mut rooms = BTreeMap::<String, usize>::new();
        for room in room_counts {
            *wings.entry(room.wing.clone()).or_insert(0) += room.total_drawers;
            *rooms.entry(room.room).or_insert(0) += room.total_drawers;
        }

        Ok(json!({
            "total_drawers": status.total_drawers,
            "wings": wings,
            "rooms": rooms,
            "palace_path": self.app.palace_root.display().to_string(),
            "protocol": PALACE_PROTOCOL,
            "aaak_dialect": AAAK_SPEC,
        }))
    }

    async fn tool_list_wings(&self) -> Result<Value, String> {
        let room_counts = self
            .app
            .store
            .room_counts()
            .await
            .map_err(|error| error.to_string())?;
        let mut wings = BTreeMap::<String, usize>::new();
        for room in room_counts {
            *wings.entry(room.wing).or_insert(0) += room.total_drawers;
        }
        Ok(json!({ "wings": wings }))
    }

    async fn tool_list_rooms(&self, wing: Option<String>) -> Result<Value, String> {
        let room_counts = self
            .app
            .store
            .room_counts()
            .await
            .map_err(|error| error.to_string())?;
        let mut rooms = BTreeMap::<String, usize>::new();
        for room in room_counts {
            if wing.as_ref().is_none_or(|selected| selected == &room.wing) {
                *rooms.entry(room.room).or_insert(0) += room.total_drawers;
            }
        }
        Ok(json!({ "wing": wing.unwrap_or_else(|| "all".to_owned()), "rooms": rooms }))
    }

    async fn tool_get_taxonomy(&self) -> Result<Value, String> {
        let room_counts = self
            .app
            .store
            .room_counts()
            .await
            .map_err(|error| error.to_string())?;
        let mut taxonomy = BTreeMap::<String, BTreeMap<String, usize>>::new();
        for room in room_counts {
            taxonomy
                .entry(room.wing)
                .or_default()
                .insert(room.room, room.total_drawers);
        }
        Ok(json!({ "taxonomy": taxonomy }))
    }

    async fn tool_search(
        &self,
        query: String,
        limit: usize,
        wing: Option<String>,
        room: Option<String>,
    ) -> Result<Value, String> {
        let mut search = SearchQuery::new(query.clone());
        search.limit = limit;
        search.wing = wing.clone();
        search.room = room.clone();
        let hits = self
            .app
            .store
            .search(search)
            .await
            .map_err(|error| error.to_string())?;

        let results = hits
            .into_iter()
            .map(|hit| {
                json!({
                    "text": hit.drawer.content,
                    "wing": hit.drawer.metadata.wing,
                    "room": hit.drawer.metadata.room,
                    "source_file": hit.drawer.metadata.source_file.as_deref().and_then(file_name).unwrap_or("?"),
                    "similarity": round3(hit.score),
                })
            })
            .collect::<Vec<_>>();

        Ok(json!({
            "query": query,
            "filters": {"wing": wing, "room": room},
            "results": results,
        }))
    }

    async fn tool_check_duplicate(&self, content: String, threshold: f32) -> Result<Value, String> {
        let mut query = SearchQuery::new(content.clone());
        query.limit = 5;
        let hits = self
            .app
            .store
            .search(query)
            .await
            .map_err(|error| error.to_string())?;

        let matches = hits
            .into_iter()
            .filter(|hit| hit.score >= threshold)
            .map(|hit| {
                json!({
                    "id": hit.drawer.id,
                    "wing": hit.drawer.metadata.wing,
                    "room": hit.drawer.metadata.room,
                    "similarity": round3(hit.score),
                    "content": excerpt(&hit.drawer.content, 200),
                })
            })
            .collect::<Vec<_>>();

        Ok(json!({
            "is_duplicate": !matches.is_empty(),
            "matches": matches,
        }))
    }

    async fn tool_add_drawer(
        &self,
        wing: String,
        room: String,
        content: String,
        source_file: Option<String>,
        added_by: String,
    ) -> Result<Value, String> {
        let duplicate = self.tool_check_duplicate(content.clone(), 0.9).await?;
        if duplicate
            .get("is_duplicate")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Ok(json!({
                "success": false,
                "reason": "duplicate",
                "matches": duplicate.get("matches").cloned().unwrap_or_else(|| json!([])),
            }));
        }

        let drawer_id = format!("drawer_{}_{}_{}", wing, room, Uuid::now_v7().simple());
        let drawer = Drawer {
            id: drawer_id.clone(),
            content,
            metadata: DrawerMetadata {
                wing: wing.clone(),
                room: room.clone(),
                source_file,
                chunk_index: 0,
                added_by,
                filed_at: Some(Utc::now().to_rfc3339()),
            },
        };
        self.app
            .store
            .add_drawer(drawer)
            .await
            .map_err(|error| error.to_string())?;

        Ok(json!({
            "success": true,
            "drawer_id": drawer_id,
            "wing": wing,
            "room": room,
        }))
    }

    async fn tool_delete_drawer(&self, drawer_id: String) -> Result<Value, String> {
        let deleted = self
            .app
            .store
            .delete_drawer(&drawer_id)
            .await
            .map_err(|error| error.to_string())?;

        if deleted {
            Ok(json!({"success": true, "drawer_id": drawer_id}))
        } else {
            Ok(json!({"success": false, "error": format!("Drawer not found: {drawer_id}")}))
        }
    }

    async fn tool_kg_query(
        &self,
        entity: String,
        as_of: Option<String>,
        direction: String,
    ) -> Result<Value, String> {
        let direction = normalize_direction(&direction);
        let facts = self
            .app
            .graph
            .query_entity(&entity, as_of.as_deref(), &direction)
            .map_err(|error| error.to_string())?;
        let count = facts.len();
        Ok(json!({
            "entity": entity,
            "as_of": as_of,
            "facts": facts,
            "count": count,
        }))
    }

    async fn tool_kg_add(
        &self,
        subject: String,
        predicate: String,
        object: String,
        valid_from: Option<String>,
        source_closet: Option<String>,
    ) -> Result<Value, String> {
        let triple_id = self
            .app
            .graph
            .add_triple(
                &subject,
                &predicate,
                &object,
                valid_from.as_deref(),
                None,
                1.0,
                source_closet.as_deref(),
                None,
            )
            .map_err(|error| error.to_string())?;

        Ok(json!({
            "success": true,
            "triple_id": triple_id,
            "fact": format!("{subject} -> {predicate} -> {object}"),
        }))
    }

    async fn tool_kg_invalidate(
        &self,
        subject: String,
        predicate: String,
        object: String,
        ended: Option<String>,
    ) -> Result<Value, String> {
        self.app
            .graph
            .invalidate(&subject, &predicate, &object, ended.as_deref())
            .map_err(|error| error.to_string())?;
        Ok(json!({
            "success": true,
            "fact": format!("{subject} -> {predicate} -> {object}"),
            "ended": ended.unwrap_or_else(|| "today".to_owned()),
        }))
    }

    async fn tool_kg_timeline(&self, entity: Option<String>) -> Result<Value, String> {
        let timeline = self
            .app
            .graph
            .timeline(entity.as_deref())
            .map_err(|error| error.to_string())?;
        let count = timeline.len();
        Ok(json!({
            "entity": entity.clone().unwrap_or_else(|| "all".to_owned()),
            "timeline": timeline,
            "count": count,
        }))
    }

    async fn tool_kg_stats(&self) -> Result<Value, String> {
        let stats = self.app.graph.stats().map_err(|error| error.to_string())?;
        serde_json::to_value(stats).map_err(|error| error.to_string())
    }

    async fn tool_diary_write(
        &self,
        agent_name: String,
        entry: String,
        topic: String,
    ) -> Result<Value, String> {
        let wing = format!("wing_{}", slugify(&agent_name));
        let now = Utc::now();
        let entry_id = format!("diary_{}_{}", wing, Uuid::now_v7().simple());
        let drawer = Drawer {
            id: entry_id.clone(),
            content: entry,
            metadata: DrawerMetadata {
                wing: wing.clone(),
                room: "diary".to_owned(),
                source_file: Some(format!("diary/{topic}")),
                chunk_index: 0,
                added_by: agent_name.clone(),
                filed_at: Some(now.to_rfc3339()),
            },
        };
        self.app
            .store
            .add_drawer(drawer)
            .await
            .map_err(|error| error.to_string())?;

        Ok(json!({
            "success": true,
            "entry_id": entry_id,
            "agent": agent_name,
            "topic": topic,
            "timestamp": now.to_rfc3339(),
        }))
    }

    async fn tool_diary_read(&self, agent_name: String, last_n: usize) -> Result<Value, String> {
        let wing = format!("wing_{}", slugify(&agent_name));
        let mut drawers = self
            .app
            .store
            .list_drawers(Some(&wing))
            .await
            .map_err(|error| error.to_string())?;
        drawers.retain(|drawer| drawer.metadata.room == "diary");
        drawers.sort_by(|a, b| b.metadata.filed_at.cmp(&a.metadata.filed_at));

        let total = drawers.len();
        let entries = drawers
            .into_iter()
            .take(last_n)
            .map(|drawer| {
                json!({
                    "date": drawer.metadata.filed_at.as_deref().map(short_date).unwrap_or_default(),
                    "timestamp": drawer.metadata.filed_at,
                    "topic": drawer.metadata.source_file.as_deref().and_then(diary_topic).unwrap_or("general"),
                    "content": drawer.content,
                })
            })
            .collect::<Vec<_>>();
        let showing = entries.len();

        Ok(json!({
            "agent": agent_name,
            "entries": entries,
            "total": total,
            "showing": showing,
        }))
    }

    async fn tool_traverse(&self, start_room: String, max_hops: usize) -> Result<Value, String> {
        let (nodes, _) = self.build_graph().await?;
        let Some(start) = nodes.get(&start_room) else {
            return Ok(json!({
                "error": format!("Room '{start_room}' not found"),
                "suggestions": fuzzy_match(&start_room, &nodes),
            }));
        };

        let mut visited = HashSet::from([start_room.clone()]);
        let mut results = vec![json!({
            "room": start_room.clone(),
            "wings": start.wings.iter().cloned().collect::<Vec<_>>(),
            "halls": start.halls.iter().cloned().collect::<Vec<_>>(),
            "count": start.count,
            "hop": 0,
        })];

        let mut frontier = vec![(start_room, 0usize)];
        while let Some((current_room, depth)) = frontier.first().cloned() {
            frontier.remove(0);
            if depth >= max_hops {
                continue;
            }

            let current = nodes.get(&current_room).expect("visited room should exist");
            for (room, data) in &nodes {
                if visited.contains(room) {
                    continue;
                }

                let shared = current
                    .wings
                    .intersection(&data.wings)
                    .cloned()
                    .collect::<Vec<_>>();
                if shared.is_empty() {
                    continue;
                }

                visited.insert(room.clone());
                results.push(json!({
                    "room": room,
                    "wings": data.wings.iter().cloned().collect::<Vec<_>>(),
                    "halls": data.halls.iter().cloned().collect::<Vec<_>>(),
                    "count": data.count,
                    "hop": depth + 1,
                    "connected_via": shared,
                }));
                if depth + 1 < max_hops {
                    frontier.push((room.clone(), depth + 1));
                }
            }
        }

        results.sort_by_key(|value| {
            (
                value.get("hop").and_then(Value::as_u64).unwrap_or(0),
                std::cmp::Reverse(value.get("count").and_then(Value::as_u64).unwrap_or(0)),
            )
        });
        Ok(json!(results))
    }

    async fn tool_find_tunnels(
        &self,
        wing_a: Option<String>,
        wing_b: Option<String>,
    ) -> Result<Value, String> {
        let (nodes, _) = self.build_graph().await?;
        let mut tunnels = Vec::new();

        for (room, data) in nodes {
            if data.wings.len() < 2 {
                continue;
            }
            if wing_a
                .as_ref()
                .is_some_and(|wing| !data.wings.contains(wing))
            {
                continue;
            }
            if wing_b
                .as_ref()
                .is_some_and(|wing| !data.wings.contains(wing))
            {
                continue;
            }

            tunnels.push(json!({
                "room": room,
                "wings": data.wings.into_iter().collect::<Vec<_>>(),
                "halls": data.halls.into_iter().collect::<Vec<_>>(),
                "count": data.count,
                "recent": data.dates.last().cloned().unwrap_or_default(),
            }));
        }

        tunnels.sort_by_key(|value| {
            std::cmp::Reverse(value.get("count").and_then(Value::as_u64).unwrap_or(0))
        });
        tunnels.truncate(50);
        Ok(json!(tunnels))
    }

    async fn tool_graph_stats(&self) -> Result<Value, String> {
        let (nodes, total_edges) = self.build_graph().await?;
        let tunnel_rooms = nodes.values().filter(|data| data.wings.len() >= 2).count();
        let mut rooms_per_wing = BTreeMap::<String, usize>::new();
        let mut top_tunnels = Vec::new();

        for (room, data) in &nodes {
            for wing in &data.wings {
                *rooms_per_wing.entry(wing.clone()).or_insert(0) += 1;
            }
            if data.wings.len() >= 2 {
                top_tunnels.push(json!({
                    "room": room,
                    "wings": data.wings.iter().cloned().collect::<Vec<_>>(),
                    "count": data.count,
                }));
            }
        }

        top_tunnels.sort_by_key(|value| {
            std::cmp::Reverse(
                value
                    .get("wings")
                    .and_then(Value::as_array)
                    .map(|wings| wings.len())
                    .unwrap_or(0),
            )
        });
        top_tunnels.truncate(10);

        Ok(json!({
            "total_rooms": nodes.len(),
            "tunnel_rooms": tunnel_rooms,
            "total_edges": total_edges,
            "rooms_per_wing": rooms_per_wing,
            "top_tunnels": top_tunnels,
        }))
    }

    async fn build_graph(&self) -> Result<(BTreeMap<String, GraphNode>, usize), String> {
        let drawers = self
            .app
            .store
            .list_drawers(None)
            .await
            .map_err(|error| error.to_string())?;
        let mut nodes = BTreeMap::<String, GraphNode>::new();

        for drawer in drawers {
            let room = drawer.metadata.room.clone();
            if room.is_empty() || room == "general" {
                continue;
            }
            let node = nodes.entry(room).or_default();
            node.wings.insert(drawer.metadata.wing.clone());
            if let Some(date) = drawer.metadata.filed_at.as_deref().map(short_date) {
                node.dates.insert(date);
            }
            node.count += 1;
        }

        let mut total_edges = 0usize;
        for data in nodes.values() {
            let wings = data.wings.len();
            if wings >= 2 {
                total_edges += wings * (wings - 1) / 2;
            }
        }

        Ok((nodes, total_edges))
    }
}

#[tool_router]
impl McpServer {
    #[tool(description = "Palace overview - total drawers, wing and room counts")]
    async fn mempalace_status(&self) -> McpResult {
        self.tool_status().await.map(Json).map_err(internal_error)
    }

    #[tool(description = "List all wings with drawer counts")]
    async fn mempalace_list_wings(&self) -> McpResult {
        self.tool_list_wings()
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "List rooms within a wing (or all rooms if no wing given)")]
    async fn mempalace_list_rooms(
        &self,
        Parameters(request): Parameters<ListRoomsRequest>,
    ) -> McpResult {
        self.tool_list_rooms(request.wing)
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "Full taxonomy: wing -> room -> drawer count")]
    async fn mempalace_get_taxonomy(&self) -> McpResult {
        self.tool_get_taxonomy()
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(
        description = "Get the AAAK dialect specification - the compressed memory format MemPalace uses."
    )]
    async fn mempalace_get_aaak_spec(&self) -> McpResult {
        Ok(Json(json!({ "aaak_spec": AAAK_SPEC })))
    }

    #[tool(
        description = "Semantic search. Returns verbatim drawer content with similarity scores."
    )]
    async fn mempalace_search(&self, Parameters(request): Parameters<SearchRequest>) -> McpResult {
        self.tool_search(
            request.query,
            request.limit.unwrap_or(5),
            request.wing,
            request.room,
        )
        .await
        .map(Json)
        .map_err(internal_error)
    }

    #[tool(description = "Check if content already exists in the palace before filing")]
    async fn mempalace_check_duplicate(
        &self,
        Parameters(request): Parameters<CheckDuplicateRequest>,
    ) -> McpResult {
        self.tool_check_duplicate(request.content, request.threshold.unwrap_or(0.9))
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "File verbatim content into the palace. Checks for duplicates first.")]
    async fn mempalace_add_drawer(
        &self,
        Parameters(request): Parameters<AddDrawerRequest>,
    ) -> McpResult {
        self.tool_add_drawer(
            request.wing,
            request.room,
            request.content,
            request.source_file,
            request.added_by.unwrap_or_else(|| "mcp".to_owned()),
        )
        .await
        .map(Json)
        .map_err(internal_error)
    }

    #[tool(description = "Delete a drawer by ID. Irreversible.")]
    async fn mempalace_delete_drawer(
        &self,
        Parameters(request): Parameters<DeleteDrawerRequest>,
    ) -> McpResult {
        self.tool_delete_drawer(request.drawer_id)
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "Query the knowledge graph for an entity's relationships.")]
    async fn mempalace_kg_query(
        &self,
        Parameters(request): Parameters<KgQueryRequest>,
    ) -> McpResult {
        self.tool_kg_query(
            request.entity,
            request.as_of,
            request.direction.unwrap_or_else(|| "both".to_owned()),
        )
        .await
        .map(Json)
        .map_err(internal_error)
    }

    #[tool(description = "Add a fact to the knowledge graph.")]
    async fn mempalace_kg_add(&self, Parameters(request): Parameters<KgAddRequest>) -> McpResult {
        self.tool_kg_add(
            request.subject,
            request.predicate,
            request.object,
            request.valid_from,
            request.source_closet,
        )
        .await
        .map(Json)
        .map_err(internal_error)
    }

    #[tool(description = "Mark a fact as no longer true.")]
    async fn mempalace_kg_invalidate(
        &self,
        Parameters(request): Parameters<KgInvalidateRequest>,
    ) -> McpResult {
        self.tool_kg_invalidate(
            request.subject,
            request.predicate,
            request.object,
            request.ended,
        )
        .await
        .map(Json)
        .map_err(internal_error)
    }

    #[tool(description = "Chronological timeline of facts.")]
    async fn mempalace_kg_timeline(
        &self,
        Parameters(request): Parameters<KgTimelineRequest>,
    ) -> McpResult {
        self.tool_kg_timeline(request.entity)
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "Knowledge graph overview.")]
    async fn mempalace_kg_stats(&self) -> McpResult {
        self.tool_kg_stats().await.map(Json).map_err(internal_error)
    }

    #[tool(description = "Write to your personal agent diary in AAAK format.")]
    async fn mempalace_diary_write(
        &self,
        Parameters(request): Parameters<DiaryWriteRequest>,
    ) -> McpResult {
        self.tool_diary_write(
            request.agent_name,
            request.entry,
            request.topic.unwrap_or_else(|| "general".to_owned()),
        )
        .await
        .map(Json)
        .map_err(internal_error)
    }

    #[tool(description = "Read your recent diary entries (in AAAK).")]
    async fn mempalace_diary_read(
        &self,
        Parameters(request): Parameters<DiaryReadRequest>,
    ) -> McpResult {
        self.tool_diary_read(request.agent_name, request.last_n.unwrap_or(10))
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "Walk the palace graph from a room.")]
    async fn mempalace_traverse(
        &self,
        Parameters(request): Parameters<TraverseRequest>,
    ) -> McpResult {
        self.tool_traverse(request.start_room, request.max_hops.unwrap_or(2))
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "Find rooms that bridge two wings.")]
    async fn mempalace_find_tunnels(
        &self,
        Parameters(request): Parameters<FindTunnelsRequest>,
    ) -> McpResult {
        self.tool_find_tunnels(request.wing_a, request.wing_b)
            .await
            .map(Json)
            .map_err(internal_error)
    }

    #[tool(description = "Palace graph overview.")]
    async fn mempalace_graph_stats(&self) -> McpResult {
        self.tool_graph_stats()
            .await
            .map(Json)
            .map_err(internal_error)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for McpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(PALACE_PROTOCOL)
            .with_server_info(Implementation::new("mempalace", env!("CARGO_PKG_VERSION")))
    }
}

fn internal_error(message: impl ToString) -> ErrorData {
    ErrorData::internal_error(message.to_string(), None)
}

fn file_name(path: &str) -> Option<&str> {
    Path::new(path).file_name().and_then(|name| name.to_str())
}

fn normalize_direction(direction: &str) -> String {
    match direction {
        "outgoing" | "incoming" | "both" => direction.to_owned(),
        _ => "both".to_owned(),
    }
}

fn short_date(value: &str) -> String {
    value.chars().take(10).collect()
}

fn diary_topic(path: &str) -> Option<&str> {
    path.strip_prefix("diary/").or(Some(path))
}

fn excerpt(content: &str, max_chars: usize) -> String {
    let compact = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        compact
    } else {
        format!("{}...", compact.chars().take(max_chars).collect::<String>())
    }
}

fn round3(value: f32) -> f32 {
    (value * 1000.0).round() / 1000.0
}

fn slugify(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .to_owned()
}

fn fuzzy_match(query: &str, nodes: &BTreeMap<String, GraphNode>) -> Vec<String> {
    let query = query.to_lowercase();
    let mut matches = nodes
        .keys()
        .filter_map(|room| {
            if room.contains(&query) {
                Some((room.clone(), 1.0f32))
            } else if query
                .split('-')
                .any(|word| !word.is_empty() && room.contains(word))
            {
                Some((room.clone(), 0.5))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    matches.into_iter().take(5).map(|(room, _)| room).collect()
}

fn seed_onnxruntime_dylib(target_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if target_path.is_file() {
        return Ok(());
    }

    let Some(parent) = target_path.parent() else {
        return Ok(());
    };
    fs::create_dir_all(parent)?;

    for candidate in bundled_onnxruntime_candidates() {
        if candidate.is_file() {
            fs::copy(candidate, target_path)?;
            break;
        }
    }

    Ok(())
}

fn bundled_onnxruntime_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("onnxruntime.dll"));
            candidates.push(parent.join(".mempalace-bin").join("onnxruntime.dll"));
        }
    }

    if let Some(home_dir) = dirs::home_dir() {
        candidates.push(home_dir.join(".mempalace-bin").join("onnxruntime.dll"));
    }

    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .map(|path| path.join(".mempalace-bin").join("onnxruntime.dll"))
            .unwrap_or_else(|| PathBuf::from(".mempalace-bin").join("onnxruntime.dll")),
    );

    candidates
}

#[cfg(test)]
mod tests {
    use super::{GraphNode, diary_topic, excerpt, fuzzy_match, slugify};
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn slugify_normalizes_agent_names() {
        assert_eq!(slugify("Codex Agent"), "codex_agent");
    }

    #[test]
    fn fuzzy_match_prefers_substrings() {
        let mut nodes = BTreeMap::new();
        nodes.insert(
            "chromadb-setup".to_owned(),
            GraphNode {
                wings: BTreeSet::new(),
                halls: BTreeSet::new(),
                dates: BTreeSet::new(),
                count: 1,
            },
        );
        assert_eq!(
            fuzzy_match("chroma", &nodes),
            vec!["chromadb-setup".to_owned()]
        );
    }

    #[test]
    fn helper_functions_trim_diary_and_excerpt() {
        assert_eq!(diary_topic("diary/general"), Some("general"));
        assert!(excerpt("a b c d e f", 3).ends_with("..."));
    }
}
