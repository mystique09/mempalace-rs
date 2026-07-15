use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    process::Command,
    time::Instant,
};

use clap::{Parser, ValueEnum};
use mempalace_core::{
    ContentKind, Drawer, DrawerMetadata, MemoryStore, MempalaceConfig, SearchQuery,
};
use mempalace_mcp::McpServer;
use mempalace_store::SqliteMemoryStore;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(
    name = "multidomain",
    about = "Run the checked-in multi-domain semantic retrieval regression suite"
)]
struct Args {
    #[arg(default_value = "benchmarks/multi_domain_queries.json")]
    corpus: PathBuf,
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = SelectedSplit::All)]
    split: SelectedSplit,
    #[arg(long)]
    output: Option<PathBuf>,
    /// Compare the direct CLI search path with the MCP wrapper for every domain.
    #[arg(long)]
    check_parity: bool,
    /// Path to the real mempalace-rs binary used by --check-parity.
    #[arg(long)]
    cli_binary: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SelectedSplit {
    All,
    Dev,
    HeldOut,
}

#[derive(Debug, Deserialize)]
struct Corpus {
    documents: Vec<CorpusDocument>,
    baseline_recall_at_5: BTreeMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct CorpusDocument {
    id: String,
    domain: String,
    content_kind: ContentKind,
    wing: String,
    room: String,
    source_file: Option<String>,
    content: String,
    #[serde(default)]
    retrieval_text: Option<String>,
    #[serde(default)]
    filed_at: Option<String>,
    target_matcher: TargetMatcher,
    queries: Vec<CorpusQuery>,
}

#[derive(Debug, Deserialize)]
struct TargetMatcher {
    #[serde(default)]
    drawer_id: Option<String>,
    #[serde(default)]
    relative_path: Option<String>,
    #[serde(default)]
    content_contains: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CorpusQuery {
    id: String,
    text: String,
    split: QuerySplit,
    #[serde(default)]
    also_relevant: Vec<String>,
    #[serde(default)]
    as_of: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum QuerySplit {
    Dev,
    HeldOut,
}

impl SelectedSplit {
    fn includes(self, split: QuerySplit) -> bool {
        match self {
            Self::All => true,
            Self::Dev => split == QuerySplit::Dev,
            Self::HeldOut => split == QuerySplit::HeldOut,
        }
    }
}

#[derive(Debug, Serialize)]
struct QueryResult {
    id: String,
    domain: String,
    split: &'static str,
    query: String,
    relevant_ids: Vec<String>,
    ranked_ids: Vec<String>,
    recall_at_5: f64,
    reciprocal_rank_at_5: f64,
    elapsed_ms: u128,
}

#[derive(Debug, Default)]
struct Metrics {
    count: usize,
    hits: f64,
    reciprocal_rank: f64,
}

impl Metrics {
    fn record(&mut self, result: &QueryResult) {
        self.count += 1;
        self.hits += result.recall_at_5;
        self.reciprocal_rank += result.reciprocal_rank_at_5;
    }

    fn recall(&self) -> f64 {
        self.hits / self.count.max(1) as f64
    }

    fn mrr(&self) -> f64 {
        self.reciprocal_rank / self.count.max(1) as f64
    }
}

fn validate_corpus(corpus: &Corpus) -> Result<(), String> {
    let document_ids = corpus
        .documents
        .iter()
        .map(|document| document.id.as_str())
        .collect::<HashSet<_>>();
    if document_ids.len() != corpus.documents.len() {
        return Err("document IDs must be unique".to_owned());
    }

    let mut query_ids = HashSet::new();
    let mut domain_counts = BTreeMap::<String, usize>::new();
    let mut kind_counts = [0usize; 6];
    for document in &corpus.documents {
        if document.queries.is_empty() {
            return Err(format!("document {} has no labelled queries", document.id));
        }
        if document.content_kind == ContentKind::Code
            && (document.target_matcher.relative_path.is_none()
                || document.target_matcher.content_contains.is_empty())
        {
            return Err(format!(
                "code document {} needs a relative path and content matcher",
                document.id
            ));
        }
        if document.target_matcher.drawer_id.is_none()
            && document.target_matcher.relative_path.is_none()
            && document.target_matcher.content_contains.is_empty()
        {
            return Err(format!("document {} has no target matcher", document.id));
        }
        for query in &document.queries {
            *domain_counts.entry(document.domain.clone()).or_default() += 1;
            kind_counts[match document.content_kind {
                ContentKind::Code => 0,
                ContentKind::Conversation => 1,
                ContentKind::Documentation => 2,
                ContentKind::Diary => 3,
                ContentKind::Prose => 4,
                ContentKind::Unknown => 5,
            }] += 1;
            if !query_ids.insert(query.id.as_str()) {
                return Err(format!("query ID {} is duplicated", query.id));
            }
            for relevant in &query.also_relevant {
                if !document_ids.contains(relevant.as_str()) {
                    return Err(format!(
                        "query {} references missing document {}",
                        query.id, relevant
                    ));
                }
            }
        }
    }
    for domain in domain_counts.keys() {
        if !corpus.baseline_recall_at_5.contains_key(domain) {
            return Err(format!("domain {domain} has no frozen Recall@5 baseline"));
        }
    }
    for (label, actual, minimum) in [
        ("code", kind_counts[0], 30),
        ("conversation", kind_counts[1], 25),
        ("documentation", kind_counts[2], 20),
        ("diary", kind_counts[3], 15),
        ("prose", kind_counts[4], 10),
    ] {
        if actual < minimum {
            return Err(format!(
                "{label} has {actual} queries; the corpus contract requires at least {minimum}"
            ));
        }
    }
    Ok(())
}

fn matches_target(drawer: &Drawer, target: &CorpusDocument) -> bool {
    let matcher = &target.target_matcher;
    matcher.drawer_id.as_ref().is_none_or(|id| drawer.id == *id)
        && matcher
            .relative_path
            .as_ref()
            .is_none_or(|path| drawer.metadata.source_file.as_deref() == Some(path.as_str()))
        && matcher
            .content_contains
            .iter()
            .all(|needle| drawer.content.contains(needle))
}

fn to_drawer(document: &CorpusDocument) -> Drawer {
    Drawer {
        id: document.id.clone(),
        content: document.content.clone(),
        retrieval_text: document.retrieval_text.clone(),
        metadata: DrawerMetadata {
            content_kind: document.content_kind,
            wing: document.wing.clone(),
            room: document.room.clone(),
            source_file: document.source_file.clone(),
            chunk_index: 0,
            added_by: "multidomain-benchmark".to_owned(),
            filed_at: document.filed_at.clone(),
        },
    }
}

fn compact_excerpt(content: &str, max_chars: usize) -> String {
    let compact = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        compact
    } else {
        format!("{}...", compact.chars().take(max_chars).collect::<String>())
    }
}

fn default_cli_binary() -> Result<PathBuf, Box<dyn Error>> {
    let example = std::env::current_exe()?;
    let profile_dir = example
        .parent()
        .and_then(|examples| examples.parent())
        .ok_or_else(|| std::io::Error::other("cannot locate the Cargo profile directory"))?;
    let binary = profile_dir.join("mempalace-rs");
    if binary.is_file() {
        Ok(binary)
    } else {
        Err(format!(
            "CLI parity requires {}; build it with `cargo build --release --bin mempalace-rs` or pass --cli-binary",
            binary.display()
        )
        .into())
    }
}

fn cli_search_excerpts(
    cli_binary: &PathBuf,
    palace: &std::path::Path,
    model: Option<&str>,
    query: &str,
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut command = Command::new(cli_binary);
    command.arg("--palace").arg(palace);
    if let Some(model) = model {
        command.arg("--model").arg(model);
    }
    let output = command
        .arg("search")
        .arg(query)
        .arg("--all-wings")
        .arg("--results")
        .arg("10")
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "CLI search failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }

    let stdout = String::from_utf8(output.stdout)?;
    let mut lines = stdout.lines();
    let mut excerpts = Vec::new();
    while let Some(line) = lines.next() {
        if line.starts_with("[relevance ") {
            let _source_file = lines.next();
            let excerpt = lines
                .next()
                .ok_or_else(|| std::io::Error::other("CLI result is missing its excerpt"))?;
            excerpts.push(excerpt.to_owned());
        }
    }
    Ok(excerpts)
}

fn evaluate(
    runtime: &tokio::runtime::Runtime,
    store: &SqliteMemoryStore,
    corpus: &Corpus,
    owner: &CorpusDocument,
    query: &CorpusQuery,
) -> Result<QueryResult, Box<dyn Error>> {
    let started = Instant::now();
    let mut search = SearchQuery::new(&query.text);
    search.limit = 5;
    search.as_of = query.as_of.clone();
    let hits = runtime.block_on(store.search(search))?;
    let ranked_drawers = hits.into_iter().map(|hit| hit.drawer).collect::<Vec<_>>();
    let ranked_ids = ranked_drawers
        .iter()
        .map(|drawer| drawer.id.clone())
        .collect::<Vec<_>>();
    let mut relevant_ids = vec![owner.id.clone()];
    relevant_ids.extend(query.also_relevant.iter().cloned());
    let relevant_targets = relevant_ids
        .iter()
        .filter_map(|id| corpus.documents.iter().find(|document| document.id == *id))
        .collect::<Vec<_>>();
    let first_rank = ranked_drawers
        .iter()
        .position(|drawer| {
            relevant_targets
                .iter()
                .any(|target| matches_target(drawer, target))
        })
        .map(|rank| rank + 1);

    Ok(QueryResult {
        id: query.id.clone(),
        domain: owner.domain.clone(),
        split: match query.split {
            QuerySplit::Dev => "dev",
            QuerySplit::HeldOut => "held_out",
        },
        query: query.text.clone(),
        relevant_ids,
        ranked_ids,
        recall_at_5: first_rank.map_or(0.0, |_| 1.0),
        reciprocal_rank_at_5: first_rank.map_or(0.0, |rank| 1.0 / rank as f64),
        elapsed_ms: started.elapsed().as_millis(),
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let corpus: Corpus = serde_json::from_reader(BufReader::new(File::open(&args.corpus)?))?;
    validate_corpus(&corpus).map_err(std::io::Error::other)?;

    let config = MempalaceConfig::load()?;
    let model_dir = args
        .model_dir
        .clone()
        .unwrap_or_else(|| config.model_cache_path());
    fs::create_dir_all(&model_dir)?;
    let palace = tempfile::tempdir()?;
    let store = SqliteMemoryStore::new(palace.path(), model_dir, args.model.as_deref())?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime
        .block_on(store.add_drawers(corpus.documents.iter().map(to_drawer).collect::<Vec<_>>()))?;

    let mut parity_cases = BTreeMap::<String, String>::new();
    if args.check_parity {
        for document in &corpus.documents {
            if parity_cases.contains_key(&document.domain) {
                continue;
            }
            parity_cases.insert(document.domain.clone(), document.queries[0].text.clone());
        }
    }

    let mut output = args
        .output
        .as_ref()
        .map(File::create)
        .transpose()?
        .map(BufWriter::new);
    let mut overall = Metrics::default();
    let mut domains = BTreeMap::<String, Metrics>::new();

    for document in &corpus.documents {
        for query in document
            .queries
            .iter()
            .filter(|query| args.split.includes(query.split))
        {
            let result = evaluate(&runtime, &store, &corpus, document, query)?;
            overall.record(&result);
            domains
                .entry(result.domain.clone())
                .or_default()
                .record(&result);
            println!(
                "{:4} {:12} R@5={:.0} MRR={:.3} {}",
                overall.count,
                result.domain,
                result.recall_at_5,
                result.reciprocal_rank_at_5,
                result.id
            );
            if let Some(writer) = output.as_mut() {
                serde_json::to_writer(&mut *writer, &result)?;
                writer.write_all(b"\n")?;
            }
        }
    }
    if let Some(writer) = output.as_mut() {
        writer.flush()?;
    }

    println!();
    println!(
        "Multi-domain Recall@5 {:.3} MRR@5 {:.3} (n={})",
        overall.recall(),
        overall.mrr(),
        overall.count
    );
    let mut regressions = Vec::new();
    for (domain, metrics) in &domains {
        let baseline = corpus.baseline_recall_at_5[domain];
        let delta = metrics.recall() - baseline;
        println!(
            "  {domain:16} Recall@5 {:.3} MRR@5 {:.3} baseline {:.3} delta {delta:+.3} (n={})",
            metrics.recall(),
            metrics.mrr(),
            baseline,
            metrics.count
        );
        if delta < -0.02 {
            regressions.push(format!("{domain} regressed by {:.3}", -delta));
        }
    }
    if let Some(path) = &args.output {
        println!("Results: {}", path.display());
    }
    if !regressions.is_empty() {
        return Err(regressions.join(", ").into());
    }
    if args.check_parity {
        let parity_domain_count = parity_cases.len();
        drop(store);
        let cli_binary = args
            .cli_binary
            .clone()
            .map(Ok)
            .unwrap_or_else(default_cli_binary)?;
        let server = runtime.block_on(McpServer::open_with_palace_and_model(
            Some(palace.path().to_path_buf()),
            args.model.clone(),
        ))?;
        for (domain, query) in parity_cases {
            let cli_order =
                cli_search_excerpts(&cli_binary, palace.path(), args.model.as_deref(), &query)?;
            let response = runtime
                .block_on(server.tool_search(query, 10, None, None, None))
                .map_err(std::io::Error::other)?;
            let results = response["results"]
                .as_array()
                .ok_or_else(|| std::io::Error::other("MCP search returned no results array"))?;
            let mcp_order = results
                .iter()
                .filter_map(|result| {
                    result["text"]
                        .as_str()
                        .map(|text| compact_excerpt(text, 220))
                })
                .collect::<Vec<_>>();
            if cli_order != mcp_order {
                return Err(format!("CLI/MCP top-10 ordering differs for {domain}").into());
            }
        }
        println!("CLI/MCP top-10 parity: PASS (all {parity_domain_count} domains)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_corpus_has_fixed_25_75_split_and_100_queries() {
        let corpus: Corpus = serde_json::from_str(include_str!(
            "../../../benchmarks/multi_domain_queries.json"
        ))
        .unwrap();
        validate_corpus(&corpus).unwrap();

        let query_count = corpus
            .documents
            .iter()
            .map(|document| document.queries.len())
            .sum::<usize>();
        let dev_count = corpus
            .documents
            .iter()
            .flat_map(|document| &document.queries)
            .filter(|query| query.split == QuerySplit::Dev)
            .count();

        assert_eq!(query_count, 100);
        assert_eq!(dev_count, 25);
    }
}
