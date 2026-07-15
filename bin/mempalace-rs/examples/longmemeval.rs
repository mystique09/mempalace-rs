use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
    fmt,
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use mempalace_core::{
    ContentKind, Drawer, DrawerMetadata, MemoryStore, MempalaceConfig, SearchQuery,
};
use mempalace_store::SqliteMemoryStore;
use serde::{
    Deserialize, Serialize,
    de::{DeserializeSeed, Error as DeError, IgnoredAny, SeqAccess, Visitor},
};

const BENCHMARK_WING: &str = "longmemeval";
const BENCHMARK_ROOM: &str = "sessions";
const SEARCH_LIMIT: usize = 50;
const METRIC_KS: [usize; 6] = [1, 3, 5, 10, 30, 50];

#[derive(Debug, Parser)]
#[command(
    name = "longmemeval",
    about = "Run the raw LongMemEval retrieval benchmark against mempalace-rs"
)]
struct Args {
    /// Path to longmemeval_s_cleaned.json.
    data: PathBuf,
    /// Number of questions to process; zero means all questions.
    #[arg(long, default_value_t = 500)]
    limit: usize,
    /// Override the embedding model repository or local model path.
    #[arg(long)]
    model: Option<String>,
    /// Override the directory used to cache embedding models.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Optional JSONL path for auditable per-question results.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Checked-in JSON file containing `dev` and `held_out` question IDs.
    #[arg(long, requires = "subset")]
    split_file: Option<PathBuf>,
    /// Evaluate only one fixed split subset.
    #[arg(long, value_enum, requires = "split_file")]
    subset: Option<SplitSubset>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SplitSubset {
    Dev,
    HeldOut,
}

impl fmt::Display for SplitSubset {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Dev => "dev",
            Self::HeldOut => "held-out",
        })
    }
}

#[derive(Debug, Deserialize)]
struct BenchmarkSplit {
    dev: Vec<String>,
    held_out: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkEntry {
    question_id: String,
    question: String,
    question_type: String,
    #[serde(default)]
    question_date: Option<String>,
    answer_session_ids: Vec<String>,
    haystack_sessions: Vec<Vec<Turn>>,
    haystack_session_ids: Vec<String>,
    haystack_dates: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Turn {
    role: String,
    content: String,
}

fn session_drawers(entry: &BenchmarkEntry) -> Result<Vec<Drawer>, String> {
    if entry.haystack_sessions.len() != entry.haystack_session_ids.len()
        || entry.haystack_sessions.len() != entry.haystack_dates.len()
    {
        return Err(format!(
            "question {} has mismatched session, id, and date counts",
            entry.question_id
        ));
    }

    let mut drawers = Vec::with_capacity(entry.haystack_sessions.len());
    for (index, (session, date)) in entry
        .haystack_sessions
        .iter()
        .zip(&entry.haystack_dates)
        .enumerate()
    {
        let content = session
            .iter()
            .filter(|turn| turn.role == "user")
            .map(|turn| turn.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if content.is_empty() {
            continue;
        }

        drawers.push(Drawer {
            id: index.to_string(),
            content,
            retrieval_text: None,
            metadata: DrawerMetadata {
                content_kind: ContentKind::Conversation,
                wing: BENCHMARK_WING.to_owned(),
                room: BENCHMARK_ROOM.to_owned(),
                // Keep the benchmark label out of the production embedding
                // representation and path-aware reranker.
                source_file: None,
                chunk_index: index as i64,
                added_by: "longmemeval-benchmark".to_owned(),
                filed_at: Some(date.clone()),
            },
        });
    }

    Ok(drawers)
}

#[derive(Debug, Clone, Copy, Serialize)]
struct RetrievalMetrics {
    recall_any: f64,
    recall_all: f64,
    ndcg: f64,
    reciprocal_rank: f64,
}

fn retrieval_metrics(
    ranked_session_ids: &[String],
    relevant_session_ids: &HashSet<String>,
    k: usize,
) -> RetrievalMetrics {
    let top_k = ranked_session_ids.iter().take(k).collect::<HashSet<_>>();
    let recall_any = f64::from(
        relevant_session_ids
            .iter()
            .any(|session_id| top_k.contains(session_id)),
    );
    let recall_all = f64::from(
        !relevant_session_ids.is_empty()
            && relevant_session_ids
                .iter()
                .all(|session_id| top_k.contains(session_id)),
    );
    let relevances = ranked_session_ids
        .iter()
        .take(k)
        .map(|session_id| f64::from(relevant_session_ids.contains(session_id)))
        .collect::<Vec<_>>();
    let dcg = discounted_cumulative_gain(&relevances);
    let ideal = vec![1.0; k.min(relevant_session_ids.len())];
    let ideal_dcg = discounted_cumulative_gain(&ideal);
    let ndcg = if ideal_dcg == 0.0 {
        0.0
    } else {
        dcg / ideal_dcg
    };
    let reciprocal_rank = ranked_session_ids
        .iter()
        .take(k)
        .position(|session_id| relevant_session_ids.contains(session_id))
        .map_or(0.0, |rank| 1.0 / (rank + 1) as f64);

    RetrievalMetrics {
        recall_any,
        recall_all,
        ndcg,
        reciprocal_rank,
    }
}

fn discounted_cumulative_gain(relevances: &[f64]) -> f64 {
    relevances
        .iter()
        .enumerate()
        .map(|(index, relevance)| relevance / (index as f64 + 2.0).log2())
        .sum()
}

#[derive(Debug, Serialize)]
struct QuestionResult {
    question_id: String,
    question_type: String,
    question: String,
    answer_session_ids: Vec<String>,
    ranked_session_ids: Vec<String>,
    metrics: BTreeMap<String, RetrievalMetrics>,
    indexing_ms: u128,
    search_ms: u128,
    elapsed_ms: u128,
}

#[derive(Debug, Default)]
struct AggregateMetrics {
    questions: usize,
    recall_any: BTreeMap<usize, f64>,
    recall_all: BTreeMap<usize, f64>,
    ndcg: BTreeMap<usize, f64>,
    reciprocal_rank_10: f64,
    per_type_recall_10: BTreeMap<String, (usize, f64)>,
    latency_ms: Vec<u128>,
    search_latency_ms: Vec<u128>,
    indexing_ms: u128,
}

impl AggregateMetrics {
    fn record(&mut self, result: &QuestionResult) {
        self.questions += 1;
        self.latency_ms.push(result.elapsed_ms);
        self.search_latency_ms.push(result.search_ms);
        self.indexing_ms += result.indexing_ms;
        for k in METRIC_KS {
            let metrics = result
                .metrics
                .get(&metric_key(k))
                .expect("every configured k should be scored");
            *self.recall_any.entry(k).or_default() += metrics.recall_any;
            *self.recall_all.entry(k).or_default() += metrics.recall_all;
            *self.ndcg.entry(k).or_default() += metrics.ndcg;
            if k == 10 {
                self.reciprocal_rank_10 += metrics.reciprocal_rank;
                let type_metrics = self
                    .per_type_recall_10
                    .entry(result.question_type.clone())
                    .or_default();
                type_metrics.0 += 1;
                type_metrics.1 += metrics.recall_any;
            }
        }
    }

    fn mean(values: &BTreeMap<usize, f64>, k: usize, questions: usize) -> f64 {
        values.get(&k).copied().unwrap_or_default() / questions as f64
    }
}

fn metric_key(k: usize) -> String {
    format!("at_{k}")
}

fn nearest_rank_percentile(samples: &[u128], percentile: f64) -> Option<u128> {
    if samples.is_empty() {
        return None;
    }

    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (percentile.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.max(1).min(sorted.len()) - 1])
}

fn wilson_interval(successes: f64, total: usize) -> (f64, f64) {
    if total == 0 {
        return (0.0, 0.0);
    }
    let z = 1.959_963_984_540_054_f64;
    let n = total as f64;
    let proportion = successes / n;
    let denominator = 1.0 + z * z / n;
    let center = (proportion + z * z / (2.0 * n)) / denominator;
    let margin =
        z * (proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)).sqrt() / denominator;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

fn evaluate_question(
    runtime: &tokio::runtime::Runtime,
    store: &SqliteMemoryStore,
    entry: BenchmarkEntry,
) -> Result<QuestionResult, Box<dyn Error>> {
    let started = Instant::now();
    let drawers = session_drawers(&entry).map_err(std::io::Error::other)?;
    if drawers.is_empty() {
        return Err(format!(
            "question {} has no user-authored sessions",
            entry.question_id
        )
        .into());
    }

    let (ranked_drawer_ids, indexing_ms, search_ms) = runtime.block_on(async {
        store.delete_wing(BENCHMARK_WING).await?;
        let indexing_started = Instant::now();
        store.add_drawers(drawers).await?;
        let indexing_ms = indexing_started.elapsed().as_millis();

        let mut query = SearchQuery::new(&entry.question);
        query.limit = SEARCH_LIMIT;
        query.wing = Some(BENCHMARK_WING.to_owned());
        query.as_of = entry.question_date.clone();
        let search_started = Instant::now();
        let hits = store.search(query).await?;
        let search_ms = search_started.elapsed().as_millis();
        Ok::<_, mempalace_core::MempalaceError>((
            hits.into_iter()
                .map(|hit| hit.drawer.id)
                .collect::<Vec<_>>(),
            indexing_ms,
            search_ms,
        ))
    })?;
    let ranked_session_ids = ranked_drawer_ids
        .into_iter()
        .map(|drawer_id| {
            let index = drawer_id.parse::<usize>().map_err(|_| {
                std::io::Error::other(format!("invalid benchmark drawer id: {drawer_id}"))
            })?;
            entry
                .haystack_session_ids
                .get(index)
                .cloned()
                .ok_or_else(|| {
                    std::io::Error::other(format!("benchmark drawer index {index} is out of range"))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let relevant = entry
        .answer_session_ids
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let metrics = METRIC_KS
        .into_iter()
        .map(|k| {
            (
                metric_key(k),
                retrieval_metrics(&ranked_session_ids, &relevant, k),
            )
        })
        .collect();

    Ok(QuestionResult {
        question_id: entry.question_id,
        question_type: entry.question_type,
        question: entry.question,
        answer_session_ids: entry.answer_session_ids,
        ranked_session_ids,
        metrics,
        indexing_ms,
        search_ms,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

struct DatasetSeed<F> {
    limit: usize,
    visit: F,
}

impl<'de, F> DeserializeSeed<'de> for DatasetSeed<F>
where
    F: FnMut(BenchmarkEntry) -> Result<(), String>,
{
    type Value = usize;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(DatasetVisitor {
            limit: self.limit,
            visit: self.visit,
        })
    }
}

struct DatasetVisitor<F> {
    limit: usize,
    visit: F,
}

impl<'de, F> Visitor<'de> for DatasetVisitor<F>
where
    F: FnMut(BenchmarkEntry) -> Result<(), String>,
{
    type Value = usize;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON array of LongMemEval questions")
    }

    fn visit_seq<A>(mut self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut processed = 0;
        while processed < self.limit {
            let Some(entry) = sequence.next_element::<BenchmarkEntry>()? else {
                return Ok(processed);
            };
            (self.visit)(entry).map_err(A::Error::custom)?;
            processed += 1;
        }

        while sequence.next_element::<IgnoredAny>()?.is_some() {}
        Ok(processed)
    }
}

fn print_summary(aggregate: &AggregateMetrics, elapsed: Duration) {
    println!();
    println!("============================================================");
    println!("  RESULTS - mempalace-rs raw session retrieval");
    println!("============================================================");
    println!("  Questions: {}", aggregate.questions);
    println!(
        "  Time:      {:.1}s ({:.2}s per question)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / aggregate.questions as f64
    );
    println!(
        "  Latency:   p50={}ms p95={}ms max={}ms",
        nearest_rank_percentile(&aggregate.latency_ms, 0.50).unwrap_or_default(),
        nearest_rank_percentile(&aggregate.latency_ms, 0.95).unwrap_or_default(),
        aggregate
            .latency_ms
            .iter()
            .max()
            .copied()
            .unwrap_or_default()
    );
    println!(
        "  Indexing:  {}ms total ({:.2}ms per question)",
        aggregate.indexing_ms,
        aggregate.indexing_ms as f64 / aggregate.questions as f64
    );
    println!(
        "  Search:    p50={}ms p95={}ms max={}ms",
        nearest_rank_percentile(&aggregate.search_latency_ms, 0.50).unwrap_or_default(),
        nearest_rank_percentile(&aggregate.search_latency_ms, 0.95).unwrap_or_default(),
        aggregate
            .search_latency_ms
            .iter()
            .max()
            .copied()
            .unwrap_or_default()
    );
    println!();
    for k in METRIC_KS {
        println!(
            "  Recall@{k:<2}: {:.3}    Recall-all@{k:<2}: {:.3}    NDCG@{k:<2}: {:.3}",
            AggregateMetrics::mean(&aggregate.recall_any, k, aggregate.questions),
            AggregateMetrics::mean(&aggregate.recall_all, k, aggregate.questions),
            AggregateMetrics::mean(&aggregate.ndcg, k, aggregate.questions),
        );
    }
    let (recall_5_low, recall_5_high) = wilson_interval(
        aggregate.recall_any.get(&5).copied().unwrap_or_default(),
        aggregate.questions,
    );
    println!("  Recall@5 95% CI: [{recall_5_low:.3}, {recall_5_high:.3}] (Wilson)");
    println!(
        "  MRR@10:   {:.3}",
        aggregate.reciprocal_rank_10 / aggregate.questions as f64
    );
    println!();
    println!("  Per-type Recall@10:");
    for (question_type, (count, recall)) in &aggregate.per_type_recall_10 {
        println!(
            "    {question_type:35} {:.3} (n={count})",
            recall / *count as f64
        );
    }
    println!("============================================================");
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if !args.data.is_file() {
        return Err(format!("dataset does not exist: {}", args.data.display()).into());
    }

    let subset = match (args.split_file.as_ref(), args.subset) {
        (Some(path), Some(subset)) => {
            let split: BenchmarkSplit = serde_json::from_reader(BufReader::new(File::open(path)?))?;
            let ids = match subset {
                SplitSubset::Dev => split.dev,
                SplitSubset::HeldOut => split.held_out,
            };
            Some((subset, ids.into_iter().collect::<HashSet<_>>()))
        }
        (None, None) => None,
        _ => return Err("--split-file and --subset must be provided together".into()),
    };

    let config = MempalaceConfig::load()?;
    let model_dir = args.model_dir.unwrap_or_else(|| config.model_cache_path());
    fs::create_dir_all(&model_dir)?;
    let palace = tempfile::tempdir()?;
    let store = SqliteMemoryStore::new(palace.path(), &model_dir, args.model.as_deref())?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let mut output = args
        .output
        .as_ref()
        .map(File::create)
        .transpose()?
        .map(BufWriter::new);
    let limit = if args.limit == 0 {
        usize::MAX
    } else {
        args.limit
    };

    println!("============================================================");
    println!("  MemPalace-rs x LongMemEval");
    println!("============================================================");
    println!("  Data:      {}", args.data.display());
    let reported_questions = subset
        .as_ref()
        .map(|(_, ids)| ids.len().min(limit))
        .unwrap_or_else(|| if args.limit == 0 { 500 } else { args.limit });
    println!("  Questions: {reported_questions}");
    println!("  Top-k:     {SEARCH_LIMIT}");
    if let Some((subset, ids)) = &subset {
        println!("  Split:     {subset} ({} question IDs)", ids.len());
    }
    println!("  Palace:    isolated temporary SQLite store");
    println!();

    let benchmark_started = Instant::now();
    let mut aggregate = AggregateMetrics::default();
    let mut selected = 0usize;
    let file = File::open(&args.data)?;
    let mut deserializer = serde_json::Deserializer::from_reader(BufReader::new(file));
    let processed = DatasetSeed {
        limit: if subset.is_some() { usize::MAX } else { limit },
        visit: |entry: BenchmarkEntry| {
            if subset
                .as_ref()
                .is_some_and(|(_, ids)| !ids.contains(&entry.question_id))
                || selected >= limit
            {
                return Ok(());
            }
            selected += 1;
            let result =
                evaluate_question(&runtime, &store, entry).map_err(|error| error.to_string())?;
            aggregate.record(&result);

            let recall_5 = result.metrics[&metric_key(5)].recall_any;
            let recall_10 = result.metrics[&metric_key(10)].recall_any;
            let status = if recall_5 > 0.0 { "HIT" } else { "miss" };
            println!(
                "  [{:4}] {:12} R@5={recall_5:.0} R@10={recall_10:.0} {status:4} {:5}ms",
                aggregate.questions, result.question_id, result.elapsed_ms
            );

            if let Some(writer) = output.as_mut() {
                serde_json::to_writer(&mut *writer, &result).map_err(|error| error.to_string())?;
                writer.write_all(b"\n").map_err(|error| error.to_string())?;
            }
            Ok(())
        },
    }
    .deserialize(&mut deserializer)?;
    deserializer.end()?;
    if processed == 0 || aggregate.questions == 0 {
        return Err("dataset contained no benchmark questions".into());
    }
    if subset.is_some() && aggregate.questions != reported_questions {
        return Err(format!(
            "split expected {reported_questions} questions but matched {} dataset entries",
            aggregate.questions
        )
        .into());
    }
    if let Some(writer) = output.as_mut() {
        writer.flush()?;
    }

    print_summary(&aggregate, benchmark_started.elapsed());
    if let Some(path) = args.output {
        println!("  Results: {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_user_turns_to_one_drawer_per_non_empty_session() {
        let entry: BenchmarkEntry = serde_json::from_value(serde_json::json!({
            "question_id": "q1",
            "question": "What degree did I graduate with?",
            "question_type": "single-session-user",
            "answer_session_ids": ["answer-session"],
            "haystack_session_ids": ["distractor", "answer-session", "empty"],
            "haystack_dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "unrelated question"},
                    {"role": "assistant", "content": "unrelated answer"}
                ],
                [
                    {"role": "user", "content": "I graduated from MIT."},
                    {"role": "assistant", "content": "Congratulations."},
                    {"role": "user", "content": "My degree was computer science."}
                ],
                [
                    {"role": "assistant", "content": "assistant-only sessions are ignored"}
                ]
            ]
        }))
        .unwrap();

        let drawers = session_drawers(&entry).unwrap();

        assert_eq!(drawers.len(), 2);
        assert_eq!(drawers[1].id, "1");
        assert_eq!(
            drawers[1].content,
            "I graduated from MIT.\nMy degree was computer science."
        );
        assert_eq!(drawers[1].metadata.source_file, None);
        assert_eq!(drawers[1].metadata.content_kind, ContentKind::Conversation);
        assert_eq!(drawers[1].metadata.filed_at.as_deref(), Some("2024-01-02"));
    }

    #[test]
    fn duplicate_session_labels_still_produce_unique_unlabelled_drawer_ids() {
        let entry: BenchmarkEntry = serde_json::from_value(serde_json::json!({
            "question_id": "q1",
            "question": "question",
            "question_type": "single-session-user",
            "answer_session_ids": ["duplicate"],
            "haystack_session_ids": ["duplicate", "duplicate"],
            "haystack_dates": ["2024-01-01", "2024-01-02"],
            "haystack_sessions": [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}]
            ]
        }))
        .unwrap();

        let drawers = session_drawers(&entry).unwrap();

        assert_eq!(drawers[0].id, "0");
        assert_eq!(drawers[1].id, "1");
    }

    #[test]
    fn scores_ranked_session_ids_against_all_ground_truth_sessions() {
        let ranked = vec![
            "noise".to_owned(),
            "answer-b".to_owned(),
            "answer-a".to_owned(),
        ];
        let relevant = ["answer-a".to_owned(), "answer-b".to_owned()]
            .into_iter()
            .collect();

        let at_one = retrieval_metrics(&ranked, &relevant, 1);
        assert_eq!(at_one.recall_any, 0.0);
        assert_eq!(at_one.recall_all, 0.0);

        let at_two = retrieval_metrics(&ranked, &relevant, 2);
        assert_eq!(at_two.recall_any, 1.0);
        assert_eq!(at_two.recall_all, 0.0);
        assert_eq!(at_two.reciprocal_rank, 0.5);
        assert!((at_two.ndcg - 0.386_852_8).abs() < 0.000_001);

        let at_three = retrieval_metrics(&ranked, &relevant, 3);
        assert_eq!(at_three.recall_all, 1.0);
        assert!((at_three.ndcg - 0.693_426_4).abs() < 0.000_001);
    }

    #[test]
    fn reports_latency_with_nearest_rank_percentiles() {
        let samples = [50, 10, 40, 20, 30];

        assert_eq!(nearest_rank_percentile(&samples, 0.50), Some(30));
        assert_eq!(nearest_rank_percentile(&samples, 0.95), Some(50));
        assert_eq!(nearest_rank_percentile(&[], 0.95), None);
    }

    #[test]
    fn reports_a_bounded_recall_confidence_interval() {
        let (low, high) = wilson_interval(479.0, 500);

        assert!((low - 0.9366).abs() < 0.001);
        assert!((high - 0.9724).abs() < 0.001);
    }
}
