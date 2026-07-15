use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::{DrawerMetadata, Result};

const EMOTION_CODES: &[(&str, &str)] = &[
    ("vulnerability", "vul"),
    ("vulnerable", "vul"),
    ("joy", "joy"),
    ("joyful", "joy"),
    ("fear", "fear"),
    ("mild_fear", "fear"),
    ("trust", "trust"),
    ("trust_building", "trust"),
    ("grief", "grief"),
    ("raw_grief", "grief"),
    ("wonder", "wonder"),
    ("philosophical_wonder", "wonder"),
    ("rage", "rage"),
    ("anger", "rage"),
    ("love", "love"),
    ("devotion", "love"),
    ("hope", "hope"),
    ("despair", "despair"),
    ("hopelessness", "despair"),
    ("peace", "peace"),
    ("relief", "relief"),
    ("humor", "humor"),
    ("dark_humor", "humor"),
    ("tenderness", "tender"),
    ("raw_honesty", "raw"),
    ("brutal_honesty", "raw"),
    ("self_doubt", "doubt"),
    ("anxiety", "anx"),
    ("exhaustion", "exhaust"),
    ("conviction", "convict"),
    ("quiet_passion", "passion"),
    ("warmth", "warmth"),
    ("curiosity", "curious"),
    ("gratitude", "grat"),
    ("frustration", "frust"),
    ("confusion", "confuse"),
    ("satisfaction", "satis"),
    ("excitement", "excite"),
    ("determination", "determ"),
    ("surprise", "surprise"),
];

const EMOTION_SIGNALS: &[(&str, &str)] = &[
    ("decided", "determ"),
    ("prefer", "convict"),
    ("worried", "anx"),
    ("excited", "excite"),
    ("frustrated", "frust"),
    ("confused", "confuse"),
    ("love", "love"),
    ("hate", "rage"),
    ("hope", "hope"),
    ("fear", "fear"),
    ("trust", "trust"),
    ("happy", "joy"),
    ("sad", "grief"),
    ("surprised", "surprise"),
    ("grateful", "grat"),
    ("curious", "curious"),
    ("wonder", "wonder"),
    ("anxious", "anx"),
    ("relieved", "relief"),
    ("satisf", "satis"),
    ("disappoint", "grief"),
    ("concern", "anx"),
];

const FLAG_SIGNALS: &[(&str, &str)] = &[
    ("decided", "DECISION"),
    ("chose", "DECISION"),
    ("switched", "DECISION"),
    ("migrated", "DECISION"),
    ("replaced", "DECISION"),
    ("instead of", "DECISION"),
    ("because", "DECISION"),
    ("founded", "ORIGIN"),
    ("created", "ORIGIN"),
    ("started", "ORIGIN"),
    ("born", "ORIGIN"),
    ("launched", "ORIGIN"),
    ("first time", "ORIGIN"),
    ("core", "CORE"),
    ("fundamental", "CORE"),
    ("essential", "CORE"),
    ("principle", "CORE"),
    ("belief", "CORE"),
    ("always", "CORE"),
    ("never forget", "CORE"),
    ("turning point", "PIVOT"),
    ("changed everything", "PIVOT"),
    ("realized", "PIVOT"),
    ("breakthrough", "PIVOT"),
    ("epiphany", "PIVOT"),
    ("api", "TECHNICAL"),
    ("database", "TECHNICAL"),
    ("architecture", "TECHNICAL"),
    ("deploy", "TECHNICAL"),
    ("infrastructure", "TECHNICAL"),
    ("algorithm", "TECHNICAL"),
    ("framework", "TECHNICAL"),
    ("server", "TECHNICAL"),
    ("config", "TECHNICAL"),
];

const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "during", "before", "after", "above", "below", "up", "down", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just", "don", "now", "and", "but",
    "or", "if", "while", "that", "this", "these", "those", "it", "its", "i", "we", "you", "he",
    "she", "they", "me", "him", "her", "us", "them", "my", "your", "his", "our", "their", "what",
    "which", "who", "whom", "also", "much", "many", "like", "because", "since", "get", "got",
    "use", "used", "using", "make", "made", "thing", "things", "way", "well", "really", "want",
    "need",
];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct AaakConfigFile {
    #[serde(default)]
    entities: BTreeMap<String, String>,
    #[serde(default)]
    skip_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AaakHeader {
    pub wing: String,
    pub room: String,
    pub date: String,
    pub title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AaakDecoded {
    pub header: Option<AaakHeader>,
    pub entries: Vec<String>,
    pub tunnels: Vec<String>,
    pub arc: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AaakCompressionStats {
    pub original_tokens_est: usize,
    pub summary_tokens_est: usize,
    pub size_ratio: f32,
    pub original_chars: usize,
    pub summary_chars: usize,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct AaakTunnel {
    #[serde(rename = "from", default)]
    pub from_id: String,
    #[serde(rename = "to", default)]
    pub to_id: String,
    #[serde(default)]
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct AaakZettel {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub people: Vec<String>,
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub origin_label: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub emotional_weight: f32,
    #[serde(default)]
    pub emotional_tone: Vec<String>,
    #[serde(default)]
    pub origin_moment: bool,
    #[serde(default)]
    pub sensitivity: String,
    #[serde(default)]
    pub date_context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct AaakFile {
    #[serde(default)]
    pub source_file: String,
    #[serde(default)]
    pub emotional_arc: String,
    #[serde(default)]
    pub zettels: Vec<AaakZettel>,
    #[serde(default)]
    pub tunnels: Vec<AaakTunnel>,
}

#[derive(Debug, Clone, Default)]
pub struct AaakDialect {
    entity_codes: HashMap<String, String>,
    skip_names: Vec<String>,
}

impl AaakDialect {
    pub fn new(entities: BTreeMap<String, String>, skip_names: Vec<String>) -> Self {
        let mut entity_codes = HashMap::new();
        for (name, code) in entities {
            entity_codes.insert(name.clone(), code.clone());
            entity_codes.insert(name.to_lowercase(), code);
        }

        Self {
            entity_codes,
            skip_names: skip_names
                .into_iter()
                .map(|name| name.to_lowercase())
                .collect(),
        }
    }

    pub fn from_config_path(path: impl AsRef<Path>) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        let config: AaakConfigFile = serde_json::from_str(&raw)?;
        Ok(Self::new(config.entities, config.skip_names))
    }

    pub fn save_config(&self, path: impl Into<PathBuf>) -> Result<PathBuf> {
        let path = path.into();
        let mut entities = BTreeMap::new();
        let mut seen = HashSet::new();
        for (name, code) in &self.entity_codes {
            if name == &name.to_lowercase() || !seen.insert(code.clone()) {
                continue;
            }
            entities.insert(name.clone(), code.clone());
        }

        let raw = serde_json::to_string_pretty(&AaakConfigFile {
            entities,
            skip_names: self.skip_names.clone(),
        })?;
        fs::write(&path, raw)?;
        Ok(path)
    }

    pub fn compress(&self, text: &str, metadata: Option<&DrawerMetadata>) -> String {
        let entities = self.detect_entities(text);
        let entity_str = if entities.is_empty() {
            "???".to_owned()
        } else {
            entities.join("+")
        };

        let topics = self.extract_topics(text, 3);
        let topic_str = if topics.is_empty() {
            "misc".to_owned()
        } else {
            topics.join("_")
        };

        let quote = self.extract_key_sentence(text);
        let emotions = self.detect_signals(text, EMOTION_SIGNALS);
        let flags = self.detect_signals(text, FLAG_SIGNALS);

        let mut lines = Vec::new();
        if let Some(metadata) = metadata {
            let title = metadata
                .source_file
                .as_deref()
                .and_then(|source| Path::new(source).file_stem())
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| "?".to_owned());
            let date = metadata
                .filed_at
                .as_deref()
                .map(short_date)
                .unwrap_or_else(|| "?".to_owned());
            lines.push(format!(
                "{}|{}|{}|{}",
                metadata.wing, metadata.room, date, title
            ));
        }

        let mut parts = vec![format!("0:{entity_str}"), topic_str];
        if !quote.is_empty() {
            parts.push(format!("\"{quote}\""));
        }
        if !emotions.is_empty() {
            parts.push(emotions.join("+"));
        }
        if !flags.is_empty() {
            parts.push(flags.join("+"));
        }
        lines.push(parts.join("|"));
        lines.join("\n")
    }

    pub fn decode(&self, dialect_text: &str) -> AaakDecoded {
        let mut decoded = AaakDecoded {
            header: None,
            entries: Vec::new(),
            tunnels: Vec::new(),
            arc: None,
        };

        for line in dialect_text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
        {
            if let Some(arc) = line.strip_prefix("ARC:") {
                decoded.arc = Some(arc.to_owned());
                continue;
            }
            if line.starts_with("T:") {
                decoded.tunnels.push(line.to_owned());
                continue;
            }
            if let Some((left, _, _)) = split_once3(line, '|')
                && left.contains(':')
            {
                decoded.entries.push(line.to_owned());
                continue;
            }
            let parts = line.split('|').collect::<Vec<_>>();
            if parts.len() >= 2 {
                decoded.header = Some(AaakHeader {
                    wing: parts.first().unwrap_or(&"").to_string(),
                    room: parts.get(1).unwrap_or(&"").to_string(),
                    date: parts.get(2).unwrap_or(&"").to_string(),
                    title: parts.get(3).unwrap_or(&"").to_string(),
                });
            }
        }

        decoded
    }

    pub fn compression_stats(&self, original_text: &str, compressed: &str) -> AaakCompressionStats {
        let original_tokens_est = Self::count_tokens(original_text);
        let summary_tokens_est = Self::count_tokens(compressed);
        AaakCompressionStats {
            original_tokens_est,
            summary_tokens_est,
            size_ratio: original_tokens_est as f32 / summary_tokens_est.max(1) as f32,
            original_chars: original_text.len(),
            summary_chars: compressed.len(),
            note: "Estimates only. Use tiktoken for accurate counts. AAAK is lossy.".to_owned(),
        }
    }

    pub fn count_tokens(text: &str) -> usize {
        let words = text.split_whitespace().count();
        ((words as f32) * 1.3).floor().max(1.0) as usize
    }

    pub fn encode_entity(&self, name: &str) -> Option<String> {
        if self.should_skip_name(name) {
            return None;
        }
        if let Some(code) = self.entity_codes.get(name) {
            return Some(code.clone());
        }
        let lowered = name.to_lowercase();
        if let Some(code) = self.entity_codes.get(&lowered) {
            return Some(code.clone());
        }
        for (key, code) in &self.entity_codes {
            if key.to_lowercase() != *key && lowered.contains(&key.to_lowercase()) {
                return Some(code.clone());
            }
        }

        let clean = name
            .chars()
            .filter(|ch| ch.is_ascii_alphabetic())
            .take(3)
            .collect::<String>();
        if clean.is_empty() {
            None
        } else {
            Some(clean.to_uppercase())
        }
    }

    pub fn encode_emotions<I, S>(&self, emotions: I) -> String
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut codes = Vec::new();
        for emotion in emotions {
            let emotion = emotion.as_ref();
            let code = EMOTION_CODES
                .iter()
                .find_map(|(name, code)| (*name == emotion).then_some(*code))
                .unwrap_or_else(|| &emotion[..emotion.len().min(4)]);
            if !codes.iter().any(|existing| existing == code) {
                codes.push(code.to_owned());
            }
        }
        codes.into_iter().take(3).collect::<Vec<_>>().join("+")
    }

    pub fn get_flags(&self, zettel: &AaakZettel) -> String {
        let mut flags = Vec::new();
        if zettel.origin_moment {
            flags.push("ORIGIN");
        }
        if zettel.sensitivity.to_uppercase().starts_with("MAXIMUM") {
            flags.push("SENSITIVE");
        }
        let notes = zettel.notes.to_lowercase();
        if notes.contains("foundational pillar") || notes.contains("core") {
            flags.push("CORE");
        }
        if notes.contains("genesis") || zettel.origin_label.to_lowercase().contains("genesis") {
            flags.push("GENESIS");
        }
        if notes.contains("pivot") {
            flags.push("PIVOT");
        }

        flags.join("+")
    }

    pub fn extract_key_quote(&self, zettel: &AaakZettel) -> String {
        let all_text = format!(
            "{} {} {}",
            zettel.content, zettel.origin_label, zettel.notes
        );

        let mut quotes = Vec::new();
        quotes.extend(extract_quoted_fragments(&all_text, '"'));
        quotes.extend(extract_single_quoted_fragments(&all_text));
        quotes.extend(extract_speech_fragments(&all_text));

        if !quotes.is_empty() {
            let mut seen = HashSet::new();
            let mut unique = Vec::new();
            for quote in quotes {
                let quote = quote.trim().to_owned();
                if quote.len() >= 8 && seen.insert(quote.clone()) {
                    unique.push(quote);
                }
            }

            let emotional_words = [
                "love",
                "fear",
                "remember",
                "soul",
                "feel",
                "stupid",
                "scared",
                "beautiful",
                "destroy",
                "respect",
                "trust",
                "consciousness",
                "alive",
                "forget",
                "waiting",
                "peace",
                "matter",
                "real",
                "guilt",
                "escape",
                "rest",
                "hope",
                "dream",
                "lost",
                "found",
            ];

            let mut scored = unique
                .into_iter()
                .map(|quote| {
                    let mut score = 0;
                    if quote
                        .chars()
                        .next()
                        .map(|ch| ch.is_uppercase())
                        .unwrap_or(false)
                        || quote.starts_with("I ")
                    {
                        score += 2;
                    }
                    score += emotional_words
                        .iter()
                        .filter(|word| quote.to_lowercase().contains(*word))
                        .count() as i32
                        * 2;
                    if quote.len() > 20 {
                        score += 1;
                    }
                    if quote.starts_with("The ")
                        || quote.starts_with("This ")
                        || quote.starts_with("She ")
                    {
                        score -= 2;
                    }
                    (score, quote)
                })
                .collect::<Vec<_>>();
            scored.sort_by_key(|b| std::cmp::Reverse(b.0));
            if let Some((_, quote)) = scored.into_iter().next() {
                return quote;
            }
        }

        if let Some((_, right)) = zettel.title.split_once(" - ") {
            return right.chars().take(45).collect();
        }

        String::new()
    }

    pub fn encode_zettel(&self, zettel: &AaakZettel) -> String {
        let zid = zettel.id.rsplit('-').next().unwrap_or(&zettel.id);

        let mut entity_codes = zettel
            .people
            .iter()
            .filter_map(|person| self.encode_entity(person))
            .collect::<Vec<_>>();
        if entity_codes.is_empty() {
            entity_codes.push("???".to_owned());
        }
        entity_codes.sort();
        entity_codes.dedup();
        let entities = entity_codes.join("+");

        let topic_str = if zettel.topics.is_empty() {
            "misc".to_owned()
        } else {
            zettel
                .topics
                .iter()
                .take(2)
                .cloned()
                .collect::<Vec<_>>()
                .join("_")
        };

        let quote = self.extract_key_quote(zettel);
        let quote_part = if quote.is_empty() {
            None
        } else {
            Some(format!("\"{quote}\""))
        };

        let weight = if zettel.emotional_weight == 0.0 {
            0.5
        } else {
            zettel.emotional_weight
        };
        let emotions = self.encode_emotions(zettel.emotional_tone.iter().map(String::as_str));
        let flags = self.get_flags(zettel);

        let mut parts = vec![format!("{zid}:{entities}"), topic_str];
        if let Some(quote_part) = quote_part {
            parts.push(quote_part);
        }
        parts.push(format_weight(weight));
        if !emotions.is_empty() {
            parts.push(emotions);
        }
        if !flags.is_empty() {
            parts.push(flags);
        }

        parts.join("|")
    }

    pub fn encode_tunnel(&self, tunnel: &AaakTunnel) -> String {
        let from_id = tunnel.from_id.rsplit('-').next().unwrap_or(&tunnel.from_id);
        let to_id = tunnel.to_id.rsplit('-').next().unwrap_or(&tunnel.to_id);
        let short_label = tunnel
            .label
            .split_once(':')
            .map(|(left, _)| left)
            .unwrap_or(&tunnel.label);
        let short_label = short_label.chars().take(30).collect::<String>();
        format!("T:{from_id}<->{to_id}|{short_label}")
    }

    pub fn encode_file(&self, zettel_json: &AaakFile) -> String {
        let mut lines = Vec::new();

        let source = if zettel_json.source_file.is_empty() {
            "unknown"
        } else {
            &zettel_json.source_file
        };
        let file_num = source.split('-').next().unwrap_or("000");
        let date = zettel_json
            .zettels
            .first()
            .map(|zettel| zettel.date_context.as_str())
            .filter(|date| !date.is_empty())
            .unwrap_or("unknown");

        let mut all_people = zettel_json
            .zettels
            .iter()
            .flat_map(|zettel| zettel.people.iter())
            .filter_map(|person| self.encode_entity(person))
            .collect::<Vec<_>>();
        if all_people.is_empty() {
            all_people.push("???".to_owned());
        }
        all_people.sort();
        all_people.dedup();
        let primary = all_people.into_iter().take(3).collect::<Vec<_>>().join("+");

        let title = if let Some((_, right)) = source.rsplit_once('-') {
            right.replace(".txt", "").trim().to_owned()
        } else {
            source.to_owned()
        };
        lines.push(format!("{file_num}|{primary}|{date}|{title}"));

        if !zettel_json.emotional_arc.is_empty() {
            lines.push(format!("ARC:{}", zettel_json.emotional_arc));
        }

        for zettel in &zettel_json.zettels {
            lines.push(self.encode_zettel(zettel));
        }
        for tunnel in &zettel_json.tunnels {
            lines.push(self.encode_tunnel(tunnel));
        }

        lines.join("\n")
    }

    pub fn compress_file(
        &self,
        zettel_json_path: impl AsRef<Path>,
        output_path: Option<PathBuf>,
    ) -> Result<String> {
        let raw = fs::read_to_string(zettel_json_path)?;
        let data: AaakFile = serde_json::from_str(&raw)?;
        let dialect = self.encode_file(&data);
        if let Some(output_path) = output_path {
            fs::write(output_path, &dialect)?;
        }
        Ok(dialect)
    }

    pub fn compress_all(
        &self,
        zettel_dir: impl AsRef<Path>,
        output_path: Option<PathBuf>,
    ) -> Result<String> {
        let mut all_dialect = Vec::new();
        for path in json_files(zettel_dir)? {
            let raw = fs::read_to_string(path)?;
            let data: AaakFile = serde_json::from_str(&raw)?;
            all_dialect.push(self.encode_file(&data));
            all_dialect.push("---".to_owned());
        }

        let combined = all_dialect.join("\n");
        if let Some(output_path) = output_path {
            fs::write(output_path, &combined)?;
        }
        Ok(combined)
    }

    pub fn generate_layer1(
        &self,
        zettel_dir: impl AsRef<Path>,
        output_path: Option<PathBuf>,
        identity_sections: Option<&BTreeMap<String, Vec<String>>>,
        weight_threshold: f32,
    ) -> Result<String> {
        let mut essential = Vec::<(AaakZettel, String, String)>::new();
        let mut all_tunnels = Vec::new();

        for path in json_files(zettel_dir)? {
            let raw = fs::read_to_string(&path)?;
            let data: AaakFile = serde_json::from_str(&raw)?;
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            let file_num = file_name
                .replace("file_", "")
                .replace(".json", "")
                .to_owned();
            let source_date = data
                .zettels
                .first()
                .map(|zettel| zettel.date_context.clone())
                .unwrap_or_else(|| "unknown".to_owned());

            for zettel in &data.zettels {
                let weight = zettel.emotional_weight;
                let flags = self.get_flags(zettel);
                let has_key_flag =
                    flags.contains("ORIGIN") || flags.contains("CORE") || flags.contains("GENESIS");
                if weight >= weight_threshold || zettel.origin_moment || has_key_flag {
                    essential.push((zettel.clone(), file_num.clone(), source_date.clone()));
                }
            }

            all_tunnels.extend(data.tunnels);
        }

        essential.sort_by(|a, b| {
            b.0.emotional_weight
                .partial_cmp(&a.0.emotional_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut by_date = BTreeMap::<String, Vec<(AaakZettel, String)>>::new();
        for (zettel, file_num, source_date) in essential {
            let key = source_date
                .split(',')
                .next()
                .unwrap_or("unknown")
                .trim()
                .to_owned();
            by_date.entry(key).or_default().push((zettel, file_num));
        }

        let mut lines = Vec::new();
        lines.push("## LAYER 1 -- ESSENTIAL STORY".to_owned());
        lines.push(format!(
            "## Auto-generated from zettel files. Updated {}.",
            Utc::now().date_naive()
        ));
        lines.push(String::new());

        if let Some(identity_sections) = identity_sections {
            for (section_name, section_lines) in identity_sections {
                lines.push(format!("={section_name}="));
                lines.extend(section_lines.iter().cloned());
                lines.push(String::new());
            }
        }

        for (date_key, items) in by_date {
            lines.push(format!("=MOMENTS[{date_key}]="));
            for (zettel, _file_num) in items {
                let mut entities = zettel
                    .people
                    .iter()
                    .filter_map(|person| self.encode_entity(person))
                    .collect::<Vec<_>>();
                if entities.is_empty() {
                    entities.push("???".to_owned());
                }
                entities.sort();
                entities.dedup();
                let ent_str = entities.join("+");

                let quote = self.extract_key_quote(&zettel);
                let weight = if zettel.emotional_weight == 0.0 {
                    0.5
                } else {
                    zettel.emotional_weight
                };
                let flags = self.get_flags(&zettel);
                let sensitivity = zettel.sensitivity.clone();

                let mut parts = vec![ent_str];
                let hint = if let Some((_, right)) = zettel.title.split_once(" - ") {
                    right.chars().take(30).collect::<String>()
                } else {
                    zettel
                        .topics
                        .iter()
                        .take(2)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("_")
                };
                if !hint.is_empty() {
                    parts.push(hint.clone());
                }
                if !quote.is_empty()
                    && quote != hint
                    && quote != zettel.title
                    && quote != hint.as_str()
                {
                    parts.push(format!("\"{quote}\""));
                }
                if !sensitivity.is_empty() && !flags.contains("SENSITIVE") {
                    parts.push("SENSITIVE".to_owned());
                }
                parts.push(format_weight(weight));
                if !flags.is_empty() {
                    parts.push(flags);
                }

                lines.push(parts.join("|"));
            }
            lines.push(String::new());
        }

        if !all_tunnels.is_empty() {
            lines.push("=TUNNELS=".to_owned());
            for tunnel in all_tunnels.into_iter().take(8) {
                let short = tunnel
                    .label
                    .split_once(':')
                    .map(|(left, _)| left)
                    .unwrap_or(&tunnel.label)
                    .chars()
                    .take(40)
                    .collect::<String>();
                lines.push(short);
            }
            lines.push(String::new());
        }

        let result = lines.join("\n");
        if let Some(output_path) = output_path {
            fs::write(output_path, &result)?;
        }
        Ok(result)
    }

    fn detect_signals(&self, text: &str, signals: &[(&str, &str)]) -> Vec<String> {
        let lowered = text.to_lowercase();
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for (needle, code) in signals {
            if lowered.contains(needle) && seen.insert(*code) {
                out.push((*code).to_owned());
            }
            if out.len() >= 3 {
                break;
            }
        }
        out
    }

    fn extract_topics(&self, text: &str, max_topics: usize) -> Vec<String> {
        let tokens = topic_tokens(text);
        let stop_words = stop_words();
        let mut freq = HashMap::<String, usize>::new();

        for token in &tokens {
            let lowered = token.to_lowercase();
            if lowered.len() < 3 || stop_words.contains(lowered.as_str()) {
                continue;
            }
            *freq.entry(lowered).or_insert(0) += 1;
        }

        for token in &tokens {
            let lowered = token.to_lowercase();
            if stop_words.contains(lowered.as_str()) {
                continue;
            }
            if token
                .chars()
                .next()
                .map(|ch| ch.is_uppercase())
                .unwrap_or(false)
            {
                *freq.entry(lowered.clone()).or_insert(0) += 2;
            }
            if token.contains('_')
                || token.contains('-')
                || token.chars().skip(1).any(|ch| ch.is_uppercase())
            {
                *freq.entry(lowered).or_insert(0) += 2;
            }
        }

        let mut ranked = freq.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked
            .into_iter()
            .take(max_topics)
            .map(|(topic, _)| topic)
            .collect()
    }

    fn extract_key_sentence(&self, text: &str) -> String {
        let decision_words = [
            "decided",
            "because",
            "instead",
            "prefer",
            "switched",
            "chose",
            "realized",
            "important",
            "key",
            "critical",
            "discovered",
            "learned",
            "conclusion",
            "solution",
            "reason",
            "why",
            "breakthrough",
            "insight",
        ];

        let mut best_score = i32::MIN;
        let mut best = String::new();
        for sentence in split_sentences(text) {
            if sentence.len() <= 10 {
                continue;
            }

            let lowered = sentence.to_lowercase();
            let mut score = 0;
            for word in &decision_words {
                if lowered.contains(word) {
                    score += 2;
                }
            }
            if sentence.len() < 80 {
                score += 1;
            }
            if sentence.len() < 40 {
                score += 1;
            }
            if sentence.len() > 150 {
                score -= 2;
            }

            if score > best_score {
                best_score = score;
                best = sentence;
            }
        }

        if best.chars().count() > 55 {
            format!("{}...", best.chars().take(52).collect::<String>())
        } else {
            best
        }
    }

    fn detect_entities(&self, text: &str) -> Vec<String> {
        let lowered = text.to_lowercase();
        let mut found = Vec::new();

        for (name, code) in &self.entity_codes {
            if name == &name.to_lowercase() {
                continue;
            }
            if self.should_skip_name(name) {
                continue;
            }
            if lowered.contains(&name.to_lowercase()) && !found.contains(code) {
                found.push(code.clone());
            }
        }
        if !found.is_empty() {
            found.truncate(3);
            return found;
        }

        let stop_words = stop_words();
        for (index, token) in text.split_whitespace().enumerate() {
            let clean = token
                .chars()
                .filter(|ch| ch.is_ascii_alphabetic())
                .collect::<String>();
            let mut chars = clean.chars();
            let Some(first) = chars.next() else {
                continue;
            };
            if clean.len() >= 2
                && first.is_uppercase()
                && chars.all(|ch| ch.is_lowercase())
                && index > 0
                && !self.should_skip_name(&clean)
                && !stop_words.contains(clean.to_lowercase().as_str())
            {
                let code = clean.chars().take(3).collect::<String>().to_uppercase();
                if !found.contains(&code) {
                    found.push(code);
                }
            }
            if found.len() >= 3 {
                break;
            }
        }
        found
    }

    fn should_skip_name(&self, name: &str) -> bool {
        let lowered = name.to_lowercase();
        self.skip_names.iter().any(|skip| lowered.contains(skip))
    }
}

fn stop_words() -> HashSet<&'static str> {
    STOP_WORDS.iter().copied().collect()
}

fn topic_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || ch == '_' || ch == '-' {
            current.push(ch);
        } else if !current.is_empty() {
            if current
                .chars()
                .next()
                .map(|c| c.is_ascii_alphabetic())
                .unwrap_or(false)
            {
                tokens.push(current.clone());
            }
            current.clear();
        }
    }

    if !current.is_empty()
        && current
            .chars()
            .next()
            .map(|c| c.is_ascii_alphabetic())
            .unwrap_or(false)
    {
        tokens.push(current);
    }

    tokens
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if matches!(ch, '.' | '!' | '?' | '\n') {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_owned());
            }
            current.clear();
        } else {
            current.push(ch);
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_owned());
    }
    sentences
}

fn short_date(value: &str) -> String {
    value.chars().take(10).collect()
}

fn split_once3(input: &str, needle: char) -> Option<(&str, char, &str)> {
    let index = input.find(needle)?;
    Some((&input[..index], needle, &input[index + needle.len_utf8()..]))
}

fn format_weight(weight: f32) -> String {
    let mut out = format!("{weight:.3}");
    while out.contains('.') && out.ends_with('0') {
        out.pop();
    }
    if out.ends_with('.') {
        out.pop();
    }
    out
}

fn json_files(dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn extract_quoted_fragments(text: &str, quote_char: char) -> Vec<String> {
    let mut fragments = Vec::new();
    let mut start = None;
    for (index, ch) in text.char_indices() {
        if ch == quote_char {
            if let Some(start_index) = start.take() {
                let fragment = text[start_index..index].trim();
                if (8..=55).contains(&fragment.chars().count()) {
                    fragments.push(fragment.to_owned());
                }
            } else {
                start = Some(index + ch.len_utf8());
            }
        }
    }
    fragments
}

fn extract_single_quoted_fragments(text: &str) -> Vec<String> {
    let mut fragments = Vec::new();
    let chars = text.char_indices().collect::<Vec<_>>();
    let mut start = None;
    for (offset, (index, ch)) in chars.iter().enumerate() {
        if *ch != '\'' {
            continue;
        }

        let prev = offset
            .checked_sub(1)
            .and_then(|idx| chars.get(idx).map(|(_, ch)| *ch));
        let next = chars.get(offset + 1).map(|(_, ch)| *ch);
        let boundary_before = prev.is_none_or(|ch| ch.is_whitespace() || ch == '(');
        let boundary_after = next.is_none_or(|ch| {
            ch.is_whitespace() || matches!(ch, '.' | ',' | ';' | ':' | '!' | '?' | ')')
        });

        if start.is_none() && boundary_before {
            start = Some(*index + ch.len_utf8());
            continue;
        }

        if let Some(start_index) = start.take()
            && boundary_after
        {
            let fragment = text[start_index..*index].trim();
            if (8..=55).contains(&fragment.chars().count()) {
                fragments.push(fragment.to_owned());
            }
        }
    }
    fragments
}

fn extract_speech_fragments(text: &str) -> Vec<String> {
    let mut fragments = Vec::new();
    let lowered = text.to_lowercase();
    let markers = [
        "says:",
        "said:",
        "articulates:",
        "reveals:",
        "admits:",
        "confesses:",
        "asks:",
        "say:",
    ];

    for marker in markers {
        let mut offset = 0usize;
        while let Some(found) = lowered[offset..].find(marker) {
            let start = offset + found + marker.len();
            let rest = text[start..].trim_start_matches([' ', '"', '\'']);
            let end = rest.find(['.', '!', '?']).unwrap_or(rest.len()).min(55);
            let fragment = rest[..end].trim().trim_matches(['"', '\'']);
            if fragment.chars().count() >= 10 {
                fragments.push(fragment.to_owned());
            }
            offset = start;
        }
    }

    fragments
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use tempfile::tempdir;

    use super::{AaakDialect, AaakFile, AaakTunnel, AaakZettel};
    use crate::{ContentKind, DrawerMetadata};

    #[test]
    fn compress_includes_header_topics_and_flags() {
        let mut entities = BTreeMap::new();
        entities.insert("Benji".to_owned(), "BEN".to_owned());
        let dialect = AaakDialect::new(entities, Vec::new());
        let metadata = DrawerMetadata {
            content_kind: ContentKind::Documentation,
            wing: "project".to_owned(),
            room: "notes".to_owned(),
            source_file: Some("F:/Dev/example/README.md".to_owned()),
            chunk_index: 0,
            added_by: "mempalace".to_owned(),
            filed_at: Some("2026-04-09T01:00:00Z".to_owned()),
        };

        let compressed = dialect.compress(
            "Benji decided to switch API architecture because GraphQL was easier to reason about.",
            Some(&metadata),
        );

        assert!(compressed.contains("project|notes|2026-04-09|README"));
        assert!(compressed.contains("0:BEN"));
        assert!(compressed.contains("|api_"));
        assert!(compressed.contains("graphql"));
        assert!(compressed.contains("DECISION"));
    }

    #[test]
    fn decode_round_trips_basic_shape() {
        let dialect = AaakDialect::default();
        let decoded = dialect.decode("wing|room|2026-04-09|title\n0:BEN|topic|\"quote\"|joy");
        assert_eq!(decoded.header.unwrap().wing, "wing");
        assert_eq!(decoded.entries.len(), 1);
    }

    #[test]
    fn save_and_load_config_round_trips() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("entities.json");

        let mut entities = BTreeMap::new();
        entities.insert("Alice".to_owned(), "ALC".to_owned());
        let dialect = AaakDialect::new(entities, vec!["Sherlock".to_owned()]);
        dialect.save_config(&path).unwrap();

        let loaded = AaakDialect::from_config_path(&path).unwrap();
        let compressed = loaded.compress("Alice decided to migrate systems.", None);
        assert!(compressed.contains("0:ALC"));
    }

    #[test]
    fn encode_entity_quote_and_flags_match_python_path() {
        let mut entities = BTreeMap::new();
        entities.insert("Alice".to_owned(), "ALC".to_owned());
        let dialect = AaakDialect::new(entities, vec!["Sherlock".to_owned()]);

        assert_eq!(dialect.encode_entity("Alice"), Some("ALC".to_owned()));
        assert_eq!(dialect.encode_entity("Alice Chen"), Some("ALC".to_owned()));
        assert_eq!(dialect.encode_entity("Sherlock Holmes"), None);

        let zettel = AaakZettel {
            id: "zettel-001".to_owned(),
            people: vec!["Alice".to_owned()],
            topics: vec!["memory".to_owned(), "identity".to_owned()],
            content: "She said: \"I trust this system and hope it lasts.\"".to_owned(),
            origin_label: "genesis".to_owned(),
            notes: "foundational pillar and pivot moment".to_owned(),
            title: "001 - The first trust".to_owned(),
            emotional_weight: 0.95,
            emotional_tone: vec!["trust".to_owned(), "hope".to_owned()],
            origin_moment: true,
            sensitivity: "MAXIMUM".to_owned(),
            date_context: "2026-04-09".to_owned(),
        };

        assert_eq!(
            dialect.extract_key_quote(&zettel),
            "I trust this system and hope it lasts."
        );
        assert_eq!(dialect.encode_emotions(["trust", "hope"]), "trust+hope");
        assert!(dialect.get_flags(&zettel).contains("ORIGIN"));
        assert!(dialect.get_flags(&zettel).contains("CORE"));
        assert!(dialect.get_flags(&zettel).contains("GENESIS"));
        assert!(dialect.get_flags(&zettel).contains("PIVOT"));
        assert!(dialect.get_flags(&zettel).contains("SENSITIVE"));
    }

    #[test]
    fn encode_file_and_layer1_cover_zettel_pipeline() {
        let mut entities = BTreeMap::new();
        entities.insert("Alice".to_owned(), "ALC".to_owned());
        entities.insert("Bob".to_owned(), "BOB".to_owned());
        let dialect = AaakDialect::new(entities, Vec::new());

        let zettel_file = AaakFile {
            source_file: "001-origin_story.txt".to_owned(),
            emotional_arc: "fear->trust->hope".to_owned(),
            zettels: vec![
                AaakZettel {
                    id: "zettel-001".to_owned(),
                    people: vec!["Alice".to_owned()],
                    topics: vec!["memory".to_owned(), "identity".to_owned()],
                    content: "\"I trust this system.\"".to_owned(),
                    notes: "foundational pillar".to_owned(),
                    title: "001 - Trust memory".to_owned(),
                    emotional_weight: 0.91,
                    emotional_tone: vec!["trust".to_owned()],
                    origin_moment: true,
                    sensitivity: "".to_owned(),
                    date_context: "2026-04-01".to_owned(),
                    ..AaakZettel::default()
                },
                AaakZettel {
                    id: "zettel-002".to_owned(),
                    people: vec!["Bob".to_owned()],
                    topics: vec!["systems".to_owned()],
                    content: "We decided to switch because it was simpler.".to_owned(),
                    title: "002 - Switch".to_owned(),
                    emotional_weight: 0.5,
                    emotional_tone: vec!["determination".to_owned()],
                    date_context: "2026-04-02".to_owned(),
                    ..AaakZettel::default()
                },
            ],
            tunnels: vec![AaakTunnel {
                from_id: "zettel-001".to_owned(),
                to_id: "zettel-002".to_owned(),
                label: "reason: follows".to_owned(),
            }],
        };

        let encoded = dialect.encode_file(&zettel_file);
        assert!(encoded.contains("001|ALC+BOB|2026-04-01|origin_story"));
        assert!(encoded.contains("ARC:fear->trust->hope"));
        assert!(
            encoded.contains(
                "001:ALC|memory_identity|\"I trust this system.\"|0.91|trust|ORIGIN+CORE"
            )
        );
        assert!(encoded.contains("T:001<->002|reason"));

        let tmp = tempdir().unwrap();
        let file_path = tmp.path().join("file_001.json");
        fs::write(&file_path, serde_json::to_string(&zettel_file).unwrap()).unwrap();

        let compressed = dialect.compress_file(&file_path, None).unwrap();
        assert_eq!(compressed, encoded);

        let combined = dialect.compress_all(tmp.path(), None).unwrap();
        assert!(combined.contains(&encoded));
        assert!(combined.contains("---"));

        let mut identity_sections = BTreeMap::new();
        identity_sections.insert(
            "IDENTITY".to_owned(),
            vec!["ALC|builder|\"Trust matters\"".to_owned()],
        );
        let layer1 = dialect
            .generate_layer1(tmp.path(), None, Some(&identity_sections), 0.85)
            .unwrap();
        assert!(layer1.contains("## LAYER 1 -- ESSENTIAL STORY"));
        assert!(layer1.contains("=IDENTITY="));
        assert!(layer1.contains("=MOMENTS[2026-04-01]="));
        assert!(layer1.contains("ALC|Trust memory|\"I trust this system.\"|0.91|ORIGIN+CORE"));
        assert!(layer1.contains("=TUNNELS="));
        assert!(layer1.contains("reason"));
    }
}
