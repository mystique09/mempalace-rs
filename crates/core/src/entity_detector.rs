use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use ignore::WalkBuilder;
use regex::Regex;

use crate::Result;

const PERSON_VERB_PATTERNS: &[&str] = &[
    r"\b{name}\s+said\b",
    r"\b{name}\s+asked\b",
    r"\b{name}\s+told\b",
    r"\b{name}\s+replied\b",
    r"\b{name}\s+laughed\b",
    r"\b{name}\s+smiled\b",
    r"\b{name}\s+cried\b",
    r"\b{name}\s+felt\b",
    r"\b{name}\s+thinks?\b",
    r"\b{name}\s+wants?\b",
    r"\b{name}\s+loves?\b",
    r"\b{name}\s+hates?\b",
    r"\b{name}\s+knows?\b",
    r"\b{name}\s+decided\b",
    r"\b{name}\s+pushed\b",
    r"\b{name}\s+wrote\b",
    r"\bhey\s+{name}\b",
    r"\bthanks?\s+{name}\b",
    r"\bhi\s+{name}\b",
    r"\bdear\s+{name}\b",
];

const DIALOGUE_PATTERNS: &[&str] = &[
    r"^>\s*{name}[:\s]",
    r"^{name}:\s",
    r"^\[{name}\]",
    r#""{name}\s+said"#,
];

const PRONOUN_PATTERNS: &[&str] = &[
    r"\bshe\b",
    r"\bher\b",
    r"\bhers\b",
    r"\bhe\b",
    r"\bhim\b",
    r"\bhis\b",
    r"\bthey\b",
    r"\bthem\b",
    r"\btheir\b",
];

const PROJECT_VERB_PATTERNS: &[&str] = &[
    r"\bbuilding\s+{name}\b",
    r"\bbuilt\s+{name}\b",
    r"\bship(?:ping|ped)?\s+{name}\b",
    r"\blaunch(?:ing|ed)?\s+{name}\b",
    r"\bdeploy(?:ing|ed)?\s+{name}\b",
    r"\binstall(?:ing|ed)?\s+{name}\b",
    r"\bthe\s+{name}\s+architecture\b",
    r"\bthe\s+{name}\s+pipeline\b",
    r"\bthe\s+{name}\s+system\b",
    r"\bthe\s+{name}\s+repo\b",
    r"\b{name}\s+v\d+\b",
    r"\b{name}\.py\b",
    r"\b{name}-core\b",
    r"\b{name}-local\b",
    r"\bimport\s+{name}\b",
    r"\bpip\s+install\s+{name}\b",
];

const PROSE_EXTENSIONS: &[&str] = &[".txt", ".md", ".rst", ".csv"];
const READABLE_EXTENSIONS: &[&str] = &[
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".csv", ".rst", ".toml", ".sh",
    ".rb", ".go", ".rs",
];
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
];

const MAX_BYTES_PER_FILE: usize = 5_000;

const STOPWORDS: &[&str] = &[
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "we",
    "our",
    "you",
    "your",
    "i",
    "my",
    "me",
    "he",
    "she",
    "his",
    "her",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "if",
    "then",
    "so",
    "not",
    "no",
    "yes",
    "ok",
    "okay",
    "just",
    "very",
    "really",
    "also",
    "already",
    "still",
    "even",
    "only",
    "here",
    "there",
    "now",
    "too",
    "up",
    "out",
    "about",
    "like",
    "use",
    "get",
    "got",
    "make",
    "made",
    "take",
    "put",
    "come",
    "go",
    "see",
    "know",
    "think",
    "true",
    "false",
    "none",
    "null",
    "new",
    "old",
    "all",
    "any",
    "some",
    "return",
    "print",
    "def",
    "class",
    "import",
    "step",
    "usage",
    "run",
    "check",
    "find",
    "add",
    "set",
    "list",
    "args",
    "dict",
    "str",
    "int",
    "bool",
    "path",
    "file",
    "type",
    "name",
    "note",
    "example",
    "option",
    "result",
    "error",
    "warning",
    "info",
    "every",
    "each",
    "more",
    "less",
    "next",
    "last",
    "first",
    "second",
    "stack",
    "layer",
    "mode",
    "test",
    "stop",
    "start",
    "copy",
    "move",
    "source",
    "target",
    "output",
    "input",
    "data",
    "item",
    "key",
    "value",
    "returns",
    "raises",
    "yields",
    "self",
    "cls",
    "kwargs",
    "world",
    "well",
    "want",
    "topic",
    "choose",
    "social",
    "cars",
    "phones",
    "healthcare",
    "ex",
    "machina",
    "deus",
    "human",
    "humans",
    "people",
    "things",
    "something",
    "nothing",
    "everything",
    "anything",
    "someone",
    "everyone",
    "anyone",
    "way",
    "time",
    "day",
    "life",
    "place",
    "thing",
    "part",
    "kind",
    "sort",
    "case",
    "point",
    "idea",
    "fact",
    "sense",
    "question",
    "answer",
    "reason",
    "number",
    "version",
    "system",
    "hey",
    "hi",
    "hello",
    "thanks",
    "thank",
    "right",
    "let",
    "click",
    "hit",
    "press",
    "tap",
    "drag",
    "drop",
    "open",
    "close",
    "save",
    "load",
    "launch",
    "install",
    "download",
    "upload",
    "scroll",
    "select",
    "enter",
    "submit",
    "cancel",
    "confirm",
    "delete",
    "paste",
    "type",
    "write",
    "read",
    "search",
    "show",
    "hide",
    "desktop",
    "documents",
    "downloads",
    "users",
    "home",
    "library",
    "applications",
    "preferences",
    "settings",
    "terminal",
    "actor",
    "vector",
    "remote",
    "control",
    "duration",
    "fetch",
    "agents",
    "tools",
    "others",
    "guards",
    "ethics",
    "regulation",
    "learning",
    "thinking",
    "memory",
    "language",
    "intelligence",
    "technology",
    "society",
    "culture",
    "future",
    "history",
    "science",
    "model",
    "models",
    "network",
    "networks",
    "training",
    "inference",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectedEntityKind {
    Person,
    Project,
    Uncertain,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetectedEntity {
    pub name: String,
    pub kind: DetectedEntityKind,
    pub confidence: f32,
    pub frequency: usize,
    pub signals: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DetectedEntities {
    pub people: Vec<DetectedEntity>,
    pub projects: Vec<DetectedEntity>,
    pub uncertain: Vec<DetectedEntity>,
}

struct ScoredEntity {
    person_score: usize,
    project_score: usize,
    person_signals: Vec<String>,
    project_signals: Vec<String>,
}

pub fn scan_for_detection(project_dir: impl AsRef<Path>, max_files: usize) -> Result<Vec<PathBuf>> {
    let project_dir = project_dir.as_ref();
    let mut builder = WalkBuilder::new(project_dir);
    builder.hidden(false);
    builder.require_git(false);
    builder.sort_by_file_path(|a, b| a.cmp(b));
    builder.filter_entry(|entry| {
        if !entry
            .file_type()
            .map(|file_type| file_type.is_dir())
            .unwrap_or(false)
        {
            return true;
        }

        let Some(name) = entry.path().file_name().map(|name| name.to_string_lossy()) else {
            return false;
        };
        !SKIP_DIRS.contains(&name.as_ref())
    });

    let mut prose = Vec::new();
    let mut readable = Vec::new();
    for result in builder.build() {
        let Ok(entry) = result else {
            continue;
        };

        if !entry
            .file_type()
            .map(|file_type| file_type.is_file())
            .unwrap_or(false)
        {
            continue;
        }

        let Some(extension) = entry
            .path()
            .extension()
            .map(|ext| format!(".{}", ext.to_string_lossy().to_lowercase()))
        else {
            continue;
        };

        if PROSE_EXTENSIONS.contains(&extension.as_str()) {
            prose.push(entry.into_path());
        } else if READABLE_EXTENSIONS.contains(&extension.as_str()) {
            readable.push(entry.into_path());
        }
    }

    let mut files = if prose.len() >= 3 {
        prose
    } else {
        prose.into_iter().chain(readable).collect::<Vec<_>>()
    };

    if max_files > 0 && files.len() > max_files {
        files.truncate(max_files);
    }
    Ok(files)
}

pub fn detect_entities(file_paths: &[PathBuf], max_files: usize) -> Result<DetectedEntities> {
    let mut all_text = Vec::new();
    let mut all_lines = Vec::new();
    let mut files_read = 0usize;

    for path in file_paths {
        if max_files > 0 && files_read >= max_files {
            break;
        }

        let Ok(raw) = fs::read(path) else {
            continue;
        };
        let raw = &raw[..raw.len().min(MAX_BYTES_PER_FILE)];
        let content = String::from_utf8_lossy(raw).into_owned();
        all_lines.extend(content.lines().map(ToOwned::to_owned));
        all_text.push(content);
        files_read += 1;
    }

    let combined_text = all_text.join("\n");
    let candidates = extract_candidates(&combined_text)?;
    if candidates.is_empty() {
        return Ok(DetectedEntities::default());
    }

    let mut detected = DetectedEntities::default();
    let mut sorted = candidates.into_iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

    for (name, frequency) in sorted {
        let scores = score_entity(&name, &combined_text, &all_lines)?;
        let entity = classify_entity(name, frequency, scores);
        match entity.kind {
            DetectedEntityKind::Person => detected.people.push(entity),
            DetectedEntityKind::Project => detected.projects.push(entity),
            DetectedEntityKind::Uncertain => detected.uncertain.push(entity),
        }
    }

    detected
        .people
        .sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    detected
        .projects
        .sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    detected
        .uncertain
        .sort_by(|left, right| right.frequency.cmp(&left.frequency));

    detected.people.truncate(15);
    detected.projects.truncate(10);
    detected.uncertain.truncate(8);
    Ok(detected)
}

fn extract_candidates(text: &str) -> Result<BTreeMap<String, usize>> {
    let single = Regex::new(r"\b([A-Z][a-z]{1,19})\b")?;
    let multi = Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")?;
    let mut counts = BTreeMap::new();

    for capture in single.captures_iter(text) {
        let Some(word) = capture.get(1).map(|value| value.as_str()) else {
            continue;
        };
        if !STOPWORDS.contains(&word.to_lowercase().as_str()) && word.len() > 1 {
            *counts.entry(word.to_owned()).or_insert(0) += 1;
        }
    }

    for capture in multi.captures_iter(text) {
        let Some(phrase) = capture.get(1).map(|value| value.as_str()) else {
            continue;
        };
        if !phrase
            .split_whitespace()
            .any(|part| STOPWORDS.contains(&part.to_lowercase().as_str()))
        {
            *counts.entry(phrase.to_owned()).or_insert(0) += 1;
        }
    }

    Ok(counts
        .into_iter()
        .filter(|(_, count)| *count >= 3)
        .collect::<BTreeMap<_, _>>())
}

fn score_entity(name: &str, text: &str, lines: &[String]) -> Result<ScoredEntity> {
    let escaped = regex::escape(name);
    let mut person_score = 0usize;
    let mut project_score = 0usize;
    let mut person_signals = Vec::new();
    let mut project_signals = Vec::new();

    for pattern in DIALOGUE_PATTERNS {
        let regex = Regex::new(&format!("(?im){}", pattern.replace("{name}", &escaped)))?;
        let matches = regex.find_iter(text).count();
        if matches > 0 {
            person_score += matches * 3;
            push_signal(&mut person_signals, format!("dialogue marker ({matches}x)"));
        }
    }

    for pattern in PERSON_VERB_PATTERNS {
        let regex = Regex::new(&format!("(?i){}", pattern.replace("{name}", &escaped)))?;
        let matches = regex.find_iter(text).count();
        if matches > 0 {
            person_score += matches * 2;
            push_signal(
                &mut person_signals,
                format!("'{name} ...' action ({matches}x)"),
            );
        }
    }

    let name_lower = name.to_lowercase();
    let mut pronoun_hits = 0usize;
    for (index, line) in lines.iter().enumerate() {
        if !line.to_lowercase().contains(&name_lower) {
            continue;
        }
        let start = index.saturating_sub(2);
        let end = (index + 3).min(lines.len());
        let window = lines[start..end].join(" ").to_lowercase();
        if PRONOUN_PATTERNS.iter().any(|pattern| {
            Regex::new(pattern)
                .ok()
                .is_some_and(|regex| regex.is_match(&window))
        }) {
            pronoun_hits += 1;
        }
    }
    if pronoun_hits > 0 {
        person_score += pronoun_hits * 2;
        push_signal(
            &mut person_signals,
            format!("pronoun nearby ({pronoun_hits}x)"),
        );
    }

    let direct = Regex::new(&format!(
        "(?i)\\bhey\\s+{escaped}\\b|\\bthanks?\\s+{escaped}\\b|\\bhi\\s+{escaped}\\b"
    ))?
    .find_iter(text)
    .count();
    if direct > 0 {
        person_score += direct * 4;
        push_signal(
            &mut person_signals,
            format!("addressed directly ({direct}x)"),
        );
    }

    for pattern in PROJECT_VERB_PATTERNS {
        let regex = Regex::new(&format!("(?i){}", pattern.replace("{name}", &escaped)))?;
        let matches = regex.find_iter(text).count();
        if matches > 0 {
            project_score += matches * 2;
            push_signal(&mut project_signals, format!("project verb ({matches}x)"));
        }
    }

    let versioned = Regex::new(&format!("(?i)\\b{escaped}[-v]\\w+"))?
        .find_iter(text)
        .count();
    if versioned > 0 {
        project_score += versioned * 3;
        push_signal(
            &mut project_signals,
            format!("versioned/hyphenated ({versioned}x)"),
        );
    }

    let code_ref = Regex::new(&format!(
        "(?i)\\b{escaped}\\.(py|js|ts|yaml|yml|json|sh)\\b"
    ))?
    .find_iter(text)
    .count();
    if code_ref > 0 {
        project_score += code_ref * 3;
        push_signal(
            &mut project_signals,
            format!("code file reference ({code_ref}x)"),
        );
    }

    Ok(ScoredEntity {
        person_score,
        project_score,
        person_signals,
        project_signals,
    })
}

fn classify_entity(name: String, frequency: usize, scores: ScoredEntity) -> DetectedEntity {
    let total = scores.person_score + scores.project_score;
    if total == 0 {
        return DetectedEntity {
            name,
            kind: DetectedEntityKind::Uncertain,
            confidence: (frequency as f32 / 50.0).min(0.4),
            frequency,
            signals: vec![format!("appears {frequency}x, no strong type signals")],
        };
    }

    let person_ratio = scores.person_score as f32 / total as f32;
    let signal_categories = scores
        .person_signals
        .iter()
        .filter_map(|signal| {
            if signal.contains("dialogue") {
                Some("dialogue")
            } else if signal.contains("action") {
                Some("action")
            } else if signal.contains("pronoun") {
                Some("pronoun")
            } else if signal.contains("addressed") {
                Some("addressed")
            } else {
                None
            }
        })
        .collect::<BTreeSet<_>>();
    let has_two_signal_types = signal_categories.len() >= 2;

    if person_ratio >= 0.7 && has_two_signal_types && scores.person_score >= 5 {
        DetectedEntity {
            name,
            kind: DetectedEntityKind::Person,
            confidence: (0.5 + person_ratio * 0.5).min(0.99),
            frequency,
            signals: if scores.person_signals.is_empty() {
                vec![format!("appears {frequency}x")]
            } else {
                scores.person_signals
            },
        }
    } else if person_ratio >= 0.7 && (!has_two_signal_types || scores.person_score < 5) {
        let mut signals = scores.person_signals;
        signals.push(format!("appears {frequency}x - pronoun-only match"));
        DetectedEntity {
            name,
            kind: DetectedEntityKind::Uncertain,
            confidence: 0.4,
            frequency,
            signals,
        }
    } else if person_ratio <= 0.3 {
        DetectedEntity {
            name,
            kind: DetectedEntityKind::Project,
            confidence: (0.5 + (1.0 - person_ratio) * 0.5).min(0.99),
            frequency,
            signals: if scores.project_signals.is_empty() {
                vec![format!("appears {frequency}x")]
            } else {
                scores.project_signals
            },
        }
    } else {
        let mut signals = scores
            .person_signals
            .into_iter()
            .chain(scores.project_signals)
            .take(3)
            .collect::<Vec<_>>();
        signals.push("mixed signals - needs review".to_owned());
        DetectedEntity {
            name,
            kind: DetectedEntityKind::Uncertain,
            confidence: 0.5,
            frequency,
            signals,
        }
    }
}

fn push_signal(signals: &mut Vec<String>, signal: String) {
    if signals.len() < 3 && !signals.contains(&signal) {
        signals.push(signal);
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::{DetectedEntityKind, detect_entities, scan_for_detection};

    #[test]
    fn scan_for_detection_prefers_prose_files() {
        let tmp = tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("docs")).unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("README.md"), "Riley said hello\n").unwrap();
        fs::write(tmp.path().join("notes.txt"), "Lantern launched soon\n").unwrap();
        fs::write(tmp.path().join("docs").join("guide.rst"), "Riley smiled\n").unwrap();
        fs::write(tmp.path().join("src").join("main.rs"), "fn main() {}\n").unwrap();

        let files = scan_for_detection(tmp.path(), 10).unwrap();
        assert_eq!(files.len(), 3);
        assert!(files.iter().all(|path| {
            matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("md" | "txt" | "rst")
            )
        }));
    }

    #[test]
    fn detect_entities_classifies_people_and_projects() {
        let tmp = tempdir().unwrap();
        let content = [
            "Riley said we should ship Lantern soon.",
            "Riley smiled and Riley asked whether Lantern v2 was ready.",
            "Thanks Riley.",
            "We are building Lantern and launched Lantern last week.",
            "The Lantern architecture needs work.",
            "pip install Lantern",
        ]
        .join("\n");
        let file = tmp.path().join("notes.md");
        fs::write(&file, content).unwrap();

        let detected = detect_entities(&[file], 10).unwrap();
        assert!(
            detected
                .people
                .iter()
                .any(|entity| entity.name == "Riley" && entity.kind == DetectedEntityKind::Person)
        );
        assert!(detected.projects.iter().any(|entity| {
            entity.name == "Lantern" && entity.kind == DetectedEntityKind::Project
        }));
    }
}
