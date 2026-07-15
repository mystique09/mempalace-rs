use std::collections::{BTreeMap, HashSet};

use serde_json::Value;

#[test]
fn multi_domain_corpus_has_portable_matchers_domain_floors_and_fixed_split() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../benchmarks/multi_domain_queries.json"
    ))
    .unwrap();
    let documents = corpus["documents"].as_array().unwrap();
    let baselines = corpus["baseline_recall_at_5"].as_object().unwrap();
    let mut query_ids = HashSet::new();
    let mut kind_counts = BTreeMap::<&str, usize>::new();
    let mut dev_count = 0;

    for document in documents {
        let kind = document["content_kind"].as_str().unwrap();
        let matcher = document["target_matcher"].as_object().unwrap();
        assert!(
            matcher.contains_key("drawer_id")
                || matcher.contains_key("relative_path")
                || matcher.contains_key("content_contains")
        );
        if kind == "code" {
            assert!(matcher["relative_path"].as_str().is_some());
            assert!(!matcher["relative_path"].as_str().unwrap().starts_with('/'));
            assert!(!matcher["content_contains"].as_array().unwrap().is_empty());
        }

        let domain = document["domain"].as_str().unwrap();
        assert!(baselines.contains_key(domain));
        for query in document["queries"].as_array().unwrap() {
            assert!(query_ids.insert(query["id"].as_str().unwrap()));
            *kind_counts.entry(kind).or_default() += 1;
            dev_count += usize::from(query["split"] == "dev");
        }
    }

    assert_eq!(query_ids.len(), 100);
    assert_eq!(dev_count, 25);
    assert!(kind_counts["code"] >= 30);
    assert!(kind_counts["conversation"] >= 25);
    assert!(kind_counts["documentation"] >= 20);
    assert!(kind_counts["diary"] >= 15);
    assert!(kind_counts["prose"] >= 10);
}

#[test]
fn longmemeval_split_is_tracked_disjoint_and_complete() {
    let split: Value = serde_json::from_str(include_str!(
        "../../../benchmarks/longmemeval_split_50_450.json"
    ))
    .unwrap();
    let dev = split["dev"].as_array().unwrap();
    let held_out = split["held_out"].as_array().unwrap();
    let dev_ids = dev
        .iter()
        .map(|id| id.as_str().unwrap())
        .collect::<HashSet<_>>();
    let held_ids = held_out
        .iter()
        .map(|id| id.as_str().unwrap())
        .collect::<HashSet<_>>();

    assert_eq!(dev.len(), 50);
    assert_eq!(held_out.len(), 450);
    assert!(dev_ids.is_disjoint(&held_ids));
    assert_eq!(dev_ids.len() + held_ids.len(), 500);
}
