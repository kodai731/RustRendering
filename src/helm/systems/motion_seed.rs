//! Motion seed catalog — pure-function module for matching source clips to motion categories.
//!
//! The JSON data file is embedded at compile time via `include_str!`. This module provides
//! loading, parsing, and candidate-finding functions with no ECS wiring.

use std::collections::HashMap;

use serde::Deserialize;

use crate::animation::editable::SourceClipId;
use crate::helm::components::tool_call::MotionCategory;
use crate::helm::systems::normalize::normalize_utterance;

const MOTION_SEED_CATALOG_JSON: &str = include_str!("../data/motion_seed_catalog.json");

#[derive(Deserialize)]
struct RawCatalog {
    categories: HashMap<String, RawCategory>,
}

#[derive(Deserialize)]
struct RawCategory {
    patterns: Vec<String>,
}

/// Motion seed catalog mapping category keys to their matching patterns.
#[derive(Clone, Debug)]
pub struct MotionSeedCatalog {
    categories: HashMap<String, Vec<String>>,
}

impl MotionSeedCatalog {
    /// Returns the patterns for a given category key.
    pub fn patterns(&self, key: &str) -> Option<&[String]> {
        self.categories.get(key).map(|v| v.as_slice())
    }
}

/// Load and verify the motion seed catalog from the embedded JSON.
///
/// Returns an error if any `MotionCategory` variant is missing a corresponding key.
pub fn load_motion_seed_catalog() -> Result<MotionSeedCatalog, String> {
    let raw: RawCatalog = serde_json::from_str(MOTION_SEED_CATALOG_JSON)
        .expect("motion_seed_catalog.json is embedded at compile time and must parse");

    let mut categories = HashMap::new();
    for (key, cat) in &raw.categories {
        categories.insert(key.clone(), cat.patterns.clone());
    }

    // Verify all MotionCategory variants have corresponding keys
    for variant in [
        MotionCategory::Walk,
        MotionCategory::Run,
        MotionCategory::Idle,
        MotionCategory::Jump,
        MotionCategory::Turn,
    ] {
        if !categories.contains_key(variant.as_str()) {
            return Err(format!("missing category key: {}", variant.as_str()));
        }
    }

    Ok(MotionSeedCatalog { categories })
}

/// Find source clips whose normalized names match any pattern for the given category.
///
/// Each clip name is normalized via `normalize_utterance` and checked against all patterns
/// for the category. Matching clips are returned sorted by their (original) name in ascending order.
pub fn find_seed_candidates(
    catalog: &MotionSeedCatalog,
    category: MotionCategory,
    clip_names: &[(SourceClipId, String)],
) -> Vec<(SourceClipId, String)> {
    let key = category.as_str();
    let patterns = match catalog.patterns(key) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut matches: Vec<(SourceClipId, String)> = clip_names
        .iter()
        .filter(|(_, name)| {
            let normalized = normalize_utterance(name);
            patterns.iter().any(|pattern| normalized.contains(pattern.as_str()))
        })
        .cloned()
        .collect();

    matches.sort_by(|a, b| a.1.cmp(&b.1));
    matches
}

/// Pick the candidate at `counter % candidates.len()`, or `None` if empty.
pub fn pick_round_robin<T: Clone>(candidates: &[T], counter: usize) -> Option<T> {
    if candidates.is_empty() {
        return None;
    }
    Some(candidates[counter % candidates.len()].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clips(names: &[&str]) -> Vec<(SourceClipId, String)> {
        names
            .iter()
            .enumerate()
            .map(|(i, name)| (i as SourceClipId, name.to_string()))
            .collect::<Vec<_>>()
    }

    #[test]
    fn load_catalog_succeeds_and_has_all_five_categories() {
        let catalog = load_motion_seed_catalog().expect("catalog should load");
        assert!(catalog.patterns("walk").is_some());
        assert!(catalog.patterns("run").is_some());
        assert!(catalog.patterns("idle").is_some());
        assert!(catalog.patterns("jump").is_some());
        assert!(catalog.patterns("turn").is_some());
    }

    #[test]
    fn walk_matches_ascii_and_japanese() {
        let catalog = load_motion_seed_catalog().unwrap();
        let clips = make_clips(&["My_Walk_01", "歩きループ", "RunFast"]);
        let candidates = find_seed_candidates(&catalog, MotionCategory::Walk, &clips);

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].1, "My_Walk_01");
        assert_eq!(candidates[1].1, "歩きループ");
    }

    #[test]
    fn non_matching_clips_are_excluded() {
        let catalog = load_motion_seed_catalog().unwrap();
        let clips = make_clips(&["idle_stand", "jump_high", "walk_forward"]);
        let candidates = find_seed_candidates(&catalog, MotionCategory::Run, &clips);

        assert!(candidates.is_empty());
    }

    #[test]
    fn candidates_are_sorted_by_name() {
        let catalog = load_motion_seed_catalog().unwrap();
        let clips = make_clips(&["z_walk", "a_walk", "m_walk"]);
        let candidates = find_seed_candidates(&catalog, MotionCategory::Walk, &clips);

        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].1, "a_walk");
        assert_eq!(candidates[1].1, "m_walk");
        assert_eq!(candidates[2].1, "z_walk");
    }

    #[test]
    fn round_robin_counter_zero() {
        let items = vec!["a", "b", "c"];
        assert_eq!(pick_round_robin(&items, 0), Some("a"));
    }

    #[test]
    fn round_robin_counter_one() {
        let items = vec!["a", "b", "c"];
        assert_eq!(pick_round_robin(&items, 1), Some("b"));
    }

    #[test]
    fn round_robin_counter_two() {
        let items = vec!["a", "b", "c"];
        assert_eq!(pick_round_robin(&items, 2), Some("c"));
    }

    #[test]
    fn round_robin_wraps_around() {
        let items = vec!["a", "b", "c"];
        assert_eq!(pick_round_robin(&items, 3), Some("a"));
        assert_eq!(pick_round_robin(&items, 5), Some("c"));
    }

    #[test]
    fn round_robin_empty_returns_none() {
        let items: Vec<&str> = vec![];
        assert_eq!(pick_round_robin(&items, 0), None);
    }
}
