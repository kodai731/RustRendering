//! Fills the speed slot from words in the utterance.
//!
//! This is slot extraction, not routing, which is why it outlived the keyword rule
//! table it used to share a file with. A modifier can only set `speed` on a route
//! the embedding router already chose, so a wrong match degrades one argument of an
//! otherwise correct call. The rules it sat beside chose the route itself and could
//! turn a save into a screenshot.
//!
//! Terms are matched against `normalize::normalize_utterance` output, so they must
//! be lowercase. Longer terms come first within a preset because the first hit wins
//! and `半分の速さ` must not be reached through `速`.

use crate::helm::components::tool_call::SpeedPreset;

const SLOW_TERMS: [&str; 8] = [
    "slowly",
    "slower",
    "slow",
    "遅く",
    "ゆっくり",
    "スロー",
    "低速",
    "半分の速さ",
];

const FAST_TERMS: [&str; 7] = [
    "faster",
    "fast",
    "quickly",
    "速く",
    "高速",
    "倍速",
    "早送り",
];

const NORMAL_TERMS: [&str; 7] = [
    "normal speed",
    "standard",
    "default speed",
    "標準",
    "等速",
    "通常速度",
    "普通",
];

const SPEED_MODIFIERS: [(SpeedPreset, &[&str]); 3] = [
    (SpeedPreset::Slow, &SLOW_TERMS),
    (SpeedPreset::Fast, &FAST_TERMS),
    (SpeedPreset::Normal, &NORMAL_TERMS),
];

pub fn extract_speed_modifier(normalized: &str) -> Option<SpeedPreset> {
    SPEED_MODIFIERS
        .iter()
        .find(|(_, terms)| terms.iter().any(|term| normalized.contains(term)))
        .map(|(preset, _)| *preset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helm::systems::normalize::normalize_utterance;

    fn extract(utterance: &str) -> Option<SpeedPreset> {
        extract_speed_modifier(&normalize_utterance(utterance))
    }

    #[test]
    fn every_term_is_lowercase_so_it_matches_normalized_text() {
        for (_, terms) in SPEED_MODIFIERS {
            for term in terms {
                assert_eq!(*term, term.to_lowercase());
            }
        }
    }

    #[test]
    fn english_modifiers_map_to_their_preset() {
        assert_eq!(extract("generate a slow walk"), Some(SpeedPreset::Slow));
        assert_eq!(extract("make him run fast"), Some(SpeedPreset::Fast));
        assert_eq!(extract("use the normal speed"), Some(SpeedPreset::Normal));
    }

    #[test]
    fn japanese_modifiers_map_to_their_preset() {
        assert_eq!(
            extract("ゆっくり歩くモーションを作って"),
            Some(SpeedPreset::Slow)
        );
        assert_eq!(extract("倍速で再生して"), Some(SpeedPreset::Fast));
        assert_eq!(extract("標準の速度に戻して"), Some(SpeedPreset::Normal));
    }

    #[test]
    fn an_utterance_without_a_modifier_yields_nothing() {
        assert_eq!(extract("play the animation"), None);
        assert_eq!(extract("シーンを保存して"), None);
    }

    #[test]
    fn full_width_input_still_matches() {
        assert_eq!(extract("ＦＡＳＴ で再生"), Some(SpeedPreset::Fast));
    }
}
