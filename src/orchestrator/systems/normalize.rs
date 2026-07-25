//! Utterance normalization used before keyword matching.
//!
//! Deliberately narrow: full-width to half-width, ASCII lowercase, whitespace
//! collapse. No stemming and no Japanese segmentation, because the keyword rules
//! are written against this exact output and anything cleverer makes them
//! unpredictable to author.

const FULLWIDTH_FIRST: u32 = 0xFF01;
const FULLWIDTH_LAST: u32 = 0xFF5E;
const FULLWIDTH_TO_ASCII_OFFSET: u32 = 0xFEE0;
const IDEOGRAPHIC_SPACE: char = '\u{3000}';

pub fn normalize_utterance(utterance: &str) -> String {
    let widened: String = utterance.chars().map(to_half_width).collect();
    collapse_whitespace(&widened.to_lowercase())
}

fn to_half_width(character: char) -> char {
    if character == IDEOGRAPHIC_SPACE {
        return ' ';
    }

    let code = character as u32;
    if (FULLWIDTH_FIRST..=FULLWIDTH_LAST).contains(&code) {
        return char::from_u32(code - FULLWIDTH_TO_ASCII_OFFSET).unwrap_or(character);
    }
    character
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<&str>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercases_ascii_and_leaves_japanese_alone() {
        assert_eq!(
            normalize_utterance("Play The Animation"),
            "play the animation"
        );
        assert_eq!(normalize_utterance("再生シテ"), "再生シテ");
    }

    #[test]
    fn converts_full_width_ascii_to_half_width() {
        assert_eq!(normalize_utterance("Ｌｉｇｈｔ０１"), "light01");
    }

    #[test]
    fn converts_the_full_width_question_mark() {
        assert_eq!(
            normalize_utterance("今何が選択されている？"),
            "今何が選択されている?"
        );
    }

    #[test]
    fn collapses_runs_of_whitespace_and_trims() {
        assert_eq!(
            normalize_utterance("  play   the\tanimation \n"),
            "play the animation"
        );
    }

    #[test]
    fn converts_the_ideographic_space() {
        assert_eq!(normalize_utterance("再生\u{3000}して"), "再生 して");
    }

    #[test]
    fn an_empty_utterance_normalizes_to_empty() {
        assert_eq!(normalize_utterance("   "), "");
    }

    #[test]
    fn normalization_is_idempotent() {
        let once = normalize_utterance("　Ｐｌａｙ  The  Ａｎｉｍａｔｉｏｎ？");
        assert_eq!(normalize_utterance(&once), once);
    }
}
