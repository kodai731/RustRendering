fn matches_keyword(text: &str, ascii_keywords: &[&str], japanese_keywords: &[&str]) -> bool {
    let lower = text.to_lowercase();

    // Japanese keywords: substring match (as before)
    if japanese_keywords.iter().any(|k| lower.contains(k)) {
        return true;
    }

    // ASCII keywords: word-boundary match (surrounded by non-alphanumeric or string edges)
    for kw in ascii_keywords {
        let bytes = lower.as_bytes();
        let pat = kw.as_bytes();
        if pat.len() > bytes.len() {
            continue;
        }
        for i in 0..=bytes.len().saturating_sub(pat.len()) {
            let before_ok = if i == 0 {
                true
            } else {
                !bytes[i - 1].is_ascii_alphanumeric()
            };
            let after_pos = i + pat.len();
            let after_ok = if after_pos >= bytes.len() {
                true
            } else {
                !bytes[after_pos].is_ascii_alphanumeric()
            };
            if before_ok && after_ok && &bytes[i..after_pos] == pat {
                return true;
            }
        }
    }

    false
}

fn has_left_bias(text: &str) -> bool {
    let lower = text.to_lowercase();
    // Japanese "左": substring match
    if lower.contains("左") {
        return true;
    }
    // ASCII "left": word-boundary match
    matches_keyword(text, &["left"], &[])
}

/// Build a GenDoP `text_motion` Movement caption from a natural-language utterance.
///
/// Scans the utterance for direction keywords (Japanese or English) and constructs
/// `"The camera continuously moves {dir} throughout the entire sequence."` where `{dir}`
/// is one or more directions joined by " and ".
///
/// Direction keywords:
/// - forward: 前, 寄, 近づ, forward, closer, in
/// - backward: 後, 引, 離れ, backward, away, out
/// - left: 左, left
/// - right: 右, right
/// - upward: 上, 上昇, up, rise
/// - downward: 下, 下降, down, lower
/// - yaw: 振る, 見回, pan, yaw (maps to "while continuously yawing {left|right}")
/// - orbit: 回り込, 回る, 周, orbit, circle, around (trucks sideways while yawing the
///   opposite way so the subject stays framed; 左/left mirrors the direction)
///
/// Returns `None` if no keywords are found.
pub fn build_movement_caption(utterance: &str) -> Option<String> {
    let mut directions: Vec<&str> = Vec::new();

    if matches_keyword(
        utterance,
        &["forward", "closer", "in", "follow", "chase"],
        &["前", "寄", "近づ", "追"],
    ) {
        directions.push("forward");
    }
    if matches_keyword(
        utterance,
        &["backward", "away", "out"],
        &["後", "引", "離れ"],
    ) {
        directions.push("backward");
    }

    let orbit = matches_keyword(
        utterance,
        &["orbit", "circle", "around"],
        &["回り込", "回る", "周"],
    );
    let yaw = orbit || matches_keyword(utterance, &["pan", "yaw"], &["振る", "見回"]);

    if orbit {
        directions.push(if has_left_bias(utterance) {
            "left"
        } else {
            "right"
        });
    } else if !yaw {
        if matches_keyword(utterance, &["left"], &["左"]) {
            directions.push("left");
        }
        if matches_keyword(utterance, &["right"], &["右"]) {
            directions.push("right");
        }
    }

    if matches_keyword(utterance, &["up", "rise"], &["上", "上昇"]) {
        directions.push("upward");
    }
    if matches_keyword(utterance, &["down", "lower"], &["下", "下降"]) {
        directions.push("downward");
    }

    if directions.is_empty() && !yaw {
        return None;
    }

    // Build the caption based on what we have
    let caption = match (!directions.is_empty(), yaw) {
        (true, true) => {
            let dirs = directions.join(" and ");
            let yaw_dir = match (has_left_bias(utterance), orbit) {
                (true, false) | (false, true) => "left",
                _ => "right",
            };
            format!(
                "The camera continuously moves {} while continuously yawing {} throughout the entire sequence.",
                dirs, yaw_dir
            )
        }
        (false, true) => {
            let yaw_dir = if has_left_bias(utterance) {
                "left"
            } else {
                "right"
            };
            format!(
                "The camera continuously yaws {} throughout the entire sequence.",
                yaw_dir
            )
        }
        (true, false) => {
            let dirs = directions.join(" and ");
            format!(
                "The camera continuously moves {} throughout the entire sequence.",
                dirs
            )
        }
        (false, false) => unreachable!(),
    };

    Some(caption)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_japanese() {
        let caption = build_movement_caption("カメラを前に動かす");
        assert_eq!(
            caption,
            Some(
                "The camera continuously moves forward throughout the entire sequence.".to_string()
            )
        );
    }

    #[test]
    fn test_backward_english() {
        let caption = build_movement_caption("move the camera backward");
        assert_eq!(
            caption,
            Some(
                "The camera continuously moves backward throughout the entire sequence."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_multiple_directions() {
        let caption = build_movement_caption("右に寄って");
        assert_eq!(
            caption,
            Some(
                "The camera continuously moves forward and right throughout the entire sequence."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_yaw_only_right() {
        let caption = build_movement_caption("振る");
        assert_eq!(
            caption,
            Some("The camera continuously yaws right throughout the entire sequence.".to_string())
        );
    }

    #[test]
    fn test_yaw_only_left() {
        let caption = build_movement_caption("左に振る");
        assert_eq!(
            caption,
            Some("The camera continuously yaws left throughout the entire sequence.".to_string())
        );
    }

    #[test]
    fn test_orbit_trucks_right_while_yawing_left() {
        assert_eq!(
            build_movement_caption("追いかけながら回り込んで"),
            Some(
                "The camera continuously moves forward and right while continuously yawing left throughout the entire sequence."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_orbit_left_mirrors_direction() {
        assert_eq!(
            build_movement_caption("orbit to the left"),
            Some(
                "The camera continuously moves left while continuously yawing right throughout the entire sequence."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_no_keywords_returns_none() {
        let caption = build_movement_caption("カメラを静止させる");
        assert_eq!(caption, None);
    }

    #[test]
    fn test_upward_and_downward() {
        let caption = build_movement_caption("up and down");
        assert_eq!(
            caption,
            Some(
                "The camera continuously moves upward and downward throughout the entire sequence."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_backward_away() {
        let caption = build_movement_caption("pull away from the subject");
        assert_eq!(
            caption,
            Some(
                "The camera continuously moves backward throughout the entire sequence."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_yaw_with_direction() {
        let caption = build_movement_caption("forward while 振る");
        assert_eq!(
            caption,
            Some(
                "The camera continuously moves forward while continuously yawing right throughout the entire sequence."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_japanese_backward() {
        let caption = build_movement_caption("カメラを後ろに引く");
        assert_eq!(
            caption,
            Some(
                "The camera continuously moves backward throughout the entire sequence."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_false_positive_into_not_forward() {
        // "into" should NOT match "in" as a forward keyword (word boundary check)
        let caption = build_movement_caption("move into the room");
        assert_eq!(caption, None);
    }

    #[test]
    fn test_false_positive_outward_not_backward() {
        // "outward" should NOT match "out" as a backward keyword (word boundary check)
        let caption = build_movement_caption("expand outward");
        assert_eq!(caption, None);
    }
}
