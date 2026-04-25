use std::collections::HashMap;

use crate::{BoneId, Skeleton};

pub const BONE_NAME_TOKEN_LENGTH: usize = 32;
pub const PAD_TOKEN: i64 = 0;
pub const UNK_TOKEN: i64 = 1;

pub fn tokenize_bone_name(name: &str) -> [i64; BONE_NAME_TOKEN_LENGTH] {
    let lower = name.to_lowercase();
    let tokens: Vec<i64> = lower.chars().map(char_to_token).collect();

    let mut result = [PAD_TOKEN; BONE_NAME_TOKEN_LENGTH];

    if tokens.is_empty() {
        return result;
    }

    let start = if tokens.len() > BONE_NAME_TOKEN_LENGTH {
        tokens.len() - BONE_NAME_TOKEN_LENGTH
    } else {
        0
    };
    let used_tokens = &tokens[start..];

    let offset = BONE_NAME_TOKEN_LENGTH - used_tokens.len();
    for (i, &token) in used_tokens.iter().enumerate() {
        result[offset + i] = token;
    }

    result
}

fn char_to_token(c: char) -> i64 {
    match c {
        '_' => 2,
        '.' => 3,
        '-' => 4,
        ' ' => 5,
        ':' => 6,
        '/' => 7,
        'a'..='z' => 8 + (c as i64 - 'a' as i64),
        '0'..='9' => 34 + (c as i64 - '0' as i64),
        _ => UNK_TOKEN,
    }
}

pub fn compute_bone_name_tokens(
    skeleton: &Skeleton,
) -> HashMap<BoneId, [i64; BONE_NAME_TOKEN_LENGTH]> {
    skeleton
        .bones
        .iter()
        .map(|bone| (bone.id, tokenize_bone_name(&bone.name)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_left_shoulder() {
        let tokens = tokenize_bone_name("LeftShoulder");
        let non_pad: Vec<i64> = tokens.iter().copied().filter(|&t| t != PAD_TOKEN).collect();

        assert_eq!(non_pad.len(), 12);
        assert_eq!(non_pad[0], 8 + 11);
        assert_eq!(non_pad[1], 8 + 4);
    }

    #[test]
    fn test_dots_and_underscores() {
        let tokens = tokenize_bone_name("Bone.003_03");
        let non_pad: Vec<i64> = tokens.iter().copied().filter(|&t| t != PAD_TOKEN).collect();

        assert!(non_pad.contains(&3));
        assert!(non_pad.contains(&2));
        assert!(non_pad.contains(&34));
    }

    #[test]
    fn test_truncation_long_name() {
        let long_name = "a".repeat(40);
        let tokens = tokenize_bone_name(&long_name);

        assert_eq!(tokens.len(), BONE_NAME_TOKEN_LENGTH);
        for &t in &tokens {
            assert_eq!(t, 8);
        }
    }

    #[test]
    fn test_empty_name() {
        let tokens = tokenize_bone_name("");

        for &t in &tokens {
            assert_eq!(t, PAD_TOKEN);
        }
    }

    #[test]
    fn test_non_ascii_becomes_unk() {
        let tokens = tokenize_bone_name("bone\u{00E9}");
        let non_pad: Vec<i64> = tokens.iter().copied().filter(|&t| t != PAD_TOKEN).collect();

        assert_eq!(*non_pad.last().unwrap(), UNK_TOKEN);
    }

    #[test]
    fn test_right_alignment() {
        let tokens = tokenize_bone_name("ab");

        for &t in &tokens[..BONE_NAME_TOKEN_LENGTH - 2] {
            assert_eq!(t, PAD_TOKEN);
        }
        assert_eq!(tokens[BONE_NAME_TOKEN_LENGTH - 2], 8);
        assert_eq!(tokens[BONE_NAME_TOKEN_LENGTH - 1], 9);
    }

    #[test]
    fn test_compute_bone_name_tokens() {
        let mut skeleton = Skeleton::new("test");
        skeleton.add_bone("root", None);
        skeleton.add_bone("child", Some(0));

        let tokens = compute_bone_name_tokens(&skeleton);
        assert_eq!(tokens.len(), 2);
        assert!(tokens.contains_key(&0));
        assert!(tokens.contains_key(&1));
    }
}
