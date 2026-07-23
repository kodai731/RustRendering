#[derive(Clone, Debug)]
pub struct FlameBoneAttachment {
    pub bone: String,
}

/// Resolve a bone specification to an index into the skeleton's bone name list.
///
/// Priority: empty → None / index parse / exact match / substring fallback.
pub fn resolve_bone_index(bone_spec: &str, bone_names: &[String]) -> Option<usize> {
    if bone_spec.is_empty() {
        return None;
    }
    if let Ok(idx) = bone_spec.parse::<usize>() {
        if idx < bone_names.len() {
            return Some(idx);
        }
    }
    if let Some(idx) = bone_names.iter().position(|name| name == bone_spec) {
        return Some(idx);
    }
    bone_names.iter().position(|name| name.contains(bone_spec))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_by_index() {
        let names = vec!["Hips".to_string(), "Spine".to_string()];
        assert_eq!(resolve_bone_index("1", &names), Some(1));
    }

    #[test]
    fn test_exact_match_wins_over_substring() {
        let names = vec!["LeftHand".to_string(), "Hand".to_string()];
        assert_eq!(resolve_bone_index("Hand", &names), Some(1));
    }

    #[test]
    fn test_substring_fallback() {
        let names = vec!["Hips".to_string(), "mixamorig:LeftHand".to_string()];
        assert_eq!(resolve_bone_index("LeftHand", &names), Some(1));
    }

    #[test]
    fn test_no_match_and_empty_return_none() {
        let names = vec!["Hips".to_string()];
        assert_eq!(resolve_bone_index("zz", &names), None);
        assert_eq!(resolve_bone_index("", &names), None);
    }
}
