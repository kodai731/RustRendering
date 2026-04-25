use std::collections::HashMap;

use crate::editable::components::curve::PropertyType;
use crate::editable::components::keyframe_copy::{CopiedKeyframe, KeyframeCopyBuffer};
use crate::editable::components::mirror::{MirrorAxis, MirrorMapping};
use crate::BoneId;

pub fn build_mirror_mapping(bone_names: &HashMap<BoneId, String>) -> MirrorMapping {
    let mut pairs = Vec::new();
    let mut matched = std::collections::HashSet::new();

    let name_to_id: HashMap<&str, BoneId> = bone_names
        .iter()
        .map(|(id, name)| (name.as_str(), *id))
        .collect();

    for (id, name) in bone_names {
        if matched.contains(id) {
            continue;
        }

        if let Some(mirror_name) = find_mirror_name(name) {
            if let Some(&mirror_id) = name_to_id.get(mirror_name.as_str()) {
                if mirror_id != *id {
                    pairs.push((*id, mirror_id));
                    matched.insert(*id);
                    matched.insert(mirror_id);
                }
            }
        }
    }

    MirrorMapping {
        pairs,
        symmetry_axis: MirrorAxis::X,
    }
}

fn find_mirror_name(name: &str) -> Option<String> {
    let patterns: &[(&str, &str)] = &[
        ("L_", "R_"),
        ("R_", "L_"),
        ("Left", "Right"),
        ("Right", "Left"),
        ("_l", "_r"),
        ("_r", "_l"),
        (".L", ".R"),
        (".R", ".L"),
        ("_L_", "_R_"),
        ("_R_", "_L_"),
    ];

    for (from, to) in patterns {
        if let Some(pos) = name.find(from) {
            let mut mirrored = String::with_capacity(name.len());
            mirrored.push_str(&name[..pos]);
            mirrored.push_str(to);
            mirrored.push_str(&name[pos + from.len()..]);
            return Some(mirrored);
        }
    }

    None
}

pub fn mirror_keyframes(
    buffer: &KeyframeCopyBuffer,
    mapping: &MirrorMapping,
) -> KeyframeCopyBuffer {
    let pair_map: HashMap<BoneId, BoneId> = mapping
        .pairs
        .iter()
        .flat_map(|(a, b)| [(*a, *b), (*b, *a)])
        .collect();

    let entries = buffer
        .entries
        .iter()
        .map(|entry| {
            let mirrored_bone_id = pair_map
                .get(&entry.bone_id)
                .copied()
                .unwrap_or(entry.bone_id);

            let mirrored_value =
                compute_mirrored_value(entry.value, entry.property_type, mapping.symmetry_axis);

            CopiedKeyframe {
                bone_id: mirrored_bone_id,
                property_type: entry.property_type,
                relative_time: entry.relative_time,
                value: mirrored_value,
                interpolation: entry.interpolation,
                in_tangent: entry.in_tangent.clone(),
                out_tangent: entry.out_tangent.clone(),
                weight_mode: entry.weight_mode,
            }
        })
        .collect();

    KeyframeCopyBuffer {
        entries,
        base_time: buffer.base_time,
        source_clip_id: buffer.source_clip_id,
    }
}

fn compute_mirrored_value(value: f32, property_type: PropertyType, axis: MirrorAxis) -> f32 {
    match axis {
        MirrorAxis::X => match property_type {
            PropertyType::TranslationX => -value,
            PropertyType::RotationY | PropertyType::RotationZ => -value,
            _ => value,
        },
        MirrorAxis::Y => match property_type {
            PropertyType::TranslationY => -value,
            PropertyType::RotationX | PropertyType::RotationZ => -value,
            _ => value,
        },
        MirrorAxis::Z => match property_type {
            PropertyType::TranslationZ => -value,
            PropertyType::RotationX | PropertyType::RotationY => -value,
            _ => value,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_mirror_mapping_l_r_prefix() {
        let mut bone_names = HashMap::new();
        bone_names.insert(1, "L_Arm".to_string());
        bone_names.insert(2, "R_Arm".to_string());
        bone_names.insert(3, "Spine".to_string());

        let mapping = build_mirror_mapping(&bone_names);
        assert_eq!(mapping.pairs.len(), 1);

        let (a, b) = mapping.pairs[0];
        assert!(
            (a == 1 && b == 2) || (a == 2 && b == 1),
            "Expected pair (1,2) or (2,1), got ({}, {})",
            a,
            b
        );
    }

    #[test]
    fn test_mirror_keyframes_translation_x_negated() {
        use crate::editable::{BezierHandle, InterpolationType, TangentWeightMode};

        let buffer = KeyframeCopyBuffer {
            entries: vec![CopiedKeyframe {
                bone_id: 1,
                property_type: PropertyType::TranslationX,
                relative_time: 0.0,
                value: 5.0,
                interpolation: InterpolationType::Linear,
                in_tangent: BezierHandle::linear(),
                out_tangent: BezierHandle::linear(),
                weight_mode: TangentWeightMode::NonWeighted,
            }],
            base_time: 0.0,
            source_clip_id: None,
        };

        let mapping = MirrorMapping {
            pairs: vec![(1, 2)],
            symmetry_axis: MirrorAxis::X,
        };

        let result = mirror_keyframes(&buffer, &mapping);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].bone_id, 2);
        assert!((result.entries[0].value - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn test_find_mirror_name_patterns() {
        assert_eq!(find_mirror_name("L_Arm"), Some("R_Arm".to_string()));
        assert_eq!(find_mirror_name("R_Leg"), Some("L_Leg".to_string()));
        assert_eq!(find_mirror_name("Hand.L"), Some("Hand.R".to_string()));
        assert_eq!(find_mirror_name("Spine"), None);
    }
}
