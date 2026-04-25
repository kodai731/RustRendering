use std::collections::HashMap;

use thyllore_anim_core::editable::{
    curve_add_keyframe_with_tangents, BezierHandle, EditableAnimationClip, InterpolationType,
    PropertyType,
};
use thyllore_anim_core::BoneId;

use crate::request::{RawAnimationCurve, RawCurveKeyframe};

pub fn convert_motion_response_to_clip(
    curves: &[RawAnimationCurve],
    clip_name: &str,
    duration: f32,
    bone_name_to_id: &HashMap<String, BoneId>,
) -> EditableAnimationClip {
    let mut clip = EditableAnimationClip::new(0, clip_name.to_string());
    clip.duration = duration;

    for raw_curve in curves {
        let bone_id = match bone_name_to_id.get(&raw_curve.bone_name) {
            Some(&id) => id,
            None => {
                log!(
                    "TextToMotion: unknown bone_name '{}', skipping",
                    raw_curve.bone_name
                );
                continue;
            }
        };

        let property_type = match convert_proto_property_type(raw_curve.property_type) {
            Some(pt) => pt,
            None => {
                log!(
                    "TextToMotion: unknown property_type {}, skipping",
                    raw_curve.property_type
                );
                continue;
            }
        };

        if !clip.tracks.contains_key(&bone_id) {
            clip.add_track(bone_id, raw_curve.bone_name.clone());
        }

        let track = clip
            .tracks
            .get_mut(&bone_id)
            .expect("track was just inserted by add_track above");
        let curve = track.get_curve_mut(property_type);

        for kf in &raw_curve.keyframes {
            let (in_tangent, out_tangent, interpolation) = convert_keyframe_tangents(kf);

            curve_add_keyframe_with_tangents(
                curve,
                kf.time,
                kf.value,
                in_tangent,
                out_tangent,
                interpolation,
            );
        }
    }

    clip
}

fn convert_proto_property_type(proto_value: i32) -> Option<PropertyType> {
    match proto_value {
        0 => Some(PropertyType::TranslationX),
        1 => Some(PropertyType::TranslationY),
        2 => Some(PropertyType::TranslationZ),
        3 => Some(PropertyType::RotationX),
        4 => Some(PropertyType::RotationY),
        5 => Some(PropertyType::RotationZ),
        _ => None,
    }
}

fn convert_keyframe_tangents(
    kf: &RawCurveKeyframe,
) -> (BezierHandle, BezierHandle, InterpolationType) {
    let interpolation = if kf.interpolation == 1 {
        InterpolationType::Bezier
    } else {
        InterpolationType::Linear
    };

    let in_tangent = BezierHandle::new(kf.tangent_in_dt, kf.tangent_in_dv);
    let out_tangent = BezierHandle::new(kf.tangent_out_dt, kf.tangent_out_dv);

    (in_tangent, out_tangent, interpolation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bone_map() -> HashMap<String, BoneId> {
        let mut map = HashMap::new();
        map.insert("hips".to_string(), 0);
        map.insert("spine".to_string(), 3);
        map.insert("leftUpperLeg".to_string(), 1);
        map
    }

    fn create_test_keyframe(time: f32, value: f32, interpolation: i32) -> RawCurveKeyframe {
        RawCurveKeyframe {
            time,
            value,
            tangent_in_dt: -0.1,
            tangent_in_dv: 0.0,
            tangent_out_dt: 0.1,
            tangent_out_dv: 0.0,
            interpolation,
        }
    }

    #[test]
    fn test_convert_empty_curves() {
        let bone_map = create_test_bone_map();
        let clip = convert_motion_response_to_clip(&[], "test", 3.0, &bone_map);
        assert!(clip.tracks.is_empty());
        assert_eq!(clip.duration, 3.0);
    }

    #[test]
    fn test_convert_single_rotation_curve() {
        let bone_map = create_test_bone_map();

        let curves = vec![RawAnimationCurve {
            bone_name: "hips".to_string(),
            property_type: 3,
            keyframes: vec![
                create_test_keyframe(0.0, 0.0, 0),
                create_test_keyframe(1.0, 45.0, 1),
                create_test_keyframe(2.0, 0.0, 0),
            ],
        }];

        let clip = convert_motion_response_to_clip(&curves, "test_motion", 2.0, &bone_map);

        assert_eq!(clip.tracks.len(), 1);
        let track = clip.tracks.get(&0).unwrap();
        assert_eq!(track.rotation_x.keyframe_count(), 3);
    }

    #[test]
    fn test_convert_unknown_bone_skipped() {
        let bone_map = create_test_bone_map();

        let curves = vec![RawAnimationCurve {
            bone_name: "unknownBone".to_string(),
            property_type: 3,
            keyframes: vec![
                create_test_keyframe(0.0, 0.0, 0),
                create_test_keyframe(1.0, 45.0, 0),
            ],
        }];

        let clip = convert_motion_response_to_clip(&curves, "test", 2.0, &bone_map);
        assert!(clip.tracks.is_empty());
    }

    #[test]
    fn test_convert_multiple_bones_multiple_properties() {
        let bone_map = create_test_bone_map();

        let curves = vec![
            RawAnimationCurve {
                bone_name: "hips".to_string(),
                property_type: 0,
                keyframes: vec![
                    create_test_keyframe(0.0, 0.0, 0),
                    create_test_keyframe(1.0, 0.5, 0),
                ],
            },
            RawAnimationCurve {
                bone_name: "hips".to_string(),
                property_type: 3,
                keyframes: vec![
                    create_test_keyframe(0.0, 0.0, 1),
                    create_test_keyframe(1.0, 30.0, 1),
                ],
            },
            RawAnimationCurve {
                bone_name: "spine".to_string(),
                property_type: 4,
                keyframes: vec![
                    create_test_keyframe(0.0, 0.0, 0),
                    create_test_keyframe(1.0, 15.0, 0),
                ],
            },
        ];

        let clip = convert_motion_response_to_clip(&curves, "multi_bone", 1.0, &bone_map);

        assert_eq!(clip.tracks.len(), 2);

        let hips = clip.tracks.get(&0).unwrap();
        assert_eq!(hips.translation_x.keyframe_count(), 2);
        assert_eq!(hips.rotation_x.keyframe_count(), 2);

        let spine = clip.tracks.get(&3).unwrap();
        assert_eq!(spine.rotation_y.keyframe_count(), 2);
    }

    #[test]
    fn test_convert_bezier_interpolation() {
        let bone_map = create_test_bone_map();

        let curves = vec![RawAnimationCurve {
            bone_name: "hips".to_string(),
            property_type: 3,
            keyframes: vec![RawCurveKeyframe {
                time: 0.5,
                value: 90.0,
                tangent_in_dt: -0.2,
                tangent_in_dv: -10.0,
                tangent_out_dt: 0.3,
                tangent_out_dv: 5.0,
                interpolation: 1,
            }],
        }];

        let clip = convert_motion_response_to_clip(&curves, "bezier_test", 1.0, &bone_map);

        let track = clip.tracks.get(&0).unwrap();
        let kf = &track.rotation_x.keyframes[0];

        assert_eq!(kf.interpolation, InterpolationType::Bezier);
        assert!((kf.in_tangent.time_offset - (-0.2)).abs() < 0.001);
        assert!((kf.in_tangent.value_offset - (-10.0)).abs() < 0.001);
        assert!((kf.out_tangent.time_offset - 0.3).abs() < 0.001);
        assert!((kf.out_tangent.value_offset - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_convert_proto_property_type_mapping() {
        assert_eq!(
            convert_proto_property_type(0),
            Some(PropertyType::TranslationX)
        );
        assert_eq!(
            convert_proto_property_type(5),
            Some(PropertyType::RotationZ)
        );
        assert_eq!(convert_proto_property_type(6), None);
        assert_eq!(convert_proto_property_type(-1), None);
    }
}
