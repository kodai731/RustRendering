use std::collections::HashMap;

use crate::editable::components::clip::EditableAnimationClip;
use crate::editable::components::curve::PropertyType;
use crate::editable::components::track::BoneTrack;
use crate::editable::systems::curve_ops::curve_resample_at_fps;
use crate::BoneId;

const ALL_PROPERTY_TYPES: [PropertyType; 9] = [
    PropertyType::TranslationX,
    PropertyType::TranslationY,
    PropertyType::TranslationZ,
    PropertyType::RotationX,
    PropertyType::RotationY,
    PropertyType::RotationZ,
    PropertyType::ScaleX,
    PropertyType::ScaleY,
    PropertyType::ScaleZ,
];

#[derive(Clone, Copy, Debug, Default)]
pub struct ResampleSummary {
    pub resampled_curve_count: usize,
    pub total_keyframe_count: usize,
}

pub fn clip_resample_at_fps(clip: &mut EditableAnimationClip, fps: f32) -> ResampleSummary {
    let duration = clip.duration;
    let mut summary = ResampleSummary::default();

    for track in clip.tracks.values_mut() {
        for property_type in ALL_PROPERTY_TYPES {
            let curve = track.get_curve_mut(property_type);
            if curve.is_empty() {
                continue;
            }
            summary.total_keyframe_count += curve_resample_at_fps(curve, duration, fps);
            summary.resampled_curve_count += 1;
        }
    }

    summary
}

pub fn clip_recalculate_duration(clip: &mut EditableAnimationClip) {
    let mut max_time: f32 = 0.0;

    for track in clip.tracks.values() {
        for curve in track.all_curves() {
            if let Some(last_kf) = curve.keyframes.last() {
                max_time = max_time.max(last_kf.time);
            }
        }
    }
    for curve in &clip.scalar_curves {
        if let Some(last_kf) = curve.keyframes.last() {
            max_time = max_time.max(last_kf.time);
        }
    }

    clip.duration = max_time.max(clip.min_duration);
}

pub fn clip_remap_bone_ids(
    clip: &mut EditableAnimationClip,
    name_to_new_id: &HashMap<String, BoneId>,
) {
    let old_tracks: Vec<(BoneId, BoneTrack)> = clip.tracks.drain().collect();
    for (_, mut track) in old_tracks {
        let new_id = match name_to_new_id.get(&track.bone_name) {
            Some(&id) => id,
            None => continue,
        };
        track.bone_id = new_id;
        clip.tracks.insert(new_id, track);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editable::systems::curve_ops::{curve_add_keyframe, curve_sample};

    #[test]
    fn resample_rebakes_curve_onto_60fps_grid() {
        let mut clip = EditableAnimationClip::new(1, "test".to_string());
        let track = clip.add_track(0, "root".to_string());
        let curve = track.get_curve_mut(PropertyType::TranslationX);
        curve_add_keyframe(curve, 0.0, 0.0);
        curve_add_keyframe(curve, 1.0, 10.0);
        clip.duration = 1.0;

        let summary = clip_resample_at_fps(&mut clip, 60.0);

        assert_eq!(summary.resampled_curve_count, 1);
        let curve = clip
            .get_track(0)
            .unwrap()
            .get_curve(PropertyType::TranslationX);
        assert_eq!(curve.keyframes.len(), 61);
        assert!((curve.keyframes.first().unwrap().time - 0.0).abs() < 1e-6);
        assert!((curve.keyframes.last().unwrap().time - 1.0).abs() < 1e-4);
        assert!((curve_sample(curve, 0.5).unwrap() - 5.0).abs() < 0.2);
    }

    #[test]
    fn recalculate_duration_keeps_the_authored_floor_without_keys() {
        let mut clip = EditableAnimationClip::new(1, "unkeyed".to_string());
        clip.min_duration = 12.0;

        clip_recalculate_duration(&mut clip);
        assert!((clip.duration - 12.0).abs() < 1e-6);

        let curve = clip
            .add_track(0, "root".to_string())
            .get_curve_mut(PropertyType::TranslationX);
        curve_add_keyframe(curve, 20.0, 1.0);
        clip_recalculate_duration(&mut clip);
        assert!((clip.duration - 20.0).abs() < 1e-6);
    }

    #[test]
    fn resample_leaves_empty_curves_untouched() {
        let mut clip = EditableAnimationClip::new(1, "test".to_string());
        clip.add_track(0, "root".to_string());
        clip.duration = 1.0;

        let summary = clip_resample_at_fps(&mut clip, 60.0);

        assert_eq!(summary.resampled_curve_count, 0);
        assert!(clip
            .get_track(0)
            .unwrap()
            .get_curve(PropertyType::TranslationX)
            .is_empty());
    }
}
