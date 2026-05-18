use crate::animation::editable::{EditableAnimationClip, PropertyType};
use crate::animation::BoneId;
use crate::ecs::resource::{BoneRestPositionCache, BoneTopologyCache};
use thyllore_ml_core::copilot::input::CONTEXT_KEYFRAME_COUNT as BONE_CONTEXT_KEYFRAME_COUNT;
use thyllore_model_core::Skeleton;

pub const BONE_CONTEXT_N_MAX: usize = 32;
pub const BONE_CONTEXT_FEATURE_DIM: usize = 6;
pub const BONE_CONTEXT_TOPOLOGY_DIM: usize = 6;
pub const BONE_CONTEXT_REST_POSITION_DIM: usize = 3;

const MIN_CURVE_STD: f32 = 0.01;
const ANCHOR_EPSILON: f32 = 1.0e-6;

pub struct BoneContextArrays {
    pub keyframes: Vec<f32>,
    pub topology: Vec<f32>,
    pub rest_positions: Vec<f32>,
    pub mask: Vec<bool>,
}

impl BoneContextArrays {
    fn empty() -> Self {
        Self {
            keyframes: vec![
                0.0;
                BONE_CONTEXT_N_MAX
                    * BONE_CONTEXT_KEYFRAME_COUNT
                    * BONE_CONTEXT_FEATURE_DIM
            ],
            topology: vec![0.0; BONE_CONTEXT_N_MAX * BONE_CONTEXT_TOPOLOGY_DIM],
            rest_positions: vec![0.0; BONE_CONTEXT_N_MAX * BONE_CONTEXT_REST_POSITION_DIM],
            mask: vec![false; BONE_CONTEXT_N_MAX],
        }
    }
}

pub struct BoneContextRequest<'a> {
    pub clip: &'a EditableAnimationClip,
    pub skeleton: &'a Skeleton,
    pub topology_cache: &'a BoneTopologyCache,
    pub rest_position_cache: &'a BoneRestPositionCache,
    pub property_type: PropertyType,
    pub anchor_time: f32,
    pub clip_duration: f32,
}

pub fn build_bone_context_arrays(request: BoneContextRequest<'_>) -> BoneContextArrays {
    let mut arrays = BoneContextArrays::empty();
    let bone_limit = request.skeleton.bones.len().min(BONE_CONTEXT_N_MAX);

    for slot in 0..bone_limit {
        let bone_id = request.skeleton.bones[slot].id;
        fill_topology_and_rest(&mut arrays, slot, bone_id, &request);
        let kf_filled = fill_keyframes_for_bone(&mut arrays, slot, bone_id, &request);
        arrays.mask[slot] = kf_filled;
    }

    arrays
}

fn fill_topology_and_rest(
    arrays: &mut BoneContextArrays,
    slot: usize,
    bone_id: BoneId,
    request: &BoneContextRequest<'_>,
) {
    let topo_features = request.topology_cache.get(bone_id).to_vec();
    let topo_offset = slot * BONE_CONTEXT_TOPOLOGY_DIM;
    for (i, value) in topo_features.iter().enumerate() {
        arrays.topology[topo_offset + i] = *value;
    }

    let rest = request.rest_position_cache.get_normalized(bone_id);
    let rest_offset = slot * BONE_CONTEXT_REST_POSITION_DIM;
    arrays.rest_positions[rest_offset] = rest[0];
    arrays.rest_positions[rest_offset + 1] = rest[1];
    arrays.rest_positions[rest_offset + 2] = rest[2];
}

fn fill_keyframes_for_bone(
    arrays: &mut BoneContextArrays,
    slot: usize,
    bone_id: BoneId,
    request: &BoneContextRequest<'_>,
) -> bool {
    let Some(track) = request.clip.get_track(bone_id) else {
        return false;
    };
    let curve = track.get_curve(request.property_type);
    let context_kfs = collect_context_before_anchor(curve, request.anchor_time);
    if context_kfs.is_empty() {
        return false;
    }

    let (mean, std) = compute_value_mean_std(&context_kfs);
    if std < MIN_CURVE_STD {
        return false;
    }

    write_keyframe_block(arrays, slot, &context_kfs, mean, std, request.clip_duration);
    true
}

fn collect_context_before_anchor<'a>(
    curve: &'a crate::animation::editable::PropertyCurve,
    anchor_time: f32,
) -> Vec<&'a crate::animation::editable::EditableKeyframe> {
    let mut sorted: Vec<&crate::animation::editable::EditableKeyframe> = curve
        .keyframes
        .iter()
        .filter(|kf| kf.time <= anchor_time + ANCHOR_EPSILON)
        .collect();
    sorted.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let total = sorted.len();
    if total > BONE_CONTEXT_KEYFRAME_COUNT {
        sorted = sorted.split_off(total - BONE_CONTEXT_KEYFRAME_COUNT);
    }
    sorted
}

fn compute_value_mean_std(
    context_kfs: &[&crate::animation::editable::EditableKeyframe],
) -> (f32, f32) {
    let count = context_kfs.len() as f32;
    let mean = context_kfs.iter().map(|kf| kf.value).sum::<f32>() / count;
    let variance = context_kfs
        .iter()
        .map(|kf| (kf.value - mean).powi(2))
        .sum::<f32>()
        / count;
    (mean, variance.sqrt())
}

fn write_keyframe_block(
    arrays: &mut BoneContextArrays,
    slot: usize,
    context_kfs: &[&crate::animation::editable::EditableKeyframe],
    mean: f32,
    std: f32,
    clip_duration: f32,
) {
    let duration = clip_duration.max(0.001);
    let block_size = BONE_CONTEXT_KEYFRAME_COUNT * BONE_CONTEXT_FEATURE_DIM;
    let block_start = slot * block_size;
    let padding_kfs = BONE_CONTEXT_KEYFRAME_COUNT - context_kfs.len();
    let padding_offset = padding_kfs * BONE_CONTEXT_FEATURE_DIM;

    for (i, kf) in context_kfs.iter().enumerate() {
        let dst = block_start + padding_offset + i * BONE_CONTEXT_FEATURE_DIM;
        arrays.keyframes[dst] = kf.time / duration;
        arrays.keyframes[dst + 1] = (kf.value - mean) / std;
        arrays.keyframes[dst + 2] = kf.in_tangent.time_offset / duration;
        arrays.keyframes[dst + 3] = kf.in_tangent.value_offset / std;
        arrays.keyframes[dst + 4] = kf.out_tangent.time_offset / duration;
        arrays.keyframes[dst + 5] = kf.out_tangent.value_offset / std;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::editable::{curve_add_keyframe, EditableAnimationClip};
    use thyllore_model_core::Skeleton;

    fn build_skeleton_with_bones(count: usize) -> Skeleton {
        let mut skeleton = Skeleton::new("test");
        for i in 0..count {
            let parent = if i == 0 { None } else { Some(0) };
            skeleton.add_bone(&format!("b{}", i), parent);
        }
        skeleton
    }

    fn make_clip_with_curve(
        bone_id: BoneId,
        property_type: PropertyType,
        keyframes: &[(f32, f32)],
    ) -> EditableAnimationClip {
        let mut clip = EditableAnimationClip::new(1, "test".to_string());
        clip.duration = 4.0;
        let track = clip.add_track(bone_id, format!("bone_{}", bone_id));
        let curve = track.get_curve_mut(property_type);
        for (t, v) in keyframes {
            curve_add_keyframe(curve, *t, *v);
        }
        clip
    }

    fn empty_caches() -> (BoneTopologyCache, BoneRestPositionCache) {
        (
            BoneTopologyCache::default(),
            BoneRestPositionCache::default(),
        )
    }

    #[test]
    fn arrays_have_fixed_padded_shape() {
        let skeleton = build_skeleton_with_bones(2);
        let clip = make_clip_with_curve(0, PropertyType::TranslationX, &[(0.0, 0.0), (1.0, 1.0)]);
        let (topo, rest) = empty_caches();

        let arrays = build_bone_context_arrays(BoneContextRequest {
            clip: &clip,
            skeleton: &skeleton,
            topology_cache: &topo,
            rest_position_cache: &rest,
            property_type: PropertyType::TranslationX,
            anchor_time: 2.0,
            clip_duration: 4.0,
        });

        assert_eq!(arrays.mask.len(), BONE_CONTEXT_N_MAX);
        assert_eq!(
            arrays.keyframes.len(),
            BONE_CONTEXT_N_MAX * BONE_CONTEXT_KEYFRAME_COUNT * BONE_CONTEXT_FEATURE_DIM
        );
        assert_eq!(
            arrays.topology.len(),
            BONE_CONTEXT_N_MAX * BONE_CONTEXT_TOPOLOGY_DIM
        );
        assert_eq!(
            arrays.rest_positions.len(),
            BONE_CONTEXT_N_MAX * BONE_CONTEXT_REST_POSITION_DIM
        );
    }

    #[test]
    fn missing_track_yields_mask_false_and_zero_block() {
        let skeleton = build_skeleton_with_bones(2);
        let clip = make_clip_with_curve(0, PropertyType::TranslationX, &[(0.0, 0.0), (1.0, 1.0)]);
        let (topo, rest) = empty_caches();

        let arrays = build_bone_context_arrays(BoneContextRequest {
            clip: &clip,
            skeleton: &skeleton,
            topology_cache: &topo,
            rest_position_cache: &rest,
            property_type: PropertyType::TranslationX,
            anchor_time: 2.0,
            clip_duration: 4.0,
        });

        assert!(arrays.mask[0]);
        assert!(!arrays.mask[1]);
        let block_size = BONE_CONTEXT_KEYFRAME_COUNT * BONE_CONTEXT_FEATURE_DIM;
        let bone1_block = &arrays.keyframes[block_size..2 * block_size];
        assert!(bone1_block.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn keyframes_after_anchor_are_excluded() {
        let skeleton = build_skeleton_with_bones(1);
        let clip = make_clip_with_curve(
            0,
            PropertyType::TranslationX,
            &[(0.5, 0.0), (1.0, 1.0), (1.5, 2.0), (3.0, 3.0)],
        );
        let (topo, rest) = empty_caches();

        let arrays = build_bone_context_arrays(BoneContextRequest {
            clip: &clip,
            skeleton: &skeleton,
            topology_cache: &topo,
            rest_position_cache: &rest,
            property_type: PropertyType::TranslationX,
            anchor_time: 1.4,
            clip_duration: 4.0,
        });

        assert!(arrays.mask[0]);
        let block_size = BONE_CONTEXT_KEYFRAME_COUNT * BONE_CONTEXT_FEATURE_DIM;
        let bone0 = &arrays.keyframes[..block_size];
        let last_kf_time_slot = block_size - BONE_CONTEXT_FEATURE_DIM;
        let last_kf_time_normalized = bone0[last_kf_time_slot];
        assert!(last_kf_time_normalized <= 1.0 / 4.0 + 1e-6);
    }

    #[test]
    fn skeleton_with_more_than_n_max_bones_truncates() {
        let skeleton = build_skeleton_with_bones(BONE_CONTEXT_N_MAX + 5);
        let clip = EditableAnimationClip::new(1, "empty".to_string());
        let (topo, rest) = empty_caches();

        let arrays = build_bone_context_arrays(BoneContextRequest {
            clip: &clip,
            skeleton: &skeleton,
            topology_cache: &topo,
            rest_position_cache: &rest,
            property_type: PropertyType::TranslationX,
            anchor_time: 0.0,
            clip_duration: 1.0,
        });

        assert_eq!(arrays.mask.len(), BONE_CONTEXT_N_MAX);
        let total_topology_slots = arrays.topology.len() / BONE_CONTEXT_TOPOLOGY_DIM;
        assert_eq!(total_topology_slots, BONE_CONTEXT_N_MAX);
    }
}
