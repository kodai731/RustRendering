use std::collections::HashMap;

use crate::animation::BoneId;

const MIN_SKELETON_HEIGHT: f32 = 1.0;

pub struct BoneRestPositionCache {
    pub positions: HashMap<BoneId, [f32; 3]>,
    pub skeleton_height: f32,
}

impl Default for BoneRestPositionCache {
    fn default() -> Self {
        Self {
            positions: HashMap::new(),
            skeleton_height: MIN_SKELETON_HEIGHT,
        }
    }
}

impl BoneRestPositionCache {
    pub fn clear(&mut self) {
        self.positions.clear();
        self.skeleton_height = MIN_SKELETON_HEIGHT;
    }

    pub fn get_normalized(&self, bone_id: BoneId) -> [f32; 3] {
        let divisor = self.skeleton_height.max(MIN_SKELETON_HEIGHT);
        match self.positions.get(&bone_id) {
            Some([x, y, z]) => [x / divisor, y / divisor, z / divisor],
            None => [0.0, 0.0, 0.0],
        }
    }
}

pub fn build_rest_position_cache(
    skeleton: &thyllore_model_core::Skeleton,
) -> BoneRestPositionCache {
    let mut cache = BoneRestPositionCache::default();
    for bone in &skeleton.bones {
        let translation = bone.local_transform.w;
        cache
            .positions
            .insert(bone.id, [translation.x, translation.y, translation.z]);
    }
    cache.skeleton_height = estimate_skeleton_height(skeleton);
    cache
}

fn estimate_skeleton_height(skeleton: &thyllore_model_core::Skeleton) -> f32 {
    let mut max_world_y: f32 = 0.0;
    for root_id in &skeleton.root_bone_ids {
        accumulate_max_y(skeleton, *root_id, 0.0, &mut max_world_y);
    }
    max_world_y.max(MIN_SKELETON_HEIGHT)
}

fn accumulate_max_y(
    skeleton: &thyllore_model_core::Skeleton,
    bone_id: BoneId,
    parent_y: f32,
    max_y: &mut f32,
) {
    let Some(bone) = skeleton.get_bone(bone_id) else {
        return;
    };
    let world_y = parent_y + bone.local_transform.w.y;
    *max_y = max_y.max(world_y.abs());
    for child_id in &bone.children {
        accumulate_max_y(skeleton, *child_id, world_y, max_y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Matrix4, Vector3};
    use thyllore_model_core::Skeleton;

    fn skeleton_with_translation(bones: &[(&str, Option<BoneId>, [f32; 3])]) -> Skeleton {
        let mut skeleton = Skeleton::new("test");
        for (name, parent_id, translation) in bones {
            let id = skeleton.add_bone(name, *parent_id);
            if let Some(bone) = skeleton.get_bone_mut(id) {
                bone.local_transform = Matrix4::from_translation(Vector3::new(
                    translation[0],
                    translation[1],
                    translation[2],
                ));
            }
        }
        skeleton
    }

    #[test]
    fn rest_position_is_normalized_by_skeleton_height() {
        let skeleton = skeleton_with_translation(&[
            ("root", None, [0.0, 4.0, 0.0]),
            ("child", Some(0), [0.0, 2.0, 0.0]),
        ]);
        let cache = build_rest_position_cache(&skeleton);

        assert!((cache.skeleton_height - 6.0).abs() < 1e-6);
        let normalized_child = cache.get_normalized(1);
        assert!((normalized_child[1] - (2.0 / 6.0)).abs() < 1e-6);
    }

    #[test]
    fn missing_bone_returns_zero() {
        let cache = BoneRestPositionCache::default();
        assert_eq!(cache.get_normalized(99), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn empty_skeleton_height_is_clamped_to_one() {
        let skeleton = Skeleton::new("empty");
        let cache = build_rest_position_cache(&skeleton);
        assert!((cache.skeleton_height - MIN_SKELETON_HEIGHT).abs() < 1e-6);
    }

    #[test]
    fn clear_resets_state() {
        let mut cache = BoneRestPositionCache {
            positions: HashMap::from([(0, [1.0, 2.0, 3.0])]),
            skeleton_height: 10.0,
        };
        cache.clear();
        assert!(cache.positions.is_empty());
        assert!((cache.skeleton_height - MIN_SKELETON_HEIGHT).abs() < 1e-6);
    }
}
