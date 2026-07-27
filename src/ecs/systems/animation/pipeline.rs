use std::collections::{HashMap, HashSet};

use cgmath::Matrix4;

use crate::animation::{BoneId, BoneLocalPose, SkeletonId};
use crate::asset::AssetStorage;
use crate::ecs::resource::{AnimationType, ClipLibrary, PoseApplyCache};
use crate::ecs::world::{Animator, World};
use crate::ecs::{apply_pose_overrides, compute_pose_global_transforms};
use crate::vulkanr::resource::graphics_resource::{GraphicsResources, NodeData};

use super::apply::{
    apply_morph_animation, apply_node_animation_to_single_mesh, apply_skinning_to_single_mesh,
    build_node_based_bone_transforms, compute_node_global_transforms, merge_updated_indices,
};
use super::collect::collect_animated_entities;
use super::evaluate::evaluate_entity_blend;
use super::post_process::{compute_spring_bone_result, find_shared_constraints};
use super::{AnimatedEntityInfo, AnimationEvalResult};
use crate::ecs::systems::constraint_solve_systems::apply_constraints;

pub fn run_animation_pipeline(
    world: &World,
    graphics: &mut GraphicsResources,
    nodes: &mut [NodeData],
    clip_library: &ClipLibrary,
    assets: &AssetStorage,
    dt: f32,
    pose_overrides: &HashMap<BoneId, BoneLocalPose>,
    pose_apply_cache: &mut PoseApplyCache,
) -> AnimationEvalResult {
    let entity_infos = collect_animated_entities(world, graphics, clip_library, assets);

    let first_time = world
        .iter_components::<Animator>()
        .next()
        .map(|(_, a)| a.time)
        .unwrap_or(0.0);

    let morph_updated: HashSet<usize> = if !clip_library.morph_animation.is_empty() {
        let updated = apply_morph_animation(graphics, &clip_library.morph_animation, first_time);
        updated.into_iter().collect()
    } else {
        HashSet::new()
    };

    if entity_infos.is_empty() {
        return AnimationEvalResult {
            updated_meshes: morph_updated.into_iter().collect(),
            bone_transforms: None,
        };
    }

    let (anim_updated, bone_transforms) = apply_blended_animations(
        &entity_infos,
        world,
        graphics,
        nodes,
        assets,
        dt,
        pose_overrides,
        &morph_updated,
        pose_apply_cache,
    );

    let morph_vec: Vec<usize> = morph_updated.into_iter().collect();
    AnimationEvalResult {
        updated_meshes: merge_updated_indices(morph_vec, anim_updated),
        bone_transforms,
    }
}

fn apply_blended_animations(
    entities: &[AnimatedEntityInfo],
    world: &World,
    graphics: &mut GraphicsResources,
    nodes: &mut [NodeData],
    assets: &AssetStorage,
    dt: f32,
    pose_overrides: &HashMap<BoneId, BoneLocalPose>,
    morph_updated: &HashSet<usize>,
    pose_apply_cache: &mut PoseApplyCache,
) -> (
    Vec<usize>,
    Option<(SkeletonId, Vec<Matrix4<f32>>, AnimationType)>,
) {
    let mut updated = Vec::new();
    let mut first_bone_transforms: Option<(SkeletonId, Vec<Matrix4<f32>>, AnimationType)> = None;

    let shared_constraints = find_shared_constraints(entities, world);

    let spring_result = compute_spring_bone_result(
        entities,
        world,
        assets,
        &shared_constraints,
        pose_overrides,
        dt,
    );

    for info in entities {
        let Some(skeleton) = assets.get_skeleton_by_skeleton_id(info.skeleton_id) else {
            continue;
        };

        let has_spring = spring_result
            .as_ref()
            .map_or(false, |(skel_id, _, _)| *skel_id == info.skeleton_id);

        let (globals, _pose) = if has_spring {
            let (_, ref cached_globals, ref cached_pose) = spring_result
                .as_ref()
                .expect("has_spring is true so spring_result is Some");

            if info.animation_type == AnimationType::Node {
                compute_node_global_transforms(nodes, skeleton, cached_pose);
            }

            (cached_globals.clone(), None)
        } else {
            let Some(mut pose) = evaluate_entity_blend(info, assets) else {
                continue;
            };

            if let Some(ref cs) = shared_constraints {
                apply_constraints(cs, skeleton, &mut pose);
            }

            if !pose_overrides.is_empty() {
                apply_pose_overrides(&mut pose, pose_overrides);
            }

            if info.animation_type == AnimationType::Node {
                compute_node_global_transforms(nodes, skeleton, &pose);
            }

            let globals = compute_pose_global_transforms(skeleton, &pose);

            (globals, Some(pose))
        };

        if first_bone_transforms.is_none() {
            let gizmo_transforms = if info.animation_type == AnimationType::Node {
                build_node_based_bone_transforms(nodes, skeleton)
            } else {
                globals.clone()
            };
            first_bone_transforms = Some((
                info.skeleton_id,
                gizmo_transforms,
                info.animation_type.clone(),
            ));
        }

        let morph_targeted = morph_updated.contains(&info.mesh_idx);

        let mesh_updated = match info.animation_type {
            AnimationType::Node => {
                let mesh_ref = &graphics.meshes[info.mesh_idx];
                let node_opt = mesh_ref
                    .node_index
                    .and_then(|idx| nodes.iter().find(|n| n.index == idx));
                if let Some(node) = node_opt {
                    let current_value = (node.global_transform, info.node_animation_scale);
                    if should_skip_node(
                        pose_apply_cache,
                        info.mesh_idx,
                        current_value,
                        morph_targeted,
                    ) {
                        continue;
                    }
                    pose_apply_cache
                        .node_cache
                        .insert(info.mesh_idx, current_value);
                }
                apply_node_animation_to_single_mesh(
                    graphics,
                    info.mesh_idx,
                    nodes,
                    info.node_animation_scale,
                )
            }
            _ => {
                if should_skip_skinned(pose_apply_cache, info.mesh_idx, &globals, morph_targeted) {
                    continue;
                }
                pose_apply_cache
                    .skinned_cache
                    .insert(info.mesh_idx, globals.clone());
                apply_skinning_to_single_mesh(graphics, info.mesh_idx, &globals, skeleton)
            }
        };

        if mesh_updated && !updated.contains(&info.mesh_idx) {
            updated.push(info.mesh_idx);
        }
    }

    (updated, first_bone_transforms)
}

#[inline]
fn should_skip_skinned(
    cache: &PoseApplyCache,
    mesh_idx: usize,
    globals: &[Matrix4<f32>],
    morph_targeted: bool,
) -> bool {
    if morph_targeted {
        return false;
    }
    if let Some(cached) = cache.skinned_cache.get(&mesh_idx) {
        if cached == &globals {
            return true;
        }
    }
    false
}

#[inline]
fn should_skip_node(
    cache: &PoseApplyCache,
    mesh_idx: usize,
    current_value: (Matrix4<f32>, f32),
    morph_targeted: bool,
) -> bool {
    if morph_targeted {
        return false;
    }
    if let Some(cached) = cache.node_cache.get(&mesh_idx) {
        if cached == &current_value {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::resource::PoseApplyCache;
    use cgmath::SquareMatrix;

    fn identity_matrices(count: usize) -> Vec<Matrix4<f32>> {
        (0..count).map(|_| Matrix4::identity()).collect()
    }

    fn translated_matrices(offset: f32, count: usize) -> Vec<Matrix4<f32>> {
        (0..count)
            .map(|i| {
                Matrix4::from_translation(cgmath::Vector3::new(offset * (i as f32 + 1.0), 0.0, 0.0))
            })
            .collect()
    }

    #[test]
    fn test_should_skip_skinned_same_globals_no_morph() {
        let mut cache = PoseApplyCache::default();
        let globals: Vec<Matrix4<f32>> = identity_matrices(4);
        cache.skinned_cache.insert(0, globals.clone());

        assert!(should_skip_skinned(&cache, 0, &globals, false));
    }

    #[test]
    fn test_should_skip_skinned_different_globals_no_morph() {
        let mut cache = PoseApplyCache::default();
        let cached_globals: Vec<Matrix4<f32>> = identity_matrices(4);
        let new_globals: Vec<Matrix4<f32>> = translated_matrices(1.0, 4);
        cache.skinned_cache.insert(0, cached_globals);

        assert!(!should_skip_skinned(&cache, 0, &new_globals, false));
    }

    #[test]
    fn test_should_skip_skinned_any_globals_morph_true() {
        let mut cache = PoseApplyCache::default();
        let globals: Vec<Matrix4<f32>> = identity_matrices(4);
        cache.skinned_cache.insert(0, globals.clone());

        assert!(!should_skip_skinned(&cache, 0, &globals, true));
    }

    #[test]
    fn test_should_skip_node_same_value_no_morph() {
        let mut cache = PoseApplyCache::default();
        let value: (Matrix4<f32>, f32) = (Matrix4::identity(), 1.0);
        cache.node_cache.insert(0, value);

        assert!(should_skip_node(&cache, 0, value, false));
    }

    #[test]
    fn test_should_skip_node_different_value_no_morph() {
        let mut cache = PoseApplyCache::default();
        let cached_value: (Matrix4<f32>, f32) = (Matrix4::identity(), 1.0);
        let new_value: (Matrix4<f32>, f32) = (
            Matrix4::from_translation(cgmath::Vector3::new(1.0, 0.0, 0.0)),
            1.0,
        );
        cache.node_cache.insert(0, cached_value);

        assert!(!should_skip_node(&cache, 0, new_value, false));
    }

    #[test]
    fn test_should_skip_node_any_value_morph_true() {
        let mut cache = PoseApplyCache::default();
        let value: (Matrix4<f32>, f32) = (Matrix4::identity(), 1.0);
        cache.node_cache.insert(0, value);

        assert!(!should_skip_node(&cache, 0, value, true));
    }
}
