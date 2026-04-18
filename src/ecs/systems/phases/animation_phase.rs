use std::collections::HashMap;

use anyhow::Result;

use crate::animation::{BoneId, BoneLocalPose};
use crate::app::FrameContext;
use crate::ecs::resource::gizmo::{BoneGizmoData, BoneSelectionState};
use crate::ecs::resource::{BonePoseOverride, ClipLibrary, NodeAssets, WeightHeatmapState};
use crate::ecs::{
    playback_upload_animations, run_animation_pipeline, transform_propagation_system,
    update_weight_heatmap,
};

pub struct AnimationUpdates {
    pub updated_meshes: Vec<usize>,
}

pub fn run_animation_phase_ecs(ctx: &mut FrameContext) -> AnimationUpdates {
    let pose_overrides: HashMap<BoneId, BoneLocalPose> = ctx
        .world
        .get_resource::<BonePoseOverride>()
        .map(|r| r.overrides.clone())
        .unwrap_or_default();

    let eval_result = {
        let clip_library = ctx.world.resource::<ClipLibrary>();
        let mut node_assets = ctx.world.resource_mut::<NodeAssets>();

        run_animation_pipeline(
            ctx.world,
            ctx.graphics,
            &mut node_assets.nodes,
            &*clip_library,
            ctx.assets,
            ctx.delta_time,
            &pose_overrides,
        )
    };

    transform_propagation_system(ctx.world);

    if let Some((skel_id, transforms, anim_type)) = &eval_result.bone_transforms {
        if ctx.world.contains_resource::<BoneGizmoData>() {
            let entity_transform = find_skin_entity_transform(ctx.world);
            let final_transforms = apply_entity_transform(transforms, &entity_transform);

            log!(
                "BoneGizmo: type={:?}, bones={}, head_pos=[{:.3},{:.3},{:.3}]",
                anim_type,
                final_transforms.len(),
                final_transforms.first().map_or(0.0, |t| t[3][0]),
                final_transforms.first().map_or(0.0, |t| t[3][1]),
                final_transforms.first().map_or(0.0, |t| t[3][2]),
            );

            let mut bone_gizmo = ctx.world.resource_mut::<BoneGizmoData>();
            bone_gizmo.cached_skeleton_id = Some(*skel_id);
            bone_gizmo.cached_animation_type = anim_type.clone();
            bone_gizmo.cached_global_transforms = final_transforms;
        }
    }

    let heatmap_updated_meshes = apply_weight_heatmap_update(ctx);

    let mut updated_meshes = eval_result.updated_meshes;
    for mesh_index in heatmap_updated_meshes {
        if !updated_meshes.contains(&mesh_index) {
            updated_meshes.push(mesh_index);
        }
    }

    AnimationUpdates { updated_meshes }
}

fn apply_weight_heatmap_update(ctx: &mut FrameContext) -> Vec<usize> {
    let selection = match ctx.world.get_resource::<BoneSelectionState>() {
        Some(state) => state.clone(),
        None => return Vec::new(),
    };

    let Some(mut heatmap) = ctx.world.get_resource_mut::<WeightHeatmapState>() else {
        return Vec::new();
    };

    update_weight_heatmap(ctx.graphics, &mut heatmap, &selection)
}

fn find_skin_entity_transform(world: &crate::ecs::World) -> cgmath::Matrix4<f32> {
    use crate::ecs::world::{Animator, GlobalTransform};
    use cgmath::SquareMatrix;

    world
        .iter_components::<Animator>()
        .next()
        .and_then(|(entity, _)| {
            world
                .get_component::<GlobalTransform>(entity)
                .map(|gt| gt.0)
        })
        .unwrap_or_else(cgmath::Matrix4::identity)
}

fn apply_entity_transform(
    bone_transforms: &[cgmath::Matrix4<f32>],
    entity_transform: &cgmath::Matrix4<f32>,
) -> Vec<cgmath::Matrix4<f32>> {
    bone_transforms
        .iter()
        .map(|bt| entity_transform * bt)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::world::{Animator, GlobalTransform, Transform};
    use cgmath::{Matrix4, SquareMatrix, Vector3};

    fn create_world_with_animated_entity(translation: Vector3<f32>) -> crate::ecs::World {
        let mut world = crate::ecs::World::new();

        let mut transform = Transform::default();
        transform.translation = translation;
        let global_matrix = Matrix4::from_translation(translation);

        let parent = world
            .entity()
            .with_name("test_model")
            .with_transform(transform)
            .with_visible(true)
            .with_animator(Animator::new())
            .build();
        world.insert_component(parent, GlobalTransform(global_matrix));

        world
            .entity()
            .with_name("test_mesh")
            .with_global_transform()
            .with_visible(true)
            .with_parent(parent)
            .with_mesh(1, 0)
            .build();

        world
    }

    #[test]
    fn find_skin_entity_transform_returns_entity_transform_for_animated_mesh() {
        let offset = Vector3::new(5.0, 3.0, -2.0);
        let world = create_world_with_animated_entity(offset);

        let result = find_skin_entity_transform(&world);

        let expected = Matrix4::from_translation(offset);
        assert_ne!(
            result,
            Matrix4::identity(),
            "BUG: find_skin_entity_transform returns identity even though animated mesh entity exists with non-identity GlobalTransform"
        );
        assert_eq!(
            result, expected,
            "find_skin_entity_transform should return the animated mesh entity's GlobalTransform"
        );
    }

    #[test]
    fn apply_entity_transform_includes_entity_offset() {
        let offset = Vector3::new(10.0, 0.0, 0.0);
        let world = create_world_with_animated_entity(offset);

        let bone_transforms = vec![
            Matrix4::identity(),
            Matrix4::from_translation(Vector3::new(0.0, 1.0, 0.0)),
        ];

        let entity_transform = find_skin_entity_transform(&world);
        let result = apply_entity_transform(&bone_transforms, &entity_transform);

        let expected_bone0 = Matrix4::from_translation(offset);
        let expected_bone1 = Matrix4::from_translation(Vector3::new(10.0, 1.0, 0.0));

        assert_ne!(
            result[0],
            Matrix4::identity(),
            "BUG: bone[0] at origin should be offset by entity transform (10,0,0), but got identity"
        );
        assert_eq!(
            result[0], expected_bone0,
            "bone[0] should be at entity position"
        );
        assert_eq!(
            result[1], expected_bone1,
            "bone[1] should be offset by entity position"
        );
    }
}

pub unsafe fn run_animation_phase_gpu(
    ctx: &mut FrameContext,
    updates: &AnimationUpdates,
) -> Result<()> {
    if !updates.updated_meshes.is_empty() {
        let mut backend = ctx.create_backend();
        playback_upload_animations(&mut backend, &updates.updated_meshes)?;
    }

    Ok(())
}
