use crate::app::FrameContext;
use crate::ecs::component::FlameBoneAttachment;
use crate::ecs::world::Entity;

pub fn flame_bone_attach_sync(ctx: &mut FrameContext) {
    use crate::ecs::component::resolve_bone_index;
    use crate::ecs::resource::gizmo::BoneGizmoData;

    let (skeleton_id, cached_global_transforms, bone_local_offsets, mesh_scale) = {
        let bone_gizmo = match ctx.world.get_resource::<BoneGizmoData>() {
            Some(bg) => bg,
            None => return,
        };
        let skeleton_id = match bone_gizmo.cached_skeleton_id {
            Some(id) => id,
            None => return,
        };
        (
            skeleton_id,
            bone_gizmo.cached_global_transforms.clone(),
            bone_gizmo.bone_local_offsets.clone(),
            bone_gizmo.mesh_scale,
        )
    };

    let skeleton = match ctx.assets.get_skeleton_by_skeleton_id(skeleton_id) {
        Some(s) => s,
        None => return,
    };
    let bone_names: Vec<String> = skeleton.bones.iter().map(|b| b.name.clone()).collect();

    let positions =
        crate::ecs::systems::bone_gizmo_systems::compute_display_transforms_with_skeleton(
            &cached_global_transforms,
            &bone_local_offsets,
            mesh_scale,
            Some(&skeleton),
        );

    let flame_entities: Vec<Entity> = ctx.world.query_flames();
    for &entity in &flame_entities {
        let attachment = match ctx.world.get_component::<FlameBoneAttachment>(entity) {
            Some(a) => a,
            None => continue,
        };
        let idx = match resolve_bone_index(&attachment.bone, &bone_names) {
            Some(i) => i,
            None => continue,
        };
        if idx >= positions.len() {
            continue;
        }
        let translation =
            cgmath::Vector3::new(positions[idx][0], positions[idx][1], positions[idx][2]);
        if let Some(transform) = ctx
            .world
            .get_component_mut::<crate::ecs::world::Transform>(entity)
        {
            transform.translation = translation;
        } else {
            ctx.world.insert_component(
                entity,
                crate::ecs::world::Transform {
                    translation,
                    rotation: cgmath::Quaternion::new(0.0, 0.0, 0.0, 1.0),
                    scale: cgmath::Vector3::new(1.0, 1.0, 1.0),
                },
            );
        }
    }
}
