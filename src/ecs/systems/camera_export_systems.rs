use std::path::Path;

use anyhow::{anyhow, Result};
use cgmath::{Deg, Euler, Quaternion, Rad, Vector3};
use thyllore_anim_core::editable::{curve_sample, EditableAnimationClip};
use thyllore_exporter_core::systems::gltf::camera::{export_gltf_camera, CameraExport};

use crate::ecs::component::{CameraComponent, CameraParam};
use crate::ecs::resource::{ActiveCamera, ClipLibrary};
use crate::ecs::systems::scalar_clip_systems::find_entity_clip_id;
use crate::ecs::world::{Entity, Name, Transform, World};

/// Build the glTF camera export for `entity`: rest pose from its components and
/// one translation/rotation key per distinct key time on its Camera domain clip.
pub fn build_camera_export(world: &World, entity: Entity) -> Option<CameraExport> {
    let transform = world.get_component::<Transform>(entity)?;
    let camera = world.get_component::<CameraComponent>(entity)?;
    let name = world
        .get_component::<Name>(entity)
        .map(|name| name.0.clone())
        .unwrap_or_else(|| format!("Camera {entity}"));

    let clip = find_entity_clip_id(world, entity).and_then(|clip_id| {
        world
            .get_resource::<ClipLibrary>()
            .and_then(|library| library.get(clip_id).map(|clip| clip.clone()))
    });
    let (translation_keys, rotation_keys) = clip
        .as_ref()
        .map(|clip| camera_pose_keys(clip, &transform))
        .unwrap_or_default();

    Some(CameraExport {
        name,
        translation: transform.translation,
        rotation: transform.rotation,
        yfov_radians: Rad::from(camera.fov_y).0,
        znear: camera.near_plane,
        zfar: camera.far_plane,
        translation_keys,
        rotation_keys,
        animation_name: clip.map(|clip| clip.name).unwrap_or_default(),
    })
}

fn camera_pose_keys(
    clip: &EditableAnimationClip,
    rest: &Transform,
) -> (Vec<(f32, Vector3<f32>)>, Vec<(f32, Quaternion<f32>)>) {
    let pose_params = [
        CameraParam::TranslationX,
        CameraParam::TranslationY,
        CameraParam::TranslationZ,
        CameraParam::RotationX,
        CameraParam::RotationY,
        CameraParam::RotationZ,
    ];
    let pose_curves: Vec<_> = clip
        .scalar_curves
        .iter()
        .filter(|curve| {
            pose_params
                .iter()
                .any(|param| param.property_type() == curve.property_type)
        })
        .collect();

    let mut times: Vec<f32> = pose_curves
        .iter()
        .flat_map(|curve| curve.keyframes.iter().map(|key| key.time))
        .collect();
    times.sort_by(|a, b| a.total_cmp(b));
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    let rest_euler: Euler<Rad<f32>> = Euler::from(rest.rotation);
    let rest_degrees = [
        Deg::from(rest_euler.x).0,
        Deg::from(rest_euler.y).0,
        Deg::from(rest_euler.z).0,
    ];
    let sample = |param: CameraParam, time: f32, fallback: f32| {
        pose_curves
            .iter()
            .find(|curve| curve.property_type == param.property_type())
            .and_then(|curve| curve_sample(curve, time))
            .unwrap_or(fallback)
    };

    let translation_keys = times
        .iter()
        .map(|&time| {
            (
                time,
                Vector3::new(
                    sample(CameraParam::TranslationX, time, rest.translation.x),
                    sample(CameraParam::TranslationY, time, rest.translation.y),
                    sample(CameraParam::TranslationZ, time, rest.translation.z),
                ),
            )
        })
        .collect();
    let rotation_keys = times
        .iter()
        .map(|&time| {
            let degrees = [
                sample(CameraParam::RotationX, time, rest_degrees[0]),
                sample(CameraParam::RotationY, time, rest_degrees[1]),
                sample(CameraParam::RotationZ, time, rest_degrees[2]),
            ];
            (
                time,
                Quaternion::from(Euler::new(
                    Rad::from(Deg(degrees[0])),
                    Rad::from(Deg(degrees[1])),
                    Rad::from(Deg(degrees[2])),
                )),
            )
        })
        .collect();
    (translation_keys, rotation_keys)
}

/// Export the active camera (rest pose + Camera domain trajectory) as a .glb.
pub fn export_active_camera_gltf(world: &World, output_path: &Path) -> Result<()> {
    let entity = world
        .get_resource::<ActiveCamera>()
        .and_then(|active| active.0)
        .ok_or_else(|| anyhow!("no active camera to export"))?;
    let camera = build_camera_export(world, entity)
        .ok_or_else(|| anyhow!("active camera entity {entity} has no camera components"))?;
    export_gltf_camera(&camera, output_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::AssetStorage;
    use crate::ecs::component::{ClipSchedule, CAMERA_DOMAIN};
    use crate::ecs::resource::TimelineState;
    use crate::ecs::systems::scalar_clip_systems::{ensure_entity_clip, scalar_clip_insert_key};

    #[test]
    fn export_samples_one_pose_key_per_distinct_key_time() {
        let mut world = World::new();
        let mut assets = AssetStorage::default();
        world.insert_resource(ClipLibrary::default());
        world.insert_resource(TimelineState::default());
        let entity = world.spawn();
        world.insert_component(entity, Name("Cam".to_string()));
        world.insert_component(
            entity,
            Transform {
                translation: Vector3::new(1.0, 2.0, 3.0),
                ..Transform::default()
            },
        );
        world.insert_component(entity, CameraComponent::default());
        world.insert_component(entity, ClipSchedule::default());
        let clip_id = ensure_entity_clip(&mut world, &mut assets, entity, &CAMERA_DOMAIN);
        {
            let mut library = world.resource_mut::<ClipLibrary>();
            let mut clip = library.get_mut(clip_id).unwrap();
            scalar_clip_insert_key(
                &mut clip,
                CameraParam::TranslationX.property_type(),
                0.0,
                1.0,
            );
            scalar_clip_insert_key(
                &mut clip,
                CameraParam::TranslationX.property_type(),
                2.0,
                5.0,
            );
            scalar_clip_insert_key(&mut clip, CameraParam::RotationY.property_type(), 1.0, 90.0);
        }

        let export = build_camera_export(&world, entity).unwrap();

        let times: Vec<f32> = export.translation_keys.iter().map(|k| k.0).collect();
        assert_eq!(times, vec![0.0, 1.0, 2.0]);
        assert!((export.translation_keys[1].1.x - 3.0).abs() < 1e-5);
        assert!((export.translation_keys[1].1.y - 2.0).abs() < 1e-5);
        assert_eq!(export.rotation_keys.len(), 3);
        assert_eq!(export.name, "Cam");
        assert_eq!(export.animation_name, "Camera");
    }
}
