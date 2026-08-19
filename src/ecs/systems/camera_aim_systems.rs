use cgmath::InnerSpace;

use crate::animation::{normalize_quat, slerp};
use crate::ecs::component::CameraAimTarget;
use crate::ecs::systems::aim_solver::{solve_aim_world_rotation, AimSolveInput};
use crate::ecs::world::{Transform, World};

/// Sync camera aim: for each entity with CameraAimTarget, rotate its Transform.rotation
/// to face the target entity's position, slerped by weight.
pub fn sync_camera_aim(world: &mut World) {
    let entities: Vec<_> = world
        .iter_components::<CameraAimTarget>()
        .map(|(e, _)| e)
        .collect();

    for entity in entities {
        let aim = match world.get_component::<CameraAimTarget>(entity) {
            Some(a) => a.clone(),
            None => continue,
        };

        let source_transform = match world.get_component::<Transform>(entity) {
            Some(t) => t.clone(),
            None => continue,
        };

        let target_translation = match world.get_component::<Transform>(aim.target) {
            Some(t) => t.translation,
            None => continue,
        };

        if aim.weight <= 0.0 {
            continue;
        }

        let up_world = if let Some(up_entity) = aim.up_target {
            match world.get_component::<Transform>(up_entity) {
                Some(t) => {
                    let diff = t.translation - source_transform.translation;
                    if diff.magnitude2() > 1e-8 {
                        Some(diff.normalize())
                    } else {
                        None
                    }
                }
                None => None,
            }
        } else {
            None
        };

        let input = AimSolveInput {
            source_pos: source_transform.translation,
            source_rot: source_transform.rotation,
            target_pos: target_translation,
            aim_axis: aim.aim_axis,
            up_axis: aim.up_axis,
            up_world,
        };

        if let Some(final_rot) = solve_aim_world_rotation(&input) {
            let current = source_transform.rotation;
            let new_rot = normalize_quat(slerp(current, final_rot, aim.weight));
            world.insert_component::<Transform>(
                entity,
                Transform {
                    translation: source_transform.translation,
                    rotation: new_rot,
                    scale: source_transform.scale,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::systems::aim_solver::rotate_vector_by_quat;
    use cgmath::{Quaternion, Vector3};
    #[test]
    fn test_aim_rotation_towards_target() {
        let mut world = World::new();

        // Camera entity at origin with identity rotation
        let camera_entity = world.spawn();
        world.insert_component::<Transform>(
            camera_entity,
            Transform {
                translation: Vector3::new(0.0, 0.0, 0.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
        );

        // Target entity at (0, 0, -5)
        let target_entity = world.spawn();
        world.insert_component::<Transform>(
            target_entity,
            Transform {
                translation: Vector3::new(0.0, 0.0, -5.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
        );

        // Attach CameraAimTarget to camera
        world.insert_component::<CameraAimTarget>(
            camera_entity,
            CameraAimTarget::look_at(target_entity),
        );

        sync_camera_aim(&mut world);

        let cam = world.get_component::<Transform>(camera_entity).unwrap();
        // Forward axis (0,0,-1) rotated by camera rotation should be near (0,0,-1)
        // because target is at (0,0,-5) which is already in the -Z direction
        let forward = rotate_vector_by_quat(cam.rotation, Vector3::new(0.0, 0.0, -1.0));
        assert!(
            (forward - Vector3::new(0.0, 0.0, -1.0)).magnitude() < 1e-5,
            "Forward should be near (0,0,-1), got {:?}",
            forward
        );

        // Move target to (5, 0, 0) and sync again
        world.insert_component::<Transform>(
            target_entity,
            Transform {
                translation: Vector3::new(5.0, 0.0, 0.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
        );

        sync_camera_aim(&mut world);

        let cam = world.get_component::<Transform>(camera_entity).unwrap();
        // Forward axis (0,0,-1) rotated should now point towards (1,0,0)
        let forward = rotate_vector_by_quat(cam.rotation, Vector3::new(0.0, 0.0, -1.0));
        assert!(
            (forward - Vector3::new(1.0, 0.0, 0.0)).magnitude() < 1e-4,
            "Forward should be near (1,0,0), got {:?}",
            forward
        );

        // Up axis (0,1,0) rotated should still be near (0,1,0)
        let up = rotate_vector_by_quat(cam.rotation, Vector3::new(0.0, 1.0, 0.0));
        assert!(
            (up - Vector3::new(0.0, 1.0, 0.0)).magnitude() < 1e-3,
            "Up should be near (0,1,0), got {:?}",
            up
        );
    }

    #[test]
    fn test_weight_zero_no_rotation_change() {
        let mut world = World::new();

        // Camera entity at origin with identity rotation
        let camera_entity = world.spawn();
        let initial_rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        world.insert_component::<Transform>(
            camera_entity,
            Transform {
                translation: Vector3::new(0.0, 0.0, 0.0),
                rotation: initial_rotation,
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
        );

        // Target entity at (5, 0, 0)
        let target_entity = world.spawn();
        world.insert_component::<Transform>(
            target_entity,
            Transform {
                translation: Vector3::new(5.0, 0.0, 0.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
        );

        // Attach CameraAimTarget with weight 0
        let mut aim = CameraAimTarget::look_at(target_entity);
        aim.weight = 0.0;
        world.insert_component::<CameraAimTarget>(camera_entity, aim);

        sync_camera_aim(&mut world);

        let cam = world.get_component::<Transform>(camera_entity).unwrap();
        assert!(
            (cam.rotation - initial_rotation).magnitude() < 1e-8,
            "Rotation should not change with weight 0, got {:?} vs {:?}",
            cam.rotation,
            initial_rotation
        );
    }
}
