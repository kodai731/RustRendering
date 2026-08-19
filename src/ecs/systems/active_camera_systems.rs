use crate::ecs::component::CameraComponent;
use crate::ecs::resource::{ActiveCamera, Camera};
use crate::ecs::systems::aim_solver::rotate_vector_by_quat;
use crate::ecs::systems::camera_systems::camera_move_to_look_at;
use crate::ecs::world::{Transform, World};
use cgmath::Vector3;

pub fn sync_active_camera_to_view(world: &mut World) {
    let Some(entity) = world.get_resource::<ActiveCamera>().and_then(|a| a.0) else {
        return;
    };

    let pose = world
        .get_component::<Transform>(entity)
        .cloned()
        .zip(world.get_component::<CameraComponent>(entity).cloned());
    let Some((transform, component)) = pose else {
        if let Some(mut active) = world.get_resource_mut::<ActiveCamera>() {
            active.0 = None;
        }
        return;
    };

    let forward = rotate_vector_by_quat(transform.rotation, Vector3::new(0.0, 0.0, -1.0));
    let mut camera = world.resource_mut::<Camera>();

    camera_move_to_look_at(&mut camera, transform.translation + forward, -forward);
    camera.fov_y = component.fov_y;
    camera.near_plane = component.near_plane;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::systems::camera_systems::{compute_camera_direction, compute_camera_position};
    use cgmath::{Deg, Quaternion};

    #[test]
    fn test_sync_active_camera_success() {
        let mut world = World::new();
        let camera = Camera::default();
        world.insert_resource(camera);

        let entity = world.spawn();
        world.insert_component(
            entity,
            Transform {
                translation: Vector3::new(0.0, 0.0, 10.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
        );
        world.insert_component(
            entity,
            CameraComponent {
                fov_y: Deg(50.0),
                near_plane: 0.1,
                far_plane: Some(1000.0),
                physical: Default::default(),
            },
        );
        world.insert_resource(ActiveCamera(Some(entity)));

        sync_active_camera_to_view(&mut world);

        let camera = world.resource::<Camera>();
        let pos = compute_camera_position(&camera);
        let dir = compute_camera_direction(&camera);

        assert!((pos.x - 0.0).abs() < 1e-4);
        assert!((pos.y - 0.0).abs() < 1e-4);
        assert!((pos.z - 10.0).abs() < 1e-4);
        assert!((dir.x - 0.0).abs() < 1e-4);
        assert!((dir.y - 0.0).abs() < 1e-4);
        assert!((dir.z - (-1.0)).abs() < 1e-4);
        assert_eq!(camera.fov_y.0, 50.0);
    }

    #[test]
    fn test_sync_active_camera_missing_component() {
        let mut world = World::new();
        let camera = Camera::default();
        world.insert_resource(camera);

        let entity = world.spawn();
        world.insert_component(
            entity,
            Transform {
                translation: Vector3::new(0.0, 0.0, 10.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
        );
        // No CameraComponent

        world.insert_resource(ActiveCamera(Some(entity)));

        sync_active_camera_to_view(&mut world);

        let active = world.resource::<ActiveCamera>();
        assert!(active.0.is_none());

        let camera = world.resource::<Camera>();
        assert_eq!(camera.distance, Camera::default().distance);
    }
}
