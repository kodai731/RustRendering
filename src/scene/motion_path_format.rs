use cgmath::Vector3;
use thyllore_scene_core::declare_scene_format;

use crate::ecs::component::MotionPath;

declare_scene_format! {
    component: MotionPath,
    record: MotionPathSceneRecord,
    items {
        snapshot: motion_path_parameter_snapshot,
        scalars: MOTION_PATH_SCALAR_PARAMS,
        ui: MOTION_PATH_UI_PARAMS,
        overwrite: overwrite_motion_path_persisted_fields,
    },
    persisted {
        center: [f32; 3] {
            get: |p| [p.center.x, p.center.y, p.center.z],
            set: |p, v| p.center = Vector3::new(v[0], v[1], v[2]),
        },
        radius: f32 { get: |p| p.radius, set: |p, v| p.radius = v },
        angular_speed: f32 {
            get: |p| p.angular_speed,
            set: |p, v| p.angular_speed = v,
        },
        phase_offset: f32 {
            get: |p| p.phase_offset,
            set: |p, v| p.phase_offset = v,
        },
        enabled: bool { get: |p| p.enabled, set: |p, v| p.enabled = v },
    },
    runtime {},
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_path_serializes_to_the_legacy_motion_path_data_json() {
        let path = MotionPath {
            center: Vector3::new(1.0, 2.0, 3.0),
            radius: 5.0,
            angular_speed: 0.5,
            phase_offset: 1.5,
            enabled: true,
        };
        let json = serde_json::to_string(&path).expect("serialize");
        assert_eq!(
            json,
            r#"{"center":[1.0,2.0,3.0],"radius":5.0,"angular_speed":0.5,"phase_offset":1.5,"enabled":true}"#
        );
    }

    #[test]
    fn test_motion_path_roundtrip() {
        let path = MotionPath {
            center: Vector3::new(-1.0, 0.5, 2.0),
            radius: 2.5,
            angular_speed: 0.7,
            phase_offset: 0.25,
            enabled: true,
        };
        let json = serde_json::to_string(&path).expect("serialize");
        let restored: MotionPath = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.center, path.center);
        assert_eq!(restored.radius, path.radius);
        assert_eq!(restored.angular_speed, path.angular_speed);
        assert_eq!(restored.phase_offset, path.phase_offset);
        assert_eq!(restored.enabled, path.enabled);
    }
}
