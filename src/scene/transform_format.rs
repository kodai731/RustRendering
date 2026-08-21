use cgmath::{Quaternion, Vector3};
use thyllore_scene_core::declare_scene_format;

use crate::ecs::world::Transform;

declare_scene_format! {
    component: Transform,
    record: TransformSceneRecord,
    items {
        snapshot: transform_parameter_snapshot,
        scalars: TRANSFORM_SCALAR_PARAMS,
        ui: TRANSFORM_UI_PARAMS,
        overwrite: overwrite_transform_persisted_fields,
    },
    persisted {
        position: [f32; 3] {
            get: |t| [t.translation.x, t.translation.y, t.translation.z],
            set: |t, v| t.translation = Vector3::new(v[0], v[1], v[2]),
        },
        // On-disk rotation order is [x, y, z, w] (identity = [0, 0, 0, 1]), unlike flame's [s, x, y, z].
        rotation: [f32; 4] {
            get: |t| [t.rotation.v.x, t.rotation.v.y, t.rotation.v.z, t.rotation.s],
            set: |t, v| t.rotation = Quaternion::new(v[3], v[0], v[1], v[2]),
        },
        scale: [f32; 3] {
            get: |t| [t.scale.x, t.scale.y, t.scale.z],
            set: |t, v| t.scale = Vector3::new(v[0], v[1], v[2]),
        },
    },
    runtime {},
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_transform_serializes_to_the_legacy_transform_data_json() {
        let json = serde_json::to_string(&Transform::default()).expect("serialize");
        assert_eq!(
            json,
            r#"{"position":[0.0,0.0,0.0],"rotation":[0.0,0.0,0.0,1.0],"scale":[1.0,1.0,1.0]}"#
        );
    }

    #[test]
    fn test_transform_roundtrip_preserves_rotation() {
        let mut transform = Transform::default();
        transform.translation = Vector3::new(1.0, 2.0, 3.0);
        transform.rotation = Quaternion::new(0.5, 0.5, 0.5, 0.5);
        transform.scale = Vector3::new(2.0, 2.0, 2.0);

        let json = serde_json::to_string(&transform).expect("serialize");
        let restored: Transform = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.translation, transform.translation);
        assert_eq!(restored.rotation, transform.rotation);
        assert_eq!(restored.scale, transform.scale);
    }
}
