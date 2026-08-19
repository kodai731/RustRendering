use cgmath::Matrix4;
use thyllore_importer_core::{load_gltf_file, CameraProjection};

#[test]
fn test_camera_only_gltf() {
    let path = format!(
        "{}/tests/testmodels/glTF/camera/camera_only.gltf",
        env!("CARGO_MANIFEST_DIR")
    );

    let result = unsafe { load_gltf_file(&path) }.expect("failed to load camera_only.gltf");

    assert_eq!(result.cameras.len(), 1, "expected exactly 1 camera");

    let camera = &result.cameras[0];

    assert_eq!(camera.name, "Camera", "camera name mismatch");
    assert_eq!(camera.node_index, 0, "node_index should be 0");

    match &camera.projection {
        CameraProjection::Perspective {
            yfov, znear, zfar, ..
        } => {
            assert!(
                (yfov - 0.6911).abs() < 1e-4,
                "yfov: expected ~0.6911, got {}",
                yfov
            );
            assert!(
                (znear - 0.1).abs() < 1e-4,
                "znear: expected ~0.1, got {}",
                znear
            );
            assert_eq!(
                *zfar,
                Some(100.0),
                "zfar: expected Some(100.0), got {:?}",
                zfar
            );
        }
        CameraProjection::Orthographic { .. } => {
            panic!("expected Perspective projection, got Orthographic");
        }
    }

    let transform: Matrix4<f32> = camera.world_transform;
    let tx = transform[3][0];
    let ty = transform[3][1];
    let tz = transform[3][2];

    assert!(
        (tx - 7.36).abs() < 1e-3,
        "translation x: expected ~7.36, got {}",
        tx
    );
    assert!(
        (ty - 4.96).abs() < 1e-3,
        "translation y: expected ~4.96, got {}",
        ty
    );
    assert!(
        (tz - 6.93).abs() < 1e-3,
        "translation z: expected ~6.93, got {}",
        tz
    );
}
