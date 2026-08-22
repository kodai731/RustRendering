use std::path::Path;

use anyhow::Result;
use cgmath::{Quaternion, Vector3};
use gltf::json::{self, validation::Checked, validation::USize64, Index};

use thyllore_anim_core::{Keyframe, TransformChannel};

use crate::systems::gltf::channels::{append_rotation_channel, append_translation_channel};
use crate::systems::gltf::glb::write_glb;

/// A camera to export: its rest pose plus an optional keyframed trajectory.
#[derive(Clone, Debug)]
pub struct CameraExport {
    pub name: String,
    pub translation: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub yfov_radians: f32,
    pub znear: f32,
    pub zfar: Option<f32>,
    pub translation_keys: Vec<(f32, Vector3<f32>)>,
    pub rotation_keys: Vec<(f32, Quaternion<f32>)>,
    pub animation_name: String,
}

pub fn export_gltf_camera(camera: &CameraExport, output_path: &Path) -> Result<()> {
    let (root, bin) = build_camera_gltf(camera);
    write_glb(&root, bin, output_path)?;
    log!("Camera glTF exported to {:?}", output_path);
    Ok(())
}

pub(crate) fn build_camera_gltf(camera: &CameraExport) -> (json::Root, Vec<u8>) {
    let mut root = json::Root::default();
    let mut bin = Vec::new();

    root.cameras.push(json::Camera {
        name: Some(camera.name.clone()),
        orthographic: None,
        perspective: Some(json::camera::Perspective {
            aspect_ratio: None,
            yfov: camera.yfov_radians,
            zfar: camera.zfar,
            znear: camera.znear,
            extensions: None,
            extras: Default::default(),
        }),
        type_: Checked::Valid(json::camera::Type::Perspective),
        extensions: None,
        extras: Default::default(),
    });

    let node_index = Index::<json::scene::Node>::new(0);
    root.push(json::scene::Node {
        name: Some(camera.name.clone()),
        camera: Some(Index::new(0)),
        translation: Some(camera.translation.into()),
        rotation: Some(json::scene::UnitQuaternion([
            camera.rotation.v.x,
            camera.rotation.v.y,
            camera.rotation.v.z,
            camera.rotation.s,
        ])),
        ..Default::default()
    });
    root.scenes.push(json::Scene {
        name: None,
        nodes: vec![node_index],
        extensions: None,
        extras: Default::default(),
    });
    root.scene = Some(Index::new(0));

    let channel = TransformChannel {
        translation: camera
            .translation_keys
            .iter()
            .map(|(time, value)| Keyframe::new(*time, *value))
            .collect(),
        rotation: camera
            .rotation_keys
            .iter()
            .map(|(time, value)| Keyframe::new(*time, *value))
            .collect(),
        scale: Vec::new(),
    };
    let mut channels = Vec::new();
    let mut samplers = Vec::new();
    let buffer_index = Index::<json::Buffer>::new(0);
    append_translation_channel(
        &mut root,
        &mut bin,
        buffer_index,
        &channel,
        node_index,
        &mut channels,
        &mut samplers,
    );
    append_rotation_channel(
        &mut root,
        &mut bin,
        buffer_index,
        &channel,
        node_index,
        &mut channels,
        &mut samplers,
    );

    if !bin.is_empty() {
        root.buffers.push(json::Buffer {
            byte_length: USize64::from(bin.len() as u64),
            name: None,
            uri: None,
            extensions: None,
            extras: Default::default(),
        });
    }
    if !channels.is_empty() {
        root.animations.push(json::Animation {
            extensions: None,
            extras: Default::default(),
            channels,
            name: Some(camera.animation_name.clone()),
            samplers,
        });
    }

    (root, bin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Deg, Rotation3};

    fn sample_camera() -> CameraExport {
        CameraExport {
            name: "Camera".to_string(),
            translation: Vector3::new(7.36, 4.96, 6.93),
            rotation: Quaternion::from_angle_y(Deg(45.0)),
            yfov_radians: 0.69,
            znear: 0.1,
            zfar: Some(100.0),
            translation_keys: vec![
                (0.0, Vector3::new(7.36, 4.96, 6.93)),
                (1.0, Vector3::new(7.0, 4.9, 6.5)),
            ],
            rotation_keys: vec![
                (0.0, Quaternion::from_angle_y(Deg(45.0))),
                (1.0, Quaternion::from_angle_y(Deg(60.0))),
            ],
            animation_name: "CameraDirection".to_string(),
        }
    }

    #[test]
    fn camera_gltf_has_a_camera_node_and_two_animation_channels() {
        let (root, bin) = build_camera_gltf(&sample_camera());

        assert_eq!(root.cameras.len(), 1);
        assert_eq!(root.nodes[0].camera, Some(Index::new(0)));
        assert_eq!(root.animations.len(), 1);
        assert_eq!(root.animations[0].channels.len(), 2);
        assert_eq!(root.buffers[0].byte_length.0 as usize, bin.len());
    }

    #[test]
    fn exported_camera_reimports_with_its_pose_and_animation() -> Result<()> {
        let dir = std::env::temp_dir().join("thyllore_camera_export_roundtrip");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("camera.glb");
        export_gltf_camera(&sample_camera(), &path)?;

        let (document, _buffers, _images) = gltf::import(&path)?;
        let node = document.nodes().find(|n| n.camera().is_some()).unwrap();
        let (translation, _, _) = node.transform().decomposed();
        assert!((translation[0] - 7.36).abs() < 1e-5);
        let animation = document.animations().next().unwrap();
        assert_eq!(animation.channels().count(), 2);

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
