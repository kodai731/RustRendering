pub(crate) mod accessors;
pub(crate) mod channels;
pub(crate) mod glb;
pub(crate) mod minimal_json;
use std::fs;
use std::path::Path;

use anyhow::{anyhow, Result};
use gltf::binary::Glb;
use gltf::json::{self};

use thyllore_anim_core::editable::{clip_to_animation, EditableAnimationClip};
use thyllore_anim_core::Skeleton;

use crate::systems::gltf::channels::{replace_animations, write_animation_channels};
use crate::systems::gltf::glb::write_glb;
use crate::systems::gltf::minimal_json::build_minimal_gltf_json;

pub fn export_gltf_animation(
    source_glb_path: &Path,
    clip: &EditableAnimationClip,
    skeleton: &Skeleton,
    output_path: &Path,
) -> Result<()> {
    let raw_bytes = fs::read(source_glb_path)?;
    export_gltf_animation_from_bytes(&raw_bytes, clip, skeleton, output_path)
}

pub fn export_gltf_animation_from_bytes(
    source_glb_bytes: &[u8],
    clip: &EditableAnimationClip,
    skeleton: &Skeleton,
    output_path: &Path,
) -> Result<()> {
    let glb =
        Glb::from_slice(source_glb_bytes).map_err(|e| anyhow!("Failed to parse GLB: {:?}", e))?;

    let mut root: json::Root = json::Root::from_slice(&glb.json)
        .map_err(|e| anyhow!("Failed to parse glTF JSON: {:?}", e))?;

    let mut bin = glb.bin.map(|b| b.into_owned()).unwrap_or_default();

    let baked_clip = clip_to_animation(clip);

    replace_animations(&mut root, &mut bin, &baked_clip, skeleton)?;

    write_glb(&root, bin, output_path)?;

    log!("glTF animation exported to {:?}", output_path);
    Ok(())
}

pub fn export_gltf_animation_only(
    clip: &EditableAnimationClip,
    skeleton: &Skeleton,
    output_path: &Path,
) -> anyhow::Result<()> {
    let baked_clip = clip_to_animation(clip);
    let mut root = build_minimal_gltf_json(skeleton)?;
    let mut bin = Vec::new();

    write_animation_channels(&mut root, &mut bin, &baked_clip, skeleton)?;

    write_glb(&root, bin, output_path)?;

    log!("Animation-only glTF exported to {:?}", output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::systems::gltf::accessors::{
        append_f32_data, build_bone_to_node_map, compute_min_max_scalar, compute_min_max_vec3,
        compute_min_max_vec4, pad_to_4byte_alignment, quaternion_to_gltf_array,
    };
    use cgmath::Quaternion;
    use gltf::json::Index;
    use thyllore_anim_core::editable::{curve_add_keyframe, PropertyType};
    use thyllore_anim_core::Skeleton;

    fn create_test_skeleton() -> Skeleton {
        let mut skeleton = Skeleton::new("test");
        skeleton.add_bone("Hips", None);
        skeleton.add_bone("Spine", Some(0));
        skeleton.add_bone("Head", Some(1));
        skeleton
    }

    fn create_test_nodes() -> Vec<json::scene::Node> {
        vec![
            json::scene::Node {
                name: Some("Armature".to_string()),
                ..Default::default()
            },
            json::scene::Node {
                name: Some("Hips".to_string()),
                ..Default::default()
            },
            json::scene::Node {
                name: Some("Spine".to_string()),
                ..Default::default()
            },
            json::scene::Node {
                name: Some("Head".to_string()),
                ..Default::default()
            },
        ]
    }

    #[test]
    fn test_bone_to_node_mapping() {
        let skeleton = create_test_skeleton();
        let nodes = create_test_nodes();

        let map = build_bone_to_node_map(&skeleton, &nodes);

        assert_eq!(map.get(&0), Some(&1));
        assert_eq!(map.get(&1), Some(&2));
        assert_eq!(map.get(&2), Some(&3));
    }

    #[test]
    fn test_bone_to_node_mapping_missing_nodes() {
        let skeleton = create_test_skeleton();
        let nodes = vec![json::scene::Node {
            name: Some("Hips".to_string()),
            ..Default::default()
        }];

        let map = build_bone_to_node_map(&skeleton, &nodes);

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&0), Some(&0));
        assert!(!map.contains_key(&1));
    }

    #[test]
    fn test_quaternion_to_gltf_array() {
        let q = Quaternion::new(1.0, 0.2, 0.3, 0.4);
        let arr = quaternion_to_gltf_array(q);

        assert_eq!(arr[0], 0.2);
        assert_eq!(arr[1], 0.3);
        assert_eq!(arr[2], 0.4);
        assert_eq!(arr[3], 1.0);
    }

    #[test]
    fn test_quaternion_identity() {
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let arr = quaternion_to_gltf_array(q);

        assert_eq!(arr, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_append_f32_data_alignment() {
        let mut bin = vec![0u8; 5];
        let data = [1.0f32, 2.0];

        let offset = append_f32_data(&mut bin, &data);

        assert_eq!(offset, 8);
        assert_eq!(bin.len(), 16);
    }

    #[test]
    fn test_compute_min_max_scalar() {
        let data = [3.0, 1.0, 4.0, 1.5, 9.0];
        let (min, max) = compute_min_max_scalar(&data);

        assert_eq!(min, 1.0);
        assert_eq!(max, 9.0);
    }

    #[test]
    fn test_compute_min_max_vec3() {
        let data = [1.0, 2.0, 3.0, -1.0, 5.0, 0.0];
        let (min, max) = compute_min_max_vec3(&data);

        assert_eq!(min, [-1.0, 2.0, 0.0]);
        assert_eq!(max, [1.0, 5.0, 3.0]);
    }

    #[test]
    fn test_pad_to_4byte_alignment() {
        let mut bin = vec![0u8; 5];
        pad_to_4byte_alignment(&mut bin);
        assert_eq!(bin.len(), 8);

        let mut bin2 = vec![0u8; 8];
        pad_to_4byte_alignment(&mut bin2);
        assert_eq!(bin2.len(), 8);
    }

    #[test]
    fn test_export_gltf_animation_only_creates_file() {
        let skeleton = create_test_skeleton();
        let mut clip = EditableAnimationClip::new(1, "test_anim".to_string());
        let track = clip.add_track(0, "Hips".to_string());
        let curve = track.get_curve_mut(PropertyType::TranslationX);
        curve_add_keyframe(curve, 0.0, 0.0);
        curve_add_keyframe(curve, 1.0, 1.0);
        clip.duration = 1.0;

        let output_path = std::env::temp_dir().join("test_export_gltf_animation_only.glb");
        export_gltf_animation_only(&clip, &skeleton, &output_path).unwrap();

        assert!(
            output_path.exists(),
            "GLB file was not created at the specified path"
        );
        let bytes = fs::read(&output_path).unwrap();
        assert!(!bytes.is_empty(), "GLB file is empty");
        drop(bytes);
        fs::remove_file(output_path).ok();
    }

    #[test]
    fn test_export_gltf_animation_only_node_count_matches_bone_count() {
        let skeleton = create_test_skeleton();
        let mut clip = EditableAnimationClip::new(1, "test_anim".to_string());
        let track = clip.add_track(0, "Hips".to_string());
        let curve = track.get_curve_mut(PropertyType::TranslationX);
        curve_add_keyframe(curve, 0.0, 0.0);
        curve_add_keyframe(curve, 1.0, 1.0);
        clip.duration = 1.0;

        let output_path = std::env::temp_dir().join("test_node_count.glb");
        export_gltf_animation_only(&clip, &skeleton, &output_path).unwrap();

        let bytes = fs::read(&output_path).unwrap();
        let glb = Glb::from_slice(&bytes).unwrap();
        let root: json::Root = json::Root::from_slice(&glb.json).unwrap();

        assert_eq!(
            root.nodes.len(),
            skeleton.bones.len(),
            "Number of nodes should equal number of bones in the skeleton"
        );
        fs::remove_file(output_path).ok();
    }

    #[test]
    fn test_export_gltf_animation_only_has_animation_channels() {
        let skeleton = create_test_skeleton();
        let mut clip = EditableAnimationClip::new(1, "test_anim".to_string());
        let track = clip.add_track(0, "Hips".to_string());
        let curve = track.get_curve_mut(PropertyType::TranslationX);
        curve_add_keyframe(curve, 0.0, 0.0);
        curve_add_keyframe(curve, 1.0, 1.0);
        clip.duration = 1.0;

        let output_path = std::env::temp_dir().join("test_channels.glb");
        export_gltf_animation_only(&clip, &skeleton, &output_path).unwrap();

        let bytes = fs::read(&output_path).unwrap();
        let glb = Glb::from_slice(&bytes).unwrap();
        let root: json::Root = json::Root::from_slice(&glb.json).unwrap();

        assert!(!root.animations.is_empty(), "No animations found in GLB");
        let anim = &root.animations[0];
        assert!(!anim.channels.is_empty(), "No animation channels found");
        fs::remove_file(output_path).ok();
    }

    #[test]
    fn test_export_gltf_animation_only_node_hierarchy_matches_skeleton() {
        // Create a skeleton with parent-child relationships:
        // Hips (0) -> Spine (1) -> Head (2)
        let skeleton = create_test_skeleton();
        let mut clip = EditableAnimationClip::new(1, "test_anim".to_string());
        let track = clip.add_track(0, "Hips".to_string());
        let curve = track.get_curve_mut(PropertyType::TranslationX);
        curve_add_keyframe(curve, 0.0, 0.0);
        curve_add_keyframe(curve, 1.0, 1.0);
        clip.duration = 1.0;

        let output_path = std::env::temp_dir().join("test_hierarchy.glb");
        export_gltf_animation_only(&clip, &skeleton, &output_path).unwrap();

        let bytes = fs::read(&output_path).unwrap();
        let glb = Glb::from_slice(&bytes).unwrap();
        let root: json::Root = json::Root::from_slice(&glb.json).unwrap();

        // Verify that the parent node's children indices match the skeleton's parent-child relationships
        // Hips (bone 0) should have Spine (bone 1) as child
        let hips_node = &root.nodes[0];
        assert!(
            hips_node.children.is_some(),
            "Hips node should have children"
        );
        let hips_children = hips_node.children.as_ref().unwrap();
        assert!(
            hips_children.contains(&Index::new(1)),
            "Hips should have Spine (index 1) as child"
        );

        // Spine (bone 1) should have Head (bone 2) as child
        let spine_node = &root.nodes[1];
        assert!(
            spine_node.children.is_some(),
            "Spine node should have children"
        );
        let spine_children = spine_node.children.as_ref().unwrap();
        assert!(
            spine_children.contains(&Index::new(2)),
            "Spine should have Head (index 2) as child"
        );

        // Head (bone 2) should have no children
        let head_node = &root.nodes[2];
        assert!(
            head_node.children.is_none() || head_node.children.as_ref().unwrap().is_empty(),
            "Head node should have no children"
        );

        fs::remove_file(output_path).ok();
    }

    #[test]
    fn test_export_gltf_animation_only_scene_contains_root_bones() {
        // Create a skeleton with root bones
        let skeleton = create_test_skeleton();
        let mut clip = EditableAnimationClip::new(1, "test_anim".to_string());
        let track = clip.add_track(0, "Hips".to_string());
        let curve = track.get_curve_mut(PropertyType::TranslationX);
        curve_add_keyframe(curve, 0.0, 0.0);
        curve_add_keyframe(curve, 1.0, 1.0);
        clip.duration = 1.0;

        let output_path = std::env::temp_dir().join("test_scene.glb");
        export_gltf_animation_only(&clip, &skeleton, &output_path).unwrap();

        let bytes = fs::read(&output_path).unwrap();
        let glb = Glb::from_slice(&bytes).unwrap();
        let root: json::Root = json::Root::from_slice(&glb.json).unwrap();

        // Verify that a scene exists and contains nodes matching root bone indices
        assert!(root.scene.is_some(), "GLB should have a default scene");
        let scene_index = root.scene.unwrap().value();
        let scene = &root.scenes[scene_index as usize];

        // The skeleton has one root bone (Hips, id 0), so the scene should contain node 0
        assert!(
            scene.nodes.contains(&Index::new(0)),
            "Scene should contain root bone node (Hips, index 0)"
        );

        // Verify that the scene nodes match all root bone IDs from the skeleton
        for &root_bone_id in &skeleton.root_bone_ids {
            assert!(
                scene.nodes.contains(&Index::new(root_bone_id)),
                "Scene should contain root bone node with id {}",
                root_bone_id
            );
        }

        fs::remove_file(output_path).ok();
    }
}
