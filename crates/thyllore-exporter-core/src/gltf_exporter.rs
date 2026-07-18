use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{anyhow, Result};
use cgmath::SquareMatrix;
use gltf::binary::Glb;
use gltf::json::accessor::{ComponentType, GenericComponentType, Type};
use gltf::json::animation::{Interpolation, Property};
use gltf::json::validation::{Checked, USize64};
use gltf::json::{self, Index};

use thyllore_anim_core::editable::{clip_to_animation, EditableAnimationClip};
use thyllore_anim_core::{AnimationClip, BoneId, Skeleton, TransformChannel};

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

fn write_glb(root: &json::Root, bin: Vec<u8>, output_path: &Path) -> Result<()> {
    let json_bytes = root
        .to_vec()
        .map_err(|e| anyhow!("Failed to serialize glTF JSON: {:?}", e))?;

    let output_glb = Glb {
        header: gltf::binary::Header {
            magic: *b"glTF",
            version: 2,
            length: 0,
        },
        json: Cow::Owned(json_bytes),
        bin: if bin.is_empty() {
            None
        } else {
            Some(Cow::Owned(bin))
        },
    };

    let file = fs::File::create(output_path)?;
    let writer = BufWriter::new(file);
    output_glb
        .to_writer(writer)
        .map_err(|e| anyhow!("Failed to write GLB: {:?}", e))?;

    Ok(())
}

fn build_minimal_gltf_json(skeleton: &Skeleton) -> Result<json::Root> {
    let mut root = json::Root::default();

    // Build a map from BoneId to node index (the position in skeleton.bones)
    let bone_id_to_node_index: HashMap<BoneId, usize> = skeleton
        .bones
        .iter()
        .enumerate()
        .map(|(idx, bone)| (bone.id, idx))
        .collect();

    // Build nodes for each bone with children and matrix
    for (i, bone) in skeleton.bones.iter().enumerate() {
        // Convert BoneId children to node indices via the HashMap
        let children: Vec<Index<json::Node>> = bone
            .children
            .iter()
            .filter_map(|&child_id| bone_id_to_node_index.get(&child_id).map(|&idx| Index::new(idx as u32)))
            .collect();

        // Convert local_transform (cgmath::Matrix4<f32>) to flat array of 16 f32 values
        // in column-major order (same as glTF)
        let mut matrix: Option<[f32; 16]> = None;
        if bone.local_transform != cgmath::Matrix4::identity() {
            let cols: [[f32; 4]; 4] = bone.local_transform.into();
            let flat: [f32; 16] = [
                cols[0][0], cols[0][1], cols[0][2], cols[0][3],
                cols[1][0], cols[1][1], cols[1][2], cols[1][3],
                cols[2][0], cols[2][1], cols[2][2], cols[2][3],
                cols[3][0], cols[3][1], cols[3][2], cols[3][3],
            ];
            matrix = Some(flat);
        }

        let node = json::scene::Node {
            name: Some(bone.name.clone()),
            children: if children.is_empty() {
                None
            } else {
                Some(children)
            },
            matrix,
            ..Default::default()
        };
        root.push(node);
    }

    // Create a scene containing all root bones
    let scene_nodes: Vec<Index<json::Node>> = skeleton
        .root_bone_ids
        .iter()
        .filter_map(|&id| bone_id_to_node_index.get(&id).map(|&idx| Index::new(idx as u32)))
        .collect();
    if !scene_nodes.is_empty() {
        root.scenes.push(json::Scene {
            extensions: None,
            extras: Default::default(),
            name: None,
            nodes: scene_nodes,
        });
        root.scene = Some(Index::<json::Scene>::new(0));
    }

    Ok(root)
}

fn write_animation_channels(
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    clip: &AnimationClip,
    skeleton: &Skeleton,
) -> Result<()> {
    let bone_to_node = build_bone_to_node_map(skeleton, &root.nodes);
    if bone_to_node.is_empty() {
        return Err(anyhow!(
            "No bone-to-node mapping found. Skeleton bone names may not match glTF node names."
        ));
    }

    let buffer_index = Index::<json::Buffer>::new(0);
    let mut channels = Vec::new();
    let mut samplers = Vec::new();

    for (&bone_id, channel) in &clip.channels {
        let Some(&node_index) = bone_to_node.get(&bone_id) else {
            continue;
        };
        let node_idx = Index::new(node_index);

        append_translation_channel(
            root,
            bin,
            buffer_index,
            channel,
            node_idx,
            &mut channels,
            &mut samplers,
        );

        append_rotation_channel(
            root,
            bin,
            buffer_index,
            channel,
            node_idx,
            &mut channels,
            &mut samplers,
        );

        append_scale_channel(
            root,
            bin,
            buffer_index,
            channel,
            node_idx,
            &mut channels,
            &mut samplers,
        );
    }

    if root.buffers.is_empty() {
        root.buffers.push(json::Buffer {
            byte_length: USize64::from(bin.len() as u64),
            name: None,
            uri: None,
            extensions: None,
            extras: Default::default(),
        });
    } else {
        root.buffers[0].byte_length = USize64::from(bin.len() as u64);
    }

    if !channels.is_empty() {
        root.animations.push(json::Animation {
            extensions: None,
            extras: Default::default(),
            channels,
            name: Some(clip.name.clone()),
            samplers,
        });
    }

    Ok(())
}

fn replace_animations(
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    clip: &AnimationClip,
    skeleton: &Skeleton,
) -> Result<()> {
    root.animations.clear();

    let bone_to_node = build_bone_to_node_map(skeleton, &root.nodes);
    if bone_to_node.is_empty() {
        return Err(anyhow!(
            "No bone-to-node mapping found. Skeleton bone names may not match glTF node names."
        ));
    }

    let buffer_index = Index::<json::Buffer>::new(0);
    let mut channels = Vec::new();
    let mut samplers = Vec::new();

    for (&bone_id, channel) in &clip.channels {
        let Some(&node_index) = bone_to_node.get(&bone_id) else {
            continue;
        };
        let node_idx = Index::new(node_index);

        append_translation_channel(
            root,
            bin,
            buffer_index,
            channel,
            node_idx,
            &mut channels,
            &mut samplers,
        );

        append_rotation_channel(
            root,
            bin,
            buffer_index,
            channel,
            node_idx,
            &mut channels,
            &mut samplers,
        );

        append_scale_channel(
            root,
            bin,
            buffer_index,
            channel,
            node_idx,
            &mut channels,
            &mut samplers,
        );
    }

    if root.buffers.is_empty() {
        root.buffers.push(json::Buffer {
            byte_length: USize64::from(bin.len() as u64),
            name: None,
            uri: None,
            extensions: None,
            extras: Default::default(),
        });
    } else {
        root.buffers[0].byte_length = USize64::from(bin.len() as u64);
    }

    if !channels.is_empty() {
        root.animations.push(json::Animation {
            extensions: None,
            extras: Default::default(),
            channels,
            name: Some(clip.name.clone()),
            samplers,
        });
    }

    Ok(())
}

fn append_translation_channel(
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    buffer_index: Index<json::Buffer>,
    channel: &TransformChannel,
    node_index: Index<json::scene::Node>,
    channels: &mut Vec<json::animation::Channel>,
    samplers: &mut Vec<json::animation::Sampler>,
) {
    if channel.translation.is_empty() {
        return;
    }

    let times: Vec<f32> = channel.translation.iter().map(|k| k.time).collect();
    let values: Vec<f32> = channel
        .translation
        .iter()
        .flat_map(|k| [k.value.x, k.value.y, k.value.z])
        .collect();

    let input_accessor = append_scalar_accessor(root, bin, buffer_index, &times);
    let output_accessor = append_vec3_accessor(root, bin, buffer_index, &values);

    let sampler_index = Index::new(samplers.len() as u32);
    samplers.push(json::animation::Sampler {
        extensions: None,
        extras: Default::default(),
        input: input_accessor,
        interpolation: Checked::Valid(Interpolation::Linear),
        output: output_accessor,
    });

    channels.push(json::animation::Channel {
        sampler: sampler_index,
        target: json::animation::Target {
            extensions: None,
            extras: Default::default(),
            node: node_index,
            path: Checked::Valid(Property::Translation),
        },
        extensions: None,
        extras: Default::default(),
    });
}

fn append_rotation_channel(
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    buffer_index: Index<json::Buffer>,
    channel: &TransformChannel,
    node_index: Index<json::scene::Node>,
    channels: &mut Vec<json::animation::Channel>,
    samplers: &mut Vec<json::animation::Sampler>,
) {
    if channel.rotation.is_empty() {
        return;
    }

    let times: Vec<f32> = channel.rotation.iter().map(|k| k.time).collect();
    let values: Vec<f32> = channel
        .rotation
        .iter()
        .flat_map(|k| quaternion_to_gltf_array(k.value))
        .collect();

    let input_accessor = append_scalar_accessor(root, bin, buffer_index, &times);
    let output_accessor = append_vec4_accessor(root, bin, buffer_index, &values);

    let sampler_index = Index::new(samplers.len() as u32);
    samplers.push(json::animation::Sampler {
        extensions: None,
        extras: Default::default(),
        input: input_accessor,
        interpolation: Checked::Valid(Interpolation::Linear),
        output: output_accessor,
    });

    channels.push(json::animation::Channel {
        sampler: sampler_index,
        target: json::animation::Target {
            extensions: None,
            extras: Default::default(),
            node: node_index,
            path: Checked::Valid(Property::Rotation),
        },
        extensions: None,
        extras: Default::default(),
    });
}

fn append_scale_channel(
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    buffer_index: Index<json::Buffer>,
    channel: &TransformChannel,
    node_index: Index<json::scene::Node>,
    channels: &mut Vec<json::animation::Channel>,
    samplers: &mut Vec<json::animation::Sampler>,
) {
    if channel.scale.is_empty() {
        return;
    }

    let times: Vec<f32> = channel.scale.iter().map(|k| k.time).collect();
    let values: Vec<f32> = channel
        .scale
        .iter()
        .flat_map(|k| [k.value.x, k.value.y, k.value.z])
        .collect();

    let input_accessor = append_scalar_accessor(root, bin, buffer_index, &times);
    let output_accessor = append_vec3_accessor(root, bin, buffer_index, &values);

    let sampler_index = Index::new(samplers.len() as u32);
    samplers.push(json::animation::Sampler {
        extensions: None,
        extras: Default::default(),
        input: input_accessor,
        interpolation: Checked::Valid(Interpolation::Linear),
        output: output_accessor,
    });

    channels.push(json::animation::Channel {
        sampler: sampler_index,
        target: json::animation::Target {
            extensions: None,
            extras: Default::default(),
            node: node_index,
            path: Checked::Valid(Property::Scale),
        },
        extensions: None,
        extras: Default::default(),
    });
}

fn append_scalar_accessor(
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    buffer_index: Index<json::Buffer>,
    data: &[f32],
) -> Index<json::Accessor> {
    let (min_val, max_val) = compute_min_max_scalar(data);
    let byte_offset = append_f32_data(bin, data);
    let byte_length = (data.len() * 4) as u64;

    let view_index = root.push(json::buffer::View {
        buffer: buffer_index,
        byte_length: USize64::from(byte_length),
        byte_offset: Some(USize64::from(byte_offset as u64)),
        byte_stride: None,
        name: None,
        target: None,
        extensions: None,
        extras: Default::default(),
    });

    root.push(json::Accessor {
        buffer_view: Some(view_index),
        byte_offset: None,
        count: USize64::from(data.len() as u64),
        component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
        extensions: None,
        extras: Default::default(),
        type_: Checked::Valid(Type::Scalar),
        min: Some(serde_json::json!([min_val])),
        max: Some(serde_json::json!([max_val])),
        name: None,
        normalized: false,
        sparse: None,
    })
}

fn append_vec3_accessor(
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    buffer_index: Index<json::Buffer>,
    data: &[f32],
) -> Index<json::Accessor> {
    let count = data.len() / 3;
    let (min_vals, max_vals) = compute_min_max_vec3(data);
    let byte_offset = append_f32_data(bin, data);
    let byte_length = (data.len() * 4) as u64;

    let view_index = root.push(json::buffer::View {
        buffer: buffer_index,
        byte_length: USize64::from(byte_length),
        byte_offset: Some(USize64::from(byte_offset as u64)),
        byte_stride: None,
        name: None,
        target: None,
        extensions: None,
        extras: Default::default(),
    });

    root.push(json::Accessor {
        buffer_view: Some(view_index),
        byte_offset: None,
        count: USize64::from(count as u64),
        component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
        extensions: None,
        extras: Default::default(),
        type_: Checked::Valid(Type::Vec3),
        min: Some(serde_json::json!(min_vals)),
        max: Some(serde_json::json!(max_vals)),
        name: None,
        normalized: false,
        sparse: None,
    })
}

fn append_vec4_accessor(
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    buffer_index: Index<json::Buffer>,
    data: &[f32],
) -> Index<json::Accessor> {
    let count = data.len() / 4;
    let (min_vals, max_vals) = compute_min_max_vec4(data);
    let byte_offset = append_f32_data(bin, data);
    let byte_length = (data.len() * 4) as u64;

    let view_index = root.push(json::buffer::View {
        buffer: buffer_index,
        byte_length: USize64::from(byte_length),
        byte_offset: Some(USize64::from(byte_offset as u64)),
        byte_stride: None,
        name: None,
        target: None,
        extensions: None,
        extras: Default::default(),
    });

    root.push(json::Accessor {
        buffer_view: Some(view_index),
        byte_offset: None,
        count: USize64::from(count as u64),
        component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
        extensions: None,
        extras: Default::default(),
        type_: Checked::Valid(Type::Vec4),
        min: Some(serde_json::json!(min_vals)),
        max: Some(serde_json::json!(max_vals)),
        name: None,
        normalized: false,
        sparse: None,
    })
}

fn append_f32_data(bin: &mut Vec<u8>, data: &[f32]) -> usize {
    pad_to_4byte_alignment(bin);
    let byte_offset = bin.len();

    for &val in data {
        bin.extend_from_slice(&val.to_le_bytes());
    }

    byte_offset
}

fn pad_to_4byte_alignment(bin: &mut Vec<u8>) {
    let remainder = bin.len() % 4;
    if remainder != 0 {
        let padding = 4 - remainder;
        bin.extend(std::iter::repeat(0u8).take(padding));
    }
}

fn build_bone_to_node_map(
    skeleton: &Skeleton,
    nodes: &[json::scene::Node],
) -> HashMap<BoneId, u32> {
    let mut map = HashMap::new();

    let node_name_to_index: HashMap<&str, u32> = nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| node.name.as_ref().map(|name| (name.as_str(), i as u32)))
        .collect();

    for bone in &skeleton.bones {
        if let Some(&node_index) = node_name_to_index.get(bone.name.as_str()) {
            map.insert(bone.id, node_index);
        }
    }

    map
}

fn quaternion_to_gltf_array(q: cgmath::Quaternion<f32>) -> [f32; 4] {
    [q.v.x, q.v.y, q.v.z, q.s]
}

fn compute_min_max_scalar(data: &[f32]) -> (f32, f32) {
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (min, max)
}

fn compute_min_max_vec3(data: &[f32]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for chunk in data.chunks(3) {
        for i in 0..3 {
            min[i] = min[i].min(chunk[i]);
            max[i] = max[i].max(chunk[i]);
        }
    }

    (min, max)
}

fn compute_min_max_vec4(data: &[f32]) -> ([f32; 4], [f32; 4]) {
    let mut min = [f32::INFINITY; 4];
    let mut max = [f32::NEG_INFINITY; 4];

    for chunk in data.chunks(4) {
        for i in 0..4 {
            min[i] = min[i].min(chunk[i]);
            max[i] = max[i].max(chunk[i]);
        }
    }

    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::Quaternion;
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

        assert!(output_path.exists(), "GLB file was not created at the specified path");
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
        assert!(hips_node.children.is_some(), "Hips node should have children");
        let hips_children = hips_node.children.as_ref().unwrap();
        assert!(hips_children.contains(&Index::new(1)), "Hips should have Spine (index 1) as child");

        // Spine (bone 1) should have Head (bone 2) as child
        let spine_node = &root.nodes[1];
        assert!(spine_node.children.is_some(), "Spine node should have children");
        let spine_children = spine_node.children.as_ref().unwrap();
        assert!(spine_children.contains(&Index::new(2)), "Spine should have Head (index 2) as child");

        // Head (bone 2) should have no children
        let head_node = &root.nodes[2];
        assert!(head_node.children.is_none() || head_node.children.as_ref().unwrap().is_empty(), "Head node should have no children");

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
        assert!(scene.nodes.contains(&Index::new(0)), "Scene should contain root bone node (Hips, index 0)");

        // Verify that the scene nodes match all root bone IDs from the skeleton
        for &root_bone_id in &skeleton.root_bone_ids {
            assert!(scene.nodes.contains(&Index::new(root_bone_id)), "Scene should contain root bone node with id {}", root_bone_id);
        }

        fs::remove_file(output_path).ok();
    }
}
