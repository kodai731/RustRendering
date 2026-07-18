use anyhow::{anyhow, Result};
use gltf::json::animation::{Interpolation, Property};
use gltf::json::validation::{Checked, USize64};
use gltf::json::{self, Index};

use thyllore_anim_core::{AnimationClip, Skeleton, TransformChannel};

use super::accessors::{
    append_scalar_accessor, append_vec3_accessor, append_vec4_accessor,
    build_bone_to_node_map, quaternion_to_gltf_array,
};

pub(crate) fn write_animation_channels(
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

pub(crate) fn replace_animations(
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

pub(crate) fn append_translation_channel(
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

pub(crate) fn append_rotation_channel(
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

pub(crate) fn append_scale_channel(
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
