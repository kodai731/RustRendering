use std::collections::HashMap;

use anyhow::Result;
use cgmath::SquareMatrix;
use gltf::json::{self, Index};

use thyllore_anim_core::BoneId;

pub(crate) fn build_minimal_gltf_json(
    skeleton: &thyllore_anim_core::Skeleton,
) -> Result<json::Root> {
    let mut root = json::Root::default();
    // Build a map from BoneId to node index (the position in skeleton.bones)
    let bone_id_to_node_index: HashMap<BoneId, usize> = skeleton
        .bones
        .iter()
        .enumerate()
        .map(|(idx, bone)| (bone.id, idx))
        .collect();

    // Build nodes for each bone with children and matrix
    for (_i, bone) in skeleton.bones.iter().enumerate() {
        // Convert BoneId children to node indices via the HashMap
        let children: Vec<Index<json::Node>> = bone
            .children
            .iter()
            .filter_map(|&child_id| {
                bone_id_to_node_index
                    .get(&child_id)
                    .map(|&idx| Index::new(idx as u32))
            })
            .collect();

        // Convert local_transform (cgmath::Matrix4<f32>) to flat array of 16 f32 values
        // in column-major order (same as glTF)
        let mut matrix: Option<[f32; 16]> = None;
        if bone.local_transform != cgmath::Matrix4::identity() {
            let cols: [[f32; 4]; 4] = bone.local_transform.into();
            let flat: [f32; 16] = [
                cols[0][0], cols[0][1], cols[0][2], cols[0][3], cols[1][0], cols[1][1], cols[1][2],
                cols[1][3], cols[2][0], cols[2][1], cols[2][2], cols[2][3], cols[3][0], cols[3][1],
                cols[3][2], cols[3][3],
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

    let root_bone_nodes: Vec<Index<json::Node>> = skeleton
        .root_bone_ids
        .iter()
        .filter_map(|&id| {
            bone_id_to_node_index
                .get(&id)
                .map(|&idx| Index::new(idx as u32))
        })
        .collect();

    // A non-identity root_transform is preserved as a parent node above the root
    // bones so importers can reconstruct Skeleton::root_transform.
    let scene_nodes: Vec<Index<json::Node>> =
        if skeleton.root_transform != cgmath::Matrix4::identity() && !root_bone_nodes.is_empty() {
            let mut name = String::from("Armature");
            while skeleton.bones.iter().any(|b| b.name == name) {
                name.push('_');
            }

            let cols: [[f32; 4]; 4] = skeleton.root_transform.into();
            let flat: [f32; 16] = [
                cols[0][0], cols[0][1], cols[0][2], cols[0][3], cols[1][0], cols[1][1], cols[1][2],
                cols[1][3], cols[2][0], cols[2][1], cols[2][2], cols[2][3], cols[3][0], cols[3][1],
                cols[3][2], cols[3][3],
            ];

            let armature_index = root.nodes.len() as u32;
            root.push(json::scene::Node {
                name: Some(name),
                children: Some(root_bone_nodes),
                matrix: Some(flat),
                ..Default::default()
            });
            vec![Index::new(armature_index)]
        } else {
            root_bone_nodes
        };

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
