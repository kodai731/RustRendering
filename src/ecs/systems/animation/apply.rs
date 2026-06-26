use anyhow::Result;
use cgmath::{Matrix4, SquareMatrix, Vector3, Vector4};

use crate::animation::{compose_transform, MorphAnimationSystem, Skeleton, SkeletonPose};
use crate::ecs::apply_skinning;
use crate::render::RenderBackend;
use crate::vulkanr::resource::graphics_resource::{GraphicsResources, NodeData};

pub(crate) fn build_node_based_bone_transforms(
    nodes: &[NodeData],
    skeleton: &Skeleton,
) -> Vec<Matrix4<f32>> {
    let mut transforms = vec![Matrix4::identity(); skeleton.bones.len()];
    for bone in &skeleton.bones {
        let matched_node = nodes.iter().find(|n| n.name == bone.name).or_else(|| {
            bone.node_index
                .and_then(|idx| nodes.iter().find(|n| n.index == idx))
        });

        if let Some(node) = matched_node {
            transforms[bone.id as usize] = node.global_transform;
        }
    }
    transforms
}

pub fn apply_skinning_to_single_mesh(
    graphics: &mut GraphicsResources,
    mesh_idx: usize,
    global_transforms: &[Matrix4<f32>],
    skeleton: &Skeleton,
) -> bool {
    if mesh_idx >= graphics.meshes.len() {
        return false;
    }

    let skin_data = {
        let mesh = &graphics.meshes[mesh_idx];
        mesh.skin_data.clone()
    };

    let Some(skin_data) = skin_data else {
        return false;
    };

    let vertex_count = skin_data.base_positions.len();
    let mut skinned_positions = vec![Vector3::new(0.0, 0.0, 0.0); vertex_count];
    let mut skinned_normals = vec![Vector3::new(0.0, 1.0, 0.0); vertex_count];

    let _ = apply_skinning(
        &skin_data,
        global_transforms,
        skeleton,
        &mut skinned_positions,
        &mut skinned_normals,
    );

    if std::env::var("THYLLORE_SKIN_DEBUG").is_ok() {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for p in &skinned_positions {
            min[0] = min[0].min(p.x);
            max[0] = max[0].max(p.x);
            min[1] = min[1].min(p.y);
            max[1] = max[1].max(p.y);
            min[2] = min[2].min(p.z);
            max[2] = max[2].max(p.z);
        }
        log!(
            "SKIN mesh={} verts={} bones={} aabb_min=[{:.3},{:.3},{:.3}] aabb_max=[{:.3},{:.3},{:.3}]",
            mesh_idx,
            vertex_count,
            skeleton.bones.len(),
            min[0],
            min[1],
            min[2],
            max[0],
            max[1],
            max[2]
        );
    }

    let mesh = &mut graphics.meshes[mesh_idx];
    for (i, pos) in skinned_positions.iter().enumerate() {
        if i < mesh.vertex_data.vertices.len() {
            mesh.vertex_data.vertices[i].pos.x = pos.x;
            mesh.vertex_data.vertices[i].pos.y = pos.y;
            mesh.vertex_data.vertices[i].pos.z = pos.z;
        }
    }
    for (i, normal) in skinned_normals.iter().enumerate() {
        if i < mesh.vertex_data.vertices.len() {
            mesh.vertex_data.vertices[i].normal.x = normal.x;
            mesh.vertex_data.vertices[i].normal.y = normal.y;
            mesh.vertex_data.vertices[i].normal.z = normal.z;
        }
    }

    true
}

pub fn apply_node_animation_to_single_mesh(
    graphics: &mut GraphicsResources,
    mesh_idx: usize,
    nodes: &[NodeData],
    scale: f32,
) -> bool {
    if mesh_idx >= graphics.meshes.len() {
        return false;
    }

    let mesh = &graphics.meshes[mesh_idx];
    if mesh.skin_data.is_some() || mesh.base_vertices.is_empty() {
        return false;
    }

    let Some(node_idx) = mesh.node_index else {
        return false;
    };

    let node_found = nodes.iter().find(|n| n.index == node_idx);
    let Some(node) = node_found else {
        return false;
    };

    let transform = node.global_transform;

    let mesh = &mut graphics.meshes[mesh_idx];
    for (i, v) in mesh.vertex_data.vertices.iter_mut().enumerate() {
        if i < mesh.base_vertices.len() {
            let base = &mesh.base_vertices[i];
            let pos = transform * Vector4::new(base.pos.x, base.pos.y, base.pos.z, 1.0);
            v.pos.x = pos.x * scale;
            v.pos.y = pos.y * scale;
            v.pos.z = pos.z * scale;
        }
    }

    true
}

pub fn compute_node_global_transforms(
    nodes: &mut [NodeData],
    skeleton: &Skeleton,
    pose: &SkeletonPose,
) {
    if nodes.is_empty() {
        return;
    }

    for bone in &skeleton.bones {
        if let Some(node) = nodes.iter_mut().find(|n| n.name == bone.name) {
            let idx = bone.id as usize;
            if idx < pose.bone_poses.len() {
                let bp = &pose.bone_poses[idx];
                node.local_transform = compose_transform(bp.translation, bp.rotation, bp.scale);
            }
        }
    }

    let node_count = nodes.len();

    fn compute_global(
        nodes: &[NodeData],
        node_idx: usize,
        computed: &mut [bool],
        global_transforms: &mut [Matrix4<f32>],
    ) -> Matrix4<f32> {
        if computed[node_idx] {
            return global_transforms[node_idx];
        }

        let local = nodes[node_idx].local_transform;
        let global = if let Some(parent_idx) = nodes[node_idx].parent_index {
            if let Some(parent_array_idx) = nodes.iter().position(|n| n.index == parent_idx) {
                let parent_global =
                    compute_global(nodes, parent_array_idx, computed, global_transforms);
                parent_global * local
            } else {
                local
            }
        } else {
            local
        };

        global_transforms[node_idx] = global;
        computed[node_idx] = true;
        global
    }

    let mut computed = vec![false; node_count];
    let mut global_transforms = vec![Matrix4::identity(); node_count];

    for i in 0..node_count {
        compute_global(nodes, i, &mut computed, &mut global_transforms);
    }

    for (i, node) in nodes.iter_mut().enumerate() {
        node.global_transform = global_transforms[i];
    }
}

pub fn apply_morph_animation(
    graphics: &mut GraphicsResources,
    morph_animation: &MorphAnimationSystem,
    time: f32,
) -> Vec<usize> {
    if morph_animation.is_empty() {
        return Vec::new();
    }

    let animation_index = morph_animation.get_animation_index(time);
    let mesh_count = morph_animation.targets.len().min(graphics.meshes.len());
    let mut updated_mesh_indices = Vec::new();

    for mesh_idx in 0..mesh_count {
        let morph_targets = &morph_animation.targets[mesh_idx];
        if morph_targets.is_empty() {
            continue;
        }

        let base_vertices = &morph_animation.base_vertices[mesh_idx];
        let vertices = &mut graphics.meshes[mesh_idx].vertex_data.vertices;

        for (i, v) in vertices.iter_mut().enumerate() {
            if i < base_vertices.len() {
                let base = base_vertices[i];
                v.pos.x = base[0];
                v.pos.y = base[1];
                v.pos.z = base[2];
            }
        }

        let morph_anim = &morph_animation.animations[animation_index];
        let scale_factor = morph_animation.scale_factor;
        for (weight_idx, &weight) in morph_anim.weights.iter().enumerate() {
            if weight_idx >= morph_targets.len() {
                break;
            }
            let morph_target = &morph_targets[weight_idx];
            for (j, delta_pos) in morph_target.positions.iter().enumerate() {
                if j < vertices.len() {
                    vertices[j].pos.x += delta_pos[0] * weight * scale_factor;
                    vertices[j].pos.y += delta_pos[1] * weight * scale_factor;
                    vertices[j].pos.z += delta_pos[2] * weight * scale_factor;
                }
            }
        }

        updated_mesh_indices.push(mesh_idx);
    }

    updated_mesh_indices
}

pub unsafe fn upload_animations(
    backend: &mut dyn RenderBackend,
    updated_meshes: &[usize],
) -> Result<()> {
    for &mesh_idx in updated_meshes {
        backend.upload_mesh_vertices(mesh_idx)?;
    }

    if !updated_meshes.is_empty() {
        backend.update_acceleration_structure(updated_meshes)?;
        backend.rebuild_tlas()?;
    }

    Ok(())
}

pub(crate) fn merge_updated_indices(morph: Vec<usize>, anim: Vec<usize>) -> Vec<usize> {
    let mut all = morph;
    for idx in anim {
        if !all.contains(&idx) {
            all.push(idx);
        }
    }
    all
}

pub fn prepare_node_animation(
    graphics: &mut GraphicsResources,
    nodes: &mut [NodeData],
    skeleton: &Skeleton,
    pose: &SkeletonPose,
    node_animation_scale: f32,
) -> Vec<usize> {
    compute_node_global_transforms(nodes, skeleton, pose);

    let mut updated_mesh_indices = Vec::new();

    for (mesh_idx, mesh) in graphics.meshes.iter_mut().enumerate() {
        if mesh.skin_data.is_some() || mesh.base_vertices.is_empty() {
            continue;
        }

        let Some(node_idx) = mesh.node_index else {
            continue;
        };

        let node_found = nodes.iter().find(|n| n.index == node_idx);
        let Some(node) = node_found else {
            continue;
        };

        let transform = node.global_transform;

        for (i, v) in mesh.vertex_data.vertices.iter_mut().enumerate() {
            if i < mesh.base_vertices.len() {
                let base = &mesh.base_vertices[i];
                let pos = transform * Vector4::new(base.pos.x, base.pos.y, base.pos.z, 1.0);
                v.pos.x = pos.x * node_animation_scale;
                v.pos.y = pos.y * node_animation_scale;
                v.pos.z = pos.z * node_animation_scale;
            }
        }

        updated_mesh_indices.push(mesh_idx);
    }

    updated_mesh_indices
}
