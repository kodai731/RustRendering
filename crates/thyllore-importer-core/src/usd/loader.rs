use std::cmp::Ordering;
use std::collections::HashMap;

use anyhow::{anyhow, Result};
use cgmath::{Matrix4, Quaternion, SquareMatrix, Vector3, Vector4};

use openusd::gf::{Matrix4d, Vec2f, Vec3f};
use openusd::schemas::geom::{Mesh, PointBased};
use openusd::schemas::skel::{
    BlendShape, InfluenceInterpolation, SkelAnimQuery, SkelBindingAPI, Skeleton as UsdSkeleton,
};
use openusd::sdf;
use openusd::usd::{PrimPredicate, Stage};

use thyllore_anim_core::{
    AnimationClip, AnimationSystem, Keyframe, MorphAnimationSystem, Skeleton, SkinData,
    TransformChannel,
};
use thyllore_math_core::{Vec2, Vec3, Vec4};
use thyllore_model_core::mesh::{Vertex, VertexData};

const MAX_SAMPLED_FRAMES: i64 = 100_000;

#[derive(Clone, Debug)]
pub struct UsdNodeInfo {
    pub index: usize,
    pub name: String,
    pub parent_index: Option<usize>,
    pub local_transform: Matrix4<f32>,
}

pub struct UsdMeshData {
    pub vertex_data: VertexData,
    pub skin_data: Option<SkinData>,
    pub skeleton_id: Option<u32>,
    pub node_index: Option<usize>,
    pub local_vertices: Vec<Vertex>,
    pub base_positions: Vec<[f32; 3]>,
}

pub struct UsdLoadResult {
    pub meshes: Vec<UsdMeshData>,
    pub nodes: Vec<UsdNodeInfo>,
    pub animation_system: AnimationSystem,
    pub clips: Vec<AnimationClip>,
    pub morph_animation: MorphAnimationSystem,
    pub has_skinned_meshes: bool,
    pub has_armature: bool,
}

pub fn load_usd_file(path: &str) -> Result<UsdLoadResult> {
    let stage = Stage::open(path).map_err(|e| anyhow!("Failed to open USD stage {path}: {e}"))?;
    let prim_paths = collect_prim_paths(&stage)?;

    let mut animation_system = AnimationSystem::new();
    let mut skeleton_id_by_path: HashMap<String, u32> = HashMap::new();
    for skel_path in filter_skeletons(&stage, &prim_paths)? {
        let usd_skel = UsdSkeleton::get(&stage, skel_path.clone())?
            .ok_or_else(|| anyhow!("Skeleton missing at {}", skel_path.as_str()))?;
        let skeleton = build_skeleton(&usd_skel)?;
        let id = animation_system.add_skeleton(skeleton);
        skeleton_id_by_path.insert(skel_path.as_str().to_string(), id);
    }
    let has_armature = !skeleton_id_by_path.is_empty();

    let time_range = read_time_range(&stage);
    let clips = build_clips(&stage, &skeleton_id_by_path, time_range)?;
    let nodes = build_nodes(&animation_system);

    let mut meshes = Vec::new();
    let mut has_skinned_meshes = false;
    for mesh_path in filter_meshes(&stage, &prim_paths)? {
        let mesh_data = build_mesh(&stage, &mesh_path, &skeleton_id_by_path)?;
        has_skinned_meshes |= mesh_data.skin_data.is_some();
        meshes.push(mesh_data);
    }

    log_blend_shape_notice(&stage, &prim_paths)?;
    log_bounds(&meshes);

    Ok(UsdLoadResult {
        meshes,
        nodes,
        animation_system,
        clips,
        morph_animation: MorphAnimationSystem::default(),
        has_skinned_meshes,
        has_armature,
    })
}

fn collect_prim_paths(stage: &Stage) -> Result<Vec<sdf::Path>> {
    let mut paths = Vec::new();
    stage.traverse(PrimPredicate::DEFAULT_PROXIES, |p| paths.push(p.clone()))?;
    Ok(paths)
}

fn filter_skeletons(stage: &Stage, prim_paths: &[sdf::Path]) -> Result<Vec<sdf::Path>> {
    let mut result = Vec::new();
    for path in prim_paths {
        if UsdSkeleton::get(stage, path.clone())?.is_some() {
            result.push(path.clone());
        }
    }
    Ok(result)
}

fn filter_meshes(stage: &Stage, prim_paths: &[sdf::Path]) -> Result<Vec<sdf::Path>> {
    let mut result = Vec::new();
    for path in prim_paths {
        if Mesh::get(stage, path.clone())?.is_some() {
            result.push(path.clone());
        }
    }
    Ok(result)
}

fn build_skeleton(usd_skel: &UsdSkeleton) -> Result<Skeleton> {
    let joints = usd_skel.joints()?;
    let short_names = usd_skel.joint_short_names()?;
    let parents = usd_skel.joint_parent_indices()?;
    let rest = usd_skel.rest_transforms()?;
    let bind = usd_skel.bind_transforms()?;

    let mut skeleton = Skeleton::new("usd_skeleton");
    for i in 0..joints.len() {
        let name = short_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| joints[i].clone());
        let parent_id = parents.get(i).copied().flatten().map(|p| p as u32);
        let bone_id = skeleton.add_bone(&name, parent_id);

        if let Some(bone) = skeleton.get_bone_mut(bone_id) {
            if let Some(local) = rest.get(i) {
                bone.local_transform = convert_matrix(local);
            }
            if let Some(world_bind) = bind.get(i) {
                let bind_matrix = convert_matrix(world_bind);
                bone.inverse_bind_pose = bind_matrix.invert().unwrap_or_else(Matrix4::identity);
            }
        }
    }
    Ok(skeleton)
}

fn build_nodes(animation_system: &AnimationSystem) -> Vec<UsdNodeInfo> {
    let Some(skeleton) = animation_system.skeletons.first() else {
        return Vec::new();
    };

    skeleton
        .bones
        .iter()
        .map(|bone| UsdNodeInfo {
            index: bone.id as usize,
            name: bone.name.clone(),
            parent_index: bone.parent_id.map(|p| p as usize),
            local_transform: bone.local_transform,
        })
        .collect()
}

fn build_clips(
    stage: &Stage,
    skeleton_id_by_path: &HashMap<String, u32>,
    time_range: (f64, f64, f64),
) -> Result<Vec<AnimationClip>> {
    let mut clips = Vec::new();
    for skel_path in skeleton_id_by_path.keys() {
        let skel_path = sdf::path(skel_path)?;
        let Some(usd_skel) = UsdSkeleton::get(stage, skel_path.clone())? else {
            continue;
        };
        let Some(anim_path) = find_animation_source(stage, &skel_path)? else {
            continue;
        };
        let Some(query) = SkelAnimQuery::new(stage, anim_path.clone())? else {
            continue;
        };

        let bone_for_anim_joint = usd_skel.map_anim_joints(query.joint_order())?;
        let clip = sample_clip(
            stage,
            &query,
            &bone_for_anim_joint,
            &leaf_name(&anim_path),
            time_range,
        )?;
        if !clip.channels.is_empty() {
            clips.push(clip);
        }
    }
    Ok(clips)
}

fn find_animation_source(stage: &Stage, skel_path: &sdf::Path) -> Result<Option<sdf::Path>> {
    let Some(binding) = SkelBindingAPI::get(stage, skel_path.clone())? else {
        return Ok(None);
    };
    binding.inherited_animation_source()
}

fn sample_clip(
    stage: &Stage,
    query: &SkelAnimQuery,
    bone_for_anim_joint: &[Option<usize>],
    name: &str,
    (start, end, time_codes_per_second): (f64, f64, f64),
) -> Result<AnimationClip> {
    let mut clip = AnimationClip::new(name);
    clip.duration = ((end - start).max(0.0) / time_codes_per_second) as f32;

    let frame_count = ((end - start).round() as i64).clamp(0, MAX_SAMPLED_FRAMES);
    let mut channels: HashMap<u32, TransformChannel> = HashMap::new();

    for step in 0..=frame_count {
        let time_code = start + step as f64;
        let time_seconds = ((time_code - start) / time_codes_per_second) as f32;
        let (translations, rotations, scales) =
            query.compute_joint_local_transform_components(stage, time_code)?;

        for (anim_index, bone) in bone_for_anim_joint.iter().enumerate() {
            let Some(bone_index) = bone else { continue };
            let channel = channels.entry(*bone_index as u32).or_default();

            if let Some(t) = translations.get(anim_index) {
                channel
                    .translation
                    .push(Keyframe::new(time_seconds, Vector3::new(t.x, t.y, t.z)));
            }
            if let Some(r) = rotations.get(anim_index) {
                channel.rotation.push(Keyframe::new(
                    time_seconds,
                    Quaternion::new(r.w, r.x, r.y, r.z),
                ));
            }
            if let Some(s) = scales.get(anim_index) {
                channel
                    .scale
                    .push(Keyframe::new(time_seconds, Vector3::new(s.x, s.y, s.z)));
            }
        }
    }

    for (bone_id, channel) in channels {
        clip.add_channel(bone_id, channel);
    }
    Ok(clip)
}

fn build_mesh(
    stage: &Stage,
    mesh_path: &sdf::Path,
    skeleton_id_by_path: &HashMap<String, u32>,
) -> Result<UsdMeshData> {
    let mesh = Mesh::get(stage, mesh_path.clone())?
        .ok_or_else(|| anyhow!("Mesh missing at {}", mesh_path.as_str()))?;

    let points = read_vec3f_array(mesh.points_attr().get::<sdf::Value>()?);
    let normals = read_vec3f_array(mesh.normals_attr().get::<sdf::Value>()?);
    let uvs = read_uvs(&mesh)?;
    let counts = read_int_array(mesh.face_vertex_counts_attr().get::<sdf::Value>()?);
    let face_indices = read_int_array(mesh.face_vertex_indices_attr().get::<sdf::Value>()?);

    let vertices = build_vertices(&points, &normals, &uvs);
    let indices = triangulate(&counts, &face_indices);
    let local_vertices = vertices.clone();
    let base_positions: Vec<[f32; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();

    let skin_data = build_skin_data(stage, mesh_path, skeleton_id_by_path, &points, &normals)?;
    let skeleton_id = skin_data.as_ref().map(|s| s.skeleton_id);

    Ok(UsdMeshData {
        vertex_data: VertexData { vertices, indices },
        skin_data,
        skeleton_id,
        node_index: None,
        local_vertices,
        base_positions,
    })
}

fn build_vertices(points: &[Vec3f], normals: &[Vec3f], uvs: &[Vec2f]) -> Vec<Vertex> {
    points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let normal = normals.get(i).copied().unwrap_or(Vec3f {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            });
            let uv = uvs.get(i).copied().unwrap_or(Vec2f { x: 0.0, y: 0.0 });
            Vertex {
                pos: Vec3::new(p.x, p.y, p.z),
                color: Vec4::new(1.0, 1.0, 1.0, 1.0),
                tex_coord: Vec2::new(uv.x, uv.y),
                normal: Vec3::new(normal.x, normal.y, normal.z),
            }
        })
        .collect()
}

fn triangulate(counts: &[i32], face_indices: &[i32]) -> Vec<u32> {
    let mut indices = Vec::new();
    let mut offset = 0usize;

    for &count in counts {
        let polygon_size = count.max(0) as usize;
        if polygon_size >= 3 && offset + polygon_size <= face_indices.len() {
            let first = face_indices[offset] as u32;
            for k in 1..polygon_size - 1 {
                indices.push(first);
                indices.push(face_indices[offset + k] as u32);
                indices.push(face_indices[offset + k + 1] as u32);
            }
        }
        offset += polygon_size;
    }
    indices
}

fn build_skin_data(
    stage: &Stage,
    mesh_path: &sdf::Path,
    skeleton_id_by_path: &HashMap<String, u32>,
    points: &[Vec3f],
    normals: &[Vec3f],
) -> Result<Option<SkinData>> {
    let Some(binding) = SkelBindingAPI::get(stage, mesh_path.clone())? else {
        return Ok(None);
    };

    let joint_indices = binding.joint_indices()?;
    let joint_weights = binding.joint_weights()?;
    if joint_indices.is_empty() || joint_weights.is_empty() {
        return Ok(None);
    }

    let Some(skel_path) = binding.inherited_skeleton()? else {
        return Ok(None);
    };
    let skel_path = skel_path.as_str().to_string();
    let Some(&skeleton_id) = skeleton_id_by_path.get(&skel_path) else {
        return Ok(None);
    };

    let remap = build_joint_remap(stage, &binding, &skel_path)?;
    let influences = binding.elements_per_element()?.max(1) as usize;
    let is_constant = matches!(binding.interpolation()?, InfluenceInterpolation::Constant);

    let (bone_indices, bone_weights) = assemble_influences(
        &joint_indices,
        &joint_weights,
        &remap,
        influences,
        is_constant,
        points.len(),
    );

    let base_positions = points.iter().map(|p| Vector3::new(p.x, p.y, p.z)).collect();
    let base_normals = if normals.len() == points.len() {
        normals
            .iter()
            .map(|n| Vector3::new(n.x, n.y, n.z))
            .collect()
    } else {
        Vec::new()
    };

    Ok(Some(SkinData {
        skeleton_id,
        bone_indices,
        bone_weights,
        base_positions,
        base_normals,
    }))
}

fn build_joint_remap(stage: &Stage, binding: &SkelBindingAPI, skel_path: &str) -> Result<Vec<u32>> {
    let subset = binding.joint_subset()?;
    if subset.is_empty() {
        return Ok(Vec::new());
    }

    let Some(usd_skel) = UsdSkeleton::get(stage, sdf::path(skel_path)?)? else {
        return Ok(Vec::new());
    };
    let mapped = usd_skel.map_anim_joints(&subset)?;
    Ok(mapped
        .into_iter()
        .map(|index| index.map(|i| i as u32).unwrap_or(0))
        .collect())
}

fn assemble_influences(
    joint_indices: &[i32],
    joint_weights: &[f32],
    remap: &[u32],
    influences: usize,
    is_constant: bool,
    vertex_count: usize,
) -> (Vec<Vector4<u32>>, Vec<Vector4<f32>>) {
    let mut bone_indices = vec![Vector4::new(0, 0, 0, 0); vertex_count];
    let mut bone_weights = vec![Vector4::new(0.0, 0.0, 0.0, 0.0); vertex_count];

    for vertex in 0..vertex_count {
        let base = if is_constant { 0 } else { vertex * influences };
        let mut entries: Vec<(u32, f32)> = Vec::new();

        for k in 0..influences {
            let index = base + k;
            if index >= joint_indices.len() || index >= joint_weights.len() {
                break;
            }
            let raw = joint_indices[index].max(0) as usize;
            let bone = if remap.is_empty() {
                raw as u32
            } else {
                remap.get(raw).copied().unwrap_or(0)
            };
            let weight = joint_weights[index];
            if weight > 0.0 {
                entries.push((bone, weight));
            }
        }

        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut indices = [0u32; 4];
        let mut weights = [0.0f32; 4];
        let mut total = 0.0;
        for (i, &(bone, weight)) in entries.iter().take(4).enumerate() {
            indices[i] = bone;
            weights[i] = weight;
            total += weight;
        }
        if total > 0.0 {
            for weight in &mut weights {
                *weight /= total;
            }
        }

        bone_indices[vertex] = Vector4::new(indices[0], indices[1], indices[2], indices[3]);
        bone_weights[vertex] = Vector4::new(weights[0], weights[1], weights[2], weights[3]);
    }

    (bone_indices, bone_weights)
}

fn read_uvs(mesh: &Mesh) -> Result<Vec<Vec2f>> {
    let value = mesh.attribute("primvars:st").get::<sdf::Value>()?;
    Ok(value
        .and_then(|v| v.try_as_vec_2f_vec())
        .unwrap_or_default())
}

fn read_vec3f_array(value: Option<sdf::Value>) -> Vec<Vec3f> {
    value
        .and_then(|v| v.try_as_vec_3f_vec())
        .unwrap_or_default()
}

fn read_int_array(value: Option<sdf::Value>) -> Vec<i32> {
    value.and_then(|v| v.try_as_int_vec()).unwrap_or_default()
}

fn read_time_range(stage: &Stage) -> (f64, f64, f64) {
    let time_codes_per_second = read_stage_double(stage, "timeCodesPerSecond").unwrap_or(24.0);
    let start = read_stage_double(stage, "startTimeCode").unwrap_or(0.0);
    let end = read_stage_double(stage, "endTimeCode").unwrap_or(start);
    let rate = if time_codes_per_second > 0.0 {
        time_codes_per_second
    } else {
        24.0
    };
    (start, end, rate)
}

fn read_stage_double(stage: &Stage, field: &str) -> Option<f64> {
    match stage.stage_metadata(field).ok().flatten() {
        Some(sdf::Value::Double(value)) => Some(value),
        Some(sdf::Value::Float(value)) => Some(value as f64),
        Some(sdf::Value::Int(value)) => Some(value as f64),
        _ => None,
    }
}

fn convert_matrix(matrix: &Matrix4d) -> Matrix4<f32> {
    let m = matrix.0;
    Matrix4::new(
        m[0] as f32,
        m[1] as f32,
        m[2] as f32,
        m[3] as f32,
        m[4] as f32,
        m[5] as f32,
        m[6] as f32,
        m[7] as f32,
        m[8] as f32,
        m[9] as f32,
        m[10] as f32,
        m[11] as f32,
        m[12] as f32,
        m[13] as f32,
        m[14] as f32,
        m[15] as f32,
    )
}

fn leaf_name(path: &sdf::Path) -> String {
    path.as_str()
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("usd_clip")
        .to_string()
}

fn log_blend_shape_notice(stage: &Stage, prim_paths: &[sdf::Path]) -> Result<()> {
    let mut blend_shape_count = 0;
    for path in prim_paths {
        if BlendShape::get(stage, path.clone())?.is_some() {
            blend_shape_count += 1;
        }
    }
    if blend_shape_count > 0 {
        log!(
            "USD import: detected {} BlendShape prim(s); morph import is not yet implemented",
            blend_shape_count
        );
    }
    Ok(())
}

fn log_bounds(meshes: &[UsdMeshData]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    let mut total_vertices = 0;

    for mesh in meshes {
        for vertex in &mesh.vertex_data.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex.pos[axis]);
                max[axis] = max[axis].max(vertex.pos[axis]);
            }
            total_vertices += 1;
        }
    }

    if total_vertices > 0 {
        let size_x = max[0] - min[0];
        let size_y = max[1] - min[1];
        let size_z = max[2] - min[2];
        log!(
            "USD Scale Info: {} vertices, size=({:.4}, {:.4}, {:.4}), max={:.4}m",
            total_vertices,
            size_x,
            size_y,
            size_z,
            size_x.max(size_y).max(size_z)
        );
    }
}
