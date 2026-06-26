use std::cmp::Ordering;
use std::collections::HashMap;

use anyhow::{anyhow, Result};
use cgmath::{Matrix4, Quaternion, SquareMatrix, Vector3, Vector4};

use openusd::gf::{Matrix4d, Vec2f, Vec3f};
use openusd::schemas::geom::{Interpolation, Mesh, PointBased};
use openusd::schemas::skel::{
    BlendShape, InfluenceInterpolation, SkelAnimQuery, SkelBindingAPI, Skeleton as UsdSkeleton,
};
use openusd::sdf;
use openusd::usd::{Attribute, PrimPredicate, Stage};

use thyllore_anim_core::{
    AnimationClip, AnimationSystem, Keyframe, MorphAnimation, MorphAnimationSystem, MorphTarget,
    Skeleton, SkinData, TransformChannel,
};
use thyllore_math_core::{Vec2, Vec3, Vec4};
use thyllore_model_core::mesh::{Vertex, VertexData};

use super::strands::{collect_strands, UsdStrandData};

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
    pub point_for_vertex: Vec<u32>,
    pub point_count: usize,
}

pub struct UsdLoadResult {
    pub meshes: Vec<UsdMeshData>,
    pub strands: Vec<UsdStrandData>,
    pub nodes: Vec<UsdNodeInfo>,
    pub animation_system: AnimationSystem,
    pub clips: Vec<AnimationClip>,
    pub morph_animation: MorphAnimationSystem,
    pub has_skinned_meshes: bool,
    pub has_armature: bool,
    pub start_time_code: f32,
    pub time_codes_per_second: f32,
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

    let mesh_paths = filter_meshes(&stage, &prim_paths)?;
    let mut meshes = Vec::new();
    let mut has_skinned_meshes = false;
    for mesh_path in &mesh_paths {
        let mesh_data = build_mesh(&stage, mesh_path, &skeleton_id_by_path)?;
        has_skinned_meshes |= mesh_data.skin_data.is_some();
        meshes.push(mesh_data);
    }

    let morph_animation = build_morph_animation(
        &stage,
        &mesh_paths,
        &meshes,
        &skeleton_id_by_path,
        time_range,
    )?;
    log_bounds(&meshes);

    let strands = collect_strands(&stage)?;
    log_strands(&strands);

    let (start_time_code, _, time_codes_per_second) = time_range;

    Ok(UsdLoadResult {
        meshes,
        strands,
        nodes,
        animation_system,
        clips,
        morph_animation,
        has_skinned_meshes,
        has_armature,
        start_time_code: start_time_code as f32,
        time_codes_per_second: time_codes_per_second as f32,
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
    let uv_indices = read_int_array(mesh.attribute("primvars:st:indices").get::<sdf::Value>()?);
    let counts = read_int_array(mesh.face_vertex_counts_attr().get::<sdf::Value>()?);
    let face_indices = read_int_array(mesh.face_vertex_indices_attr().get::<sdf::Value>()?);

    let corner_count = face_indices.len();
    let normal_mapping = resolve_mapping(
        read_interpolation(&mesh.normals_attr()),
        normals.len(),
        points.len(),
        counts.len(),
        corner_count,
    );
    let uv_element_count = if uv_indices.is_empty() {
        uvs.len()
    } else {
        uv_indices.len()
    };
    let uv_mapping = resolve_mapping(
        read_interpolation(&mesh.attribute("primvars:st")),
        uv_element_count,
        points.len(),
        counts.len(),
        corner_count,
    );

    let expanded = expand_mesh(MeshSource {
        points: &points,
        normals: &normals,
        normal_mapping,
        uvs: &uvs,
        uv_indices: &uv_indices,
        uv_mapping,
        counts: &counts,
        face_indices: &face_indices,
    });

    let base_positions = expanded
        .point_for_vertex
        .iter()
        .map(|&point| {
            let p = points[point as usize];
            [p.x, p.y, p.z]
        })
        .collect();
    let local_vertices = expanded.vertices.clone();

    let skin_data = build_skin_data(
        stage,
        mesh_path,
        skeleton_id_by_path,
        &points,
        &expanded.point_for_vertex,
        &expanded.vertices,
    )?;
    let skeleton_id = skin_data.as_ref().map(|s| s.skeleton_id);

    Ok(UsdMeshData {
        vertex_data: VertexData {
            vertices: expanded.vertices,
            indices: expanded.indices,
        },
        skin_data,
        skeleton_id,
        node_index: None,
        local_vertices,
        base_positions,
        point_for_vertex: expanded.point_for_vertex,
        point_count: points.len(),
    })
}

#[derive(Clone, Copy, PartialEq)]
enum PrimvarMapping {
    Constant,
    Uniform,
    PerPoint,
    PerCorner,
}

impl PrimvarMapping {
    fn element_index(self, face: usize, corner: usize, point: usize) -> usize {
        match self {
            PrimvarMapping::Constant => 0,
            PrimvarMapping::Uniform => face,
            PrimvarMapping::PerPoint => point,
            PrimvarMapping::PerCorner => corner,
        }
    }
}

fn read_interpolation(attr: &Attribute) -> Option<Interpolation> {
    attr.get_metadata::<String>("interpolation")
        .ok()
        .flatten()
        .and_then(|token| Interpolation::from_token(&token))
}

fn resolve_mapping(
    authored: Option<Interpolation>,
    element_count: usize,
    point_count: usize,
    face_count: usize,
    corner_count: usize,
) -> PrimvarMapping {
    if let Some(interpolation) = authored {
        return match interpolation {
            Interpolation::Constant => PrimvarMapping::Constant,
            Interpolation::Uniform => PrimvarMapping::Uniform,
            Interpolation::Varying | Interpolation::Vertex => PrimvarMapping::PerPoint,
            Interpolation::FaceVarying => PrimvarMapping::PerCorner,
        };
    }

    if element_count == point_count {
        PrimvarMapping::PerPoint
    } else if element_count == corner_count {
        PrimvarMapping::PerCorner
    } else if element_count == face_count {
        PrimvarMapping::Uniform
    } else if element_count <= 1 {
        PrimvarMapping::Constant
    } else {
        PrimvarMapping::PerPoint
    }
}

struct MeshSource<'a> {
    points: &'a [Vec3f],
    normals: &'a [Vec3f],
    normal_mapping: PrimvarMapping,
    uvs: &'a [Vec2f],
    uv_indices: &'a [i32],
    uv_mapping: PrimvarMapping,
    counts: &'a [i32],
    face_indices: &'a [i32],
}

struct ExpandedMesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    point_for_vertex: Vec<u32>,
}

#[derive(PartialEq, Eq, Hash)]
struct VertexKey {
    point: u32,
    normal: [u32; 3],
    uv: [u32; 2],
}

impl VertexKey {
    fn new(point: u32, normal: Vec3f, uv: Vec2f) -> Self {
        Self {
            point,
            normal: [normal.x.to_bits(), normal.y.to_bits(), normal.z.to_bits()],
            uv: [uv.x.to_bits(), uv.y.to_bits()],
        }
    }
}

const NO_INDICES: &[i32] = &[];

fn expand_mesh(source: MeshSource) -> ExpandedMesh {
    let default_normal = Vec3f {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    let default_uv = Vec2f { x: 0.0, y: 0.0 };

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut point_for_vertex = Vec::new();
    let mut vertex_for_key: HashMap<VertexKey, u32> = HashMap::new();

    let mut offset = 0usize;
    for (face, &count) in source.counts.iter().enumerate() {
        let polygon_size = count.max(0) as usize;
        if polygon_size < 3 || offset + polygon_size > source.face_indices.len() {
            offset += polygon_size;
            continue;
        }

        let mut corner_vertices = Vec::with_capacity(polygon_size);
        for k in 0..polygon_size {
            let corner = offset + k;
            let point = source.face_indices[corner].max(0) as usize;

            let normal = sample_attribute(
                source.normals,
                NO_INDICES,
                source.normal_mapping.element_index(face, corner, point),
                default_normal,
            );
            let uv = sample_attribute(
                source.uvs,
                source.uv_indices,
                source.uv_mapping.element_index(face, corner, point),
                default_uv,
            );

            let key = VertexKey::new(point as u32, normal, uv);
            let vertex_id = *vertex_for_key.entry(key).or_insert_with(|| {
                let id = vertices.len() as u32;
                let position = source.points.get(point).copied().unwrap_or(Vec3f {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                });
                vertices.push(make_vertex(position, normal, uv));
                point_for_vertex.push(point as u32);
                id
            });
            corner_vertices.push(vertex_id);
        }

        for k in 1..polygon_size - 1 {
            indices.push(corner_vertices[0]);
            indices.push(corner_vertices[k]);
            indices.push(corner_vertices[k + 1]);
        }
        offset += polygon_size;
    }

    ExpandedMesh {
        vertices,
        indices,
        point_for_vertex,
    }
}

fn sample_attribute<T: Copy>(values: &[T], indices: &[i32], element: usize, fallback: T) -> T {
    let resolved = if indices.is_empty() {
        element
    } else {
        indices
            .get(element)
            .map(|&i| i.max(0) as usize)
            .unwrap_or(0)
    };
    values.get(resolved).copied().unwrap_or(fallback)
}

fn make_vertex(position: Vec3f, normal: Vec3f, uv: Vec2f) -> Vertex {
    Vertex {
        pos: Vec3::new(position.x, position.y, position.z),
        color: Vec4::new(1.0, 1.0, 1.0, 1.0),
        tex_coord: Vec2::new(uv.x, uv.y),
        normal: Vec3::new(normal.x, normal.y, normal.z),
    }
}

fn build_skin_data(
    stage: &Stage,
    mesh_path: &sdf::Path,
    skeleton_id_by_path: &HashMap<String, u32>,
    points: &[Vec3f],
    point_for_vertex: &[u32],
    vertices: &[Vertex],
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

    let (point_indices, point_weights) = assemble_influences(
        &joint_indices,
        &joint_weights,
        &remap,
        influences,
        is_constant,
        points.len(),
    );

    let bone_indices = point_for_vertex
        .iter()
        .map(|&point| point_indices[point as usize])
        .collect();
    let bone_weights = point_for_vertex
        .iter()
        .map(|&point| point_weights[point as usize])
        .collect();

    let base_positions = point_for_vertex
        .iter()
        .map(|&point| {
            let p = points[point as usize];
            Vector3::new(p.x, p.y, p.z)
        })
        .collect();
    let base_normals = vertices
        .iter()
        .map(|v| Vector3::new(v.normal[0], v.normal[1], v.normal[2]))
        .collect();

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

struct MorphWeightTrack {
    order: Vec<String>,
    animations: Vec<MorphAnimation>,
}

fn build_morph_animation(
    stage: &Stage,
    mesh_paths: &[sdf::Path],
    meshes: &[UsdMeshData],
    skeleton_id_by_path: &HashMap<String, u32>,
    time_range: (f64, f64, f64),
) -> Result<MorphAnimationSystem> {
    let track = build_morph_weight_track(stage, skeleton_id_by_path, time_range)?;
    if track.order.is_empty() || track.animations.is_empty() {
        return Ok(MorphAnimationSystem::default());
    }

    let mut targets = Vec::with_capacity(meshes.len());
    let mut base_vertices = Vec::with_capacity(meshes.len());
    let mut morphed_mesh_count = 0;
    for (mesh, mesh_path) in meshes.iter().zip(mesh_paths) {
        let mesh_targets = build_mesh_blend_shapes(
            stage,
            mesh_path,
            &track.order,
            &mesh.point_for_vertex,
            mesh.point_count,
        )?;
        if !mesh_targets.is_empty() {
            morphed_mesh_count += 1;
        }
        targets.push(mesh_targets);
        base_vertices.push(mesh.base_positions.clone());
    }

    if morphed_mesh_count == 0 {
        return Ok(MorphAnimationSystem::default());
    }

    log!(
        "USD import: morph imported for {} mesh(es), {} blend shape(s), {} weight keyframe(s)",
        morphed_mesh_count,
        track.order.len(),
        track.animations.len()
    );

    Ok(MorphAnimationSystem {
        animations: track.animations,
        targets,
        base_vertices,
        scale_factor: 1.0,
    })
}

fn build_morph_weight_track(
    stage: &Stage,
    skeleton_id_by_path: &HashMap<String, u32>,
    time_range: (f64, f64, f64),
) -> Result<MorphWeightTrack> {
    let mut selected: Option<SkelAnimQuery> = None;
    let mut ignored_sources = 0;
    for skel_path in skeleton_id_by_path.keys() {
        let skel_path = sdf::path(skel_path)?;
        let Some(anim_path) = find_animation_source(stage, &skel_path)? else {
            continue;
        };
        let Some(query) = SkelAnimQuery::new(stage, anim_path)? else {
            continue;
        };
        if query.blend_shape_order().is_empty() {
            continue;
        }
        if selected.is_none() {
            selected = Some(query);
        } else {
            ignored_sources += 1;
        }
    }

    let Some(query) = selected else {
        return Ok(MorphWeightTrack {
            order: Vec::new(),
            animations: Vec::new(),
        });
    };
    if ignored_sources > 0 {
        log!(
            "USD import: {} blend shape animation source(s) ignored (engine supports one global morph track)",
            ignored_sources
        );
    }

    let order = query.blend_shape_order().to_vec();
    let animations = sample_blend_shape_weights(stage, &query, time_range)?;
    Ok(MorphWeightTrack { order, animations })
}

fn sample_blend_shape_weights(
    stage: &Stage,
    query: &SkelAnimQuery,
    (start, end, time_codes_per_second): (f64, f64, f64),
) -> Result<Vec<MorphAnimation>> {
    let frame_count = if query.blend_shape_weights_might_be_time_varying() {
        ((end - start).round() as i64).clamp(0, MAX_SAMPLED_FRAMES)
    } else {
        0
    };

    let mut animations = Vec::with_capacity(frame_count as usize + 1);
    for step in 0..=frame_count {
        let time_code = start + step as f64;
        let time_seconds = ((time_code - start) / time_codes_per_second) as f32;
        let weights = query.compute_blend_shape_weights(stage, time_code)?;
        animations.push(MorphAnimation {
            key_frame: time_seconds,
            weights,
        });
    }
    Ok(animations)
}

fn build_mesh_blend_shapes(
    stage: &Stage,
    mesh_path: &sdf::Path,
    order: &[String],
    point_for_vertex: &[u32],
    point_count: usize,
) -> Result<Vec<MorphTarget>> {
    let Some(binding) = SkelBindingAPI::get(stage, mesh_path.clone())? else {
        return Ok(Vec::new());
    };
    let names = binding.blend_shapes()?;
    let target_paths = binding.blend_shape_targets()?;
    if names.is_empty() || target_paths.is_empty() {
        return Ok(Vec::new());
    }

    let slot_for_name: HashMap<&str, usize> = order
        .iter()
        .enumerate()
        .map(|(slot, name)| (name.as_str(), slot))
        .collect();

    let mut targets = vec![MorphTarget::default(); order.len()];
    let mut matched = 0;
    for (name, target_path) in names.iter().zip(&target_paths) {
        let Some(&slot) = slot_for_name.get(name.as_str()) else {
            continue;
        };
        let Some(blend_shape) = BlendShape::get(stage, target_path.clone())? else {
            continue;
        };
        targets[slot] = build_morph_target(&blend_shape, point_for_vertex, point_count)?;
        matched += 1;
    }

    if matched == 0 {
        return Ok(Vec::new());
    }
    Ok(targets)
}

fn build_morph_target(
    blend_shape: &BlendShape,
    point_for_vertex: &[u32],
    point_count: usize,
) -> Result<MorphTarget> {
    let offsets = blend_shape.offsets()?;
    let normal_offsets = blend_shape.normal_offsets()?;
    let point_indices = blend_shape.point_indices()?;

    let positions = expand_offsets(&offsets, &point_indices, point_for_vertex, point_count);
    let normals = if normal_offsets.is_empty() {
        Vec::new()
    } else {
        expand_offsets(
            &normal_offsets,
            &point_indices,
            point_for_vertex,
            point_count,
        )
    };

    Ok(MorphTarget {
        positions,
        normals,
        tangents: Vec::new(),
    })
}

fn expand_offsets(
    offsets: &[Vec3f],
    point_indices: &[i32],
    point_for_vertex: &[u32],
    point_count: usize,
) -> Vec<[f32; 3]> {
    let mut per_point = vec![[0.0f32; 3]; point_count];

    if point_indices.is_empty() {
        for (i, offset) in offsets.iter().enumerate().take(point_count) {
            per_point[i] = [offset.x, offset.y, offset.z];
        }
    } else {
        for (slot, &point_index) in point_indices.iter().enumerate() {
            let i = point_index.max(0) as usize;
            if i < point_count && slot < offsets.len() {
                per_point[i] = [offsets[slot].x, offsets[slot].y, offsets[slot].z];
            }
        }
    }

    point_for_vertex
        .iter()
        .map(|&point| per_point[point as usize])
        .collect()
}

fn log_strands(strands: &[UsdStrandData]) {
    if strands.is_empty() {
        return;
    }
    let total_curves: usize = strands.iter().map(|s| s.curve_count()).sum();
    let total_points: usize = strands.iter().map(|s| s.point_count()).sum();
    log!(
        "USD import: {} strand set(s), {} curve(s), {} control point(s)",
        strands.len(),
        total_curves,
        total_points
    );
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
