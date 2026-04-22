use anyhow::{Context, Result};
use cgmath::{Matrix4, Quaternion, SquareMatrix, Vector3};
use std::collections::HashMap;
use thyllore_anim_core::{
    AimConstraintData, BoneId, ConstraintType, IkConstraintData, ParentConstraintData,
    PositionConstraintData, RotationConstraintData, ScaleConstraintData,
};

#[derive(Clone, Debug)]
pub struct LoadedConstraint {
    pub constraint_type: ConstraintType,
    pub priority: u32,
}

#[derive(Clone, Debug)]
pub struct BoneNode {
    pub name: String,
    pub parent: Option<String>,
    pub local_transform: Matrix4<f32>,
    pub default_translation: [f32; 3],
    pub default_rotation: Quaternion<f32>,
    pub default_scaling: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct FbxAxesInfo {
    pub up_axis: i32,
    pub up_axis_sign: i32,
    pub front_axis: i32,
    pub front_axis_sign: i32,
    pub coord_axis: i32,
    pub coord_axis_sign: i32,
}

impl Default for FbxAxesInfo {
    fn default() -> Self {
        Self {
            up_axis: 1,
            up_axis_sign: 1,
            front_axis: 2,
            front_axis_sign: 1,
            coord_axis: 0,
            coord_axis_sign: 1,
        }
    }
}

fn convert_coordinate_axis(axis: ufbx::CoordinateAxis) -> (i32, i32) {
    match axis {
        ufbx::CoordinateAxis::PositiveX => (0, 1),
        ufbx::CoordinateAxis::NegativeX => (0, -1),
        ufbx::CoordinateAxis::PositiveY => (1, 1),
        ufbx::CoordinateAxis::NegativeY => (1, -1),
        ufbx::CoordinateAxis::PositiveZ => (2, 1),
        ufbx::CoordinateAxis::NegativeZ => (2, -1),
        ufbx::CoordinateAxis::Unknown => (1, 1),
    }
}

fn read_axes_from_scene(settings: &ufbx::SceneSettings) -> FbxAxesInfo {
    let (up_axis, up_axis_sign) = convert_coordinate_axis(settings.axes.up);
    let (front_axis, front_axis_sign) = convert_coordinate_axis(settings.axes.front);
    let (coord_axis, coord_axis_sign) = convert_coordinate_axis(settings.axes.right);

    FbxAxesInfo {
        up_axis,
        up_axis_sign,
        front_axis,
        front_axis_sign,
        coord_axis,
        coord_axis_sign,
    }
}

#[derive(Clone, Debug, Default)]
pub struct FbxModel {
    pub fbx_data: Vec<FbxData>,
    pub animations: Vec<FbxAnimation>,
    pub nodes: HashMap<String, BoneNode>,
    pub unit_scale: f32,
    pub fps: f32,
    pub constraints: Vec<LoadedConstraint>,
    pub axes: FbxAxesInfo,
    pub source_path: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FbxAnimation {
    pub name: String,
    pub duration: f32,
    pub bone_animations: HashMap<String, BoneAnimation>,
}

#[derive(Clone, Debug)]
pub struct BoneAnimation {
    pub bone_name: String,
    pub translation_keys: Vec<KeyFrame<[f32; 3]>>,
    pub rotation_keys: Vec<KeyFrame<Quaternion<f32>>>,
    pub scale_keys: Vec<KeyFrame<[f32; 3]>>,
}

#[derive(Clone, Debug)]
pub struct KeyFrame<T> {
    pub time: f32,
    pub value: T,
}

#[derive(Clone, Debug)]
pub struct ClusterInfo {
    pub bone_name: String,
    pub transform: Matrix4<f32>,
    pub transform_link: Matrix4<f32>,
    pub inverse_bind_pose: Matrix4<f32>,
    pub vertex_indices: Vec<usize>,
    pub vertex_weights: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct MeshPart {
    pub mesh_name: String,
    pub local_positions: Vec<Vector3<f32>>,
    pub parent_bone: Option<String>,
    pub local_transform: Matrix4<f32>,
    pub vertex_offset: usize,
    pub vertex_count: usize,
}

#[derive(Clone, Debug)]
pub struct FbxData {
    pub positions: Vec<Vector3<f32>>,
    pub local_positions: Vec<Vector3<f32>>,
    pub normals: Vec<Vector3<f32>>,
    pub local_normals: Vec<Vector3<f32>>,
    pub indices: Vec<u32>,
    pub tex_coords: Vec<[f32; 2]>,
    pub clusters: Vec<ClusterInfo>,
    pub mesh_parts: Vec<MeshPart>,
    pub parent_node: Option<String>,
    pub mesh_node_name: Option<String>,
    pub material_name: Option<String>,
    pub diffuse_texture: Option<String>,
    pub diffuse_color: [f32; 3],
}

impl FbxData {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            local_positions: Vec::new(),
            normals: Vec::new(),
            local_normals: Vec::new(),
            indices: Vec::new(),
            tex_coords: Vec::new(),
            clusters: Vec::new(),
            mesh_parts: Vec::new(),
            parent_node: None,
            mesh_node_name: None,
            material_name: None,
            diffuse_texture: None,
            diffuse_color: [0.8, 0.8, 0.8],
        }
    }
}

fn ufbx_matrix_to_cgmath(m: &ufbx::Matrix) -> Matrix4<f32> {
    Matrix4::new(
        m.m00 as f32,
        m.m10 as f32,
        m.m20 as f32,
        0.0,
        m.m01 as f32,
        m.m11 as f32,
        m.m21 as f32,
        0.0,
        m.m02 as f32,
        m.m12 as f32,
        m.m22 as f32,
        0.0,
        m.m03 as f32,
        m.m13 as f32,
        m.m23 as f32,
        1.0,
    )
}

fn decompose_transform(m: &Matrix4<f32>) -> ([f32; 3], Quaternion<f32>, [f32; 3]) {
    let translation = [m[3][0], m[3][1], m[3][2]];

    let sx = (m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2]).sqrt();
    let sy = (m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2]).sqrt();
    let sz = (m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2]).sqrt();
    let scale = [sx, sy, sz];

    let inv_sx = if sx > 1e-6 { 1.0 / sx } else { 0.0 };
    let inv_sy = if sy > 1e-6 { 1.0 / sy } else { 0.0 };
    let inv_sz = if sz > 1e-6 { 1.0 / sz } else { 0.0 };

    let r00 = m[0][0] * inv_sx;
    let r01 = m[0][1] * inv_sx;
    let r02 = m[0][2] * inv_sx;
    let r10 = m[1][0] * inv_sy;
    let r11 = m[1][1] * inv_sy;
    let r12 = m[1][2] * inv_sy;
    let r20 = m[2][0] * inv_sz;
    let r21 = m[2][1] * inv_sz;
    let r22 = m[2][2] * inv_sz;

    let trace = r00 + r11 + r22;
    let rotation = if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        Quaternion::new(0.25 / s, (r12 - r21) * s, (r20 - r02) * s, (r01 - r10) * s)
    } else if r00 > r11 && r00 > r22 {
        let s = 2.0 * (1.0 + r00 - r11 - r22).sqrt();
        Quaternion::new((r12 - r21) / s, 0.25 * s, (r10 + r01) / s, (r20 + r02) / s)
    } else if r11 > r22 {
        let s = 2.0 * (1.0 + r11 - r00 - r22).sqrt();
        Quaternion::new((r20 - r02) / s, (r10 + r01) / s, 0.25 * s, (r12 + r21) / s)
    } else {
        let s = 2.0 * (1.0 + r22 - r00 - r11).sqrt();
        Quaternion::new((r01 - r10) / s, (r20 + r02) / s, (r12 + r21) / s, 0.25 * s)
    };

    (translation, rotation, scale)
}

pub fn load_fbx_with_ufbx(path: &str) -> Result<FbxModel> {
    log!("=== Loading FBX file with ufbx: {} ===", path);

    let opts = ufbx::LoadOpts {
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        ..Default::default()
    };
    let scene = ufbx::load_file(path, opts)
        .map_err(|e| anyhow::anyhow!("ufbx load failed: {}", e.description))
        .context(format!("Failed to load FBX: {}", path))?;

    let unit_scale = scene.settings.original_unit_meters as f32;
    log!(
        "Scene: original_unit_meters={}, meshes={}, nodes={}, anim_stacks={}, constraints={}",
        unit_scale,
        scene.meshes.len(),
        scene.nodes.len(),
        scene.anim_stacks.len(),
        scene.constraints.len()
    );

    for (idx, c) in scene.constraints.iter().enumerate() {
        log!(
            "  Constraint[{}]: type={}, node={:?}, targets={}",
            idx,
            c.type_name,
            c.node.as_ref().map(|n| n.element.name.to_string()),
            c.targets.len()
        );
    }

    let axes = read_axes_from_scene(&scene.settings);
    log!(
        "FBX axes: up={}(sign={}), front={}(sign={}), coord={}(sign={})",
        axes.up_axis,
        axes.up_axis_sign,
        axes.front_axis,
        axes.front_axis_sign,
        axes.coord_axis,
        axes.coord_axis_sign
    );

    let fps = scene.settings.frames_per_second as f32;

    let mut fbx_model = FbxModel {
        unit_scale,
        fps,
        axes,
        source_path: Some(path.to_string()),
        ..Default::default()
    };

    build_bone_hierarchy(&scene, &mut fbx_model);

    let mesh_to_node = build_mesh_node_mapping(&scene);
    let mut split_infos: Vec<MeshSplitInfo> = Vec::new();

    for (mesh_idx, ufbx_mesh) in scene.meshes.iter().enumerate() {
        let typed_id = ufbx_mesh.element.typed_id as usize;
        let parts = extract_mesh_data_by_material(ufbx_mesh);

        for (fbx_data, vertex_map) in parts {
            split_infos.push(MeshSplitInfo {
                ufbx_mesh_typed_id: typed_id,
                vertex_map,
            });
            fbx_model.fbx_data.push(fbx_data);
        }

        log!(
            "Mesh {}: {} vertices, {} indices, {} materials",
            mesh_idx,
            ufbx_mesh.num_vertices,
            ufbx_mesh.num_indices,
            ufbx_mesh.materials.len()
        );
    }

    extract_skin_data(&scene, &mut fbx_model, &split_infos);
    extract_animations(&scene, &mut fbx_model);

    let bone_name_to_id = build_bone_name_to_id(&fbx_model.nodes);
    fbx_model.constraints = extract_constraints(&scene, &bone_name_to_id);

    assign_mesh_parent_nodes(&mut fbx_model, &mesh_to_node, &split_infos);

    log!(
        "=== FBX loading complete: {} meshes, {} animations, {} constraints ===",
        fbx_model.fbx_data.len(),
        fbx_model.animations.len(),
        fbx_model.constraints.len()
    );

    Ok(fbx_model)
}

struct MeshSplitInfo {
    ufbx_mesh_typed_id: usize,
    vertex_map: HashMap<u32, u32>,
}

struct MaterialPart {
    positions: Vec<Vector3<f32>>,
    local_positions: Vec<Vector3<f32>>,
    normals: Vec<Vector3<f32>>,
    local_normals: Vec<Vector3<f32>>,
    tex_coords: Vec<[f32; 2]>,
    indices: Vec<u32>,
    vertex_map: HashMap<u32, u32>,
}

impl MaterialPart {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            local_positions: Vec::new(),
            normals: Vec::new(),
            local_normals: Vec::new(),
            tex_coords: Vec::new(),
            indices: Vec::new(),
            vertex_map: HashMap::new(),
        }
    }
}

fn extract_mesh_data_by_material(mesh: &ufbx::Mesh) -> Vec<(FbxData, HashMap<u32, u32>)> {
    let num_materials = mesh.materials.len().max(1);
    let mut parts: Vec<MaterialPart> = (0..num_materials).map(|_| MaterialPart::new()).collect();

    let max_tris = mesh.max_face_triangles;
    let mut tri_indices = vec![0u32; max_tris * 3];

    for face_idx in 0..mesh.faces.len() {
        let mat_idx = if !mesh.face_material.is_empty() {
            (mesh.face_material[face_idx] as usize).min(num_materials - 1)
        } else {
            0
        };

        let face = mesh.faces[face_idx];
        let num_tris = mesh.triangulate_face(&mut tri_indices, face);
        let num_corners = (num_tris * 3) as usize;

        let part = &mut parts[mat_idx];

        for &idx in &tri_indices[..num_corners] {
            let uidx = idx as usize;
            let ctrl_idx = mesh.vertex_indices[uidx];
            let next_id = part.vertex_map.len() as u32;
            let mapped = *part.vertex_map.entry(ctrl_idx).or_insert(next_id);

            if mapped == next_id {
                let pos = mesh.vertex_position[uidx];
                let v = Vector3::new(pos.x as f32, pos.y as f32, pos.z as f32);
                part.positions.push(v);
                part.local_positions.push(v);

                if mesh.vertex_normal.exists {
                    let n = mesh.vertex_normal[uidx];
                    let normal = Vector3::new(n.x as f32, n.y as f32, n.z as f32);
                    part.normals.push(normal);
                    part.local_normals.push(normal);
                } else {
                    part.normals.push(Vector3::new(0.0, 1.0, 0.0));
                    part.local_normals.push(Vector3::new(0.0, 1.0, 0.0));
                }

                if mesh.vertex_uv.exists {
                    let uv = mesh.vertex_uv[uidx];
                    part.tex_coords.push([uv.x as f32, 1.0 - uv.y as f32]);
                } else {
                    part.tex_coords.push([0.5, 0.5]);
                }
            }

            part.indices.push(mapped);
        }
    }

    let mut results = Vec::new();

    for (mat_idx, part) in parts.into_iter().enumerate() {
        if part.indices.is_empty() {
            continue;
        }

        let mut fbx_data = FbxData::new();
        fbx_data.positions = part.positions;
        fbx_data.local_positions = part.local_positions;
        fbx_data.normals = part.normals;
        fbx_data.local_normals = part.local_normals;
        fbx_data.tex_coords = part.tex_coords;
        fbx_data.indices = part.indices;

        if mat_idx < mesh.materials.len() {
            let mat = &mesh.materials[mat_idx];
            fbx_data.material_name = Some(mat.element.name.to_string());
            fbx_data.diffuse_texture = extract_texture_path(mat);
            let dc = &mat.fbx.diffuse_color.value_vec4;
            fbx_data.diffuse_color = [dc.x as f32, dc.y as f32, dc.z as f32];
        }

        results.push((fbx_data, part.vertex_map));
    }

    results
}

fn extract_texture_path(mat: &ufbx::Material) -> Option<String> {
    if let Some(ref tex) = mat.pbr.base_color.texture {
        let filename = tex.filename.to_string();
        if !filename.is_empty() {
            return Some(filename);
        }
    }

    if let Some(ref tex) = mat.fbx.diffuse_color.texture {
        let filename = tex.filename.to_string();
        if !filename.is_empty() {
            return Some(filename);
        }
    }

    None
}

fn extract_skin_data(scene: &ufbx::Scene, fbx_model: &mut FbxModel, split_infos: &[MeshSplitInfo]) {
    for ufbx_mesh in &scene.meshes {
        if ufbx_mesh.skin_deformers.is_empty() {
            continue;
        }

        let typed_id = ufbx_mesh.element.typed_id as usize;
        let skin = &ufbx_mesh.skin_deformers[0];

        for cluster in &skin.clusters {
            let bone_name = cluster
                .bone_node
                .as_ref()
                .map(|n| n.element.name.to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            let geometry_to_bone = ufbx_matrix_to_cgmath(&cluster.geometry_to_bone);

            let transform_link = geometry_to_bone.invert().unwrap_or(Matrix4::identity());

            for (fbx_idx, info) in split_infos.iter().enumerate() {
                if info.ufbx_mesh_typed_id != typed_id {
                    continue;
                }

                let mut vertex_indices = Vec::new();
                let mut vertex_weights = Vec::new();

                for i in 0..cluster.num_weights {
                    let ctrl_idx = cluster.vertices[i];
                    let weight = cluster.weights[i] as f32;

                    if let Some(&mapped) = info.vertex_map.get(&ctrl_idx) {
                        vertex_indices.push(mapped as usize);
                        vertex_weights.push(weight);
                    }
                }

                if !vertex_indices.is_empty() {
                    fbx_model.fbx_data[fbx_idx].clusters.push(ClusterInfo {
                        bone_name: bone_name.clone(),
                        transform: Matrix4::identity(),
                        transform_link,
                        inverse_bind_pose: geometry_to_bone,
                        vertex_indices,
                        vertex_weights,
                    });
                }
            }
        }

        log!("Extracted skin data for mesh typed_id={}", typed_id);
    }
}

fn build_bone_hierarchy(scene: &ufbx::Scene, fbx_model: &mut FbxModel) {
    for node in &scene.nodes {
        if node.is_root {
            continue;
        }

        let name = node.element.name.to_string();
        let parent = node
            .parent
            .as_ref()
            .filter(|p| !p.is_root)
            .map(|p| p.element.name.to_string());

        let local_transform = ufbx_matrix_to_cgmath(&node.node_to_parent);

        let (default_translation, default_rotation, default_scaling) =
            decompose_transform(&local_transform);

        let bone_node = BoneNode {
            name: name.clone(),
            parent,
            local_transform,
            default_translation,
            default_rotation,
            default_scaling,
        };

        fbx_model.nodes.insert(name, bone_node);
    }

    log!("Built bone hierarchy with {} nodes", fbx_model.nodes.len());
}

fn build_mesh_node_mapping(scene: &ufbx::Scene) -> HashMap<usize, String> {
    let mut mesh_to_node = HashMap::new();

    for node in &scene.nodes {
        if let Some(ref mesh) = node.mesh {
            let mesh_typed_id = mesh.element.typed_id as usize;
            let node_name = node.element.name.to_string();
            mesh_to_node.insert(mesh_typed_id, node_name);
        }
    }

    mesh_to_node
}

fn extract_animations(scene: &ufbx::Scene, fbx_model: &mut FbxModel) {
    for anim_stack in &scene.anim_stacks {
        let raw_name = anim_stack.element.name.to_string();
        let anim_name = if raw_name.is_empty() {
            "DefaultAnimation".to_string()
        } else {
            raw_name.replace('|', "-")
        };

        log!("Processing AnimStack: {}", anim_name);

        let bake_opts = ufbx::BakeOpts::default();
        let baked = match ufbx::bake_anim(scene, &anim_stack.anim, bake_opts) {
            Ok(b) => b,
            Err(e) => {
                log!(
                    "Failed to bake animation '{}': {}",
                    anim_name,
                    e.description
                );
                continue;
            }
        };

        let duration = (anim_stack.time_end - anim_stack.time_begin) as f32;

        let mut bone_animations = HashMap::new();

        for bake_node in &baked.nodes {
            let node_idx = bake_node.typed_id as usize;
            if node_idx >= scene.nodes.len() {
                continue;
            }
            let node = &scene.nodes[node_idx];
            let bone_name = node.element.name.to_string();

            if bone_name.is_empty() {
                continue;
            }

            let translation_keys: Vec<KeyFrame<[f32; 3]>> = bake_node
                .translation_keys
                .iter()
                .map(|k| KeyFrame {
                    time: k.time as f32,
                    value: [k.value.x as f32, k.value.y as f32, k.value.z as f32],
                })
                .collect();

            let rotation_keys: Vec<KeyFrame<Quaternion<f32>>> = bake_node
                .rotation_keys
                .iter()
                .map(|k| KeyFrame {
                    time: k.time as f32,
                    value: Quaternion::new(
                        k.value.w as f32,
                        k.value.x as f32,
                        k.value.y as f32,
                        k.value.z as f32,
                    ),
                })
                .collect();

            let scale_keys: Vec<KeyFrame<[f32; 3]>> = bake_node
                .scale_keys
                .iter()
                .map(|k| KeyFrame {
                    time: k.time as f32,
                    value: [k.value.x as f32, k.value.y as f32, k.value.z as f32],
                })
                .collect();

            let has_keys =
                translation_keys.len() > 1 || rotation_keys.len() > 1 || scale_keys.len() > 1;

            if has_keys {
                bone_animations.insert(
                    bone_name.clone(),
                    BoneAnimation {
                        bone_name,
                        translation_keys,
                        rotation_keys,
                        scale_keys,
                    },
                );
            }
        }

        log!(
            "AnimStack '{}': duration={:.4}s, {} bone animations",
            anim_name,
            duration,
            bone_animations.len()
        );

        fbx_model.animations.push(FbxAnimation {
            name: anim_name,
            duration,
            bone_animations,
        });
    }
}

fn build_bone_name_to_id(nodes: &HashMap<String, BoneNode>) -> HashMap<String, u32> {
    let mut name_to_id = HashMap::new();
    let mut sorted_names: Vec<&String> = nodes.keys().collect();
    sorted_names.sort();

    for (id, name) in sorted_names.iter().enumerate() {
        name_to_id.insert((*name).clone(), id as u32);
    }

    name_to_id
}

fn extract_constraints(
    scene: &ufbx::Scene,
    bone_name_to_id: &HashMap<String, u32>,
) -> Vec<LoadedConstraint> {
    let mut result = Vec::new();

    for constraint in &scene.constraints {
        let constrained_bone_name = constraint.node.as_ref().map(|n| n.element.name.to_string());

        let constrained_bone_id = constrained_bone_name
            .as_ref()
            .and_then(|name| bone_name_to_id.get(name))
            .copied()
            .unwrap_or(0);

        let weight = constraint.weight as f32;
        let enabled = constraint.active;

        let loaded = match constraint.type_ {
            ufbx::ConstraintType::Position => Some(extract_position_constraint(
                constraint,
                bone_name_to_id,
                constrained_bone_id,
                enabled,
                weight,
            )),
            ufbx::ConstraintType::Rotation => Some(extract_rotation_constraint(
                constraint,
                bone_name_to_id,
                constrained_bone_id,
                enabled,
                weight,
            )),
            ufbx::ConstraintType::Scale => Some(extract_scale_constraint(
                constraint,
                bone_name_to_id,
                constrained_bone_id,
                enabled,
                weight,
            )),
            ufbx::ConstraintType::Parent => Some(extract_parent_constraint(
                constraint,
                bone_name_to_id,
                constrained_bone_id,
                enabled,
                weight,
            )),
            ufbx::ConstraintType::Aim => Some(extract_aim_constraint(
                constraint,
                bone_name_to_id,
                constrained_bone_id,
                enabled,
                weight,
            )),
            ufbx::ConstraintType::SingleChainIk => Some(extract_ik_constraint(
                constraint,
                bone_name_to_id,
                scene,
                constrained_bone_id,
                enabled,
                weight,
            )),
            _ => {
                log!("Unsupported constraint type: {}", constraint.type_name);
                None
            }
        };

        if let Some(loaded_constraint) = loaded {
            log!(
                "Loaded constraint: {:?} on bone '{}'",
                constraint.type_name,
                constrained_bone_name.as_deref().unwrap_or("Unknown")
            );
            result.push(loaded_constraint);
        }
    }

    result
}

fn extract_position_constraint(
    constraint: &ufbx::Constraint,
    bone_name_to_id: &HashMap<String, u32>,
    constrained_bone_id: u32,
    enabled: bool,
    weight: f32,
) -> LoadedConstraint {
    let target_bone_id = resolve_first_target(constraint, bone_name_to_id);
    let affect_axes = [
        constraint.constrain_translation[0],
        constraint.constrain_translation[1],
        constraint.constrain_translation[2],
    ];

    LoadedConstraint {
        constraint_type: ConstraintType::Position(PositionConstraintData {
            constrained_bone: constrained_bone_id,
            target_bone: target_bone_id,
            offset: Vector3::new(0.0, 0.0, 0.0),
            affect_axes,
            enabled,
            weight,
        }),
        priority: ConstraintType::Position(PositionConstraintData::default()).default_priority(),
    }
}

fn extract_rotation_constraint(
    constraint: &ufbx::Constraint,
    bone_name_to_id: &HashMap<String, u32>,
    constrained_bone_id: u32,
    enabled: bool,
    weight: f32,
) -> LoadedConstraint {
    let target_bone_id = resolve_first_target(constraint, bone_name_to_id);
    let affect_axes = [
        constraint.constrain_rotation[0],
        constraint.constrain_rotation[1],
        constraint.constrain_rotation[2],
    ];

    LoadedConstraint {
        constraint_type: ConstraintType::Rotation(RotationConstraintData {
            constrained_bone: constrained_bone_id,
            target_bone: target_bone_id,
            offset: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            affect_axes,
            enabled,
            weight,
        }),
        priority: ConstraintType::Rotation(RotationConstraintData::default()).default_priority(),
    }
}

fn extract_scale_constraint(
    constraint: &ufbx::Constraint,
    bone_name_to_id: &HashMap<String, u32>,
    constrained_bone_id: u32,
    enabled: bool,
    weight: f32,
) -> LoadedConstraint {
    let target_bone_id = resolve_first_target(constraint, bone_name_to_id);
    let affect_axes = [
        constraint.constrain_scale[0],
        constraint.constrain_scale[1],
        constraint.constrain_scale[2],
    ];

    LoadedConstraint {
        constraint_type: ConstraintType::Scale(ScaleConstraintData {
            constrained_bone: constrained_bone_id,
            target_bone: target_bone_id,
            offset: Vector3::new(1.0, 1.0, 1.0),
            affect_axes,
            enabled,
            weight,
        }),
        priority: ConstraintType::Scale(ScaleConstraintData::default()).default_priority(),
    }
}

fn extract_parent_constraint(
    constraint: &ufbx::Constraint,
    bone_name_to_id: &HashMap<String, u32>,
    constrained_bone_id: u32,
    enabled: bool,
    weight: f32,
) -> LoadedConstraint {
    let sources: Vec<(BoneId, f32)> = constraint
        .targets
        .iter()
        .filter_map(|t| {
            let name = t.node.element.name.to_string();
            bone_name_to_id.get(&name).map(|&id| (id, t.weight as f32))
        })
        .collect();

    LoadedConstraint {
        constraint_type: ConstraintType::Parent(ParentConstraintData {
            constrained_bone: constrained_bone_id,
            sources,
            affect_translation: constraint.constrain_translation.iter().any(|&v| v),
            affect_rotation: constraint.constrain_rotation.iter().any(|&v| v),
            enabled,
            weight,
        }),
        priority: ConstraintType::Parent(ParentConstraintData::default()).default_priority(),
    }
}

fn extract_aim_constraint(
    constraint: &ufbx::Constraint,
    bone_name_to_id: &HashMap<String, u32>,
    constrained_bone_id: u32,
    enabled: bool,
    weight: f32,
) -> LoadedConstraint {
    let target_bone_id = resolve_first_target(constraint, bone_name_to_id);

    let aim_axis = Vector3::new(
        constraint.aim_vector.x as f32,
        constraint.aim_vector.y as f32,
        constraint.aim_vector.z as f32,
    );

    let up_axis = Vector3::new(
        constraint.aim_up_vector.x as f32,
        constraint.aim_up_vector.y as f32,
        constraint.aim_up_vector.z as f32,
    );

    let up_target = constraint
        .aim_up_node
        .as_ref()
        .and_then(|n| bone_name_to_id.get(&n.element.name.to_string()).copied());

    LoadedConstraint {
        constraint_type: ConstraintType::Aim(AimConstraintData {
            source_bone: constrained_bone_id,
            target_bone: target_bone_id,
            aim_axis,
            up_axis,
            up_target,
            enabled,
            weight,
        }),
        priority: ConstraintType::Aim(AimConstraintData::default()).default_priority(),
    }
}

fn extract_ik_constraint(
    constraint: &ufbx::Constraint,
    bone_name_to_id: &HashMap<String, u32>,
    scene: &ufbx::Scene,
    constrained_bone_id: u32,
    enabled: bool,
    weight: f32,
) -> LoadedConstraint {
    let target_bone_id = resolve_first_target(constraint, bone_name_to_id);

    let effector_bone = constraint
        .ik_effector
        .as_ref()
        .and_then(|n| bone_name_to_id.get(&n.element.name.to_string()).copied())
        .unwrap_or(constrained_bone_id);

    let pole_vector = Vector3::new(
        constraint.ik_pole_vector.x as f32,
        constraint.ik_pole_vector.y as f32,
        constraint.ik_pole_vector.z as f32,
    );

    let chain_length = compute_ik_chain_length(constraint, scene);

    LoadedConstraint {
        constraint_type: ConstraintType::Ik(IkConstraintData {
            chain_length,
            target_bone: target_bone_id,
            effector_bone,
            pole_vector: Some(pole_vector),
            pole_target: None,
            twist: 0.0,
            enabled,
            weight,
        }),
        priority: ConstraintType::Ik(IkConstraintData::default()).default_priority(),
    }
}

fn resolve_first_target(
    constraint: &ufbx::Constraint,
    bone_name_to_id: &HashMap<String, u32>,
) -> BoneId {
    constraint
        .targets
        .first()
        .and_then(|t| {
            bone_name_to_id
                .get(&t.node.element.name.to_string())
                .copied()
        })
        .unwrap_or(0)
}

fn compute_ik_chain_length(constraint: &ufbx::Constraint, scene: &ufbx::Scene) -> u32 {
    let effector = match &constraint.ik_effector {
        Some(n) => n,
        None => return 2,
    };
    let end_node = match &constraint.ik_end_node {
        Some(n) => n,
        None => return 2,
    };

    let mut count = 0u32;
    let mut current = Some(effector.element.typed_id);
    let end_id = end_node.element.typed_id;

    while let Some(node_id) = current {
        if node_id == end_id {
            break;
        }
        count += 1;
        if count > 10 {
            break;
        }
        let node = &scene.nodes[node_id as usize];
        current = node.parent.as_ref().map(|p| p.element.typed_id);
    }

    count.max(2)
}

fn assign_mesh_parent_nodes(
    fbx_model: &mut FbxModel,
    mesh_to_node: &HashMap<usize, String>,
    split_infos: &[MeshSplitInfo],
) {
    let has_animations = !fbx_model.animations.is_empty();
    let animated_nodes: std::collections::HashSet<String> = if has_animations {
        fbx_model.animations[0]
            .bone_animations
            .keys()
            .cloned()
            .collect()
    } else {
        std::collections::HashSet::new()
    };

    let mut name_usage_count: HashMap<String, usize> = HashMap::new();

    for (fbx_idx, fbx_data) in fbx_model.fbx_data.iter_mut().enumerate() {
        let ufbx_id = split_infos.get(fbx_idx).map(|info| info.ufbx_mesh_typed_id);

        if let Some(id) = ufbx_id {
            if let Some(node_name) = mesh_to_node.get(&id) {
                let count = name_usage_count.entry(node_name.clone()).or_insert(0);
                let unique_name = if *count == 0 {
                    node_name.clone()
                } else {
                    format!("{}_mat{}", node_name, count)
                };
                *count += 1;
                fbx_data.mesh_node_name = Some(unique_name);

                if fbx_data.clusters.is_empty() && !animated_nodes.is_empty() {
                    fbx_data.parent_node = Some(node_name.clone());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fbx_model_default() {
        let model = FbxModel::default();
        assert_eq!(model.fbx_data.len(), 0);
        assert_eq!(model.animations.len(), 0);
        assert_eq!(model.nodes.len(), 0);
    }

    #[test]
    fn test_fbx_data_new() {
        let data = FbxData::new();
        assert_eq!(data.positions.len(), 0);
        assert_eq!(data.local_positions.len(), 0);
        assert_eq!(data.indices.len(), 0);
        assert_eq!(data.tex_coords.len(), 0);
        assert_eq!(data.clusters.len(), 0);
        assert_eq!(data.mesh_parts.len(), 0);
        assert_eq!(data.parent_node, None);
        assert_eq!(data.material_name, None);
        assert_eq!(data.diffuse_texture, None);
    }

    #[test]
    fn test_fbx_data_add_position() {
        let mut data = FbxData::new();
        data.positions.push(Vector3::new(1.0, 2.0, 3.0));

        assert_eq!(data.positions.len(), 1);
        assert_eq!(data.positions[0].x, 1.0);
        assert_eq!(data.positions[0].y, 2.0);
        assert_eq!(data.positions[0].z, 3.0);
    }

    #[test]
    fn test_fbx_data_add_index() {
        let mut data = FbxData::new();
        data.indices.push(0);
        data.indices.push(1);
        data.indices.push(2);

        assert_eq!(data.indices.len(), 3);
        assert_eq!(data.indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_fbx_data_add_tex_coord() {
        let mut data = FbxData::new();
        data.tex_coords.push([0.5, 0.5]);

        assert_eq!(data.tex_coords.len(), 1);
        assert_eq!(data.tex_coords[0], [0.5, 0.5]);
    }

    #[test]
    fn test_fbx_data_set_parent_node() {
        let mut data = FbxData::new();
        data.parent_node = Some("ParentBone".to_string());

        assert_eq!(data.parent_node, Some("ParentBone".to_string()));
    }

    #[test]
    fn test_fbx_data_set_material_name() {
        let mut data = FbxData::new();
        data.material_name = Some("Material01".to_string());

        assert_eq!(data.material_name, Some("Material01".to_string()));
    }

    #[test]
    fn test_fbx_data_set_diffuse_texture() {
        let mut data = FbxData::new();
        data.diffuse_texture = Some("texture.png".to_string());

        assert_eq!(data.diffuse_texture, Some("texture.png".to_string()));
    }

    #[test]
    fn test_fbx_animation_name() {
        let animation = FbxAnimation {
            name: "Walk".to_string(),
            duration: 1.0,
            bone_animations: HashMap::new(),
        };

        assert_eq!(animation.name, "Walk");
        assert_eq!(animation.duration, 1.0);
        assert_eq!(animation.bone_animations.len(), 0);
    }

    #[test]
    fn test_keyframe_creation() {
        let keyframe = KeyFrame {
            time: 0.5,
            value: [1.0, 2.0, 3.0],
        };

        assert_eq!(keyframe.time, 0.5);
        assert_eq!(keyframe.value, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_bone_animation_structure() {
        let bone_anim = BoneAnimation {
            bone_name: "Bone01".to_string(),
            translation_keys: Vec::new(),
            rotation_keys: Vec::new(),
            scale_keys: Vec::new(),
        };

        assert_eq!(bone_anim.bone_name, "Bone01");
        assert_eq!(bone_anim.translation_keys.len(), 0);
        assert_eq!(bone_anim.rotation_keys.len(), 0);
        assert_eq!(bone_anim.scale_keys.len(), 0);
    }

    #[test]
    fn test_fbx_model_add_data() {
        let mut model = FbxModel::default();
        let data = FbxData::new();
        model.fbx_data.push(data);

        assert_eq!(model.fbx_data.len(), 1);
    }
}
