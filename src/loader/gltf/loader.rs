use crate::animation::{
    AnimationClip, AnimationSystem, Interpolation, Keyframe, MorphAnimation, MorphAnimationSystem,
    MorphTarget, Skeleton, SkinData, TransformChannel,
};
use crate::ecs::component::SpringBoneSetup;
use crate::math::*;
use crate::vulkanr::data::{Vertex, VertexData};
use anyhow::Result;
use cgmath::{Matrix4, Quaternion, SquareMatrix, Vector3, Vector4};
use gltf::buffer::Data;
use gltf::{Document, Node};
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct ImageData {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub struct GltfMeshData {
    pub vertex_data: VertexData,
    pub skin_data: Option<SkinData>,
    pub morph_targets: Vec<MorphTarget>,
    pub base_positions: Vec<[f32; 3]>,
    pub skeleton_id: Option<u32>,
    pub image_data: Vec<ImageData>,
    pub node_index: Option<usize>,
    pub local_vertices: Vec<Vertex>,
    pub base_color_factor: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub index: usize,
    pub name: String,
    pub parent_index: Option<usize>,
    pub local_transform: Matrix4<f32>,
}

impl Default for NodeInfo {
    fn default() -> Self {
        Self {
            index: 0,
            name: String::new(),
            parent_index: None,
            local_transform: Matrix4::identity(),
        }
    }
}

pub struct GltfLoadResult {
    pub meshes: Vec<GltfMeshData>,
    pub nodes: Vec<NodeInfo>,
    pub animation_system: AnimationSystem,
    pub clips: Vec<AnimationClip>,
    pub morph_animation: MorphAnimationSystem,
    pub has_skinned_meshes: bool,
    pub has_armature: bool,
    pub spring_bone_setup: Option<SpringBoneSetup>,
}

#[derive(Clone, Debug, Default)]
struct Joint {
    index: u16,
    name: String,
    child_joint_indices: Vec<u16>,
    inverse_bind_pose: [[f32; 4]; 4],
    transform: [[f32; 4]; 4],
    original_node_index: Option<usize>,
}

#[derive(Clone, Debug, Default)]
struct JointAnimation {
    key_frames: Vec<f32>,
    translations: Vec<Mat4>,
    rotations: Vec<Mat4>,
    scales: Vec<Mat4>,
}

#[derive(Clone, Debug)]
struct NodeAnimation {
    node_index: usize,
    translation_keyframes: Vec<f32>,
    translations: Vec<Vector3<f32>>,
    translation_in_tangents: Vec<Vector3<f32>>,
    translation_out_tangents: Vec<Vector3<f32>>,
    rotation_keyframes: Vec<f32>,
    rotations: Vec<Quaternion<f32>>,
    rotation_in_tangents: Vec<Quaternion<f32>>,
    rotation_out_tangents: Vec<Quaternion<f32>>,
    scale_keyframes: Vec<f32>,
    scales: Vec<Vector3<f32>>,
    scale_in_tangents: Vec<Vector3<f32>>,
    scale_out_tangents: Vec<Vector3<f32>>,
    interpolation: Interpolation,
    default_translation: Vector3<f32>,
    default_rotation: Quaternion<f32>,
    default_scale: Vector3<f32>,
}

impl Default for NodeAnimation {
    fn default() -> Self {
        NodeAnimation {
            node_index: 0,
            translation_keyframes: Vec::new(),
            translations: Vec::new(),
            translation_in_tangents: Vec::new(),
            translation_out_tangents: Vec::new(),
            rotation_keyframes: Vec::new(),
            rotations: Vec::new(),
            rotation_in_tangents: Vec::new(),
            rotation_out_tangents: Vec::new(),
            scale_keyframes: Vec::new(),
            scales: Vec::new(),
            scale_in_tangents: Vec::new(),
            scale_out_tangents: Vec::new(),
            interpolation: Interpolation::Linear,
            default_translation: Vector3::new(0.0, 0.0, 0.0),
            default_rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            default_scale: Vector3::new(1.0, 1.0, 1.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct RRNode {
    index: u16,
    name: String,
    transform: [[f32; 4]; 4],
    children: Vec<u16>,
}

#[derive(Clone, Debug, Default)]
struct NodeJointMap {
    node_to_joint: HashMap<u16, u16>,
    joint_to_node: HashMap<u16, u16>,
}

impl NodeJointMap {
    fn make_from_skin(&mut self, skin: &gltf::Skin) {
        self.node_to_joint.clear();
        self.joint_to_node.clear();
        for (joint_index, joint_node) in skin.joints().enumerate() {
            self.node_to_joint
                .insert(joint_node.index() as u16, joint_index as u16);
            self.joint_to_node
                .insert(joint_index as u16, joint_node.index() as u16);
        }
    }

    fn get_node_index(&self, joint_index: u16) -> Option<u16> {
        self.joint_to_node.get(&joint_index).copied()
    }

    fn get_joint_index(&self, node_index: u16) -> Option<u16> {
        self.node_to_joint.get(&node_index).copied()
    }

    fn contain_node_index(&self, node_index: u16) -> bool {
        self.node_to_joint.contains_key(&node_index)
    }
}

#[derive(Clone, Debug, Default)]
struct MorphAnimationRaw {
    key_frame: f32,
    weights: Vec<f32>,
}

struct MeshBuildData {
    vertex_data: VertexData,
    bone_indices: Vec<Vector4<u32>>,
    bone_weights: Vec<Vector4<f32>>,
    base_positions: Vec<[f32; 3]>,
    base_normals: Vec<Vector3<f32>>,
    morph_targets: Vec<MorphTarget>,
    image_data: Vec<ImageData>,
    has_joints: bool,
    node_index: usize,
    local_vertices: Vec<Vertex>,
    base_color_factor: [f32; 4],
}

struct GltfParseContext {
    meshes: Vec<MeshBuildData>,
    morph_animations: Vec<MorphAnimationRaw>,
    joints: Vec<Joint>,
    joint_animations: Vec<Vec<JointAnimation>>,
    node_animations: Vec<NodeAnimation>,
    node_joint_map: NodeJointMap,
    rrnodes: Vec<RRNode>,
    node_infos: Vec<NodeInfo>,
    has_skinned_meshes: bool,
    has_armature: bool,
    skeleton_root_transform: Option<[[f32; 4]; 4]>,
    spring_bone_setup: Option<SpringBoneSetup>,
}

impl Default for GltfParseContext {
    fn default() -> Self {
        Self {
            meshes: Vec::new(),
            morph_animations: Vec::new(),
            joints: Vec::new(),
            joint_animations: Vec::new(),
            node_animations: Vec::new(),
            node_joint_map: NodeJointMap::default(),
            rrnodes: Vec::new(),
            node_infos: Vec::new(),
            has_skinned_meshes: false,
            has_armature: false,
            skeleton_root_transform: None,
            spring_bone_setup: None,
        }
    }
}

pub unsafe fn load_gltf_file(path: &str) -> Result<GltfLoadResult> {
    log!("Loading glTF file: {}", path);
    let (gltf, buffers, images) = gltf::import(format!("{}", path))?;
    let mut ctx = GltfParseContext::default();
    parse_gltf_imported(&mut ctx, &gltf, &buffers, &images)?;
    Ok(build_result(ctx))
}

#[cfg(feature = "text-to-mesh")]
pub unsafe fn load_gltf_from_slice(data: &[u8]) -> Result<GltfLoadResult> {
    log!("Loading glTF from memory ({} bytes)", data.len());
    let (gltf, buffers, images) = gltf::import_slice(data)?;
    let mut ctx = GltfParseContext::default();
    parse_gltf_imported(&mut ctx, &gltf, &buffers, &images)?;
    Ok(build_result(ctx))
}

unsafe fn parse_gltf_imported(
    ctx: &mut GltfParseContext,
    gltf: &Document,
    buffers: &Vec<gltf::buffer::Data>,
    images: &Vec<gltf::image::Data>,
) -> Result<()> {
    log!(
        "glTF: {} skins, {} nodes, {} meshes, {} animations",
        gltf.skins().count(),
        gltf.nodes().count(),
        gltf.meshes().count(),
        gltf.animations().count()
    );

    let node_parent_map = build_node_parent_map(gltf);
    ctx.has_armature = gltf.skins().count() > 0;

    for (i, skin) in gltf.skins().enumerate() {
        log!(
            "Skin {}: name={:?}, {} joints",
            i,
            skin.name(),
            skin.joints().count()
        );
        ctx.node_joint_map.make_from_skin(&skin);
        set_joints(ctx, &skin, buffers);
        ctx.skeleton_root_transform =
            determine_skeleton_root_transform(gltf, &skin, &node_parent_map);
    }

    for scene in gltf.scenes() {
        for node in scene.nodes() {
            process_node(
                gltf,
                buffers,
                images,
                &node,
                ctx,
                &Matrix4::identity(),
                None,
            )?;
        }
    }

    load_white_texture_if_none(ctx);
    initialize_joint_animation(ctx);

    let morph_target_count = ctx
        .meshes
        .last()
        .map(|m| m.morph_targets.len())
        .unwrap_or(0);
    for animation in gltf.animations() {
        process_animation(buffers, animation, ctx, morph_target_count)?;
    }

    ctx.spring_bone_setup = extract_spring_bone_extension(gltf, &ctx.node_joint_map);

    log!(
        "Loaded: has_skinned_meshes={}, {} node_animations, {} joint_animations",
        ctx.has_skinned_meshes,
        ctx.node_animations.len(),
        ctx.joint_animations.len()
    );

    Ok(())
}

fn extract_spring_bone_extension(
    gltf: &Document,
    node_joint_map: &NodeJointMap,
) -> Option<SpringBoneSetup> {
    let extension_json = gltf.extension_value("VRMC_springBone")?;

    let resolve = |node_index: u32| -> Option<u32> {
        node_joint_map
            .get_joint_index(node_index as u16)
            .map(|j| j as u32)
    };

    let setup = super::spring_bone_extension::parse_vrmc_spring_bone(extension_json, &resolve);

    if let Some(ref s) = setup {
        log!(
            "VRMC_springBone loaded: {} chains, {} colliders, {} groups",
            s.chains.len(),
            s.colliders.len(),
            s.collider_groups.len()
        );
    }

    setup
}

fn set_joints(ctx: &mut GltfParseContext, skin: &gltf::Skin, buffers: &Vec<Data>) {
    ctx.joints.clear();

    let temp_joints: Vec<_> = skin.joints().collect();
    for (joint_index, node) in temp_joints.iter().enumerate() {
        let joint_transform = mat4_from_array(node.transform().matrix());
        let Some(node_index) = ctx.node_joint_map.get_node_index(joint_index as u16) else {
            continue;
        };
        log!(
            "Joint Pushed: Node Index: {}, Node Name: {}, Joint Index: {}",
            node_index,
            node.name().unwrap_or(""),
            joint_index
        );
        ctx.joints.push(Joint {
            index: joint_index as u16,
            name: node.name().unwrap_or("").to_string(),
            child_joint_indices: Vec::new(),
            inverse_bind_pose: [[0.0; 4]; 4],
            transform: array_from_mat4(joint_transform),
            original_node_index: Some(node.index()),
        });
    }

    if skin.inverse_bind_matrices().is_some() {
        let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
        if let Some(iter) = reader.read_inverse_bind_matrices() {
            log!("Inverse bind poses: {:?}", iter.len());
            for (i, mat) in iter.enumerate() {
                let inverse_bind_pose = mat4_from_array(mat);
                ctx.joints[i].inverse_bind_pose = array_from_mat4(inverse_bind_pose);
            }
        }
    }
}

fn build_node_parent_map(gltf: &Document) -> HashMap<usize, usize> {
    let mut parent_map = HashMap::new();
    for node in gltf.nodes() {
        for child in node.children() {
            parent_map.insert(child.index(), node.index());
        }
    }
    parent_map
}

fn compute_node_global_transform(
    gltf: &Document,
    node_index: usize,
    parent_map: &HashMap<usize, usize>,
) -> Matrix4<f32> {
    let mut transform_chain = Vec::new();
    let mut current_index = Some(node_index);

    while let Some(idx) = current_index {
        if let Some(node) = gltf.nodes().nth(idx) {
            transform_chain.push(mat4_from_array(node.transform().matrix()));
        }
        current_index = parent_map.get(&idx).copied();
    }

    let mut global = Matrix4::identity();
    for transform in transform_chain.iter().rev() {
        global = global * transform;
    }
    global
}

fn determine_skeleton_root_transform(
    gltf: &Document,
    skin: &gltf::Skin,
    parent_map: &HashMap<usize, usize>,
) -> Option<[[f32; 4]; 4]> {
    if let Some(skeleton_node) = skin.skeleton() {
        let global_transform =
            compute_node_global_transform(gltf, skeleton_node.index(), parent_map);
        log!(
            "Skeleton root transform from skin.skeleton: node {} ({:?})",
            skeleton_node.index(),
            skeleton_node.name()
        );
        return Some(array_from_mat4(global_transform));
    }

    let joint_indices: std::collections::HashSet<usize> =
        skin.joints().map(|j| j.index()).collect();
    let mut root_joint_node: Option<gltf::Node> = None;

    for joint in skin.joints() {
        let has_parent_joint = parent_map
            .get(&joint.index())
            .map(|parent_idx| joint_indices.contains(parent_idx))
            .unwrap_or(false);

        if !has_parent_joint {
            root_joint_node = Some(joint);
            break;
        }
    }

    if let Some(root_joint) = root_joint_node {
        if let Some(&parent_index) = parent_map.get(&root_joint.index()) {
            let parent_global = compute_node_global_transform(gltf, parent_index, parent_map);
            log!(
                "Skeleton root transform from root joint's parent: node {}",
                parent_index
            );
            return Some(array_from_mat4(parent_global));
        }
    }

    None
}

struct RawVertexAttributes {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tex_coords: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    joint_indices: Vec<[u16; 4]>,
    joint_weights: Vec<[f32; 4]>,
    has_joints: bool,
}

fn read_vertex_attributes<'a, 's, F>(reader: &gltf::mesh::Reader<'a, 's, F>) -> RawVertexAttributes
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .map(|iter| iter.map(|p| [p[0], p[1], p[2]]).collect())
        .unwrap_or_default();

    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(|iter| iter.collect())
        .unwrap_or_default();

    let tex_coords: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|iter| iter.into_f32().collect())
        .unwrap_or_default();

    let mut has_joints = false;
    let joint_indices: Vec<[u16; 4]> = reader
        .read_joints(0)
        .map(|iter| {
            has_joints = true;
            iter.into_u16().collect()
        })
        .unwrap_or_default();

    let joint_weights: Vec<[f32; 4]> = reader
        .read_weights(0)
        .map(|iter| iter.into_f32().collect())
        .unwrap_or_default();

    let colors: Vec<[f32; 4]> = reader
        .read_colors(0)
        .map(|iter| iter.into_rgba_f32().collect())
        .unwrap_or_default();

    RawVertexAttributes {
        positions,
        normals,
        tex_coords,
        colors,
        joint_indices,
        joint_weights,
        has_joints,
    }
}

fn transform_positions(
    raw_positions: &[[f32; 3]],
    has_joints: bool,
    cumulative_transform: &Matrix4<f32>,
    node_name: &str,
) -> Vec<[f32; 3]> {
    if node_name.contains("NurbsPath.009") {
        log!(
            "=== Load-time transform for {} (has_joints={}) ===",
            node_name,
            has_joints
        );
        let ct = cumulative_transform;
        let scale = (
            (ct[0][0] * ct[0][0] + ct[0][1] * ct[0][1] + ct[0][2] * ct[0][2]).sqrt(),
            (ct[1][0] * ct[1][0] + ct[1][1] * ct[1][1] + ct[1][2] * ct[1][2]).sqrt(),
            (ct[2][0] * ct[2][0] + ct[2][1] * ct[2][1] + ct[2][2] * ct[2][2]).sqrt(),
        );
        log!(
            "  cumulative_transform: scale=[{:.1},{:.1},{:.1}] trans=[{:.2},{:.2},{:.2}]",
            scale.0,
            scale.1,
            scale.2,
            ct[3][0],
            ct[3][1],
            ct[3][2]
        );
        if !raw_positions.is_empty() {
            let raw = raw_positions[0];
            let pos = *cumulative_transform * [raw[0], raw[1], raw[2], 1.0].to_vec4();
            log!(
                "  raw[0]=({:.3},{:.3},{:.3}) -> transformed=({:.2},{:.2},{:.2})",
                raw[0],
                raw[1],
                raw[2],
                pos.x,
                pos.y,
                pos.z
            );
        }
    }

    if has_joints {
        raw_positions.to_vec()
    } else {
        raw_positions
            .iter()
            .map(|p| {
                let pos = *cumulative_transform * [p[0], p[1], p[2], 1.0].to_vec4();
                [pos.x, pos.y, pos.z]
            })
            .collect()
    }
}

fn compute_smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let vertex_count = positions.len();
    let mut normals = vec![[0.0f32; 3]; vertex_count];

    let triangles = if indices.is_empty() {
        (0..vertex_count as u32).collect::<Vec<_>>()
    } else {
        indices.to_vec()
    };

    for tri in triangles.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count {
            continue;
        }

        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];

        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let face_normal = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        for &idx in &[i0, i1, i2] {
            normals[idx][0] += face_normal[0];
            normals[idx][1] += face_normal[1];
            normals[idx][2] += face_normal[2];
        }
    }

    for n in &mut normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-8 {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        } else {
            *n = [0.0, 1.0, 0.0];
        }
    }

    normals
}

fn build_vertices(
    mesh_data: &mut MeshBuildData,
    attrs: &RawVertexAttributes,
    positions: &[[f32; 3]],
) {
    for i in 0..positions.len() {
        let pos = positions[i];
        let raw_pos = attrs.positions[i];
        let normal = attrs.normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]);
        let tex_coord = attrs.tex_coords.get(i).copied().unwrap_or([0.0, 0.0]);
        let color = attrs.colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]);

        mesh_data.vertex_data.vertices.push(Vertex {
            pos: Vec3::new(pos[0], pos[1], pos[2]),
            color: Vec4::new(color[0], color[1], color[2], color[3]),
            tex_coord: Vec2::new(tex_coord[0], tex_coord[1]),
            normal: Vec3::new(normal[0], normal[1], normal[2]),
        });

        mesh_data.local_vertices.push(Vertex {
            pos: Vec3::new(raw_pos[0], raw_pos[1], raw_pos[2]),
            color: Vec4::new(color[0], color[1], color[2], color[3]),
            tex_coord: Vec2::new(tex_coord[0], tex_coord[1]),
            normal: Vec3::new(normal[0], normal[1], normal[2]),
        });

        mesh_data.base_positions.push(pos);
        mesh_data
            .base_normals
            .push(Vector3::new(normal[0], normal[1], normal[2]));

        let ji = attrs.joint_indices.get(i).copied().unwrap_or([0, 0, 0, 0]);
        let jw = attrs
            .joint_weights
            .get(i)
            .copied()
            .unwrap_or([0.0, 0.0, 0.0, 0.0]);
        mesh_data.bone_indices.push(Vector4::new(
            ji[0] as u32,
            ji[1] as u32,
            ji[2] as u32,
            ji[3] as u32,
        ));
        mesh_data
            .bone_weights
            .push(Vector4::new(jw[0], jw[1], jw[2], jw[3]));
    }
}

fn read_morph_targets<'a, 's, F>(reader: &gltf::mesh::Reader<'a, 's, F>) -> Vec<MorphTarget>
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    let mut morph_targets = Vec::new();
    for (positions, normals, tangents) in reader.read_morph_targets() {
        let mut morph_target = MorphTarget::default();
        if let Some(pos_iter) = positions {
            morph_target.positions = pos_iter.collect::<Vec<_>>();
        }
        if let Some(norm_iter) = normals {
            morph_target.normals = norm_iter.collect::<Vec<_>>();
        }
        if let Some(tan_iter) = tangents {
            morph_target.tangents = tan_iter.collect::<Vec<_>>();
        }
        morph_targets.push(morph_target);
    }
    morph_targets
}

fn load_primitive_texture(
    primitive: &gltf::Primitive,
    images: &[gltf::image::Data],
) -> Option<ImageData> {
    let material = primitive
        .material()
        .pbr_metallic_roughness()
        .base_color_texture()?;

    let image_index = material.texture().source().index();
    if image_index < images.len() {
        let image = &images[image_index];
        log!(
            "PBR texture: image_index={}, format={:?}, {}x{}, pixel_bytes={}",
            image_index,
            image.format,
            image.width,
            image.height,
            image.pixels.len()
        );
        let first_pixels: Vec<u8> = image.pixels.iter().take(16).copied().collect();
        log!("  first 16 bytes: {:?}", first_pixels);

        let converted = convert_image_data(image);
        log!(
            "  converted: {}x{}, data_len={}, first_16={:?}",
            converted.width,
            converted.height,
            converted.data.len(),
            &converted.data[..converted.data.len().min(16)]
        );
        Some(converted)
    } else {
        None
    }
}

fn register_node_info(
    ctx: &mut GltfParseContext,
    node: &Node,
    node_transform: Matrix4<f32>,
    parent_node_index: Option<usize>,
) {
    ctx.rrnodes.push(RRNode {
        index: node.index() as u16,
        name: node.name().unwrap_or("").to_string(),
        transform: array_from_mat4(node_transform),
        children: node.children().map(|c| c.index() as u16).collect(),
    });

    ctx.node_infos.push(NodeInfo {
        index: node.index(),
        name: node.name().unwrap_or("").to_string(),
        parent_index: parent_node_index,
        local_transform: node_transform,
    });
}

fn update_joint_hierarchy(
    ctx: &mut GltfParseContext,
    node: &Node,
    node_transform: Matrix4<f32>,
    parent_node_index: Option<usize>,
) {
    if let Some(&joint_index) = ctx.node_joint_map.node_to_joint.get(&(node.index() as u16)) {
        ctx.joints[joint_index as usize].transform = array_from_mat4(node_transform);

        if let Some(parent_index) = parent_node_index {
            if let Some(&parent_joint_index) =
                ctx.node_joint_map.node_to_joint.get(&(parent_index as u16))
            {
                ctx.joints[parent_joint_index as usize]
                    .child_joint_indices
                    .push(joint_index);
            }
        }
    }
}

unsafe fn process_node(
    gltf: &Document,
    buffers: &Vec<Data>,
    images: &Vec<gltf::image::Data>,
    node: &Node,
    ctx: &mut GltfParseContext,
    parent_transform: &Matrix4<f32>,
    parent_node_index: Option<usize>,
) -> Result<()> {
    let node_transform = mat4_from_array(node.transform().matrix());
    let cumulative_transform = *parent_transform * node_transform;
    let node_name = node.name().unwrap_or("");

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let mut attrs = read_vertex_attributes(&reader);

            let indices: Vec<u32> = reader
                .read_indices()
                .map(|iter| iter.into_u32().collect())
                .unwrap_or_default();

            if attrs.normals.is_empty() && !attrs.positions.is_empty() {
                attrs.normals = compute_smooth_normals(&attrs.positions, &indices);
            }

            let base_color_factor = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_factor();

            let mut mesh_data = MeshBuildData {
                vertex_data: VertexData::default(),
                bone_indices: Vec::new(),
                bone_weights: Vec::new(),
                base_positions: Vec::new(),
                base_normals: Vec::new(),
                morph_targets: Vec::new(),
                image_data: Vec::new(),
                has_joints: attrs.has_joints,
                node_index: node.index(),
                local_vertices: Vec::new(),
                base_color_factor,
            };
            if attrs.has_joints {
                ctx.has_skinned_meshes = true;
            }

            let positions = transform_positions(
                &attrs.positions,
                attrs.has_joints,
                &cumulative_transform,
                node_name,
            );
            build_vertices(&mut mesh_data, &attrs, &positions);

            mesh_data.vertex_data.indices = indices;

            mesh_data.morph_targets = read_morph_targets(&reader);

            if let Some(image_data) = load_primitive_texture(&primitive, images) {
                mesh_data.image_data.push(image_data);
            }

            if node_name.contains("NurbsPath.009") && !mesh_data.local_vertices.is_empty() {
                let lv = &mesh_data.local_vertices[0];
                log!(
                    "  After processing: local_vertices[0]=({:.3},{:.3},{:.3}), count={}",
                    lv.pos.x,
                    lv.pos.y,
                    lv.pos.z,
                    mesh_data.local_vertices.len()
                );
            }

            ctx.meshes.push(mesh_data);
        }
    }

    register_node_info(ctx, node, node_transform, parent_node_index);
    update_joint_hierarchy(ctx, node, node_transform, parent_node_index);

    for child in node.children() {
        process_node(
            gltf,
            buffers,
            images,
            &child,
            ctx,
            &cumulative_transform,
            Some(node.index()),
        )?;
    }

    Ok(())
}

fn convert_image_data(image: &gltf::image::Data) -> ImageData {
    let data = match image.format {
        gltf::image::Format::R8G8B8A8 => image.pixels.clone(),
        gltf::image::Format::R8G8B8 => {
            let mut rgba_data = Vec::with_capacity(image.pixels.len() / 3 * 4);
            for chunk in image.pixels.chunks(3) {
                rgba_data.push(chunk[0]);
                rgba_data.push(chunk[1]);
                rgba_data.push(chunk[2]);
                rgba_data.push(255);
            }
            rgba_data
        }
        _ => {
            log!("Unsupported image format: {:?}", image.format);
            vec![255, 255, 255, 255]
        }
    };

    ImageData {
        data,
        width: image.width,
        height: image.height,
    }
}

fn load_white_texture_if_none(ctx: &mut GltfParseContext) {
    for mesh in &mut ctx.meshes {
        if mesh.image_data.is_empty() {
            mesh.image_data.push(ImageData {
                data: vec![255, 255, 255, 255],
                width: 1,
                height: 1,
            });
        }
    }
}

fn initialize_joint_animation(ctx: &mut GltfParseContext) {
    for _ in 0..ctx.joints.len() {
        ctx.joint_animations.push(Vec::new());
    }
}

enum ParsedChannelData {
    Translation(TranslationChannelData),
    Rotation(RotationChannelData),
    Scale(ScaleChannelData),
    MorphWeights,
}

struct TranslationChannelData {
    joint_matrices: Vec<Mat4>,
    values: Vec<Vector3<f32>>,
    in_tangents: Vec<Vector3<f32>>,
    out_tangents: Vec<Vector3<f32>>,
}

struct RotationChannelData {
    joint_matrices: Vec<Mat4>,
    quats: Vec<Quaternion<f32>>,
    in_tangents: Vec<Quaternion<f32>>,
    out_tangents: Vec<Quaternion<f32>>,
}

struct ScaleChannelData {
    joint_matrices: Vec<Mat4>,
    values: Vec<Vector3<f32>>,
    in_tangents: Vec<Vector3<f32>>,
    out_tangents: Vec<Vector3<f32>>,
}

unsafe fn process_animation(
    buffers: &Vec<Data>,
    animation: gltf::Animation,
    ctx: &mut GltfParseContext,
    morph_target_count: usize,
) -> Result<()> {
    use gltf::animation::util::ReadOutputs;

    for channel in animation.channels() {
        let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
        let Some(inputs) = reader.read_inputs() else {
            continue;
        };
        let key_frames: Vec<f32> = inputs.collect();

        let gltf_interp = channel.sampler().interpolation();
        let is_cubic = gltf_interp == gltf::animation::Interpolation::CubicSpline;
        let interp = convert_gltf_interpolation(gltf_interp);

        let Some(outputs) = reader.read_outputs() else {
            continue;
        };

        let parsed = match outputs {
            ReadOutputs::Translations(translations) => {
                ParsedChannelData::Translation(parse_translation_outputs(translations, is_cubic))
            }
            ReadOutputs::Rotations(rotations) => {
                ParsedChannelData::Rotation(parse_rotation_outputs(rotations, is_cubic))
            }
            ReadOutputs::Scales(scales) => {
                ParsedChannelData::Scale(parse_scale_outputs(scales, is_cubic))
            }
            ReadOutputs::MorphTargetWeights(weights) => {
                collect_morph_weights(weights, morph_target_count, &key_frames, ctx);
                ParsedChannelData::MorphWeights
            }
        };

        let target = channel.target();
        let node = target.node();

        let is_joint_node = ctx
            .node_joint_map
            .node_to_joint
            .contains_key(&(node.index() as u16));

        if is_joint_node && ctx.has_skinned_meshes {
            let Some(&joint_id) = ctx.node_joint_map.node_to_joint.get(&(node.index() as u16))
            else {
                continue;
            };
            apply_joint_animation(&parsed, &key_frames, joint_id, ctx);
        } else {
            let node_animation = find_or_create_node_animation(ctx, &node, interp);
            apply_node_animation(&parsed, &key_frames, is_cubic, node_animation);
        }
    }

    Ok(())
}

fn parse_translation_outputs(
    translations: gltf::animation::util::Translations,
    is_cubic: bool,
) -> TranslationChannelData {
    let mut data = TranslationChannelData {
        joint_matrices: Vec::new(),
        values: Vec::new(),
        in_tangents: Vec::new(),
        out_tangents: Vec::new(),
    };

    if is_cubic {
        let all: Vec<_> = translations.collect();
        for chunk in all.chunks(3) {
            if chunk.len() == 3 {
                let vec = Vector3::new(chunk[1][0], chunk[1][1], chunk[1][2]);
                data.joint_matrices.push(Matrix4::from_translation(vec));
                data.values.push(vec);
                data.in_tangents
                    .push(Vector3::new(chunk[0][0], chunk[0][1], chunk[0][2]));
                data.out_tangents
                    .push(Vector3::new(chunk[2][0], chunk[2][1], chunk[2][2]));
            }
        }
    } else {
        for t in translations {
            let vec = Vector3::new(t[0], t[1], t[2]);
            data.joint_matrices.push(Matrix4::from_translation(vec));
            data.values.push(vec);
        }
    }

    data
}

fn parse_rotation_outputs(
    rotations: gltf::animation::util::Rotations,
    is_cubic: bool,
) -> RotationChannelData {
    let mut data = RotationChannelData {
        joint_matrices: Vec::new(),
        quats: Vec::new(),
        in_tangents: Vec::new(),
        out_tangents: Vec::new(),
    };

    if is_cubic {
        let all: Vec<_> = rotations.into_f32().collect();
        for chunk in all.chunks(3) {
            if chunk.len() == 3 {
                let quat = Quaternion::new(chunk[1][3], chunk[1][0], chunk[1][1], chunk[1][2]);
                data.quats.push(quat);
                data.joint_matrices.push(Matrix4::from(quat));
                data.in_tangents.push(Quaternion::new(
                    chunk[0][3],
                    chunk[0][0],
                    chunk[0][1],
                    chunk[0][2],
                ));
                data.out_tangents.push(Quaternion::new(
                    chunk[2][3],
                    chunk[2][0],
                    chunk[2][1],
                    chunk[2][2],
                ));
            }
        }
    } else {
        for r in rotations.into_f32() {
            let quat = Quaternion::new(r[3], r[0], r[1], r[2]);
            data.quats.push(quat);
            data.joint_matrices.push(Matrix4::from(quat));
        }
    }

    data
}

fn parse_scale_outputs(scales: gltf::animation::util::Scales, is_cubic: bool) -> ScaleChannelData {
    let mut data = ScaleChannelData {
        joint_matrices: Vec::new(),
        values: Vec::new(),
        in_tangents: Vec::new(),
        out_tangents: Vec::new(),
    };

    if is_cubic {
        let all: Vec<_> = scales.collect();
        for chunk in all.chunks(3) {
            if chunk.len() == 3 {
                let vec = Vector3::new(chunk[1][0], chunk[1][1], chunk[1][2]);
                data.joint_matrices
                    .push(Matrix4::from_nonuniform_scale(vec.x, vec.y, vec.z));
                data.values.push(vec);
                data.in_tangents
                    .push(Vector3::new(chunk[0][0], chunk[0][1], chunk[0][2]));
                data.out_tangents
                    .push(Vector3::new(chunk[2][0], chunk[2][1], chunk[2][2]));
            }
        }
    } else {
        for s in scales {
            let vec = Vector3::new(s[0], s[1], s[2]);
            data.joint_matrices
                .push(Matrix4::from_nonuniform_scale(vec.x, vec.y, vec.z));
            data.values.push(vec);
        }
    }

    data
}

fn collect_morph_weights(
    morph_target_weights: gltf::animation::util::MorphTargetWeights,
    morph_target_count: usize,
    key_frames: &[f32],
    ctx: &mut GltfParseContext,
) {
    if morph_target_count == 0 {
        return;
    }

    let mut current_weight_set = Vec::new();
    let mut grouped_weights = Vec::new();

    for w in morph_target_weights.into_f32() {
        current_weight_set.push(w);
        if current_weight_set.len() >= morph_target_count {
            grouped_weights.push(current_weight_set.clone());
            current_weight_set.clear();
        }
    }

    for (i, weight_set) in grouped_weights.iter().enumerate() {
        if i < key_frames.len() {
            ctx.morph_animations.push(MorphAnimationRaw {
                key_frame: key_frames[i],
                weights: weight_set.clone(),
            });
        }
    }
}

fn apply_joint_animation(
    parsed: &ParsedChannelData,
    key_frames: &[f32],
    joint_id: u16,
    ctx: &mut GltfParseContext,
) {
    let mut translations = Vec::new();
    let mut rotations = Vec::new();
    let mut scales = Vec::new();

    match parsed {
        ParsedChannelData::Translation(d) => translations = d.joint_matrices.clone(),
        ParsedChannelData::Rotation(d) => rotations = d.joint_matrices.clone(),
        ParsedChannelData::Scale(d) => scales = d.joint_matrices.clone(),
        ParsedChannelData::MorphWeights => {}
    }

    ctx.joint_animations[joint_id as usize].push(JointAnimation {
        key_frames: key_frames.to_vec(),
        translations,
        rotations,
        scales,
    });
}

fn find_or_create_node_animation<'a>(
    ctx: &'a mut GltfParseContext,
    node: &Node,
    interp: Interpolation,
) -> &'a mut NodeAnimation {
    let node_index = node.index();
    let existing_index = ctx
        .node_animations
        .iter()
        .position(|na| na.node_index == node_index);

    if let Some(idx) = existing_index {
        return &mut ctx.node_animations[idx];
    }

    let (default_trans, default_rot, default_scale) =
        decompose(&mat4_from_array(node.transform().matrix()));

    ctx.node_animations.push(NodeAnimation {
        node_index,
        default_translation: default_trans,
        default_rotation: default_rot,
        default_scale,
        interpolation: interp,
        ..Default::default()
    });

    ctx.node_animations
        .last_mut()
        .expect("just pushed node_animation")
}

fn apply_node_animation(
    parsed: &ParsedChannelData,
    key_frames: &[f32],
    is_cubic: bool,
    anim: &mut NodeAnimation,
) {
    match parsed {
        ParsedChannelData::Translation(d) => {
            for (i, &kf) in key_frames.iter().enumerate().take(d.values.len()) {
                anim.translation_keyframes.push(kf);
                anim.translations.push(d.values[i]);
                if is_cubic && i < d.in_tangents.len() {
                    anim.translation_in_tangents.push(d.in_tangents[i]);
                    anim.translation_out_tangents.push(d.out_tangents[i]);
                }
            }
        }

        ParsedChannelData::Rotation(d) => {
            for (i, &kf) in key_frames.iter().enumerate().take(d.quats.len()) {
                anim.rotation_keyframes.push(kf);
                anim.rotations.push(d.quats[i]);
                if is_cubic && i < d.in_tangents.len() {
                    anim.rotation_in_tangents.push(d.in_tangents[i]);
                    anim.rotation_out_tangents.push(d.out_tangents[i]);
                }
            }
        }

        ParsedChannelData::Scale(d) => {
            for (i, &kf) in key_frames.iter().enumerate().take(d.values.len()) {
                anim.scale_keyframes.push(kf);
                anim.scales.push(d.values[i]);
                if is_cubic && i < d.in_tangents.len() {
                    anim.scale_in_tangents.push(d.in_tangents[i]);
                    anim.scale_out_tangents.push(d.out_tangents[i]);
                }
            }
        }

        ParsedChannelData::MorphWeights => {}
    }
}

fn build_result(ctx: GltfParseContext) -> GltfLoadResult {
    let mut animation_system = AnimationSystem::new();
    let mut clips = Vec::new();

    let skeleton_id = if !ctx.joints.is_empty() {
        let skeleton = convert_joints_to_skeleton(&ctx.joints, &ctx.skeleton_root_transform);
        Some(animation_system.add_skeleton(skeleton))
    } else {
        None
    };

    collect_animation_clips(&ctx, skeleton_id, &mut animation_system, &mut clips);

    let scale = if ctx.has_armature { 0.01 } else { 1.0 };
    log!(
        "glTF scale: {} (has_armature={}, has_skinned_meshes={})",
        scale,
        ctx.has_armature,
        ctx.has_skinned_meshes
    );

    let (meshes, morph_system) = build_meshes_and_morph(
        ctx.meshes,
        &ctx.morph_animations,
        &ctx.joints,
        skeleton_id,
        scale,
    );

    log_gltf_scale_info(&meshes, ctx.has_skinned_meshes);

    if !morph_system.animations.is_empty() {
        log!(
            "Morph animation loaded: {} keyframes, {} meshes",
            morph_system.animations.len(),
            morph_system.targets.len()
        );
    }

    GltfLoadResult {
        meshes,
        nodes: ctx.node_infos,
        animation_system,
        clips,
        morph_animation: morph_system,
        has_skinned_meshes: ctx.has_skinned_meshes,
        has_armature: ctx.has_armature,
        spring_bone_setup: ctx.spring_bone_setup,
    }
}

fn collect_animation_clips(
    ctx: &GltfParseContext,
    skeleton_id: Option<u32>,
    animation_system: &mut AnimationSystem,
    clips: &mut Vec<AnimationClip>,
) {
    if !ctx.joint_animations.is_empty() {
        let clip = convert_joint_animations_to_clip(&ctx.joint_animations);
        log!(
            "Joint animation clip: duration={}, channels={}",
            clip.duration,
            clip.channels.len()
        );
        if clip.duration > 0.0 && !clip.channels.is_empty() {
            clips.push(clip);
        }
    }

    if let Some(skel_id) = skeleton_id {
        if !ctx.node_animations.is_empty() {
            let clip = convert_node_animations_to_clip(
                &ctx.node_animations,
                &ctx.rrnodes,
                animation_system,
                skel_id,
            );
            log!(
                "Node animation clip: duration={}, channels={}",
                clip.duration,
                clip.channels.len()
            );
            if clip.duration > 0.0 && !clip.channels.is_empty() {
                if let Some(skeleton) = animation_system.get_skeleton_mut(skel_id) {
                    initialize_skeleton_from_clip(skeleton, &clip, 0.0);
                    log!("Initialized skeleton bones with animation t=0 values");
                }
                clips.push(clip);
            }
        }
    }
}

fn build_meshes_and_morph(
    source_meshes: Vec<MeshBuildData>,
    morph_animations: &[MorphAnimationRaw],
    joints: &[Joint],
    skeleton_id: Option<u32>,
    scale: f32,
) -> (Vec<GltfMeshData>, MorphAnimationSystem) {
    let mut meshes = Vec::new();
    let mut morph_system = MorphAnimationSystem::new();
    morph_system.scale_factor = scale;

    for anim in morph_animations {
        morph_system.animations.push(MorphAnimation {
            key_frame: anim.key_frame,
            weights: anim.weights.clone(),
        });
    }

    for mesh in source_meshes {
        let mut vertex_data = mesh.vertex_data;

        if scale != 1.0 {
            for v in &mut vertex_data.vertices {
                v.pos.x *= scale;
                v.pos.y *= scale;
                v.pos.z *= scale;
            }
        }

        let skin_data = if !joints.is_empty() && skeleton_id.is_some() && mesh.has_joints {
            Some(SkinData {
                skeleton_id: skeleton_id
                    .expect("skeleton_id verified as Some by is_some() check above"),
                bone_indices: mesh.bone_indices,
                bone_weights: mesh.bone_weights,
                base_positions: mesh
                    .base_positions
                    .iter()
                    .map(|p| Vector3::new(p[0], p[1], p[2]))
                    .collect(),
                base_normals: mesh.base_normals,
            })
        } else {
            None
        };

        morph_system.targets.push(mesh.morph_targets.clone());

        let base_verts: Vec<[f32; 3]> = mesh
            .base_positions
            .iter()
            .map(|p| [p[0] * scale, p[1] * scale, p[2] * scale])
            .collect();
        morph_system.base_vertices.push(base_verts);

        let local_vertices: Vec<Vertex> = mesh.local_vertices.clone();

        meshes.push(GltfMeshData {
            vertex_data,
            skin_data,
            morph_targets: mesh.morph_targets,
            base_positions: mesh.base_positions,
            skeleton_id,
            image_data: mesh.image_data,
            node_index: Some(mesh.node_index),
            local_vertices,
            base_color_factor: mesh.base_color_factor,
        });
    }

    (meshes, morph_system)
}

fn log_gltf_scale_info(meshes: &[GltfMeshData], has_skinned_meshes: bool) {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut min_z = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut max_z = f32::MIN;
    let mut total_vertices = 0;

    for mesh in meshes {
        for v in &mesh.vertex_data.vertices {
            min_x = min_x.min(v.pos.x);
            min_y = min_y.min(v.pos.y);
            min_z = min_z.min(v.pos.z);
            max_x = max_x.max(v.pos.x);
            max_y = max_y.max(v.pos.y);
            max_z = max_z.max(v.pos.z);
            total_vertices += 1;
        }
    }

    if total_vertices == 0 {
        return;
    }

    let size_x = max_x - min_x;
    let size_y = max_y - min_y;
    let size_z = max_z - min_z;
    let max_dimension = size_x.max(size_y).max(size_z);

    log!("=== glTF Scale Info ===");
    log!("  Total vertices: {}", total_vertices);
    log!(
        "  Bounding box min: ({:.4}, {:.4}, {:.4})",
        min_x,
        min_y,
        min_z
    );
    log!(
        "  Bounding box max: ({:.4}, {:.4}, {:.4})",
        max_x,
        max_y,
        max_z
    );
    log!("  Size: ({:.4}, {:.4}, {:.4})", size_x, size_y, size_z);
    log!("  Max dimension: {:.4} (glTF spec: meters)", max_dimension);

    if has_skinned_meshes {
        log!("  Skinned mesh: vertex positions are in bind pose (small bounds expected)");
        return;
    }

    if max_dimension > 100.0 {
        log_warn!(
            "Model appears very large ({:.4}m). Might be in mm or cm units.",
            max_dimension
        );
    } else if max_dimension < 0.01 {
        log_warn!(
            "Model appears very small ({:.4}m). Check unit scale.",
            max_dimension
        );
    }
}

fn initialize_skeleton_from_clip(skeleton: &mut Skeleton, clip: &AnimationClip, time: f32) {
    use crate::animation::{compose_transform, decompose_transform};

    for (&bone_id, channel) in &clip.channels {
        if let Some(bone) = skeleton.get_bone_mut(bone_id) {
            let (rest_t, rest_r, rest_s) = decompose_transform(&bone.local_transform);

            let translation = channel.sample_translation(time).unwrap_or(rest_t);
            let rotation = channel.sample_rotation(time).unwrap_or(rest_r);
            let scale = channel.sample_scale(time).unwrap_or(rest_s);

            bone.local_transform = compose_transform(translation, rotation, scale);
        }
    }
}

fn convert_joints_to_skeleton(
    joints: &[Joint],
    skeleton_root_transform: &Option<[[f32; 4]; 4]>,
) -> Skeleton {
    let mut skeleton = Skeleton::new("gltf_skeleton");

    if let Some(transform) = skeleton_root_transform {
        skeleton.root_transform = mat4_from_array(*transform);
        log!("Skeleton root_transform set from glTF: diag=[{:.4}, {:.4}, {:.4}], trans=[{:.4}, {:.4}, {:.4}]",
            skeleton.root_transform[0][0], skeleton.root_transform[1][1], skeleton.root_transform[2][2],
            skeleton.root_transform[3][0], skeleton.root_transform[3][1], skeleton.root_transform[3][2]);
    }

    for joint in joints {
        let parent_id = find_parent_joint_id(joints, joint.index);
        let bone_id = skeleton.add_bone(&joint.name, parent_id);

        if let Some(bone) = skeleton.get_bone_mut(bone_id) {
            bone.local_transform = mat4_from_array(joint.transform);
            bone.inverse_bind_pose = mat4_from_array(joint.inverse_bind_pose);
            bone.node_index = joint.original_node_index;
        }
    }

    skeleton
}

fn convert_gltf_interpolation(gltf_interp: gltf::animation::Interpolation) -> Interpolation {
    match gltf_interp {
        gltf::animation::Interpolation::Step => Interpolation::Step,
        gltf::animation::Interpolation::Linear => Interpolation::Linear,
        gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
    }
}

fn find_parent_joint_id(joints: &[Joint], child_index: u16) -> Option<u32> {
    for (idx, joint) in joints.iter().enumerate() {
        if joint.child_joint_indices.contains(&child_index) {
            return Some(idx as u32);
        }
    }
    None
}

fn convert_joint_animations_to_clip(joint_animations: &[Vec<JointAnimation>]) -> AnimationClip {
    let mut clip = AnimationClip::new("gltf_joint_animation");
    let mut max_duration = 0.0f32;

    for (joint_idx, anims) in joint_animations.iter().enumerate() {
        if anims.is_empty() {
            continue;
        }

        let mut all_times: Vec<f32> = Vec::new();
        for anim in anims {
            for &time in &anim.key_frames {
                if !all_times.iter().any(|t| (*t - time).abs() < 0.0001) {
                    all_times.push(time);
                }
            }
        }
        all_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(&last) = all_times.last() {
            if last > max_duration {
                max_duration = last;
            }
        }

        let mut channel = TransformChannel::default();

        for &time in &all_times {
            let mut combined_translate = Matrix4::identity();
            let mut combined_rotation = Matrix4::identity();
            let mut combined_scale = Matrix4::identity();

            for anim in anims {
                let key_frame_id = identify_key_frame_index_step(&anim.key_frames, time);

                if key_frame_id < anim.scales.len() {
                    combined_scale = anim.scales[key_frame_id] * combined_scale;
                }
                if key_frame_id < anim.rotations.len() {
                    combined_rotation = anim.rotations[key_frame_id] * combined_rotation;
                }
                if key_frame_id < anim.translations.len() {
                    combined_translate = anim.translations[key_frame_id] * combined_translate;
                }
            }

            channel.translation.push(Keyframe::with_interpolation(
                time,
                Vector3::new(
                    combined_translate[3][0],
                    combined_translate[3][1],
                    combined_translate[3][2],
                ),
                Interpolation::Step,
            ));

            channel.rotation.push(Keyframe::with_interpolation(
                time,
                matrix_to_quaternion(&combined_rotation),
                Interpolation::Step,
            ));

            channel.scale.push(Keyframe::with_interpolation(
                time,
                Vector3::new(
                    combined_scale[0][0],
                    combined_scale[1][1],
                    combined_scale[2][2],
                ),
                Interpolation::Step,
            ));
        }

        if !channel.translation.is_empty()
            || !channel.rotation.is_empty()
            || !channel.scale.is_empty()
        {
            clip.add_channel(joint_idx as u32, channel);
        }
    }

    clip.duration = max_duration;
    clip
}

fn identify_key_frame_index_step(key_frames: &[f32], time: f32) -> usize {
    if key_frames.is_empty() {
        return 0;
    }
    let period = *key_frames
        .last()
        .expect("key_frames verified non-empty by early return above");
    if period <= 0.0 {
        return 0;
    }
    let time = time.rem_euclid(period);
    let idx = key_frames.partition_point(|&kf| kf <= time);
    idx.min(key_frames.len() - 1)
}

fn convert_node_animations_to_clip(
    node_animations: &[NodeAnimation],
    rrnodes: &[RRNode],
    animation_system: &AnimationSystem,
    skeleton_id: u32,
) -> AnimationClip {
    let mut clip = AnimationClip::new("gltf_node_animation");

    let skeleton = match animation_system.get_skeleton(skeleton_id) {
        Some(s) => s,
        None => return clip,
    };

    let mut max_duration = 0.0f32;

    for node_anim in node_animations {
        let node = rrnodes
            .iter()
            .find(|n| n.index as usize == node_anim.node_index);
        let bone_id = node.and_then(|n| skeleton.bone_name_to_id.get(&n.name).copied());

        let Some(bid) = bone_id else {
            continue;
        };

        let mut channel = TransformChannel::default();
        let interp = &node_anim.interpolation;

        for (i, &time) in node_anim.translation_keyframes.iter().enumerate() {
            if i < node_anim.translations.len() {
                let mut kf =
                    Keyframe::with_interpolation(time, node_anim.translations[i], interp.clone());
                if i < node_anim.translation_in_tangents.len() {
                    kf.in_tangent = Some(node_anim.translation_in_tangents[i]);
                    kf.out_tangent = Some(node_anim.translation_out_tangents[i]);
                }
                channel.translation.push(kf);
                if time > max_duration {
                    max_duration = time;
                }
            }
        }

        for (i, &time) in node_anim.rotation_keyframes.iter().enumerate() {
            if i < node_anim.rotations.len() {
                let mut kf =
                    Keyframe::with_interpolation(time, node_anim.rotations[i], interp.clone());
                if i < node_anim.rotation_in_tangents.len() {
                    kf.in_tangent = Some(node_anim.rotation_in_tangents[i]);
                    kf.out_tangent = Some(node_anim.rotation_out_tangents[i]);
                }
                channel.rotation.push(kf);
                if time > max_duration {
                    max_duration = time;
                }
            }
        }

        for (i, &time) in node_anim.scale_keyframes.iter().enumerate() {
            if i < node_anim.scales.len() {
                let mut kf =
                    Keyframe::with_interpolation(time, node_anim.scales[i], interp.clone());
                if i < node_anim.scale_in_tangents.len() {
                    kf.in_tangent = Some(node_anim.scale_in_tangents[i]);
                    kf.out_tangent = Some(node_anim.scale_out_tangents[i]);
                }
                channel.scale.push(kf);
                if time > max_duration {
                    max_duration = time;
                }
            }
        }

        if !channel.translation.is_empty()
            || !channel.rotation.is_empty()
            || !channel.scale.is_empty()
        {
            clip.add_channel(bid, channel);
        }
    }

    clip.duration = max_duration;
    clip
}

fn mat4_from_array(arr: [[f32; 4]; 4]) -> Matrix4<f32> {
    Matrix4::from_cols(
        Vector4::new(arr[0][0], arr[0][1], arr[0][2], arr[0][3]),
        Vector4::new(arr[1][0], arr[1][1], arr[1][2], arr[1][3]),
        Vector4::new(arr[2][0], arr[2][1], arr[2][2], arr[2][3]),
        Vector4::new(arr[3][0], arr[3][1], arr[3][2], arr[3][3]),
    )
}

fn array_from_mat4(m: Matrix4<f32>) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[0][1], m[0][2], m[0][3]],
        [m[1][0], m[1][1], m[1][2], m[1][3]],
        [m[2][0], m[2][1], m[2][2], m[2][3]],
        [m[3][0], m[3][1], m[3][2], m[3][3]],
    ]
}

fn matrix_to_quaternion(m: &Matrix4<f32>) -> Quaternion<f32> {
    let trace = m[0][0] + m[1][1] + m[2][2];

    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        Quaternion::new(
            0.25 * s,
            (m[1][2] - m[2][1]) / s,
            (m[2][0] - m[0][2]) / s,
            (m[0][1] - m[1][0]) / s,
        )
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        Quaternion::new(
            (m[1][2] - m[2][1]) / s,
            0.25 * s,
            (m[1][0] + m[0][1]) / s,
            (m[2][0] + m[0][2]) / s,
        )
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        Quaternion::new(
            (m[2][0] - m[0][2]) / s,
            (m[1][0] + m[0][1]) / s,
            0.25 * s,
            (m[2][1] + m[1][2]) / s,
        )
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        Quaternion::new(
            (m[0][1] - m[1][0]) / s,
            (m[2][0] + m[0][2]) / s,
            (m[2][1] + m[1][2]) / s,
            0.25 * s,
        )
    }
}
