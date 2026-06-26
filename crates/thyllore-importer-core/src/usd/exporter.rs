use std::collections::HashSet;

use anyhow::{anyhow, Result};
use cgmath::{Matrix4, Quaternion, SquareMatrix, Vector3, Vector4};

use openusd::gf::{Matrix4d, Quatf, Vec2f, Vec3f};
use openusd::sdf::{self, Layer, LayerFormat, Specifier, Value, Variability};

use thyllore_anim_core::{decompose_transform, AnimationClip, MorphAnimation, Skeleton};
use thyllore_model_core::mesh::Vertex;
use thyllore_model_core::SkinData;

const MAX_SAMPLED_FRAMES: i64 = 100_000;
const MAX_INFLUENCES: usize = 4;

const ROOT_PATH: &str = "/Model";
const SKELETON_PATH: &str = "/Model/Skel";
const ANIMATION_PATH: &str = "/Model/Anim";

pub struct UsdExportBlendShape {
    pub name: String,
    pub position_offsets: Vec<[f32; 3]>,
    pub normal_offsets: Vec<[f32; 3]>,
}

pub struct UsdExportMesh<'a> {
    pub name: &'a str,
    pub vertices: &'a [Vertex],
    pub indices: &'a [u32],
    pub skin: Option<&'a SkinData>,
    pub blend_shapes: &'a [UsdExportBlendShape],
}

pub struct UsdExportScene<'a> {
    pub skeleton: Option<&'a Skeleton>,
    pub clip: Option<&'a AnimationClip>,
    pub blend_shape_order: &'a [String],
    pub blend_shape_weights: &'a [MorphAnimation],
    pub meshes: &'a [UsdExportMesh<'a>],
    pub start_time_code: f64,
    pub end_time_code: f64,
    pub time_codes_per_second: f64,
}

#[derive(Clone, Copy)]
struct ExportTime {
    start: f64,
    end: f64,
    rate: f64,
}

pub fn save_usd_file(path: &str, scene: &UsdExportScene) -> Result<()> {
    let mut layer = Layer::new_anonymous(path);
    author_time_metadata(&mut layer, scene)?;

    layer
        .create_prim(sdf::path(ROOT_PATH)?, Specifier::Def, "SkelRoot")
        .map_err(|e| anyhow!("failed to author SkelRoot: {e}"))?;
    layer
        .set_default_prim("Model")
        .map_err(|e| anyhow!("failed to set default prim: {e}"))?;

    let joint_paths = scene.skeleton.map(build_joint_paths).unwrap_or_default();
    author_skeleton_and_animation(&mut layer, scene, &joint_paths)?;

    for mesh in scene.meshes {
        author_mesh(&mut layer, mesh, scene.skeleton.is_some())?;
    }

    let format = LayerFormat::from_extension(extension_of(path))
        .ok_or_else(|| anyhow!("unsupported export extension for {path}"))?;
    layer
        .save_as(path, format)
        .map_err(|e| anyhow!("failed to write USD file {path}: {e}"))
}

fn author_time_metadata(layer: &mut Layer, scene: &UsdExportScene) -> Result<()> {
    let mut root = layer
        .pseudo_root_mut()
        .map_err(|e| anyhow!("failed to access layer root: {e}"))?;
    root.add(
        sdf::FieldKey::StartTimeCode,
        Value::Double(scene.start_time_code),
    );
    root.add(
        sdf::FieldKey::EndTimeCode,
        Value::Double(scene.end_time_code),
    );
    root.add(
        sdf::FieldKey::TimeCodesPerSecond,
        Value::Double(scene.time_codes_per_second),
    );
    root.add(
        sdf::FieldKey::FramesPerSecond,
        Value::Double(scene.time_codes_per_second),
    );
    Ok(())
}

fn author_skeleton_and_animation(
    layer: &mut Layer,
    scene: &UsdExportScene,
    joint_paths: &[String],
) -> Result<()> {
    let Some(skeleton) = scene.skeleton else {
        return Ok(());
    };

    author_skeleton(layer, skeleton, joint_paths)?;
    author_animation(layer, scene, skeleton, joint_paths)
}

fn author_skeleton(layer: &mut Layer, skeleton: &Skeleton, joint_paths: &[String]) -> Result<()> {
    {
        let mut prim = layer
            .create_prim(sdf::path(SKELETON_PATH)?, Specifier::Def, "Skeleton")
            .map_err(|e| anyhow!("failed to author Skeleton: {e}"))?;
        prim.add_applied_schema("SkelBindingAPI")
            .map_err(|e| anyhow!("failed to apply SkelBindingAPI: {e}"))?;
    }

    let bind = skeleton
        .bones
        .iter()
        .map(|b| {
            export_matrix(
                &b.inverse_bind_pose
                    .invert()
                    .unwrap_or_else(Matrix4::identity),
            )
        })
        .collect();
    let rest = skeleton
        .bones
        .iter()
        .map(|b| export_matrix(&b.local_transform))
        .collect();

    set_uniform_default(
        layer,
        SKELETON_PATH,
        "joints",
        "token[]",
        token_vec(joint_paths),
    )?;
    set_uniform_default(
        layer,
        SKELETON_PATH,
        "bindTransforms",
        "matrix4d[]",
        Value::Matrix4dVec(bind),
    )?;
    set_uniform_default(
        layer,
        SKELETON_PATH,
        "restTransforms",
        "matrix4d[]",
        Value::Matrix4dVec(rest),
    )?;

    let mut rel = layer
        .create_relationship(
            sdf::path(&format!("{SKELETON_PATH}.skel:animationSource"))?,
            Variability::Varying,
            false,
        )
        .map_err(|e| anyhow!("failed to author animationSource: {e}"))?;
    rel.set_target_paths([sdf::path(ANIMATION_PATH)?]);
    Ok(())
}

fn author_animation(
    layer: &mut Layer,
    scene: &UsdExportScene,
    skeleton: &Skeleton,
    joint_paths: &[String],
) -> Result<()> {
    layer
        .create_prim(sdf::path(ANIMATION_PATH)?, Specifier::Def, "SkelAnimation")
        .map_err(|e| anyhow!("failed to author SkelAnimation: {e}"))?;
    set_uniform_default(
        layer,
        ANIMATION_PATH,
        "joints",
        "token[]",
        token_vec(joint_paths),
    )?;

    let time = ExportTime {
        start: scene.start_time_code,
        end: scene.end_time_code,
        rate: scene.time_codes_per_second.max(1.0),
    };
    author_joint_tracks(layer, skeleton, scene.clip, time)?;
    author_blend_shape_tracks(layer, scene, time)?;
    Ok(())
}

fn author_joint_tracks(
    layer: &mut Layer,
    skeleton: &Skeleton,
    clip: Option<&AnimationClip>,
    time: ExportTime,
) -> Result<()> {
    let rest = rest_pose(skeleton);

    let Some(clip) = clip else {
        let (t, r, s) = split_pose(&rest);
        set_varying_default(
            layer,
            ANIMATION_PATH,
            "translations",
            "float3[]",
            Value::Vec3fVec(t),
        )?;
        set_varying_default(
            layer,
            ANIMATION_PATH,
            "rotations",
            "quatf[]",
            Value::QuatfVec(r),
        )?;
        set_varying_default(
            layer,
            ANIMATION_PATH,
            "scales",
            "float3[]",
            Value::Vec3fVec(s),
        )?;
        return Ok(());
    };

    new_attr(layer, ANIMATION_PATH, "translations", "float3[]")?;
    new_attr(layer, ANIMATION_PATH, "rotations", "quatf[]")?;
    new_attr(layer, ANIMATION_PATH, "scales", "float3[]")?;

    for step in 0..=frame_count(time) {
        let time_code = time.start + step as f64;
        let seconds = ((time_code - time.start) / time.rate) as f32;
        let (t, r, s) = sample_pose(skeleton, clip, &rest, seconds);

        set_time_sample(
            layer,
            ANIMATION_PATH,
            "translations",
            time_code,
            Value::Vec3fVec(t),
        )?;
        set_time_sample(
            layer,
            ANIMATION_PATH,
            "rotations",
            time_code,
            Value::QuatfVec(r),
        )?;
        set_time_sample(
            layer,
            ANIMATION_PATH,
            "scales",
            time_code,
            Value::Vec3fVec(s),
        )?;
    }
    Ok(())
}

fn author_blend_shape_tracks(
    layer: &mut Layer,
    scene: &UsdExportScene,
    time: ExportTime,
) -> Result<()> {
    if scene.blend_shape_order.is_empty() {
        return Ok(());
    }

    set_uniform_default(
        layer,
        ANIMATION_PATH,
        "blendShapes",
        "token[]",
        token_vec(scene.blend_shape_order),
    )?;

    if scene.blend_shape_weights.is_empty() {
        let zeros = vec![0.0f32; scene.blend_shape_order.len()];
        set_varying_default(
            layer,
            ANIMATION_PATH,
            "blendShapeWeights",
            "float[]",
            Value::FloatVec(zeros),
        )?;
        return Ok(());
    }

    new_attr(layer, ANIMATION_PATH, "blendShapeWeights", "float[]")?;
    for animation in scene.blend_shape_weights {
        let time_code = time.start + (animation.key_frame as f64) * time.rate;
        set_time_sample(
            layer,
            ANIMATION_PATH,
            "blendShapeWeights",
            time_code,
            Value::FloatVec(animation.weights.clone()),
        )?;
    }
    Ok(())
}

fn author_mesh(layer: &mut Layer, mesh: &UsdExportMesh, has_skeleton: bool) -> Result<()> {
    let mesh_path = format!("{ROOT_PATH}/{}", sanitize_identifier(mesh.name));
    let skin = mesh.skin.filter(|_| has_skeleton);
    let has_blends = !mesh.blend_shapes.is_empty();

    {
        let mut prim = layer
            .create_prim(sdf::path(&mesh_path)?, Specifier::Def, "Mesh")
            .map_err(|e| anyhow!("failed to author Mesh: {e}"))?;
        if skin.is_some() || has_blends {
            prim.add_applied_schema("SkelBindingAPI")
                .map_err(|e| anyhow!("failed to apply SkelBindingAPI: {e}"))?;
        }
    }

    author_mesh_geometry(layer, &mesh_path, mesh)?;

    if let Some(skin) = skin {
        author_mesh_skin(layer, &mesh_path, skin)?;
    }
    if has_blends {
        author_mesh_blend_shapes(layer, &mesh_path, mesh)?;
    }
    Ok(())
}

fn author_mesh_geometry(layer: &mut Layer, mesh_path: &str, mesh: &UsdExportMesh) -> Result<()> {
    let points = mesh
        .vertices
        .iter()
        .map(|v| Vec3f {
            x: v.pos[0],
            y: v.pos[1],
            z: v.pos[2],
        })
        .collect();
    let normals = mesh
        .vertices
        .iter()
        .map(|v| Vec3f {
            x: v.normal[0],
            y: v.normal[1],
            z: v.normal[2],
        })
        .collect();
    let uvs = mesh
        .vertices
        .iter()
        .map(|v| Vec2f {
            x: v.tex_coord[0],
            y: v.tex_coord[1],
        })
        .collect();

    let triangle_count = mesh.indices.len() / 3;
    let counts = vec![3i32; triangle_count];
    let face_indices = mesh.indices.iter().map(|&i| i as i32).collect();

    set_varying_default(
        layer,
        mesh_path,
        "points",
        "point3f[]",
        Value::Vec3fVec(points),
    )?;
    set_varying_default(
        layer,
        mesh_path,
        "faceVertexCounts",
        "int[]",
        Value::IntVec(counts),
    )?;
    set_varying_default(
        layer,
        mesh_path,
        "faceVertexIndices",
        "int[]",
        Value::IntVec(face_indices),
    )?;

    set_primvar(
        layer,
        mesh_path,
        "normals",
        "normal3f[]",
        Value::Vec3fVec(normals),
    )?;
    set_primvar(
        layer,
        mesh_path,
        "primvars:st",
        "texCoord2f[]",
        Value::Vec2fVec(uvs),
    )?;
    Ok(())
}

fn author_mesh_skin(layer: &mut Layer, mesh_path: &str, skin: &SkinData) -> Result<()> {
    let indices = flatten_indices(&skin.bone_indices);
    let weights = flatten_weights(&skin.bone_weights);

    set_influence_primvar(
        layer,
        mesh_path,
        "primvars:skel:jointIndices",
        "int[]",
        Value::IntVec(indices),
    )?;
    set_influence_primvar(
        layer,
        mesh_path,
        "primvars:skel:jointWeights",
        "float[]",
        Value::FloatVec(weights),
    )?;

    let mut rel = layer
        .create_relationship(
            sdf::path(&format!("{mesh_path}.skel:skeleton"))?,
            Variability::Varying,
            false,
        )
        .map_err(|e| anyhow!("failed to author skel:skeleton: {e}"))?;
    rel.set_target_paths([sdf::path(SKELETON_PATH)?]);
    Ok(())
}

fn author_mesh_blend_shapes(
    layer: &mut Layer,
    mesh_path: &str,
    mesh: &UsdExportMesh,
) -> Result<()> {
    let mut names = Vec::with_capacity(mesh.blend_shapes.len());
    let mut target_paths = Vec::with_capacity(mesh.blend_shapes.len());

    for blend_shape in mesh.blend_shapes {
        let leaf = sanitize_identifier(&blend_shape.name);
        let bs_path = format!("{mesh_path}/{leaf}");
        author_blend_shape(layer, &bs_path, blend_shape)?;
        names.push(blend_shape.name.clone());
        target_paths.push(sdf::path(&bs_path)?);
    }

    set_uniform_default(
        layer,
        mesh_path,
        "skel:blendShapes",
        "token[]",
        Value::TokenVec(names),
    )?;
    let mut rel = layer
        .create_relationship(
            sdf::path(&format!("{mesh_path}.skel:blendShapeTargets"))?,
            Variability::Varying,
            false,
        )
        .map_err(|e| anyhow!("failed to author blendShapeTargets: {e}"))?;
    rel.set_target_paths(target_paths);
    Ok(())
}

fn author_blend_shape(
    layer: &mut Layer,
    bs_path: &str,
    blend_shape: &UsdExportBlendShape,
) -> Result<()> {
    layer
        .create_prim(sdf::path(bs_path)?, Specifier::Def, "BlendShape")
        .map_err(|e| anyhow!("failed to author BlendShape: {e}"))?;

    let offsets = vec3f_vec_from(&blend_shape.position_offsets);
    set_uniform_default(
        layer,
        bs_path,
        "offsets",
        "vector3f[]",
        Value::Vec3fVec(offsets),
    )?;

    if !blend_shape.normal_offsets.is_empty() {
        let normals = vec3f_vec_from(&blend_shape.normal_offsets);
        set_uniform_default(
            layer,
            bs_path,
            "normalOffsets",
            "vector3f[]",
            Value::Vec3fVec(normals),
        )?;
    }
    Ok(())
}

fn rest_pose(skeleton: &Skeleton) -> Vec<(Vector3<f32>, Quaternion<f32>, Vector3<f32>)> {
    skeleton
        .bones
        .iter()
        .map(|bone| decompose_transform(&bone.local_transform))
        .collect()
}

fn split_pose(
    pose: &[(Vector3<f32>, Quaternion<f32>, Vector3<f32>)],
) -> (Vec<Vec3f>, Vec<Quatf>, Vec<Vec3f>) {
    let translations = pose.iter().map(|(t, _, _)| to_vec3f(*t)).collect();
    let rotations = pose.iter().map(|(_, r, _)| to_quatf(*r)).collect();
    let scales = pose.iter().map(|(_, _, s)| to_vec3f(*s)).collect();
    (translations, rotations, scales)
}

fn sample_pose(
    skeleton: &Skeleton,
    clip: &AnimationClip,
    rest: &[(Vector3<f32>, Quaternion<f32>, Vector3<f32>)],
    seconds: f32,
) -> (Vec<Vec3f>, Vec<Quatf>, Vec<Vec3f>) {
    let mut translations = Vec::with_capacity(skeleton.bones.len());
    let mut rotations = Vec::with_capacity(skeleton.bones.len());
    let mut scales = Vec::with_capacity(skeleton.bones.len());

    for bone in &skeleton.bones {
        let (rest_t, rest_r, rest_s) = rest[bone.id as usize];
        let channel = clip.channels.get(&bone.id);

        let t = channel
            .and_then(|c| c.sample_translation(seconds))
            .unwrap_or(rest_t);
        let r = channel
            .and_then(|c| c.sample_rotation(seconds))
            .unwrap_or(rest_r);
        let s = channel
            .and_then(|c| c.sample_scale(seconds))
            .unwrap_or(rest_s);

        translations.push(to_vec3f(t));
        rotations.push(to_quatf(r));
        scales.push(to_vec3f(s));
    }
    (translations, rotations, scales)
}

fn build_joint_paths(skeleton: &Skeleton) -> Vec<String> {
    let mut paths = vec![String::new(); skeleton.bones.len()];
    let mut used: HashSet<String> = HashSet::new();

    for bone in &skeleton.bones {
        let name = sanitize_identifier(&bone.name);
        let parent_path = bone
            .parent_id
            .and_then(|p| paths.get(p as usize))
            .filter(|p| !p.is_empty());

        let mut full = match parent_path {
            Some(parent) => format!("{parent}/{name}"),
            None => name,
        };
        if used.contains(&full) {
            full = format!("{full}_{}", bone.id);
        }

        used.insert(full.clone());
        if let Some(slot) = paths.get_mut(bone.id as usize) {
            *slot = full;
        }
    }
    paths
}

fn flatten_indices(indices: &[Vector4<u32>]) -> Vec<i32> {
    let mut out = Vec::with_capacity(indices.len() * MAX_INFLUENCES);
    for influence in indices {
        out.extend_from_slice(&[
            influence.x as i32,
            influence.y as i32,
            influence.z as i32,
            influence.w as i32,
        ]);
    }
    out
}

fn flatten_weights(weights: &[Vector4<f32>]) -> Vec<f32> {
    let mut out = Vec::with_capacity(weights.len() * MAX_INFLUENCES);
    for influence in weights {
        out.extend_from_slice(&[influence.x, influence.y, influence.z, influence.w]);
    }
    out
}

fn set_uniform_default(
    layer: &mut Layer,
    prim_path: &str,
    name: &str,
    type_name: &str,
    value: Value,
) -> Result<()> {
    let mut attr = new_named_attr(layer, prim_path, name, type_name, Variability::Uniform)?;
    attr.set_default(value);
    Ok(())
}

fn set_varying_default(
    layer: &mut Layer,
    prim_path: &str,
    name: &str,
    type_name: &str,
    value: Value,
) -> Result<()> {
    let mut attr = new_named_attr(layer, prim_path, name, type_name, Variability::Varying)?;
    attr.set_default(value);
    Ok(())
}

fn set_primvar(
    layer: &mut Layer,
    prim_path: &str,
    name: &str,
    type_name: &str,
    value: Value,
) -> Result<()> {
    let mut attr = new_named_attr(layer, prim_path, name, type_name, Variability::Varying)?;
    attr.set_default(value);
    attr.add("interpolation", Value::Token("vertex".into()));
    Ok(())
}

fn set_influence_primvar(
    layer: &mut Layer,
    prim_path: &str,
    name: &str,
    type_name: &str,
    value: Value,
) -> Result<()> {
    let mut attr = new_named_attr(layer, prim_path, name, type_name, Variability::Varying)?;
    attr.set_default(value);
    attr.add("interpolation", Value::Token("vertex".into()));
    attr.add("elementSize", Value::Int(MAX_INFLUENCES as i32));
    Ok(())
}

fn set_time_sample(
    layer: &mut Layer,
    prim_path: &str,
    name: &str,
    time_code: f64,
    value: Value,
) -> Result<()> {
    let attr_path = sdf::path(&format!("{prim_path}.{name}"))?;
    let mut attr = layer
        .attribute_mut(attr_path)
        .ok_or_else(|| anyhow!("attribute {prim_path}.{name} not found for time sample"))?;
    attr.set_time_sample(time_code, value);
    Ok(())
}

fn new_attr<'a>(
    layer: &'a mut Layer,
    prim_path: &str,
    name: &str,
    type_name: &str,
) -> Result<openusd::sdf::AttributeSpecMut<'a>> {
    new_named_attr(layer, prim_path, name, type_name, Variability::Varying)
}

fn new_named_attr<'a>(
    layer: &'a mut Layer,
    prim_path: &str,
    name: &str,
    type_name: &str,
    variability: Variability,
) -> Result<openusd::sdf::AttributeSpecMut<'a>> {
    let attr_path = sdf::path(&format!("{prim_path}.{name}"))?;
    layer
        .create_attribute(attr_path, type_name, variability, false)
        .map_err(|e| anyhow!("failed to author {prim_path}.{name}: {e}"))
}

fn frame_count(time: ExportTime) -> i64 {
    ((time.end - time.start).round() as i64).clamp(0, MAX_SAMPLED_FRAMES)
}

fn token_vec(values: &[String]) -> Value {
    Value::TokenVec(values.to_vec())
}

fn vec3f_vec_from(offsets: &[[f32; 3]]) -> Vec<Vec3f> {
    offsets
        .iter()
        .map(|o| Vec3f {
            x: o[0],
            y: o[1],
            z: o[2],
        })
        .collect()
}

fn to_vec3f(v: Vector3<f32>) -> Vec3f {
    Vec3f {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}

fn to_quatf(q: Quaternion<f32>) -> Quatf {
    Quatf {
        w: q.s,
        x: q.v.x,
        y: q.v.y,
        z: q.v.z,
    }
}

fn export_matrix(m: &Matrix4<f32>) -> Matrix4d {
    Matrix4d([
        m.x.x as f64,
        m.x.y as f64,
        m.x.z as f64,
        m.x.w as f64,
        m.y.x as f64,
        m.y.y as f64,
        m.y.z as f64,
        m.y.w as f64,
        m.z.x as f64,
        m.z.y as f64,
        m.z.z as f64,
        m.z.w as f64,
        m.w.x as f64,
        m.w.y as f64,
        m.w.z as f64,
        m.w.w as f64,
    ])
}

fn sanitize_identifier(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    for (i, ch) in name.chars().enumerate() {
        let valid = ch.is_ascii_alphanumeric() || ch == '_';
        if i == 0 && ch.is_ascii_digit() {
            result.push('_');
        }
        result.push(if valid { ch } else { '_' });
    }
    if result.is_empty() {
        result.push('_');
    }
    result
}

fn extension_of(path: &str) -> &str {
    path.rsplit('.').next().unwrap_or_default()
}
