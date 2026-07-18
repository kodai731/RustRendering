use std::io::{Seek, Write};
use std::path::Path;

use cgmath::{InnerSpace, Matrix4, Vector3};
use fbxcel::low::FbxVersion;
use fbxcel::writer::v7400::binary::{FbxFooter, Writer};

use thyllore_anim_core::editable::{
    BezierHandle, EditableAnimationClip, EditableKeyframe, InterpolationType, PropertyCurve,
    TangentWeightMode,
};
use thyllore_anim_core::Skeleton;
use thyllore_importer_core::fbx::fbx::FbxAxesInfo;

pub(crate) type FbxWriteResult<T> = Result<T, Box<dyn std::error::Error>>;

pub(crate) const KTIME_ONE_SECOND: f64 = 46_186_158_000.0;

pub(crate) struct UidAllocator {
    next: i64,
}

impl UidAllocator {
    pub(crate) fn new() -> Self {
        Self { next: 1_000_000 }
    }

    pub(crate) fn allocate(&mut self) -> i64 {
        let uid = self.next;
        self.next += 1;
        uid
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum FbxChannel {
    Translation,
    Rotation,
    Scale,
}

impl FbxChannel {
    pub(crate) fn property_name(&self) -> &'static str {
        match self {
            FbxChannel::Translation => "Lcl Translation",
            FbxChannel::Rotation => "Lcl Rotation",
            FbxChannel::Scale => "Lcl Scaling",
        }
    }

    pub(crate) fn short_name(&self) -> &'static str {
        match self {
            FbxChannel::Translation => "T",
            FbxChannel::Rotation => "R",
            FbxChannel::Scale => "S",
        }
    }
}

pub(crate) struct FbxBoneExport {
    pub model_uid: i64,
    pub name: String,
    pub is_root: bool,
    pub parent_model_uid: Option<i64>,
    pub translation: [f64; 3],
    pub rotation: [f64; 3],
    pub scaling: [f64; 3],
    pub node_attribute_uid: Option<i64>,
}

pub(crate) struct FbxCurveNodeExport {
    pub uid: i64,
    pub bone_model_uid: i64,
    pub channel: FbxChannel,
    pub default_values: [f64; 3],
    pub curve_uids: [Option<i64>; 3],
}

pub(crate) struct FbxCurveExport {
    pub uid: i64,
    pub key_times: Vec<i64>,
    pub key_values: Vec<f32>,
    pub key_attr_flags: Vec<i32>,
    pub key_attr_data: Vec<f32>,
    pub key_attr_ref_count: Vec<i32>,
    pub default_value: f64,
}

pub(crate) enum FbxConnection {
    OO {
        child: i64,
        parent: i64,
    },
    OP {
        child: i64,
        parent: i64,
        property: String,
    },
}

pub(crate) struct FbxExportData {
    pub clip_name: String,
    pub duration_ktime: i64,
    pub needs_coord_conversion: bool,
    pub axes: FbxAxesInfo,
    pub fps: f32,
    pub bones: Vec<FbxBoneExport>,
    pub stack_uid: i64,
    pub layer_uid: i64,
    pub document_uid: i64,
    pub curve_nodes: Vec<FbxCurveNodeExport>,
    pub curves: Vec<FbxCurveExport>,
    pub connections: Vec<FbxConnection>,
}

pub(crate) fn seconds_to_ktime(seconds: f32) -> i64 {
    (seconds as f64 * KTIME_ONE_SECOND) as i64
}

fn convert_interpolation_to_flags(interp: InterpolationType) -> i32 {
    match interp {
        InterpolationType::Linear => 0x0000_0004,
        InterpolationType::Bezier => 0x0000_0408,
        InterpolationType::Stepped => 0x0000_0002,
    }
}

fn convert_tangent_to_fbx_slope_weight(handle: &BezierHandle, key_interval: f32) -> (f32, f32) {
    let slope = if handle.time_offset.abs() > 1e-8 {
        handle.value_offset / handle.time_offset
    } else {
        0.0
    };

    let weight = if key_interval.abs() > 1e-8 {
        (handle.time_offset.abs() / (key_interval / 3.0)).clamp(0.0, 1.0)
    } else {
        0.333
    };

    (slope, weight)
}

pub(crate) fn decompose_matrix_to_trs(m: &Matrix4<f32>) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let tx = m[3][0] as f64;
    let ty = m[3][1] as f64;
    let tz = m[3][2] as f64;

    let sx = Vector3::new(m[0][0], m[0][1], m[0][2]).magnitude() as f64;
    let sy = Vector3::new(m[1][0], m[1][1], m[1][2]).magnitude() as f64;
    let sz = Vector3::new(m[2][0], m[2][1], m[2][2]).magnitude() as f64;

    let sx_safe = if sx > 1e-8 { sx } else { 1.0 };
    let sy_safe = if sy > 1e-8 { sy } else { 1.0 };
    let sz_safe = if sz > 1e-8 { sz } else { 1.0 };

    let r00 = m[0][0] as f64 / sx_safe;
    let r01 = m[0][1] as f64 / sx_safe;
    let r02 = m[0][2] as f64 / sx_safe;
    let _r10 = m[1][0] as f64 / sy_safe;
    let r11 = m[1][1] as f64 / sy_safe;
    let r12 = m[1][2] as f64 / sy_safe;
    let _r20 = m[2][0] as f64 / sz_safe;
    let r21 = m[2][1] as f64 / sz_safe;
    let r22 = m[2][2] as f64 / sz_safe;

    let ry = (-r02).asin();
    let (rx, rz) = if ry.cos().abs() > 1e-6 {
        (r12.atan2(r22), r01.atan2(r00))
    } else {
        ((-r21).atan2(r11), 0.0)
    };

    (
        [tx, ty, tz],
        [rx.to_degrees(), ry.to_degrees(), rz.to_degrees()],
        [sx, sy, sz],
    )
}

fn validate_bone_names(clip: &EditableAnimationClip, skeleton: &Skeleton) -> Vec<String> {
    clip.tracks
        .values()
        .filter(|track| !skeleton.bone_name_to_id.contains_key(&track.bone_name))
        .map(|track| track.bone_name.clone())
        .collect()
}

pub(crate) fn build_bone_export_list(
    skeleton: &Skeleton,
    uid_alloc: &mut UidAllocator,
    mesh_node_names: &std::collections::HashSet<String>,
    inv_unit_scale: f32,
    needs_coord_conversion: bool,
) -> Vec<FbxBoneExport> {
    let mut bones = Vec::new();
    let mut skeleton_idx_to_export_idx: Vec<Option<usize>> =
        Vec::with_capacity(skeleton.bones.len());

    for bone in &skeleton.bones {
        if mesh_node_names.contains(&bone.name) {
            skeleton_idx_to_export_idx.push(None);
            continue;
        }

        let export_idx = bones.len();
        skeleton_idx_to_export_idx.push(Some(export_idx));

        let model_uid = uid_alloc.allocate();
        let is_root = bone.parent_id.is_none();

        let is_root_or_root_child = bone.parent_id.is_none()
            || bone.name == "RootNode"
            || bone
                .parent_id
                .and_then(|pid| skeleton.get_bone(pid))
                .map_or(false, |parent| parent.name == "RootNode");

        let export_transform = if needs_coord_conversion && is_root_or_root_child {
            reverse_coord_conversion_for_export(&bone.local_transform)
        } else {
            bone.local_transform
        };
        let (mut translation, rotation, scaling) = decompose_matrix_to_trs(&export_transform);

        let scale = inv_unit_scale as f64;
        translation[0] *= scale;
        translation[1] *= scale;
        translation[2] *= scale;

        let node_attribute_uid = if needs_coord_conversion {
            Some(uid_alloc.allocate())
        } else {
            None
        };

        bones.push(FbxBoneExport {
            model_uid,
            name: bone.name.clone(),
            is_root,
            parent_model_uid: None,
            translation,
            rotation,
            scaling,
            node_attribute_uid,
        });
    }

    resolve_bone_parent_uids(&mut bones, skeleton, &skeleton_idx_to_export_idx);
    bones
}

fn resolve_bone_parent_uids(
    bones: &mut [FbxBoneExport],
    skeleton: &Skeleton,
    skeleton_idx_to_export_idx: &[Option<usize>],
) {
    let model_uids: Vec<i64> = bones.iter().map(|b| b.model_uid).collect();

    for (skel_idx, bone) in skeleton.bones.iter().enumerate() {
        let Some(export_idx) = skeleton_idx_to_export_idx[skel_idx] else {
            continue;
        };

        if let Some(parent_id) = bone.parent_id {
            if let Some(parent_export_idx) = skeleton_idx_to_export_idx[parent_id as usize] {
                bones[export_idx].parent_model_uid = Some(model_uids[parent_export_idx]);
            }
        }
    }
}

fn reverse_coord_conversion_for_export(local_transform: &Matrix4<f32>) -> Matrix4<f32> {
    use thyllore_math_core::coordinate_system::world_to_fbx;
    let inv = world_to_fbx();
    inv * *local_transform
}

fn build_key_attr_arrays(
    keyframes: &[EditableKeyframe],
    flags_out: &mut Vec<i32>,
    data_out: &mut Vec<f32>,
    ref_count_out: &mut Vec<i32>,
) {
    if keyframes.is_empty() {
        return;
    }

    let mut current_flag: i32 = 0;
    let mut current_data: [f32; 4] = [0.0; 4];
    let mut current_count: i32 = 0;

    // FBX SDK tangent weight flags (OR'd with interpolation flags for Bezier curves)
    const WEIGHTED_RIGHT: i32 = 0x0100_0000;
    const WEIGHTED_NEXT_LEFT: i32 = 0x0400_0000;

    for (i, kf) in keyframes.iter().enumerate() {
        let mut flag = convert_interpolation_to_flags(kf.interpolation);
        let next_kf = keyframes.get(i + 1);

        // Set weighted tangent flags when current or next keyframe uses weighted mode
        if kf.weight_mode == TangentWeightMode::Weighted {
            flag |= WEIGHTED_RIGHT;
        }
        if next_kf.is_some_and(|n| n.weight_mode == TangentWeightMode::Weighted) {
            flag |= WEIGHTED_NEXT_LEFT;
        }

        let key_interval = next_kf.map(|n| n.time - kf.time).unwrap_or(1.0 / 30.0);

        let (right_slope, right_weight) =
            convert_tangent_to_fbx_slope_weight(&kf.out_tangent, key_interval);

        let (next_left_slope, next_left_weight) = next_kf
            .map(|n| convert_tangent_to_fbx_slope_weight(&n.in_tangent, key_interval))
            .unwrap_or((0.0, 0.0));

        let data = [right_slope, next_left_slope, right_weight, next_left_weight];

        if current_count > 0 && current_flag == flag && current_data == data {
            current_count += 1;
        } else {
            if current_count > 0 {
                flags_out.push(current_flag);
                data_out.extend_from_slice(&current_data);
                ref_count_out.push(current_count);
            }
            current_flag = flag;
            current_data = data;
            current_count = 1;
        }
    }

    if current_count > 0 {
        flags_out.push(current_flag);
        data_out.extend_from_slice(&current_data);
        ref_count_out.push(current_count);
    }
}

pub(crate) fn build_curve_export(
    curve: &PropertyCurve,
    uid: i64,
    value_scale: f32,
) -> FbxCurveExport {
    let key_times: Vec<i64> = curve
        .keyframes
        .iter()
        .map(|kf| seconds_to_ktime(kf.time))
        .collect();

    let key_values: Vec<f32> = curve
        .keyframes
        .iter()
        .map(|kf| kf.value * value_scale)
        .collect();

    let default_value = curve
        .keyframes
        .first()
        .map(|kf| kf.value as f64 * value_scale as f64)
        .unwrap_or(0.0);

    let mut key_attr_flags = Vec::new();
    let mut key_attr_data = Vec::new();
    let mut key_attr_ref_count = Vec::new();
    build_key_attr_arrays(
        &curve.keyframes,
        &mut key_attr_flags,
        &mut key_attr_data,
        &mut key_attr_ref_count,
    );

    FbxCurveExport {
        uid,
        key_times,
        key_values,
        key_attr_flags,
        key_attr_data,
        key_attr_ref_count,
        default_value,
    }
}

pub(crate) fn build_channel_exports(
    curves: [&PropertyCurve; 3],
    bone_model_uid: i64,
    channel: FbxChannel,
    uid_alloc: &mut UidAllocator,
    inv_unit_scale: f32,
) -> Option<(FbxCurveNodeExport, Vec<FbxCurveExport>)> {
    let has_any = curves.iter().any(|c| !c.is_empty());
    if !has_any {
        return None;
    }

    let value_scale = match channel {
        FbxChannel::Translation => inv_unit_scale,
        FbxChannel::Rotation | FbxChannel::Scale => 1.0,
    };

    let curvenode_uid = uid_alloc.allocate();
    let mut curve_uids: [Option<i64>; 3] = [None; 3];
    let mut curve_exports = Vec::new();

    for (i, curve) in curves.iter().enumerate() {
        if !curve.is_empty() {
            let curve_uid = uid_alloc.allocate();
            curve_uids[i] = Some(curve_uid);
            curve_exports.push(build_curve_export(curve, curve_uid, value_scale));
        }
    }

    let scale_f64 = value_scale as f64;
    let default_values = [
        curves[0]
            .keyframes
            .first()
            .map(|k| k.value as f64 * scale_f64)
            .unwrap_or(0.0),
        curves[1]
            .keyframes
            .first()
            .map(|k| k.value as f64 * scale_f64)
            .unwrap_or(0.0),
        curves[2]
            .keyframes
            .first()
            .map(|k| k.value as f64 * scale_f64)
            .unwrap_or(0.0),
    ];

    let node = FbxCurveNodeExport {
        uid: curvenode_uid,
        bone_model_uid,
        channel,
        default_values,
        curve_uids,
    };

    Some((node, curve_exports))
}

fn generate_connections(data: &mut FbxExportData) {
    let mut conns = Vec::new();

    for bone in &data.bones {
        let parent_uid = bone.parent_model_uid.unwrap_or(0);
        conns.push(FbxConnection::OO {
            child: bone.model_uid,
            parent: parent_uid,
        });

        if let Some(attr_uid) = bone.node_attribute_uid {
            conns.push(FbxConnection::OO {
                child: attr_uid,
                parent: bone.model_uid,
            });
        }
    }

    conns.push(FbxConnection::OO {
        child: data.stack_uid,
        parent: 0,
    });

    conns.push(FbxConnection::OO {
        child: data.layer_uid,
        parent: data.stack_uid,
    });

    for cn in &data.curve_nodes {
        conns.push(FbxConnection::OO {
            child: cn.uid,
            parent: data.layer_uid,
        });

        conns.push(FbxConnection::OP {
            child: cn.uid,
            parent: cn.bone_model_uid,
            property: cn.channel.property_name().to_string(),
        });

        let axis_names = ["d|X", "d|Y", "d|Z"];
        for (i, axis) in axis_names.iter().enumerate() {
            if let Some(curve_uid) = cn.curve_uids[i] {
                conns.push(FbxConnection::OP {
                    child: curve_uid,
                    parent: cn.uid,
                    property: axis.to_string(),
                });
            }
        }
    }

    data.connections = conns;
}

fn build_export_data(
    clip: &EditableAnimationClip,
    skeleton: &Skeleton,
    needs_coord_conversion: bool,
    axes: FbxAxesInfo,
    fps: f32,
) -> anyhow::Result<FbxExportData> {
    let missing = validate_bone_names(clip, skeleton);
    if !missing.is_empty() {
        anyhow::bail!("Bone name mismatch. Missing bones: {:?}", missing);
    }

    let inv_unit_scale = 100.0_f32;

    let mut uid_alloc = UidAllocator::new();
    let empty_set = std::collections::HashSet::new();
    let bones = build_bone_export_list(
        skeleton,
        &mut uid_alloc,
        &empty_set,
        inv_unit_scale,
        needs_coord_conversion,
    );

    let stack_uid = uid_alloc.allocate();
    let layer_uid = uid_alloc.allocate();
    let document_uid = uid_alloc.allocate();
    let duration_ktime = seconds_to_ktime(clip.duration);

    let bone_name_to_model_uid: std::collections::HashMap<&str, i64> = bones
        .iter()
        .map(|b| (b.name.as_str(), b.model_uid))
        .collect();

    let mut curve_nodes = Vec::new();
    let mut curves = Vec::new();

    for track in clip.tracks.values() {
        let bone_model_uid = match bone_name_to_model_uid.get(track.bone_name.as_str()) {
            Some(&uid) => uid,
            None => continue,
        };

        if let Some((node, node_curves)) = build_channel_exports(
            [
                &track.translation_x,
                &track.translation_y,
                &track.translation_z,
            ],
            bone_model_uid,
            FbxChannel::Translation,
            &mut uid_alloc,
            inv_unit_scale,
        ) {
            curve_nodes.push(node);
            curves.extend(node_curves);
        }

        if let Some((node, node_curves)) = build_channel_exports(
            [&track.rotation_x, &track.rotation_y, &track.rotation_z],
            bone_model_uid,
            FbxChannel::Rotation,
            &mut uid_alloc,
            inv_unit_scale,
        ) {
            curve_nodes.push(node);
            curves.extend(node_curves);
        }

        if let Some((node, node_curves)) = build_channel_exports(
            [&track.scale_x, &track.scale_y, &track.scale_z],
            bone_model_uid,
            FbxChannel::Scale,
            &mut uid_alloc,
            inv_unit_scale,
        ) {
            curve_nodes.push(node);
            curves.extend(node_curves);
        }
    }

    let mut data = FbxExportData {
        clip_name: clip.name.clone(),
        duration_ktime,
        needs_coord_conversion,
        axes,
        fps,
        bones,
        stack_uid,
        layer_uid,
        document_uid,
        curve_nodes,
        curves,
        connections: Vec::new(),
    };

    generate_connections(&mut data);
    Ok(data)
}

pub(crate) fn write_property_i32<W: Write + Seek>(
    writer: &mut Writer<W>,
    name: &str,
    type1: &str,
    type2: &str,
    flags: &str,
    value: i32,
) -> FbxWriteResult<()> {
    let mut attrs = writer.new_node("P")?;
    attrs.append_string_direct(name)?;
    attrs.append_string_direct(type1)?;
    attrs.append_string_direct(type2)?;
    attrs.append_string_direct(flags)?;
    attrs.append_i32(value)?;
    drop(attrs);
    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_property_f64<W: Write + Seek>(
    writer: &mut Writer<W>,
    name: &str,
    type1: &str,
    type2: &str,
    flags: &str,
    value: f64,
) -> FbxWriteResult<()> {
    let mut attrs = writer.new_node("P")?;
    attrs.append_string_direct(name)?;
    attrs.append_string_direct(type1)?;
    attrs.append_string_direct(type2)?;
    attrs.append_string_direct(flags)?;
    attrs.append_f64(value)?;
    drop(attrs);
    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_property_f64x3<W: Write + Seek>(
    writer: &mut Writer<W>,
    name: &str,
    type1: &str,
    type2: &str,
    flags: &str,
    x: f64,
    y: f64,
    z: f64,
) -> FbxWriteResult<()> {
    let mut attrs = writer.new_node("P")?;
    attrs.append_string_direct(name)?;
    attrs.append_string_direct(type1)?;
    attrs.append_string_direct(type2)?;
    attrs.append_string_direct(flags)?;
    attrs.append_f64(x)?;
    attrs.append_f64(y)?;
    attrs.append_f64(z)?;
    drop(attrs);
    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_property_ktime<W: Write + Seek>(
    writer: &mut Writer<W>,
    name: &str,
    value: i64,
) -> FbxWriteResult<()> {
    let mut attrs = writer.new_node("P")?;
    attrs.append_string_direct(name)?;
    attrs.append_string_direct("KTime")?;
    attrs.append_string_direct("Time")?;
    attrs.append_string_direct("")?;
    attrs.append_i64(value)?;
    drop(attrs);
    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_header_extension<W: Write + Seek>(
    writer: &mut Writer<W>,
) -> FbxWriteResult<()> {
    drop(writer.new_node("FBXHeaderExtension")?);

    {
        let mut attrs = writer.new_node("FBXHeaderVersion")?;
        attrs.append_i32(1003)?;
        drop(attrs);
        writer.close_node()?;
    }

    {
        let mut attrs = writer.new_node("FBXVersion")?;
        attrs.append_i32(7400)?;
        drop(attrs);
        writer.close_node()?;
    }

    {
        let mut attrs = writer.new_node("Creator")?;
        attrs.append_string_direct("Thyllore Animation Engine")?;
        drop(attrs);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn fps_to_time_mode(fps: f32) -> i32 {
    let rounded = fps.round() as i32;
    match rounded {
        120 => 1,
        100 => 2,
        60 => 3,
        50 => 4,
        48 => 5,
        30 => 6,
        24 => 11,
        _ => 0,
    }
}

pub(crate) fn write_global_settings<W: Write + Seek>(
    writer: &mut Writer<W>,
    duration_ktime: i64,
    axes: &FbxAxesInfo,
    fps: f32,
    unit_scale_factor: f64,
) -> FbxWriteResult<()> {
    drop(writer.new_node("GlobalSettings")?);

    {
        let mut attrs = writer.new_node("Version")?;
        attrs.append_i32(1000)?;
        drop(attrs);
        writer.close_node()?;
    }

    let time_mode = fps_to_time_mode(fps);

    drop(writer.new_node("Properties70")?);
    write_property_i32(writer, "UpAxis", "int", "Integer", "", axes.up_axis)?;
    write_property_i32(
        writer,
        "UpAxisSign",
        "int",
        "Integer",
        "",
        axes.up_axis_sign,
    )?;
    write_property_i32(writer, "FrontAxis", "int", "Integer", "", axes.front_axis)?;
    write_property_i32(
        writer,
        "FrontAxisSign",
        "int",
        "Integer",
        "",
        axes.front_axis_sign,
    )?;
    write_property_i32(writer, "CoordAxis", "int", "Integer", "", axes.coord_axis)?;
    write_property_i32(
        writer,
        "CoordAxisSign",
        "int",
        "Integer",
        "",
        axes.coord_axis_sign,
    )?;
    write_property_f64(
        writer,
        "UnitScaleFactor",
        "double",
        "Number",
        "",
        unit_scale_factor,
    )?;
    write_property_i32(writer, "TimeMode", "enum", "", "", time_mode)?;
    write_property_f64(
        writer,
        "CustomFrameRate",
        "double",
        "Number",
        "",
        fps as f64,
    )?;
    write_property_ktime(writer, "TimeSpanStart", 0)?;
    write_property_ktime(writer, "TimeSpanStop", duration_ktime)?;
    writer.close_node()?;

    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_documents<W: Write + Seek>(
    writer: &mut Writer<W>,
    document_uid: i64,
) -> FbxWriteResult<()> {
    drop(writer.new_node("Documents")?);

    {
        let mut attrs = writer.new_node("Count")?;
        attrs.append_i32(1)?;
        drop(attrs);
        writer.close_node()?;
    }

    {
        let mut attrs = writer.new_node("Document")?;
        attrs.append_i64(document_uid)?;
        attrs.append_string_direct("")?;
        attrs.append_string_direct("Scene")?;
        drop(attrs);

        {
            let mut ra = writer.new_node("RootNode")?;
            ra.append_i64(0)?;
            drop(ra);
            writer.close_node()?;
        }

        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_references<W: Write + Seek>(writer: &mut Writer<W>) -> FbxWriteResult<()> {
    drop(writer.new_node("References")?);
    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_object_type<W: Write + Seek>(
    writer: &mut Writer<W>,
    type_name: &str,
    count: i32,
) -> FbxWriteResult<()> {
    let mut attrs = writer.new_node("ObjectType")?;
    attrs.append_string_direct(type_name)?;
    drop(attrs);

    {
        let mut ca = writer.new_node("Count")?;
        ca.append_i32(count)?;
        drop(ca);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_definitions<W: Write + Seek>(
    writer: &mut Writer<W>,
    data: &FbxExportData,
) -> FbxWriteResult<()> {
    let model_count = data.bones.len() as i32;
    let node_attribute_count = data
        .bones
        .iter()
        .filter(|b| b.node_attribute_uid.is_some())
        .count() as i32;
    let curve_node_count = data.curve_nodes.len() as i32;
    let curve_count = data.curves.len() as i32;

    let total = 1 + model_count + node_attribute_count + 1 + 1 + curve_node_count + curve_count;

    drop(writer.new_node("Definitions")?);

    {
        let mut attrs = writer.new_node("Version")?;
        attrs.append_i32(100)?;
        drop(attrs);
        writer.close_node()?;
    }

    {
        let mut attrs = writer.new_node("Count")?;
        attrs.append_i32(total)?;
        drop(attrs);
        writer.close_node()?;
    }

    write_object_type(writer, "GlobalSettings", 1)?;
    write_object_type(writer, "Model", model_count)?;

    if node_attribute_count > 0 {
        write_object_type(writer, "NodeAttribute", node_attribute_count)?;
    }

    write_object_type(writer, "AnimationStack", 1)?;
    write_object_type(writer, "AnimationLayer", 1)?;

    if curve_node_count > 0 {
        write_object_type(writer, "AnimationCurveNode", curve_node_count)?;
    }
    if curve_count > 0 {
        write_object_type(writer, "AnimationCurve", curve_count)?;
    }

    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_bone_model<W: Write + Seek>(
    writer: &mut Writer<W>,
    bone: &FbxBoneExport,
) -> FbxWriteResult<()> {
    let fbx_name = format!("{}\x00\x01Model", bone.name);
    let subclass = if bone.node_attribute_uid.is_some() {
        "LimbNode"
    } else {
        "Null"
    };
    let mut attrs = writer.new_node("Model")?;
    attrs.append_i64(bone.model_uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct(subclass)?;
    drop(attrs);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(232)?;
        drop(va);
        writer.close_node()?;
    }

    drop(writer.new_node("Properties70")?);
    write_property_f64x3(
        writer,
        "Lcl Translation",
        "Lcl Translation",
        "",
        "A",
        bone.translation[0],
        bone.translation[1],
        bone.translation[2],
    )?;
    write_property_f64x3(
        writer,
        "Lcl Rotation",
        "Lcl Rotation",
        "",
        "A",
        bone.rotation[0],
        bone.rotation[1],
        bone.rotation[2],
    )?;
    write_property_f64x3(
        writer,
        "Lcl Scaling",
        "Lcl Scaling",
        "",
        "A",
        bone.scaling[0],
        bone.scaling[1],
        bone.scaling[2],
    )?;
    write_property_i32(writer, "RotationOrder", "enum", "", "", 0)?;
    writer.close_node()?;

    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_node_attribute<W: Write + Seek>(
    writer: &mut Writer<W>,
    bone: &FbxBoneExport,
) -> FbxWriteResult<()> {
    let Some(attr_uid) = bone.node_attribute_uid else {
        return Ok(());
    };

    let fbx_name = format!("{}\x00\x01NodeAttribute", bone.name);
    let mut attrs = writer.new_node("NodeAttribute")?;
    attrs.append_i64(attr_uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("LimbNode")?;
    drop(attrs);

    {
        let mut ta = writer.new_node("TypeFlags")?;
        ta.append_string_direct("Skeleton")?;
        drop(ta);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_anim_stack<W: Write + Seek>(
    writer: &mut Writer<W>,
    data: &FbxExportData,
) -> FbxWriteResult<()> {
    let fbx_name = format!("{}\x00\x01AnimStack", data.clip_name);
    let mut attrs = writer.new_node("AnimationStack")?;
    attrs.append_i64(data.stack_uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("")?;
    drop(attrs);

    drop(writer.new_node("Properties70")?);
    write_property_ktime(writer, "LocalStart", 0)?;
    write_property_ktime(writer, "LocalStop", data.duration_ktime)?;
    write_property_ktime(writer, "ReferenceStart", 0)?;
    write_property_ktime(writer, "ReferenceStop", data.duration_ktime)?;
    writer.close_node()?;

    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_anim_layer<W: Write + Seek>(
    writer: &mut Writer<W>,
    layer_uid: i64,
) -> FbxWriteResult<()> {
    let mut attrs = writer.new_node("AnimationLayer")?;
    attrs.append_i64(layer_uid)?;
    attrs.append_string_direct("BaseLayer\x00\x01AnimLayer")?;
    attrs.append_string_direct("")?;
    drop(attrs);
    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_anim_curve_node<W: Write + Seek>(
    writer: &mut Writer<W>,
    cn: &FbxCurveNodeExport,
) -> FbxWriteResult<()> {
    let fbx_name = format!("{}\x00\x01AnimCurveNode", cn.channel.short_name());
    let mut attrs = writer.new_node("AnimationCurveNode")?;
    attrs.append_i64(cn.uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("")?;
    drop(attrs);

    drop(writer.new_node("Properties70")?);
    write_property_f64(writer, "d|X", "Number", "", "A", cn.default_values[0])?;
    write_property_f64(writer, "d|Y", "Number", "", "A", cn.default_values[1])?;
    write_property_f64(writer, "d|Z", "Number", "", "A", cn.default_values[2])?;
    writer.close_node()?;

    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_anim_curve<W: Write + Seek>(
    writer: &mut Writer<W>,
    curve: &FbxCurveExport,
) -> FbxWriteResult<()> {
    let mut attrs = writer.new_node("AnimationCurve")?;
    attrs.append_i64(curve.uid)?;
    attrs.append_string_direct("\x00\x01AnimCurve")?;
    attrs.append_string_direct("")?;
    drop(attrs);

    {
        let mut da = writer.new_node("Default")?;
        da.append_f64(curve.default_value)?;
        drop(da);
        writer.close_node()?;
    }

    {
        let mut ka = writer.new_node("KeyVer")?;
        ka.append_i32(4008)?;
        drop(ka);
        writer.close_node()?;
    }

    {
        let mut ta = writer.new_node("KeyTime")?;
        ta.append_arr_i64_from_iter(None, curve.key_times.iter().copied())?;
        drop(ta);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("KeyValueFloat")?;
        va.append_arr_f32_from_iter(None, curve.key_values.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut fa = writer.new_node("KeyAttrFlags")?;
        fa.append_arr_i32_from_iter(None, curve.key_attr_flags.iter().copied())?;
        drop(fa);
        writer.close_node()?;
    }

    {
        let mut da = writer.new_node("KeyAttrDataFloat")?;
        da.append_arr_f32_from_iter(None, curve.key_attr_data.iter().copied())?;
        drop(da);
        writer.close_node()?;
    }

    {
        let mut ra = writer.new_node("KeyAttrRefCount")?;
        ra.append_arr_i32_from_iter(None, curve.key_attr_ref_count.iter().copied())?;
        drop(ra);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_objects<W: Write + Seek>(
    writer: &mut Writer<W>,
    data: &FbxExportData,
) -> FbxWriteResult<()> {
    drop(writer.new_node("Objects")?);

    for bone in &data.bones {
        write_bone_model(writer, bone)?;
    }

    for bone in &data.bones {
        write_node_attribute(writer, bone)?;
    }

    write_anim_stack(writer, data)?;
    write_anim_layer(writer, data.layer_uid)?;

    for cn in &data.curve_nodes {
        write_anim_curve_node(writer, cn)?;
    }

    for curve in &data.curves {
        write_anim_curve(writer, curve)?;
    }

    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_connections<W: Write + Seek>(
    writer: &mut Writer<W>,
    data: &FbxExportData,
) -> FbxWriteResult<()> {
    drop(writer.new_node("Connections")?);

    for conn in &data.connections {
        match conn {
            FbxConnection::OO { child, parent } => {
                let mut attrs = writer.new_node("C")?;
                attrs.append_string_direct("OO")?;
                attrs.append_i64(*child)?;
                attrs.append_i64(*parent)?;
                drop(attrs);
                writer.close_node()?;
            }
            FbxConnection::OP {
                child,
                parent,
                property,
            } => {
                let mut attrs = writer.new_node("C")?;
                attrs.append_string_direct("OP")?;
                attrs.append_i64(*child)?;
                attrs.append_i64(*parent)?;
                attrs.append_string_direct(property)?;
                drop(attrs);
                writer.close_node()?;
            }
        }
    }

    writer.close_node()?;
    Ok(())
}

fn write_fbx_binary<W: Write + Seek>(
    mut writer: Writer<W>,
    data: &FbxExportData,
) -> FbxWriteResult<()> {
    write_header_extension(&mut writer)?;
    write_global_settings(&mut writer, data.duration_ktime, &data.axes, data.fps, 1.0)?;
    write_documents(&mut writer, data.document_uid)?;
    write_references(&mut writer)?;
    write_definitions(&mut writer, data)?;
    write_objects(&mut writer, data)?;
    write_connections(&mut writer, data)?;
    writer.finalize_and_flush(&FbxFooter::default())?;
    Ok(())
}

pub fn export_animation_fbx(
    clip: &EditableAnimationClip,
    skeleton: &Skeleton,
    path: &Path,
    needs_coord_conversion: bool,
    axes: FbxAxesInfo,
    fps: f32,
) -> anyhow::Result<()> {
    let export_data = build_export_data(clip, skeleton, needs_coord_conversion, axes, fps)?;

    let file = std::fs::File::create(path)?;
    let writer = Writer::new(file, FbxVersion::V7_4)
        .map_err(|e| anyhow::anyhow!("FBX writer init failed: {}", e))?;

    write_fbx_binary(writer, &export_data)
        .map_err(|e| anyhow::anyhow!("FBX write failed: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seconds_to_ktime() {
        assert_eq!(seconds_to_ktime(1.0), 46_186_158_000);
        assert_eq!(seconds_to_ktime(0.0), 0);
    }

    #[test]
    fn test_interpolation_flags() {
        assert_eq!(
            convert_interpolation_to_flags(InterpolationType::Linear),
            0x04
        );
        assert_eq!(
            convert_interpolation_to_flags(InterpolationType::Stepped),
            0x02
        );
        assert_eq!(
            convert_interpolation_to_flags(InterpolationType::Bezier),
            0x0408
        );
    }

    #[test]
    fn test_tangent_conversion_linear() {
        let handle = BezierHandle::linear();
        let (slope, _weight) = convert_tangent_to_fbx_slope_weight(&handle, 1.0);
        assert!((slope - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_key_attr_rle_identical_keys() {
        let kfs = vec![
            EditableKeyframe::new(0, 0.0, 0.0),
            EditableKeyframe::new(1, 1.0, 1.0),
            EditableKeyframe::new(2, 2.0, 2.0),
        ];

        let mut flags = Vec::new();
        let mut data = Vec::new();
        let mut ref_count = Vec::new();
        build_key_attr_arrays(&kfs, &mut flags, &mut data, &mut ref_count);

        assert_eq!(ref_count.len(), 1);
        assert_eq!(ref_count[0], 3);
        assert_eq!(flags[0], 0x04);
    }

    #[test]
    fn test_key_attr_rle_mixed_interpolation() {
        let mut kf0 = EditableKeyframe::new(0, 0.0, 0.0);
        kf0.interpolation = InterpolationType::Linear;

        let mut kf1 = EditableKeyframe::new(1, 1.0, 1.0);
        kf1.interpolation = InterpolationType::Bezier;

        let mut kf2 = EditableKeyframe::new(2, 2.0, 2.0);
        kf2.interpolation = InterpolationType::Linear;

        let kfs = vec![kf0, kf1, kf2];
        let mut flags = Vec::new();
        let mut data = Vec::new();
        let mut ref_count = Vec::new();
        build_key_attr_arrays(&kfs, &mut flags, &mut data, &mut ref_count);

        assert!(ref_count.len() >= 2);
    }

    #[test]
    fn test_uid_allocator_monotonic() {
        let mut alloc = UidAllocator::new();
        let uid1 = alloc.allocate();
        let uid2 = alloc.allocate();
        let uid3 = alloc.allocate();

        assert!(uid2 > uid1);
        assert!(uid3 > uid2);
        assert_eq!(uid1, 1_000_000);
    }

    #[test]
    fn test_validate_bone_names_all_match() {
        let mut skeleton = Skeleton::new("test");
        skeleton.add_bone("Hips", None);
        skeleton.add_bone("Spine", Some(0));

        let mut clip = EditableAnimationClip::new(1, "test_clip".to_string());
        clip.add_track(0, "Hips".to_string());
        clip.add_track(1, "Spine".to_string());

        let missing = validate_bone_names(&clip, &skeleton);
        assert!(missing.is_empty());
    }

    #[test]
    fn test_validate_bone_names_missing() {
        let mut skeleton = Skeleton::new("test");
        skeleton.add_bone("Hips", None);

        let mut clip = EditableAnimationClip::new(1, "test_clip".to_string());
        clip.add_track(0, "Hips".to_string());
        clip.add_track(1, "NonExistent".to_string());

        let missing = validate_bone_names(&clip, &skeleton);
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0], "NonExistent");
    }

    #[test]
    fn test_coord_conversion_roundtrip() {
        use cgmath::SquareMatrix;
        use thyllore_math_core::coordinate_system::{fbx_to_world, world_to_fbx};

        let original = Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0));
        let roundtrip = world_to_fbx() * fbx_to_world() * original;

        for col in 0..4 {
            for row in 0..4 {
                assert!(
                    (roundtrip[col][row] - original[col][row]).abs() < 1e-5,
                    "Mismatch at [{col}][{row}]: {} vs {}",
                    roundtrip[col][row],
                    original[col][row]
                );
            }
        }
    }

    fn build_test_skeleton_with_rootnode() -> Skeleton {
        let mut skeleton = Skeleton::new("test");
        skeleton.add_bone("RootNode", None);
        skeleton.add_bone("Armature", Some(0));
        skeleton.add_bone("Hips", Some(1));
        skeleton.add_bone("Spine", Some(2));
        skeleton
    }

    #[test]
    fn test_build_bone_export_list_with_coord_conversion() {
        let skeleton = build_test_skeleton_with_rootnode();
        let mut uid_alloc = UidAllocator::new();
        let empty_set = std::collections::HashSet::new();

        let bones_with = build_bone_export_list(&skeleton, &mut uid_alloc, &empty_set, 1.0, true);
        let mut uid_alloc2 = UidAllocator::new();
        let bones_without =
            build_bone_export_list(&skeleton, &mut uid_alloc2, &empty_set, 1.0, false);

        let hips_with = bones_with.iter().find(|b| b.name == "Hips").unwrap();
        let hips_without = bones_without.iter().find(|b| b.name == "Hips").unwrap();
        assert_eq!(hips_with.translation, hips_without.translation);
        assert_eq!(hips_with.rotation, hips_without.rotation);

        let rootnode_with = bones_with.iter().find(|b| b.name == "RootNode").unwrap();
        let rootnode_without = bones_without.iter().find(|b| b.name == "RootNode").unwrap();
        assert_eq!(rootnode_without.translation, [0.0, 0.0, 0.0]);
        assert_eq!(rootnode_with.translation, rootnode_without.translation);
    }

    #[test]
    fn test_weighted_tangent_flags_in_key_attr() {
        let mut kf0 = EditableKeyframe::new(0, 0.0, 0.0);
        kf0.interpolation = InterpolationType::Bezier;
        kf0.weight_mode = TangentWeightMode::Weighted;

        let mut kf1 = EditableKeyframe::new(1, 1.0, 1.0);
        kf1.interpolation = InterpolationType::Bezier;
        kf1.weight_mode = TangentWeightMode::Weighted;

        let mut kf2 = EditableKeyframe::new(2, 2.0, 2.0);
        kf2.interpolation = InterpolationType::Bezier;
        kf2.weight_mode = TangentWeightMode::NonWeighted;

        let kfs = vec![kf0, kf1, kf2];
        let mut flags = Vec::new();
        let mut data = Vec::new();
        let mut ref_count = Vec::new();
        build_key_attr_arrays(&kfs, &mut flags, &mut data, &mut ref_count);

        // kf0: weighted out + next(kf1) weighted in => WEIGHTED_RIGHT | WEIGHTED_NEXT_LEFT
        // kf1: weighted out + next(kf2) non-weighted in => WEIGHTED_RIGHT only
        // kf2: non-weighted out + no next => no weight flags
        // All three have different flag combinations, so RLE won't merge them
        let base_bezier = 0x0000_0408;
        let weighted_right = 0x0100_0000;
        let weighted_next_left = 0x0400_0000;

        assert!(flags.len() >= 2, "Expected at least 2 RLE groups");
        assert_eq!(
            flags[0],
            base_bezier | weighted_right | weighted_next_left,
            "kf0 should have both weight flags"
        );
    }

    #[test]
    fn test_non_weighted_tangent_no_extra_flags() {
        let mut kf0 = EditableKeyframe::new(0, 0.0, 0.0);
        kf0.interpolation = InterpolationType::Bezier;
        kf0.weight_mode = TangentWeightMode::NonWeighted;

        let mut kf1 = EditableKeyframe::new(1, 1.0, 1.0);
        kf1.interpolation = InterpolationType::Bezier;
        kf1.weight_mode = TangentWeightMode::NonWeighted;

        let kfs = vec![kf0, kf1];
        let mut flags = Vec::new();
        let mut data = Vec::new();
        let mut ref_count = Vec::new();
        build_key_attr_arrays(&kfs, &mut flags, &mut data, &mut ref_count);

        let base_bezier = 0x0000_0408;
        let weighted_right = 0x0100_0000;
        let weighted_next_left = 0x0400_0000;
        for &flag in &flags {
            assert_eq!(
                flag & weighted_right,
                0,
                "NonWeighted should not have WEIGHTED_RIGHT flag"
            );
            assert_eq!(
                flag & weighted_next_left,
                0,
                "NonWeighted should not have WEIGHTED_NEXT_LEFT flag"
            );
        }
    }

    #[test]
    fn test_build_bone_export_list_without_coord_conversion() {
        let skeleton = build_test_skeleton_with_rootnode();
        let mut uid_alloc = UidAllocator::new();
        let empty_set = std::collections::HashSet::new();

        let bones = build_bone_export_list(&skeleton, &mut uid_alloc, &empty_set, 1.0, false);

        assert_eq!(bones.len(), 4);
        for bone in &bones {
            assert_eq!(bone.translation, [0.0, 0.0, 0.0]);
            assert_eq!(bone.rotation, [0.0, 0.0, 0.0]);
            assert_eq!(bone.scaling, [1.0, 1.0, 1.0]);
        }
    }
}
