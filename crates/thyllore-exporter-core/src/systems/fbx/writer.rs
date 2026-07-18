use std::io::{Seek, Write};

use fbxcel::writer::v7400::binary::{FbxFooter, Writer};

use crate::fbx_animation::{
    write_anim_curve, write_anim_curve_node, write_anim_layer, write_anim_stack, write_bone_model,
    write_connections, write_documents, write_global_settings, write_header_extension,
    write_node_attribute, write_object_type, write_property_f64, write_property_f64x3,
    write_references, FbxWriteResult,
};

use crate::components::fbx::*;

pub(crate) fn write_full_definitions<W: Write + Seek>(
    writer: &mut Writer<W>,
    data: &FullFbxExportData,
) -> FbxWriteResult<()> {
    let model_count = data.anim_data.bones.len() as i32 + data.mesh_models.len() as i32;
    let node_attribute_count = data
        .anim_data
        .bones
        .iter()
        .filter(|b| b.node_attribute_uid.is_some())
        .count() as i32;
    let geometry_count = data.geometries.len() as i32;
    let material_count = data.materials.len() as i32;
    let texture_count = data.textures.len() as i32;
    let video_count = data.textures.len() as i32;
    let deformer_count = data.skins.len() as i32;
    let sub_deformer_count: i32 = data.skins.iter().map(|s| s.clusters.len() as i32).sum();
    let curve_node_count = data.anim_data.curve_nodes.len() as i32;
    let curve_count = data.anim_data.curves.len() as i32;

    let total = 1
        + model_count
        + node_attribute_count
        + geometry_count
        + material_count
        + texture_count
        + video_count
        + deformer_count
        + sub_deformer_count
        + 1
        + 1
        + curve_node_count
        + curve_count;

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

    if geometry_count > 0 {
        write_object_type(writer, "Geometry", geometry_count)?;
    }
    if material_count > 0 {
        write_object_type(writer, "Material", material_count)?;
    }
    if texture_count > 0 {
        write_object_type(writer, "Texture", texture_count)?;
    }
    if video_count > 0 {
        write_object_type(writer, "Video", video_count)?;
    }
    if deformer_count > 0 {
        write_object_type(writer, "Deformer", deformer_count + sub_deformer_count)?;
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

pub(crate) fn write_mesh_model<W: Write + Seek>(
    writer: &mut Writer<W>,
    mesh: &FbxMeshModelExport,
) -> FbxWriteResult<()> {
    let fbx_name = format!("{}\x00\x01Model", mesh.name);
    let mut attrs = writer.new_node("Model")?;
    attrs.append_i64(mesh.uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("Mesh")?;
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
        mesh.translation[0],
        mesh.translation[1],
        mesh.translation[2],
    )?;
    write_property_f64x3(
        writer,
        "Lcl Rotation",
        "Lcl Rotation",
        "",
        "A",
        mesh.rotation[0],
        mesh.rotation[1],
        mesh.rotation[2],
    )?;
    write_property_f64x3(
        writer,
        "Lcl Scaling",
        "Lcl Scaling",
        "",
        "A",
        mesh.scaling[0],
        mesh.scaling[1],
        mesh.scaling[2],
    )?;
    writer.close_node()?;

    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_geometry<W: Write + Seek>(
    writer: &mut Writer<W>,
    geo: &FbxGeometryExport,
    has_material: bool,
) -> FbxWriteResult<()> {
    let fbx_name = format!("\x00\x01Geometry");
    let mut attrs = writer.new_node("Geometry")?;
    attrs.append_i64(geo.uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("Mesh")?;
    drop(attrs);

    {
        let mut va = writer.new_node("Vertices")?;
        va.append_arr_f64_from_iter(None, geo.positions.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("PolygonVertexIndex")?;
        va.append_arr_i32_from_iter(None, geo.polygon_vertex_index.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    if !geo.normals.is_empty() {
        write_layer_element_normal(writer, &geo.normals)?;
    }

    if !geo.uv_values.is_empty() {
        write_layer_element_uv(writer, &geo.uv_values)?;
    }

    if has_material {
        write_layer_element_material(writer)?;
    }

    if !geo.normals.is_empty() || !geo.uv_values.is_empty() || has_material {
        write_layer(
            writer,
            !geo.normals.is_empty(),
            !geo.uv_values.is_empty(),
            has_material,
        )?;
    }

    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_layer_element_normal<W: Write + Seek>(
    writer: &mut Writer<W>,
    normals: &[f64],
) -> FbxWriteResult<()> {
    drop(writer.new_node("LayerElementNormal")?);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(101)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Name")?;
        va.append_string_direct("")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("MappingInformationType")?;
        va.append_string_direct("ByVertice")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("ReferenceInformationType")?;
        va.append_string_direct("Direct")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Normals")?;
        va.append_arr_f64_from_iter(None, normals.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_layer_element_uv<W: Write + Seek>(
    writer: &mut Writer<W>,
    uv_values: &[f64],
) -> FbxWriteResult<()> {
    drop(writer.new_node("LayerElementUV")?);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(101)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Name")?;
        va.append_string_direct("UVMap")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("MappingInformationType")?;
        va.append_string_direct("ByVertice")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("ReferenceInformationType")?;
        va.append_string_direct("Direct")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("UV")?;
        va.append_arr_f64_from_iter(None, uv_values.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_layer_element_material<W: Write + Seek>(writer: &mut Writer<W>) -> FbxWriteResult<()> {
    drop(writer.new_node("LayerElementMaterial")?);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(101)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Name")?;
        va.append_string_direct("")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("MappingInformationType")?;
        va.append_string_direct("AllSame")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("ReferenceInformationType")?;
        va.append_string_direct("IndexToDirect")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Materials")?;
        va.append_arr_i32_from_iter(None, [0].iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_layer<W: Write + Seek>(
    writer: &mut Writer<W>,
    has_normal: bool,
    has_uv: bool,
    has_material: bool,
) -> FbxWriteResult<()> {
    drop(writer.new_node("Layer")?);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(100)?;
        drop(va);
        writer.close_node()?;
    }

    if has_normal {
        drop(writer.new_node("LayerElement")?);

        {
            let mut va = writer.new_node("Type")?;
            va.append_string_direct("LayerElementNormal")?;
            drop(va);
            writer.close_node()?;
        }

        {
            let mut va = writer.new_node("TypedIndex")?;
            va.append_i32(0)?;
            drop(va);
            writer.close_node()?;
        }

        writer.close_node()?;
    }

    if has_uv {
        drop(writer.new_node("LayerElement")?);

        {
            let mut va = writer.new_node("Type")?;
            va.append_string_direct("LayerElementUV")?;
            drop(va);
            writer.close_node()?;
        }

        {
            let mut va = writer.new_node("TypedIndex")?;
            va.append_i32(0)?;
            drop(va);
            writer.close_node()?;
        }

        writer.close_node()?;
    }

    if has_material {
        drop(writer.new_node("LayerElement")?);

        {
            let mut va = writer.new_node("Type")?;
            va.append_string_direct("LayerElementMaterial")?;
            drop(va);
            writer.close_node()?;
        }

        {
            let mut va = writer.new_node("TypedIndex")?;
            va.append_i32(0)?;
            drop(va);
            writer.close_node()?;
        }

        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_material<W: Write + Seek>(
    writer: &mut Writer<W>,
    mat: &FbxMaterialExport,
) -> FbxWriteResult<()> {
    let fbx_name = format!("{}\x00\x01Material", mat.name);
    let mut attrs = writer.new_node("Material")?;
    attrs.append_i64(mat.uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("")?;
    drop(attrs);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(102)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("ShadingModel")?;
        va.append_string_direct("phong")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("MultiLayer")?;
        va.append_i32(0)?;
        drop(va);
        writer.close_node()?;
    }

    drop(writer.new_node("Properties70")?);

    write_property_f64x3(
        writer,
        "DiffuseColor",
        "Color",
        "",
        "A",
        mat.diffuse_color[0],
        mat.diffuse_color[1],
        mat.diffuse_color[2],
    )?;

    write_property_f64(writer, "DiffuseFactor", "Number", "", "A", 1.0)?;
    write_property_f64(writer, "Opacity", "Number", "", "A", 1.0)?;

    write_property_f64x3(
        writer,
        "AmbientColor",
        "Color",
        "",
        "A",
        mat.diffuse_color[0],
        mat.diffuse_color[1],
        mat.diffuse_color[2],
    )?;

    write_property_f64x3(writer, "SpecularColor", "Color", "", "A", 0.9, 0.9, 0.9)?;

    write_property_f64(writer, "Shininess", "Number", "", "A", 20.0)?;
    write_property_f64(writer, "ShininessExponent", "Number", "", "A", 20.0)?;

    writer.close_node()?;

    writer.close_node()?;
    Ok(())
}

fn write_texture<W: Write + Seek>(
    writer: &mut Writer<W>,
    tex: &FbxTextureExport,
) -> FbxWriteResult<()> {
    let fbx_name = format!("\x00\x01Texture");
    let mut attrs = writer.new_node("Texture")?;
    attrs.append_i64(tex.texture_uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("")?;
    drop(attrs);

    {
        let mut va = writer.new_node("Type")?;
        va.append_string_direct("TextureDiffuse")?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("FileName")?;
        va.append_string_direct(&tex.filename)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("RelativeFilename")?;
        va.append_string_direct(&tex.relative_filename)?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_video<W: Write + Seek>(
    writer: &mut Writer<W>,
    tex: &FbxTextureExport,
) -> FbxWriteResult<()> {
    let fbx_name = format!("\x00\x01Video");
    let mut attrs = writer.new_node("Video")?;
    attrs.append_i64(tex.video_uid)?;
    attrs.append_string_direct(&fbx_name)?;
    attrs.append_string_direct("Clip")?;
    drop(attrs);

    {
        let mut va = writer.new_node("FileName")?;
        va.append_string_direct(&tex.filename)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("RelativeFilename")?;
        va.append_string_direct(&tex.relative_filename)?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_skin_deformer<W: Write + Seek>(
    writer: &mut Writer<W>,
    skin: &FbxSkinExport,
) -> FbxWriteResult<()> {
    let mut attrs = writer.new_node("Deformer")?;
    attrs.append_i64(skin.skin_uid)?;
    attrs.append_string_direct("\x00\x01Deformer")?;
    attrs.append_string_direct("Skin")?;
    drop(attrs);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(101)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Link_DeformAcuracy")?;
        va.append_f64(50.0)?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_cluster<W: Write + Seek>(
    writer: &mut Writer<W>,
    cluster: &FbxClusterExport,
) -> FbxWriteResult<()> {
    let mut attrs = writer.new_node("Deformer")?;
    attrs.append_i64(cluster.uid)?;
    attrs.append_string_direct("\x00\x01SubDeformer")?;
    attrs.append_string_direct("Cluster")?;
    drop(attrs);

    {
        let mut va = writer.new_node("Version")?;
        va.append_i32(100)?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Indexes")?;
        va.append_arr_i32_from_iter(None, cluster.indices.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Weights")?;
        va.append_arr_f64_from_iter(None, cluster.weights.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("Transform")?;
        va.append_arr_f64_from_iter(None, cluster.transform.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    {
        let mut va = writer.new_node("TransformLink")?;
        va.append_arr_f64_from_iter(None, cluster.transform_link.iter().copied())?;
        drop(va);
        writer.close_node()?;
    }

    writer.close_node()?;
    Ok(())
}

fn write_full_objects<W: Write + Seek>(
    writer: &mut Writer<W>,
    data: &FullFbxExportData,
) -> FbxWriteResult<()> {
    drop(writer.new_node("Objects")?);

    for bone in &data.anim_data.bones {
        write_bone_model(writer, bone)?;
    }

    for bone in &data.anim_data.bones {
        write_node_attribute(writer, bone)?;
    }

    let has_material = !data.materials.is_empty();
    for geo in &data.geometries {
        write_geometry(writer, geo, has_material)?;
    }

    for mesh_model in &data.mesh_models {
        write_mesh_model(writer, mesh_model)?;
    }

    for material in &data.materials {
        write_material(writer, material)?;
    }

    for texture in &data.textures {
        write_texture(writer, texture)?;
        write_video(writer, texture)?;
    }

    for skin in &data.skins {
        write_skin_deformer(writer, skin)?;
        for cluster in &skin.clusters {
            write_cluster(writer, cluster)?;
        }
    }

    write_anim_stack(writer, &data.anim_data)?;
    write_anim_layer(writer, data.anim_data.layer_uid)?;

    for cn in &data.anim_data.curve_nodes {
        write_anim_curve_node(writer, cn)?;
    }

    for curve in &data.anim_data.curves {
        write_anim_curve(writer, curve)?;
    }

    writer.close_node()?;
    Ok(())
}

pub(crate) fn write_full_fbx_binary<W: Write + Seek>(
    mut writer: Writer<W>,
    data: &FullFbxExportData,
) -> FbxWriteResult<()> {
    write_header_extension(&mut writer)?;
    let unit_scale_factor = (data.unit_scale * 100.0) as f64;
    write_global_settings(
        &mut writer,
        data.anim_data.duration_ktime,
        &data.anim_data.axes,
        data.anim_data.fps,
        unit_scale_factor,
    )?;
    write_documents(&mut writer, data.anim_data.document_uid)?;
    write_references(&mut writer)?;
    write_full_definitions(&mut writer, data)?;
    write_full_objects(&mut writer, data)?;
    write_connections(&mut writer, &data.anim_data)?;
    writer.finalize_and_flush(&FbxFooter::default())?;
    Ok(())
}
