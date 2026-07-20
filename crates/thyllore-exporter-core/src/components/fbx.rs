use crate::fbx_animation::FbxExportData;

pub(crate) struct FbxGeometryExport {
    pub uid: i64,
    pub mesh_model_uid: i64,
    pub positions: Vec<f64>,
    pub polygon_vertex_index: Vec<i32>,
    pub normals: Vec<f64>,
    pub uv_values: Vec<f64>,
}

pub(crate) struct FbxMeshModelExport {
    pub uid: i64,
    pub name: String,
    pub parent_bone_uid: Option<i64>,
    pub translation: [f64; 3],
    pub rotation: [f64; 3],
    pub scaling: [f64; 3],
}

pub(crate) struct FbxMaterialExport {
    pub uid: i64,
    pub name: String,
    pub mesh_model_uid: i64,
    pub diffuse_color: [f64; 3],
}

pub(crate) struct FbxTextureExport {
    pub texture_uid: i64,
    pub video_uid: i64,
    pub material_uid: i64,
    pub filename: String,
    pub relative_filename: String,
}

pub(crate) struct FbxSkinExport {
    pub skin_uid: i64,
    pub geometry_uid: i64,
    pub clusters: Vec<FbxClusterExport>,
}

pub(crate) struct FbxClusterExport {
    pub uid: i64,
    pub bone_model_uid: i64,
    pub indices: Vec<i32>,
    pub weights: Vec<f64>,
    pub transform: [f64; 16],
    pub transform_link: [f64; 16],
}

pub(crate) struct FullFbxExportData {
    pub anim_data: FbxExportData,
    pub geometries: Vec<FbxGeometryExport>,
    pub mesh_models: Vec<FbxMeshModelExport>,
    pub materials: Vec<FbxMaterialExport>,
    pub textures: Vec<FbxTextureExport>,
    pub skins: Vec<FbxSkinExport>,
    pub unit_scale: f32,
}
