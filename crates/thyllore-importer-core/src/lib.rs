#[macro_use]
mod logger_compat;

pub mod fbx;
pub mod gltf;
mod model_result;
mod texture;
pub mod usd;

pub use fbx::{
    load_fbx_to_graphics_resources, FbxLoadResult, FbxMeshData, FbxModel, FbxNodeInfo,
    LoadedConstraint,
};
pub use gltf::{load_gltf_file, GltfLoadResult, GltfMeshData, ImageData, NodeInfo};
pub use model_result::{LoadedMesh, LoadedNode, ModelLoadResult, TextureData, TextureSource};
pub use texture::load_png_image;
pub use usd::{
    load_usd_file, save_usd_file, UsdCurveData, UsdExportBlendShape, UsdExportMesh, UsdExportScene,
    UsdLoadResult, UsdMeshData, UsdNodeInfo,
};

#[cfg(feature = "auto-rig")]
pub use gltf::load_gltf_from_slice;
