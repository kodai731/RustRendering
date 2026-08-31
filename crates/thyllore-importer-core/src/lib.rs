#[macro_use]
extern crate thyllore_log_core;

pub mod fbx;
pub mod gltf;
mod model_result;
pub mod primitive;
mod texture;

pub use fbx::{
    load_fbx_to_graphics_resources, FbxLoadResult, FbxMeshData, FbxModel, FbxNodeInfo,
    LoadedConstraint,
};
pub use gltf::{load_gltf_file, GltfLoadResult, GltfMeshData, ImageData, NodeInfo};
pub use model_result::{LoadedMesh, LoadedNode, ModelLoadResult, TextureData, TextureSource};
pub use texture::load_png_image;

#[cfg(feature = "auto-rig")]
pub use gltf::load_gltf_from_slice;
