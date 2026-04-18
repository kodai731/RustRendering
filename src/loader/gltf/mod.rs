mod loader;
pub mod spring_bone_extension;

#[cfg(feature = "auto-rig")]
pub use loader::load_gltf_from_slice;
pub use loader::{load_gltf_file, GltfLoadResult, GltfMeshData, ImageData, NodeInfo};
