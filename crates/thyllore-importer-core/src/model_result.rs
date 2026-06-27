use cgmath::{Deg, Matrix4, Quaternion, Rotation3};

use thyllore_anim_core::spring_bone::SpringBoneSetup;
use thyllore_anim_core::{
    AnimationClip, AnimationSystem, MorphAnimationSystem, Skeleton, SkeletonId, SkinData,
};
use thyllore_model_core::mesh::{Vertex, VertexData};

use crate::fbx::{self, LoadedConstraint};
use crate::gltf;
use crate::usd;

#[derive(Clone, Debug)]
pub struct TextureData {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug)]
pub struct LoadedMesh {
    pub vertex_data: VertexData,
    pub skin_data: Option<SkinData>,
    pub skeleton_id: Option<SkeletonId>,
    pub node_index: Option<usize>,
    pub local_vertices: Vec<Vertex>,
    pub texture: Option<TextureSource>,
    pub base_color_factor: [f32; 4],
}

impl Default for LoadedMesh {
    fn default() -> Self {
        Self {
            vertex_data: VertexData::default(),
            skin_data: None,
            skeleton_id: None,
            node_index: None,
            local_vertices: Vec::new(),
            texture: None,
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

#[derive(Clone, Debug)]
pub enum TextureSource {
    Embedded(TextureData),
    File(String),
}

/// Root-level transform that brings imported model data into the engine's
/// Y-up, meters world. Derived from source-file metadata (e.g. USD `upAxis` /
/// `metersPerUnit`), never hardcoded per asset.
#[derive(Clone, Copy, Debug)]
pub struct RootTransform {
    pub rotation: Quaternion<f32>,
    pub scale: f32,
}

#[derive(Clone, Debug)]
pub struct LoadedNode {
    pub index: usize,
    pub name: String,
    pub parent_index: Option<usize>,
    pub local_transform: Matrix4<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct ModelLoadResult {
    pub meshes: Vec<LoadedMesh>,
    pub nodes: Vec<LoadedNode>,
    pub skeletons: Vec<Skeleton>,
    pub animation_system: AnimationSystem,
    pub clips: Vec<AnimationClip>,
    pub morph_animation: MorphAnimationSystem,
    pub has_skinned_meshes: bool,
    pub node_animation_scale: f32,
    pub constraints: Vec<LoadedConstraint>,
    pub spring_bone_setup: Option<SpringBoneSetup>,
    pub curves: Vec<usd::UsdCurveData>,
    pub points: Vec<usd::UsdPointData>,
    pub root_transform: Option<RootTransform>,
}

/// Build the root transform from USD stage metadata. USD permits only Y-up or
/// Z-up, so the orientation is a deterministic rotation; `metersPerUnit`
/// normalizes the stage to the engine's meter scale.
fn usd_root_transform(up_axis: usd::UsdUpAxis, meters_per_unit: f32) -> RootTransform {
    let rotation = match up_axis {
        usd::UsdUpAxis::Z => Quaternion::from_angle_x(Deg(-90.0)),
        usd::UsdUpAxis::Y => Quaternion::new(1.0, 0.0, 0.0, 0.0),
    };
    RootTransform {
        rotation,
        scale: meters_per_unit,
    }
}

impl ModelLoadResult {
    pub fn from_gltf(result: gltf::GltfLoadResult) -> Self {
        let meshes = result
            .meshes
            .into_iter()
            .map(|m| LoadedMesh {
                vertex_data: m.vertex_data,
                skin_data: m.skin_data,
                skeleton_id: m.skeleton_id,
                node_index: m.node_index,
                local_vertices: m.local_vertices,
                texture: m.image_data.first().map(|img| {
                    TextureSource::Embedded(TextureData {
                        data: img.data.clone(),
                        width: img.width,
                        height: img.height,
                    })
                }),
                base_color_factor: m.base_color_factor,
            })
            .collect();

        let nodes = result
            .nodes
            .into_iter()
            .map(|n| LoadedNode {
                index: n.index,
                name: n.name,
                parent_index: n.parent_index,
                local_transform: n.local_transform,
            })
            .collect();

        let node_animation_scale = if result.has_armature { 0.01 } else { 1.0 };

        let skeletons = result.animation_system.skeletons.clone();

        Self {
            meshes,
            nodes,
            skeletons,
            animation_system: result.animation_system,
            clips: result.clips,
            morph_animation: result.morph_animation,
            has_skinned_meshes: result.has_skinned_meshes,
            node_animation_scale,
            constraints: Vec::new(),
            spring_bone_setup: result.spring_bone_setup,
            curves: Vec::new(),
            points: Vec::new(),
            root_transform: None,
        }
    }

    pub fn from_usd(result: usd::UsdLoadResult) -> Self {
        let meshes = result
            .meshes
            .into_iter()
            .map(|m| LoadedMesh {
                vertex_data: m.vertex_data,
                skin_data: m.skin_data,
                skeleton_id: m.skeleton_id,
                node_index: m.node_index,
                local_vertices: m.local_vertices,
                texture: None,
                base_color_factor: [1.0, 1.0, 1.0, 1.0],
            })
            .collect();

        let nodes = result
            .nodes
            .into_iter()
            .map(|n| LoadedNode {
                index: n.index,
                name: n.name,
                parent_index: n.parent_index,
                local_transform: n.local_transform,
            })
            .collect();

        let skeletons = result.animation_system.skeletons.clone();

        let root_transform = Some(usd_root_transform(result.up_axis, result.meters_per_unit));

        Self {
            meshes,
            nodes,
            skeletons,
            animation_system: result.animation_system,
            clips: result.clips,
            morph_animation: result.morph_animation,
            has_skinned_meshes: result.has_skinned_meshes,
            node_animation_scale: 1.0,
            constraints: Vec::new(),
            spring_bone_setup: None,
            curves: result.curves,
            points: result.points,
            root_transform,
        }
    }

    pub fn from_fbx(result: fbx::FbxLoadResult) -> Self {
        let meshes = result
            .meshes
            .into_iter()
            .map(|m| LoadedMesh {
                vertex_data: m.vertex_data,
                skin_data: m.skin_data,
                skeleton_id: m.skeleton_id,
                node_index: m.node_index,
                local_vertices: m.local_vertices,
                texture: m.texture_path.map(TextureSource::File),
                base_color_factor: [1.0, 1.0, 1.0, 1.0],
            })
            .collect();

        let nodes = result
            .nodes
            .into_iter()
            .map(|n| LoadedNode {
                index: n.index,
                name: n.name,
                parent_index: n.parent_index,
                local_transform: n.local_transform,
            })
            .collect();

        let skeletons = result.animation_system.skeletons.clone();

        Self {
            meshes,
            nodes,
            skeletons,
            animation_system: result.animation_system,
            clips: result.clips,
            morph_animation: MorphAnimationSystem::default(),
            has_skinned_meshes: result.has_skinned_meshes,
            node_animation_scale: 1.0,
            constraints: result.constraints,
            spring_bone_setup: None,
            curves: Vec::new(),
            points: Vec::new(),
            root_transform: None,
        }
    }
}
