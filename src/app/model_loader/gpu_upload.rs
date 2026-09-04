use std::ffi::c_void;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::Result;
use cgmath::Vector4;
use vulkanalia::prelude::v1_0::*;

use super::scene_model::restore_batch_playback;
use crate::asset::AssetStorage;
use crate::ecs::resource::{NodeAssets, TimelineState};
use crate::ecs::world::World;
use crate::loader::load_png_image;
use crate::loader::{ModelLoadResult, TextureSource};
use crate::render::MaterialUBO;
use crate::vulkanr::buffer::{RRIndexBuffer, RRVertexBuffer};
use crate::vulkanr::command::RRCommandPool;
use crate::vulkanr::data as vulkan_data;
use crate::vulkanr::device::RRDevice;
use crate::vulkanr::resource::graphics_resource::{GraphicsResources, MaterialId, MeshBuffer};
use crate::vulkanr::swapchain::RRSwapchain;
use crate::vulkanr::vulkan::Instance;
use thyllore_vulkan_core::resource::image::{
    create_image_view, create_texture_image_pixel, create_texture_sampler,
};

pub(super) unsafe fn ensure_graphics_capacity(
    load_result: &ModelLoadResult,
    instance: &Instance,
    device: &RRDevice,
    swapchain: &RRSwapchain,
    graphics: &mut GraphicsResources,
) -> Result<()> {
    let mesh_count = load_result.meshes.len();
    let reserved_scene_objects = 4;
    let required_objects = graphics.objects.get_next_slot() + mesh_count + reserved_scene_objects;

    graphics.objects.ensure_capacity(
        instance,
        device,
        swapchain.swapchain_images.len(),
        required_objects,
    )?;

    Ok(())
}

pub(super) unsafe fn create_mesh_buffer(
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    graphics: &mut GraphicsResources,
    loaded_mesh: &crate::loader::LoadedMesh,
    mesh_index: usize,
    model_path: &str,
) -> Result<MeshBuffer> {
    let mut mesh = MeshBuffer::default();

    match &loaded_mesh.texture {
        Some(TextureSource::Embedded(tex)) => {
            (mesh.image, mesh.image_memory, mesh.mip_level) = create_texture_image_pixel(
                instance,
                device,
                command_pool,
                &tex.data,
                tex.width,
                tex.height,
            )?;
        }
        Some(TextureSource::File(texture_path)) => {
            let resolved = resolve_texture_path(texture_path, model_path);
            let load_path = resolved.to_string_lossy();
            match load_png_image(&load_path) {
                Ok((image_data, width, height)) => {
                    (mesh.image, mesh.image_memory, mesh.mip_level) = create_texture_image_pixel(
                        instance,
                        device,
                        command_pool,
                        &image_data,
                        width,
                        height,
                    )?;
                }
                Err(e) => {
                    log_warn!("Failed to load texture {}: {}", load_path, e);
                    let white_pixel = vec![255u8, 255, 255, 255];
                    (mesh.image, mesh.image_memory, mesh.mip_level) = create_texture_image_pixel(
                        instance,
                        device,
                        command_pool,
                        &white_pixel,
                        1,
                        1,
                    )?;
                }
            }
        }
        None => {
            let white_pixel = vec![255u8, 255, 255, 255];
            (mesh.image, mesh.image_memory, mesh.mip_level) =
                create_texture_image_pixel(instance, device, command_pool, &white_pixel, 1, 1)?;
        }
    }

    mesh.image_view = create_image_view(
        device,
        mesh.image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        mesh.mip_level,
    )?;
    mesh.sampler = create_texture_sampler(device, mesh.mip_level)?;

    mesh.vertex_data = loaded_mesh.vertex_data.clone();
    mesh.skin_data = loaded_mesh.skin_data.clone();
    mesh.skeleton_id = loaded_mesh.skeleton_id;
    mesh.node_index = loaded_mesh.node_index;
    mesh.base_vertices = loaded_mesh.local_vertices.clone();
    mesh.base_colors = Some(
        mesh.vertex_data
            .vertices
            .iter()
            .map(|v| cgmath::Vector4::new(v.color.x, v.color.y, v.color.z, v.color.w))
            .collect(),
    );

    mesh.vertex_buffer = RRVertexBuffer::new(
        instance,
        device,
        command_pool,
        (size_of::<vulkan_data::Vertex>() * mesh.vertex_data.vertices.len()) as vk::DeviceSize,
        mesh.vertex_data.vertices.as_ptr() as *const c_void,
        mesh.vertex_data.vertices.len(),
    )?;

    mesh.index_buffer = RRIndexBuffer::new(
        instance,
        device,
        command_pool,
        (size_of::<u32>() * mesh.vertex_data.indices.len()) as u64,
        mesh.vertex_data.indices.as_ptr() as *const c_void,
        mesh.vertex_data.indices.len(),
    )?;

    mesh.object_index = graphics.objects.allocate_slot();
    log!(
        "Allocated object_index {} for mesh {}",
        mesh.object_index,
        mesh_index
    );

    Ok(mesh)
}

pub(super) unsafe fn create_material_for_mesh(
    instance: &Instance,
    device: &RRDevice,
    graphics: &mut GraphicsResources,
    mesh: &MeshBuffer,
    mesh_index: usize,
    base_color_factor: [f32; 4],
) -> Result<MaterialId> {
    let material_name = format!("material_{}", mesh_index);
    let material_properties = MaterialUBO {
        base_color: Vector4::new(
            base_color_factor[0],
            base_color_factor[1],
            base_color_factor[2],
            base_color_factor[3],
        ),
        ..MaterialUBO::default()
    };

    let material_id = graphics.materials.create_material_with_texture(
        instance,
        device,
        &material_name,
        mesh.image_view,
        mesh.sampler,
        material_properties,
    )?;

    log!("Created material {} for mesh {}", material_id, mesh_index);
    Ok(material_id)
}

pub(super) unsafe fn apply_initial_pose(
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    graphics: &mut GraphicsResources,
    world: &mut World,
    assets: &AssetStorage,
    load_result: &ModelLoadResult,
) -> Result<()> {
    use crate::ecs::{compute_pose_global_transforms, create_pose_from_rest, sample_clip_to_pose};

    if load_result.clips.is_empty() {
        return Ok(());
    }

    log!("Applying initial pose (time=0) for animation...");

    if !load_result.clips.is_empty() {
        if world.contains_resource::<TimelineState>() {
            let mut timeline = world.resource_mut::<TimelineState>();
            timeline.playing = false;
            timeline.current_time = 0.0;
        }
        restore_batch_playback(world);
    }

    let skeleton_id = graphics.meshes.first().and_then(|m| m.skeleton_id);

    if let Some(skel_id) = skeleton_id {
        let (current_time, looping) = if world.contains_resource::<TimelineState>() {
            let timeline = world.resource::<TimelineState>();
            (timeline.current_time, timeline.looping)
        } else {
            (0.0, true)
        };
        let skeleton = assets.get_skeleton_by_skeleton_id(skel_id);
        let first_clip = assets.animation_clips.values().next().map(|a| &a.clip);

        if let (Some(skeleton), Some(clip)) = (skeleton, first_clip) {
            let mut pose = create_pose_from_rest(skeleton);
            sample_clip_to_pose(clip, current_time, skeleton, &mut pose, looping);
            let globals = compute_pose_global_transforms(skeleton, &pose);
            let skeleton_clone = skeleton.clone();

            for mesh_idx in 0..graphics.meshes.len() {
                apply_skinning_to_mesh(
                    instance,
                    device,
                    command_pool,
                    graphics,
                    &globals,
                    &skeleton_clone,
                    mesh_idx,
                )?;
            }
        }
    }

    let has_node_animation = !load_result.has_skinned_meshes && !graphics.meshes.is_empty();
    if has_node_animation {
        let mut node_assets = world.resource_mut::<NodeAssets>();
        let node_animation_scale = load_result.node_animation_scale;

        let skel_id = graphics.meshes.first().and_then(|m| m.skeleton_id);
        let skeleton_clone = skel_id.and_then(|id| assets.get_skeleton_by_skeleton_id(id).cloned());
        let clip_clone = assets
            .animation_clips
            .values()
            .next()
            .map(|a| a.clip.clone());

        let updated_meshes = if let (Some(skeleton), Some(clip)) = (&skeleton_clone, &clip_clone) {
            let mut pose = create_pose_from_rest(skeleton);
            sample_clip_to_pose(clip, 0.0, skeleton, &mut pose, false);

            crate::ecs::systems::animation::apply::prepare_node_animation(
                graphics,
                &mut node_assets.nodes,
                skeleton,
                &pose,
                node_animation_scale,
            )
        } else {
            Vec::new()
        };

        for mesh_idx in updated_meshes {
            if let Err(e) = upload_mesh_vertices(instance, device, command_pool, graphics, mesh_idx)
            {
                log!(
                    "Failed to upload initial node animation mesh {}: {}",
                    mesh_idx,
                    e
                );
            }
        }
    }

    log!("Initial pose applied successfully");
    Ok(())
}

unsafe fn apply_skinning_to_mesh(
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    graphics: &mut GraphicsResources,
    global_transforms: &[cgmath::Matrix4<f32>],
    skeleton: &crate::animation::Skeleton,
    mesh_idx: usize,
) -> Result<()> {
    use crate::ecs::systems::animation::apply::apply_skinning_to_single_mesh;

    if !apply_skinning_to_single_mesh(graphics, mesh_idx, global_transforms, skeleton) {
        return Ok(());
    }

    let mesh = &mut graphics.meshes[mesh_idx];
    if let Err(e) = mesh.vertex_buffer.update(
        instance,
        device,
        command_pool,
        (size_of::<vulkan_data::Vertex>() * mesh.vertex_data.vertices.len()) as vk::DeviceSize,
        mesh.vertex_data.vertices.as_ptr() as *const c_void,
        mesh.vertex_data.vertices.len(),
    ) {
        log!(
            "Failed to update vertex buffer for mesh {}: {}",
            mesh_idx,
            e
        );
    }

    Ok(())
}

unsafe fn upload_mesh_vertices(
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    graphics: &mut GraphicsResources,
    mesh_idx: usize,
) -> Result<()> {
    if mesh_idx >= graphics.meshes.len() {
        return Ok(());
    }

    let mesh = &mut graphics.meshes[mesh_idx];
    let vertices = &mesh.vertex_data.vertices;
    let vertex_count = vertices.len();
    let vertex_stride = size_of::<vulkan_data::Vertex>();

    mesh.vertex_buffer.update(
        instance,
        device,
        command_pool,
        (vertex_stride * vertex_count) as vk::DeviceSize,
        vertices.as_ptr() as *const c_void,
        vertex_count,
    )?;

    Ok(())
}

fn resolve_texture_path(texture_path: &str, model_path: &str) -> PathBuf {
    let original = Path::new(texture_path);
    if original.exists() {
        return original.to_path_buf();
    }

    let file_stem = original.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let file_name = original.file_name().and_then(|s| s.to_str()).unwrap_or("");

    let model_dir = Path::new(model_path)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let model_root = model_dir.parent().unwrap_or(model_dir);

    let texture_dir = original.parent().unwrap_or_else(|| Path::new("."));
    let texture_root = texture_dir.parent().unwrap_or(texture_dir);

    let mut search_dirs = vec![
        model_dir.to_path_buf(),
        model_dir.join("textures"),
        model_root.join("textures"),
    ];

    if texture_dir != model_dir {
        search_dirs.push(texture_dir.to_path_buf());
        search_dirs.push(texture_dir.join("textures"));
        search_dirs.push(texture_root.join("textures"));
    }

    let candidate_names: Vec<String> = vec![
        file_name.to_string(),
        format!("{}.png", file_name),
        format!("{}.png", file_stem),
        format!("{}.jpg", file_stem),
    ];

    for dir in &search_dirs {
        for name in &candidate_names {
            let candidate = dir.join(name);
            if candidate.exists() {
                log!(
                    "Resolved texture: {} -> {}",
                    texture_path,
                    candidate.display()
                );
                return candidate;
            }
        }
    }

    log!("Texture not found, using original path: {}", texture_path);
    original.to_path_buf()
}
