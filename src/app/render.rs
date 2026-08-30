use crate::app::init::MAX_FRAMES_IN_FLIGHT;
use crate::app::App;
use crate::ecs::resource::billboard::BillboardData;
use crate::ecs::resource::gizmo::{GridGizmoData, LightGizmoData};
use crate::ecs::resource::{Camera, GridMeshData};
use crate::ecs::systems::render_data_systems::{
    gizmo_mesh_render_data, gizmo_selectable_render_data, grid_mesh_render_data,
};
use crate::vulkanr::context::{
    CommandState, FrameSync, PipelineState, RenderTargets, SwapchainState,
};
use crate::vulkanr::descriptor::{CompositeGBufferViews, FlameImageBindings};
use crate::vulkanr::renderer::deferred::create_gbuffer_framebuffer;
use crate::vulkanr::renderer::scene_renderer::render_scene_objects;
use crate::vulkanr::vulkan::*;

use anyhow::{anyhow, Result};
use std::fs::OpenOptions;

impl App {
    pub unsafe fn begin_frame(&mut self) -> Result<usize> {
        self.handle_viewport_resize()?;
        self.handle_model_loading()?;

        let current_fence = self.resource::<FrameSync>().current_fence();
        self.rrdevice
            .device
            .wait_for_fences(&[current_fence], true, u64::MAX)?;

        self.update_auto_exposure();
        self.read_object_id_readback();

        let swapchain = self.resource::<SwapchainState>().swapchain.swapchain;
        let image_available = self.resource::<FrameSync>().current_image_available();
        let result = self.rrdevice.device.acquire_next_image_khr(
            swapchain,
            u64::MAX,
            image_available,
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return Err(anyhow!("SWAPCHAIN_OUT_OF_DATE")),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.resource::<SwapchainState>().images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.rrdevice
                .device
                .wait_for_fences(&[image_in_flight], true, u64::MAX)?;
        }

        let current_fence = self.resource::<FrameSync>().current_fence();
        self.resource_mut::<SwapchainState>().images_in_flight[image_index] = current_fence;

        Ok(image_index)
    }

    unsafe fn handle_viewport_resize(&mut self) -> Result<()> {
        let pending = self
            .data
            .ecs_world
            .resource_mut::<crate::ecs::resource::ViewportInput>()
            .resize_pending
            .take();
        let Some((width, height)) = pending else {
            return Ok(());
        };

        self.rrdevice.device.device_wait_idle()?;
        let command_pool = self.resource::<CommandState>().pool.command_pool;
        self.data
            .viewport
            .resize(&self.instance, &self.rrdevice, command_pool, width, height)?;

        self.resize_gbuffer(width, height)?;
        self.update_postprocessing_descriptors_on_resize()?;

        if self
            .data
            .ecs_world
            .contains_resource::<crate::ecs::resource::ObjectIdReadback>()
        {
            let mut readback = self
                .data
                .ecs_world
                .resource_mut::<crate::ecs::resource::ObjectIdReadback>();
            readback.pending_pixel = None;
            readback.copy_in_flight = false;
            readback.last_read_object_id = None;
            readback.last_read_world_position = None;
        }

        Ok(())
    }

    unsafe fn update_postprocessing_descriptors_on_resize(&mut self) -> Result<()> {
        if let (Some(ref hdr_buffer), Some(ref tonemap_descriptor)) = (
            &self.data.viewport.hdr_buffer,
            &self.data.raytracing.tonemap_descriptor,
        ) {
            tonemap_descriptor.update_hdr_sampler(
                &self.rrdevice,
                hdr_buffer.color_image_view,
                hdr_buffer.sampler,
            )?;

            let bloom_view_and_sampler =
                self.data.viewport.bloom_chain.as_ref().and_then(|chain| {
                    chain
                        .mip_levels
                        .first()
                        .map(|mip| (mip.image_view, chain.sampler))
                });

            if let Some((bloom_view, bloom_sampler)) = bloom_view_and_sampler {
                tonemap_descriptor.update_bloom_sampler(
                    &self.rrdevice,
                    bloom_view,
                    bloom_sampler,
                )?;
            } else {
                tonemap_descriptor.update_bloom_sampler(
                    &self.rrdevice,
                    hdr_buffer.color_image_view,
                    hdr_buffer.sampler,
                )?;
            }
        }

        if let (Some(ref hdr_buffer), Some(ref bloom_chain), Some(ref bloom_descriptors)) = (
            &self.data.viewport.hdr_buffer,
            &self.data.viewport.bloom_chain,
            &self.data.raytracing.bloom_descriptors,
        ) {
            let mip_views: Vec<vk::ImageView> = bloom_chain
                .mip_levels
                .iter()
                .map(|m| m.image_view)
                .collect();

            bloom_descriptors.update_image_views(
                &self.rrdevice,
                hdr_buffer.color_image_view,
                &mip_views,
                bloom_chain.sampler,
            )?;
        }

        {
            let render_targets = self.resource::<crate::vulkanr::context::RenderTargets>();
            let depth_image_view = render_targets.render.gbuffer_depth_image_view;

            if let (Some(ref hdr_buffer), Some(ref dof_descriptor)) = (
                &self.data.viewport.hdr_buffer,
                &self.data.raytracing.dof_descriptor,
            ) {
                let depth_sampler = self
                    .data
                    .raytracing
                    .gbuffer_sampler
                    .unwrap_or(hdr_buffer.sampler);

                dof_descriptor.update_image_views(
                    &self.rrdevice,
                    hdr_buffer.color_image_view,
                    hdr_buffer.sampler,
                    depth_image_view,
                    depth_sampler,
                )?;
            }
        }

        if let (Some(ref dof_buffer), Some(ref tonemap_descriptor)) = (
            &self.data.viewport.dof_buffer,
            &self.data.raytracing.tonemap_descriptor,
        ) {
            tonemap_descriptor.update_hdr_sampler(
                &self.rrdevice,
                dof_buffer.output_image_view,
                dof_buffer.sampler,
            )?;
        }

        if let (Some(ref gbuffer), Some(gbuffer_sampler), Some(ref tonemap_descriptor)) = (
            &self.data.raytracing.gbuffer,
            self.data.raytracing.gbuffer_sampler,
            &self.data.raytracing.tonemap_descriptor,
        ) {
            tonemap_descriptor.update_position_sampler(
                &self.rrdevice,
                gbuffer.position_image_view,
                gbuffer_sampler,
            )?;
        }

        self.update_auto_exposure_descriptors_on_resize()?;

        Ok(())
    }

    pub unsafe fn load_model(&mut self, path: &str) -> Result<()> {
        log!("Loading new model from: {}", path);
        self.rrdevice.device.device_wait_idle()?;

        let command_pool = self.resource::<CommandState>().pool.clone();
        let swapchain = self.resource::<SwapchainState>().swapchain.clone();
        match Self::load_model_from_path_with_resources(
            &self.instance,
            &self.rrdevice,
            &mut self.data,
            &command_pool,
            &swapchain,
            path,
            false,
        ) {
            Ok(_) => {
                {
                    let mut model_state = self
                        .data
                        .ecs_world
                        .resource_mut::<crate::ecs::resource::ModelState>();
                    model_state.model_path = path.to_string();
                    model_state.load_status = format!("Loaded: {}", path);
                }
                {
                    let mut timeline = self
                        .data
                        .ecs_world
                        .resource_mut::<crate::ecs::resource::TimelineState>();
                    timeline.current_time = 0.0;
                }
                {
                    let mut scene_state =
                        self.data.ecs_world.resource_mut::<crate::ecs::SceneState>();
                    scene_state.clear();
                }

                msg_info!("Model loaded: {}", path);
            }
            Err(e) => {
                let mut model_state = self
                    .data
                    .ecs_world
                    .resource_mut::<crate::ecs::resource::ModelState>();
                model_state.load_status = format!("Error: {}", e);
                msg_error!("Failed to load model: {:?}", e);
            }
        }

        Ok(())
    }

    #[cfg(feature = "auto-rig")]
    pub unsafe fn load_model_from_glb(&mut self, glb_data: &[u8]) -> Result<()> {
        log!("Loading generated mesh from GLB ({} bytes)", glb_data.len());
        self.rrdevice.device.device_wait_idle()?;

        let gltf_result = crate::loader::gltf::load_gltf_from_slice(glb_data)?;
        let load_result = crate::loader::ModelLoadResult::from_gltf(gltf_result);

        let command_pool = self.resource::<CommandState>().pool.clone();
        let swapchain = self.resource::<SwapchainState>().swapchain.clone();
        match crate::app::model_loader::load_model_from_file_system_with_result(
            &load_result,
            crate::scene::ModelReference::GENERATED_MESH,
            &self.instance,
            &self.rrdevice,
            &command_pool,
            &swapchain,
            &mut self.data.graphics_resources,
            &mut self.data.raytracing,
            &mut self.data.ecs_world,
            &mut self.data.ecs_assets,
            false,
            None,
        ) {
            Ok(parent_entity) => {
                {
                    let mut model_state = self
                        .data
                        .ecs_world
                        .resource_mut::<crate::ecs::resource::ModelState>();
                    model_state.model_path =
                        crate::scene::ModelReference::GENERATED_MESH.to_string();
                    model_state.load_status = "Loaded: Generated Mesh".to_string();
                }
                {
                    let mut timeline = self
                        .data
                        .ecs_world
                        .resource_mut::<crate::ecs::resource::TimelineState>();
                    timeline.current_time = 0.0;
                }
                {
                    let mut scene_state =
                        self.data.ecs_world.resource_mut::<crate::ecs::SceneState>();
                    scene_state.clear();
                }
                {
                    let cache =
                        crate::ecs::resource::GltfModelCache::from_glb_data(glb_data.to_vec());
                    self.data.ecs_world.insert_resource(cache);
                }
                self.data.ecs_world.insert_component(
                    parent_entity,
                    crate::ecs::component::GlbSource::InMemory(glb_data.to_vec()),
                );

                msg_info!("Generated mesh loaded successfully");
            }
            Err(e) => {
                let mut model_state = self
                    .data
                    .ecs_world
                    .resource_mut::<crate::ecs::resource::ModelState>();
                model_state.load_status = format!("Error: {}", e);
                return Err(e);
            }
        }

        Ok(())
    }

    pub unsafe fn load_model_additive(&mut self, path: &str) -> Result<()> {
        log!("Additively loading model from: {}", path);
        self.rrdevice.device.device_wait_idle()?;

        let command_pool = self.resource::<CommandState>().pool.clone();
        let swapchain = self.resource::<SwapchainState>().swapchain.clone();

        crate::app::model_loader::load_model_additive(
            path,
            &self.instance,
            &self.rrdevice,
            &command_pool,
            &swapchain,
            &mut self.data.graphics_resources,
            &mut self.data.raytracing,
            &mut self.data.ecs_world,
            &mut self.data.ecs_assets,
        )?;

        msg_info!("Model added: {}", path);
        Ok(())
    }

    pub unsafe fn delete_entities(&mut self, entities: &[u64]) -> Result<()> {
        self.rrdevice.device.device_wait_idle()?;

        for &entity in entities {
            let mesh_ref = self
                .data
                .ecs_world
                .get_component::<crate::ecs::world::MeshRef>(entity)
                .cloned();

            if let Some(mesh_ref) = mesh_ref {
                let graphics_mesh_index = self
                    .data
                    .ecs_assets
                    .get_mesh(mesh_ref.mesh_asset_id)
                    .map(|m| m.graphics_mesh_index);

                if let Some(idx) = graphics_mesh_index {
                    if idx < self.data.graphics_resources.meshes.len() {
                        self.data.graphics_resources.meshes[idx].render_to_gbuffer = false;
                        self.data.graphics_resources.meshes[idx].destroy(&self.rrdevice);
                    }
                }
            }

            self.data.ecs_world.despawn(entity);
        }

        let command_pool = self.resource::<CommandState>().pool.clone();
        crate::app::model_loader::rebuild_acceleration_structures(
            &self.instance,
            &self.rrdevice,
            &command_pool,
            &self.data.graphics_resources,
            &mut self.data.raytracing,
        )?;

        log!("Deleted {} entities with GPU cleanup", entities.len());
        Ok(())
    }

    unsafe fn handle_model_loading(&mut self) -> Result<()> {
        Ok(())
    }

    unsafe fn update_auto_exposure_descriptors_on_resize(&self) -> Result<()> {
        let (hdr_image_view, hdr_sampler) =
            if let Some(ref dof_buffer) = self.data.viewport.dof_buffer {
                (dof_buffer.output_image_view, dof_buffer.sampler)
            } else if let Some(ref hdr_buffer) = self.data.viewport.hdr_buffer {
                (hdr_buffer.color_image_view, hdr_buffer.sampler)
            } else {
                return Ok(());
            };

        let ae_buffers = match self.data.viewport.auto_exposure_buffers {
            Some(ref buf) => buf,
            None => return Ok(()),
        };

        if let Some(ref hist_desc) = self.data.raytracing.auto_exposure_histogram_descriptor {
            hist_desc.update_bindings(
                &self.rrdevice,
                hdr_image_view,
                hdr_sampler,
                ae_buffers.histogram_buffer,
                (256 * 4) as u64,
            )?;
        }

        if let Some(ref avg_desc) = self.data.raytracing.auto_exposure_average_descriptor {
            avg_desc.update_bindings(
                &self.rrdevice,
                ae_buffers.histogram_buffer,
                (256 * 4) as u64,
                ae_buffers.luminance_buffer,
                (2 * 4) as u64,
            )?;
        }
        Ok(())
    }

    unsafe fn resize_gbuffer(&mut self, new_width: u32, new_height: u32) -> Result<()> {
        let needs_resize = self
            .data
            .raytracing
            .gbuffer
            .as_ref()
            .map(|gb| gb.width != new_width || gb.height != new_height)
            .unwrap_or(false);

        if !needs_resize {
            self.attach_gbuffer_depth_to_hdr()?;
            return Ok(());
        }

        let command_pool = self.resource::<CommandState>().pool.command_pool;

        if let Some(ref mut gbuffer) = self.data.raytracing.gbuffer {
            gbuffer.resize(&self.instance, &self.rrdevice, new_width, new_height)?;
            gbuffer.transition_layouts(&self.rrdevice, command_pool)?;
        }

        let Some(ref gbuffer) = self.data.raytracing.gbuffer else {
            return Ok(());
        };
        let position_view = gbuffer.position_image_view;
        let normal_view = gbuffer.normal_image_view;
        let shadow_mask_view = gbuffer.shadow_mask_image_view;
        let albedo_view = gbuffer.albedo_image_view;
        let object_id_view = gbuffer.object_id_image_view;
        self.recreate_gbuffer_framebuffer()?;
        self.attach_gbuffer_depth_to_hdr()?;
        self.recreate_water_on_resize()?;
        self.update_gbuffer_descriptors(
            position_view,
            normal_view,
            shadow_mask_view,
            albedo_view,
            object_id_view,
        )?;
        self.recreate_onion_skin_on_resize()?;
        self.recreate_flame_on_resize()?;
        log!("G-Buffer resized to: {}x{}", new_width, new_height);
        Ok(())
    }

    /// The HDR framebuffer borrows the G-buffer depth view; a resized HDR
    /// buffer has no framebuffer until the current depth is attached here.
    unsafe fn attach_gbuffer_depth_to_hdr(&mut self) -> Result<()> {
        let depth_view = {
            let rt = self.resource::<RenderTargets>();
            rt.render.gbuffer_depth_image_view
        };
        if depth_view == vk::ImageView::null() {
            return Ok(());
        }
        if let Some(hdr_buffer) = &mut self.data.viewport.hdr_buffer {
            hdr_buffer.attach_depth(&self.rrdevice, depth_view)?;
        }
        Ok(())
    }

    unsafe fn recreate_water_on_resize(&mut self) -> Result<()> {
        let depth_view = {
            let rt = self.resource::<RenderTargets>();
            rt.render.gbuffer_depth_image_view
        };
        let hdr_view = match &self.data.viewport.hdr_buffer {
            Some(hdr) => hdr.color_image_view,
            None => return Ok(()),
        };
        if depth_view == vk::ImageView::null() {
            return Ok(());
        }

        let width = self.data.viewport.width;
        let height = self.data.viewport.height;

        // Destroy old buffer if it exists
        if let Some(mut old_buffer) = self.data.viewport.water_buffer.take() {
            old_buffer.destroy(&self.rrdevice.device);
        }

        // Recreate with new dimensions
        let water_buffer = thyllore_vulkan_core::resource::WaterBuffer::new(
            &self.rrdevice,
            width,
            height,
            hdr_view,
            depth_view,
        )?;
        self.data.viewport.water_buffer = Some(water_buffer);

        Ok(())
    }

    unsafe fn recreate_flame_on_resize(&mut self) -> Result<()> {
        let (Some(ref flame_buffer), Some(ref flame_descriptor)) = (
            &self.data.viewport.flame_buffer,
            &self.data.raytracing.flame_descriptor,
        ) else {
            return Ok(());
        };
        flame_descriptor.update_image_views(
            &self.rrdevice,
            FlameImageBindings {
                history_image_views: flame_buffer.history_image_views,
                flame_sampler: flame_buffer.sampler,
                sdf_image_view: self.data.raytracing.flame_sdf_image_view,
                sdf_sampler: self.data.raytracing.flame_sdf_sampler,
                scene_depth_view: self
                    .resource::<RenderTargets>()
                    .render
                    .gbuffer_depth_image_view,
            },
        )?;

        if let Some(mut state) = self
            .data
            .ecs_world
            .get_resource_mut::<crate::ecs::resource::FlameTemporalState>()
        {
            state.previous = None;
        }

        Ok(())
    }

    unsafe fn recreate_gbuffer_framebuffer(&mut self) -> Result<()> {
        let mut render_targets = self.resource_mut::<RenderTargets>();
        let device = &self.rrdevice.device;

        if render_targets.render.gbuffer_framebuffer != vk::Framebuffer::null() {
            device.destroy_framebuffer(render_targets.render.gbuffer_framebuffer, None);
            render_targets.render.gbuffer_framebuffer = vk::Framebuffer::null();
        }
        if render_targets.render.gbuffer_depth_image_view != vk::ImageView::null() {
            device.destroy_image_view(render_targets.render.gbuffer_depth_image_view, None);
            render_targets.render.gbuffer_depth_image_view = vk::ImageView::null();
        }
        if render_targets.render.gbuffer_depth_image != vk::Image::null() {
            device.destroy_image(render_targets.render.gbuffer_depth_image, None);
            render_targets.render.gbuffer_depth_image = vk::Image::null();
        }
        if render_targets.render.gbuffer_depth_image_memory != vk::DeviceMemory::null() {
            device.free_memory(render_targets.render.gbuffer_depth_image_memory, None);
            render_targets.render.gbuffer_depth_image_memory = vk::DeviceMemory::null();
        }

        if let Some(ref gbuffer) = self.data.raytracing.gbuffer {
            create_gbuffer_framebuffer(
                &self.instance,
                &self.rrdevice,
                &mut render_targets.render,
                gbuffer,
            )?;
        }

        Ok(())
    }

    unsafe fn update_gbuffer_descriptors(
        &mut self,
        position_view: vk::ImageView,
        normal_view: vk::ImageView,
        shadow_mask_view: vk::ImageView,
        albedo_view: vk::ImageView,
        object_id_view: vk::ImageView,
    ) -> Result<()> {
        let gbuffer_sampler = self
            .data
            .raytracing
            .gbuffer_sampler
            .unwrap_or(vk::Sampler::null());
        let object_id_sampler = self
            .data
            .raytracing
            .object_id_sampler
            .unwrap_or(vk::Sampler::null());

        if let Some(ref composite_desc) = self.data.raytracing.composite_descriptor {
            composite_desc.update_gbuffer_views(
                &self.rrdevice,
                CompositeGBufferViews {
                    position_image_view: position_view,
                    position_sampler: gbuffer_sampler,
                    normal_image_view: normal_view,
                    normal_sampler: gbuffer_sampler,
                    shadow_mask_image_view: shadow_mask_view,
                    shadow_mask_sampler: gbuffer_sampler,
                    albedo_image_view: albedo_view,
                    albedo_sampler: gbuffer_sampler,
                    object_id_image_view: object_id_view,
                    object_id_sampler,
                },
            )?;
        }

        if let Some(ref ray_query_desc) = self.data.raytracing.ray_query_descriptor {
            ray_query_desc.update_gbuffer_views(
                &self.rrdevice,
                position_view,
                normal_view,
                shadow_mask_view,
            )?;
        }

        {
            let swapchain = self.resource::<SwapchainState>().swapchain.clone();
            let mut billboard = self.resource_mut::<BillboardData>();
            billboard
                .render_state
                .descriptor_set
                .update_position_sampler(
                    &self.rrdevice,
                    &swapchain,
                    position_view,
                    gbuffer_sampler,
                )?;
        }

        Ok(())
    }

    unsafe fn recreate_onion_skin_on_resize(&mut self) -> Result<()> {
        if let (Some(ref mut onion_pass), Some(ref offscreen)) = (
            &mut self.data.raytracing.onion_skin_pass,
            &self.data.viewport.offscreen,
        ) {
            onion_pass.recreate_on_resize(
                &self.instance,
                &self.rrdevice,
                offscreen.resolve_color_image_view,
                offscreen.width,
                offscreen.height,
            )?;
        }

        Ok(())
    }

    unsafe fn read_object_id_readback(&mut self) {
        use crate::ecs::resource::ObjectIdReadback;

        if !self.data.ecs_world.contains_resource::<ObjectIdReadback>() {
            return;
        }

        let readback = self.data.ecs_world.resource::<ObjectIdReadback>();
        if !readback.copy_in_flight {
            return;
        }
        drop(readback);

        let Some(ref gbuffer) = self.data.raytracing.gbuffer else {
            return;
        };

        let memory = self
            .rrdevice
            .device
            .map_memory(
                gbuffer.readback_staging_memory,
                0,
                thyllore_vulkan_core::resource::READBACK_STAGING_SIZE,
                vk::MemoryMapFlags::empty(),
            )
            .ok();

        let picked = memory.map(|ptr| {
            let object_id = *(ptr as *const u32);
            let position_ptr = (ptr as *const u8)
                .add(thyllore_vulkan_core::resource::READBACK_POSITION_OFFSET as usize)
                as *const f32;
            let world_position = [*position_ptr, *position_ptr.add(1), *position_ptr.add(2)];
            self.rrdevice
                .device
                .unmap_memory(gbuffer.readback_staging_memory);
            (object_id, world_position)
        });

        if let Some((object_id, world_position)) = picked {
            let mut readback = self.data.ecs_world.resource_mut::<ObjectIdReadback>();
            readback.last_read_object_id = Some(object_id);
            readback.last_read_world_position = (object_id != 0).then_some(world_position);
            readback.copy_in_flight = false;
        }
    }

    unsafe fn update_auto_exposure(&mut self) {
        let ae_enabled = self
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::AutoExposure>()
            .map(|ae| ae.enabled)
            .unwrap_or(false);

        // Get frame number from BatchRun if available, otherwise use internal counter
        let frame = match self
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::BatchRun>()
        {
            Some(batch_run) => batch_run.frames_rendered,
            None => match self
                .data
                .ecs_world
                .get_resource_mut::<crate::ecs::resource::ExposureDumpSink>()
            {
                Some(mut sink) => {
                    sink.last_frame += 1;
                    sink.last_frame
                }
                None => 0,
            },
        };

        if !ae_enabled {
            // Still record exposure dump even when AE is disabled
            if let Some(mut sink) = self
                .data
                .ecs_world
                .get_resource_mut::<crate::ecs::resource::ExposureDumpSink>()
            {
                let exposure_value = self
                    .data
                    .ecs_world
                    .get_resource::<crate::ecs::resource::Exposure>()
                    .map(|e| e.exposure_value)
                    .unwrap_or(1.0);
                let line = format!(
                    "{{\"frame\":{},\"adapted\":0.0,\"exposure_value\":{},\"ae_enabled\":false}}\n",
                    frame, exposure_value
                );
                append_jsonl(&sink.path, &line);
            }
            self.restore_manual_exposure_if_needed();
            return;
        }

        self.save_manual_exposure_if_needed();
        // batch 決定性: AE 読み戻しを直前フレーム完了後に固定する
        if self
            .data
            .ecs_world
            .contains_resource::<crate::ecs::resource::BatchRun>()
        {
            let _ = self.rrdevice.device.device_wait_idle();
        }

        let adapted = match self.data.viewport.auto_exposure_buffers {
            Some(ref ae_buffers) => {
                let current_slot = self.resource::<FrameSync>().current_frame;
                ae_buffers.read_adapted_exposure(&self.rrdevice.device, current_slot)
            }
            None => return,
        };

        if adapted > 0.0 {
            if let Some(mut exposure) = self
                .data
                .ecs_world
                .get_resource_mut::<crate::ecs::resource::Exposure>()
            {
                exposure.exposure_value = adapted;
            }
        }

        // Record exposure dump
        if let Some(mut sink) = self
            .data
            .ecs_world
            .get_resource_mut::<crate::ecs::resource::ExposureDumpSink>()
        {
            let exposure_value = self
                .data
                .ecs_world
                .get_resource::<crate::ecs::resource::Exposure>()
                .map(|e| e.exposure_value)
                .unwrap_or(1.0);
            let line = format!(
                "{{\"frame\":{},\"adapted\":{},\"exposure_value\":{},\"ae_enabled\":true}}\n",
                frame, adapted, exposure_value
            );
            append_jsonl(&sink.path, &line);
        }
    }

    fn save_manual_exposure_if_needed(&mut self) {
        let already_saved = self
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::AutoExposure>()
            .map(|ae| ae.saved_manual_exposure.is_some())
            .unwrap_or(true);

        if already_saved {
            return;
        }

        let current_exposure = self
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::Exposure>()
            .map(|e| e.exposure_value)
            .unwrap_or(1.0);

        if let Some(mut ae) = self
            .data
            .ecs_world
            .get_resource_mut::<crate::ecs::resource::AutoExposure>()
        {
            ae.saved_manual_exposure = Some(current_exposure);
        }
    }

    fn restore_manual_exposure_if_needed(&mut self) {
        let saved = self
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::AutoExposure>()
            .and_then(|ae| ae.saved_manual_exposure);

        let restore_value = match saved {
            Some(v) => v,
            None => return,
        };

        if let Some(mut exposure) = self
            .data
            .ecs_world
            .get_resource_mut::<crate::ecs::resource::Exposure>()
        {
            exposure.exposure_value = restore_value;
        }

        if let Some(mut ae) = self
            .data
            .ecs_world
            .get_resource_mut::<crate::ecs::resource::AutoExposure>()
        {
            ae.saved_manual_exposure = None;
        }
    }

    pub unsafe fn render(&mut self, image_index: usize, draw_data: &imgui::DrawData) -> Result<()> {
        let frame_slot = self.resource::<FrameSync>().current_frame;

        Self::update_imgui_buffers(
            &self.instance,
            &self.rrdevice,
            &mut self.data,
            draw_data,
            frame_slot,
        )?;

        self.record_command_buffer(image_index, draw_data, frame_slot)?;

        let image_available = self.resource::<FrameSync>().current_image_available();
        let render_finished = self.resource::<FrameSync>().current_render_finished();
        let current_fence = self.resource::<FrameSync>().current_fence();

        let wait_semaphores = &[image_available];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers =
            &[self.resource::<CommandState>().buffers.command_buffers[image_index]];
        let signal_semaphores = &[render_finished];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.rrdevice.device.reset_fences(&[current_fence])?;
        self.rrdevice.device.queue_submit(
            self.rrdevice.graphics_queue,
            &[submit_info],
            current_fence,
        )?;

        let swapchain = self.resource::<SwapchainState>().swapchain.swapchain;
        let swapchains = &[swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);
        let present_result = self
            .rrdevice
            .device
            .queue_present_khr(self.rrdevice.present_queue, &present_info);
        let changed = present_result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || present_result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        if changed || self.resized {
            self.resized = false;
            return Err(anyhow!("SWAPCHAIN_OUT_OF_DATE"));
        } else if let Err(e) = present_result {
            return Err(anyhow!(e));
        }

        self.resource_mut::<FrameSync>()
            .advance(MAX_FRAMES_IN_FLIGHT);
        let current_frame = self.resource::<FrameSync>().current_frame;
        self.frame = current_frame;

        Ok(())
    }
    pub unsafe fn save_screenshot(&self, image_index: usize) -> Result<String> {
        let device = &self.rrdevice.device;
        let swapchain = &self.resource::<SwapchainState>().swapchain;
        let swapchain_image = swapchain.swapchain_images[image_index];
        let extent = swapchain.swapchain_extent;
        let width = extent.width;
        let height = extent.height;
        let image_size = (width * height * 4) as vk::DeviceSize;
        let command_pool = self.resource::<CommandState>().pool.command_pool;

        let (buffer, buffer_memory, command_buffer) = self.copy_image_to_buffer(
            swapchain_image,
            extent.width,
            extent.height,
            image_size,
            command_pool,
            vk::ImageLayout::PRESENT_SRC_KHR,
        )?;

        let path = Self::encode_and_save_png(device, buffer_memory, image_size, width, height)?;

        device.free_command_buffers(command_pool, &[command_buffer]);
        device.free_memory(buffer_memory, None);
        device.destroy_buffer(buffer, None);

        Ok(path)
    }

    pub unsafe fn save_flame_history_npy(&self, path: &std::path::Path) -> Result<()> {
        use crate::app::util::{f16_to_f32, write_npy_f32};

        let device = &self.rrdevice.device;
        let flame_buffer = self
            .data
            .viewport
            .flame_buffer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("flame buffer not initialized"))?;

        let flames = self.data.ecs_world.query_flames();
        let history_index = if let Some(first) = flames.first() {
            if let Some(temporal) = self
                .data
                .ecs_world
                .get_component::<crate::ecs::component::FlameTemporalAccum>(*first)
            {
                (temporal.frame_index as usize) & 1
            } else {
                0
            }
        } else {
            0
        };

        let history_image = flame_buffer.history_images[history_index];
        let width = flame_buffer.width;
        let height = flame_buffer.height;
        let image_size = (width * height * 8) as vk::DeviceSize;
        let command_pool = self.resource::<CommandState>().pool.command_pool;

        let (buffer, buffer_memory, command_buffer) = self.copy_image_to_buffer(
            history_image,
            width,
            height,
            image_size,
            command_pool,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;

        let data_ptr =
            device.map_memory(buffer_memory, 0, image_size, vk::MemoryMapFlags::empty())?;
        let slice = std::slice::from_raw_parts(data_ptr as *const u8, image_size as usize);

        let mut f32_data = Vec::with_capacity((width * height * 4) as usize);
        for chunk in slice.chunks_exact(8) {
            let r_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let g_bits = u16::from_le_bytes([chunk[2], chunk[3]]);
            let b_bits = u16::from_le_bytes([chunk[4], chunk[5]]);
            let a_bits = u16::from_le_bytes([chunk[6], chunk[7]]);
            f32_data.push(f16_to_f32(r_bits));
            f32_data.push(f16_to_f32(g_bits));
            f32_data.push(f16_to_f32(b_bits));
            f32_data.push(f16_to_f32(a_bits));
        }

        device.unmap_memory(buffer_memory);
        device.free_command_buffers(command_pool, &[command_buffer]);
        device.free_memory(buffer_memory, None);
        device.destroy_buffer(buffer, None);

        write_npy_f32(path, &[height as usize, width as usize, 4], &f32_data)?;

        Ok(())
    }

    pub unsafe fn copy_image_to_buffer(
        &self,
        image: vk::Image,
        width: u32,
        height: u32,
        image_size: vk::DeviceSize,
        command_pool: vk::CommandPool,
        layout: vk::ImageLayout,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, vk::CommandBuffer)> {
        let device = &self.rrdevice.device;

        let (buffer, buffer_memory) = self.allocate_transfer_buffer(device, image_size)?;

        let command_buffer = allocate_one_time_command_buffer(device, command_pool)?;

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(command_buffer, &begin_info)?;

        record_image_to_buffer_copy(
            device,
            command_buffer,
            image,
            buffer,
            width,
            height,
            layout,
            layout,
        );

        device.end_command_buffer(command_buffer)?;

        let command_buffers_slice = [command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers_slice);
        device.queue_submit(
            self.rrdevice.graphics_queue,
            &[submit_info.build()],
            vk::Fence::null(),
        )?;
        device.queue_wait_idle(self.rrdevice.graphics_queue)?;

        Ok((buffer, buffer_memory, command_buffer))
    }

    unsafe fn allocate_transfer_buffer(
        &self,
        device: &crate::vulkanr::core::device::Device,
        image_size: vk::DeviceSize,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(image_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = device.create_buffer(&buffer_info, None)?;

        let mem_requirements = device.get_buffer_memory_requirements(buffer);
        let memory_type_index = self.get_memory_type_index(
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);
        let buffer_memory = device.allocate_memory(&alloc_info, None)?;
        device.bind_buffer_memory(buffer, buffer_memory, 0)?;

        Ok((buffer, buffer_memory))
    }

    unsafe fn encode_and_save_png(
        device: &crate::vulkanr::core::device::Device,
        buffer_memory: vk::DeviceMemory,
        image_size: vk::DeviceSize,
        width: u32,
        height: u32,
    ) -> Result<String> {
        use std::fs::File;
        use std::io::BufWriter;
        use std::time::SystemTime;

        let data = device.map_memory(buffer_memory, 0, image_size, vk::MemoryMapFlags::empty())?;
        let slice = std::slice::from_raw_parts(data as *const u8, image_size as usize);

        let mut rgba_data = vec![0u8; (width * height * 4) as usize];
        for i in (0..rgba_data.len()).step_by(4) {
            rgba_data[i] = slice[i + 2];
            rgba_data[i + 1] = slice[i + 1];
            rgba_data[i + 2] = slice[i];
            rgba_data[i + 3] = slice[i + 3];
        }

        device.unmap_memory(buffer_memory);

        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_secs();
        let filename = format!("log/screenshot_{}.png", timestamp);
        std::fs::create_dir_all("log")?;

        let file = File::create(&filename)?;
        let writer = BufWriter::new(file);
        let mut encoder = png::Encoder::new(writer, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut png_writer = encoder.write_header()?;
        png_writer.write_image_data(&rgba_data)?;

        let absolute_path = std::fs::canonicalize(&filename)
            .unwrap_or_else(|_| std::path::PathBuf::from(&filename));
        let path_str = absolute_path.to_string_lossy().to_string();

        log!("Screenshot saved to: {}", path_str);

        Ok(path_str)
    }

    pub unsafe fn begin_offscreen_render_pass(
        &self,
        command_buffer: vk::CommandBuffer,
        offscreen: &crate::vulkanr::resource::OffscreenFramebuffer,
    ) {
        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(offscreen.extent());

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.1, 0.1, 0.1, 1.0],
            },
        };
        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
        };
        let clear_values = [color_clear_value, depth_clear_value];

        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(offscreen.render_pass)
            .framebuffer(offscreen.framebuffer)
            .render_area(render_area)
            .clear_values(&clear_values);

        self.rrdevice.device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
    }

    pub unsafe fn record_3d_rendering_to_offscreen(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        offscreen: &crate::vulkanr::resource::OffscreenFramebuffer,
    ) -> Result<()> {
        let extent = offscreen.extent();

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        self.rrdevice
            .device
            .cmd_set_viewport(command_buffer, 0, &[viewport]);

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(extent);
        self.rrdevice
            .device
            .cmd_set_scissor(command_buffer, 0, &[scissor]);

        let frame_set = self.data.graphics_resources.frame_set.sets[image_index];
        let camera_pos = {
            use crate::ecs::systems::camera_systems::compute_camera_position;
            compute_camera_position(&self.resource::<Camera>())
        };

        let render_data_vec = vec![
            crate::ecs::systems::render_data_systems::grid_mesh_render_data(
                &self.resource::<GridMeshData>(),
            ),
            crate::ecs::systems::render_data_systems::gizmo_mesh_render_data(
                &self.resource::<GridGizmoData>(),
            ),
            crate::ecs::systems::render_data_systems::gizmo_selectable_render_data(
                &self.resource::<LightGizmoData>(),
                camera_pos,
            ),
        ];
        let render_data_refs: Vec<_> = render_data_vec.iter().collect();

        crate::vulkanr::renderer::scene_renderer::render_scene_objects(
            &render_data_refs,
            command_buffer,
            image_index,
            frame_set,
            &self.data.graphics_resources.objects,
            &self.rrdevice,
            self.pipeline_storage(),
            &self.data.buffer_registry,
        );

        self.render_billboard(command_buffer, image_index);
        self.render_models(command_buffer, image_index);

        Ok(())
    }

    pub unsafe fn begin_main_render_pass(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) {
        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.resource::<SwapchainState>().swapchain.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
        };
        let clear_values = [color_clear_value, depth_clear_value];

        let render_targets = self.resource::<RenderTargets>();
        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_targets.render.render_pass)
            .framebuffer(render_targets.render.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(&clear_values);

        self.rrdevice.device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
    }

    unsafe fn render_models(&self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let mesh_count = self.data.graphics_resources.meshes.len();

        for i in 0..mesh_count {
            let mesh = &self.data.graphics_resources.meshes[i];

            if !mesh.render_to_gbuffer {
                continue;
            }

            let pipeline = &self.resource::<PipelineState>().model_pipeline;
            self.rrdevice.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );

            self.rrdevice.device.cmd_set_line_width(command_buffer, 1.0);

            self.rrdevice.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[mesh.vertex_buffer.buffer],
                &[0],
            );

            self.rrdevice.device.cmd_bind_index_buffer(
                command_buffer,
                mesh.index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );

            let frame_set = self.data.graphics_resources.frame_set.sets[image_index];
            self.rrdevice.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline_layout,
                0,
                &[frame_set],
                &[],
            );

            if let Some(material_id) = self.data.graphics_resources.get_material_id(i) {
                if let Some(material) = self.data.graphics_resources.materials.get(material_id) {
                    self.rrdevice.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline_layout,
                        1,
                        &[material.descriptor_set],
                        &[],
                    );
                }
            }

            let object_set_idx = self
                .data
                .graphics_resources
                .objects
                .get_set_index(image_index, mesh.object_index);
            let object_set = self.data.graphics_resources.objects.sets[object_set_idx];
            self.rrdevice.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline_layout,
                2,
                &[object_set],
                &[],
            );

            self.rrdevice.device.cmd_draw_indexed(
                command_buffer,
                mesh.index_buffer.indices,
                1,
                0,
                0,
                0,
            );
        }
    }

    pub unsafe fn record_3d_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<()> {
        let extent = self.resource::<SwapchainState>().swapchain.swapchain_extent;

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        self.rrdevice
            .device
            .cmd_set_viewport(command_buffer, 0, &[viewport]);

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(extent);
        self.rrdevice
            .device
            .cmd_set_scissor(command_buffer, 0, &[scissor]);

        let frame_set = self.data.graphics_resources.frame_set.sets[image_index];
        let camera_pos = {
            use crate::ecs::systems::camera_systems::compute_camera_position;
            compute_camera_position(&self.resource::<Camera>())
        };

        let render_data_vec = vec![
            grid_mesh_render_data(&self.resource::<GridMeshData>()),
            gizmo_mesh_render_data(&self.resource::<GridGizmoData>()),
            gizmo_selectable_render_data(&self.resource::<LightGizmoData>(), camera_pos),
        ];
        let render_data_refs: Vec<_> = render_data_vec.iter().collect();

        render_scene_objects(
            &render_data_refs,
            command_buffer,
            image_index,
            frame_set,
            &self.data.graphics_resources.objects,
            &self.rrdevice,
            self.pipeline_storage(),
            &self.data.buffer_registry,
        );

        self.render_billboard(command_buffer, image_index);

        self.render_models(command_buffer, image_index);

        Ok(())
    }

    unsafe fn render_billboard(&self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let billboard = self.resource::<BillboardData>();
        let pipeline_storage = self.pipeline_storage();

        let vertex_buffer = match self
            .data
            .buffer_registry
            .get_vertex_buffer(billboard.mesh.current_vertex_buffer_handle())
        {
            Some(b) => b,
            None => return,
        };
        let index_buffer = match self
            .data
            .buffer_registry
            .get_index_buffer(billboard.mesh.current_index_buffer_handle())
        {
            Some(b) => b,
            None => return,
        };

        let pipeline_id = match billboard.render_info.pipeline_id {
            Some(id) => id,
            None => return,
        };
        let pipeline = match pipeline_storage.get(pipeline_id) {
            Some(p) => p,
            None => return,
        };

        self.rrdevice.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline,
        );

        self.rrdevice
            .device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);

        self.rrdevice.device.cmd_bind_index_buffer(
            command_buffer,
            index_buffer,
            0,
            vk::IndexType::UINT32,
        );

        self.rrdevice.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            0,
            &[billboard.render_state.descriptor_set.descriptor_sets[image_index]],
            &[],
        );

        self.rrdevice.device.cmd_draw_indexed(
            command_buffer,
            billboard.mesh.indices.len() as u32,
            1,
            0,
            0,
            0,
        );
    }

    pub unsafe fn record_imgui_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        draw_data: &imgui::DrawData,
    ) -> Result<()> {
        if draw_data.total_vtx_count == 0 || draw_data.total_idx_count == 0 {
            return Ok(());
        }

        let frame_slot = self.resource::<FrameSync>().current_frame;

        let pipeline = self
            .data
            .imgui
            .pipeline
            .ok_or_else(|| anyhow!("ImGui pipeline not initialized"))?;
        let pipeline_layout = self
            .data
            .imgui
            .pipeline_layout
            .ok_or_else(|| anyhow!("ImGui pipeline layout not initialized"))?;
        let descriptor_set = self
            .data
            .imgui
            .descriptor_set
            .ok_or_else(|| anyhow!("ImGui descriptor set not initialized"))?;
        let vertex_buffer = self.data.imgui.vertex_buffers[frame_slot]
            .ok_or_else(|| anyhow!("ImGui vertex buffer not initialized"))?;
        let index_buffer = self.data.imgui.index_buffers[frame_slot]
            .ok_or_else(|| anyhow!("ImGui index buffer not initialized"))?;

        self.setup_imgui_render_state(
            command_buffer,
            draw_data,
            pipeline,
            pipeline_layout,
            descriptor_set,
            vertex_buffer,
            index_buffer,
        );

        self.record_imgui_draw_commands(command_buffer, draw_data, pipeline_layout, descriptor_set);

        Ok(())
    }

    unsafe fn setup_imgui_render_state(
        &self,
        command_buffer: vk::CommandBuffer,
        draw_data: &imgui::DrawData,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
    ) {
        self.rrdevice.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline,
        );

        self.rrdevice.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            0,
            &[descriptor_set],
            &[],
        );

        self.rrdevice
            .device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);
        self.rrdevice.device.cmd_bind_index_buffer(
            command_buffer,
            index_buffer,
            0,
            vk::IndexType::UINT16,
        );

        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(fb_width)
            .height(fb_height)
            .min_depth(0.0)
            .max_depth(1.0);
        self.rrdevice
            .device
            .cmd_set_viewport(command_buffer, 0, &[viewport]);

        let scale = [
            2.0 / draw_data.display_size[0],
            2.0 / draw_data.display_size[1],
        ];
        let translate = [
            -1.0 - draw_data.display_pos[0] * scale[0],
            -1.0 - draw_data.display_pos[1] * scale[1],
        ];
        let push_constants = [scale[0], scale[1], translate[0], translate[1]];

        self.rrdevice.device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            std::slice::from_raw_parts(
                push_constants.as_ptr() as *const u8,
                std::mem::size_of_val(&push_constants),
            ),
        );
    }

    unsafe fn record_imgui_draw_commands(
        &self,
        command_buffer: vk::CommandBuffer,
        draw_data: &imgui::DrawData,
        pipeline_layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
    ) {
        let font_texture_id = descriptor_set.as_raw() as usize;
        let viewport_texture_id = self.data.viewport.texture_id();
        let viewport_descriptor_set = self.data.viewport.descriptor_set;
        let mut current_texture_id = font_texture_id;

        let mut vertex_offset: u32 = 0;
        let mut index_offset: u32 = 0;

        for draw_list in draw_data.draw_lists() {
            for cmd in draw_list.commands() {
                match cmd {
                    imgui::DrawCmd::Elements { count, cmd_params } => {
                        let texture_id = cmd_params.texture_id.id();

                        if texture_id != current_texture_id {
                            current_texture_id = texture_id;
                            let new_descriptor_set = if texture_id == viewport_texture_id {
                                viewport_descriptor_set
                            } else {
                                descriptor_set
                            };
                            self.rrdevice.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline_layout,
                                0,
                                &[new_descriptor_set],
                                &[],
                            );
                        }

                        let clip_rect = cmd_params.clip_rect;
                        let scissor = vk::Rect2D::builder()
                            .offset(vk::Offset2D {
                                x: ((clip_rect[0] - draw_data.display_pos[0])
                                    * draw_data.framebuffer_scale[0])
                                    .max(0.0) as i32,
                                y: ((clip_rect[1] - draw_data.display_pos[1])
                                    * draw_data.framebuffer_scale[1])
                                    .max(0.0) as i32,
                            })
                            .extent(vk::Extent2D {
                                width: ((clip_rect[2] - clip_rect[0])
                                    * draw_data.framebuffer_scale[0])
                                    as u32,
                                height: ((clip_rect[3] - clip_rect[1])
                                    * draw_data.framebuffer_scale[1])
                                    as u32,
                            });
                        self.rrdevice
                            .device
                            .cmd_set_scissor(command_buffer, 0, &[scissor]);

                        self.rrdevice.device.cmd_draw_indexed(
                            command_buffer,
                            count as u32,
                            1,
                            index_offset + cmd_params.idx_offset as u32,
                            (vertex_offset + cmd_params.vtx_offset as u32) as i32,
                            0,
                        );
                    }
                    _ => {}
                }
            }

            vertex_offset += draw_list.vtx_buffer().len() as u32;
            index_offset += draw_list.idx_buffer().len() as u32;
        }
    }
}

unsafe fn allocate_one_time_command_buffer(
    device: &crate::vulkanr::core::device::Device,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    let cmd_alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffers = device.allocate_command_buffers(&cmd_alloc_info)?;
    Ok(command_buffers[0])
}

unsafe fn record_image_to_buffer_copy(
    device: &crate::vulkanr::core::device::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    buffer: vk::Buffer,
    width: u32,
    height: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    let barrier_to_transfer = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
        .src_access_mask(vk::AccessFlags::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier_to_transfer.build()],
    );

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    device.cmd_copy_image_to_buffer(
        command_buffer,
        image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        buffer,
        &[region.build()],
    );

    let barrier_back = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ);

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier_back.build()],
    );
}

fn append_jsonl(path: &str, line: &str) {
    use std::fs::OpenOptions;
    use std::io::Write;
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(line.as_bytes());
    }
}
