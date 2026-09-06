use std::rc::Rc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::model_loader::load_model_from_file_system;
use crate::app::{App, AppData};
use crate::vulkanr::command::RRCommandPool;
use crate::vulkanr::context::{CommandState, SwapchainState};
use crate::vulkanr::device::RRDevice;
use crate::vulkanr::swapchain::RRSwapchain;
use crate::vulkanr::vulkan::Instance;

impl App {
    pub(crate) unsafe fn load_model_from_path_with_resources(
        instance: &Instance,
        rrdevice: &RRDevice,
        data: &mut AppData,
        rrcommand_pool: &Rc<RRCommandPool>,
        rrswapchain: &RRSwapchain,
        model_path: &str,
        scene_will_provide_clips: bool,
    ) -> Result<()> {
        rrdevice.device.device_wait_idle()?;

        if let Some(ref mut gpu) = data.onion_skin_gpu {
            gpu.destroy(rrdevice);
        }
        data.onion_skin_gpu = None;

        load_model_from_file_system(
            model_path,
            instance,
            rrdevice,
            rrcommand_pool,
            rrswapchain,
            &mut data.graphics_resources,
            &mut data.raytracing,
            &mut data.ecs_world,
            &mut data.ecs_assets,
            scene_will_provide_clips,
        )
    }

    pub unsafe fn load_model(&mut self, path: &str) -> Result<()> {
        log!("Loading new model from: {}", path);
        self.rrdevice.device.device_wait_idle()?;

        let water_state = crate::scene::build_water_scene_data(&self.data.ecs_world);
        let flame_state = crate::scene::build_flame_scene_data(&self.data.ecs_world);
        let wind_state = crate::scene::build_wind_scene_data(&self.data.ecs_world);

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

                if let Some(ref water) = water_state {
                    crate::scene::apply_water_state_to_world(
                        &mut self.data.ecs_world,
                        &mut self.data.ecs_assets,
                        water,
                    );
                }
                if let Some(ref flame) = flame_state {
                    crate::scene::apply_flame_state_to_world(
                        &mut self.data.ecs_world,
                        &mut self.data.ecs_assets,
                        flame,
                    );
                }
                if let Some(ref wind) = wind_state {
                    crate::scene::apply_wind_state_to_world(
                        &mut self.data.ecs_world,
                        &mut self.data.ecs_assets,
                        wind,
                    );
                }
                if water_state.is_some() {
                    let command_pool = self.resource::<CommandState>().pool.clone();
                    let waters = crate::ecs::systems::collect_water_instances(&self.data.ecs_world);
                    let mesh_transforms = crate::ecs::systems::collect_mesh_transforms(
                        &self.data.ecs_world,
                        &self.data.ecs_assets,
                    );
                    crate::app::model_loader::rebuild_acceleration_structures(
                        &self.instance,
                        &self.rrdevice,
                        &command_pool,
                        &self.data.graphics_resources,
                        &mut self.data.raytracing,
                        &waters,
                        &mesh_transforms,
                    )?;
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

    pub fn dump_debug_info(&self) {
        log!("========== DUMP DEBUG INFORMATION ==========");

        use crate::ecs::{ClipLibrary, ModelState};
        let clip_library = self.resource::<ClipLibrary>();
        let model_state = self.resource::<ModelState>();

        log!("--- Model Info ---");
        log!("  current_model_path: {}", model_state.model_path);
        log!(
            "  meshes count: {}",
            self.data.graphics_resources.meshes.len()
        );
        log!("  has_skinned_meshes: {}", model_state.has_skinned_meshes);
        log!("  animation clips count: {}", clip_library.clip_count());
        log!(
            "  morph_animations count: {}",
            clip_library.morph_animation.animations.len()
        );
        log!(
            "  skeletons count: {}",
            clip_library.animation.skeletons.len()
        );

        log!("--- GraphicsResources Info ---");
        log!(
            "  meshes count: {}",
            self.data.graphics_resources.meshes.len()
        );
        log!(
            "  materials count: {}",
            self.data.graphics_resources.materials.materials.len()
        );
        log!(
            "  mesh_material_ids: {:?}",
            self.data.graphics_resources.mesh_material_ids
        );

        for (i, mesh) in self.data.graphics_resources.meshes.iter().enumerate() {
            log!(
                "  mesh[{}]: render_to_gbuffer={}, vertex_buffer={:?}, indices={}",
                i,
                mesh.render_to_gbuffer,
                mesh.vertex_buffer.buffer,
                mesh.index_buffer.indices
            );
            log!(
                "    vertex_data.vertices count: {}",
                mesh.vertex_data.vertices.len()
            );
            log!("    object_index: {}", mesh.object_index);

            if !mesh.vertex_data.vertices.is_empty() {
                let v = &mesh.vertex_data.vertices[0];
                log!(
                    "    vertex_data[0].pos: ({:.4}, {:.4}, {:.4})",
                    v.pos.x,
                    v.pos.y,
                    v.pos.z
                );

                let mut min_x = f32::MAX;
                let mut max_x = f32::MIN;
                let mut min_y = f32::MAX;
                let mut max_y = f32::MIN;
                let mut min_z = f32::MAX;
                let mut max_z = f32::MIN;
                for v in &mesh.vertex_data.vertices {
                    min_x = min_x.min(v.pos.x);
                    max_x = max_x.max(v.pos.x);
                    min_y = min_y.min(v.pos.y);
                    max_y = max_y.max(v.pos.y);
                    min_z = min_z.min(v.pos.z);
                    max_z = max_z.max(v.pos.z);
                }
                log!(
                    "    bounds: X[{:.2}, {:.2}], Y[{:.2}, {:.2}], Z[{:.2}, {:.2}]",
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    min_z,
                    max_z
                );
            }
        }

        log!("--- Camera Info ---");
        log!(
            "  pivot: {:?}",
            self.resource::<crate::ecs::resource::Camera>().pivot
        );

        log!("--- Animation Info ---");
        let timeline = self
            .data
            .ecs_world
            .resource::<crate::ecs::resource::TimelineState>();
        log!("  animation_playing: {}", timeline.playing);
        log!("  clips count: {}", clip_library.clip_count());

        log!("========== END DEBUG INFORMATION ==========");
    }
}
