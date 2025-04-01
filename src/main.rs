#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]
mod vulkanr {
    pub mod buffer;
    pub mod command;
    pub mod data;
    pub mod descriptor;
    pub mod device;
    pub mod image;
    pub mod pipeline;
    pub mod render;
    pub mod swapchain;
    pub mod vulkan;
    pub mod window;
}
use vulkanr::buffer::*;
use vulkanr::command::*;
use vulkanr::data::*;
use vulkanr::descriptor::*;
use vulkanr::device::*;
use vulkanr::image::*;
use vulkanr::pipeline::*;
use vulkanr::render::*;
use vulkanr::swapchain::*;
use vulkanr::vulkan::*;
use vulkanr::window::*;

// imgui
//use imgui::*;

mod support;

mod math {
    pub mod math;
}
use math::math::*;

mod gltf {
    pub mod gltf;
}
use gltf::gltf::*;

use anyhow::{anyhow, Result};
use core::result::Result::Ok;
use log::*;
const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
use thiserror::Error;
use vulkanalia::bytecode::Bytecode;
const MAX_FRAMES_IN_FLIGHT: usize = 2; // how many frames should be processed concurrently GPU-GPU synchronization
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;
use std::time::Instant;

use cgmath::num_traits::AsPrimitive;
use cgmath::Vector4;
use glium::buffer::Content;
use imgui::{Condition, MouseButton};
use std::borrow::BorrowMut;
use std::path::Path;
use std::rc::Rc;
use vulkanalia::vk::CommandPool;

fn main() -> Result<()> {
    pretty_env_logger::init();
    // imgui
    let system = support::init(file!());
    let mut value = 0;
    let choices = ["test test this is 1", "test test this is 2"];
    let mut gui_data = GUIData::default();

    // App
    let mut app = unsafe { App::create(&system.app_window)? };
    let destroying = false;
    let minimized = false;

    // event_loop.run(move |event, _, control_flow| {
    //     *control_flow = ControlFlow::Poll;
    //     match event {
    //         Event::MainEventsCleared if !destroying && !minimized => {
    //             unsafe { app.render(&window) }.unwrap()
    //         }
    //         Event::WindowEvent {
    //             event: WindowEvent::Resized(size),
    //             ..
    //         } => {
    //             if size.width == 0 || size.height == 0 {
    //                 minimized = true;
    //             } else {
    //                 minimized = false;
    //                 app.resized = true;
    //             }
    //         }
    //         Event::WindowEvent {
    //             event: WindowEvent::CloseRequested,
    //             ..
    //         } => {
    //             destroying = true;
    //             *control_flow = ControlFlow::Exit;
    //             unsafe {
    //                 app.device.device_wait_idle().unwrap();
    //             }
    //             unsafe {
    //                 app.destroy();
    //             }
    //         }
    //         _ => {}
    //     }

    // });

    system.main_loop(move |_, ui| {}, &mut app, &mut gui_data);

    Ok(())
}

impl support::System {
    pub fn main_loop<F: FnMut(&mut bool, &mut Ui) + 'static>(
        self,
        mut run_ui: F,
        app: &mut App,
        gui_data: &mut GUIData,
    ) {
        let support::System {
            event_loop,
            window,
            display,
            mut imgui,
            mut platform,
            mut renderer,
            font_size,
            app_window,
            app_display,
        } = self;
        let mut last_frame = Instant::now();

        event_loop
            .run(move |event, window_target| match event {
                Event::NewEvents(_) => {
                    let now = Instant::now();
                    imgui.io_mut().update_delta_time(now - last_frame);
                    last_frame = now;
                }
                Event::AboutToWait => {
                    platform
                        .prepare_frame(imgui.io_mut(), &window)
                        .expect("Failed to prepare frame");
                    window.request_redraw();
                    platform
                        .prepare_frame(imgui.io_mut(), &app_window)
                        .expect("Failed to prepare frame");
                    app_window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    let ui = imgui.frame();
                    let mouse_pos = ui.io().mouse_pos;
                    let mouse_wheel = ui.io().mouse_wheel;
                    // initialize gui_data
                    gui_data.is_left_clicked = false;
                    gui_data.is_wheel_clicked = false;
                    gui_data.monitor_value = 0.0;

                    if ui.is_mouse_down(MouseButton::Left) {
                        gui_data.is_left_clicked = true;
                    }
                    if ui.is_mouse_down(MouseButton::Middle) {
                        gui_data.is_wheel_clicked = true;
                    }

                    let mut run = true;
                    run_ui(&mut run, ui);
                    if !run {
                        window_target.exit();
                    }

                    unsafe { app.render(&app_window, mouse_pos, mouse_wheel, gui_data) }.unwrap();

                    ui.window("debug window")
                        .size([600.0, 220.0], Condition::FirstUseEver)
                        .build(|| {
                            ui.button("button");
                            if ui.button("reset camera") {
                                unsafe {
                                    app.reset_camera();
                                }
                            }
                            if ui.button("reset camera up") {
                                unsafe {
                                    app.reset_camera_up();
                                }
                            }
                            ui.separator();
                            // let mouse_pos = ui.io().mouse_pos;
                            ui.text(format!(
                                "Mouse Position: ({:.1},{:.1})",
                                mouse_pos[0], mouse_pos[1]
                            ));
                            ui.text(format!(
                                "is left clicked: ({:.1})",
                                gui_data.is_left_clicked
                            ));
                            ui.text(format!(
                                "is wheel clicked: ({:.1})",
                                gui_data.is_wheel_clicked
                            ));
                            ui.text(format!("monitor value: ({:.1})", gui_data.monitor_value));
                        });

                    let mut target = display.draw();
                    target.clear_color_srgb(0.0, 0.0, 0.5, 1.0);
                    platform.prepare_render(ui, &app_window);
                    let draw_data = imgui.render();
                    renderer
                        .render(&mut target, draw_data)
                        .expect("Rendering failed");
                    target.finish().expect("Failed to swap buffers");
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(new_size),
                    ..
                } => {
                    if new_size.width > 0 && new_size.height > 0 {
                        display.resize((new_size.width, new_size.height));
                    }
                    platform.handle_event(imgui.io_mut(), &window, &event);
                }
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => window_target.exit(),
                event => {
                    platform.handle_event(imgui.io_mut(), &window, &event);
                }
            })
            .expect("EventLoop error");
    }
}

#[derive(Clone, Debug)]
struct GUIData {
    is_left_clicked: bool,
    is_wheel_clicked: bool,
    monitor_value: f32,
}

impl Default for GUIData {
    fn default() -> Self {
        Self {
            is_left_clicked: false,
            is_wheel_clicked: false,
            monitor_value: 0.0,
        }
    }
}

/// Vulkan app
#[derive(Clone, Debug)]
struct App {
    entry: Entry,
    instance: Instance,
    rrdevice: RRDevice,
    data: AppData,
    frame: usize,
    resized: bool,
    start: Instant,
}

#[derive(Clone, Debug, Default)]
struct AppData {
    messenger: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    rrswapchain: RRSwapchain,
    rrrender: RRRender,
    rrcommand_pool: Rc<RRCommandPool>,
    rrcommand_buffer: RRCommandBuffer,
    model_pipeline: RRPipeline,
    model_descriptor_set: RRDescriptorSet,
    model_vertex_buffer: RRVertexBuffer,
    model_index_buffer: RRIndexBuffer,
    grid_pipeline: RRPipeline,
    grid_descriptor_set: RRDescriptorSet,
    grid_vertex_buffer: RRVertexBuffer,
    grid_index_buffer: RRIndexBuffer,
    command_pool: vk::CommandPool,
    image_available_semaphores: Vec<vk::Semaphore>, // semaphores are used to synchronize operations within or across command queues.
    render_finish_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>, // CPU-GPU sync. Fences are mainly designed to synchronize your application itself with rendering operation
    images_in_flight: Vec<vk::Fence>,
    //rrdata_model: RRData,
    //rrdata_grid: RRData,
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    mip_levels: u32,
    msaa_samples: vk::SampleCountFlags,
    color_image: vk::Image, // We only need one render target since only one drawing operation is active at a time
    color_image_memory: vk::DeviceMemory,
    color_image_view: vk::ImageView,
    camera_direction: [f32; 3],
    camera_pos: [f32; 3],
    initial_camera_pos: [f32; 3],
    camera_up: [f32; 3],
    grid_vertices: Vec<Vertex>,
    grid_indices: Vec<u32>,
    is_left_clicked: bool,
    clicked_mouse_pos: [f32; 2],
    is_wheel_clicked: bool,
}

impl App {
    unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = Self::create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        let rrdevice = RRDevice::new(
            &entry,
            &instance,
            &data.surface,
            VALIDATION_ENABLED,
            VALIDATION_LAYER,
            DEVICE_EXTENSIONS,
            PORTABILITY_MACOS_VERSION,
        )?;
        data.rrswapchain = RRSwapchain::new(window, &instance, &data.surface, &rrdevice);
        data.rrcommand_pool = Rc::new(RRCommandPool::new(&instance, &data.surface, &rrdevice));
        data.rrrender = RRRender::new(
            &instance,
            &rrdevice,
            &data.rrswapchain,
            &data.rrcommand_pool.borrow_mut(),
        );
        data.model_descriptor_set = RRDescriptorSet::new(&rrdevice, &data.rrswapchain);
        data.grid_descriptor_set = RRDescriptorSet::new(&rrdevice, &data.rrswapchain);
        data.model_pipeline = RRPipeline::new(
            &rrdevice,
            &data.rrswapchain,
            &data.rrrender,
            &data.model_descriptor_set,
            "src/shaders/vert.spv",
            "src/shaders/frag.spv",
        );
        data.grid_pipeline = RRPipeline::new(
            &rrdevice,
            &data.rrswapchain,
            &data.rrrender,
            &data.grid_descriptor_set,
            "src/shaders/gridVert.spv",
            "src/shaders/gridFrag.spv",
        );
        println!("created pipeline");

        let _ = Self::load_model(&instance, &rrdevice, &mut data)?;
        println!("loaded model");

        let tex_coord = Vec2::new(0.0, 0.0);
        let mut color = Vec4::new(0.0, 0.0, 1.0, 1.0);
        let _ = Self::create_grid_data(&mut data, 0, color, tex_coord)?;
        color = Vec4::new(0.0, 1.0, 0.0, 1.0);
        let _ = Self::create_grid_data(&mut data, 1, color, tex_coord)?;
        color = Vec4::new(1.0, 0.0, 0.0, 1.0);
        let _ = Self::create_grid_data(&mut data, 2, color, tex_coord)?;
        println!("created grid data ");
        // let _ = Self::create_texture_image(&instance, &device, &mut data)?;
        // data.texture_image = RRImage::new(&instance, &rrdevice, &data.rrcommand_pool.borrow_mut());
        data.model_vertex_buffer = RRVertexBuffer::new(
            &instance,
            &rrdevice,
            &data.rrcommand_pool,
            (size_of::<Vertex>() * data.vertices.len()) as vk::DeviceSize,
            data.vertices.as_ptr() as *const c_void,
            data.vertices.len(),
        );
        data.grid_vertex_buffer = RRVertexBuffer::new(
            &instance,
            &rrdevice,
            &data.rrcommand_pool,
            (size_of::<Vertex>() * data.grid_vertices.len()) as vk::DeviceSize,
            data.grid_vertices.as_ptr() as *const c_void,
            data.grid_vertices.len(),
        );
        println!("created vertex buffers model and grid");
        data.model_index_buffer = RRIndexBuffer::new(
            &instance,
            &rrdevice,
            &data.rrcommand_pool,
            (size_of::<u32>() * data.indices.len()) as u64,
            data.indices.as_ptr() as *const c_void,
            data.indices.len(),
        );
        data.grid_index_buffer = RRIndexBuffer::new(
            &instance,
            &rrdevice,
            &data.rrcommand_pool,
            (size_of::<u32>() * data.grid_indices.len()) as u64,
            data.grid_indices.as_ptr() as *const c_void,
            data.grid_indices.len(),
        );
        println!("created index buffers model and grid");

        data.model_descriptor_set.rrdata =
            RRData::create_uniform_buffers(&instance, &rrdevice, &data.rrswapchain);
        data.grid_descriptor_set.rrdata =
            RRData::create_uniform_buffers(&instance, &rrdevice, &data.rrswapchain);
        println!("created uniform buffers");

        data.model_descriptor_set.rrdata.image_view = create_image_view(
            &rrdevice,
            data.texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            data.mip_levels,
        )?;
        data.model_descriptor_set.rrdata.sampler =
            create_texture_sampler(&rrdevice, data.mip_levels)?;
        data.grid_descriptor_set.rrdata.image_view = create_image_view(
            &rrdevice,
            data.texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            data.mip_levels,
        )?;
        data.grid_descriptor_set.rrdata.sampler =
            create_texture_sampler(&rrdevice, data.mip_levels)?;

        if let Err(e) = RRDescriptorSet::create_descriptor_set(
            &rrdevice,
            &data.rrswapchain,
            &mut data.model_descriptor_set,
        ) {
            eprintln!("failed to create model descriptor set: {:?}", e);
        };
        println!("created model descriptor set");
        if let Err(e) = RRDescriptorSet::create_descriptor_set(
            &rrdevice,
            &data.rrswapchain,
            &mut data.grid_descriptor_set,
        ) {
            eprintln!("failed to create grid descriptor set: {:?}", e);
        }
        println!("created grid descriptor set");
        data.rrcommand_buffer = RRCommandBuffer::new(&data.rrcommand_pool);
        if let Err(e) = create_command_buffers(
            &rrdevice,
            &data.rrrender,
            &data.rrswapchain,
            &data.grid_pipeline,
            &data.grid_descriptor_set,
            &data.grid_vertex_buffer,
            &data.grid_index_buffer,
            &data.model_pipeline,
            &data.model_descriptor_set,
            &data.model_vertex_buffer,
            &data.model_index_buffer,
            &mut data.rrcommand_buffer,
        ) {
            eprintln!("failed to create command buffers: {:?}", e);
        }
        println!("created command buffer");

        let _ = Self::create_sync_objects(&rrdevice.device, &mut data)?;
        println!("created sync objects");
        let frame = 0 as usize;
        let resized = false;
        let start = Instant::now();
        data.initial_camera_pos = [0.0, -1.0, -2.0];
        data.camera_pos = data.initial_camera_pos;
        let camera_pos = vec3(data.camera_pos[0], data.camera_pos[1], data.camera_pos[2]);
        let camera_direction = camera_pos.normalize();
        let camera_up = Vector3::cross(camera_direction, vec3(1.0, 0.0, 0.0));
        data.camera_direction = [camera_direction.x, camera_direction.y, camera_direction.z];
        data.camera_up = [camera_up.x, camera_up.y, camera_up.z];
        data.is_left_clicked = false;

        println!("initialized finished");
        Ok(Self {
            entry,
            instance,
            rrdevice,
            data,
            frame,
            resized,
            start,
        })
    }

    unsafe fn render(
        &mut self,
        window: &Window,
        mouse_pos: [f32; 2],
        mouse_wheel: f32,
        gui_data: &mut GUIData,
    ) -> Result<()> {
        // Acquire an image from the swapchain
        // Execute the command buffer with that image as attachment in the framebuffer
        // Return the image to the swapchain for presentation
        self.rrdevice.device.wait_for_fences(
            &[self.data.in_flight_fences[self.frame]],
            true,
            u64::MAX,
        )?; // wait until all fences signaled

        let result = self.rrdevice.device.acquire_next_image_khr(
            self.data.rrswapchain.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            //TODO: Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        // sync CPU(swapchain image)
        if !self.data.images_in_flight[image_index as usize].is_null() {
            self.rrdevice.device.wait_for_fences(
                &[self.data.images_in_flight[image_index as usize]],
                true,
                u64::MAX,
            )?;
        }

        self.data.images_in_flight[image_index as usize] = self.data.in_flight_fences[self.frame];

        self.update_uniform_buffer(image_index, mouse_pos, mouse_wheel, gui_data)?;

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.rrcommand_buffer.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finish_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages) // Each entry in the wait_stages array corresponds to the semaphore with the same index in wait_semaphores.
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.rrdevice
            .device
            .reset_fences(&[self.data.in_flight_fences[self.frame]])?;
        self.rrdevice.device.queue_submit(
            self.rrdevice.graphics_queue,
            &[submit_info],
            self.data.in_flight_fences[self.frame],
        )?;

        let swapchains = &[self.data.rrswapchain.swapchain];
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
            // TODO: self.recreate_swapchain(window)?;
        } else if let Err(e) = present_result {
            return Err(anyhow!(e));
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    unsafe fn destroy(&mut self) {
        // buffer
        // self.rrdevice
        //     .device
        //     .destroy_buffer(self.data.vertex_buffer, None);
        // self.rrdevice
        //     .device
        //     .free_memory(self.data.vertex_buffer_memory, None);
        // self.rrdevice
        //     .device
        //     .destroy_buffer(self.data.index_buffer, None);
        // self.rrdevice
        //     .device
        //     .free_memory(self.data.index_buffer_memory, None);
        // // texture image
        // self.rrdevice
        //     .device
        //     .destroy_image(self.data.texture_image, None);
        // self.rrdevice
        //     .device
        //     .free_memory(self.data.texture_image_memory, None);
        // self.rrdevice
        //     .device
        //     .destroy_image_view(self.data.texture_image_view, None);
        // self.rrdevice
        //     .device
        //     .destroy_sampler(self.data.texture_sampler, None);
        // // semaphore
        // self.data
        //     .image_available_semaphores
        //     .iter()
        //     .for_each(|s| self.rrdevice.device.destroy_semaphore(*s, None));
        // self.data
        //     .render_finish_semaphores
        //     .iter()
        //     .for_each(|s| self.rrdevice.device.destroy_semaphore(*s, None));
        // // fence
        // self.data
        //     .in_flight_fences
        //     .iter()
        //     .for_each(|f| self.rrdevice.device.destroy_fence(*f, None));
        // // relate to swapchain
        // self.destroy_swapchain();
        // // descriptor set layouts
        // self.rrdevice
        //     .device
        //     .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        // // command pool
        // self.rrdevice
        //     .device
        //     .destroy_command_pool(self.data.command_pool, None);
        // // device
        // self.rrdevice.device.destroy_device(None);
        // // surface
        // self.instance.destroy_surface_khr(self.data.surface, None);
        //
        // if VALIDATION_ENABLED {
        //     self.instance
        //         .destroy_debug_utils_messenger_ext(self.data.messenger, None);
        // }
        // self.instance.destroy_instance(None)
    }

    unsafe fn create_instance(
        window: &Window,
        entry: &Entry,
        data: &mut AppData,
    ) -> Result<Instance> {
        let application_info = vk::ApplicationInfo::builder()
            .application_name(b"Vulkan Tutorial\0")
            .application_version(vk::make_version(1, 0, 0))
            .engine_name(b"No Engine\0")
            .engine_version(vk::make_version(1, 0, 0))
            .api_version(vk::make_version(1, 0, 0));

        let mut extensions = vk_window::get_required_instance_extensions(window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

        if VALIDATION_ENABLED {
            extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
        }

        // for Mac ablability
        let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
            info!("Enabling extensions for macOS portability.");
            extensions.push(
                vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION
                    .name
                    .as_ptr(),
            );
            extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

        let available_layers = entry
            .enumerate_instance_layer_properties()?
            .iter()
            .map(|l| l.layer_name)
            .collect::<HashSet<_>>();

        if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
            return Err(anyhow!("Validation layer requested but not supported"));
        }

        let layers = if VALIDATION_ENABLED {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };

        let mut info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .flags(flags);

        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .user_callback(Some(Self::debug_callback));

        if VALIDATION_ENABLED {
            info = info.push_next(&mut debug_info);
        }

        let instance = entry.create_instance(&info, None)?;

        if VALIDATION_ENABLED {
            data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
        }

        Ok(instance)
    }

    extern "system" fn debug_callback(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        type_: vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _: *mut c_void,
    ) -> vk::Bool32 {
        let data = unsafe { *data };
        let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

        if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
            error!("({:?}) {}", type_, message);
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
            warn!("({:?}) {}", type_, message);
        } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
            debug!("({:?}) {}", type_, message);
        } else {
            trace!("({:?}) {}", type_, message);
        }

        vk::FALSE
    }

    unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            data.image_available_semaphores
                .push(device.create_semaphore(&semaphore_info, None)?);
            data.render_finish_semaphores
                .push(device.create_semaphore(&semaphore_info, None)?);
            data.in_flight_fences
                .push(device.create_fence(&fence_info, None)?);
        }

        data.images_in_flight = data
            .rrswapchain
            .swapchain_images
            .iter()
            .map(|_| vk::Fence::null())
            .collect();

        Ok(())
    }

    //unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
    // self.rrdevice.device.device_wait_idle()?;
    // self.destroy_swapchain();
    // Self::create_swapchain(
    //     window,
    //     &self.instance,
    //     &self.rrdevice.device,
    //     &mut self.data,
    // )?;
    // Self::create_swapchain_image_view(&self.rrdevice.device, &mut self.data)?;
    // Self::create_render_pass(&self.instance, &self.rrdevice.device, &mut self.data)?;
    // Self::create_pipeline(&self.rrdevice.device, &mut self.data)?;
    // Self::create_color_objects(&self.instance, &self.rrdevice.device, &mut self.data)?;
    // Self::create_depth_objects(&self.instance, &self.rrdevice.device, &mut self.data)?;
    // Self::create_framebuffers(&self.rrdevice.device, &mut self.data)?;
    // Self::create_uniform_buffers(&self.instance, &self.rrdevice.device, &mut self.data)?;
    // Self::create_uniform_buffers_grid(&self.instance, &self.rrdevice.device, &mut self.data)?;
    // Self::create_descriptor_pool(&self.rrdevice.device, &mut self.data)?;
    // Self::create_descriptor_sets(&self.rrdevice.device, &mut self.data)?;
    // Self::create_descriptor_sets_grid(&self.rrdevice.device, &mut self.data)?;
    // Self::create_command_buffers(&self.rrdevice.device, &mut self.data)?;
    // self.data
    //     .images_in_flight
    //     .resize(self.data.swapchain_images.len(), vk::Fence::null());
    //
    //Ok(())
    // }

    unsafe fn destroy_swapchain(&mut self) {
        // // depth objects
        // self.rrdevice
        //     .device
        //     .destroy_image(self.data.depth_image, None);
        // self.rrdevice
        //     .device
        //     .free_memory(self.data.depth_image_memory, None);
        // self.rrdevice
        //     .device
        //     .destroy_image_view(self.data.depth_image_view, None);
        // // color objects
        // self.rrdevice
        //     .device
        //     .destroy_image(self.data.color_image, None);
        // self.rrdevice
        //     .device
        //     .free_memory(self.data.color_image_memory, None);
        // self.rrdevice
        //     .device
        //     .destroy_image_view(self.data.color_image_view, None);
        // // descriptor pool
        // self.rrdevice
        //     .device
        //     .destroy_descriptor_pool(self.data.descriptor_pool, None);
        // // uniform buffers
        // self.data
        //     .uniform_buffers
        //     .iter()
        //     .for_each(|b| self.rrdevice.device.destroy_buffer(*b, None));
        // self.data
        //     .uniform_buffer_memories
        //     .iter()
        //     .for_each(|m| self.rrdevice.device.free_memory(*m, None));
        // // framebuffers
        // self.data
        //     .framebuffers
        //     .iter()
        //     .for_each(|f| self.rrdevice.device.destroy_framebuffer(*f, None));
        // // command buffers
        // self.rrdevice
        //     .device
        //     .free_command_buffers(self.data.command_pool, &self.data.command_buffers);
        // // The pipeline layout will be referenced throughout the program's lifetime
        // self.rrdevice
        //     .device
        //     .destroy_pipeline_layout(self.data.pipeline_layout, None);
        // // render pass
        // self.rrdevice
        //     .device
        //     .destroy_render_pass(self.data.render_pass, None);
        // // graphics pipeline
        // self.rrdevice
        //     .device
        //     .destroy_pipeline(self.data.pipeline, None);
        // // swapchain imageviews
        // self.data
        //     .swapchain_image_views
        //     .iter()
        //     .for_each(|v| self.rrdevice.device.destroy_image_view(*v, None));
        // // swapchain
        // self.rrdevice
        //     .device
        //     .destroy_swapchain_khr(self.data.swapchain, None);
    }

    unsafe fn create_grid_data(
        data: &mut AppData,
        index: i32,
        color: Vec4,
        tex_coord: Vec2,
    ) -> Result<()> {
        for i in 0..100 {
            let mut pos1 = Vec3::new(0.0, 0.0, 0.0);
            if index == 0 {
                // fix x coordinate
                pos1.x = i as f32 * 0.1;
                pos1.z = 100.0;
            } else if index == 1 {
                //fix x coodinate
                pos1.x = i as f32 * 0.1;
                pos1.y = 100.0f32;
            } else if index == 2 {
                // fix z coordinate
                pos1.z = i as f32 * 0.1;
                pos1.x = 100.0;
            }
            let mut pos2 = Vec3::new(0.0, 0.0, 0.0);
            if index == 0 {
                // fix x coordinate
                pos2.x = pos1.x;
                pos2.z = -100.0;
            } else if index == 1 {
                // fix x coordinate
                pos2.x = pos1.x;
                pos2.y = -100.0;
            } else if index == 2 {
                // fix z coordinate
                pos2.z = pos1.z;
                pos2.x = -100.0;
            }
            let vertex1 = Vertex::new(pos1, color, tex_coord);
            let vertex2 = Vertex::new(pos2, color, tex_coord);
            let vertex3 = Vertex::new(-pos1, color, tex_coord);
            let vertex4 = Vertex::new(-pos2, color, tex_coord);
            data.grid_vertices.push(vertex1);
            data.grid_indices.push(data.grid_indices.len() as u32);
            data.grid_vertices.push(vertex2);
            data.grid_indices.push(data.grid_indices.len() as u32);
            data.grid_vertices.push(vertex3);
            data.grid_indices.push(data.grid_indices.len() as u32);
            data.grid_vertices.push(vertex4);
            data.grid_indices.push(data.grid_indices.len() as u32);
        }

        Ok(())
    }

    // unsafe fn create_descriptor_set_layout_grid(device: &Device, data: &mut AppData) -> Result<()> {
    //     // The descriptor layout specifies the types of resources that are going to be accessed by the pipeline,
    //     // just like a render pass specifies the types of attachments that will be accessed
    //     let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
    //         .binding(0)
    //         .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
    //         .descriptor_count(1)
    //         .stage_flags(vk::ShaderStageFlags::VERTEX);
    //
    //     let bindings = &[ubo_binding];
    //     let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);
    //     data.grid_descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;
    //
    //     Ok(())
    // }
    //
    // unsafe fn create_uniform_buffers_grid(
    //     instance: &Instance,
    //     device: &Device,
    //     data: &mut AppData,
    // ) -> Result<()> {
    //     data.grid_uniform_buffers.clear();
    //     data.grid_uniform_buffer_memories.clear();
    //
    //     for _ in 0..data.swapchain_images.len() {
    //         let (uniform_buffer, uniform_buffer_memory) = Self::create_buffer(
    //             instance,
    //             device,
    //             data,
    //             size_of::<UniformBufferObject>() as u64,
    //             vk::BufferUsageFlags::UNIFORM_BUFFER,
    //             vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    //         )?;
    //         data.grid_uniform_buffers.push(uniform_buffer);
    //         data.grid_uniform_buffer_memories
    //             .push(uniform_buffer_memory);
    //     }
    //
    //     Ok(())
    // }

    unsafe fn update_uniform_buffer(
        &mut self,
        image_index: usize,
        mouse_pos: [f32; 2],
        mouse_wheel: f32,
        gui_data: &mut GUIData,
    ) -> Result<()> {
        //let mut model = Mat4::from_axis_angle(vec3(0.0, 0.0, 1.0), Deg(0.0));
        let model = Mat4::identity();

        let mut camera_pos = vec3_from_array(self.data.camera_pos);
        let mut camera_direction = vec3_from_array(self.data.camera_direction);
        let mut camera_up = vec3_from_array(self.data.camera_up);

        let mouse_pos = Vector2::new(mouse_pos[0], mouse_pos[1]);

        let last_view = view(camera_pos, camera_direction, camera_up);
        let base_x_4 = last_view * vec4(1.0, 0.0, 0.0, 0.0);
        let base_y_4 = last_view * vec4(0.0, -1.0, 0.0, 0.0);
        let base_x = vec3(base_x_4.x, base_x_4.y, base_x_4.z);
        let base_y = vec3(base_y_4.x, base_y_4.y, base_y_4.z);

        if gui_data.is_left_clicked || self.data.is_left_clicked {
            // first clicked
            if !self.data.is_left_clicked {
                self.data.clicked_mouse_pos = [mouse_pos[0], mouse_pos[1]];
                self.data.is_left_clicked = true;
            }
            let clicked_mouse_pos = vec2_from_array(self.data.clicked_mouse_pos);

            let diff = mouse_pos - clicked_mouse_pos;
            let distance = Vector2::distance(mouse_pos, clicked_mouse_pos);
            gui_data.monitor_value = distance;
            if 0.001 < distance {
                let mut rotate_x = Mat3::identity();
                let mut rotate_y = Mat3::identity();
                let theta_x = -diff.x * 0.005;
                let theta_y = -diff.y * 0.005;
                let _ = rodrigues(
                    &mut rotate_x,
                    Rad(theta_x).cos(),
                    Rad(theta_x).sin(),
                    &base_y,
                );
                let _ = rodrigues(
                    &mut rotate_y,
                    Rad(theta_y).cos(),
                    Rad(theta_y).sin(),
                    &base_x,
                );
                let rotate = rotate_y * rotate_x;
                camera_up = rotate * camera_up;
                camera_direction = rotate * camera_direction;

                if !gui_data.is_left_clicked {
                    // left button released
                    self.data.camera_direction = array3_from_vec(camera_direction);
                    self.data.camera_up = array3_from_vec(camera_up);
                    self.data.is_left_clicked = false;
                }
            }
        }

        if gui_data.is_wheel_clicked || self.data.is_wheel_clicked {
            // first clicked
            if !self.data.is_wheel_clicked {
                self.data.clicked_mouse_pos = [mouse_pos[0], mouse_pos[1]];
                self.data.is_wheel_clicked = true;
            }
            let clicked_mouse_pos = vec2_from_array(self.data.clicked_mouse_pos);
            let diff = mouse_pos - clicked_mouse_pos;
            let distance = Vector2::distance(mouse_pos, clicked_mouse_pos);
            gui_data.monitor_value = distance;
            if 0.001 < distance {
                let translate_x_v = base_x * diff.x * 0.01;
                let translate_y_v = base_y * -diff.y * 0.01;
                camera_pos += translate_x_v + translate_y_v;

                if !gui_data.is_wheel_clicked {
                    // left button released
                    self.data.camera_pos = array3_from_vec(camera_pos);
                    self.data.is_wheel_clicked = false;
                }
            }
        }

        if mouse_wheel != 0.0 {
            let diff_view = camera_direction * mouse_wheel * -0.03;
            camera_pos += diff_view;
            self.data.camera_pos = array3_from_vec(camera_pos);
        }

        let view = view(camera_pos, camera_direction, camera_up);

        let correction = Mat4::new(
            // column-major order
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0, // cgmath was originally designed for OpenGL, where the Y coordinate of the clip coordinates is inverted.
            0.0,
            0.0,
            1.0 / 2.0,
            0.0, // depth [-1.0, 1.0] (OpenGL) -> [0.0, 1.0] (Vulkan)
            0.0,
            0.0,
            1.0 / 2.0,
            1.0,
        );
        let proj = correction
            * cgmath::perspective(
                Deg(45.0),
                self.data.rrswapchain.swapchain_extent.width as f32
                    / self.data.rrswapchain.swapchain_extent.height as f32,
                0.1,
                10.0,
            );

        let ubo = UniformBufferObject { model, view, proj };
        let ubo_memory =
            self.data.model_descriptor_set.rrdata.rruniform_buffers[image_index].buffer_memory;
        let memory = self.rrdevice.device.map_memory(
            ubo_memory,
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        memcpy(&ubo, memory.cast(), 1);
        self.rrdevice.device.unmap_memory(ubo_memory);

        // update for grid
        let model_grid = Mat4::identity();
        let grid_ubo_memory =
            self.data.grid_descriptor_set.rrdata.rruniform_buffers[image_index].buffer_memory;
        let ubo_grid = UniformBufferObject {
            model: model_grid,
            view: view,
            proj: proj,
        };
        let memory_grid = self.rrdevice.device.map_memory(
            grid_ubo_memory,
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        memcpy(&ubo_grid, memory_grid.cast(), 1);
        self.rrdevice.device.unmap_memory(
            self.data.grid_descriptor_set.rrdata.rruniform_buffers[image_index].buffer_memory,
        );

        Ok(())
    }

    // unsafe fn create_descriptor_sets_grid(device: &Device, data: &mut AppData) -> Result<()> {
    //     let layouts = vec![data.grid_descriptor_set_layout; data.swapchain_images.len()]; //  create one descriptor set for each swapchain image, all with the same layout.
    //     let info = vk::DescriptorSetAllocateInfo::builder()
    //         .descriptor_pool(data.descriptor_pool)
    //         .set_layouts(&layouts);
    //     data.grid_descriptor_sets = device.allocate_descriptor_sets(&info)?;
    //
    //     for i in 0..data.swapchain_images.len() {
    //         let info = vk::DescriptorBufferInfo::builder()
    //             .buffer(data.grid_uniform_buffers[i])
    //             .offset(0)
    //             .range(size_of::<UniformBufferObject>() as u64);
    //         // The configuration of descriptors is updated using the update_descriptor_sets function,
    //         // which takes an array of vk::WriteDescriptorSet structs as parameter.
    //         let buffer_info = &[info];
    //
    //         let ubo_write = vk::WriteDescriptorSet::builder()
    //             .dst_set(data.grid_descriptor_sets[i])
    //             .dst_binding(0)
    //             .dst_array_element(0) // Remember that descriptors can be arrays, so we also need to specify the first index in the array that we want to update
    //             .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
    //             .buffer_info(buffer_info);
    //
    //         device.update_descriptor_sets(&[ubo_write], &[] as &[vk::CopyDescriptorSet]);
    //     }
    //
    //     Ok(())
    // }

    unsafe fn load_model(
        instance: &Instance,
        rrdevice: &RRDevice,
        data: &mut AppData,
    ) -> Result<()> {
        let mut reader = BufReader::new(File::open("src/resources/VikingRoom/viking_room.obj")?);

        // let (models, _) = tobj::load_obj_buf(
        //     &mut reader,
        //     &tobj::LoadOptions {
        //         triangulate: true,
        //         ..Default::default()
        //     },
        //     |_| Ok(Default::default()),
        // )?;

        // let mut unique_vertices = HashMap::new();

        // for model in models {
        //     for index in &model.mesh.indices {
        //         let pos_offset = (3 * index) as usize;
        //         let tex_coord_offset = (2 * index) as usize;

        //         let vertex = Vertex {
        //             pos: vec3(
        //                 model.mesh.positions[pos_offset],
        //                 model.mesh.positions[pos_offset + 1],
        //                 model.mesh.positions[pos_offset + 2],
        //             ),
        //             color: vec3(1.0, 1.0, 1.0),
        //             tex_coord: vec2(
        //                 model.mesh.texcoords[tex_coord_offset],
        //                 // The OBJ format assumes a coordinate system where a vertical coordinate of 0 means the bottom of the image,
        //                 // however we've uploaded our image into Vulkan in a top to bottom orientation where 0 means the top of the image.
        //                 1.0 - model.mesh.texcoords[tex_coord_offset + 1],
        //             ),
        //         };
        //         if let Some(index) = unique_vertices.get(&vertex) {
        //             data.indices.push(*index as u32);
        //         } else {
        //             let index = data.vertices.len();
        //             unique_vertices.insert(vertex, index);
        //             data.vertices.push(vertex);
        //             data.indices.push(data.indices.len() as u32);
        //         }
        //     }
        // }

        // gltf model
        let grass_path = "src/resources/yard_grass.glb";
        let gltf_data = load_gltf(grass_path)?;
        (
            data.texture_image,
            data.texture_image_memory,
            data.mip_levels,
        ) = create_texture_image_pixel(
            instance,
            rrdevice,
            data.rrcommand_pool.borrow_mut(),
            &gltf_data.image_data[0].data,
            gltf_data.image_data[0].width,
            gltf_data.image_data[0].height,
        )?;

        for i in 0..gltf_data.positions.len() {
            data.indices.push(i as u32);
            let vertex = Vertex::new(
                Vec3::new_array(gltf_data.positions[i]) * 0.01f32,
                Vec4::new(0.0, 1.0, 0.0, 1.0),
                Vec2::new_array(gltf_data.tex_coords[i]),
            );
            data.vertices.push(vertex);
        }

        Ok(())
    }

    unsafe fn reset_camera(&mut self) {
        self.data.camera_pos = self.data.initial_camera_pos;
        let camera_pos = vec3_from_array(self.data.camera_pos);
        let camera_direction = camera_pos.normalize();
        let camera_up = Vector3::cross(camera_direction, vec3(1.0, 0.0, 0.0));
        self.data.camera_direction = array3_from_vec(camera_direction);
        self.data.camera_up = array3_from_vec(camera_up);
    }

    unsafe fn reset_camera_up(&mut self) {
        let camera_pos = vec3_from_array(self.data.camera_pos);
        let mut camera_direction = vec3_from_array(self.data.camera_direction);
        let mut camera_up = vec3_from_array(self.data.camera_up);
        let horizon = Vector3::cross(camera_up, camera_direction);
        camera_up = vec3(0.0, -1.0, 0.0);
        camera_direction = Vector3::cross(horizon, camera_up);
        self.data.camera_up = array3_from_vec(camera_up);
        self.data.camera_direction = array3_from_vec(camera_direction);
    }
}
