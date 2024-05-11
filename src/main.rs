#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

// imgui
use imgui::*;

mod support;

use anyhow::{anyhow, Result};
use core::result::Result::Ok;
use log::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_2::*;
use vulkanalia::window as vk_window;
use vulkanalia::Version;
const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;
use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
use thiserror::Error;
use vulkanalia::bytecode::Bytecode;
const MAX_FRAMES_IN_FLIGHT: usize = 2; // how many frames should be processed concurrently GPU-GPU synchronization
use cgmath::Rad;
use cgmath::{point3, Deg, InnerSpace, MetricSpace, Vector2};
use cgmath::{prelude::*, Vector3};
use cgmath::{vec2, vec3, vec4};
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;
use std::time::Instant;

type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;
type Vec4 = cgmath::Vector4<f32>;
type Mat3 = cgmath::Matrix3<f32>;
type Mat4 = cgmath::Matrix4<f32>;

use glium::glutin::surface::WindowSurface;
use glium::Surface;
use imgui::{Context, FontConfig, FontGlyphRanges, FontSource, Ui};
use imgui_glium_renderer::Renderer;
use imgui_winit_support::winit::dpi::LogicalSize;
use imgui_winit_support::winit::event::{Event, WindowEvent};
use imgui_winit_support::winit::event_loop::EventLoop;
use imgui_winit_support::winit::window::{Window, WindowBuilder};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::path::Path;

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
    last_translate_x: [f32; 3],
    last_translate_y: [f32; 3],
}

impl Default for GUIData {
    fn default() -> Self {
        Self {
            is_left_clicked: false,
            is_wheel_clicked: false,
            monitor_value: 0.0,
            last_translate_x: [0.0, 0.0, 0.0],
            last_translate_y: [0.0, 0.0, 0.0],
        }
    }
}

/// Vulkan app
#[derive(Clone, Debug)]
struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    frame: usize,
    resized: bool,
    start: Instant,
}

#[derive(Clone, Debug, Default)]
struct AppData {
    messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool, // Command pools manage the memory that is used to store the buffers
    command_buffers: Vec<vk::CommandBuffer>, // have to record a command buffer for every image in the swapchain once again
    image_available_semaphores: Vec<vk::Semaphore>, // semaphores are used to synchronize operations within or across command queues.
    render_finish_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>, // CPU-GPU sync. Fences are mainly designed to synchronize your application itself with rendering operation
    images_in_flight: Vec<vk::Fence>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffer_memories: Vec<vk::DeviceMemory>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    mip_levels: u32,
    msaa_samples: vk::SampleCountFlags,
    color_image: vk::Image, // We only need one render target since only one drawing operation is active at a time
    color_image_memory: vk::DeviceMemory,
    color_image_view: vk::ImageView,
    last_mouse_pos: [f32; 2],
    camera_direction: [f32; 3],
    camera_pos: [f32; 3],
    initial_camera_pos: [f32; 3],
    camera_up: [f32; 3],
    grid_descriptor_set_layout: vk::DescriptorSetLayout,
    grid_pipeline_layout: vk::PipelineLayout,
    grid_pipeline: vk::Pipeline,
    grid_vertex_buffer: vk::Buffer,
    grid_vertex_buffer_memory: vk::DeviceMemory,
    grid_index_buffer: vk::Buffer,
    grid_index_buffer_memory: vk::DeviceMemory,
    grid_vertices: Vec<Vertex>,
    grid_indices: Vec<u32>,
    grid_descriptor_sets: Vec<vk::DescriptorSet>,
    grid_uniform_buffers: Vec<vk::Buffer>,
    grid_uniform_buffer_memories: Vec<vk::DeviceMemory>,
    is_left_clicked: bool,
    clicked_mouse_pos: [f32; 2],
}

impl App {
    unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = Self::create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        let _ = pick_physical_device(&instance, &mut data);
        let device = Self::create_logical_device(&entry, &instance, &mut data)?;
        let _ = Self::create_swapchain(window, &instance, &device, &mut data)?;
        let _ = Self::create_swapchain_image_view(&device, &mut data)?;
        let _ = Self::create_render_pass(&instance, &device, &mut data)?;
        let _ = Self::create_descriptor_set_layout(&device, &mut data)?;
        let _ = Self::create_descriptor_set_layout_grid(&device, &mut data)?;
        let _ = Self::create_pipeline(&device, &mut data)?;
        let _ = Self::create_pipeline_grid(&device, &mut data)?;
        let _ = Self::create_command_pool(&instance, &device, &mut data)?;
        let _ = Self::create_color_objects(&instance, &device, &mut data)?;
        let _ = Self::create_depth_objects(&instance, &device, &mut data)?;
        let _ = Self::create_framebuffers(&device, &mut data)?;
        let _ = Self::create_texture_image(&instance, &device, &mut data)?;
        let _ = Self::create_texture_image_view(&device, &mut data)?;
        let _ = Self::create_texture_sampler(&device, &mut data)?;
        let _ = Self::load_model(&mut data)?;
        let _ = Self::create_vertex_buffer(&instance, &device, &mut data)?;
        let _ = Self::create_index_buffer(&instance, &device, &mut data)?;
        let _ = Self::create_vertex_buffer_grid(&instance, &device, &mut data)?;
        let _ = Self::create_index_buffer_grid(&instance, &device, &mut data)?;
        let _ = Self::create_uniform_buffers(&instance, &device, &mut data)?;
        let _ = Self::create_uniform_buffers_grid(&instance, &device, &mut data)?;
        let _ = Self::create_descriptor_pool(&device, &mut data)?;
        let _ = Self::create_descriptor_sets(&device, &mut data)?;
        let _ = Self::create_descriptor_sets_grid(&device, &mut data)?;
        let _ = Self::create_command_buffers(&device, &mut data)?;
        let _ = Self::create_sync_objects(&device, &mut data)?;
        let frame = 0 as usize;
        let resized = false;
        let start = Instant::now();
        data.initial_camera_pos = [0.0, -1.0, -2.0];
        data.camera_pos = data.initial_camera_pos;
        let camera_pos = vec3(data.camera_pos[0], data.camera_pos[1], data.camera_pos[2]);
        let camera_direction = camera_pos.normalize();
        let camera_up = Vec3::cross(camera_direction, vec3(1.0, 0.0, 0.0));
        data.camera_direction = [camera_direction.x, camera_direction.y, camera_direction.z];
        data.camera_up = [camera_up.x, camera_up.y, camera_up.z];
        data.is_left_clicked = false;

        Ok(Self {
            entry,
            instance,
            data,
            device,
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
        self.device
            .wait_for_fences(&[self.data.in_flight_fences[self.frame]], true, u64::MAX)?; // wait until all fences signaled

        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        // sync CPU(swapchain image)
        if !self.data.images_in_flight[image_index as usize].is_null() {
            self.device.wait_for_fences(
                &[self.data.images_in_flight[image_index as usize]],
                true,
                u64::MAX,
            )?;
        }

        self.data.images_in_flight[image_index as usize] = self.data.in_flight_fences[self.frame];

        self.update_uniform_buffer(image_index, mouse_pos, mouse_wheel, gui_data)?;

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index as usize]];
        let signal_semaphores = &[self.data.render_finish_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages) // Each entry in the wait_stages array corresponds to the semaphore with the same index in wait_semaphores.
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device
            .reset_fences(&[self.data.in_flight_fences[self.frame]])?;
        self.device.queue_submit(
            self.data.graphics_queue,
            &[submit_info],
            self.data.in_flight_fences[self.frame],
        )?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);
        let present_result = self
            .device
            .queue_present_khr(self.data.present_queue, &present_info);
        let changed = present_result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || present_result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        if changed || self.resized {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = present_result {
            return Err(anyhow!(e));
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    unsafe fn destroy(&mut self) {
        // buffer
        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device
            .free_memory(self.data.vertex_buffer_memory, None);
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device.free_memory(self.data.index_buffer_memory, None);
        // texture image
        self.device.destroy_image(self.data.texture_image, None);
        self.device
            .free_memory(self.data.texture_image_memory, None);
        self.device
            .destroy_image_view(self.data.texture_image_view, None);
        self.device.destroy_sampler(self.data.texture_sampler, None);
        // semaphore
        self.data
            .image_available_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data
            .render_finish_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        // fence
        self.data
            .in_flight_fences
            .iter()
            .for_each(|f| self.device.destroy_fence(*f, None));
        // relate to swapchain
        self.destroy_swapchain();
        // descriptor set layouts
        self.device
            .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        // command pool
        self.device
            .destroy_command_pool(self.data.command_pool, None);
        // device
        self.device.destroy_device(None);
        // surface
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }
        self.instance.destroy_instance(None)
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

    unsafe fn create_logical_device(
        entry: &Entry,
        instance: &Instance,
        data: &mut AppData,
    ) -> Result<Device> {
        let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
        let mut unique_indices = HashSet::new();
        unique_indices.insert(indices.graphics);
        unique_indices.insert(indices.present);
        let queue_priorities = &[1.0];
        let queue_infos = unique_indices
            .iter()
            .map(|i| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(*i)
                    .queue_priorities(queue_priorities)
            })
            .collect::<Vec<_>>();

        let layers = if VALIDATION_ENABLED {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            vec![]
        };

        let mut extensions = DEVICE_EXTENSIONS
            .iter()
            .map(|n| n.as_ptr())
            .collect::<Vec<_>>();
        if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
            extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
        }

        let features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .sample_rate_shading(true)
            .fill_mode_non_solid(true);

        let info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .enabled_features(&features);

        let device = instance.create_device(data.physical_device, &info, None)?;

        data.graphics_queue = device.get_device_queue(indices.graphics, 0);
        data.present_queue = device.get_device_queue(indices.present, 0);
        Ok(device)
    }

    unsafe fn create_swapchain(
        window: &Window,
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
        let support = SwapchainSupport::get(instance, data, data.physical_device)?;
        let surface_format = SwapchainSupport::get_swapchain_surface_format(&support.formats);
        let present_mode = SwapchainSupport::get_swapchain_present_mode(&support.present_modes);
        let extent = SwapchainSupport::get_swapchain_extent(window, support.capabilities);

        let mut image_count = support.capabilities.min_image_count + 1;
        if support.capabilities.max_image_count != 0
            && image_count > support.capabilities.max_image_count
        {
            image_count = support.capabilities.max_image_count;
        }

        let mut queue_family_indices = vec![];
        let image_sharing_mode = if indices.graphics != indices.present {
            queue_family_indices.push(indices.graphics);
            queue_family_indices.push(indices.present);
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(data.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        data.swapchain = device.create_swapchain_khr(&info, None)?;
        data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;
        data.swapchain_format = surface_format.format;
        data.swapchain_extent = extent;
        Ok(())
    }

    unsafe fn create_swapchain_image_view(device: &Device, data: &mut AppData) -> Result<()> {
        data.swapchain_image_views = data
            .swapchain_images
            .iter()
            .map(|i| {
                Self::create_image_view(
                    device,
                    *i,
                    data.swapchain_format,
                    vk::ImageAspectFlags::COLOR,
                    1,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }

    unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
        let vert = include_bytes!("./shaders/vert.spv");
        let frag = include_bytes!("./shaders/frag.spv");
        let vert_shader_module = Self::create_shader_module(device, &vert[..])?;
        let frag_shader_module = Self::create_shader_module(device, &frag[..])?;

        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(b"main\0");
        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(b"main\0");

        let binding_discriptions = &[Vertex::binding_description()];
        let attribute_descriptions = Vertex::attribute_descriptions();
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(binding_discriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(data.swapchain_extent.width as f32)
            .height(data.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(data.swapchain_extent);

        let viewports = &[viewport];
        let scissors = &[scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(true) // https://registry.khronos.org/vulkan/specs/1.0/html/vkspec.html#primsrast-sampleshading
            .min_sample_shading(0.9) //  Minimum fraction for sample shading; closer to one is smoother.
            .sample_shading_enable(false)
            .rasterization_samples(data.msaa_samples);

        let attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let attachments = &[attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        // NOTE: This will cause the configuration of these values to be ignored and you will be required to specify the data at drawing time.
        let dynamic_state = &[vk::DynamicState::VIEWPORT, vk::DynamicState::LINE_WIDTH];

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(dynamic_state);

        let descriptor_set_layouts = &[data.descriptor_set_layout];
        let layout_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(descriptor_set_layouts);
        data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

        let stages = &[vert_stage, frag_stage];

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);

        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .layout(data.pipeline_layout)
            .render_pass(data.render_pass)
            .subpass(0);

        data.pipeline = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
            .0[0];

        device.destroy_shader_module(vert_shader_module, None);
        device.destroy_shader_module(frag_shader_module, None);
        Ok(())
    }

    unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
        let bytecode = Bytecode::new(bytecode).unwrap();
        let info = vk::ShaderModuleCreateInfo::builder()
            .code_size(bytecode.code_size())
            .code(bytecode.code());

        Ok(device.create_shader_module(&info, None)?)
    }

    unsafe fn create_pipeline_grid(device: &Device, data: &mut AppData) -> Result<()> {
        let vert = include_bytes!("./shaders/gridVert.spv");
        let frag = include_bytes!("./shaders/gridFrag.spv");
        let vert_shader_module = Self::create_shader_module(device, &vert[..])?;
        let frag_shader_module = Self::create_shader_module(device, &frag[..])?;

        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(b"main\0");
        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(b"main\0");

        let binding_discriptions = &[Vertex::binding_description()];
        let attribute_descriptions = Vertex::attribute_descriptions();
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(binding_discriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::LINE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(data.swapchain_extent.width as f32)
            .height(data.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(data.swapchain_extent);

        let viewports = &[viewport];
        let scissors = &[scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::LINE)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(true) // https://registry.khronos.org/vulkan/specs/1.0/html/vkspec.html#primsrast-sampleshading
            .min_sample_shading(0.9) //  Minimum fraction for sample shading; closer to one is smoother.
            .sample_shading_enable(false)
            .rasterization_samples(data.msaa_samples);

        let attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let attachments = &[attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        // NOTE: This will cause the configuration of these values to be ignored and you will be required to specify the data at drawing time.
        let dynamic_state = &[vk::DynamicState::VIEWPORT, vk::DynamicState::LINE_WIDTH];

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(dynamic_state);

        let descriptor_set_layouts = &[data.grid_descriptor_set_layout];
        let layout_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(descriptor_set_layouts);
        data.grid_pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

        let stages = &[vert_stage, frag_stage];

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);

        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .layout(data.grid_pipeline_layout)
            .render_pass(data.render_pass)
            .subpass(0);

        data.grid_pipeline = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
            .0[0];

        device.destroy_shader_module(vert_shader_module, None);
        device.destroy_shader_module(frag_shader_module, None);
        Ok(())
    }

    unsafe fn create_render_pass(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        // we need to tell Vulkan about the framebuffer attachments that will be used while rendering.
        // We need to specify how many color and depth buffers there will be, how many samples to use for each of them and how their contents should be handled throughout the rendering operations.
        // All of this information is wrapped in a render pass object
        let color_attachment = vk::AttachmentDescription::builder()
            .format(data.swapchain_format)
            .samples(data.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE) // for stencil buffer
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE) // for stencil buffer
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        // That's because multisampled images cannot be presented directly.
        // We first need to resolve them to a regular image.
        let color_resolve_attachment = vk::AttachmentDescription::builder()
            .format(data.swapchain_format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        //  Subpasses are subsequent rendering operations that depend on the contents of framebuffers in previous passes
        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_resolve_attachement_ref = vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        // The index of the attachment in this array is directly referenced from the fragment shader with the layout(location = 0) out vec4 outColor directive!
        let color_attachments = &[color_attachment_ref];
        let resolve_attachments = &[color_resolve_attachement_ref];

        let depth_stencil_attachment = vk::AttachmentDescription::builder()
            .format(Self::get_depth_format(instance, data)?)
            .samples(data.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(color_attachments)
            .depth_stencil_attachment(&depth_stencil_attachment_ref)
            .resolve_attachments(resolve_attachments);

        // The subpasses in a render pass automatically take care of image layout transitions.
        // These transitions are controlled by subpass dependencies, which specify memory and execution dependencies between subpasses
        // The depth image is first accessed in the early fragment test pipeline stage
        // and because we have a load operation that clears, we should specify the access mask for writes.
        let dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            ) //  wait for the swapchain to finish reading from the image before we can access it
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

        let attachments = &[
            color_attachment,
            depth_stencil_attachment,
            color_resolve_attachment,
        ];
        let subpasses = &[subpass];
        let dependencies = &[dependency];
        let info = vk::RenderPassCreateInfo::builder()
            .attachments(attachments)
            .subpasses(subpasses)
            .dependencies(dependencies);

        data.render_pass = device.create_render_pass(&info, None)?;
        Ok(())
    }

    unsafe fn create_framebuffers(device: &Device, data: &mut AppData) -> Result<()> {
        data.framebuffers = data
            .swapchain_image_views
            .iter()
            .map(|i| {
                let attachments = &[data.color_image_view, data.depth_image_view, *i];
                let create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(data.render_pass) // they use the same number and type of attachments.
                    .attachments(attachments)
                    .width(data.swapchain_extent.width)
                    .height(data.swapchain_extent.height)
                    .layers(1);
                device.create_framebuffer(&create_info, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }

    unsafe fn create_command_pool(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
        let info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::empty())
            .queue_family_index(indices.graphics); // Each command pool can only allocate command buffers that are submitted on a single type of queue.

        data.command_pool = device.create_command_pool(&info, None)?;

        Ok(())
    }

    unsafe fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(data.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(data.framebuffers.len() as u32);

        data.command_buffers = device.allocate_command_buffers(&info)?;

        for (i, command_buffer) in data.command_buffers.iter().enumerate() {
            let inheritance = vk::CommandBufferInheritanceInfo::builder(); //  only relevant for secondary command buffers.
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::empty())
                .inheritance_info(&inheritance);

            device.begin_command_buffer(*command_buffer, &begin_info)?;

            let render_area = vk::Rect2D::builder()
                .offset(vk::Offset2D::default())
                .extent(data.swapchain_extent);
            let color_clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            };
            let depth_clear_value = vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0, // The range of depths in the depth buffer is 0.0 to 1.0 in Vulkan, where 1.0 lies at the far view plane and 0.0 at the near view plane
                    stencil: 0,
                },
            };

            let clear_values = &[color_clear_value, depth_clear_value];
            let info = vk::RenderPassBeginInfo::builder()
                .render_pass(data.render_pass)
                .framebuffer(data.framebuffers[i])
                .render_area(render_area)
                .clear_values(clear_values);
            device.cmd_begin_render_pass(*command_buffer, &info, vk::SubpassContents::INLINE);

            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                data.pipeline,
            );
            device.cmd_bind_vertex_buffers(*command_buffer, 0, &[data.vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(
                *command_buffer,
                data.index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                data.pipeline_layout,
                0,
                &[data.descriptor_sets[i]],
                &[],
            );
            device.cmd_draw_indexed(*command_buffer, data.indices.len() as u32, 1, 0, 0, 0);

            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                data.grid_pipeline,
            );
            device.cmd_bind_vertex_buffers(*command_buffer, 0, &[data.grid_vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(
                *command_buffer,
                data.grid_index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                data.grid_pipeline_layout,
                0,
                &[data.grid_descriptor_sets[i]],
                &[],
            );
            device.cmd_draw_indexed(*command_buffer, data.grid_indices.len() as u32, 1, 0, 0, 0);
            device.cmd_end_render_pass(*command_buffer);
            device.end_command_buffer(*command_buffer)?;
        }

        Ok(())
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
            .swapchain_images
            .iter()
            .map(|_| vk::Fence::null())
            .collect();

        Ok(())
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();
        Self::create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        Self::create_swapchain_image_view(&self.device, &mut self.data)?;
        Self::create_render_pass(&self.instance, &self.device, &mut self.data)?;
        Self::create_pipeline(&self.device, &mut self.data)?;
        Self::create_color_objects(&self.instance, &self.device, &mut self.data)?;
        Self::create_depth_objects(&self.instance, &self.device, &mut self.data)?;
        Self::create_framebuffers(&self.device, &mut self.data)?;
        Self::create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        Self::create_uniform_buffers_grid(&self.instance, &self.device, &mut self.data)?;
        Self::create_descriptor_pool(&self.device, &mut self.data)?;
        Self::create_descriptor_sets(&self.device, &mut self.data)?;
        Self::create_descriptor_sets_grid(&self.device, &mut self.data)?;
        Self::create_command_buffers(&self.device, &mut self.data)?;
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
        // depth objects
        self.device.destroy_image(self.data.depth_image, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        self.device
            .destroy_image_view(self.data.depth_image_view, None);
        // color objects
        self.device.destroy_image(self.data.color_image, None);
        self.device.free_memory(self.data.color_image_memory, None);
        self.device
            .destroy_image_view(self.data.color_image_view, None);
        // descriptor pool
        self.device
            .destroy_descriptor_pool(self.data.descriptor_pool, None);
        // uniform buffers
        self.data
            .uniform_buffers
            .iter()
            .for_each(|b| self.device.destroy_buffer(*b, None));
        self.data
            .uniform_buffer_memories
            .iter()
            .for_each(|m| self.device.free_memory(*m, None));
        // framebuffers
        self.data
            .framebuffers
            .iter()
            .for_each(|f| self.device.destroy_framebuffer(*f, None));
        // command buffers
        self.device
            .free_command_buffers(self.data.command_pool, &self.data.command_buffers);
        // The pipeline layout will be referenced throughout the program's lifetime
        self.device
            .destroy_pipeline_layout(self.data.pipeline_layout, None);
        // render pass
        self.device.destroy_render_pass(self.data.render_pass, None);
        // graphics pipeline
        self.device.destroy_pipeline(self.data.pipeline, None);
        // swapchain imageviews
        self.data
            .swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));
        // swapchain
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }

    unsafe fn create_vertex_buffer(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        // NOTE :  Driver developers recommend that you also store multiple buffers, like the vertex and index buffer, into a single vk::Buffer and use offsets in commands like cmd_bind_vertex_buffers.
        // The advantage is that your data is more cache friendly in that case,
        // because it's closer together. It is even possible to reuse the same chunk of memory for multiple resources
        // if they are not used during the same render operations, provided that their data is refreshed, of course.
        // This is known as aliasing
        let size = (size_of::<Vertex>() * data.vertices.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let map_memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

        memcpy(
            data.vertices.as_ptr(),
            map_memory.cast(),
            data.vertices.len(),
        );
        device.unmap_memory(staging_buffer_memory);

        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, //  we're not able to use map_memory, instead can be copied
        )?;

        data.vertex_buffer = vertex_buffer;
        data.vertex_buffer_memory = vertex_buffer_memory;

        Self::copy_buffer(device, data, staging_buffer, vertex_buffer, size)?;

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        Ok(())
    }

    unsafe fn create_vertex_buffer_grid(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        // create data
        // -100.0 - 100.0
        let tex_coord = vec2(0.0, 0.0);
        let mut color = vec3(0.0, 0.0, 1.0);
        let _ = Self::create_grid_data(data, 0, color, tex_coord)?;
        color = vec3(0.0, 1.0, 0.0);
        let _ = Self::create_grid_data(data, 1, color, tex_coord)?;
        color = vec3(1.0, 0.0, 0.0);
        let _ = Self::create_grid_data(data, 2, color, tex_coord)?;

        let size = (size_of::<Vertex>() * data.grid_vertices.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let map_memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

        memcpy(
            data.grid_vertices.as_ptr(),
            map_memory.cast(),
            data.grid_vertices.len(),
        );
        device.unmap_memory(staging_buffer_memory);

        let (grid_vertex_buffer, grid_vertex_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, //  we're not able to use map_memory, instead can be copied
        )?;

        data.grid_vertex_buffer = grid_vertex_buffer;
        data.grid_vertex_buffer_memory = grid_vertex_buffer_memory;

        Self::copy_buffer(device, data, staging_buffer, grid_vertex_buffer, size)?;

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        Ok(())
    }

    unsafe fn create_grid_data(
        data: &mut AppData,
        index: i32,
        color: Vector3<f32>,
        tex_coord: Vector2<f32>,
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
                pos1.y = 100.0;
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

    unsafe fn copy_buffer(
        device: &Device,
        data: &AppData,
        source: vk::Buffer,
        destination: vk::Buffer,
        size: vk::DeviceSize,
    ) -> Result<()> {
        let command_buffer = Self::begin_single_time_commands(device, data)?;
        let regions = vk::BufferCopy::builder().size(size);
        device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);
        Self::end_single_time_commands(device, data, command_buffer)?;

        Ok(())
    }

    unsafe fn create_index_buffer(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        let size = (size_of::<u32>() * data.indices.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let map_memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

        memcpy(data.indices.as_ptr(), map_memory.cast(), data.indices.len());
        device.unmap_memory(staging_buffer_memory);

        let (index_buffer, index_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, //  we're not able to use map_memory, instead can be copied
        )?;

        data.index_buffer = index_buffer;
        data.index_buffer_memory = index_buffer_memory;

        Self::copy_buffer(device, data, staging_buffer, index_buffer, size)?;

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        Ok(())
    }

    unsafe fn create_index_buffer_grid(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        let size = (size_of::<u32>() * data.grid_indices.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let map_memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

        memcpy(
            data.indices.as_ptr(),
            map_memory.cast(),
            data.grid_indices.len(),
        );
        device.unmap_memory(staging_buffer_memory);

        let (index_buffer, index_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, //  we're not able to use map_memory, instead can be copied
        )?;

        data.grid_index_buffer = index_buffer;
        data.grid_index_buffer_memory = index_buffer_memory;

        Self::copy_buffer(device, data, staging_buffer, index_buffer, size)?;

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        Ok(())
    }

    unsafe fn get_memory_type_index(
        instance: &Instance,
        data: &AppData,
        properties: vk::MemoryPropertyFlags,
        requirements: vk::MemoryRequirements,
    ) -> Result<u32> {
        let memory = instance.get_physical_device_memory_properties(data.physical_device);
        (0..memory.memory_type_count)
            .find(|i| {
                let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
                let memory_type = memory.memory_types[*i as usize];
                suitable & memory_type.property_flags.contains(properties)
            })
            .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
    }

    unsafe fn create_buffer(
        instance: &Instance,
        device: &Device,
        data: &AppData,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device.create_buffer(&buffer_info, None)?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(Self::get_memory_type_index(
                instance,
                data,
                properties,
                requirements,
            )?);
        let buffer_memory = device.allocate_memory(&memory_info, None)?;
        device.bind_buffer_memory(buffer, buffer_memory, 0)?;

        Ok((buffer, buffer_memory))
    }

    unsafe fn create_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
        // The descriptor layout specifies the types of resources that are going to be accessed by the pipeline,
        // just like a render pass specifies the types of attachments that will be accessed
        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = &[ubo_binding, sampler_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);
        data.descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

        Ok(())
    }

    unsafe fn create_descriptor_set_layout_grid(device: &Device, data: &mut AppData) -> Result<()> {
        // The descriptor layout specifies the types of resources that are going to be accessed by the pipeline,
        // just like a render pass specifies the types of attachments that will be accessed
        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let bindings = &[ubo_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);
        data.grid_descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

        Ok(())
    }

    unsafe fn create_uniform_buffers(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        data.uniform_buffers.clear();
        data.uniform_buffer_memories.clear();

        for _ in 0..data.swapchain_images.len() {
            let (uniform_buffer, uniform_buffer_memory) = Self::create_buffer(
                instance,
                device,
                data,
                size_of::<UniformBufferObject>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            )?;
            data.uniform_buffers.push(uniform_buffer);
            data.uniform_buffer_memories.push(uniform_buffer_memory);
        }

        Ok(())
    }

    unsafe fn create_uniform_buffers_grid(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        data.grid_uniform_buffers.clear();
        data.grid_uniform_buffer_memories.clear();

        for _ in 0..data.swapchain_images.len() {
            let (uniform_buffer, uniform_buffer_memory) = Self::create_buffer(
                instance,
                device,
                data,
                size_of::<UniformBufferObject>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            )?;
            data.grid_uniform_buffers.push(uniform_buffer);
            data.grid_uniform_buffer_memories
                .push(uniform_buffer_memory);
        }

        Ok(())
    }

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

        let last_mouse_pos = Vec2::new(self.data.last_mouse_pos[0], self.data.last_mouse_pos[1]);
        let mouse_pos = Vec2::new(mouse_pos[0], mouse_pos[1]);

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
            let distance = Vec2::distance(mouse_pos, clicked_mouse_pos);
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

        if gui_data.is_wheel_clicked {
            let diff = mouse_pos - last_mouse_pos;
            let distance = Vec2::distance(mouse_pos, last_mouse_pos);
            gui_data.monitor_value = distance;
            if 0.001 < distance && distance < 100.0 {
                let translate_x_v = base_x * diff.x * 0.01;
                let translate_y_v = base_y * -diff.y * 0.01;
                camera_pos += translate_x_v + translate_y_v;

                gui_data.last_translate_x =
                    array3_from_vec(vec3_from_array(gui_data.last_translate_x) + translate_x_v);
                gui_data.last_translate_y =
                    array3_from_vec(vec3_from_array(gui_data.last_translate_y) + translate_y_v);
            }
        }

        let diff_view = camera_direction * mouse_wheel * -0.03;
        camera_pos += diff_view;

        let view = view(camera_pos, camera_direction, camera_up);

        self.data.camera_pos = [camera_pos.x, camera_pos.y, camera_pos.z];
        self.data.last_mouse_pos = [mouse_pos.x, mouse_pos.y];

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
                self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
                0.1,
                10.0,
            );

        let ubo = UniformBufferObject { model, view, proj };
        let memory = self.device.map_memory(
            self.data.uniform_buffer_memories[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        memcpy(&ubo, memory.cast(), 1);
        self.device
            .unmap_memory(self.data.uniform_buffer_memories[image_index]);

        // update for grid
        let model_grid = Mat4::identity();
        let ubo_grid = UniformBufferObject {
            model: model_grid,
            view: view,
            proj: proj,
        };
        let memory_grid = self.device.map_memory(
            self.data.grid_uniform_buffer_memories[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        memcpy(&ubo_grid, memory_grid.cast(), 1);
        self.device
            .unmap_memory(self.data.grid_uniform_buffer_memories[image_index]);

        Ok(())
    }

    unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
        let ubo_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count((data.swapchain_images.len() * 4) as u32); // This pool size structure is referenced by the main vk::DescriptorPoolCreateInfo
                                                                         //along with the maximum number of descriptor sets that may be allocated:

        let sampler_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count((data.swapchain_images.len() * 4) as u32);

        let pool_sizes = &[ubo_size, sampler_size];
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets((data.swapchain_images.len() * 4) as u32);
        data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

        Ok(())
    }

    unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
        /*
        A descriptor is a way for shaders to freely access resources like buffers and images
        Usage of descriptors consists of three parts:

        Specify a descriptor layout during pipeline creation
        Allocate a descriptor set from a descriptor pool
        Bind the descriptor set during rendering
         */
        let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()]; //  create one descriptor set for each swapchain image, all with the same layout.
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(data.descriptor_pool)
            .set_layouts(&layouts);
        data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

        for i in 0..data.swapchain_images.len() {
            let info = vk::DescriptorBufferInfo::builder()
                .buffer(data.uniform_buffers[i])
                .offset(0)
                .range(size_of::<UniformBufferObject>() as u64);
            // The configuration of descriptors is updated using the update_descriptor_sets function,
            // which takes an array of vk::WriteDescriptorSet structs as parameter.
            let buffer_info = &[info];

            let info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(data.texture_image_view)
                .sampler(data.texture_sampler);
            let image_info = &[info];

            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(data.descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0) // Remember that descriptors can be arrays, so we also need to specify the first index in the array that we want to update
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info);
            let sampler_write = vk::WriteDescriptorSet::builder()
                .dst_set(data.descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(image_info);

            device.update_descriptor_sets(
                &[ubo_write, sampler_write],
                &[] as &[vk::CopyDescriptorSet],
            );
        }

        Ok(())
    }

    unsafe fn create_descriptor_sets_grid(device: &Device, data: &mut AppData) -> Result<()> {
        let layouts = vec![data.grid_descriptor_set_layout; data.swapchain_images.len()]; //  create one descriptor set for each swapchain image, all with the same layout.
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(data.descriptor_pool)
            .set_layouts(&layouts);
        data.grid_descriptor_sets = device.allocate_descriptor_sets(&info)?;

        for i in 0..data.swapchain_images.len() {
            let info = vk::DescriptorBufferInfo::builder()
                .buffer(data.grid_uniform_buffers[i])
                .offset(0)
                .range(size_of::<UniformBufferObject>() as u64);
            // The configuration of descriptors is updated using the update_descriptor_sets function,
            // which takes an array of vk::WriteDescriptorSet structs as parameter.
            let buffer_info = &[info];

            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(data.grid_descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0) // Remember that descriptors can be arrays, so we also need to specify the first index in the array that we want to update
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info);

            device.update_descriptor_sets(&[ubo_write], &[] as &[vk::CopyDescriptorSet]);
        }

        Ok(())
    }

    unsafe fn create_texture_image(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        /*TODO :
         Try to experiment with this by creating a setup_command_buffer that the helper functions record commands into,
         and add a flush_setup_commands to execute the commands that have been recorded so far.
         It's best to do this after the texture mapping works to check if the texture resources are still set up correctly.
        */
        let image = File::open("src/resources/VikingRoom/viking_room.png")?;
        let decoder = png::Decoder::new(image);
        let mut reader = decoder.read_info()?;
        let mut pixels = vec![0; reader.info().raw_bytes()];
        reader.next_frame(&mut pixels)?;
        let size = reader.info().raw_bytes() as u64;
        let (width, height) = reader.info().size();
        data.mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

        if width != 1024 || height != 1024 || reader.info().color_type != png::ColorType::Rgba {
            panic!("invalid texture image");
        }

        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
        memcpy(pixels.as_ptr(), memory.cast(), pixels.len());
        device.unmap_memory(staging_buffer_memory);

        let (texture_image, texture_image_memory) = Self::create_image(
            instance,
            device,
            data,
            width,
            height,
            data.mip_levels,
            vk::SampleCountFlags::_1,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        data.texture_image = texture_image;
        data.texture_image_memory = texture_image_memory;

        Self::transition_image_layout(
            device,
            data,
            data.texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            data.mip_levels,
        )?;
        Self::copy_buffer_to_image(
            device,
            data,
            staging_buffer,
            data.texture_image,
            width,
            height,
        )?;

        Self::generate_mipmaps(
            instance,
            device,
            data,
            data.texture_image,
            vk::Format::R8G8B8A8_SRGB,
            width,
            height,
            data.mip_levels,
        )?;

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        Ok(())
    }

    unsafe fn create_image(
        instance: &Instance,
        device: &Device,
        data: &AppData,
        width: u32,
        height: u32,
        mip_levels: u32,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory)> {
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED) // Not usable by the GPU and the very first transition will discard the texels.
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(samples)
            .flags(vk::ImageCreateFlags::empty());

        let image = device.create_image(&info, None)?;
        let requirements = device.get_image_memory_requirements(image);
        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(Self::get_memory_type_index(
                instance,
                data,
                properties,
                requirements,
            )?);
        let image_memory = device.allocate_memory(&info, None)?;
        device.bind_image_memory(image, image_memory, 0)?;

        Ok((image, image_memory))
    }

    unsafe fn begin_single_time_commands(
        device: &Device,
        data: &AppData,
    ) -> Result<vk::CommandBuffer> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(data.command_pool)
            .command_buffer_count(1);
        let command_buffer = device.allocate_command_buffers(&info)?[0];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(command_buffer, &info)?;

        Ok(command_buffer)
    }

    unsafe fn end_single_time_commands(
        device: &Device,
        data: &AppData,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        device.end_command_buffer(command_buffer)?;

        let command_buffers = &[command_buffer];
        let info = vk::SubmitInfo::builder().command_buffers(command_buffers);
        device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
        device.queue_wait_idle(data.graphics_queue)?;

        device.free_command_buffers(data.command_pool, &[command_buffer]);

        Ok(())
    }

    unsafe fn transition_image_layout(
        device: &Device,
        data: &AppData,
        image: vk::Image,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) -> Result<()> {
        let command_buffer = Self::begin_single_time_commands(device, data)?;

        let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            match format {
                vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT => {
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                }
                _ => vk::ImageAspectFlags::DEPTH,
            }
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let subresource = vk::ImageSubresourceRange::builder()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1);

        let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
            match (old_layout, new_layout) {
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER, // transfer writes must occur in the pipeline transfer stage
                ),
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ) => (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                ),
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => {
                    (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    )
                }
                _ => return Err(anyhow!("Unsupported image layout transition")),
            };

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED) // barrier between queue families
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(subresource)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage_mask, // perations will wait on the barrier.
            dst_stage_mask, //
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        Self::end_single_time_commands(device, data, command_buffer)?;

        Ok(())
    }

    unsafe fn copy_buffer_to_image(
        device: &Device,
        data: &AppData,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let command_buffer = Self::begin_single_time_commands(device, data)?;
        let subresources = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);

        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(subresources)
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        //  has already been transitioned to the layout that is optimal for copying pixels
        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        Self::end_single_time_commands(device, data, command_buffer)?;

        Ok(())
    }

    unsafe fn create_texture_image_view(device: &Device, data: &mut AppData) -> Result<()> {
        data.texture_image_view = Self::create_image_view(
            device,
            data.texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            data.mip_levels,
        )?;

        Ok(())
    }

    unsafe fn create_image_view(
        device: &Device,
        image: vk::Image,
        format: vk::Format,
        aspects: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> Result<vk::ImageView> {
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(aspects)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1);

        let info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(format)
            .subresource_range(subresource_range);

        Ok(device.create_image_view(&info, None)?)
    }

    unsafe fn create_texture_sampler(device: &Device, data: &mut AppData) -> Result<()> {
        // Textures are usually accessed through samplers, which will apply filtering and transformations to compute the final color that is retrieved.
        // anisotorpic filtering, addressing mode(clamp, repeat, ...)
        let info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR) // concerns the oversampling problem
            .min_filter(vk::Filter::LINEAR) // concerns the undersampling problem
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false) // If a comparison function is enabled, then texels will first be compared to a value, and the result of that comparison is used in filtering operations., mainly used like shadow maps
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(data.mip_levels as f32);
        data.texture_sampler = device.create_sampler(&info, None)?;

        Ok(())
    }

    unsafe fn create_depth_objects(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        // The stencil component is used for stencil tests, which is an additional test that can be combined with depth testing.
        let format = Self::get_depth_format(instance, data)?;
        let (depth_image, depth_image_memory) = Self::create_image(
            instance,
            device,
            data,
            data.swapchain_extent.width,
            data.swapchain_extent.height,
            1,
            data.msaa_samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        data.depth_image = depth_image;
        data.depth_image_memory = depth_image_memory;
        data.depth_image_view = Self::create_image_view(
            device,
            data.depth_image,
            format,
            vk::ImageAspectFlags::DEPTH,
            1,
        )?;

        Self::transition_image_layout(
            device,
            data,
            data.depth_image,
            format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            1,
        )?;

        Ok(())
    }

    unsafe fn get_suppoted_format(
        instance: &Instance,
        data: &AppData,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Result<vk::Format> {
        candidates
            .iter()
            .cloned()
            .find(|f| {
                let properties =
                    instance.get_physical_device_format_properties(data.physical_device, *f);
                match tiling {
                    vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                    vk::ImageTiling::OPTIMAL => {
                        properties.optimal_tiling_features.contains(features)
                    }
                    _ => false,
                }
            })
            .ok_or_else(|| anyhow!("Failed to find supported format"))
    }

    unsafe fn get_depth_format(instance: &Instance, data: &AppData) -> Result<vk::Format> {
        let candidates = &[
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];

        Self::get_suppoted_format(
            instance,
            data,
            candidates,
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    fn load_model(data: &mut AppData) -> Result<()> {
        let mut reader = BufReader::new(File::open("src/resources/VikingRoom/viking_room.obj")?);

        let (models, _) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions {
                triangulate: true,
                ..Default::default()
            },
            |_| Ok(Default::default()),
        )?;

        let mut unique_vertices = HashMap::new();

        for model in models {
            for index in &model.mesh.indices {
                let pos_offset = (3 * index) as usize;
                let tex_coord_offset = (2 * index) as usize;

                let vertex = Vertex {
                    pos: vec3(
                        model.mesh.positions[pos_offset],
                        model.mesh.positions[pos_offset + 1],
                        model.mesh.positions[pos_offset + 2],
                    ),
                    color: vec3(1.0, 1.0, 1.0),
                    tex_coord: vec2(
                        model.mesh.texcoords[tex_coord_offset],
                        // The OBJ format assumes a coordinate system where a vertical coordinate of 0 means the bottom of the image,
                        // however we've uploaded our image into Vulkan in a top to bottom orientation where 0 means the top of the image.
                        1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                    ),
                };
                if let Some(index) = unique_vertices.get(&vertex) {
                    data.indices.push(*index as u32);
                } else {
                    let index = data.vertices.len();
                    unique_vertices.insert(vertex, index);
                    data.vertices.push(vertex);
                    data.indices.push(data.indices.len() as u32);
                }
            }
        }

        Ok(())
    }

    unsafe fn generate_mipmaps(
        instance: &Instance,
        device: &Device,
        data: &AppData,
        image: vk::Image,
        format: vk::Format,
        width: u32,
        height: u32,
        mip_levels: u32,
    ) -> Result<()> {
        if !instance
            .get_physical_device_format_properties(data.physical_device, format)
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            return Err(anyhow!(
                "Texture image format does not support linear blitting"
            ));
        }

        let command_buffer = Self::begin_single_time_commands(device, data)?;

        let subresource = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(1)
            .level_count(1);

        let mut barrier = vk::ImageMemoryBarrier::builder()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(subresource);

        let mut mip_width = width;
        let mut mip_height = height;

        for i in 1..mip_levels {
            barrier.subresource_range.base_mip_level = i - 1;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[barrier],
            );

            let src_subresource = vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(i - 1)
                .base_array_layer(0)
                .layer_count(1);

            let dst_subresource = vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(i)
                .base_array_layer(0)
                .layer_count(1);

            let blit = vk::ImageBlit::builder()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width as i32,
                        y: mip_height as i32,
                        z: 1, // a 2D image has a depth of 1.
                    },
                ])
                .src_subresource(src_subresource)
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                        y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(dst_subresource);

            device.cmd_blit_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );

            barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[barrier],
            );

            if mip_width > 1 {
                mip_width /= 2;
            }
            if mip_height > 1 {
                mip_height /= 2;
            }
        }

        barrier.subresource_range.base_mip_level = mip_levels - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        Self::end_single_time_commands(device, data, command_buffer)?;

        Ok(())
    }

    unsafe fn get_max_msaa_samples(instance: &Instance, data: &AppData) -> vk::SampleCountFlags {
        let properties = instance.get_physical_device_properties(data.physical_device);
        let counts = properties.limits.framebuffer_color_sample_counts
            & properties.limits.framebuffer_depth_sample_counts;
        [
            vk::SampleCountFlags::_64,
            vk::SampleCountFlags::_32,
            vk::SampleCountFlags::_16,
            vk::SampleCountFlags::_8,
            vk::SampleCountFlags::_4,
            vk::SampleCountFlags::_2,
        ]
        .iter()
        .cloned()
        .find(|c| counts.contains(*c))
        .unwrap_or(vk::SampleCountFlags::_1)
    }

    unsafe fn create_color_objects(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        //  this color buffer doesn't need mipmaps since it's not going to be used as a texture:
        let (color_image, color_image_memory) = Self::create_image(
            instance,
            device,
            data,
            data.swapchain_extent.width,
            data.swapchain_extent.height,
            1,
            data.msaa_samples,
            data.swapchain_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        data.color_image = color_image;
        data.color_image_memory = color_image_memory;

        data.color_image_view = Self::create_image_view(
            device,
            data.color_image,
            data.swapchain_format,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        Ok(())
    }

    unsafe fn reset_camera(&mut self) {
        self.data.camera_pos = self.data.initial_camera_pos;
        let camera_pos = vec3_from_array(self.data.camera_pos);
        let camera_direction = camera_pos.normalize();
        let camera_up = Vec3::cross(camera_direction, vec3(1.0, 0.0, 0.0));
        self.data.camera_direction = array3_from_vec(camera_direction);
        self.data.camera_up = array3_from_vec(camera_up);
    }

    unsafe fn reset_camera_up(&mut self) {
        let camera_pos = vec3_from_array(self.data.camera_pos);
        let mut camera_direction = vec3_from_array(self.data.camera_direction);
        let mut camera_up = vec3_from_array(self.data.camera_up);
        let horizon = Vec3::cross(camera_up, camera_direction);
        camera_up = vec3(0.0, -1.0, 0.0);
        camera_direction = Vec3::cross(horizon, camera_up);
        self.data.camera_up = array3_from_vec(camera_up);
        self.data.camera_direction = array3_from_vec(camera_direction);
    }
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);
unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, data, physical_device) {
            warn!(
                "Skipping Physical Device (`{}`): {}",
                properties.device_name, error
            );
        } else {
            info!("Selected Physical Device (`{}`).", properties.device_name);
            data.physical_device = physical_device;
            data.msaa_samples = App::get_max_msaa_samples(instance, data);
            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device"))
}

unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let properties = instance.get_physical_device_properties(physical_device);
    if properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
        return Err(anyhow!(SuitabilityError(
            "Only discrete GPUs are supported"
        )));
    }

    let features = instance.get_physical_device_features(physical_device);
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError("No sampler anisotropy")));
    }
    if features.geometry_shader != vk::TRUE {
        return Err(anyhow!(SuitabilityError(
            "Missing geometry shader supported"
        )));
    }

    QueueFamilyIndices::get(instance, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support")));
    }

    Ok(())
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError("Device Extensions Not Supported")))
    }
}

#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}
// TODO: support for trasfer bit, see https://kylemayes.github.io/vulkanalia/vertex/staging_buffer.html

impl QueueFamilyIndices {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);
        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, properties) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                index as u32,
                data.surface,
            )? {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families")))
        }
    }
}

#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }

    fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        formats
            .iter()
            .cloned()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or_else(|| formats[0])
    }

    fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        present_modes
            .iter()
            .cloned()
            .find(|m| *m == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO)
    }

    fn get_swapchain_extent(
        window: &Window,
        capabilities: vk::SurfaceCapabilitiesKHR,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let size = window.inner_size();
            let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
            vk::Extent2D::builder()
                .width(clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                    size.width,
                ))
                .height(clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                    size.height,
                ))
                .build()
        }
    }
}

#[repr(C)] // for compatibility of C struct
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: Vec3,
    color: Vec3,
    tex_coord: Vec2,
}

impl Vertex {
    const fn new(pos: Vec3, color: Vec3, tex_coord: Vec2) -> Self {
        Self {
            pos,
            color,
            tex_coord,
        }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        //  at which rate to load data from memory throughout the vertices
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        // how to extract a vertex attribute from a chunk of vertex data originating from a binding description
        // two attributes, position and color
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0) // directive of the input in the vertex shader
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();

        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec3>() as u32)
            .build();

        let tex_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32)
            .build();

        [pos, color, tex_coord]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.color == other.pos && self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}

unsafe fn view(
    camera_pos: cgmath::Vector3<f32>,
    direction: cgmath::Vector3<f32>,
    up: cgmath::Vector3<f32>,
) -> cgmath::Matrix4<f32> {
    let n_z = cgmath::Vector3::normalize(direction);
    let n_x = cgmath::Vector3::normalize(cgmath::Vector3::cross(up, n_z));
    let n_y = cgmath::Vector3::cross(n_x, n_z);
    let orientation = cgmath::Matrix4::new(
        n_x.x, n_y.x, n_z.x, 0.0, n_x.y, n_y.y, n_z.y, 0.0, n_x.z, n_y.z, n_z.z, 0.0, 0.0, 0.0,
        0.0, 1.0,
    );
    let translate = cgmath::Matrix4::new(
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -camera_pos.x,
        -camera_pos.y,
        -camera_pos.z,
        1.0,
    );
    return orientation * translate;
}

unsafe fn rodrigues(
    rotate: &mut cgmath::Matrix3<f32>,
    c: f32,
    s: f32,
    n: &cgmath::Vector3<f32>,
) -> Result<()> {
    let ac = 1.0 - c;
    let xyac = n.x * n.y * ac;
    let yzac = n.y * n.z * ac;
    let zxac = n.x * n.z * ac;
    let xs = n.x * s;
    let ys = n.y * s;
    let zs = n.z * s;
    // rotate = glm::mat3(c + n.x * n.x * ac, n.x * n.y * ac + n.z * s, n.z * n.x * ac - n.y * s,
    //     n.x * n.y * ac - n.z * s, c + n.y * n.y * ac, n.y * n.z * ac + n.x * s,
    //     n.z * n.x * ac + n.y * s, n.y * n.z * ac - n.x * s, c + n.z * n.z * ac);
    *rotate = cgmath::Matrix3::new(
        c + n.x * n.x * ac,
        xyac + zs,
        zxac - ys,
        xyac - zs,
        c + n.y * n.y * ac,
        yzac + xs,
        zxac + ys,
        yzac - xs,
        c + n.z * n.z * ac,
    );
    Ok(())
}

fn vec3_from_array(a: [f32; 3]) -> Vector3<f32> {
    vec3(a[0], a[1], a[2])
}

fn array3_from_vec(v: Vector3<f32>) -> [f32; 3] {
    [v.x, v.y, v.z]
}

fn vec2_from_array(a: [f32; 2]) -> Vector2<f32> {
    vec2(a[0], a[1])
}

fn array2_from_vec(v: Vector2<f32>) -> [f32; 2] {
    [v.x, v.y]
}
