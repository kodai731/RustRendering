use cgmath::{SquareMatrix, Vector3};

use crate::app::App;
use crate::vulkanr::context::CommandState;
use crate::vulkanr::vulkan::*;

pub(super) fn save_flame_history_npy_if_requested(app: &mut App) {
    let batch = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::BatchRun>();
    if let Some(batch) = batch {
        let sequence_dir = batch.sequence_dir.clone();
        let total_count = batch.total_count;
        let captures_remaining = batch.captures_remaining;
        if let Some(sink) = app
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::FlameDumpSink>()
        {
            let path = sink.path.clone();
            let npy_path = if let Some(ref sequence_dir) = sequence_dir {
                let frame_index = total_count - captures_remaining;
                sequence_dir.join(format!("flame_{:02}.npy", frame_index))
            } else {
                crate::ecs::systems::flame_dump_npy_path(&path)
            };
            if let Err(e) = unsafe { app.save_flame_history_npy(&npy_path) } {
                eprintln!("Failed to save flame history npy: {:?}", e);
            }
            let ubo_path = {
                let mut p = npy_path.clone();
                let stem = p.file_stem().unwrap();
                p.set_file_name(format!("{}.ubo.bin", stem.to_string_lossy()));
                p
            };
            let flame_entities = app.data.ecs_world.query_flames();
            if let Some(first) = flame_entities.first() {
                if let (Some(effect), Some(baked), Some(temporal_accum)) = (
                    app.data
                        .ecs_world
                        .get_component::<crate::ecs::component::FlameEffect>(*first),
                    app.data
                        .ecs_world
                        .get_component::<crate::ecs::component::FlameBaked>(*first),
                    app.data
                        .ecs_world
                        .get_component::<crate::ecs::component::FlameTemporalAccum>(*first),
                ) {
                    let ubo = thyllore_effect_core::build_flame_ubo(effect, baked, temporal_accum);
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            &ubo as *const thyllore_effect_core::FlameUBO as *const u8,
                            std::mem::size_of::<thyllore_effect_core::FlameUBO>(),
                        )
                    };
                    if let Err(e) = std::fs::write(&ubo_path, bytes) {
                        eprintln!("Failed to save flame UBO bin: {:?}", e);
                    }
                }
            }
        }
    }
}

pub(super) fn save_water_probe_if_requested(app: &mut App) {
    let batch = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::BatchRun>();
    if let Some(batch) = batch {
        let water_probe_path = match &batch.water_probe_path {
            Some(p) => p.clone(),
            None => return,
        };

        let hdr = app
            .data
            .viewport
            .hdr_buffer
            .as_ref()
            .expect("hdr_buffer not initialized");
        let w = hdr.width;
        let h = hdr.height;
        let image = hdr.color_image;
        let image_size = (w * h * 8) as vk::DeviceSize;
        let command_pool = app.resource::<CommandState>().pool.command_pool;

        let (buffer, buffer_memory, command_buffer) = unsafe {
            app.copy_image_to_buffer(
                image,
                w,
                h,
                image_size,
                command_pool,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )
        }
        .expect("copy_image_to_buffer failed for water probe");

        let device = &app.rrdevice.device;
        let data_ptr = unsafe {
            device
                .map_memory(buffer_memory, 0, image_size, vk::MemoryMapFlags::empty())
                .expect("map_memory failed")
        };
        let slice =
            unsafe { std::slice::from_raw_parts(data_ptr as *const u8, image_size as usize) };

        let mut f32_data: Vec<f32> = Vec::with_capacity((w * h * 4) as usize);
        for chunk in slice.chunks_exact(8) {
            let r_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let g_bits = u16::from_le_bytes([chunk[2], chunk[3]]);
            let b_bits = u16::from_le_bytes([chunk[4], chunk[5]]);
            let a_bits = u16::from_le_bytes([chunk[6], chunk[7]]);
            f32_data.push(thyllore_math_core::f16_to_f32(r_bits));
            f32_data.push(thyllore_math_core::f16_to_f32(g_bits));
            f32_data.push(thyllore_math_core::f16_to_f32(b_bits));
            f32_data.push(thyllore_math_core::f16_to_f32(a_bits));
        }

        unsafe {
            device.unmap_memory(buffer_memory);
            device.free_command_buffers(command_pool, &[command_buffer]);
            device.free_memory(buffer_memory, None);
            device.destroy_buffer(buffer, None);
        }

        // Camera view and proj matrices from ProjectionData (same as the frame UBO)
        let proj_data = app
            .data
            .ecs_world
            .resource::<crate::ecs::resource::ProjectionData>();
        let inv_view_proj = crate::ecs::systems::water::probe::inverse_view_proj_f64(
            proj_data.proj,
            proj_data.view,
        );

        // Camera position from inverse(view) — translation component (inv[3].xyz)
        let inv_view = proj_data.view.invert().unwrap();
        let camera_pos = Vector3::new(inv_view[3][0], inv_view[3][1], inv_view[3][2]);

        // Water torus effect
        let waters: Vec<_> = app.data.ecs_world.query_waters();
        if let Some(&first) = waters.first() {
            if let Some(effect) = app
                .data
                .ecs_world
                .get_component::<crate::ecs::component::WaterTorusEffect>(first)
            {
                let inverse_model = {
                    let model = thyllore_effect_core::build_water_model_matrix(effect);
                    model.invert().unwrap_or(cgmath::Matrix4::identity())
                };
                let major_radius = effect.major_radius;
                let minor_over_major = effect.minor_radius / effect.major_radius;

                // Determine which root to compare based on WaterRenderSettings.debug_view
                let debug_view = app
                    .data
                    .ecs_world
                    .get_resource::<crate::ecs::resource::WaterRenderSettings>()
                    .map(|s| s.debug_view)
                    .unwrap_or(3);
                let which = if debug_view == 4 {
                    crate::ecs::systems::water::probe::ProbeRoot::Exit
                } else {
                    crate::ecs::systems::water::probe::ProbeRoot::Nearest
                };

                let report = crate::ecs::systems::water::probe::compute_water_probe_report(
                    &f32_data,
                    w,
                    h,
                    inv_view_proj,
                    inverse_model,
                    major_radius,
                    camera_pos,
                    minor_over_major,
                    which,
                );

                let json_path = {
                    let mut p = water_probe_path.clone();
                    let stem = p.file_stem().unwrap();
                    p.set_file_name(format!("{}.json", stem.to_string_lossy()));
                    p
                };
                if let Err(e) =
                    std::fs::write(&json_path, serde_json::to_string_pretty(&report).unwrap())
                {
                    eprintln!("Failed to write water probe json: {:?}", e);
                }

                let npy_path = {
                    let mut p = water_probe_path.clone();
                    let stem = p.file_stem().unwrap();
                    p.set_file_name(format!("{}.npy", stem.to_string_lossy()));
                    p
                };
                if let Err(e) = thyllore_math_core::write_npy_f32(
                    &npy_path,
                    &[h as usize, w as usize, 4],
                    &f32_data,
                ) {
                    eprintln!("Failed to write water probe npy: {:?}", e);
                }

                println!(
                    "water probe dumped to {} ({} pixels, {} mismatch, root={})",
                    json_path.display(),
                    report.pixels,
                    report.count_mismatch,
                    report.root
                );
            }
        } else {
            eprintln!("water probe skipped: no water torus effect entity");
        }
    }
}
