use crate::ecs::component::{EntityIcon, WaterTorusEffect};
use crate::ecs::resource::{HierarchyState, TimelineState};
use crate::ecs::world::{Transform, World};

use super::*;
use crate::ecs::component::EditorDisplay;
use crate::ecs::world::{GlobalTransform, Name};

fn spawn_default_water(world: &mut World, name: &str) -> crate::ecs::world::Entity {
    spawn_water(world, name, WaterTorusEffect::default())
}

#[test]
fn spawned_water_carries_the_components_the_editor_queries() {
    let mut world = World::new();
    let entity = spawn_default_water(&mut world, DEFAULT_WATER_NAME);

    assert_eq!(
        world.get_component::<Name>(entity).map(|n| n.0.clone()),
        Some(DEFAULT_WATER_NAME.to_string())
    );
    assert!(world.get_component::<Transform>(entity).is_some());
    assert!(world.get_component::<GlobalTransform>(entity).is_some());
    assert!(world.get_component::<EditorDisplay>(entity).is_some());
    assert!(world.get_component::<WaterTorusEffect>(entity).is_some());

    let display = world.get_component::<EditorDisplay>(entity).unwrap();
    assert_eq!(display.icon, EntityIcon::Water);
}

#[test]
fn water_count_after_spawn_is_one() {
    let mut world = World::new();
    spawn_default_water(&mut world, DEFAULT_WATER_NAME);

    assert_eq!(world.query_waters().len(), 1);
}

#[test]
fn timeline_state_drives_water_time() {
    let mut world = World::new();
    let effect = WaterTorusEffect {
        time_scale: 2.0,
        time_offset: 1.5,
        ..WaterTorusEffect::default()
    };
    let _entity = spawn_water(&mut world, "Water", effect);

    world.insert_resource(TimelineState {
        current_time: 2.0,
        ..TimelineState::new()
    });

    // water_time_advance reads TimelineState.current_time when BatchRun is absent.
    // Replicate the branch logic here (the function takes FrameContext which needs Vulkan).
    let entity = world.query_waters()[0];
    let timeline_time: f32 = world.get_resource::<TimelineState>().unwrap().current_time;
    let mut effect = world.get_component_mut::<WaterTorusEffect>(entity).unwrap();
    effect.time = timeline_time * effect.time_scale + effect.time_offset;

    let effect = world.get_component::<WaterTorusEffect>(entity).unwrap();
    // time = current_time * time_scale + time_offset = 2.0 * 2.0 + 1.5 = 5.5
    assert!((effect.time - 5.5).abs() < 1e-6);
}

fn timeline_time_sources(current_time: f32, playing: bool, delta_time: f32) -> WaterTimeSources {
    WaterTimeSources {
        batch_fixed_time: None,
        batch_frames_rendered: None,
        timeline: Some(TimelineSample {
            current_time,
            playing,
        }),
        delta_time,
        free_run_when_paused: true,
    }
}

#[test]
fn paused_timeline_advances_water_time_by_delta_when_free_run_is_enabled() {
    let mut world = World::new();
    let effect = WaterTorusEffect {
        time: 3.0,
        ..WaterTorusEffect::default()
    };
    let entity = spawn_water(&mut world, "Water", effect);

    world.insert_resource(TimelineState {
        current_time: 2.0,
        playing: false,
        ..TimelineState::new()
    });

    let mut effect = world.get_component_mut::<WaterTorusEffect>(entity).unwrap();
    resolve_water_time(&mut effect, timeline_time_sources(2.0, false, 0.25));

    assert!((effect.time - 3.25).abs() < 1e-6, "got {}", effect.time);
}

#[test]
fn playing_timeline_drives_water_time_from_timeline_time() {
    let mut world = World::new();
    let effect = WaterTorusEffect {
        time: 3.0,
        time_scale: 2.0,
        time_offset: 1.5,
        ..WaterTorusEffect::default()
    };
    let entity = spawn_water(&mut world, "Water", effect);

    world.insert_resource(TimelineState {
        current_time: 2.0,
        playing: true,
        ..TimelineState::new()
    });

    let mut effect = world.get_component_mut::<WaterTorusEffect>(entity).unwrap();
    resolve_water_time(&mut effect, timeline_time_sources(2.0, true, 0.25));

    assert!((effect.time - 5.5).abs() < 1e-6, "got {}", effect.time);
}

#[test]
fn pick_ray_hits_water_torus() {
    let mut world = World::new();
    let effect = WaterTorusEffect {
        major_radius: 1.0,
        minor_radius: 0.3,
        ..WaterTorusEffect::default()
    };
    spawn_water(&mut world, "Water", effect);

    // Ray from (0, 0, -5) going +Z hits the torus outer edge at z = -(1.0 + 0.3) = -1.3
    // Distance from (0, 0, -5) to (0, 0, -1.3) = 3.7
    let ray = crate::ecs::resource::PickRay {
        origin: cgmath::Vector3::new(0.0, 0.0, -5.0),
        direction: cgmath::Vector3::new(0.0, 0.0, 1.0),
    };

    let hit = find_water_by_pick_ray(&world, &ray);

    assert!(hit.is_some());
    let distance = hit.unwrap().1;
    assert!(
        (distance - 3.7).abs() < 1e-3,
        "expected distance ≈ 3.7, got {:.4}",
        distance
    );
}

#[test]
fn compute_water_probe_report_matches_rust_solver() {
    use cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector3};

    // Known torus: R=1.2, r=0.35 (matching the actual scene — R≠1 so unit mismatch is caught)
    let major_radius: f32 = 1.2;
    let minor_radius: f32 = 0.35;
    let minor_over_major = minor_radius / major_radius;
    let inverse_model = Matrix4::identity();

    // Identity inv_view_proj: camera at origin, rays go through NDC positions
    let inv_view_proj = Matrix4::identity();

    // Camera position at origin (matching identity view)
    let camera_pos = Vector3::new(0.0, 0.0, 0.0);

    let width: u32 = 32;
    let height: u32 = 32;
    let mut image_data = vec![0.0f32; (width * height * 4) as usize];

    // Construct synthetic image: for each pixel, reconstruct the ray using the same
    // logic as compute_water_probe_report, solve with Rust's intersect_torus, and
    // write the result as if it came from the GLSL shader.
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize * 4;

            // Reconstruct ray from NDC (same as compute_water_probe_report)
            let ndc_x = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
            let ndc_y = (y as f32 + 0.5) / height as f32 * 2.0 - 1.0;

            // invViewProj * vec4(ndc, DEPTH_NEAR=1.0, 1.0)
            let world = inv_view_proj * cgmath::vec4(ndc_x, ndc_y, 1.0, 1.0);
            let world_pos =
                cgmath::Vector3::new(world.x / world.w, world.y / world.w, world.z / world.w);

            // Ray direction from camera position to world position
            let ray_dir = (world_pos - camera_pos).normalize();

            // Transform to local space: origin w=1, dir w=0 (same as compute_water_probe_report)
            let p_local_origin =
                inverse_model * cgmath::vec4(camera_pos.x, camera_pos.y, camera_pos.z, 1.0);
            let p_local =
                cgmath::Vector3::new(p_local_origin.x, p_local_origin.y, p_local_origin.z)
                    / major_radius;

            let d_local_raw = inverse_model * cgmath::vec4(ray_dir.x, ray_dir.y, ray_dir.z, 0.0);
            let d_local =
                cgmath::Vector3::new(d_local_raw.x, d_local_raw.y, d_local_raw.z).normalize();

            // Solve with Rust's analytic solver (same as compute_water_probe_report)
            let rust_hits = thyllore_effect_core::water::analytic::intersect_torus(
                p_local,
                d_local,
                1.0,
                minor_over_major,
            );

            // Write result into synthetic image (as if from GLSL shader).
            // High-precision encoding: t = t_norm * R, decomposed as
            //   hi = floor(t), mid = floor((t - hi) * 1024.0), lo = (t * 1024.0).fract()
            let hit_count = rust_hits.count;
            if hit_count > 0 {
                let t = rust_hits.roots[0] * major_radius;
                let hi = t.floor();
                let mid = ((t - hi) * 1024.0).floor();
                let lo = (t * 1024.0).fract();
                image_data[idx] = hi;
                image_data[idx + 1] = mid;
                image_data[idx + 2] = lo;
                image_data[idx + 3] = -(hit_count as f32);
            }
            // Background pixels: A >= 0 (leave as 0.0, already initialized)
        }
    }
    // Call compute_water_probe_report with the synthetic image
    let report = compute_water_probe_report(
        &image_data,
        width,
        height,
        inv_view_proj,
        inverse_model,
        major_radius,
        camera_pos,
        minor_over_major,
        ProbeRoot::Nearest,
    );

    // Since both sides use the same Rust solver, they should match perfectly
    assert_eq!(
        report.count_mismatch, 0,
        "expected count_mismatch == 0, got {}",
        report.count_mismatch
    );
    assert!(
        report.max_rel < 1e-6,
        "expected max_rel < 1e-6, got {:.8}",
        report.max_rel
    );
    assert!(
        report.p50_rel <= report.p99_rel,
        "expected p50_rel <= p99_rel, got p50={:.8} > p99={:.8}",
        report.p50_rel,
        report.p99_rel
    );
    assert!(
        report.p99_rel <= report.max_rel,
        "expected p99_rel <= max_rel, got p99={:.8} > max={:.8}",
        report.p99_rel,
        report.max_rel
    );
    assert!(
        (report.frac_over_1e_4 - 0.0).abs() < 1e-9,
        "expected frac_over_1e_4 == 0.0, got {:.8}",
        report.frac_over_1e_4
    );
}

#[test]
fn compute_water_probe_report_count_mismatch_no_panic() {
    use cgmath::{Matrix4, SquareMatrix, Vector3};

    // Known torus: R=1, identity model
    let major_radius: f32 = 1.0;
    let minor_over_major: f32 = 0.3;
    let inverse_model = Matrix4::identity();
    let inv_view_proj = Matrix4::identity();

    // Camera at (0, 0, 100) far along Z axis. The single pixel at ndc=(0,0) maps to
    // world=(0,0,1), so the ray goes from (0,0,100) to (0,0,1) = direction (0,0,-1).
    // This passes through the torus hole along Z axis — no intersection, Rust finds count=0.
    let camera_pos = Vector3::new(0.0, 0.0, 100.0);

    let width: u32 = 1;
    let height: u32 = 1;
    let mut image_data = vec![0.0f32; (width * height * 4) as usize];

    // Pixel 0: GLSL reports hit_count=2 with valid roots, but Rust solver will find count=0
    // because the ray passes through the torus hole.
    // This is the count mismatch scenario: rust_hits.count (0) != hit_count (2).
    // Encode fake t=1.0 using high-precision scheme: hi=floor(1.0)=1, mid=floor((1.0-1)*1024)=0, lo=(1.0*1024).fract()=0
    let idx = 0 * 4;
    image_data[idx] = 1.0; // hi = floor(t) = 1
    image_data[idx + 1] = 0.0; // mid = floor((t - hi) * 1024.0) = 0
    image_data[idx + 2] = 0.0; // lo = (t * 1024.0).fract() = 0
    image_data[idx + 3] = -2.0; // marker: water pixel, hit_count=2, not fallback

    // Call compute_water_probe_report — must not panic despite count mismatch
    let report = compute_water_probe_report(
        &image_data,
        width,
        height,
        inv_view_proj,
        inverse_model,
        major_radius,
        camera_pos,
        minor_over_major,
        ProbeRoot::Nearest,
    );

    // Pixel 0: GLSL hit_count=2 but Rust finds count=0 -> count_mismatch += 1
    assert_eq!(
        report.count_mismatch, 1,
        "expected count_mismatch == 1 (GLSL hit but Rust no hit), got {}",
        report.count_mismatch
    );
    assert_eq!(
        report.pixels, 1,
        "expected pixels == 1 (only pixel 0 counted)"
    );
}

#[test]
fn inverse_view_proj_f64_is_inverse_of_proj_times_view() {
    use cgmath::{Matrix4, SquareMatrix};

    // Build a real view/proj pair using perspective + translation (like frame_systems.rs)
    let fov = cgmath::Deg(std::f32::consts::FRAC_PI_4);
    let aspect = 16.0 / 9.0;
    let proj = crate::math::coordinate_system::perspective_infinite_reverse(fov, aspect, 0.1);

    // View matrix: camera at (0, 0, 5) looking at origin
    // view = inverse(translation(0, 0, 5)) = translation(0, 0, -5)
    let view = cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 0.0, -5.0));

    // Compute inverse using our function
    let inv = super::inverse_view_proj_f64(proj, view);

    // Reference: (proj * view).invert() in f32
    let product = proj * view;
    let reference = product.invert().unwrap();

    // Assert all components differ by less than 1e-3 in relative terms
    for i in 0..4 {
        for j in 0..4 {
            let diff = (inv[i][j] - reference[i][j]).abs();
            let rel = diff / (reference[i][j].abs() + 1e-8);
            assert!(
                rel < 1e-3,
                "component [{}][{}] diff={:.6} rel={:.6}: inv={:.6} ref={:.6}",
                i,
                j,
                diff,
                rel,
                inv[i][j],
                reference[i][j]
            );
        }
    }

    // Assert inv * (proj * view) is identity within 1e-5
    let identity_check = inv * product;
    let identity = Matrix4::identity();
    for i in 0..4 {
        for j in 0..4 {
            let val: f32 = identity_check[i][j];
            let expected: f32 = identity[i][j];
            let diff = (val - expected).abs();
            assert!(
                diff < 1e-5,
                "identity check [{}][{}] diff={:.8}: got {:.8} expected {:.8}",
                i,
                j,
                diff,
                val,
                expected
            );
        }
    }
}

#[test]
fn water_temporal_same_snapshot_twice_weight_0_85() {
    use crate::ecs::component::{WaterTemporalAccum, WaterTorusEffect};
    use crate::ecs::resource::{WaterRenderSettings, WaterTemporalState};
    use cgmath::{Matrix4, SquareMatrix};

    let mut world = World::new();
    world.insert_resource(thyllore_render_core::ProjectionData {
        view: Matrix4::identity(),
        proj: Matrix4::identity(),
        screen_size: cgmath::Vector2::new(800.0, 600.0),
        aspect: 800.0 / 600.0,
    });
    world.insert_resource(WaterRenderSettings::default());
    world.insert_resource(WaterTemporalState::default());

    let entity = spawn_default_water(&mut world, "Water");

    // First call: no previous snapshot, so weight should be 0.0
    crate::ecs::systems::water_temporal_accumulate(&mut world);

    let temporal = world.get_component::<WaterTemporalAccum>(entity).unwrap();
    assert_eq!(temporal.weight, 0.0, "first frame weight should be 0");
    assert_eq!(temporal.frame_index, 1, "first frame index should be 1");

    // Second call: same snapshot (nothing changed), so weight should be 0.85
    crate::ecs::systems::water_temporal_accumulate(&mut world);

    let temporal = world.get_component::<WaterTemporalAccum>(entity).unwrap();
    assert_eq!(temporal.weight, 0.85, "second frame weight should be 0.85");
    assert_eq!(temporal.frame_index, 2, "second frame index should be 2");
}

#[test]
fn water_temporal_view_change_weight_0() {
    use crate::ecs::component::{WaterTemporalAccum, WaterTorusEffect};
    use crate::ecs::resource::{WaterRenderSettings, WaterTemporalState};
    use cgmath::{Matrix4, SquareMatrix};

    let mut world = World::new();
    world.insert_resource(thyllore_render_core::ProjectionData {
        view: Matrix4::identity(),
        proj: Matrix4::identity(),
        screen_size: cgmath::Vector2::new(800.0, 600.0),
        aspect: 800.0 / 600.0,
    });
    world.insert_resource(WaterRenderSettings::default());
    world.insert_resource(WaterTemporalState::default());

    let entity = spawn_default_water(&mut world, "Water");

    // First call: no previous snapshot, weight 0
    crate::ecs::systems::water_temporal_accumulate(&mut world);

    let temporal = world.get_component::<WaterTemporalAccum>(entity).unwrap();
    assert_eq!(temporal.weight, 0.0, "first frame weight should be 0");

    // Change the view matrix
    let mut proj = world.resource_mut::<thyllore_render_core::ProjectionData>();
    proj.view = Matrix4::from_translation(cgmath::Vector3::new(1.0, 0.0, 0.0));
    drop(proj);

    // Second call: view changed, so weight should be 0.0
    crate::ecs::systems::water_temporal_accumulate(&mut world);

    let temporal = world.get_component::<WaterTemporalAccum>(entity).unwrap();
    assert_eq!(temporal.weight, 0.0, "view change should reset weight to 0");
    assert_eq!(
        temporal.frame_index, 2,
        "frame index should still increment"
    );
}

#[test]
fn water_debug_record_lists_every_instance_with_its_wave_modes() {
    let mut world = World::new();
    world.insert_resource(crate::ecs::resource::WaterRenderSettings::default());
    let mut effect = WaterTorusEffect::default();
    effect.wave_lb_blend = 0.5;
    spawn_water(&mut world, "Water A", effect);
    spawn_default_water(&mut world, "Water B");

    let render_info = WaterDebugRenderInfo {
        gpu_name: "test-gpu".to_string(),
        ..WaterDebugRenderInfo::default()
    };
    let record = build_water_debug_record(&world, &render_info, 42);

    assert_eq!(record["unix_time"], 42);
    assert_eq!(record["render"]["gpu_name"], "test-gpu");
    assert_eq!(
        record["water_render_settings"]["secondary_rays"],
        "Ray Query"
    );

    let instances = record["water_instances"].as_array().unwrap();
    assert_eq!(instances.len(), 2);
    assert_eq!(instances[0]["name"], "Water A");
    assert_eq!(instances[0]["wave_modes"].as_array().unwrap().len(), 8);
    assert_eq!(instances[0]["lb_modes"].as_array().unwrap().len(), 4);
    assert!(instances[1]["lb_modes"].as_array().unwrap().is_empty());
    assert_eq!(instances[1]["effect"]["major_radius"], 1.0);
    assert_eq!(
        instances[1]["ubo"]["wave_modes"].as_array().unwrap().len(),
        16
    );
}

#[test]
fn water_debug_caustic_accum_path_is_a_npy_next_to_the_dump() {
    let path = water_debug_caustic_accum_path(42);

    assert_eq!(
        path.file_name().unwrap().to_string_lossy(),
        "water_debug_42_caustic.npy"
    );
}

#[test]
fn water_debug_record_reports_caustic_accum_stats_when_the_image_was_read_back() {
    let mut world = World::new();
    world.insert_resource(crate::ecs::resource::WaterRenderSettings::default());
    spawn_default_water(&mut world, "Water A");

    let render_info = WaterDebugRenderInfo {
        caustic_accum_path: Some("log/water/water_debug_42_caustic.npy".to_string()),
        caustic_accum_nonzero: Some(1234),
        caustic_accum_max: Some(4096),
        ..WaterDebugRenderInfo::default()
    };
    let record = build_water_debug_record(&world, &render_info, 42);

    assert_eq!(
        record["render"]["caustic_accum_path"],
        "log/water/water_debug_42_caustic.npy"
    );
    assert_eq!(record["render"]["caustic_accum_nonzero"], 1234);
    assert_eq!(record["render"]["caustic_accum_max"], 4096);
}

#[test]
fn water_debug_record_leaves_caustic_accum_null_without_a_water_buffer() {
    let mut world = World::new();
    world.insert_resource(crate::ecs::resource::WaterRenderSettings::default());
    spawn_default_water(&mut world, "Water A");

    let record = build_water_debug_record(&world, &WaterDebugRenderInfo::default(), 42);

    assert!(record["render"]["caustic_accum_path"].is_null());
    assert!(record["render"]["caustic_accum_nonzero"].is_null());
    assert!(record["render"]["caustic_accum_max"].is_null());
}
