use super::*;
fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

#[test]
fn pick_pixel_is_absent_without_the_flag() {
    assert_eq!(pick_pixel_resolve_from_args(&args(&["bin"])).unwrap(), None);
}

#[test]
fn pick_pixel_parses_a_pixel_pair() {
    assert_eq!(
        pick_pixel_resolve_from_args(&args(&["bin", "--batch-pick", "947,150"])).unwrap(),
        Some((947, 150))
    );
}

#[test]
fn pick_pixel_rejects_a_malformed_pair() {
    assert!(pick_pixel_resolve_from_args(&args(&["bin", "--batch-pick", "947"])).is_err());
    assert!(pick_pixel_resolve_from_args(&args(&["bin", "--batch-pick", "a,b"])).is_err());
}

#[test]
fn resolve_returns_none_without_flag() {
    let resolved = batch_run_resolve_from_args(&args(&["thyllore-animation"])).unwrap();
    assert!(resolved.is_none());
}

#[test]
fn resolve_parses_output_and_default_frames() {
    let resolved =
        batch_run_resolve_from_args(&args(&["bin", "--batch-screenshot", "/tmp/out.png"]))
            .unwrap()
            .unwrap();
    assert_eq!(resolved.output, PathBuf::from("/tmp/out.png"));
    assert_eq!(resolved.screenshot_frame, DEFAULT_SCREENSHOT_FRAME);
    assert!(matches!(resolved.state, BatchRunState::WaitingForFrame));
}

#[test]
fn resolve_parses_explicit_frames() {
    let resolved = batch_run_resolve_from_args(&args(&[
        "bin",
        "--batch-screenshot",
        "/tmp/out.png",
        "--batch-frames",
        "30",
    ]))
    .unwrap()
    .unwrap();
    assert_eq!(resolved.screenshot_frame, 30);
}

#[test]
fn resolve_rejects_missing_output() {
    assert!(batch_run_resolve_from_args(&args(&["bin", "--batch-screenshot"])).is_err());
    assert!(
        batch_run_resolve_from_args(&args(&["bin", "--batch-screenshot", "--batch-frames"]))
            .is_err()
    );
}

#[test]
fn resolve_rejects_non_png_output() {
    assert!(
        batch_run_resolve_from_args(&args(&["bin", "--batch-screenshot", "/tmp/out.jpg"])).is_err()
    );
}

#[test]
fn resolve_rejects_invalid_frames() {
    assert!(batch_run_resolve_from_args(&args(&[
        "bin",
        "--batch-screenshot",
        "/tmp/out.png",
        "--batch-frames",
        "0"
    ]))
    .is_err());
    assert!(batch_run_resolve_from_args(&args(&[
        "bin",
        "--batch-screenshot",
        "/tmp/out.png",
        "--batch-frames",
        "abc"
    ]))
    .is_err());
}

#[test]
fn resolve_rejects_frames_without_screenshot() {
    assert!(batch_run_resolve_from_args(&args(&["bin", "--batch-frames", "30"])).is_err());
}

#[test]
fn resolve_flame_mode_and_steps() {
    let overrides = resolve_engine_cli_overrides(&args(&[
        "bin",
        "--batch-flame-mode",
        "raymarch",
        "--batch-flame-steps",
        "512",
    ]))
    .unwrap();
    assert!(overrides.batch_run.is_none());
    assert_eq!(
        overrides.flame_mode,
        Some(FlameShadingMode::ReferenceRaymarch)
    );
    assert_eq!(overrides.flame_steps, Some(512));
}

#[test]
fn resolve_rejects_invalid_flame_overrides() {
    assert!(flame_mode_resolve_from_args(&args(&["bin", "--batch-flame-mode", "x"])).is_err());
    assert!(flame_steps_resolve_from_args(&args(&["bin", "--batch-flame-steps", "0"])).is_err());
    assert!(flame_steps_resolve_from_args(&args(&["bin", "--batch-flame-steps", "abc"])).is_err());
}

#[test]
fn resolve_camera_pose() {
    let pose = camera_pose_resolve_from_args(&args(&["bin", "--batch-camera", "30,5,4"]))
        .unwrap()
        .unwrap();
    assert_eq!(
        pose,
        BatchCameraPose {
            yaw_degrees: 30.0,
            pitch_degrees: 5.0,
            distance: 4.0,
            pivot: None
        }
    );
    let pose = camera_pose_resolve_from_args(&args(&["bin", "--batch-camera", "30,5,4,0,1.2,0"]))
        .unwrap()
        .unwrap();
    assert_eq!(pose.pivot, Some([0.0, 1.2, 0.0]));
    assert!(camera_pose_resolve_from_args(&args(&["bin"]))
        .unwrap()
        .is_none());
}

#[test]
fn resolve_rejects_invalid_camera_pose() {
    for value in ["30,5", "a,b,c", "30,5,0", "30,5,-1", "30,5,4,0,1"] {
        assert!(
            camera_pose_resolve_from_args(&args(&["bin", "--batch-camera", value])).is_err(),
            "expected error for '{value}'"
        );
    }
}

#[test]
fn tick_requests_screenshot_at_target_frame() {
    let mut world = World::new();
    world.insert_resource(UIEventQueue::default());
    world.insert_resource(BatchRun::new(PathBuf::from("/tmp/out.png"), 2, Vec::new()));

    batch_run_tick(&world);
    assert!(matches!(
        world.resource::<BatchRun>().state,
        BatchRunState::WaitingForFrame
    ));

    batch_run_tick(&world);
    assert!(matches!(
        world.resource::<BatchRun>().state,
        BatchRunState::ScreenshotRequested
    ));
}

#[test]
fn record_ignores_keyboard_screenshot_while_waiting() {
    let world = {
        let mut world = World::new();
        world.insert_resource(BatchRun::new(
            PathBuf::from("/tmp/out.png"),
            100,
            Vec::new(),
        ));
        world
    };

    batch_run_record_screenshot(&world, Ok("log/screenshot_1.png".to_string()));
    assert!(matches!(
        world.resource::<BatchRun>().state,
        BatchRunState::WaitingForFrame
    ));
}

#[test]
fn record_stores_error_result() {
    let mut world = World::new();
    world.insert_resource(BatchRun::new(PathBuf::from("/tmp/out.png"), 1, Vec::new()));
    world.resource_mut::<BatchRun>().state = BatchRunState::ScreenshotRequested;

    batch_run_record_screenshot(&world, Err("save failed".to_string()));

    let batch = world.resource::<BatchRun>();
    assert!(batch.is_completed());
    let (ok, line) = batch_run_report(&batch);
    assert!(!ok);
    assert!(line.contains("save failed"));
}

#[test]
fn report_incomplete_state_is_error() {
    let batch = BatchRun::new(PathBuf::from("/tmp/out.png"), 1, Vec::new());
    let (ok, line) = batch_run_report(&batch);
    assert!(!ok);
    assert!(line.contains("before screenshot completed"));
}

#[test]
fn flame_style_path_only_defaults_to_all_groups() {
    let args: Vec<String> = vec![
        "--batch-flame-style".into(),
        "assets/flames/styles/pillar.style.ron".into(),
    ];
    let (path, groups) = flame_style_resolve_from_args(&args).unwrap().unwrap();
    assert_eq!(path, "assets/flames/styles/pillar.style.ron");
    assert_eq!(groups, thyllore_effect_core::StyleGroups::default());
}

#[test]
fn flame_style_group_subset() {
    let args: Vec<String> = vec!["--batch-flame-style".into(), "s.ron,motion,optics".into()];
    let (_, groups) = flame_style_resolve_from_args(&args).unwrap().unwrap();
    assert!(groups.motion && groups.optics && !groups.texture);
}

#[test]
fn flame_style_unknown_group_error() {
    let args: Vec<String> = vec!["--batch-flame-style".into(), "s.ron,shape".into()];
    assert!(flame_style_resolve_from_args(&args).is_err());
}

#[test]
fn flame_style_ron_roundtrip_applies() {
    let ron_text = r#"FlameStyle(
            version: 1,
            name: "pillar-ref",
            motion: (twist_gain: Some(6.0), meander_amp_over_r0: Some(0.5)),
            optics: (tau0: Some(4.0)),
        )"#;
    let style: thyllore_effect_core::FlameStyle = ron::from_str(ron_text).unwrap();
    let mut effect = FlameEffect::default();
    effect.radius = 2.0;
    let applied = thyllore_effect_core::apply_flame_style(
        &mut effect,
        &style,
        thyllore_effect_core::StyleGroups::default(),
    );
    assert_eq!(effect.twist.gain, 6.0);
    assert_eq!(effect.meander.amp, 1.0);
    assert_eq!(effect.optical_depth, 4.0);
    assert_eq!(applied.len(), 3);
}

#[test]
fn flame_style_dump_load_roundtrip() {
    let effect = FlameEffect::default();
    let path = std::env::temp_dir().join("thyllore_style_test.style.ron");
    let path_str = path.to_str().unwrap();
    dump_flame_style_to_path(&effect, path_str);
    let style = load_flame_style_from_path(path_str).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        style,
        thyllore_effect_core::flame_style_from_effect(&effect, "thyllore_style_test")
    );
}

#[test]
fn shipped_style_assets_parse() {
    for entry in std::fs::read_dir(crate::paths::FLAMES_STYLE_DIR).unwrap() {
        let path = entry.unwrap().path();
        if path.to_string_lossy().ends_with(".style.ron") {
            let content = std::fs::read_to_string(&path).unwrap();
            ron::from_str::<thyllore_effect_core::FlameStyle>(&content)
                .unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
        }
    }
}

#[test]
fn flame_set_combined_form() {
    let args: Vec<String> = vec!["--batch-flame-set=noise_amplitude=0.35".into()];
    let pairs = flame_set_resolve_from_args(&args).unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].0, "noise_amplitude");
    assert!((pairs[0].1 - 0.35).abs() < 1e-6);
}

#[test]
fn flame_set_separate_form() {
    let args: Vec<String> = vec!["--batch-flame-set".into(), "noise_amplitude=0.35".into()];
    let pairs = flame_set_resolve_from_args(&args).unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].0, "noise_amplitude");
    assert!((pairs[0].1 - 0.35).abs() < 1e-6);
}

#[test]
fn flame_set_unknown_key_error() {
    let args: Vec<String> = vec!["--batch-flame-set".into(), "invalid_key=1.0".into()];
    let err = flame_set_resolve_from_args(&args).unwrap_err();
    assert!(err.to_string().contains("invalid_key"),);
}

#[test]
fn apply_flame_overrides_no_panic_for_all_keys() {
    for key in flame_set_valid_keys() {
        let mut effect = FlameEffect::default();
        let overrides: Vec<(String, f32)> = vec![(key.to_string(), 1.0)];
        apply_flame_overrides(&mut effect, &overrides);
    }
}

/// Every key the pre-registry FLAME_SET_KEYS table accepted must keep working.
#[test]
fn flame_set_legacy_keys_stay_accepted() {
    let legacy_keys = [
        "warp_amp",
        "warp_freq",
        "rise_speed",
        "taper_power",
        "radius_tip_ratio",
        "edge_low",
        "edge_high",
        "white_boost",
        "bend_amount",
        "bend_power",
        "wind_x",
        "wind_z",
        "noise_amplitude",
        "noise_contrast",
        "noise_frequency",
        "noise_scroll_speed",
        "sigma_t",
        "intensity",
        "height",
        "radius",
        "time",
        "time_scale",
        "time_offset",
        "rot_z_deg",
        "temperature_base_k",
        "temperature_tip_k",
        "envelope_peak",
        "envelope_base",
        "envelope_tail",
        "radial_sharpness",
        "emitter_kind",
        "ring_major_radius",
        "ring_angular_speed",
        "noise_aniso_y",
        "warp_y_scale",
        "occlusion_lum_ref",
        "contour_wiggle_amp",
        "aniso_axis_advect",
        "rte_bands",
        "sigma_dispersion",
        "boundary_amp",
        "near_fade_radius",
        "carve_residual",
        "tip_carve_depth",
        "tip_carve_reach",
        "warp_reach",
        "swirl_gain",
        "swirl_speed",
        "spread_gain",
        "support_margin",
        "meander_amp",
        "meander_frequency",
        "mix_lo",
        "mix_hi",
        "mix_height_gain",
        "mix_scale",
        "mix_radial_gain",
        "density_exp",
        "temp_exp",
        "wien_c_k",
        "wave_segments",
        "boundary_freq",
        "boundary_speed",
        "boundary_radius_ratio",
        "edge_outer_sharpen",
        "noise_scale_mode",
        "erosion_noise_gain",
        "twist_gain",
        "twist_speed",
        "burnout_gain",
        "noise_shaping_scale",
        "optical_depth",
        "branch_period",
        "branch_life",
        "branch_gain",
        "branch_core_radius",
        "branch_core_offset",
        "branch_reach",
        "branch_spread",
        "branch_spawn_height",
        "branch_spawn_range",
        "branch_seed",
    ];
    let valid = flame_set_valid_keys();
    for key in legacy_keys {
        assert!(valid.contains(&key), "legacy key {key} no longer accepted");
    }
}

#[test]
fn batch_run_update_orbit_inserts_missing_transform() {
    let mut world = World::new();

    // Spawn an entity with only FlameEffect (no Transform)
    let e = world.spawn();
    world.insert_component(e, FlameEffect::default());

    // Insert BatchRun and BatchFlameOrbit resources
    world.insert_resource(BatchRun::new(PathBuf::from("/tmp/out.png"), 1, Vec::new()));
    world.resource_mut::<BatchRun>().frames_rendered = 1;
    world.insert_resource(crate::ecs::resource::BatchFlameOrbit {
        radius: 2.0,
        period_seconds: 4.0,
        initial: None,
    });

    // Call once: initializes `initial` from the (missing) Transform to (0,0,0)
    // and inserts MotionPath component for the flame entity
    batch_run_update_orbit(&mut world);

    // Assert that the entity now has a MotionPath component
    let motion_path = world.get_component::<crate::ecs::component::MotionPath>(e);
    assert!(
        motion_path.is_some(),
        "MotionPath should have been inserted"
    );

    let motion_path = motion_path.unwrap();
    assert_eq!(motion_path.center, cgmath::Vector3::new(0.0, 0.0, 0.0));
    assert!((motion_path.radius - 2.0).abs() < 1e-5);
    assert!(
        (motion_path.angular_speed - 2.0 * std::f32::consts::PI / 4.0).abs() < 1e-5,
        "angular_speed: got {}, expected {}",
        motion_path.angular_speed,
        2.0 * std::f32::consts::PI / 4.0
    );

    // Call sync_motion_paths to update Transform from MotionPath
    crate::ecs::systems::sync_motion_paths(&mut world);

    // Assert that the entity now has a Transform component (inserted by sync_motion_paths)
    let transform = world.get_component::<crate::ecs::world::Transform>(e);
    assert!(
        transform.is_some(),
        "Transform should have been inserted by sync_motion_paths"
    );

    let transform = transform.unwrap();
    let offset = compute_orbit_offset(2.0, 4.0, 1.0 / 60.0);

    assert!(
        (transform.translation.x - offset[0]).abs() < 1e-5,
        "translation.x: got {}, expected {}",
        transform.translation.x,
        offset[0]
    );
    assert!(
        (transform.translation.z - offset[2]).abs() < 1e-5,
        "translation.z: got {}, expected {}",
        transform.translation.z,
        offset[2]
    );
}

#[test]
fn test_flame_preset_resolve_valid() {
    let args = vec![String::from("--batch-flame-preset"), String::from("candle")];
    let result = flame_preset_resolve_from_args(&args).unwrap();
    assert_eq!(result, Some(String::from("candle")));
}

#[test]
fn test_flame_preset_then_override_order() {
    // "candle" preset sets height=0.28, radius=0.07, intensity=2.0, etc.
    let mut effect = FlameEffect::default();
    thyllore_effect_core::apply_flame_preset(&mut effect, "candle");

    // Now apply an individual override for height via flame_set
    let overrides: Vec<(String, f32)> = vec![(String::from("height"), 1.5)];
    apply_flame_overrides(&mut effect, &overrides);

    // The override should be final (1.5), not the preset value (0.28)
    assert!(
        (effect.height - 1.5).abs() < 1e-5,
        "height should be overridden to 1.5, got {}",
        effect.height
    );
    // Other candle preset values should remain
    assert!(
        (effect.radius - 0.07).abs() < 1e-5,
        "radius should still be candle's 0.07, got {}",
        effect.radius
    );
}

#[test]
fn test_orbit_motion_path_equivalence() {
    use crate::ecs::component::{motion_path_position, MotionPath};
    use std::f32::consts::PI;

    let center = cgmath::Vector3::new(1.0, 2.0, 3.0);
    let radius = 1.5;
    let period = 2.0;
    let path = MotionPath {
        center,
        radius,
        angular_speed: 2.0 * PI / period,
        phase_offset: 0.0,
        enabled: true,
    };

    for &t in &[0.0, 0.7, 1.9, 3.3] {
        let mp_pos = motion_path_position(&path, t);
        let offset = compute_orbit_offset(radius, period, t);
        let orbit_pos = cgmath::Vector3::new(
            center.x + offset[0],
            center.y + offset[1],
            center.z + offset[2],
        );

        assert!(
            (mp_pos.x - orbit_pos.x).abs() < 1e-5,
            "t={}: x diff {} (mp={}, orbit={})",
            t,
            (mp_pos.x - orbit_pos.x).abs(),
            mp_pos.x,
            orbit_pos.x
        );
        assert!(
            (mp_pos.y - orbit_pos.y).abs() < 1e-5,
            "t={}: y diff {} (mp={}, orbit={})",
            t,
            (mp_pos.y - orbit_pos.y).abs(),
            mp_pos.y,
            orbit_pos.y
        );
        assert!(
            (mp_pos.z - orbit_pos.z).abs() < 1e-5,
            "t={}: z diff {} (mp={}, orbit={})",
            t,
            (mp_pos.z - orbit_pos.z).abs(),
            mp_pos.z,
            orbit_pos.z
        );
    }
}

#[test]
fn flame_texture_fit_path_only_defaults_blend_to_one() {
    let resolved =
        flame_texture_fit_resolve_from_args(&args(&["bin", "--batch-flame-texture", "image.png"]))
            .unwrap()
            .unwrap();
    assert_eq!(resolved.0, "image.png");
    assert!((resolved.1 - 1.0).abs() < 1e-6);
    assert!(!resolved.2);
}

#[test]
fn flame_texture_fit_path_with_blend() {
    let resolved = flame_texture_fit_resolve_from_args(&args(&[
        "bin",
        "--batch-flame-texture",
        "image.png,0.4",
    ]))
    .unwrap()
    .unwrap();
    assert_eq!(resolved.0, "image.png");
    assert!((resolved.1 - 0.4).abs() < 1e-6);
    assert!(!resolved.2);
}

#[test]
fn flame_texture_fit_invalid_blend_is_err() {
    assert!(flame_texture_fit_resolve_from_args(&args(&[
        "bin",
        "--batch-flame-texture",
        "image.png,abc"
    ]))
    .is_err());
}

#[test]
fn flame_texture_fit_profile() {
    let resolved = flame_texture_fit_resolve_from_args(&args(&[
        "bin",
        "--batch-flame-texture",
        "image.png,0.5,profile",
    ]))
    .unwrap()
    .unwrap();
    assert_eq!(resolved.0, "image.png");
    assert!((resolved.1 - 0.5).abs() < 1e-6);
    assert!(resolved.2);
}

#[test]
fn anim_edit_specs_parse_all_forms() {
    let edits = anim_edits_resolve_from_args(&args(&[
        "bin",
        "--batch-anim-edit",
        "debug_keys=42",
        "--batch-anim-edit",
        "key=height@1.5=2.25",
        "--batch-anim-edit",
        "clear",
    ]))
    .unwrap();
    assert_eq!(edits[0], BatchAnimEdit::DebugKeys { seed: 42 });
    assert_eq!(
        edits[1],
        BatchAnimEdit::Key {
            property_type: crate::ecs::component::FlameParam::Height.property_type(),
            time: 1.5,
            value: 2.25
        }
    );
    assert_eq!(edits[2], BatchAnimEdit::Clear);
}

#[test]
fn anim_edit_invalid_specs_are_err() {
    for spec in [
        "debug_keys=abc",
        "key=height@1.5",
        "key=no_such_param@1.0=2.0",
        "key=height@-1.0=2.0",
        "bogus",
    ] {
        assert!(
            anim_edits_resolve_from_args(&args(&["bin", "--batch-anim-edit", spec])).is_err(),
            "{spec} should be rejected"
        );
    }
}

#[test]
fn debug_actions_parse_names_and_view_mode() {
    let actions = debug_actions_resolve_from_args(&args(&[
        "bin",
        "--batch-debug-action",
        "reset_camera",
        "--batch-debug-action",
        "view_mode=normal",
    ]))
    .unwrap();
    assert_eq!(actions[0], BatchDebugAction::ResetCamera);
    assert_eq!(
        actions[1],
        BatchDebugAction::ViewMode(crate::ecs::resource::DebugViewMode::Normal)
    );
    assert!(
        debug_actions_resolve_from_args(&args(&["bin", "--batch-debug-action", "bogus"])).is_err()
    );
}

#[test]
fn anim_edits_apply_and_dump_reflect_clip_state() {
    let mut world = World::new();
    crate::ecs::systems::spawn_flame(
        &mut world,
        crate::ecs::systems::DEFAULT_FLAME_NAME,
        FlameEffect::default(),
    );
    world.insert_resource(ClipLibrary::new());
    world.insert_resource(TimelineState::new());
    world.insert_resource(crate::ecs::resource::EditHistory::new(10));
    let mut assets = AssetStorage::new();

    batch_apply_anim_edits(
        &mut world,
        &mut assets,
        &[
            BatchAnimEdit::DebugKeys { seed: 7 },
            BatchAnimEdit::Key {
                property_type: crate::ecs::component::FlameParam::Height.property_type(),
                time: 9.0,
                value: 3.5,
            },
        ],
    );
    assert!(
        (world.resource::<TimelineState>().current_time).abs() < 1e-6,
        "key edit must restore timeline time"
    );

    let dump = batch_anim_dump_json(&world);
    let entities = dump["entities"].as_array().unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0]["domain"], "Flame");
    let clip_id = entities[0]["clip_id"]
        .as_u64()
        .expect("flame clip scheduled");
    let clips = dump["clips"].as_array().unwrap();
    let clip = clips
        .iter()
        .find(|c| c["id"].as_u64() == Some(clip_id))
        .expect("clip in dump");
    let curves = clip["scalar_curves"].as_array().unwrap();
    assert_eq!(
        curves.len(),
        crate::ecs::component::FLAME_DOMAIN.channels.len()
    );
    let height = curves
        .iter()
        .find(|c| c["property"] == "height")
        .expect("height curve");
    let keyframes = height["keyframes"].as_array().unwrap();
    assert_eq!(
        keyframes.len(),
        crate::ecs::systems::scalar_clip_systems::DEBUG_KEYS_PER_CURVE + 1
    );
    assert!(keyframes
        .iter()
        .any(|k| (k["time"].as_f64().unwrap() - 9.0).abs() < 1e-6
            && (k["value"].as_f64().unwrap() - 3.5).abs() < 1e-6));
    assert!((clip["duration"].as_f64().unwrap() - 9.0).abs() < 1e-6);
}

#[test]
fn flame_clip_preview_parses_and_rejects_invalid() {
    let actions = debug_actions_resolve_from_args(&args(&[
        "bin",
        "--batch-debug-action",
        "flame_clip_preview=3.5",
    ]))
    .unwrap();
    assert_eq!(
        actions[0],
        BatchDebugAction::FlameClipPreview { end_seconds: 3.5 }
    );
    for bad in ["flame_clip_preview=abc", "flame_clip_preview=-1"] {
        assert!(
            debug_actions_resolve_from_args(&args(&["bin", "--batch-debug-action", bad])).is_err(),
            "{bad} should be rejected"
        );
    }
}

#[test]
fn flame_clip_preview_sets_drag_preview_without_touching_instance() {
    let mut world = World::new();
    world.insert_resource(ClipLibrary::new());
    world.insert_resource(TimelineState::new());
    world.insert_resource(crate::ecs::resource::TimelineInteractionState::default());
    let mut assets = AssetStorage::new();
    let flame = crate::ecs::systems::spawn_flame_with_clip(
        &mut world,
        &mut assets,
        "Flame",
        FlameEffect::default(),
    );

    batch_apply_debug_actions(
        &world,
        &[BatchDebugAction::FlameClipPreview { end_seconds: 3.0 }],
    );

    let preview = world
        .resource::<crate::ecs::resource::TimelineInteractionState>()
        .drag_preview
        .expect("preview set");
    assert_eq!(preview.entity, flame);
    assert!((preview.start_time - 0.0).abs() < 1e-6);
    assert!((preview.end_time - 3.0).abs() < 1e-6);

    let instance = world
        .get_component::<ClipSchedule>(flame)
        .unwrap()
        .first_instance()
        .cloned()
        .unwrap();
    assert!(
        (instance.clip_out - 0.0).abs() < 1e-6,
        "preview must not commit the trim"
    );

    let dump = batch_anim_dump_json(&world);
    assert!(
        (dump["timeline"]["drag_preview"]["end_time"]
            .as_f64()
            .unwrap()
            - 3.0)
            .abs()
            < 1e-6
    );
}

#[test]
fn debug_actions_apply_sets_view_mode_and_queues_events() {
    let mut world = World::new();
    world.insert_resource(DebugViewState::default());
    world.insert_resource(UIEventQueue::new());
    batch_apply_debug_actions(
        &world,
        &[
            BatchDebugAction::ViewMode(crate::ecs::resource::DebugViewMode::Normal),
            BatchDebugAction::ResetCamera,
        ],
    );
    assert_eq!(
        world.resource::<DebugViewState>().debug_view_mode,
        crate::ecs::resource::DebugViewMode::Normal
    );
    let events: Vec<UIEvent> = world.resource_mut::<UIEventQueue>().drain().collect();
    assert!(matches!(events[0], UIEvent::ResetCamera));
}

#[test]
fn water_debug_dump_action_marks_the_batch_run_in_every_capture_mode() {
    let single = batch_run_resolve_from_args(&args(&[
        "bin",
        "--batch-screenshot",
        "out.png",
        "--batch-debug-action",
        "dump_water_debug",
    ]))
    .unwrap()
    .expect("single-shot batch");
    assert!(single.dump_water_debug);

    let sequence = batch_run_resolve_from_args(&args(&[
        "bin",
        "--batch-screenshot-sequence",
        "out,3,2",
        "--batch-debug-action",
        "dump_water_debug",
    ]))
    .unwrap()
    .expect("sequence batch");
    assert!(sequence.dump_water_debug);

    let without = batch_run_resolve_from_args(&args(&["bin", "--batch-screenshot", "out.png"]))
        .unwrap()
        .expect("batch without debug action");
    assert!(!without.dump_water_debug);
}

#[test]
fn water_debug_dump_action_still_queues_its_event_outside_a_batch_run() {
    let mut world = World::new();
    world.insert_resource(UIEventQueue::new());

    batch_apply_debug_actions(&world, &[BatchDebugAction::WaterDebugDump]);

    let events: Vec<UIEvent> = world.resource_mut::<UIEventQueue>().drain().collect();
    assert!(matches!(events[0], UIEvent::DumpWaterDebug));
}

#[test]
fn sequence_analyze_resolve_dir_only() {
    let args = args(&[
        "bin",
        "--batch-sequence-analyze",
        "data/flames",
        "--batch-sequence-dump",
        "out.json",
    ]);
    let result = batch_sequence_analyze_resolve_from_args(&args)
        .unwrap()
        .unwrap();
    assert_eq!(result.directories.len(), 1);
    assert_eq!(result.directories[0].0, "data/flames");
    assert_eq!(result.directories[0].1, None);
    assert_eq!(result.directories[0].2, None);
    assert_eq!(result.dump_path, "out.json");
}

#[test]
fn sequence_analyze_resolve_dir_with_range() {
    let args = args(&[
        "bin",
        "--batch-sequence-analyze",
        "data/flames,5,10",
        "--batch-sequence-dump",
        "out.json",
    ]);
    let result = batch_sequence_analyze_resolve_from_args(&args)
        .unwrap()
        .unwrap();
    assert_eq!(result.directories.len(), 1);
    assert_eq!(result.directories[0].0, "data/flames");
    assert_eq!(result.directories[0].1, Some(5));
    assert_eq!(result.directories[0].2, Some(10));
}

#[test]
fn sequence_analyze_resolve_multiple_dirs() {
    let args = args(&[
        "bin",
        "--batch-sequence-analyze",
        "data/a",
        "--batch-sequence-analyze",
        "data/b,1,5",
        "--batch-sequence-dump",
        "out.json",
    ]);
    let result = batch_sequence_analyze_resolve_from_args(&args)
        .unwrap()
        .unwrap();
    assert_eq!(result.directories.len(), 2);
    assert_eq!(result.directories[0].0, "data/a");
    assert_eq!(result.directories[1].0, "data/b");
    assert_eq!(result.directories[1].1, Some(1));
    assert_eq!(result.directories[1].2, Some(5));
}

#[test]
fn sequence_analyze_resolve_missing_dump() {
    let args = args(&["bin", "--batch-sequence-analyze", "data/flames"]);
    let result = batch_sequence_analyze_resolve_from_args(&args);
    assert!(result.is_err());
}

#[test]
fn sequence_analyze_resolve_invalid_range() {
    let args = args(&[
        "bin",
        "--batch-sequence-analyze",
        "data/flames,abc,10",
        "--batch-sequence-dump",
        "out.json",
    ]);
    let result = batch_sequence_analyze_resolve_from_args(&args);
    assert!(result.is_err());
}

#[test]
fn sequence_analyze_resolve_none_without_flag() {
    let args = args(&["bin", "--batch-screenshot", "data/flames"]);
    let result = batch_sequence_analyze_resolve_from_args(&args).unwrap();
    assert!(result.is_none());
}

#[test]
fn sequence_analyze_run_returns_none_without_flag() {
    let args: Vec<String> = vec![
        "bin".to_string(),
        "--batch-screenshot".to_string(),
        "data/flames".to_string(),
    ];
    let result = run_sequence_analyze_from_args(args);
    assert!(result.is_none());
}

#[test]
fn sequence_analyze_end_to_end() {
    let temp_dir = tempfile::tempdir().unwrap();
    let dir_path = temp_dir.path();

    // Write meta.json with custom fps
    let meta_path = dir_path.join("meta.json");
    std::fs::write(&meta_path, r#"{"fps": 30.0}"#).unwrap();

    // Write 3 dummy 2x2 RGB PNGs with distinct colors
    for i in 0..3 {
        let value = (i + 1) as u8 * 50; // 50, 100, 150
        let png_path = dir_path.join(format!("frame_{:04}.png", i));
        write_test_png(&png_path, 2, 2, value);
    }

    let dump_path = temp_dir.path().join("output.json");
    let args = vec![
        "bin".to_string(),
        "--batch-sequence-analyze".to_string(),
        dir_path.to_string_lossy().to_string(),
        "--batch-sequence-dump".to_string(),
        dump_path.to_string_lossy().to_string(),
    ];

    let result = run_sequence_analyze_from_args(args);
    assert!(result.is_some());
    let result = result.unwrap();
    assert!(result.is_ok(), "sequence analysis failed: {:?}", result);

    // Verify output JSON
    let content = std::fs::read_to_string(&dump_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(json.get("sequences").is_some());
    let sequences = json["sequences"].as_array().unwrap();
    assert_eq!(sequences.len(), 1);

    let entry = &sequences[0];
    assert!(entry.get("dir").is_some());
    assert!(entry.get("descriptors").is_some());
    let descriptors = &entry["descriptors"];
    assert!(descriptors.get("f1_width").is_some());
    assert!(descriptors.get("f2_rough").is_some());
    assert!(descriptors.get("meta").is_some());

    // Verify fps from meta.json is used
    let meta = &descriptors["meta"];
    assert!((meta["fps"].as_f64().unwrap() - 30.0).abs() < 1e-6);
}

#[test]
fn sequence_analyze_range_filter() {
    let temp_dir = tempfile::tempdir().unwrap();
    let dir_path = temp_dir.path();

    // Write meta.json
    std::fs::write(dir_path.join("meta.json"), r#"{"fps": 10.0}"#).unwrap();

    // Write 5 dummy 2x2 RGB PNGs
    for i in 0..5 {
        let value = (i + 1) as u8 * 30;
        let png_path = dir_path.join(format!("frame_{:04}.png", i));
        write_test_png(&png_path, 2, 2, value);
    }

    let dump_path = temp_dir.path().join("output.json");
    let args = vec![
        "bin".to_string(),
        "--batch-sequence-analyze".to_string(),
        format!("{},1,3", dir_path.to_string_lossy()),
        "--batch-sequence-dump".to_string(),
        dump_path.to_string_lossy().to_string(),
    ];

    let result = run_sequence_analyze_from_args(args);
    assert!(result.is_some());
    let result = result.unwrap();
    assert!(result.is_ok(), "sequence analysis failed: {:?}", result);

    let content = std::fs::read_to_string(&dump_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    let sequences = json["sequences"].as_array().unwrap();
    assert_eq!(sequences.len(), 1);

    // Verify frame count in meta (should be 3 frames: 1, 2, 3)
    let meta = &sequences[0]["descriptors"]["meta"];
    assert_eq!(meta["frame_count"].as_u64().unwrap(), 3);
}

#[test]
fn sequence_analyze_jpg_error() {
    let temp_dir = tempfile::tempdir().unwrap();
    let dir_path = temp_dir.path();

    // Write a fake JPG file
    std::fs::write(dir_path.join("frame_0001.jpg"), b"fake jpg").unwrap();

    let dump_path = temp_dir.path().join("output.json");
    let args = vec![
        "bin".to_string(),
        "--batch-sequence-analyze".to_string(),
        dir_path.to_string_lossy().to_string(),
        "--batch-sequence-dump".to_string(),
        dump_path.to_string_lossy().to_string(),
    ];

    let result = run_sequence_analyze_from_args(args);
    assert!(result.is_some());
    let result = result.unwrap();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("JPG") || err_msg.contains("jpg"));
}

/// Write a simple 2x2 RGB PNG with all pixels having the same color value.
fn write_test_png(path: &Path, width: u32, height: u32, value: u8) {
    let file = std::fs::File::create(path).unwrap();
    let writer = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    let mut pixels = vec![value; (width * height * 3) as usize];
    writer.write_image_data(&pixels).unwrap();
    writer.finish().unwrap();
}
