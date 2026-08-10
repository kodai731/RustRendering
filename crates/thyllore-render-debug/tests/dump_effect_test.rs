use thyllore_render_debug::dump_effect::{camera_from_dump, effect_from_dump};

fn load_sample() -> serde_json::Value {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/wall_probe_sample.json"
    ))
    .expect("read wall_probe_sample.json");
    serde_json::from_slice(&bytes).expect("parse JSON")
}

#[test]
fn test_effect_from_dump_values() {
    let dump = load_sample();
    let flame = &dump["flames"][0];
    let effect = effect_from_dump(flame);

    // Verify key values from the sample dump
    assert!(
        (effect.noise_amplitude - 1.6).abs() < 1e-4,
        "noise_amplitude={}",
        effect.noise_amplitude
    );
    assert!(
        (effect.noise_frequency - 6.0).abs() < 1e-4,
        "noise_frequency={}",
        effect.noise_frequency
    );
    assert_eq!(
        effect.emitter_kind, 1,
        "emitter_kind={}",
        effect.emitter_kind
    );
    assert!(
        (effect.edge_low - 0.27).abs() < 1e-4,
        "edge_low={}",
        effect.edge_low
    );
    assert!(
        (effect.time - 2.1973181).abs() < 1e-5,
        "time={}",
        effect.time
    );

    // Verify restored effect matches specific scalar values from the dump
    assert!(
        (effect.warp_amp - 1.4).abs() < 1e-4,
        "warp_amp={}",
        effect.warp_amp
    );
    assert!(
        (effect.noise_scroll_speed - 1.0).abs() < 1e-4,
        "noise_scroll_speed={}",
        effect.noise_scroll_speed
    );
    assert!(
        (effect.rise_speed - 1.5).abs() < 1e-4,
        "rise_speed={}",
        effect.rise_speed
    );
    assert!(
        (effect.taper_power - 1.4).abs() < 1e-4,
        "taper_power={}",
        effect.taper_power
    );
    assert!(
        (effect.radial_sharpness - 4.0).abs() < 1e-4,
        "radial_sharpness={}",
        effect.radial_sharpness
    );
    assert!(
        (effect.ring_major_radius - 1.5).abs() < 1e-4,
        "ring_major_radius={}",
        effect.ring_major_radius
    );
}

#[test]
fn test_camera_from_dump() {
    let dump = load_sample();
    let cam = camera_from_dump(&dump);

    assert!((cam.position[0] + 0.1801402).abs() < 1e-6);
    assert!((cam.fov_y_degrees - 45.0).abs() < 1e-6);
    assert!((cam.viewport_size_px[0] - 1994.0).abs() < 1e-6);
    assert!((cam.viewport_size_px[1] - 855.0).abs() < 1e-6);
}

/// Drift detection: list every key in flames[0] of the sample dump, subtract the ones
/// that effect_from_dump reads and the allowlisted ignored keys, assert the remainder is empty.
/// If a new key is added to the dump format but not consumed by the reader (and not on the
/// allowlist), this test will fail — catching reader update omissions.
#[test]
fn test_no_unread_keys() {
    let dump = load_sample();
    let flame = &dump["flames"][0];

    let flame_obj = flame.as_object().expect("flames[0] is an object");

    // Keys that effect_from_dump reads from the JSON
    static READ_KEYS: &[&str] = &[
        "position",
        "height",
        "radius",
        "sigma_t",
        "intensity",
        "noise_amplitude",
        "noise_frequency",
        "time",
        "edge_low",
        "edge_high",
        "white_boost",
        "bend_amount",
        "bend_power",
        "occlusion_lum_ref",
        "aniso_axis_advect",
        "rte_bands",
        "edge_temperature_blend",
        "boundary_amp",
        "boundary_freq",
        "boundary_speed",
        "boundary_radius_ratio",
        "baked_blend",
        "emitter_kind",
        "use_blackbody",
        "color_base",
        "color_tip",
        "temperature_base_k",
        "temperature_tip_k",
        "light_position_world",
        "self_shadow_strength",
        "coefficients",
        "warp_amp",
        "warp_freq",
        "warp_y_scale",
        "noise_scroll_speed",
        "noise_aniso_y",
        "rise_speed",
        "taper_power",
        "radius_tip_ratio",
        "contour_wiggle_amp",
        "sigma_dispersion",
        "radial_sharpness",
        "envelope_base",
        "envelope_peak",
        "envelope_tail",
        "ring_angular_speed",
        "ring_major_radius",
        "time_scale",
        "time_offset",
        "temporal_weight",
        "frame_index",
        "rotation",
        "wind_direction",
    ];

    // Keys that are in the dump but intentionally not consumed by the reader
    static ALLOWLIST: &[&str] = &[
        "wall_probe",
        "baked_envelope",
        "baked_radius",
        "baked_color",
        "kernel_blob_amp",
        "kernel_blob_size",
        "turbulence_model",
    ];

    let mut unread: Vec<String> = flame_obj
        .keys()
        .filter(|k| !READ_KEYS.contains(&k.as_str()) && !ALLOWLIST.contains(&k.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    unread.sort();

    assert!(
        unread.is_empty(),
        "flames[0] has keys not read by effect_from_dump and not on the allowlist: {:?}",
        unread
    );
}
