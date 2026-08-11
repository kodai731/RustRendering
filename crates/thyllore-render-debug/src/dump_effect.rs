use cgmath::{Vector2, Vector3};
use thyllore_effect_core::{
    refresh_flame_coefficients, FlameBaked, FlameCoefficients, FlameEffect, FlameTemporalAccum,
};

/// Camera description reconstructed from a flame dump JSON.
#[derive(Debug)]
pub struct DumpCamera {
    pub position: [f32; 3],
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub fov_y_degrees: f32,
    pub viewport_size_px: [f32; 2],
}

fn read_array3(v: &serde_json::Value) -> Option<[f32; 3]> {
    let arr = v.as_array()?;
    if arr.len() < 3 {
        return None;
    }
    Some([
        arr[0].as_f64()? as f32,
        arr[1].as_f64()? as f32,
        arr[2].as_f64()? as f32,
    ])
}

/// Reconstruct a [`FlameEffect`] plus its baked/temporal state from the
/// `flames[0]` entry of a flame dump JSON.
///
/// Starts from defaults and overwrites every field whose key appears in the
/// JSON. Fields missing from the dump (e.g. `near_fade_radius`,
/// `carve_residual`) keep their default values.
pub fn flame_from_dump(
    flame_json: &serde_json::Value,
) -> (FlameEffect, FlameBaked, FlameTemporalAccum) {
    let mut effect = FlameEffect::default();
    let mut baked = FlameBaked::default();
    let mut temporal = FlameTemporalAccum::default();

    // position
    if let Some(pos) = read_array3(flame_json.get("position").unwrap()) {
        effect.position = Vector3::new(pos[0], pos[1], pos[2]);
    }

    // scalar fields
    macro_rules! apply_scalar {
        ($field:ident, $json_field:expr) => {
            if let Some(v) = flame_json.get($json_field).and_then(|f| f.as_f64()) {
                effect.$field = v as f32;
            }
        };
    }

    apply_scalar!(height, "height");
    apply_scalar!(radius, "radius");
    apply_scalar!(sigma_t, "sigma_t");
    apply_scalar!(intensity, "intensity");
    apply_scalar!(noise_amplitude, "noise_amplitude");
    apply_scalar!(noise_contrast, "noise_contrast");
    apply_scalar!(noise_frequency, "noise_frequency");
    apply_scalar!(time, "time");
    apply_scalar!(edge_low, "edge_low");
    apply_scalar!(edge_high, "edge_high");
    apply_scalar!(white_boost, "white_boost");
    apply_scalar!(bend_amount, "bend_amount");
    apply_scalar!(bend_power, "bend_power");
    apply_scalar!(occlusion_lum_ref, "occlusion_lum_ref");
    apply_scalar!(aniso_axis_advect, "aniso_axis_advect");
    apply_scalar!(rte_bands, "rte_bands");
    apply_scalar!(edge_temperature_blend, "edge_temperature_blend");
    apply_scalar!(boundary_amp, "boundary_amp");
    apply_scalar!(boundary_freq, "boundary_freq");
    apply_scalar!(boundary_speed, "boundary_speed");
    apply_scalar!(boundary_radius_ratio, "boundary_radius_ratio");
    apply_scalar!(tip_carve_depth, "tip_carve_depth");
    apply_scalar!(tip_carve_reach, "tip_carve_reach");
    apply_scalar!(warp_reach, "warp_reach");
    if let Some(v) = flame_json.get("baked_blend").and_then(|f| f.as_f64()) {
        baked.blend = v as f32;
    }
    apply_scalar!(warp_amp, "warp_amp");
    apply_scalar!(warp_freq, "warp_freq");
    apply_scalar!(warp_y_scale, "warp_y_scale");
    apply_scalar!(noise_scroll_speed, "noise_scroll_speed");
    apply_scalar!(noise_aniso_y, "noise_aniso_y");
    apply_scalar!(rise_speed, "rise_speed");
    apply_scalar!(taper_power, "taper_power");
    apply_scalar!(radius_tip_ratio, "radius_tip_ratio");
    apply_scalar!(contour_wiggle_amp, "contour_wiggle_amp");
    apply_scalar!(sigma_dispersion, "sigma_dispersion");
    apply_scalar!(radial_sharpness, "radial_sharpness");
    apply_scalar!(envelope_base, "envelope_base");
    apply_scalar!(envelope_peak, "envelope_peak");
    apply_scalar!(envelope_tail, "envelope_tail");
    apply_scalar!(ring_angular_speed, "ring_angular_speed");
    apply_scalar!(ring_major_radius, "ring_major_radius");
    apply_scalar!(time_scale, "time_scale");
    apply_scalar!(time_offset, "time_offset");
    if let Some(v) = flame_json.get("temporal_weight").and_then(|f| f.as_f64()) {
        temporal.weight = v as f32;
    }
    if let Some(v) = flame_json.get("frame_index").and_then(|f| f.as_u64()) {
        temporal.frame_index = v;
    }

    // rotation [w, x, y, z] (4-element array)
    if let Some(rot) = flame_json.get("rotation").and_then(|v| v.as_array()) {
        if rot.len() >= 4 {
            effect.rotation = cgmath::Quaternion::new(
                rot[0].as_f64().unwrap() as f32,
                rot[1].as_f64().unwrap() as f32,
                rot[2].as_f64().unwrap() as f32,
                rot[3].as_f64().unwrap() as f32,
            );
        }
    }

    // wind_direction [x, y] (2-element array)
    if let Some(wd) = flame_json.get("wind_direction").and_then(|v| v.as_array()) {
        if wd.len() >= 2 {
            effect.wind_direction = Vector2::new(
                wd[0].as_f64().unwrap() as f32,
                wd[1].as_f64().unwrap() as f32,
            );
        }
    }

    // emitter_kind from integer
    if let Some(v) = flame_json.get("emitter_kind").and_then(|f| f.as_i64()) {
        effect.emitter_kind = v as u32;
    }

    // use_blackbody from boolean
    if let Some(v) = flame_json.get("use_blackbody").and_then(|f| f.as_bool()) {
        effect.use_blackbody = v;
    }

    // color_base / color_tip (array of 3 floats)
    if let Some(cb) = read_array3(flame_json.get("color_base").unwrap()) {
        effect.color_base = cb;
    }
    if let Some(ct) = read_array3(flame_json.get("color_tip").unwrap()) {
        effect.color_tip = ct;
    }

    // temperature fields
    apply_scalar!(temperature_base_k, "temperature_base_k");
    apply_scalar!(temperature_tip_k, "temperature_tip_k");

    // light_position_world
    if let Some(lp) = read_array3(flame_json.get("light_position_world").unwrap()) {
        effect.light_position_world = Vector3::new(lp[0], lp[1], lp[2]);
    }

    // self_shadow_strength
    apply_scalar!(self_shadow_strength, "self_shadow_strength");

    // coefficients from JSON (if present)
    if let Some(coeffs_json) = flame_json.get("coefficients") {
        let mut coeffs = FlameCoefficients::default();
        if let Some(arr) = coeffs_json
            .get("height_primitive")
            .and_then(|a| a.as_array())
        {
            for (i, row) in arr.iter().enumerate().take(3) {
                if let Some(inner) = row.as_array() {
                    for (j, val) in inner.iter().enumerate().take(4) {
                        if let Some(v) = val.as_f64() {
                            coeffs.height_primitive[i][j] = v as f32;
                        }
                    }
                }
            }
        }
        if let Some(arr) = coeffs_json.get("radial").and_then(|a| a.as_array()) {
            for (i, row) in arr.iter().enumerate().take(2) {
                if let Some(inner) = row.as_array() {
                    for (j, val) in inner.iter().enumerate().take(4) {
                        if let Some(v) = val.as_f64() {
                            coeffs.radial[i][j] = v as f32;
                        }
                    }
                }
            }
        }
        if let Some(arr) = coeffs_json.get("height").and_then(|a| a.as_array()) {
            for (i, row) in arr.iter().enumerate().take(2) {
                if let Some(inner) = row.as_array() {
                    for (j, val) in inner.iter().enumerate().take(4) {
                        if let Some(v) = val.as_f64() {
                            coeffs.height[i][j] = v as f32;
                        }
                    }
                }
            }
        }
        if let Some(arr) = coeffs_json.get("radius_scale").and_then(|a| a.as_array()) {
            for (i, row) in arr.iter().enumerate().take(2) {
                if let Some(inner) = row.as_array() {
                    for (j, val) in inner.iter().enumerate().take(4) {
                        if let Some(v) = val.as_f64() {
                            coeffs.radius_scale[i][j] = v as f32;
                        }
                    }
                }
            }
        }
        effect.coefficients = coeffs;
    }

    refresh_flame_coefficients(&mut effect, &baked);
    (effect, baked, temporal)
}

pub fn effect_from_dump(flame_json: &serde_json::Value) -> FlameEffect {
    flame_from_dump(flame_json).0
}

/// Reconstruct a [`DumpCamera`] from the top-level `camera` and `viewport_size_px` of a dump.
pub fn camera_from_dump(dump: &serde_json::Value) -> DumpCamera {
    let cam = dump.get("camera").expect("missing camera");
    let viewport = dump
        .get("viewport_size_px")
        .expect("missing viewport_size_px");

    let position: [f32; 3] = read_array3(cam.get("position").unwrap()).unwrap();
    let forward: [f32; 3] = read_array3(cam.get("forward").unwrap()).unwrap();
    let right: [f32; 3] = read_array3(cam.get("right").unwrap()).unwrap();
    let up: [f32; 3] = read_array3(cam.get("up").unwrap()).unwrap();
    let fov_y_degrees = cam.get("fov_y_degrees").unwrap().as_f64().unwrap() as f32;

    let vp = viewport
        .as_array()
        .expect("viewport_size_px is not an array");
    let viewport_size_px: [f32; 2] = [
        vp[0].as_f64().unwrap() as f32,
        vp[1].as_f64().unwrap() as f32,
    ];

    DumpCamera {
        position,
        forward,
        right,
        up,
        fov_y_degrees,
        viewport_size_px,
    }
}
