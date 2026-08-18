use crate::flame::*;
use thyllore_color_core::blackbody_rgb;

/// Emission color: either the authored base/tip pair or a blackbody pair
/// sampled from the base/tip temperatures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameColor {
    pub base: [f32; 3],
    pub tip: [f32; 3],
    pub temperature_base_k: f32,
    pub temperature_tip_k: f32,
    pub use_blackbody: bool,
    pub occlusion_lum_ref: f32,
}

impl Default for FlameColor {
    fn default() -> Self {
        Self {
            base: [1.0, 0.45, 0.1],
            tip: [1.0, 0.1, 0.02],
            temperature_base_k: 3200.0,
            temperature_tip_k: 1500.0,
            use_blackbody: true,
            occlusion_lum_ref: 1.0,
        }
    }
}

/// (base, mid, tip) linear colors of the legacy 3-point ramp.
pub fn resolve_flame_colors(color: &FlameColor) -> ([f32; 3], [f32; 3], [f32; 3]) {
    if color.use_blackbody {
        let mid_temp = (color.temperature_base_k + color.temperature_tip_k) / 2.0;
        return (
            blackbody_rgb(color.temperature_base_k),
            blackbody_rgb(mid_temp),
            blackbody_rgb(color.temperature_tip_k),
        );
    }
    let mid = std::array::from_fn(|i| (color.base[i] + color.tip[i]) / 2.0);
    (color.base, mid, color.tip)
}

/// Planckian chromaticity sampled from the tip temperature (index 0) to the
/// base temperature (index 7).
pub fn build_temperature_ramp(color: &FlameColor) -> [[f32; 4]; 8] {
    let cold = color.temperature_tip_k;
    let hot = color.temperature_base_k;
    std::array::from_fn(|index| {
        let kelvin = cold + (hot - cold) * index as f32 / 7.0;
        let rgb = blackbody_rgb(kelvin);
        [rgb[0], rgb[1], rgb[2], 1.0]
    })
}

/// Legacy 3-point ramp blended toward the baked ramp by `baked.blend`; all zero
/// when nothing is baked.
pub fn build_color_ramp(color: &FlameColor, baked_state: &FlameBaked) -> [[f32; 4]; 8] {
    let baked = match baked_state.color {
        Some(ref b) if baked_state.blend > 0.0 => b,
        _ => return [[0.0; 4]; 8],
    };
    let blend = baked_state.blend;
    let (base, mid, tip) = resolve_flame_colors(color);

    std::array::from_fn(|i| {
        let h = (i as f32 + 0.5) / 8.0;
        let (from, to, t) = if h < 0.5 {
            (base, mid, h * 2.0)
        } else {
            (mid, tip, (h - 0.5) * 2.0)
        };
        let legacy: [f32; 3] = std::array::from_fn(|c| from[c] + (to[c] - from[c]) * t);
        [
            legacy[0] + (baked[i][0] - legacy[0]) * blend,
            legacy[1] + (baked[i][1] - legacy[1]) * blend,
            legacy[2] + (baked[i][2] - legacy[2]) * blend,
            0.0,
        ]
    })
}
