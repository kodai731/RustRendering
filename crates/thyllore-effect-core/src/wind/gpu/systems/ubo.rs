use crate::wind::analytic::WindShellParams;
use crate::wind::{build_wind_model_matrix, WindTornadoEffect, WindUBO};
use cgmath::{Matrix4, SquareMatrix};

pub fn build_wind_ubo(effect: &WindTornadoEffect) -> WindUBO {
    let model = build_wind_model_matrix(effect);
    let inverse_model = model.invert().unwrap_or(Matrix4::identity());
    let params = WindShellParams::from_effect(effect);

    WindUBO {
        model,
        inverse_model,
        shape: [
            params.height,
            params.wall_radius_base,
            params.wall_radius_slope,
            params.wall_width_q,
        ],
        core: [
            params.core_radius_sq,
            params.core_strength,
            params.wall_strength,
            params.top_fade,
        ],
        optics: [
            params.sigma_t,
            effect.ambient_brightness,
            effect.time,
            params.h_top,
        ],
        albedo: [
            effect.albedo[0],
            effect.albedo[1],
            effect.albedo[2],
            params.spread_offset,
        ],
        inv_view_proj: Matrix4::identity(),
    }
}

impl Default for WindUBO {
    fn default() -> Self {
        build_wind_ubo(&WindTornadoEffect::default())
    }
}
