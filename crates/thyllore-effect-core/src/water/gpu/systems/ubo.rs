use crate::water::*;
use cgmath::{Matrix4, SquareMatrix};

pub fn build_water_ubo(effect: &WaterTorusEffect) -> WaterUBO {
    let model = build_water_model_matrix(effect);
    let inverse_model = model.invert().unwrap_or(Matrix4::identity());

    WaterUBO {
        model,
        inverse_model,
        radii: [effect.major_radius, effect.minor_radius, 0.0, 0.0],
        absorption: [
            effect.absorption[0],
            effect.absorption[1],
            effect.absorption[2],
            effect.ior,
        ],
        flow: [
            effect.flow_longitudinal,
            effect.flow_meridional,
            effect.time,
            0.0,
        ],
        composite: [effect.reflect_strength, effect.refract_strength, 0.0, 0.0],
        tint: [effect.tint[0], effect.tint[1], effect.tint[2], 0.0],
        temporal: [0.0, 0.0, 0.0, 0.0],
    }
}
