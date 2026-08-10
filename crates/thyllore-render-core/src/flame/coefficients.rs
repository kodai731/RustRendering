use super::*;
use thyllore_math_core::{
    fit_chebyshev, integrate_chebyshev, pack_coefficients_vec4, parametric_height_falloff,
};

pub const HEIGHT_PRIMITIVE_COEFFICIENT_COUNT: usize = 12;
pub const HEIGHT_COEFFICIENT_COUNT: usize = 8;
pub const RADIAL_COEFFICIENT_COUNT: usize = 8;

pub struct FlameProfile {
    pub sigma_t: f32,
    pub height_falloff: Box<dyn Fn(f64) -> f64>,
    pub radial_falloff: Box<dyn Fn(f64) -> f64>,
}

impl Default for FlameProfile {
    fn default() -> Self {
        Self {
            sigma_t: 1.0,
            height_falloff: Box::new(default_height_falloff),
            radial_falloff: Box::new(default_radial_falloff),
        }
    }
}

pub fn default_height_falloff(height01: f64) -> f64 {
    parametric_height_falloff(height01, 0.25, 0.05, 1.25)
}

pub fn default_radial_falloff(radius01: f64) -> f64 {
    biweight_radial_falloff(radius01, 4.0)
}

/// Compact-support biweight radial falloff with the support radius derived from sharpness.
fn biweight_radial_falloff(radius01: f64, radial_sharpness: f32) -> f64 {
    let support = crate::flame_radial::flame_radial_support_radius(radial_sharpness) as f64;
    let inside = (1.0 - (radius01 / support) * (radius01 / support)).max(0.0);
    inside * inside
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameCoefficients {
    pub height_primitive: [[f32; 4]; 3],
    pub radial: [[f32; 4]; 2],
    pub height: [[f32; 4]; 2],
    pub radius_scale: [[f32; 4]; 2],
}

impl Default for FlameCoefficients {
    fn default() -> Self {
        fit_flame_coefficients(&FlameProfile::default())
    }
}

pub fn fit_flame_coefficients(profile: &FlameProfile) -> FlameCoefficients {
    let height_primitive_source = fit_chebyshev(
        &profile.height_falloff,
        (0.0, 1.0),
        HEIGHT_PRIMITIVE_COEFFICIENT_COUNT - 1,
    );
    let height_primitive = integrate_chebyshev(&height_primitive_source);
    let height_series = fit_chebyshev(
        &profile.height_falloff,
        (0.0, 1.0),
        HEIGHT_COEFFICIENT_COUNT,
    );
    let radial_series = fit_chebyshev(
        &profile.radial_falloff,
        (0.0, 1.0),
        RADIAL_COEFFICIENT_COUNT,
    );

    let height_primitive_slots = pack_coefficients_vec4(&height_primitive, 3);
    let height_slots = pack_coefficients_vec4(&height_series, 2);
    let radial_slots = pack_coefficients_vec4(&radial_series, 2);
    FlameCoefficients {
        height_primitive: [
            height_primitive_slots[0],
            height_primitive_slots[1],
            height_primitive_slots[2],
        ],
        radial: [radial_slots[0], radial_slots[1]],
        height: [height_slots[0], height_slots[1]],
        radius_scale: [[0.0; 4]; 2],
    }
}
pub fn profile_from_effect(effect: &FlameEffect, baked: &FlameBaked) -> FlameProfile {
    let peak = effect.envelope_peak as f64;
    let base = effect.envelope_base as f64;
    let tail = effect.envelope_tail as f64;
    let radial_sharpness = effect.radial_sharpness;
    let baked_envelope = baked.envelope;
    let baked_blend = baked.blend;
    FlameProfile {
        sigma_t: effect.sigma_t,
        height_falloff: Box::new(move |h: f64| {
            if let Some(ref envelope) = baked_envelope {
                if baked_blend > 0.0 {
                    let parametric = parametric_height_falloff(h, peak, base, tail);
                    let lut_value = lut_lerp33(envelope, h);
                    let baked_blend_f64 = baked_blend as f64;
                    return (1.0 - baked_blend_f64) * parametric + baked_blend_f64 * lut_value;
                }
            }
            parametric_height_falloff(h, peak, base, tail)
        }),
        radial_falloff: Box::new(move |r: f64| biweight_radial_falloff(r, radial_sharpness)),
    }
}

fn lut_lerp33(lut: &[f32; 33], h: f64) -> f64 {
    let h = h.max(0.0).min(1.0);
    let idx = h * 32.0;
    let i = idx.floor() as usize;
    let frac = idx - i as f64;
    let i = i.min(31);
    let v0 = lut[i] as f64;
    let v1 = lut[i + 1] as f64;
    v0 + frac * (v1 - v0)
}

pub fn refresh_flame_coefficients(effect: &mut FlameEffect, baked: &FlameBaked) {
    effect.coefficients = fit_flame_coefficients(&profile_from_effect(effect, baked));

    let baked_radius = baked.radius;
    let baked_blend = baked.blend;
    let radius_tip_ratio = effect.radius_tip_ratio as f64;
    let taper_power = effect.taper_power as f64;

    if baked_radius.is_some() && baked_blend > 0.0 {
        let blend = baked_blend as f64;
        let lut = baked_radius.unwrap();
        let rel: Box<dyn Fn(f64) -> f64> = Box::new(move |h: f64| {
            (1.0 - blend) * (1.0 + (radius_tip_ratio - 1.0) * h.powf(taper_power))
                + blend * lut_lerp33(&lut, h)
        });
        let series = fit_chebyshev(&*rel, (0.0, 1.0), 8);
        let slots = pack_coefficients_vec4(&series, 2);
        effect.coefficients.radius_scale[0] = slots[0];
        effect.coefficients.radius_scale[1] = slots[1];
    } else {
        let rel: Box<dyn Fn(f64) -> f64> =
            Box::new(move |h: f64| 1.0 + (radius_tip_ratio - 1.0) * h.powf(taper_power));
        let series = fit_chebyshev(&*rel, (0.0, 1.0), 8);
        let slots = pack_coefficients_vec4(&series, 2);
        effect.coefficients.radius_scale[0] = slots[0];
        effect.coefficients.radius_scale[1] = slots[1];
    }
}
