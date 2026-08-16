use crate::flame::*;
use cgmath::Vector2;

/// Rising warp of the noise coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameWarp {
    pub amp: f32,
    pub freq: f32,
    pub rise_speed: f32,
    pub taper_power: f32,
    pub y_scale: f32,
    /// Penetration depth of the tip-asymptotic warp strain, in the same
    /// remaining-luminous-fraction units as the tip carve reach.
    pub reach: f32,
}

impl Default for FlameWarp {
    fn default() -> Self {
        Self {
            amp: 1.4,
            freq: 5.0,
            rise_speed: 1.5,
            taper_power: 1.4,
            y_scale: 0.6,
            reach: crate::flame_wave::WARP_REACH_DEFAULT,
        }
    }
}

/// Horizontal wind bending the column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameWind {
    pub direction: Vector2<f32>,
    pub bend_amount: f32,
    pub bend_power: f32,
}

impl Default for FlameWind {
    fn default() -> Self {
        Self {
            direction: Vector2::new(0.0, 0.0),
            bend_amount: 0.0,
            bend_power: 1.7,
        }
    }
}

pub fn build_warp_style(warp: &FlameWarp) -> FlameWarpStyle {
    FlameWarpStyle {
        warp_amp: warp.amp,
        warp_freq: warp.freq,
        rise_speed: warp.rise_speed,
        taper_power: warp.taper_power,
    }
}

pub fn build_wind_bend(wind: &FlameWind) -> FlameWindBend {
    FlameWindBend {
        wind_direction: [wind.direction.x, wind.direction.y],
        bend_amount: wind.bend_amount,
        bend_power: wind.bend_power,
    }
}
