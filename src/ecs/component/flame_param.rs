use thyllore_anim_core::editable::PropertyType;

use super::flame::FlameEffect;
use super::scalar_channel::{ScalarChannel, ScalarChannelDomain};
use crate::ecs::world::{Entity, World};

/// Flame's animatable channels. Everything outside the flame domain (curve
/// editor, events, batch CLI, scene files) addresses them through the generic
/// `ScalarChannel` registry; this enum only backs the flame-side field access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlameParam {
    Height,
    Radius,
    Intensity,
    SigmaT,
    TemperatureBaseK,
    TemperatureTipK,
    WarpAmp,
    WarpFreq,
    RiseSpeed,
    NoiseAmplitude,
    WhiteBoost,
    BendAmount,
    WindX,
    WindZ,
    EdgeLow,
    EdgeHigh,
}

impl FlameParam {
    pub const ALL: [FlameParam; 16] = [
        FlameParam::Height,
        FlameParam::Radius,
        FlameParam::Intensity,
        FlameParam::SigmaT,
        FlameParam::TemperatureBaseK,
        FlameParam::TemperatureTipK,
        FlameParam::WarpAmp,
        FlameParam::WarpFreq,
        FlameParam::RiseSpeed,
        FlameParam::NoiseAmplitude,
        FlameParam::WhiteBoost,
        FlameParam::BendAmount,
        FlameParam::WindX,
        FlameParam::WindZ,
        FlameParam::EdgeLow,
        FlameParam::EdgeHigh,
    ];

    /// Stable scalar-curve code persisted in clip files (`PropertyType::Custom(code)`).
    /// Never reorder or reuse codes. Flame owns the 0..=15 block.
    pub const fn code(self) -> u16 {
        match self {
            FlameParam::Height => 0,
            FlameParam::Radius => 1,
            FlameParam::Intensity => 2,
            FlameParam::SigmaT => 3,
            FlameParam::TemperatureBaseK => 4,
            FlameParam::TemperatureTipK => 5,
            FlameParam::WarpAmp => 6,
            FlameParam::WarpFreq => 7,
            FlameParam::RiseSpeed => 8,
            FlameParam::NoiseAmplitude => 9,
            FlameParam::WhiteBoost => 10,
            FlameParam::BendAmount => 11,
            FlameParam::WindX => 12,
            FlameParam::WindZ => 13,
            FlameParam::EdgeLow => 14,
            FlameParam::EdgeHigh => 15,
        }
    }

    pub fn from_code(code: u16) -> Option<FlameParam> {
        FlameParam::ALL.iter().copied().find(|p| p.code() == code)
    }

    pub const fn property_type(self) -> PropertyType {
        PropertyType::Custom(self.code())
    }

    pub fn from_property_type(property_type: PropertyType) -> Option<FlameParam> {
        match property_type {
            PropertyType::Custom(code) => FlameParam::from_code(code),
            _ => None,
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            FlameParam::Height => "Height",
            FlameParam::Radius => "Radius",
            FlameParam::Intensity => "Intensity",
            FlameParam::SigmaT => "Sigma T",
            FlameParam::TemperatureBaseK => "Temp Base K",
            FlameParam::TemperatureTipK => "Temp Tip K",
            FlameParam::WarpAmp => "Warp Amp",
            FlameParam::WarpFreq => "Warp Freq",
            FlameParam::RiseSpeed => "Rise Speed",
            FlameParam::NoiseAmplitude => "Noise Amp",
            FlameParam::WhiteBoost => "White Boost",
            FlameParam::BendAmount => "Bend",
            FlameParam::WindX => "Wind X",
            FlameParam::WindZ => "Wind Z",
            FlameParam::EdgeLow => "Edge Low",
            FlameParam::EdgeHigh => "Edge High",
        }
    }

    /// Stable snake_case identifier used by batch CLI flags and anim dumps.
    pub const fn cli_name(self) -> &'static str {
        match self {
            FlameParam::Height => "height",
            FlameParam::Radius => "radius",
            FlameParam::Intensity => "intensity",
            FlameParam::SigmaT => "sigma_t",
            FlameParam::TemperatureBaseK => "temperature_base_k",
            FlameParam::TemperatureTipK => "temperature_tip_k",
            FlameParam::WarpAmp => "warp_amp",
            FlameParam::WarpFreq => "warp_freq",
            FlameParam::RiseSpeed => "rise_speed",
            FlameParam::NoiseAmplitude => "noise_amplitude",
            FlameParam::WhiteBoost => "white_boost",
            FlameParam::BendAmount => "bend_amount",
            FlameParam::WindX => "wind_x",
            FlameParam::WindZ => "wind_z",
            FlameParam::EdgeLow => "edge_low",
            FlameParam::EdgeHigh => "edge_high",
        }
    }

    pub fn from_cli_name(name: &str) -> Option<FlameParam> {
        FlameParam::ALL
            .iter()
            .copied()
            .find(|p| p.cli_name() == name)
    }

    /// Stable PascalCase identifier persisted in scene files.
    pub const fn scene_name(self) -> &'static str {
        match self {
            FlameParam::Height => "Height",
            FlameParam::Radius => "Radius",
            FlameParam::Intensity => "Intensity",
            FlameParam::SigmaT => "SigmaT",
            FlameParam::TemperatureBaseK => "TemperatureBaseK",
            FlameParam::TemperatureTipK => "TemperatureTipK",
            FlameParam::WarpAmp => "WarpAmp",
            FlameParam::WarpFreq => "WarpFreq",
            FlameParam::RiseSpeed => "RiseSpeed",
            FlameParam::NoiseAmplitude => "NoiseAmplitude",
            FlameParam::WhiteBoost => "WhiteBoost",
            FlameParam::BendAmount => "BendAmount",
            FlameParam::WindX => "WindX",
            FlameParam::WindZ => "WindZ",
            FlameParam::EdgeLow => "EdgeLow",
            FlameParam::EdgeHigh => "EdgeHigh",
        }
    }

    /// Conservative sub-range of each UI slider for generated debug keys.
    /// EdgeLow / EdgeHigh are disjoint so any drawn pair keeps low < high.
    pub const fn debug_value_range(self) -> (f32, f32) {
        match self {
            FlameParam::Height => (0.5, 4.0),
            FlameParam::Radius => (0.2, 2.0),
            FlameParam::Intensity => (0.5, 5.0),
            FlameParam::SigmaT => (0.5, 5.0),
            FlameParam::TemperatureBaseK => (800.0, 3000.0),
            FlameParam::TemperatureTipK => (800.0, 3000.0),
            FlameParam::WarpAmp => (0.0, 1.5),
            FlameParam::WarpFreq => (0.5, 8.0),
            FlameParam::RiseSpeed => (0.0, 2.5),
            FlameParam::NoiseAmplitude => (0.0, 1.5),
            FlameParam::WhiteBoost => (0.0, 4.0),
            FlameParam::BendAmount => (0.0, 1.0),
            FlameParam::WindX => (-1.0, 1.0),
            FlameParam::WindZ => (-1.0, 1.0),
            FlameParam::EdgeLow => (0.0, 0.4),
            FlameParam::EdgeHigh => (0.6, 1.0),
        }
    }

    const fn channel(self) -> ScalarChannel {
        ScalarChannel {
            code: self.code(),
            display_name: self.display_name(),
            cli_name: self.cli_name(),
            scene_name: self.scene_name(),
            debug_value_range: self.debug_value_range(),
        }
    }
}

pub static FLAME_CHANNELS: [ScalarChannel; FlameParam::ALL.len()] = {
    let mut channels = [FlameParam::Height.channel(); FlameParam::ALL.len()];
    let mut i = 0;
    while i < FlameParam::ALL.len() {
        channels[i] = FlameParam::ALL[i].channel();
        i += 1;
    }
    channels
};

pub static FLAME_DOMAIN: ScalarChannelDomain = ScalarChannelDomain {
    name: "Flame",
    channels: &FLAME_CHANNELS,
    has_component: flame_has_component,
    entities: flame_entities,
    read: flame_channel_read,
    local_time: flame_local_time,
};

fn flame_has_component(world: &World, entity: Entity) -> bool {
    world.get_component::<FlameEffect>(entity).is_some()
}

fn flame_entities(world: &World) -> Vec<Entity> {
    world.query_flames()
}

fn flame_channel_read(world: &World, entity: Entity, property_type: PropertyType) -> Option<f32> {
    let param = FlameParam::from_property_type(property_type)?;
    world
        .get_component::<FlameEffect>(entity)
        .map(|effect| flame_param_value(effect, param))
}

fn flame_local_time(world: &World, entity: Entity) -> Option<f32> {
    world
        .get_component::<FlameEffect>(entity)
        .map(|effect| effect.time)
}

pub fn apply_flame_param_value(effect: &mut FlameEffect, param: FlameParam, value: f32) {
    match param {
        FlameParam::Height => effect.height = value,
        FlameParam::Radius => effect.radius = value,
        FlameParam::Intensity => effect.intensity = value,
        FlameParam::SigmaT => effect.sigma_t = value,
        FlameParam::TemperatureBaseK => effect.temperature_base_k = value,
        FlameParam::TemperatureTipK => effect.temperature_tip_k = value,
        FlameParam::WarpAmp => effect.warp_amp = value,
        FlameParam::WarpFreq => effect.warp_freq = value,
        FlameParam::RiseSpeed => effect.rise_speed = value,
        FlameParam::NoiseAmplitude => effect.noise_amplitude = value,
        FlameParam::WhiteBoost => effect.white_boost = value,
        FlameParam::BendAmount => effect.bend_amount = value,
        FlameParam::WindX => effect.wind_direction.x = value,
        FlameParam::WindZ => effect.wind_direction.y = value,
        FlameParam::EdgeLow => effect.edge_low = value,
        FlameParam::EdgeHigh => effect.edge_high = value,
    }
}

pub fn flame_param_value(effect: &FlameEffect, param: FlameParam) -> f32 {
    match param {
        FlameParam::Height => effect.height,
        FlameParam::Radius => effect.radius,
        FlameParam::Intensity => effect.intensity,
        FlameParam::SigmaT => effect.sigma_t,
        FlameParam::TemperatureBaseK => effect.temperature_base_k,
        FlameParam::TemperatureTipK => effect.temperature_tip_k,
        FlameParam::WarpAmp => effect.warp_amp,
        FlameParam::WarpFreq => effect.warp_freq,
        FlameParam::RiseSpeed => effect.rise_speed,
        FlameParam::NoiseAmplitude => effect.noise_amplitude,
        FlameParam::WhiteBoost => effect.white_boost,
        FlameParam::BendAmount => effect.bend_amount,
        FlameParam::WindX => effect.wind_direction.x,
        FlameParam::WindZ => effect.wind_direction.y,
        FlameParam::EdgeLow => effect.edge_low,
        FlameParam::EdgeHigh => effect.edge_high,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_roundtrip_all_params() {
        for param in FlameParam::ALL {
            assert_eq!(FlameParam::from_code(param.code()), Some(param));
            assert_eq!(
                FlameParam::from_property_type(param.property_type()),
                Some(param)
            );
        }
        assert_eq!(FlameParam::from_code(999), None);
        assert_eq!(
            FlameParam::from_property_type(PropertyType::TranslationX),
            None
        );
    }

    #[test]
    fn test_cli_name_roundtrip_all_params() {
        for param in FlameParam::ALL {
            assert_eq!(FlameParam::from_cli_name(param.cli_name()), Some(param));
        }
        assert_eq!(FlameParam::from_cli_name("no_such_param"), None);
    }

    #[test]
    fn test_codes_are_unique() {
        let mut codes: Vec<u16> = FlameParam::ALL.iter().map(|p| p.code()).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), FlameParam::ALL.len());
    }

    #[test]
    fn test_channel_table_mirrors_enum() {
        assert_eq!(FLAME_CHANNELS.len(), FlameParam::ALL.len());
        for (channel, param) in FLAME_CHANNELS.iter().zip(FlameParam::ALL) {
            assert_eq!(channel.code, param.code());
            assert_eq!(channel.cli_name, param.cli_name());
            assert_eq!(channel.scene_name, param.scene_name());
            assert_eq!(channel.property_type(), param.property_type());
        }
    }

    #[test]
    fn test_param_value_mirrors_apply_for_all_params() {
        let mut effect = FlameEffect::default();
        for (i, param) in FlameParam::ALL.into_iter().enumerate() {
            let value = 10.0 + i as f32;
            apply_flame_param_value(&mut effect, param, value);
            assert!(
                (flame_param_value(&effect, param) - value).abs() < 1e-6,
                "{param:?}"
            );
        }
    }
}
