use thyllore_anim_core::editable::PropertyType;
use thyllore_effect_core::{find_scalar_param, ScalarParam, WIND_SCALAR_PARAMS};

use super::scalar_channel::{ScalarChannel, ScalarChannelDomain};
use super::wind::WindTornadoEffect;
use crate::ecs::world::{Entity, World};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WindParam {
    ColumnHeight,
    CoreRadius,
    CoreStrength,
    WallRadiusBase,
    WallRadiusTop,
    WallWidthQ,
    WallStrength,
    TopFade,
    Density,
    AlbedoR,
    AlbedoG,
    AlbedoB,
    AmbientBrightness,
    PhaseG,
    SunIntensity,
    RiseInitialHeight,
    RiseDuration,
    SpreadStart,
    SpreadRate,
    DissipateStart,
    DissipateTime,
}

impl WindParam {
    pub const ALL: [WindParam; 21] = [
        WindParam::ColumnHeight,
        WindParam::CoreRadius,
        WindParam::CoreStrength,
        WindParam::WallRadiusBase,
        WindParam::WallRadiusTop,
        WindParam::WallWidthQ,
        WindParam::WallStrength,
        WindParam::TopFade,
        WindParam::Density,
        WindParam::AlbedoR,
        WindParam::AlbedoG,
        WindParam::AlbedoB,
        WindParam::AmbientBrightness,
        WindParam::PhaseG,
        WindParam::SunIntensity,
        WindParam::RiseInitialHeight,
        WindParam::RiseDuration,
        WindParam::SpreadStart,
        WindParam::SpreadRate,
        WindParam::DissipateStart,
        WindParam::DissipateTime,
    ];

    pub const fn code(self) -> u16 {
        match self {
            WindParam::ColumnHeight => 512,
            WindParam::CoreRadius => 513,
            WindParam::CoreStrength => 514,
            WindParam::WallRadiusBase => 515,
            WindParam::WallRadiusTop => 516,
            WindParam::WallWidthQ => 517,
            WindParam::WallStrength => 518,
            WindParam::TopFade => 519,
            WindParam::Density => 520,
            WindParam::AlbedoR => 521,
            WindParam::AlbedoG => 522,
            WindParam::AlbedoB => 523,
            WindParam::AmbientBrightness => 524,
            WindParam::PhaseG => 531,
            WindParam::SunIntensity => 532,
            WindParam::RiseInitialHeight => 525,
            WindParam::RiseDuration => 526,
            WindParam::SpreadStart => 527,
            WindParam::SpreadRate => 528,
            WindParam::DissipateStart => 529,
            WindParam::DissipateTime => 530,
        }
    }

    pub fn from_code(code: u16) -> Option<WindParam> {
        WindParam::ALL.iter().copied().find(|p| p.code() == code)
    }

    pub const fn property_type(self) -> PropertyType {
        PropertyType::Custom(self.code())
    }

    pub fn from_property_type(property_type: PropertyType) -> Option<WindParam> {
        match property_type {
            PropertyType::Custom(code) => WindParam::from_code(code),
            _ => None,
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            WindParam::ColumnHeight => "Column Height",
            WindParam::CoreRadius => "Core Radius",
            WindParam::CoreStrength => "Core Strength",
            WindParam::WallRadiusBase => "Wall Radius Base",
            WindParam::WallRadiusTop => "Wall Radius Top",
            WindParam::WallWidthQ => "Wall Width Q",
            WindParam::WallStrength => "Wall Strength",
            WindParam::TopFade => "Top Fade",
            WindParam::Density => "Density",
            WindParam::AlbedoR => "Albedo R",
            WindParam::AlbedoG => "Albedo G",
            WindParam::AlbedoB => "Albedo B",
            WindParam::AmbientBrightness => "Ambient Brightness",
            WindParam::PhaseG => "Phase G",
            WindParam::SunIntensity => "Sun Intensity",
            WindParam::RiseInitialHeight => "Rise Initial Height",
            WindParam::RiseDuration => "Rise Duration",
            WindParam::SpreadStart => "Spread Start",
            WindParam::SpreadRate => "Spread Rate",
            WindParam::DissipateStart => "Dissipate Start",
            WindParam::DissipateTime => "Dissipate Time",
        }
    }

    pub const fn cli_name(self) -> &'static str {
        match self {
            WindParam::ColumnHeight => "column_height",
            WindParam::CoreRadius => "core_radius",
            WindParam::CoreStrength => "core_strength",
            WindParam::WallRadiusBase => "wall_radius_base",
            WindParam::WallRadiusTop => "wall_radius_top",
            WindParam::WallWidthQ => "wall_width_q",
            WindParam::WallStrength => "wall_strength",
            WindParam::TopFade => "top_fade",
            WindParam::Density => "density",
            WindParam::AlbedoR => "albedo_r",
            WindParam::AlbedoG => "albedo_g",
            WindParam::AlbedoB => "albedo_b",
            WindParam::AmbientBrightness => "ambient_brightness",
            WindParam::PhaseG => "phase_g",
            WindParam::SunIntensity => "sun_intensity",
            WindParam::RiseInitialHeight => "rise_initial_height",
            WindParam::RiseDuration => "rise_duration",
            WindParam::SpreadStart => "spread_start",
            WindParam::SpreadRate => "spread_rate",
            WindParam::DissipateStart => "dissipate_start",
            WindParam::DissipateTime => "dissipate_time",
        }
    }

    pub fn from_cli_name(name: &str) -> Option<WindParam> {
        WindParam::ALL
            .iter()
            .copied()
            .find(|p| p.cli_name() == name)
    }

    pub const fn scene_name(self) -> &'static str {
        match self {
            WindParam::ColumnHeight => "ColumnHeight",
            WindParam::CoreRadius => "CoreRadius",
            WindParam::CoreStrength => "CoreStrength",
            WindParam::WallRadiusBase => "WallRadiusBase",
            WindParam::WallRadiusTop => "WallRadiusTop",
            WindParam::WallWidthQ => "WallWidthQ",
            WindParam::WallStrength => "WallStrength",
            WindParam::TopFade => "TopFade",
            WindParam::Density => "Density",
            WindParam::AlbedoR => "AlbedoR",
            WindParam::AlbedoG => "AlbedoG",
            WindParam::AlbedoB => "AlbedoB",
            WindParam::AmbientBrightness => "AmbientBrightness",
            WindParam::PhaseG => "PhaseG",
            WindParam::SunIntensity => "SunIntensity",
            WindParam::RiseInitialHeight => "RiseInitialHeight",
            WindParam::RiseDuration => "RiseDuration",
            WindParam::SpreadStart => "SpreadStart",
            WindParam::SpreadRate => "SpreadRate",
            WindParam::DissipateStart => "DissipateStart",
            WindParam::DissipateTime => "DissipateTime",
        }
    }

    pub const fn debug_value_range(self) -> (f32, f32) {
        match self {
            WindParam::ColumnHeight => (0.5, 10.0),
            WindParam::CoreRadius => (0.0, 2.0),
            WindParam::CoreStrength => (0.0, 4.0),
            WindParam::WallRadiusBase => (0.05, 5.0),
            WindParam::WallRadiusTop => (0.05, 5.0),
            WindParam::WallWidthQ => (0.01, 2.0),
            WindParam::WallStrength => (0.0, 4.0),
            WindParam::TopFade => (0.05, 1.0),
            WindParam::Density => (0.0, 20.0),
            WindParam::AlbedoR => (0.0, 1.0),
            WindParam::AlbedoG => (0.0, 1.0),
            WindParam::AlbedoB => (0.0, 1.0),
            WindParam::AmbientBrightness => (0.0, 5.0),
            WindParam::PhaseG => (-0.95, 0.95),
            WindParam::SunIntensity => (0.0, 10.0),
            WindParam::RiseInitialHeight => (0.0, 1.0),
            WindParam::RiseDuration => (0.1, 10.0),
            WindParam::SpreadStart => (0.0, 10.0),
            WindParam::SpreadRate => (0.0, 5.0),
            WindParam::DissipateStart => (0.0, 10.0),
            WindParam::DissipateTime => (0.0, 10.0),
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

pub static WIND_CHANNELS: [ScalarChannel; WindParam::ALL.len()] = {
    let mut channels = [WindParam::ColumnHeight.channel(); WindParam::ALL.len()];
    let mut i = 0;
    while i < WindParam::ALL.len() {
        channels[i] = WindParam::ALL[i].channel();
        i += 1;
    }
    channels
};

pub static WIND_DOMAIN: ScalarChannelDomain = ScalarChannelDomain {
    name: "Wind",
    channels: &WIND_CHANNELS,
    has_component: wind_has_component,
    entities: wind_entities,
    read: wind_channel_read,
    local_time: wind_local_time,
};

fn wind_has_component(world: &World, entity: Entity) -> bool {
    world.get_component::<WindTornadoEffect>(entity).is_some()
}

fn wind_entities(world: &World) -> Vec<Entity> {
    world.query_winds()
}

fn wind_channel_read(world: &World, entity: Entity, property_type: PropertyType) -> Option<f32> {
    let param = WindParam::from_property_type(property_type)?;
    world
        .get_component::<WindTornadoEffect>(entity)
        .map(|effect| wind_param_value(effect, param))
}

fn wind_local_time(world: &World, entity: Entity) -> Option<f32> {
    world
        .get_component::<WindTornadoEffect>(entity)
        .map(|effect| effect.time)
}

fn scalar_param(param: WindParam) -> &'static ScalarParam<WindTornadoEffect> {
    find_scalar_param(WIND_SCALAR_PARAMS, param.cli_name())
        .expect("every WindParam cli_name is registered in WIND_SCALAR_PARAMS")
}

pub fn apply_wind_param_value(effect: &mut WindTornadoEffect, param: WindParam, value: f32) {
    (scalar_param(param).set)(effect, value)
}

pub fn wind_param_value(effect: &WindTornadoEffect, param: WindParam) -> f32 {
    (scalar_param(param).get)(effect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_and_cli_name_roundtrip_all_params() {
        for param in WindParam::ALL {
            assert_eq!(WindParam::from_code(param.code()), Some(param));
            assert_eq!(
                WindParam::from_property_type(param.property_type()),
                Some(param)
            );
            assert_eq!(WindParam::from_cli_name(param.cli_name()), Some(param));
        }
        assert_eq!(WindParam::from_code(999), None);
        assert_eq!(WindParam::from_cli_name("no_such_param"), None);
    }

    #[test]
    fn test_every_cli_name_is_in_the_scalar_registry() {
        for param in WindParam::ALL {
            assert!(
                find_scalar_param(WIND_SCALAR_PARAMS, param.cli_name()).is_some(),
                "{:?}",
                param
            );
        }
    }

    #[test]
    fn test_codes_are_unique_and_start_at_512() {
        let mut codes: Vec<u16> = WindParam::ALL.iter().map(|p| p.code()).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), WindParam::ALL.len());
        assert!(codes.iter().all(|code| *code >= 512));
    }

    #[test]
    fn test_channel_table_mirrors_enum() {
        assert_eq!(WIND_CHANNELS.len(), WindParam::ALL.len());
        for (channel, param) in WIND_CHANNELS.iter().zip(WindParam::ALL) {
            assert_eq!(channel.code, param.code());
            assert_eq!(channel.cli_name, param.cli_name());
            assert_eq!(channel.property_type(), param.property_type());
        }
    }

    #[test]
    fn test_param_value_mirrors_apply_for_all_params() {
        let mut effect = WindTornadoEffect::default();
        for (i, param) in WindParam::ALL.into_iter().enumerate() {
            let value = 10.0 + i as f32;
            apply_wind_param_value(&mut effect, param, value);
            assert!(
                (wind_param_value(&effect, param) - value).abs() < 1e-6,
                "{param:?}"
            );
        }
    }
}
