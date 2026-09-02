use thyllore_anim_core::editable::PropertyType;
use thyllore_effect_core::{find_scalar_param, ScalarParam, WATER_SCALAR_PARAMS};

use super::scalar_channel::{ScalarChannel, ScalarChannelDomain};
use super::water::{AppliedWaterPreset, WaterTorusEffect};
use crate::ecs::world::{Entity, World};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WaterParam {
    MajorRadius,
    MinorRadius,
    Ior,
    AbsorptionR,
    AbsorptionG,
    AbsorptionB,
    FlowLongitudinal,
    FlowMeridional,
    WaveAmplitude,
    WaveFrequency,
    WaveSpeed,
    ReflectStrength,
    RefractStrength,
    CausticStrength,
    LightIntensity,
    HighlightSharpness,
    SkyBrightness,
    ScatterStrength,
    ScatterAnisotropy,
    TintR,
    TintG,
    TintB,
}

impl WaterParam {
    pub const ALL: [WaterParam; 22] = [
        WaterParam::MajorRadius,
        WaterParam::MinorRadius,
        WaterParam::Ior,
        WaterParam::AbsorptionR,
        WaterParam::AbsorptionG,
        WaterParam::AbsorptionB,
        WaterParam::FlowLongitudinal,
        WaterParam::FlowMeridional,
        WaterParam::WaveAmplitude,
        WaterParam::WaveFrequency,
        WaterParam::WaveSpeed,
        WaterParam::ReflectStrength,
        WaterParam::RefractStrength,
        WaterParam::CausticStrength,
        WaterParam::LightIntensity,
        WaterParam::HighlightSharpness,
        WaterParam::SkyBrightness,
        WaterParam::ScatterStrength,
        WaterParam::ScatterAnisotropy,
        WaterParam::TintR,
        WaterParam::TintG,
        WaterParam::TintB,
    ];
    pub const fn code(self) -> u16 {
        match self {
            WaterParam::MajorRadius => 256,
            WaterParam::MinorRadius => 257,
            WaterParam::Ior => 258,
            WaterParam::AbsorptionR => 259,
            WaterParam::AbsorptionG => 260,
            WaterParam::AbsorptionB => 261,
            WaterParam::FlowLongitudinal => 262,
            WaterParam::FlowMeridional => 263,
            WaterParam::WaveAmplitude => 264,
            WaterParam::WaveFrequency => 265,
            WaterParam::WaveSpeed => 266,
            WaterParam::ReflectStrength => 267,
            WaterParam::RefractStrength => 268,
            WaterParam::CausticStrength => 272,
            WaterParam::LightIntensity => 273,
            WaterParam::HighlightSharpness => 274,
            WaterParam::SkyBrightness => 275,
            WaterParam::ScatterStrength => 276,
            WaterParam::ScatterAnisotropy => 277,
            WaterParam::TintR => 269,
            WaterParam::TintG => 270,
            WaterParam::TintB => 271,
        }
    }

    pub fn from_code(code: u16) -> Option<WaterParam> {
        WaterParam::ALL.iter().copied().find(|p| p.code() == code)
    }

    pub const fn property_type(self) -> PropertyType {
        PropertyType::Custom(self.code())
    }

    pub fn from_property_type(property_type: PropertyType) -> Option<WaterParam> {
        match property_type {
            PropertyType::Custom(code) => WaterParam::from_code(code),
            _ => None,
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            WaterParam::MajorRadius => "Major Radius",
            WaterParam::MinorRadius => "Minor Radius",
            WaterParam::Ior => "IOR",
            WaterParam::AbsorptionR => "Absorption R",
            WaterParam::AbsorptionG => "Absorption G",
            WaterParam::AbsorptionB => "Absorption B",
            WaterParam::FlowLongitudinal => "Flow Longitudinal",
            WaterParam::FlowMeridional => "Flow Meridional",
            WaterParam::WaveAmplitude => "Wave Amplitude",
            WaterParam::WaveFrequency => "Wave Frequency",
            WaterParam::WaveSpeed => "Wave Speed",
            WaterParam::ReflectStrength => "Reflect Strength",
            WaterParam::RefractStrength => "Refract Strength",
            WaterParam::CausticStrength => "Caustic Strength",
            WaterParam::LightIntensity => "Light Intensity",
            WaterParam::HighlightSharpness => "Highlight Sharpness",
            WaterParam::SkyBrightness => "Sky Brightness",
            WaterParam::ScatterStrength => "Scatter Strength",
            WaterParam::ScatterAnisotropy => "Scatter Anisotropy",
            WaterParam::TintR => "Tint R",
            WaterParam::TintG => "Tint G",
            WaterParam::TintB => "Tint B",
        }
    }

    pub const fn cli_name(self) -> &'static str {
        match self {
            WaterParam::MajorRadius => "major_radius",
            WaterParam::MinorRadius => "minor_radius",
            WaterParam::Ior => "ior",
            WaterParam::AbsorptionR => "absorption_r",
            WaterParam::AbsorptionG => "absorption_g",
            WaterParam::AbsorptionB => "absorption_b",
            WaterParam::FlowLongitudinal => "flow_longitudinal",
            WaterParam::FlowMeridional => "flow_meridional",
            WaterParam::WaveAmplitude => "wave_amplitude",
            WaterParam::WaveFrequency => "wave_frequency",
            WaterParam::WaveSpeed => "wave_speed",
            WaterParam::RefractStrength => "refract_strength",
            WaterParam::ReflectStrength => "reflect_strength",
            WaterParam::CausticStrength => "caustic_strength",
            WaterParam::LightIntensity => "light_intensity",
            WaterParam::HighlightSharpness => "highlight_sharpness",
            WaterParam::SkyBrightness => "sky_brightness",
            WaterParam::ScatterStrength => "scatter_strength",
            WaterParam::ScatterAnisotropy => "scatter_anisotropy",
            WaterParam::TintR => "tint_r",
            WaterParam::TintG => "tint_g",
            WaterParam::TintB => "tint_b",
        }
    }

    pub fn from_cli_name(name: &str) -> Option<WaterParam> {
        WaterParam::ALL
            .iter()
            .copied()
            .find(|p| p.cli_name() == name)
    }

    pub const fn scene_name(self) -> &'static str {
        match self {
            WaterParam::MajorRadius => "MajorRadius",
            WaterParam::MinorRadius => "MinorRadius",
            WaterParam::Ior => "Ior",
            WaterParam::AbsorptionR => "AbsorptionR",
            WaterParam::AbsorptionG => "AbsorptionG",
            WaterParam::AbsorptionB => "AbsorptionB",
            WaterParam::FlowLongitudinal => "FlowLongitudinal",
            WaterParam::FlowMeridional => "FlowMeridional",
            WaterParam::WaveAmplitude => "WaveAmplitude",
            WaterParam::WaveFrequency => "WaveFrequency",
            WaterParam::RefractStrength => "RefractStrength",
            WaterParam::ReflectStrength => "ReflectStrength",
            WaterParam::CausticStrength => "CausticStrength",
            WaterParam::LightIntensity => "LightIntensity",
            WaterParam::HighlightSharpness => "HighlightSharpness",
            WaterParam::SkyBrightness => "SkyBrightness",
            WaterParam::ScatterStrength => "ScatterStrength",
            WaterParam::ScatterAnisotropy => "ScatterAnisotropy",
            WaterParam::WaveSpeed => "WaveSpeed",
            WaterParam::TintR => "TintR",
            WaterParam::TintG => "TintG",
            WaterParam::TintB => "TintB",
        }
    }

    pub const fn debug_value_range(self) -> (f32, f32) {
        match self {
            WaterParam::MajorRadius => (0.5, 5.0),
            WaterParam::MinorRadius => (0.1, 2.0),
            WaterParam::Ior => (1.0, 2.5),
            WaterParam::AbsorptionR => (0.0, 5.0),
            WaterParam::AbsorptionG => (0.0, 5.0),
            WaterParam::AbsorptionB => (0.0, 5.0),
            WaterParam::FlowLongitudinal => (-2.0, 2.0),
            WaterParam::FlowMeridional => (-2.0, 2.0),
            WaterParam::WaveAmplitude => (0.0, 0.5),
            WaterParam::WaveFrequency => (1.0, 20.0),
            WaterParam::WaveSpeed => (0.0, 5.0),
            WaterParam::ReflectStrength => (0.0, 1.0),
            WaterParam::RefractStrength => (0.0, 1.0),
            WaterParam::CausticStrength => (0.0, 2.0),
            WaterParam::LightIntensity => (0.0, 20.0),
            WaterParam::HighlightSharpness => (1.0, 1024.0),
            WaterParam::SkyBrightness => (0.0, 2.0),
            WaterParam::ScatterStrength => (0.0, 10.0),
            WaterParam::ScatterAnisotropy => (-0.9, 0.9),
            WaterParam::TintR => (0.0, 1.0),
            WaterParam::TintG => (0.0, 1.0),
            WaterParam::TintB => (0.0, 1.0),
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

pub static WATER_CHANNELS: [ScalarChannel; WaterParam::ALL.len()] = {
    let mut channels = [WaterParam::MajorRadius.channel(); WaterParam::ALL.len()];
    let mut i = 0;
    while i < WaterParam::ALL.len() {
        channels[i] = WaterParam::ALL[i].channel();
        i += 1;
    }
    channels
};

pub static WATER_DOMAIN: ScalarChannelDomain = ScalarChannelDomain {
    name: "Water",
    channels: &WATER_CHANNELS,
    has_component: water_has_component,
    entities: water_entities,
    read: water_channel_read,
    local_time: water_local_time,
};

fn water_has_component(world: &World, entity: Entity) -> bool {
    world.get_component::<WaterTorusEffect>(entity).is_some()
}

fn water_entities(world: &World) -> Vec<Entity> {
    world.query_waters()
}

fn water_channel_read(world: &World, entity: Entity, property_type: PropertyType) -> Option<f32> {
    let param = WaterParam::from_property_type(property_type)?;
    world
        .get_component::<WaterTorusEffect>(entity)
        .map(|effect| water_param_value(effect, param))
}

fn water_local_time(world: &World, entity: Entity) -> Option<f32> {
    world
        .get_component::<WaterTorusEffect>(entity)
        .map(|effect| effect.time)
}

fn scalar_param(param: WaterParam) -> &'static ScalarParam<WaterTorusEffect> {
    find_scalar_param(WATER_SCALAR_PARAMS, param.cli_name())
        .expect("every WaterParam cli_name is registered in WATER_SCALAR_PARAMS")
}

pub fn apply_water_param_value(effect: &mut WaterTorusEffect, param: WaterParam, value: f32) {
    (scalar_param(param).set)(effect, value)
}

pub fn water_param_value(effect: &WaterTorusEffect, param: WaterParam) -> f32 {
    (scalar_param(param).get)(effect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_roundtrip_all_params() {
        for param in WaterParam::ALL {
            assert_eq!(WaterParam::from_code(param.code()), Some(param));
            assert_eq!(
                WaterParam::from_property_type(param.property_type()),
                Some(param)
            );
        }
        assert_eq!(WaterParam::from_code(999), None);
        assert_eq!(
            WaterParam::from_property_type(PropertyType::TranslationX),
            None
        );
    }

    #[test]
    fn test_cli_name_roundtrip_all_params() {
        for param in WaterParam::ALL {
            assert_eq!(WaterParam::from_cli_name(param.cli_name()), Some(param));
        }
        assert_eq!(WaterParam::from_cli_name("no_such_param"), None);
    }

    #[test]
    fn test_every_cli_name_is_in_the_scalar_registry() {
        for param in WaterParam::ALL {
            assert!(
                find_scalar_param(WATER_SCALAR_PARAMS, param.cli_name()).is_some(),
                "{:?}",
                param
            );
        }
    }

    #[test]
    fn test_codes_are_unique() {
        let mut codes: Vec<u16> = WaterParam::ALL.iter().map(|p| p.code()).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), WaterParam::ALL.len());
    }

    #[test]
    fn test_channel_table_mirrors_enum() {
        assert_eq!(WATER_CHANNELS.len(), WaterParam::ALL.len());
        for (channel, param) in WATER_CHANNELS.iter().zip(WaterParam::ALL) {
            assert_eq!(channel.code, param.code());
            assert_eq!(channel.cli_name, param.cli_name());
            assert_eq!(channel.scene_name, param.scene_name());
            assert_eq!(channel.property_type(), param.property_type());
        }
    }

    #[test]
    fn test_param_value_mirrors_apply_for_all_params() {
        let mut effect = WaterTorusEffect::default();
        for (i, param) in WaterParam::ALL.into_iter().enumerate() {
            let value = 10.0 + i as f32;
            apply_water_param_value(&mut effect, param, value);
            assert!(
                (water_param_value(&effect, param) - value).abs() < 1e-6,
                "{param:?}"
            );
        }
    }
}
