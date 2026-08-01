use thyllore_anim_core::editable::PropertyType;

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
    /// Never reorder or reuse codes.
    pub fn code(self) -> u16 {
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

    pub fn property_type(self) -> PropertyType {
        PropertyType::Custom(self.code())
    }

    pub fn from_property_type(property_type: PropertyType) -> Option<FlameParam> {
        match property_type {
            PropertyType::Custom(code) => FlameParam::from_code(code),
            _ => None,
        }
    }

    pub fn display_name(self) -> &'static str {
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
    pub fn cli_name(self) -> &'static str {
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

    /// Conservative sub-range of each UI slider for generated debug keys.
    /// EdgeLow / EdgeHigh are disjoint so any drawn pair keeps low < high.
    pub fn debug_value_range(self) -> (f32, f32) {
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
}
