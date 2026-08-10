#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlameShadingMode {
    #[default]
    Analytic,
    ReferenceRaymarch,
    DebugThickness,
    NoiseRaymarch,
    DebugDepthClamp,
}

impl FlameShadingMode {
    pub const ALL: [FlameShadingMode; 5] = [
        FlameShadingMode::Analytic,
        FlameShadingMode::ReferenceRaymarch,
        FlameShadingMode::NoiseRaymarch,
        FlameShadingMode::DebugThickness,
        FlameShadingMode::DebugDepthClamp,
    ];

    pub fn label(self) -> &'static str {
        match self {
            FlameShadingMode::Analytic => "Analytic",
            FlameShadingMode::ReferenceRaymarch => "Reference Raymarch",
            FlameShadingMode::DebugThickness => "Debug Thickness",
            FlameShadingMode::NoiseRaymarch => "Noise Raymarch",
            FlameShadingMode::DebugDepthClamp => "Debug Depth Clamp",
        }
    }

    pub fn as_shader_value(self) -> i32 {
        match self {
            FlameShadingMode::Analytic => 0,
            FlameShadingMode::ReferenceRaymarch => 1,
            FlameShadingMode::DebugThickness => 2,
            FlameShadingMode::NoiseRaymarch => 3,
            FlameShadingMode::DebugDepthClamp => 4,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "analytic" => Some(FlameShadingMode::Analytic),
            "raymarch" => Some(FlameShadingMode::ReferenceRaymarch),
            "thickness" => Some(FlameShadingMode::DebugThickness),
            "noise" => Some(FlameShadingMode::NoiseRaymarch),
            "depthclamp" => Some(FlameShadingMode::DebugDepthClamp),
            _ => None,
        }
    }
}

/// Numeric debug visualization of the wave-path intermediates: the shader
/// replaces the flame color with a colormap of the selected quantity, sampled
/// at the max-density node along each ray (Emission Total integrates instead).
/// History blending is bypassed so the display is the raw current-frame value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlameDebugView {
    Off,
    ShapedNoise,
    Erosion,
    Argument,
    Density,
    SigmaNoise,
    EmissionTotal,
    JitterPsi,
    NoiseCoord,
    SegmentGrid,
    WarpStrain,
    WarpStretch,
}

impl FlameDebugView {
    pub const ALL: [FlameDebugView; 12] = [
        FlameDebugView::Off,
        FlameDebugView::ShapedNoise,
        FlameDebugView::Erosion,
        FlameDebugView::Argument,
        FlameDebugView::Density,
        FlameDebugView::SigmaNoise,
        FlameDebugView::EmissionTotal,
        FlameDebugView::JitterPsi,
        FlameDebugView::NoiseCoord,
        FlameDebugView::SegmentGrid,
        FlameDebugView::WarpStrain,
        FlameDebugView::WarpStretch,
    ];

    pub fn label(self) -> &'static str {
        match self {
            FlameDebugView::Off => "Off",
            FlameDebugView::ShapedNoise => "Shaped Noise",
            FlameDebugView::Erosion => "Erosion",
            FlameDebugView::Argument => "Eroded Argument",
            FlameDebugView::Density => "Smooth Density",
            FlameDebugView::SigmaNoise => "Sigma (unresolved)",
            FlameDebugView::EmissionTotal => "Emission Total",
            FlameDebugView::JitterPsi => "Jitter Psi",
            FlameDebugView::NoiseCoord => "Noise Coord w",
            FlameDebugView::SegmentGrid => "Segment Grid",
            FlameDebugView::WarpStrain => "Warp Strain",
            FlameDebugView::WarpStretch => "Warp Stretch",
        }
    }

    pub fn as_shader_value(self) -> i32 {
        match self {
            FlameDebugView::Off => 0,
            FlameDebugView::ShapedNoise => 1,
            FlameDebugView::Erosion => 2,
            FlameDebugView::Argument => 3,
            FlameDebugView::Density => 4,
            FlameDebugView::SigmaNoise => 5,
            FlameDebugView::EmissionTotal => 6,
            FlameDebugView::JitterPsi => 7,
            FlameDebugView::NoiseCoord => 8,
            FlameDebugView::SegmentGrid => 9,
            FlameDebugView::WarpStrain => 10,
            FlameDebugView::WarpStretch => 11,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "off" => Some(FlameDebugView::Off),
            "shaped" | "shaped-noise" => Some(FlameDebugView::ShapedNoise),
            "erosion" => Some(FlameDebugView::Erosion),
            "argument" => Some(FlameDebugView::Argument),
            "density" => Some(FlameDebugView::Density),
            "sigma" => Some(FlameDebugView::SigmaNoise),
            "emission" => Some(FlameDebugView::EmissionTotal),
            "jitter" => Some(FlameDebugView::JitterPsi),
            "wcoord" | "noise-coord" => Some(FlameDebugView::NoiseCoord),
            "grid" | "segment-grid" => Some(FlameDebugView::SegmentGrid),
            "strain" | "warp-strain" => Some(FlameDebugView::WarpStrain),
            "stretch" | "warp-stretch" => Some(FlameDebugView::WarpStretch),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameRenderSettings {
    pub shading_mode: FlameShadingMode,
    pub reference_step_count: u32,
    pub noise_step_count: u32,
    pub debug_view: FlameDebugView,
}

impl Default for FlameRenderSettings {
    fn default() -> Self {
        Self {
            shading_mode: FlameShadingMode::Analytic,
            reference_step_count: 128,
            noise_step_count: 8,
            debug_view: FlameDebugView::Off,
        }
    }
}

impl FlameRenderSettings {
    pub fn resolved_step_count(&self) -> u32 {
        match self.shading_mode {
            FlameShadingMode::Analytic | FlameShadingMode::DebugThickness => 1,
            FlameShadingMode::ReferenceRaymarch => self.reference_step_count.max(1),
            FlameShadingMode::NoiseRaymarch => self.noise_step_count.max(1),
            FlameShadingMode::DebugDepthClamp => 1,
        }
    }
}
