#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WindShadingMode {
    #[default]
    ClosedForm,
    ReferenceQuadrature,
}

impl WindShadingMode {
    pub const ALL: [WindShadingMode; 2] = [
        WindShadingMode::ClosedForm,
        WindShadingMode::ReferenceQuadrature,
    ];

    pub fn label(self) -> &'static str {
        match self {
            WindShadingMode::ClosedForm => "Closed Form",
            WindShadingMode::ReferenceQuadrature => "Reference Quadrature",
        }
    }

    pub fn as_shader_value(self) -> i32 {
        match self {
            WindShadingMode::ClosedForm => 0,
            WindShadingMode::ReferenceQuadrature => 1,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "closed" | "closed-form" => Some(WindShadingMode::ClosedForm),
            "reference" | "quadrature" => Some(WindShadingMode::ReferenceQuadrature),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WindDebugView {
    #[default]
    Off,
    OpticalDepth,
    KnotCount,
}

impl WindDebugView {
    pub const ALL: [WindDebugView; 3] = [
        WindDebugView::Off,
        WindDebugView::OpticalDepth,
        WindDebugView::KnotCount,
    ];

    pub fn label(self) -> &'static str {
        match self {
            WindDebugView::Off => "Off",
            WindDebugView::OpticalDepth => "Optical Depth",
            WindDebugView::KnotCount => "Knot Count",
        }
    }

    pub fn as_shader_value(self) -> i32 {
        match self {
            WindDebugView::Off => 0,
            WindDebugView::OpticalDepth => 1,
            WindDebugView::KnotCount => 2,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "off" => Some(WindDebugView::Off),
            "depth" | "optical-depth" => Some(WindDebugView::OpticalDepth),
            "knots" | "knot-count" => Some(WindDebugView::KnotCount),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindRenderSettings {
    pub shading_mode: WindShadingMode,
    pub reference_step_count: u32,
    pub debug_view: WindDebugView,
    pub batch_fixed_time: Option<f32>,
    pub free_run_when_paused: bool,
}

impl Default for WindRenderSettings {
    fn default() -> Self {
        Self {
            shading_mode: WindShadingMode::ClosedForm,
            reference_step_count: 256,
            debug_view: WindDebugView::Off,
            batch_fixed_time: None,
            free_run_when_paused: true,
        }
    }
}
