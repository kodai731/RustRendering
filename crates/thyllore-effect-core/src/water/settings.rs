#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WaterSecondaryRays {
    #[default]
    RayQuery,
    ScreenSpace,
    RayTracingPipeline,
}

impl WaterSecondaryRays {
    pub const ALL: [WaterSecondaryRays; 3] = [
        WaterSecondaryRays::RayQuery,
        WaterSecondaryRays::ScreenSpace,
        WaterSecondaryRays::RayTracingPipeline,
    ];

    pub fn label(self) -> &'static str {
        match self {
            WaterSecondaryRays::RayQuery => "Ray Query",
            WaterSecondaryRays::ScreenSpace => "Screen Space",
            WaterSecondaryRays::RayTracingPipeline => "Ray Tracing Pipeline",
        }
    }

    pub fn as_shader_value(self) -> i32 {
        match self {
            WaterSecondaryRays::RayQuery => 0,
            WaterSecondaryRays::ScreenSpace => 1,
            WaterSecondaryRays::RayTracingPipeline => 2,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "rayquery" | "ray-query" => Some(WaterSecondaryRays::RayQuery),
            "screenspace" | "screen-space" => Some(WaterSecondaryRays::ScreenSpace),
            "raytracing" | "ray-tracing" => Some(WaterSecondaryRays::RayTracingPipeline),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterRenderSettings {
    pub secondary_rays: WaterSecondaryRays,
    pub debug_view: i32,
    pub caustic_debug: i32,
    pub batch_history_weight: Option<f32>,
    pub batch_fixed_time: Option<f32>,
    /// Keep the water surface animating while the timeline is not playing.
    pub free_run_when_paused: bool,
}

impl Default for WaterRenderSettings {
    fn default() -> Self {
        Self {
            secondary_rays: WaterSecondaryRays::RayQuery,
            debug_view: 0,
            caustic_debug: 0,
            batch_history_weight: None,
            batch_fixed_time: None,
            free_run_when_paused: true,
        }
    }
}
