#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DebugViewMode {
    Final = 0,
    Position = 1,
    Normal = 2,
    ShadowMask = 3,
    NdotL = 4,
    LightDirection = 5,
    ViewDepth = 6,
    ObjectID = 7,
    SelectionView = 8,
    SelectionUBO = 9,
}

impl Default for DebugViewMode {
    fn default() -> Self {
        DebugViewMode::Final
    }
}

impl DebugViewMode {
    pub fn as_int(&self) -> i32 {
        *self as i32
    }

    pub fn from_int(value: i32) -> Self {
        match value {
            0 => DebugViewMode::Final,
            1 => DebugViewMode::Position,
            2 => DebugViewMode::Normal,
            3 => DebugViewMode::ShadowMask,
            4 => DebugViewMode::NdotL,
            5 => DebugViewMode::LightDirection,
            6 => DebugViewMode::ViewDepth,
            7 => DebugViewMode::ObjectID,
            8 => DebugViewMode::SelectionView,
            9 => DebugViewMode::SelectionUBO,
            other => {
                debug_assert!(false, "Invalid DebugViewMode value: {}", other);
                DebugViewMode::Final
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DebugViewMode::Final => "Final (Lit + Shadow)",
            DebugViewMode::Position => "Position (World Space)",
            DebugViewMode::Normal => "Normal (World Space)",
            DebugViewMode::ShadowMask => "Shadow Mask",
            DebugViewMode::NdotL => "N dot L (Green=Lit, Red=Back)",
            DebugViewMode::LightDirection => "Light Direction",
            DebugViewMode::ViewDepth => "View Depth (R=billboard, G=gbuffer)",
            DebugViewMode::ObjectID => "ObjectID (Color per ID)",
            DebugViewMode::SelectionView => "Selection View (Orange=Selected)",
            DebugViewMode::SelectionUBO => "SelectionUBO (R=count, G=id0)",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DebugViewState {
    pub debug_view_mode: DebugViewMode,
    pub billboard_click_rect: Option<[f32; 4]>,
    pub show_click_debug: bool,
    pub black_background: bool,
}
