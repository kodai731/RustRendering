#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum EntityIcon {
    #[default]
    Empty,
    Model,
    Mesh,
    Light,
    Camera,
    Grid,
    Gizmo,
    Billboard,
    Flame,
}

impl EntityIcon {
    pub fn to_char(&self) -> char {
        match self {
            EntityIcon::Empty => ' ',
            EntityIcon::Model => 'M',
            EntityIcon::Mesh => 'm',
            EntityIcon::Light => 'L',
            EntityIcon::Camera => 'C',
            EntityIcon::Grid => 'G',
            EntityIcon::Gizmo => 'g',
            EntityIcon::Billboard => 'B',
            EntityIcon::Flame => 'F',
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EditorDisplay {
    pub icon: EntityIcon,
    pub expanded: bool,
    pub hidden_in_hierarchy: bool,
}

impl EditorDisplay {
    pub fn new(icon: EntityIcon) -> Self {
        Self {
            icon,
            expanded: false,
            hidden_in_hierarchy: false,
        }
    }

    pub fn with_expanded(mut self, expanded: bool) -> Self {
        self.expanded = expanded;
        self
    }
}
