#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Override,
    Additive,
}

impl BlendMode {
    pub fn default_ease_in(&self) -> EaseType {
        match self {
            BlendMode::Override => EaseType::Linear,
            BlendMode::Additive => EaseType::Linear,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EaseType {
    #[default]
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Stepped,
}
