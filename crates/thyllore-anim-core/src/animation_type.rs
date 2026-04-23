#[derive(Clone, Debug, PartialEq)]
pub enum AnimationType {
    None,
    Skeletal,
    Node,
}

impl Default for AnimationType {
    fn default() -> Self {
        Self::None
    }
}
