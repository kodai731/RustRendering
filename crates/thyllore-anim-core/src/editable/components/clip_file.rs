use serde::{Deserialize, Serialize};

use super::clip::EditableAnimationClip;

pub const ANIMATION_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationClipFile {
    pub version: u32,
    pub clip: EditableAnimationClip,
}

impl AnimationClipFile {
    pub fn new(clip: EditableAnimationClip) -> Self {
        Self {
            version: ANIMATION_FORMAT_VERSION,
            clip,
        }
    }
}
