use super::clip::EditableAnimationClip;
use super::keyframe::SourceClipId;

#[derive(Clone, Debug)]
pub struct SourceClip {
    pub id: SourceClipId,
    pub editable_clip: EditableAnimationClip,
    pub ref_count: u32,
}

impl SourceClip {
    pub fn new(id: SourceClipId, editable_clip: EditableAnimationClip) -> Self {
        Self {
            id,
            editable_clip,
            ref_count: 0,
        }
    }

    pub fn name(&self) -> &str {
        &self.editable_clip.name
    }

    pub fn duration(&self) -> f32 {
        self.editable_clip.duration
    }

    pub fn source_path(&self) -> Option<&str> {
        self.editable_clip.source_path.as_deref()
    }
}
