use crate::ecs::resource::{ClipDragPreview, ClipDragState};

#[derive(Clone, Debug, Default)]
pub struct TimelineInteractionState {
    pub scrubbing: bool,
    pub dragging_clip: Option<ClipDragState>,
    /// Externally injected preview (batch debug action). Interactive drags
    /// compute their preview from the live mouse position instead.
    pub drag_preview: Option<ClipDragPreview>,
}
