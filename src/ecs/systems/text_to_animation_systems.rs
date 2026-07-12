use crate::animation::editable::EditableAnimationClip;
use crate::asset::AssetStorage;
use crate::ecs::resource::{ModelState, TextToAnimationState, TextToAnimationStatus};
use crate::ecs::world::Entity;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RigPresence {
    Present,
    Absent,
}

pub fn detect_rig_presence(model_state: &ModelState, assets: &AssetStorage) -> RigPresence {
    if model_state.has_skinned_meshes && !assets.skeletons.is_empty() {
        RigPresence::Present
    } else {
        RigPresence::Absent
    }
}

pub fn text_to_animation_begin(
    state: &mut TextToAnimationState,
    prompt: String,
    duration_seconds: f32,
    target_entity: Entity,
    rig_present: RigPresence,
) {
    state.target_entity = Some(target_entity);
    state.prompt = prompt;
    state.duration_seconds = duration_seconds;
    state.error_message = None;
    state.generated_clip = None;
    state.generation_time_ms = None;
    state.model_used = None;
    state.skipped_auto_rig = rig_present == RigPresence::Present;

    state.status = match rig_present {
        RigPresence::Present => TextToAnimationStatus::GeneratingMotion,
        RigPresence::Absent => TextToAnimationStatus::AutoRigging,
    };
}

pub fn text_to_animation_advance_to_apply_rig(state: &mut TextToAnimationState) {
    if state.status == TextToAnimationStatus::AutoRigging {
        state.status = TextToAnimationStatus::AutoRigApplying;
    }
}

pub fn text_to_animation_advance_to_motion(state: &mut TextToAnimationState) {
    if state.status == TextToAnimationStatus::AutoRigApplying {
        state.status = TextToAnimationStatus::GeneratingMotion;
    }
}

pub fn text_to_animation_set_motion_result(
    state: &mut TextToAnimationState,
    clip: EditableAnimationClip,
    generation_time_ms: f32,
    model_used: String,
) {
    state.generated_clip = Some(clip);
    state.generation_time_ms = Some(generation_time_ms);
    state.model_used = Some(model_used);
    state.status = TextToAnimationStatus::ApplyingClip;
}

pub fn text_to_animation_mark_done(state: &mut TextToAnimationState) {
    state.status = TextToAnimationStatus::Done;
    state.generated_clip = None;
}

pub fn text_to_animation_mark_error(state: &mut TextToAnimationState, message: String) {
    state.status = TextToAnimationStatus::Error;
    state.error_message = Some(message);
}

pub fn text_to_animation_cancel(state: &mut TextToAnimationState) {
    state.status = TextToAnimationStatus::Idle;
    state.target_entity = None;
    state.error_message = None;
    state.generated_clip = None;
    state.generation_time_ms = None;
    state.model_used = None;
    state.skipped_auto_rig = false;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{AssetStorage, SkeletonAsset};
    use crate::ecs::resource::ModelState;

    fn make_model_state(has_skinned: bool) -> ModelState {
        let mut s = ModelState::default();
        s.has_skinned_meshes = has_skinned;
        s
    }

    fn make_assets_with_skeleton() -> AssetStorage {
        let mut assets = AssetStorage::new();
        assets.add_skeleton(SkeletonAsset {
            id: 0,
            skeleton_id: 0,
            skeleton: crate::animation::Skeleton::new("test"),
        });
        assets
    }

    #[test]
    fn detect_rig_present_when_skinned_and_skeleton_loaded() {
        let model = make_model_state(true);
        let assets = make_assets_with_skeleton();
        assert_eq!(detect_rig_presence(&model, &assets), RigPresence::Present);
    }

    #[test]
    fn detect_rig_absent_when_no_skin_flag() {
        let model = make_model_state(false);
        let assets = make_assets_with_skeleton();
        assert_eq!(detect_rig_presence(&model, &assets), RigPresence::Absent);
    }

    #[test]
    fn detect_rig_absent_when_no_skeleton_asset() {
        let model = make_model_state(true);
        let assets = AssetStorage::new();
        assert_eq!(detect_rig_presence(&model, &assets), RigPresence::Absent);
    }

    #[test]
    fn begin_with_rig_goes_to_generating_motion() {
        let mut state = TextToAnimationState::default();
        text_to_animation_begin(&mut state, "walk".to_string(), 2.0, 7, RigPresence::Present);
        assert_eq!(state.status, TextToAnimationStatus::GeneratingMotion);
        assert!(state.skipped_auto_rig);
    }

    #[test]
    fn begin_without_rig_goes_to_auto_rigging() {
        let mut state = TextToAnimationState::default();
        text_to_animation_begin(&mut state, "walk".to_string(), 2.0, 7, RigPresence::Absent);
        assert_eq!(state.status, TextToAnimationStatus::AutoRigging);
        assert!(!state.skipped_auto_rig);
    }

    #[test]
    fn cancel_resets_state() {
        let mut state = TextToAnimationState::default();
        state.status = TextToAnimationStatus::GeneratingMotion;
        state.error_message = Some("err".to_string());
        text_to_animation_cancel(&mut state);
        assert_eq!(state.status, TextToAnimationStatus::Idle);
        assert!(state.error_message.is_none());
    }
}
