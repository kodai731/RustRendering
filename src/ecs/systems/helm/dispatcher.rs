//! Turns a bound `ToolCall` into either a `UIEvent` or a report.
//!
//! The dispatcher never mutates the world. Edits go out as `UIEvent`s so the
//! existing undo/redo and event dispatch phase keep working unchanged, and no
//! second mutation path is created.

use crate::ecs::events::UIEvent;
use crate::ecs::resource::{HierarchyState, TimelineState};
use crate::ecs::world::{Entity, World};
use crate::helm::components::tool_call::{
    FocusTarget, MotionCategory, ObjectName, ShotPreset, SpeedPreset, ToolCall, VisibilityState,
};
use crate::helm::systems::seek::{resolve_seek_time, TimelineContext};

use super::name_resolver::{
    list_entity_names, read_entity_name, resolve_entity_by_name, NameResolution,
};

/// `UIEvent` is deliberately not comparable — several of its payloads come from
/// `thyllore-anim-core` and do not implement `PartialEq` — so this type is not
/// either. Callers match on the variant they expect.
#[derive(Clone, Debug)]
pub enum DispatchOutcome {
    Command(UIEvent),
    Report(String),
    MotionRequest {
        category: MotionCategory,
        speed: SpeedPreset,
    },
    CameraDirectionRequest {
        utterance: String,
        target: Option<Entity>,
    },
    Rejected(DispatchError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchError {
    ObjectNotFound(String),
    ObjectNameAmbiguous {
        requested: String,
        candidates: Vec<String>,
    },
    NothingSelected,
}

pub fn dispatch_tool_call(
    world: &World,
    timeline: &TimelineContext,
    call: &ToolCall,
) -> DispatchOutcome {
    match call {
        ToolCall::ListObjects => report_object_list(world),
        ToolCall::DescribeSelection => report_selection(world),
        ToolCall::GetPlaybackState => report_playback_state(world),
        ToolCall::TakeScreenshot => DispatchOutcome::Command(UIEvent::TakeScreenshot),

        ToolCall::PlayAnimation => DispatchOutcome::Command(UIEvent::TimelinePlay),
        ToolCall::PauseAnimation => DispatchOutcome::Command(UIEvent::TimelinePause),
        ToolCall::StopAnimation => DispatchOutcome::Command(UIEvent::TimelineStop),
        ToolCall::ToggleLoop => DispatchOutcome::Command(UIEvent::TimelineToggleLoop),

        ToolCall::SetPlaybackSpeed(preset) => {
            DispatchOutcome::Command(UIEvent::TimelineSetSpeed(preset.to_multiplier()))
        }
        ToolCall::SeekTime(position) => DispatchOutcome::Command(UIEvent::TimelineSetTime(
            resolve_seek_time(timeline, *position),
        )),

        ToolCall::SelectObject(name) => {
            dispatch_for_named_object(world, name, |entity| UIEvent::SelectEntity(entity))
        }
        ToolCall::SetObjectVisibility(name, state) => {
            dispatch_visibility_change(world, name, *state)
        }

        ToolCall::FocusCamera(target) => dispatch_camera_focus(world, *target),

        ToolCall::Undo => DispatchOutcome::Command(UIEvent::Undo),
        ToolCall::Redo => DispatchOutcome::Command(UIEvent::Redo),
        ToolCall::SaveScene => DispatchOutcome::Command(UIEvent::SaveScene),

        ToolCall::GenerateMotion(category, speed) => DispatchOutcome::MotionRequest {
            category: *category,
            speed: *speed,
        },

        ToolCall::CameraShot(preset, speed) => dispatch_camera_shot(world, *preset, *speed),

        ToolCall::CameraDirection(name) => dispatch_camera_direction(world, name),
    }
}

fn dispatch_for_named_object(
    world: &World,
    name: &ObjectName,
    to_event: impl Fn(Entity) -> UIEvent,
) -> DispatchOutcome {
    match resolve_entity_by_name(world, name.as_str()) {
        NameResolution::Resolved(entity) => DispatchOutcome::Command(to_event(entity)),
        NameResolution::NotFound => {
            DispatchOutcome::Rejected(DispatchError::ObjectNotFound(name.0.clone()))
        }
        NameResolution::Ambiguous(entities) => {
            DispatchOutcome::Rejected(DispatchError::ObjectNameAmbiguous {
                requested: name.0.clone(),
                candidates: entities
                    .iter()
                    .filter_map(|entity| read_entity_name(world, *entity))
                    .collect(),
            })
        }
    }
}

fn dispatch_visibility_change(
    world: &World,
    name: &ObjectName,
    state: VisibilityState,
) -> DispatchOutcome {
    let visibility = state.to_visibility();
    dispatch_for_named_object(world, name, move |entity| {
        UIEvent::SetEntityVisible(entity, visibility)
    })
}

fn dispatch_camera_focus(world: &World, target: FocusTarget) -> DispatchOutcome {
    match target {
        FocusTarget::Model => DispatchOutcome::Command(UIEvent::MoveCameraToModel),
        FocusTarget::Reset => DispatchOutcome::Command(UIEvent::ResetCamera),
        FocusTarget::Selection => match read_selected_entity(world) {
            Some(entity) => DispatchOutcome::Command(UIEvent::FocusOnEntity(entity)),
            None => DispatchOutcome::Rejected(DispatchError::NothingSelected),
        },
    }
}

fn dispatch_camera_shot(world: &World, preset: ShotPreset, speed: SpeedPreset) -> DispatchOutcome {
    match preset {
        ShotPreset::LookAtSelection | ShotPreset::OrbitAroundSelection => {
            match read_selected_entity(world) {
                Some(entity) => DispatchOutcome::Command(UIEvent::CameraShot {
                    preset,
                    speed,
                    target: Some(entity),
                }),
                None => DispatchOutcome::Rejected(DispatchError::NothingSelected),
            }
        }
        ShotPreset::DollyIn
        | ShotPreset::DollyOut
        | ShotPreset::CraneUp
        | ShotPreset::CraneDown => DispatchOutcome::Command(UIEvent::CameraShot {
            preset,
            speed,
            target: None,
        }),
    }
}

fn dispatch_camera_direction(world: &World, utterance: &str) -> DispatchOutcome {
    let target = match resolve_entity_by_name(world, utterance) {
        NameResolution::Resolved(entity) => Some(entity),
        NameResolution::NotFound => read_selected_entity(world),
        NameResolution::Ambiguous(entities) => {
            read_selected_entity(world).or_else(|| entities.first().copied())
        }
    };
    DispatchOutcome::CameraDirectionRequest {
        utterance: utterance.to_string(),
        target,
    }
}

fn read_selected_entity(world: &World) -> Option<Entity> {
    world
        .get_resource::<HierarchyState>()
        .and_then(|state| state.selected_entity)
}

fn report_object_list(world: &World) -> DispatchOutcome {
    let names = list_entity_names(world);
    if names.is_empty() {
        return DispatchOutcome::Report("The scene has no named objects.".to_string());
    }
    DispatchOutcome::Report(format!("{} objects: {}", names.len(), names.join(", ")))
}

fn report_selection(world: &World) -> DispatchOutcome {
    let Some(entity) = read_selected_entity(world) else {
        return DispatchOutcome::Rejected(DispatchError::NothingSelected);
    };

    let label = read_entity_name(world, entity).unwrap_or_else(|| format!("entity {}", entity));
    DispatchOutcome::Report(format!("Selected: {}", label))
}

fn report_playback_state(world: &World) -> DispatchOutcome {
    let Some(timeline) = world.get_resource::<TimelineState>() else {
        return DispatchOutcome::Report("No timeline is active.".to_string());
    };

    let motion = if timeline.playing {
        "playing"
    } else {
        "paused"
    };
    let looping = if timeline.looping { "on" } else { "off" };
    DispatchOutcome::Report(format!(
        "{} at {:.2}s, speed {:.2}x, loop {}",
        motion, timeline.current_time, timeline.speed, looping
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::world::{Name, Visibility};
    use crate::helm::components::tool_call::SeekPosition;

    fn spawn_named(world: &mut World, name: &str) -> Entity {
        let entity = world.spawn();
        world.insert_component(entity, Name(name.to_string()));
        entity
    }

    fn select(world: &mut World, entity: Entity) {
        let mut state = HierarchyState::default();
        state.selected_entity = Some(entity);
        world.insert_resource(state);
    }

    fn empty_timeline() -> TimelineContext {
        TimelineContext::default()
    }

    fn dispatch(world: &World, call: ToolCall) -> DispatchOutcome {
        dispatch_tool_call(world, &empty_timeline(), &call)
    }

    fn expect_command(outcome: DispatchOutcome) -> UIEvent {
        match outcome {
            DispatchOutcome::Command(event) => event,
            other => panic!("expected a command, got {:?}", other),
        }
    }

    fn expect_report(outcome: DispatchOutcome) -> String {
        match outcome {
            DispatchOutcome::Report(text) => text,
            other => panic!("expected a report, got {:?}", other),
        }
    }

    fn expect_rejection(outcome: DispatchOutcome) -> DispatchError {
        match outcome {
            DispatchOutcome::Rejected(error) => error,
            other => panic!("expected a rejection, got {:?}", other),
        }
    }

    #[test]
    fn playback_tools_map_to_their_timeline_events() {
        let world = World::new();
        assert!(matches!(
            expect_command(dispatch(&world, ToolCall::PlayAnimation)),
            UIEvent::TimelinePlay
        ));
        assert!(matches!(
            expect_command(dispatch(&world, ToolCall::PauseAnimation)),
            UIEvent::TimelinePause
        ));
        assert!(matches!(
            expect_command(dispatch(&world, ToolCall::StopAnimation)),
            UIEvent::TimelineStop
        ));
        assert!(matches!(
            expect_command(dispatch(&world, ToolCall::ToggleLoop)),
            UIEvent::TimelineToggleLoop
        ));
    }

    #[test]
    fn edit_history_tools_map_to_their_events() {
        let world = World::new();
        assert!(matches!(
            expect_command(dispatch(&world, ToolCall::Undo)),
            UIEvent::Undo
        ));
        assert!(matches!(
            expect_command(dispatch(&world, ToolCall::Redo)),
            UIEvent::Redo
        ));
        assert!(matches!(
            expect_command(dispatch(&world, ToolCall::SaveScene)),
            UIEvent::SaveScene
        ));
    }

    #[test]
    fn speed_presets_reach_the_event_as_multipliers() {
        let world = World::new();
        assert!(matches!(
            expect_command(dispatch(&world, ToolCall::SetPlaybackSpeed(SpeedPreset::Slow))),
            UIEvent::TimelineSetSpeed(multiplier) if multiplier == 0.5
        ));
        assert!(matches!(
            expect_command(dispatch(&world, ToolCall::SetPlaybackSpeed(SpeedPreset::Fast))),
            UIEvent::TimelineSetSpeed(multiplier) if multiplier == 2.0
        ));
    }

    #[test]
    fn seek_uses_the_supplied_timeline_context() {
        let world = World::new();
        let timeline = TimelineContext {
            current_time: 1.0,
            duration: 6.0,
            keyframe_times: vec![0.0, 3.0],
        };

        let to_end = dispatch_tool_call(&world, &timeline, &ToolCall::SeekTime(SeekPosition::End));
        assert!(matches!(
            expect_command(to_end),
            UIEvent::TimelineSetTime(time) if time == 6.0
        ));

        let to_next = dispatch_tool_call(
            &world,
            &timeline,
            &ToolCall::SeekTime(SeekPosition::NextKey),
        );
        assert!(matches!(
            expect_command(to_next),
            UIEvent::TimelineSetTime(time) if time == 3.0
        ));
    }

    #[test]
    fn selecting_a_known_object_resolves_to_its_entity() {
        let mut world = World::new();
        let hero = spawn_named(&mut world, "Hero");

        let outcome = dispatch(&world, ToolCall::SelectObject(ObjectName("Hero".into())));
        assert!(matches!(
            expect_command(outcome),
            UIEvent::SelectEntity(entity) if entity == hero
        ));
    }

    #[test]
    fn selecting_an_unknown_object_is_rejected_before_any_event() {
        let mut world = World::new();
        spawn_named(&mut world, "Hero");

        let outcome = dispatch(&world, ToolCall::SelectObject(ObjectName("Dragon".into())));
        assert_eq!(
            expect_rejection(outcome),
            DispatchError::ObjectNotFound("Dragon".into())
        );
    }

    #[test]
    fn an_ambiguous_name_is_rejected_with_the_candidate_names() {
        let mut world = World::new();
        spawn_named(&mut world, "Light01");
        spawn_named(&mut world, "Light02");

        let outcome = dispatch(&world, ToolCall::SelectObject(ObjectName("light".into())));
        assert_eq!(
            expect_rejection(outcome),
            DispatchError::ObjectNameAmbiguous {
                requested: "light".into(),
                candidates: vec!["Light01".into(), "Light02".into()],
            }
        );
    }

    #[test]
    fn hiding_an_object_carries_the_visibility_state() {
        let mut world = World::new();
        let floor = spawn_named(&mut world, "Floor");

        let outcome = dispatch(
            &world,
            ToolCall::SetObjectVisibility(ObjectName("Floor".into()), VisibilityState::Hide),
        );
        assert!(matches!(
            expect_command(outcome),
            UIEvent::SetEntityVisible(entity, Visibility::Hidden) if entity == floor
        ));
    }

    #[test]
    fn showing_an_object_carries_the_visibility_state() {
        let mut world = World::new();
        let floor = spawn_named(&mut world, "Floor");

        let outcome = dispatch(
            &world,
            ToolCall::SetObjectVisibility(ObjectName("Floor".into()), VisibilityState::Show),
        );
        assert!(matches!(
            expect_command(outcome),
            UIEvent::SetEntityVisible(entity, Visibility::Shown) if entity == floor
        ));
    }

    #[test]
    fn focusing_the_model_and_resetting_need_no_selection() {
        let world = World::new();
        assert!(matches!(
            expect_command(dispatch(&world, ToolCall::FocusCamera(FocusTarget::Model))),
            UIEvent::MoveCameraToModel
        ));
        assert!(matches!(
            expect_command(dispatch(&world, ToolCall::FocusCamera(FocusTarget::Reset))),
            UIEvent::ResetCamera
        ));
    }

    #[test]
    fn focusing_the_selection_requires_a_selection() {
        let world = World::new();
        let outcome = dispatch(&world, ToolCall::FocusCamera(FocusTarget::Selection));
        assert_eq!(expect_rejection(outcome), DispatchError::NothingSelected);
    }

    #[test]
    fn focusing_the_selection_uses_the_selected_entity() {
        let mut world = World::new();
        let hero = spawn_named(&mut world, "Hero");
        select(&mut world, hero);

        let outcome = dispatch(&world, ToolCall::FocusCamera(FocusTarget::Selection));
        assert!(matches!(
            expect_command(outcome),
            UIEvent::FocusOnEntity(entity) if entity == hero
        ));
    }

    #[test]
    fn listing_objects_reports_every_name() {
        let mut world = World::new();
        spawn_named(&mut world, "Hero");
        spawn_named(&mut world, "Camera01");

        assert_eq!(
            expect_report(dispatch(&world, ToolCall::ListObjects)),
            "2 objects: Camera01, Hero"
        );
    }

    #[test]
    fn listing_objects_in_an_empty_scene_reports_rather_than_fails() {
        let world = World::new();
        assert_eq!(
            expect_report(dispatch(&world, ToolCall::ListObjects)),
            "The scene has no named objects."
        );
    }

    #[test]
    fn describing_the_selection_requires_a_selection() {
        let world = World::new();
        let outcome = dispatch(&world, ToolCall::DescribeSelection);
        assert_eq!(expect_rejection(outcome), DispatchError::NothingSelected);
    }

    #[test]
    fn describing_the_selection_names_the_selected_entity() {
        let mut world = World::new();
        let hero = spawn_named(&mut world, "Hero");
        select(&mut world, hero);

        assert_eq!(
            expect_report(dispatch(&world, ToolCall::DescribeSelection)),
            "Selected: Hero"
        );
    }

    #[test]
    fn playback_state_reports_the_timeline_resource() {
        let mut world = World::new();
        let mut timeline = TimelineState::new();
        timeline.playing = true;
        timeline.current_time = 1.5;
        timeline.speed = 2.0;
        timeline.looping = false;
        world.insert_resource(timeline);

        assert_eq!(
            expect_report(dispatch(&world, ToolCall::GetPlaybackState)),
            "playing at 1.50s, speed 2.00x, loop off"
        );
    }

    #[test]
    fn playback_state_without_a_timeline_reports_rather_than_panics() {
        let world = World::new();
        assert_eq!(
            expect_report(dispatch(&world, ToolCall::GetPlaybackState)),
            "No timeline is active."
        );
    }

    #[test]
    fn motion_generation_is_deferred_to_curve_copilot() {
        let world = World::new();
        let outcome = dispatch(
            &world,
            ToolCall::GenerateMotion(MotionCategory::Walk, SpeedPreset::Slow),
        );
        assert!(matches!(
            outcome,
            DispatchOutcome::MotionRequest {
                category: MotionCategory::Walk,
                speed: SpeedPreset::Slow,
            }
        ));
    }

    #[test]
    fn a_rejected_call_never_produces_a_command() {
        let world = World::new();
        let outcome = dispatch(&world, ToolCall::SelectObject(ObjectName("Ghost".into())));
        assert!(matches!(outcome, DispatchOutcome::Rejected(_)));
    }

    #[test]
    fn camera_shot_dolly_in_without_selection_has_no_target() {
        let world = World::new();
        let outcome = dispatch(
            &world,
            ToolCall::CameraShot(ShotPreset::DollyIn, SpeedPreset::Fast),
        );
        assert!(matches!(
            outcome,
            DispatchOutcome::Command(UIEvent::CameraShot {
                preset: ShotPreset::DollyIn,
                speed: SpeedPreset::Fast,
                target: None,
            })
        ));
    }

    #[test]
    fn camera_shot_look_at_selection_without_selection_is_rejected() {
        let world = World::new();
        let outcome = dispatch(
            &world,
            ToolCall::CameraShot(ShotPreset::LookAtSelection, SpeedPreset::Normal),
        );
        assert_eq!(expect_rejection(outcome), DispatchError::NothingSelected);
    }

    #[test]
    fn camera_shot_look_at_selection_with_selection_has_target() {
        let mut world = World::new();
        let hero = spawn_named(&mut world, "Hero");
        select(&mut world, hero);

        let outcome = dispatch(
            &world,
            ToolCall::CameraShot(ShotPreset::LookAtSelection, SpeedPreset::Normal),
        );
        assert!(matches!(
            expect_command(outcome),
            UIEvent::CameraShot {
                preset: ShotPreset::LookAtSelection,
                speed: SpeedPreset::Normal,
                target: Some(e)
            } if e == hero
        ));
    }
}
