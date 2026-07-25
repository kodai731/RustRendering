use crate::orchestrator::components::tool_call::SeekPosition;

/// Timeline facts a seek needs. Kept separate from `TimelineState` so the
/// resolution stays a pure function that unit tests can drive directly.
#[derive(Clone, Debug, Default)]
pub struct TimelineContext {
    pub current_time: f32,
    pub duration: f32,
    pub keyframe_times: Vec<f32>,
}

/// Resolves a symbolic seek target to a concrete time. When no keyframe lies in
/// the requested direction the playhead stays where it is, so a repeated
/// "next keyframe" never walks off the end of the clip.
pub fn resolve_seek_time(context: &TimelineContext, position: SeekPosition) -> f32 {
    match position {
        SeekPosition::Start => 0.0,
        SeekPosition::End => context.duration.max(0.0),
        SeekPosition::NextKey => find_next_keyframe(context).unwrap_or(context.current_time),
        SeekPosition::PrevKey => find_previous_keyframe(context).unwrap_or(context.current_time),
    }
}

fn find_next_keyframe(context: &TimelineContext) -> Option<f32> {
    context
        .keyframe_times
        .iter()
        .copied()
        .filter(|time| *time > context.current_time)
        .fold(None, |nearest: Option<f32>, time| {
            Some(nearest.map_or(time, |best| best.min(time)))
        })
}

fn find_previous_keyframe(context: &TimelineContext) -> Option<f32> {
    context
        .keyframe_times
        .iter()
        .copied()
        .filter(|time| *time < context.current_time)
        .fold(None, |nearest: Option<f32>, time| {
            Some(nearest.map_or(time, |best| best.max(time)))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context_with_keys(current_time: f32, keyframe_times: &[f32]) -> TimelineContext {
        TimelineContext {
            current_time,
            duration: 10.0,
            keyframe_times: keyframe_times.to_vec(),
        }
    }

    #[test]
    fn start_resolves_to_zero_regardless_of_position() {
        let context = context_with_keys(4.5, &[0.0, 2.0, 8.0]);
        assert_eq!(resolve_seek_time(&context, SeekPosition::Start), 0.0);
    }

    #[test]
    fn end_resolves_to_the_clip_duration() {
        let context = context_with_keys(4.5, &[0.0, 2.0, 8.0]);
        assert_eq!(resolve_seek_time(&context, SeekPosition::End), 10.0);
    }

    #[test]
    fn end_never_returns_a_negative_time() {
        let context = TimelineContext {
            current_time: 0.0,
            duration: -1.0,
            keyframe_times: Vec::new(),
        };
        assert_eq!(resolve_seek_time(&context, SeekPosition::End), 0.0);
    }

    #[test]
    fn next_key_picks_the_nearest_later_keyframe() {
        let context = context_with_keys(2.5, &[0.0, 2.0, 4.0, 8.0]);
        assert_eq!(resolve_seek_time(&context, SeekPosition::NextKey), 4.0);
    }

    #[test]
    fn prev_key_picks_the_nearest_earlier_keyframe() {
        let context = context_with_keys(5.0, &[0.0, 2.0, 4.0, 8.0]);
        assert_eq!(resolve_seek_time(&context, SeekPosition::PrevKey), 4.0);
    }

    #[test]
    fn keyframe_order_in_the_input_does_not_matter() {
        let context = context_with_keys(2.5, &[8.0, 0.0, 4.0, 2.0]);
        assert_eq!(resolve_seek_time(&context, SeekPosition::NextKey), 4.0);
        assert_eq!(resolve_seek_time(&context, SeekPosition::PrevKey), 2.0);
    }

    #[test]
    fn a_keyframe_at_the_current_time_is_not_a_neighbour() {
        let context = context_with_keys(4.0, &[0.0, 4.0, 8.0]);
        assert_eq!(resolve_seek_time(&context, SeekPosition::NextKey), 8.0);
        assert_eq!(resolve_seek_time(&context, SeekPosition::PrevKey), 0.0);
    }

    #[test]
    fn seeking_past_the_last_keyframe_holds_the_playhead() {
        let context = context_with_keys(9.0, &[0.0, 4.0, 8.0]);
        assert_eq!(resolve_seek_time(&context, SeekPosition::NextKey), 9.0);
    }

    #[test]
    fn seeking_before_the_first_keyframe_holds_the_playhead() {
        let context = context_with_keys(0.0, &[0.0, 4.0, 8.0]);
        assert_eq!(resolve_seek_time(&context, SeekPosition::PrevKey), 0.0);
    }

    #[test]
    fn a_clip_without_keyframes_holds_the_playhead() {
        let context = context_with_keys(3.0, &[]);
        assert_eq!(resolve_seek_time(&context, SeekPosition::NextKey), 3.0);
        assert_eq!(resolve_seek_time(&context, SeekPosition::PrevKey), 3.0);
    }
}
