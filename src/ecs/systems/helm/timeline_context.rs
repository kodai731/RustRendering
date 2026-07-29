use crate::animation::editable::{BoneTrack, EditableAnimationClip};
use crate::ecs::resource::{ClipLibrary, TimelineState};
use crate::helm::systems::seek::TimelineContext;

/// Build a `TimelineContext` from the ECS timeline state and clip library.
pub fn build_timeline_context(timeline: &TimelineState, clips: &ClipLibrary) -> TimelineContext {
    let current_time = timeline.current_time;

    let Some(clip_id) = timeline.current_clip_id else {
        return TimelineContext {
            current_time,
            ..Default::default()
        };
    };

    let clip: &EditableAnimationClip = match clips.get(clip_id) {
        Some(c) => c,
        None => {
            return TimelineContext {
                current_time,
                ..Default::default()
            };
        }
    };

    let duration = clip.duration;

    let mut keyframe_times: Vec<f32> = clip
        .tracks
        .values()
        .flat_map(|track: &BoneTrack| track.collect_all_keyframe_times())
        .collect();

    // Deduplicate by f32::to_bits equality (exact bit representation match)
    keyframe_times.sort_unstable_by(|a, b| a.total_cmp(b));
    keyframe_times.dedup_by(|a, b| a.to_bits() == b.to_bits());

    TimelineContext {
        current_time,
        duration,
        keyframe_times,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::editable::{EditableKeyframe, PropertyCurve};
    use crate::animation::BoneId;
    use std::collections::HashMap;

    fn make_clip(id: u64, duration: f32, tracks: HashMap<BoneId, BoneTrack>) -> EditableAnimationClip {
        let mut clip = EditableAnimationClip::new(id, "test".to_string());
        clip.duration = duration;
        clip.tracks = tracks;
        clip
    }

    fn insert_keyframes(curve: &mut PropertyCurve, times: &[f32]) {
        let mut id_counter: u64 = 1;
        for time in times {
            curve.keyframes.push(EditableKeyframe::new(id_counter, *time, 0.0));
            id_counter += 1;
        }
    }

    #[test]
    fn clip_not_selected_returns_default_with_current_time() {
        let timeline = TimelineState::new();
        let clips = ClipLibrary::new();

        let context = build_timeline_context(&timeline, &clips);

        assert_eq!(context.current_time, 0.0);
        assert_eq!(context.duration, 0.0);
        assert!(context.keyframe_times.is_empty());
    }

    #[test]
    fn clip_not_found_returns_default_with_current_time() {
        let mut timeline = TimelineState::new();
        timeline.current_clip_id = Some(999);
        timeline.current_time = 1.5;
        let clips = ClipLibrary::new();

        let context = build_timeline_context(&timeline, &clips);

        assert_eq!(context.current_time, 1.5);
        assert_eq!(context.duration, 0.0);
        assert!(context.keyframe_times.is_empty());
    }

    #[test]
    fn multiple_tracks_deduplicated_and_sorted() {
        let mut timeline = TimelineState::new();
        timeline.current_time = 2.0;

        let mut clips = ClipLibrary::new();
        let source_id: crate::animation::editable::SourceClipId = 1;

        let mut track_a = BoneTrack::new(1, "bone_a".to_string(), 1);
        insert_keyframes(&mut track_a.translation_x, &[3.0, 1.0, 5.0]);

        // Overlapping keys at 1.0 and 3.0, plus a unique one at 4.0
        let mut track_b = BoneTrack::new(2, "bone_b".to_string(), 11);
        insert_keyframes(&mut track_b.translation_x, &[1.0, 4.0, 3.0]);

        let mut tracks = HashMap::new();
        tracks.insert(1, track_a);
        tracks.insert(2, track_b);

        let clip = make_clip(source_id, 10.0, tracks);
        clips.source_clips.insert(
            source_id,
            crate::animation::editable::SourceClip::new(source_id, clip),
        );
        timeline.current_clip_id = Some(source_id);

        let context = build_timeline_context(&timeline, &clips);

        // Expected: [1.0, 3.0, 4.0, 5.0] — deduplicated and sorted ascending
        assert_eq!(context.keyframe_times, vec![1.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn duration_is_pulled_from_clip() {
        let mut timeline = TimelineState::new();
        timeline.current_time = 0.0;

        let mut clips = ClipLibrary::new();
        let source_id: crate::animation::editable::SourceClipId = 1;

        let clip = make_clip(source_id, 7.5, HashMap::new());
        clips.source_clips.insert(
            source_id,
            crate::animation::editable::SourceClip::new(source_id, clip),
        );
        timeline.current_clip_id = Some(source_id);

        let context = build_timeline_context(&timeline, &clips);

        assert_eq!(context.duration, 7.5);
    }
}
