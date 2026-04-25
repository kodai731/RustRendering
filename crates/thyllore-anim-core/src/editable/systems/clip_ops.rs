use std::collections::HashMap;

use crate::editable::components::clip::EditableAnimationClip;
use crate::editable::components::track::BoneTrack;
use crate::BoneId;

pub fn clip_recalculate_duration(clip: &mut EditableAnimationClip) {
    let mut max_time: f32 = 0.0;

    for track in clip.tracks.values() {
        for curve in track.all_curves() {
            if let Some(last_kf) = curve.keyframes.last() {
                max_time = max_time.max(last_kf.time);
            }
        }
    }

    clip.duration = max_time;
}

pub fn clip_remap_bone_ids(
    clip: &mut EditableAnimationClip,
    name_to_new_id: &HashMap<String, BoneId>,
) {
    let old_tracks: Vec<(BoneId, BoneTrack)> = clip.tracks.drain().collect();
    for (_, mut track) in old_tracks {
        let new_id = match name_to_new_id.get(&track.bone_name) {
            Some(&id) => id,
            None => continue,
        };
        track.bone_id = new_id;
        clip.tracks.insert(new_id, track);
    }
}
