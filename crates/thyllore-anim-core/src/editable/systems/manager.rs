use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use anyhow::{Context, Result};

use crate::{AnimationClip, AnimationClipId, BoneId};

use super::super::components::clip::EditableAnimationClip;
use super::super::components::keyframe::SourceClipId;
use super::clip_convert::{clip_from_animation, clip_to_animation};

#[derive(Clone, Debug, Default)]
pub struct EditableClipManager {
    clips: HashMap<SourceClipId, EditableAnimationClip>,
    dirty_clips: HashSet<SourceClipId>,
    next_clip_id: SourceClipId,
    editable_to_anim_id: HashMap<SourceClipId, AnimationClipId>,
}

impl EditableClipManager {
    pub fn new() -> Self {
        Self {
            clips: HashMap::new(),
            dirty_clips: HashSet::new(),
            next_clip_id: 1,
            editable_to_anim_id: HashMap::new(),
        }
    }

    pub fn create_from_imported(
        &mut self,
        clip: &AnimationClip,
        bone_names: &HashMap<BoneId, String>,
    ) -> SourceClipId {
        let id = self.next_clip_id;
        self.next_clip_id += 1;

        let editable = clip_from_animation(id, clip, bone_names);
        self.clips.insert(id, editable);
        self.editable_to_anim_id.insert(id, clip.id);
        id
    }

    pub fn create_empty(&mut self, name: String) -> SourceClipId {
        let id = self.next_clip_id;
        self.next_clip_id += 1;

        let clip = EditableAnimationClip::new(id, name);
        self.clips.insert(id, clip);
        id
    }

    pub fn register_clip(&mut self, mut clip: EditableAnimationClip) -> SourceClipId {
        let id = self.next_clip_id;
        self.next_clip_id += 1;
        clip.id = id;
        self.clips.insert(id, clip);
        id
    }

    pub fn get(&self, id: SourceClipId) -> Option<&EditableAnimationClip> {
        self.clips.get(&id)
    }

    pub fn get_mut(&mut self, id: SourceClipId) -> Option<&mut EditableAnimationClip> {
        self.dirty_clips.insert(id);
        self.clips.get_mut(&id)
    }

    pub fn remove(&mut self, id: SourceClipId) -> Option<EditableAnimationClip> {
        self.dirty_clips.remove(&id);
        self.clips.remove(&id)
    }

    pub fn to_playable_clip(&self, id: SourceClipId) -> Option<AnimationClip> {
        self.clips.get(&id).map(|clip| clip_to_animation(clip))
    }

    pub fn is_dirty(&self, id: SourceClipId) -> bool {
        self.dirty_clips.contains(&id)
    }

    pub fn mark_clean(&mut self, id: SourceClipId) {
        self.dirty_clips.remove(&id);
    }

    pub fn mark_dirty(&mut self, id: SourceClipId) {
        if self.clips.contains_key(&id) {
            self.dirty_clips.insert(id);
        }
    }

    pub fn dirty_clip_ids(&self) -> impl Iterator<Item = &SourceClipId> {
        self.dirty_clips.iter()
    }

    pub fn all_clip_ids(&self) -> impl Iterator<Item = &SourceClipId> {
        self.clips.keys()
    }

    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    pub fn clear(&mut self) {
        self.clips.clear();
        self.dirty_clips.clear();
        self.editable_to_anim_id.clear();
    }

    pub fn sync_dirty_to_animation_clips(&mut self, anim_clips: &mut Vec<AnimationClip>) {
        for editable_id in self.dirty_clips.drain() {
            let (clip, anim_id) = match (
                self.clips.get(&editable_id),
                self.editable_to_anim_id.get(&editable_id),
            ) {
                (Some(c), Some(&id)) => (c, id),
                _ => continue,
            };

            let mut playable = clip_to_animation(clip);
            playable.id = anim_id;

            if let Some(target) = anim_clips.iter_mut().find(|c| c.id == anim_id) {
                *target = playable;
            }
        }
    }

    pub fn clip_names(&self) -> Vec<(SourceClipId, String)> {
        self.clips
            .iter()
            .map(|(&id, clip)| (id, clip.name.clone()))
            .collect()
    }

    pub fn save_to_file(&self, id: SourceClipId, path: &Path) -> Result<()> {
        let clip = self.clips.get(&id).context("Clip not found")?;

        let file =
            fs::File::create(path).with_context(|| format!("Failed to create file: {:?}", path))?;
        let writer = BufWriter::new(file);

        ron::ser::to_writer_pretty(writer, clip, ron::ser::PrettyConfig::default())
            .with_context(|| format!("Failed to serialize clip to: {:?}", path))?;

        Ok(())
    }

    pub fn load_from_file(&mut self, path: &Path) -> Result<SourceClipId> {
        let file =
            fs::File::open(path).with_context(|| format!("Failed to open file: {:?}", path))?;
        let reader = BufReader::new(file);

        let mut clip: EditableAnimationClip = ron::de::from_reader(reader)
            .with_context(|| format!("Failed to deserialize clip from: {:?}", path))?;

        let id = self.next_clip_id;
        self.next_clip_id += 1;
        clip.id = id;
        clip.source_path = Some(path.to_string_lossy().to_string());

        self.clips.insert(id, clip);
        Ok(id)
    }
}
