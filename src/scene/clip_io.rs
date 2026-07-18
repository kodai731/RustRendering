use std::fs;
use std::path::Path;

use crate::animation::editable::EditableAnimationClip;

use super::error::{SceneError, SceneResult};
use super::format::{AnimationClipFile, ANIMATION_FORMAT_VERSION};

pub fn save_animation_clip(path: &Path, clip: &EditableAnimationClip) -> SceneResult<()> {
    thyllore_exporter_core::systems::ron::export_ron_clip(clip, path)
        .map_err(|e| SceneError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))
}

pub fn load_animation_clip(path: &Path) -> SceneResult<EditableAnimationClip> {
    if !path.exists() {
        return Err(SceneError::AnimationNotFound(path.to_path_buf()));
    }

    let content = fs::read_to_string(path)?;
    let clip_file: AnimationClipFile = ron::from_str(&content)?;

    if clip_file.version != ANIMATION_FORMAT_VERSION {
        return Err(SceneError::VersionMismatch {
            expected: ANIMATION_FORMAT_VERSION,
            found: clip_file.version,
        });
    }

    log!("Loaded animation clip from: {}", path.display());
    Ok(clip_file.clip)
}
