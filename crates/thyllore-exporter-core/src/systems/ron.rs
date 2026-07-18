use std::fs;
use std::path::Path;

use thyllore_anim_core::editable::{AnimationClipFile, EditableAnimationClip};

pub fn export_ron_clip(clip: &EditableAnimationClip, output_path: &Path) -> anyhow::Result<()> {
    let clip_file = AnimationClipFile::new(clip.clone());
    let config = ron::ser::PrettyConfig::new()
        .depth_limit(10)
        .separate_tuple_members(true)
        .enumerate_arrays(false);

    let content = ron::ser::to_string_pretty(&clip_file, config)?;

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(output_path, content)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_and_reload_ron_clip() {
        let clip = EditableAnimationClip::new(1, "test_clip".to_string());
        let output_path = std::path::PathBuf::from("/tmp/test_ron_clip_export.ron");

        export_ron_clip(&clip, &output_path).unwrap();

        let content = fs::read_to_string(&output_path).unwrap();
        let loaded: AnimationClipFile = ron::from_str(&content).unwrap();

        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.clip.name, "test_clip");
        assert_eq!(loaded.clip.id, 1);

        fs::remove_file(&output_path).ok();
    }
}
