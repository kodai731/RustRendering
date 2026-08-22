pub mod caption;
pub mod detokenize;
pub mod keyframes;
pub mod session;
pub mod text_encoder;

use crate::copilot::camera_direction::detokenize::detokenize_sequence;
use crate::copilot::camera_direction::session::{CameraDirectionOnnxPaths, CameraDirectionSession};
use crate::copilot::camera_direction::text_encoder::TextEncoder;

/// Generate camera poses from a natural-language utterance.
///
/// Convenience function that chains: build_movement_caption -> TextEncoder encode ->
/// CameraDirectionSession generate_tokens -> detokenize_sequence.
pub fn generate_camera_poses(
    paths: &CameraDirectionOnnxPaths,
    caption: &str,
) -> anyhow::Result<Vec<[[f32; 4]; 4]>> {
    let mut encoder = TextEncoder::from_paths(&paths.text_encoder, &paths.tokenizer)?;
    let cond_embeds = encoder.encode(caption)?;

    let mut session = CameraDirectionSession::from_paths(paths)?;
    let tokens = session.generate_tokens(&cond_embeds)?;

    detokenize_sequence(&tokens).map_err(|e| anyhow::anyhow!("detokenization failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end decode against the production ONNX artifacts (`GENDOP_ONNX_DIR`). The
    /// forward caption must yield a finite 30-pose trajectory whose camera-local Z decreases
    /// (GenDoP "forward" = -Z), which is what collapses when the text encoder pads wrongly.
    #[test]
    fn forward_caption_decodes_to_a_forward_trajectory_when_onnx_dir_is_set() -> anyhow::Result<()>
    {
        let Ok(dir) = std::env::var("GENDOP_ONNX_DIR") else {
            return Ok(());
        };
        let base = std::path::PathBuf::from(dir);
        let paths = CameraDirectionOnnxPaths {
            decoder_step0: base.join("decoder_step0.onnx"),
            decoder_with_past: base.join("decoder_with_past.onnx"),
            embd_table: base.join("embd_table.bin"),
            text_encoder: base.join("text_encoder.onnx"),
            tokenizer: base.join("tokenizer.json"),
        };
        let caption = "The camera continuously moves forward throughout the entire sequence.";

        let poses = generate_camera_poses(&paths, caption)?;

        assert_eq!(poses.len(), 30, "expected 30 poses");
        assert!(
            poses.iter().flatten().flatten().all(|v| v.is_finite()),
            "pose matrices must be finite"
        );
        let z_first = poses[0][2][3];
        let z_last = poses[29][2][3];
        assert!(
            z_last < z_first - 0.2,
            "forward caption should move along -Z: first {z_first} last {z_last}"
        );
        Ok(())
    }
}
