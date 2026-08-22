use std::path::Path;

use anyhow::{ensure, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

/// ONNX text encoder for the GenDoP camera direction copilot.
///
/// Converts a natural-language utterance into condition embeddings (cond_embeds)
/// by running it through a CLIP-style text encoder ONNX model. Unlike
/// `SentenceEncoder`, this returns the raw per-token hidden states without
/// mean pooling or normalization — shape [1, 77, 1024] flattened to Vec<f32>.
pub struct TextEncoder {
    tokenizer: Tokenizer,
    session: Session,
}

impl TextEncoder {
    /// Load the text encoder from ONNX and tokenizer JSON files.
    pub fn from_paths(text_encoder_onnx: &Path, tokenizer_json: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_json).map_err(|e| anyhow::anyhow!("{e}"))?;
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(text_encoder_onnx)?;

        Ok(Self { tokenizer, session })
    }

    /// Encode an utterance into condition embeddings.
    ///
    /// Tokenizes the utterance to a fixed length of 77 tokens (CLIP's
    /// num_cond_tokens), runs the ONNX model, and returns the
    /// `last_hidden_state` output as a flat Vec<f32> of length 78848
    /// (1 * 77 * 1024).
    pub fn encode(&mut self, utterance: &str) -> Result<Vec<f32>> {
        let pad_id = self.pad_token_id()?;

        // Configure padding and truncation on the tokenizer in-place.
        self.tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(77),
            pad_id,
            ..Default::default()
        }));
        self.tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 77,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("truncation failed: {e}"))?;

        let encoded = self
            .tokenizer
            .encode(utterance, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let ids: Vec<i64> = encoded.get_ids().iter().map(|id| *id as i64).collect();
        let mask: Vec<i64> = encoded
            .get_attention_mask()
            .iter()
            .map(|m| *m as i64)
            .collect();

        ensure!(
            ids.len() == 77,
            "tokenized input_ids length {} != 77",
            ids.len()
        );
        ensure!(
            mask.len() == 77,
            "attention_mask length {} != 77",
            mask.len()
        );

        let shape = vec![1_i64, 77_i64];
        let inputs_ids_tensor = Tensor::from_array((shape.clone(), ids))?;
        let attention_mask_tensor = Tensor::from_array((shape, mask))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => inputs_ids_tensor,
            "attention_mask" => attention_mask_tensor
        ])?;

        // The exported graph runs in fp16 (same as the decoder ONNX artifacts), so
        // last_hidden_state comes back as f16 -- extract as f16 then widen to f32 to
        // match CameraDirectionSession::generate_tokens's cond_embeds signature.
        let (output_shape, hidden) =
            outputs["last_hidden_state"].try_extract_tensor::<half::f16>()?;

        // Verify shape: [1, 77, 1024]
        ensure!(
            output_shape.len() == 3,
            "last_hidden_state rank {} != 3",
            output_shape.len()
        );
        ensure!(
            output_shape[0] == 1 && output_shape[1] == 77 && output_shape[2] == 1024,
            "last_hidden_state shape [{}, {}, {}]",
            output_shape[0],
            output_shape[1],
            output_shape[2]
        );

        Ok(hidden.iter().map(|v| v.to_f32()).collect())
    }

    /// CLIP has no dedicated pad token; HF's `CLIPTokenizer` sets `pad_token = eos_token`
    /// (`<|endoftext|>`) by default, matching GenDoP's `padding="max_length"` behavior.
    /// Falling back to token id `0` here would silently pad with whatever ordinary word
    /// happens to occupy that vocab slot.
    fn pad_token_id(&self) -> Result<u32> {
        self.tokenizer
            .token_to_id("<|endoftext|>")
            .context("tokenizer has no <|endoftext|> token to use as pad")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TextEncoder::encode` output must match the PyTorch/ONNXRuntime reference bit-for-bit
    /// (own verification, 2026-08-22: same utterance run through `text_encoder.onnx` directly
    /// in Python produced identical first-10 values `[-1.0761719, 0.3203125, -0.19140625, ...]`).
    #[test]
    fn encode_matches_onnxruntime_reference() -> Result<()> {
        let onnx_dir = match std::env::var("GENDOP_ONNX_DIR") {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Skipping encode_matches_onnxruntime_reference: GENDOP_ONNX_DIR not set ({e})");
                return Ok(());
            }
        };
        let dir = Path::new(&onnx_dir);
        let mut enc =
            TextEncoder::from_paths(&dir.join("text_encoder.onnx"), &dir.join("tokenizer.json"))?;
        let out = enc.encode("The camera slowly circles the subject while pulling back.")?;

        assert_eq!(out.len(), 77 * 1024);
        let expected_first10: [f32; 10] = [
            -1.0761719, 0.3203125, -0.19140625, -1.0839844, -0.4267578, 0.35791016, 1.6337891,
            -0.19055176, -2.3535156, -1.0048828,
        ];
        for (got, want) in out[..10].iter().zip(expected_first10.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got} want {want}");
        }
        Ok(())
    }

    /// Skipped unless `GENDOP_ONNX_DIR` points at a real exported tokenizer.json (same
    /// gate as `session.rs`'s `test_camera_direction_parity`) -- CLIP's real BPE vocab
    /// and special tokens can't be faked with a small hand-written tokenizer.json.
    #[test]
    fn tokenize_fixed_length_77_with_real_clip_pad_token() -> Result<()> {
        let onnx_dir = match std::env::var("GENDOP_ONNX_DIR") {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Skipping tokenize_fixed_length_77_with_real_clip_pad_token: GENDOP_ONNX_DIR not set ({e})");
                return Ok(());
            }
        };
        let tokenizer_path = Path::new(&onnx_dir).join("tokenizer.json");
        let mut tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;

        let pad_id = tokenizer
            .token_to_id("<|endoftext|>")
            .context("tokenizer has no <|endoftext|> token")?;

        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(77),
            pad_id,
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 77,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let encoded = tokenizer
            .encode("The camera slowly circles the subject.", true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids: Vec<u32> = encoded.get_ids().to_vec();
        let mask: Vec<u32> = encoded.get_attention_mask().to_vec();

        assert_eq!(ids.len(), 77, "input_ids should be exactly 77 tokens");
        assert_eq!(
            mask.len(),
            77,
            "attention_mask should match input_ids length"
        );
        // Padding fills with the real CLIP eos/pad token, not an arbitrary vocab slot.
        assert_eq!(*ids.last().unwrap(), pad_id);
        Ok(())
    }
}
