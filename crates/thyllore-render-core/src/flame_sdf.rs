use std::fs;
use std::io::{self, Read, Write};

#[derive(Clone, Debug)]
pub struct FlameSdfImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

/// Build a silhouette mask from luma values.
/// A pixel is "inside" (true) if its luma < threshold.
/// If `invert` is true, the condition is flipped (luma >= threshold is inside).
pub fn build_silhouette_mask(luma: &[f32], width: usize, height: usize, threshold: f32, invert: bool) -> Vec<bool> {
    let total = width * height;
    let mut mask = Vec::with_capacity(total);
    for i in 0..total {
        let inside = if invert {
            luma[i] >= threshold
        } else {
            luma[i] < threshold
        };
        mask.push(inside);
    }
    mask
}

/// Integer downsampling of a boolean mask to fit within `max_dim`.
/// A block is true if the majority of its pixels are true.
pub fn downsample_mask(mask: &[bool], width: usize, height: usize, max_dim: usize) -> (Vec<bool>, usize, usize) {
    let scale_x = (width as f32 / max_dim as f32).ceil() as usize;
    let scale_y = (height as f32 / max_dim as f32).ceil() as usize;
    let out_w = width.div_ceil(scale_x);
    let out_h = height.div_ceil(scale_y);

    let total = out_w * out_h;
    let mut out_mask = Vec::with_capacity(total);

    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut true_count = 0usize;
            let mut total_count = 0usize;
            for dy in 0..scale_y {
                let sy = (oy * scale_y + dy).min(height - 1);
                for dx in 0..scale_x {
                    let sx = (ox * scale_x + dx).min(width - 1);
                    if mask[sy * width + sx] {
                        true_count += 1;
                    }
                    total_count += 1;
                }
            }
            out_mask.push(true_count > total_count / 2);
        }
    }

    (out_mask, out_w, out_h)
}

/// Compute signed distance transform from a boolean mask.
/// Boundary pixels are those where any of the 4 neighbors has a different mask value.
/// Distance is negative for inside pixels, positive for outside.
/// Normalized by height. If no boundaries exist (all true or all false), returns all 1.0.
pub fn compute_signed_distance(mask: &[bool], width: usize, height: usize) -> Vec<f32> {
    let total = width * height;

    // Identify boundary pixels
    let mut boundaries: Vec<(usize, usize)> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let val = mask[idx];
            let is_boundary = {
                let mut b = false;
                // Check 4 neighbors
                if y > 0 && mask[(y - 1) * width + x] != val {
                    b = true;
                }
                if !b && y < height - 1 && mask[(y + 1) * width + x] != val {
                    b = true;
                }
                if !b && x > 0 && mask[y * width + (x - 1)] != val {
                    b = true;
                }
                if !b && x < width - 1 && mask[y * width + (x + 1)] != val {
                    b = true;
                }
                b
            };
            if is_boundary {
                boundaries.push((x, y));
            }
        }
    }

    // If no boundaries, return all 1.0
    if boundaries.is_empty() {
        return vec![1.0f32; total];
    }

    let mut distances = Vec::with_capacity(total);
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let inside = mask[idx];

            // Find minimum distance to any boundary pixel
            let mut min_dist = f32::MAX;
            for &(bx, by) in &boundaries {
                let dx = (x as f32 - bx as f32).abs();
                let dy = (y as f32 - by as f32).abs();
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                }
            }

            // Sign: negative for inside, positive for outside
            let signed = if inside { -min_dist } else { min_dist };
            // Normalize by height
            distances.push(signed / height as f32);
        }
    }

    distances
}

/// Save FlameSdfImage to a binary file.
/// Format: magic b"FSDF1" (5 bytes) + u32 LE width + u32 LE height + f32 LE data array.
pub fn save_flame_sdf(path: &str, image: &FlameSdfImage) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(b"FSDF1")?;
    file.write_all(&image.width.to_le_bytes())?;
    file.write_all(&image.height.to_le_bytes())?;
    for &val in &image.data {
        file.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}

/// Load FlameSdfImage from a binary file.
pub fn load_flame_sdf(path: &str) -> io::Result<FlameSdfImage> {
    let mut file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    if bytes.len() < 13 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
    }

    if &bytes[..5] != b"FSDF1" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic"));
    }

    let width = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
    let height = u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]);

    let expected_len = 13 + (width * height) as usize * 4;
    if bytes.len() < expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated data array",
        ));
    }

    let mut data = Vec::with_capacity((width * height) as usize);
    let data_bytes = &bytes[13..expected_len];
    for chunk in data_bytes.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        data.push(val);
    }

    Ok(FlameSdfImage { width, height, data })
}

/// Encode FlameSdfImage as RGBA8 bytes for Vulkan texture upload.
/// Each pixel d is mapped to ((d.clamp(-0.5, 0.5) + 0.5) * 255.0).round() as u8
/// for all 4 channels (RGBA), alpha always 255.
pub fn encode_sdf_rgba8(sdf: &FlameSdfImage) -> Vec<u8> {
    let mut out = Vec::with_capacity((sdf.width * sdf.height * 4) as usize);
    for &d in &sdf.data {
        let v = ((d.clamp(-0.5, 0.5) + 0.5) * 255.0).round() as u8;
        out.push(v);
        out.push(v);
        out.push(v);
        out.push(255);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdf_center_negative_corners_positive() {
        // 8x8 mask with a 3x3 block of true pixels in center (cols 2-4, rows 2-4)
        let mut mask = vec![false; 64];
        for y in 2..=4 {
            for x in 2..=4 {
                mask[y * 8 + x] = true;
            }
        }

        let sdf = compute_signed_distance(&mask, 8, 8);

        // Center pixel (3, 3) is inside and not on boundary -> should be negative
        let center_idx = 3 * 8 + 3;
        assert!(
            sdf[center_idx] < 0.0,
            "center pixel should be negative, got {}",
            sdf[center_idx]
        );

        // Corner (0, 0) is outside -> should be positive
        let corner_idx = 0 * 8 + 0;
        assert!(
            sdf[corner_idx] > 0.0,
            "corner pixel should be positive, got {}",
            sdf[corner_idx]
        );

        // Corner (7, 7) is outside -> should be positive
        let corner_idx = 7 * 8 + 7;
        assert!(
            sdf[corner_idx] > 0.0,
            "corner pixel (7,7) should be positive, got {}",
            sdf[corner_idx]
        );
    }

    #[test]
    fn test_save_load_roundtrip() {
        let temp_path = std::env::temp_dir().join("flame_sdf_test.fsdf");

        let image = FlameSdfImage {
            width: 4,
            height: 4,
            data: vec![1.0, -0.5, 0.25, -1.0, 0.0, 0.5, -0.75, 0.3, -0.1, 0.9, -0.3, 0.6, 0.4, -0.8, 0.15, -0.2],
        };

        save_flame_sdf(temp_path.to_str().unwrap(), &image).unwrap();
        let loaded = load_flame_sdf(temp_path.to_str().unwrap()).unwrap();

        assert_eq!(loaded.width, image.width);
        assert_eq!(loaded.height, image.height);
        assert_eq!(loaded.data, image.data);

        fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_determinism() {
        let mask: Vec<bool> = [true, false, true, false, false, true, false, true, true, false, true, false, false, true, false, true].to_vec();

        let sdf1 = compute_signed_distance(&mask, 4, 4);
        let sdf2 = compute_signed_distance(&mask, 4, 4);

        assert_eq!(sdf1, sdf2, "same input should produce identical output");
    }
}
