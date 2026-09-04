use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::json;

use super::flag_value_resolve_from_args;

const BATCH_SEQUENCE_ANALYZE_FLAG: &str = "--batch-sequence-analyze";
const BATCH_SEQUENCE_DUMP_FLAG: &str = "--batch-sequence-dump";

#[derive(Clone, Debug)]
pub struct SequenceAnalyzeArgs {
    pub directories: Vec<(String, Option<u64>, Option<u64>)>,
    pub dump_path: String,
}

pub(super) fn batch_sequence_analyze_resolve_from_args(
    args: &[String],
) -> Result<Option<SequenceAnalyzeArgs>> {
    let directories: Vec<String> = args
        .windows(2)
        .enumerate()
        .filter(|(_, window)| window[0] == BATCH_SEQUENCE_ANALYZE_FLAG)
        .map(|(_, window)| window[1].clone())
        .collect();

    if directories.is_empty() {
        return Ok(None);
    }

    let dump_path = flag_value_resolve_from_args(args, BATCH_SEQUENCE_DUMP_FLAG)?
        .ok_or_else(|| anyhow::anyhow!("{BATCH_SEQUENCE_DUMP_FLAG} is required when {BATCH_SEQUENCE_ANALYZE_FLAG} is specified"))?;

    let parsed: Vec<(String, Option<u64>, Option<u64>)> = directories
        .iter()
        .map(|spec| {
            let parts: Vec<&str> = spec.split(',').collect();
            match parts.len() {
                1 => {
                    let dir = parts[0].trim().to_string();
                    Ok((dir, None, None))
                }
                2 | 3 => {
                    let dir = parts[0].trim().to_string();
                    let from: Option<u64> = if parts.len() >= 2 && !parts[1].is_empty() {
                        Some(parts[1].trim().parse::<u64>().map_err(|_| {
                            anyhow::anyhow!("invalid range in {BATCH_SEQUENCE_ANALYZE_FLAG} value '{spec}': from must be a number")
                        })?)
                    } else {
                        None
                    };
                    let to: Option<u64> = if parts.len() == 3 && !parts[2].is_empty() {
                        Some(parts[2].trim().parse::<u64>().map_err(|_| {
                            anyhow::anyhow!("invalid range in {BATCH_SEQUENCE_ANALYZE_FLAG} value '{spec}': to must be a number")
                        })?)
                    } else {
                        None
                    };
                    Ok((dir, from, to))
                }
                _ => bail!("invalid {BATCH_SEQUENCE_ANALYZE_FLAG} value '{spec}': expected <dir>[,<from>,<to>]"),
            }
        })
        .collect::<Result<_>>()?;

    Ok(Some(SequenceAnalyzeArgs {
        directories: parsed,
        dump_path,
    }))
}

/// Headless sequence analysis: find frame_*.png files in each directory, compute luminance,
/// extract descriptors, and write JSON to the dump path.
pub fn run_sequence_analyze_from_args(args: Vec<String>) -> Option<Result<()>> {
    let args_slice: Vec<String> = args;
    if !args_slice.iter().any(|a| a == BATCH_SEQUENCE_ANALYZE_FLAG) {
        return None;
    }

    Some(run_sequence_analyze(&args_slice))
}

pub(super) fn run_sequence_analyze(args: &[String]) -> Result<()> {
    let analyze_args = batch_sequence_analyze_resolve_from_args(args)?.ok_or_else(|| {
        anyhow::anyhow!("{BATCH_SEQUENCE_ANALYZE_FLAG} requires at least one directory argument")
    })?;

    let mut sequences = Vec::new();

    for (dir, from, to) in &analyze_args.directories {
        let entries =
            std::fs::read_dir(dir).with_context(|| format!("failed to read directory {dir}"))?;

        let mut frame_files: Vec<(u64, PathBuf)> = Vec::new();
        for entry in entries {
            let entry = entry.with_context(|| format!("failed to read entry in {dir}"))?;
            let path = entry.path();
            let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if filename.starts_with("frame_") && filename.ends_with(".png") {
                // Extract number from frame_NNNN.png
                let stem = &filename[6..filename.len() - 4];
                if let Ok(num) = stem.parse::<u64>() {
                    frame_files.push((num, path));
                }
            } else if filename.starts_with("frame_")
                && (filename.ends_with(".jpg") || filename.ends_with(".jpeg"))
            {
                bail!(
                    "found JPG file in directory {dir}: {filename} — only PNG files are supported"
                );
            }
        }

        frame_files.sort_by_key(|(num, _)| *num);

        // Apply range filter
        let filtered: Vec<(u64, PathBuf)> = if let (Some(f), Some(t)) = (from, to) {
            frame_files
                .into_iter()
                .filter(|(num, _)| *num >= *f && *num <= *t)
                .collect()
        } else if let Some(f) = from {
            frame_files
                .into_iter()
                .filter(|(num, _)| *num >= *f)
                .collect()
        } else if let Some(t) = to {
            frame_files
                .into_iter()
                .filter(|(num, _)| *num <= *t)
                .collect()
        } else {
            frame_files
        };

        if filtered.is_empty() {
            eprintln!("warning: no frame_*.png files found in {dir}");
            continue;
        }

        // Read FPS from meta.json
        let fps = read_fps_from_meta(dir);

        // Read each PNG and compute average luminance
        let mut frames: Vec<Vec<f32>> = Vec::new();
        let mut width: usize = 0;
        let mut height: usize = 0;

        for (num, path) in &filtered {
            let (w, h, luminance) = read_png_luminance(path).with_context(|| {
                format!("failed to read PNG at {} (frame #{num})", path.display())
            })?;
            if frames.is_empty() {
                width = w;
                height = h;
            } else if w != width || h != height {
                bail!(
                    "inconsistent frame size: expected {}x{}, got {}x{} at {}",
                    width,
                    height,
                    w,
                    h,
                    path.display()
                );
            }
            frames.push(luminance);
        }

        // Extract descriptors
        let descriptors =
            thyllore_texture_fit_core::sequence_descriptors::extract_sequence_descriptors(
                &frames, width, height, fps,
            )
            .with_context(|| format!("failed to extract descriptors for {dir}"))?;

        let descriptors_json = serde_json::to_value(&descriptors)?;
        sequences.push(json!({
            "dir": dir,
            "descriptors": descriptors_json,
        }));
    }

    let output = json!({"sequences": sequences});
    let dump_path = PathBuf::from(&analyze_args.dump_path);
    if let Some(parent) = dump_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(&dump_path, serde_json::to_string_pretty(&output)?).with_context(|| {
        format!(
            "failed to write sequence analysis dump to {}",
            dump_path.display()
        )
    })?;

    Ok(())
}

pub(super) fn read_fps_from_meta(dir: &str) -> f32 {
    let meta_path = PathBuf::from(dir).join("meta.json");
    if let Ok(content) = std::fs::read_to_string(&meta_path) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(fps) = value.get("fps").and_then(|v| v.as_f64()) {
                return fps as f32;
            }
        }
    }
    10.0
}

pub(super) fn read_png_luminance(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let decoder = png::Decoder::new(reader);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;

    let (width, height) = (info.width as usize, info.height as usize);

    let bytes_per_pixel = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        _ => bail!("unsupported PNG color type: {:?}", info.color_type),
    };

    let buf = &buf[..info.buffer_size()];
    let mut luminance = Vec::with_capacity(width * height);
    for chunk in buf.chunks(bytes_per_pixel) {
        let lum = match bytes_per_pixel {
            3 => (chunk[0] as f32 + chunk[1] as f32 + chunk[2] as f32) / 3.0,
            4 => (chunk[0] as f32 + chunk[1] as f32 + chunk[2] as f32) / 3.0,
            1 => chunk[0] as f32,
            2 => chunk[0] as f32,
            _ => unreachable!(),
        };
        luminance.push(lum);
    }

    Ok((width, height, luminance))
}
