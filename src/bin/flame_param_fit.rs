use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use serde_json::json;
use thyllore_effect_core::flame_fit::*;

fn median(sorted: &[f32]) -> Option<f32> {
    if sorted.is_empty() {
        return None;
    }
    percentile(sorted, 0.5)
}

fn classify_background(lum: &[f32], width: usize, height: usize) -> &'static str {
    let corner_size = 8usize.min(width.min(height));
    let corners: [[usize; 2]; 4] = [
        [0, 0],
        [width - corner_size, 0],
        [0, height - corner_size],
        [width - corner_size, height - corner_size],
    ];

    let mut white_count = 0u32;
    let mut black_count = 0u32;

    for &[cx, cy] in &corners {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for y in cy..cy + corner_size {
            for x in cx..cx + corner_size {
                sum += lum[y * width + x];
                count += 1;
            }
        }
        let mean = sum / count as f32;
        if mean > 0.8 {
            white_count += 1;
        } else if mean < 0.2 {
            black_count += 1;
        }
    }

    if white_count >= 3 {
        "white_bg"
    } else if black_count >= 3 {
        "black_bg"
    } else {
        "mixed"
    }
}

fn process_image(path: &std::path::Path) -> io::Result<serde_json::Value> {
    let filename = path.file_name().unwrap().to_string_lossy().to_string();

    let mut file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let decoder = png::Decoder::new(io::Cursor::new(&bytes));
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let width = info.width as usize;
    let height = info.height as usize;
    let color = info.color_type;
    let bytes_per_pixel = match color {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unsupported PNG color type",
            ))
        }
    };

    let total_pixels = width * height;
    let buf = &buf[..info.buffer_size()];

    // Build linear RGB and luminance buffers
    let mut lum = vec![0.0f32; total_pixels];
    let mut linear_rgb: Vec<[f32; 3]> = Vec::with_capacity(total_pixels);
    let mut srgb8_values: Vec<[u8; 3]> = Vec::with_capacity(total_pixels);

    for i in 0..total_pixels {
        let offset = i * bytes_per_pixel;
        let r = buf[offset] as f32 / 255.0;
        let g = buf[offset + 1] as f32 / 255.0;
        let b = buf[offset + 2] as f32 / 255.0;

        let rl = srgb_to_linear(r);
        let gl = srgb_to_linear(g);
        let bl = srgb_to_linear(b);

        linear_rgb.push([rl, gl, bl]);
        srgb8_values.push([buf[offset], buf[offset + 1], buf[offset + 2]]);
        lum[i] = luminance([rl, gl, bl]);
    }

    let bg_class = classify_background(&lum, width, height);

    // Saturated fraction
    let sat_count = srgb8_values.iter().filter(|s| is_saturated(**s)).count();
    let saturated_fraction = sat_count as f32 / total_pixels as f32;

    // Mask and coverage
    let mask = flame_mask(&lum, width, height, 0.12);
    let coverage = mask.iter().filter(|&&b| b).count() as f32 / total_pixels as f32;

    // Taper
    let profile = row_width_profile(&mask, width, height);
    let taper = fit_taper(&profile);
    let (tip_ratio, taper_power) = match taper {
        Some((t, p)) => (Some(t), Some(p)),
        None => (None, None),
    };

    // Edge width
    let edge_width = edge_width_profile(&lum, width, height, 0.1, 0.9);

    // Wiggle
    let wiggle = boundary_wiggle(&mask, width, height);
    let (wiggle_amplitude, wiggle_frequency) = match wiggle {
        Some((a, f)) => (Some(a), Some(f)),
        None => (None, None),
    };

    // Vertical profile peak
    let vprofile = vertical_luminance_profile(&lum, width, height);
    let peak_position = if vprofile.is_empty() {
        None
    } else {
        let max_idx = (0..vprofile.len())
            .max_by(|&a, &b| vprofile[a].partial_cmp(&vprofile[b]).unwrap())
            .unwrap();
        Some(max_idx as f32 / vprofile.len() as f32)
    };

    // Envelope estimate (for black_bg images)
    let (env_tip, env_power) = match taper {
        Some((t, p)) => (t, p),
        None => (0.10, 1.4),
    };
    // Crop the vertical profile to the flame's active row span and re-normalize to max 1
    let envelope_profile = crop_profile_to_span(&vprofile, 0.05)
        .map(|mut cropped| {
            let max_val = cropped.iter().cloned().fold(0.0f32, f32::max);
            if max_val > 1e-9 {
                for v in &mut cropped {
                    *v /= max_val;
                }
            }
            cropped
        })
        .unwrap_or(vprofile.clone());
    let envelope_estimate = fit_envelope_from_profile(&envelope_profile, env_tip, env_power);
    let (envelope_peak, envelope_base, envelope_tail) = match envelope_estimate {
        Some((p, v0, q)) => (Some(p), Some(v0), Some(q)),
        None => (None, None, None),
    };

    // Saturated envelope estimate (for black_bg images)
    let saturated_envelope_estimate =
        fit_envelope_from_profile_saturated(&envelope_profile, env_tip, env_power);
    let (envelope_peak_sat, envelope_base_sat, envelope_tail_sat, saturation_k) =
        match saturated_envelope_estimate {
            Some((p, v0, q, k)) => (Some(p), Some(v0), Some(q), Some(k)),
            None => (None, None, None, None),
        };

    if bg_class == "white_bg" {
        // Compute alpha = 1 - min_channel(linear srgb) and report alpha_p50/p95
        let mut alphas: Vec<f32> = linear_rgb
            .iter()
            .map(|c| {
                let min_c = c.iter().cloned().fold(f32::MAX, f32::min);
                1.0 - min_c
            })
            .collect();
        alphas.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha_p50 = percentile(&alphas, 0.5).unwrap_or(0.0);
        let alpha_p95 = percentile(&alphas, 0.95).unwrap_or(0.0);

        Ok(json!({
            "filename": filename,
            "bg_class": bg_class,
            "saturated_fraction": saturated_fraction,
            "mask_coverage": coverage,
            "tip_ratio": tip_ratio,
            "taper_power": taper_power,
            "edge_width": edge_width,
            "wiggle_amplitude": wiggle_amplitude,
            "wiggle_frequency": wiggle_frequency,
            "vertical_profile_peak": peak_position,
            "alpha_p50": alpha_p50,
            "alpha_p95": alpha_p95,
        }))
    } else {
        // Compute CCT stats over non-saturated pixels with luminance > 0.15
        let mut ccts: Vec<f32> = Vec::new();
        for i in 0..total_pixels {
            if !is_saturated(srgb8_values[i]) && lum[i] > 0.15 {
                if let Some(xy) = chromaticity_xy(linear_rgb[i]) {
                    if let Some(cct) = mccamy_cct(xy) {
                        ccts.push(cct);
                    }
                }
            }
        }
        ccts.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let cct_p10 = percentile(&ccts, 0.1);
        let cct_p50 = percentile(&ccts, 0.5);
        let cct_p95 = percentile(&ccts, 0.95);

        Ok(json!({
            "filename": filename,
            "bg_class": bg_class,
            "saturated_fraction": saturated_fraction,
            "cct_p10": cct_p10,
            "cct_p50": cct_p50,
            "cct_p95": cct_p95,
            "mask_coverage": coverage,
            "tip_ratio": tip_ratio,
            "taper_power": taper_power,
            "edge_width": edge_width,
            "wiggle_amplitude": wiggle_amplitude,
            "wiggle_frequency": wiggle_frequency,
            "vertical_profile_peak": peak_position,
            "envelope_peak": envelope_peak,
            "envelope_base": envelope_base,
            "envelope_tail": envelope_tail,
            "envelope_peak_sat": envelope_peak_sat,
            "envelope_base_sat": envelope_base_sat,
            "envelope_tail_sat": envelope_tail_sat,
            "saturation_k": saturation_k,
        }))
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: flame_param_fit <input_dir>");
        std::process::exit(1);
    }

    let input_dir = PathBuf::from(&args[1]);
    if !input_dir.is_dir() {
        eprintln!("Error: {} is not a directory", args[1]);
        std::process::exit(1);
    }

    let mut entries: Vec<PathBuf> = fs::read_dir(&input_dir)
        .expect("Failed to read input directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .collect();
    entries.sort();

    let mut black_bg_cct_p95: Vec<f32> = Vec::new();
    let mut black_bg_cct_p10: Vec<f32> = Vec::new();
    let mut black_bg_taper_tip: Vec<f32> = Vec::new();
    let mut black_bg_taper_power: Vec<f32> = Vec::new();
    let mut black_bg_edge_width: Vec<f32> = Vec::new();
    let mut black_bg_wiggle: Vec<f32> = Vec::new();
    let mut black_bg_envelope_peak: Vec<f32> = Vec::new();
    let mut black_bg_envelope_base: Vec<f32> = Vec::new();
    let mut black_bg_envelope_tail: Vec<f32> = Vec::new();

    for path in &entries {
        match process_image(path) {
            Ok(json_value) => {
                println!("{}", serde_json::to_string(&json_value).unwrap());

                // Collect stats for black_bg images
                if json_value.get("bg_class").and_then(|v| v.as_str()) == Some("black_bg") {
                    if let Some(v) = json_value.get("cct_p95").and_then(|v| v.as_f64()) {
                        black_bg_cct_p95.push(v as f32);
                    }
                    if let Some(v) = json_value.get("cct_p10").and_then(|v| v.as_f64()) {
                        black_bg_cct_p10.push(v as f32);
                    }
                    if let Some(v) = json_value.get("tip_ratio").and_then(|v| v.as_f64()) {
                        black_bg_taper_tip.push(v as f32);
                    }
                    if let Some(v) = json_value.get("taper_power").and_then(|v| v.as_f64()) {
                        black_bg_taper_power.push(v as f32);
                    }
                    if let Some(v) = json_value.get("edge_width").and_then(|v| v.as_f64()) {
                        black_bg_edge_width.push(v as f32);
                    }
                    if let Some(v) = json_value.get("wiggle_amplitude").and_then(|v| v.as_f64()) {
                        black_bg_wiggle.push(v as f32);
                    }
                    if let Some(v) = json_value.get("envelope_peak").and_then(|v| v.as_f64()) {
                        black_bg_envelope_peak.push(v as f32);
                    }
                    if let Some(v) = json_value.get("envelope_base").and_then(|v| v.as_f64()) {
                        black_bg_envelope_base.push(v as f32);
                    }
                    if let Some(v) = json_value.get("envelope_tail").and_then(|v| v.as_f64()) {
                        black_bg_envelope_tail.push(v as f32);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error processing {:?}: {}", path, e);
            }
        }
    }

    // Print aggregated line for black_bg images
    if !black_bg_cct_p95.is_empty() {
        black_bg_cct_p95.sort_by(|a, b| a.partial_cmp(b).unwrap());
        black_bg_cct_p10.sort_by(|a, b| a.partial_cmp(b).unwrap());
        black_bg_taper_tip.sort_by(|a, b| a.partial_cmp(b).unwrap());
        black_bg_taper_power.sort_by(|a, b| a.partial_cmp(b).unwrap());
        black_bg_edge_width.sort_by(|a, b| a.partial_cmp(b).unwrap());
        black_bg_wiggle.sort_by(|a, b| a.partial_cmp(b).unwrap());
        black_bg_envelope_peak.sort_by(|a, b| a.partial_cmp(b).unwrap());
        black_bg_envelope_base.sort_by(|a, b| a.partial_cmp(b).unwrap());
        black_bg_envelope_tail.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let agg = json!({
            "aggregated": true,
            "temperature_base_k": median(&black_bg_cct_p95),
            "temperature_tip_k": median(&black_bg_cct_p10),
            "taper_tip_ratio": median(&black_bg_taper_tip),
            "taper_power": median(&black_bg_taper_power),
            "edge_width": median(&black_bg_edge_width),
            "wiggle_amplitude": median(&black_bg_wiggle),
            "envelope_peak": median(&black_bg_envelope_peak),
            "envelope_base": median(&black_bg_envelope_base),
            "envelope_tail": median(&black_bg_envelope_tail),
        });
        println!("{}", serde_json::to_string(&agg).unwrap());
    }
}
