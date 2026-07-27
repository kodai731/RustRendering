use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use serde_json::json;
use thyllore_render_core::flame_sdf::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: flame_sdf_bake <input.png> <output.fsdf> [--threshold 0.92] [--invert] [--max-dim 128]");
        std::process::exit(1);
    }

    let input_path: PathBuf = args[1].parse().unwrap();
    let output_path: PathBuf = args[2].parse().unwrap();

    let mut threshold = 0.92f32;
    let mut invert = false;
    let mut max_dim = 128usize;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--threshold" => {
                i += 1;
                threshold = args[i].parse().unwrap();
            }
            "--invert" => {
                invert = true;
            }
            "--max-dim" => {
                i += 1;
                max_dim = args[i].parse().unwrap();
            }
            _ => {}
        }
        i += 1;
    }

    let result = run(&input_path, &output_path, threshold, invert, max_dim);
    match result {
        Ok(output) => println!("{}", output),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

fn run(
    input_path: &PathBuf,
    output_path: &PathBuf,
    threshold: f32,
    invert: bool,
    max_dim: usize,
) -> io::Result<String> {
    let mut file = fs::File::open(input_path)?;
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

    // Build luminance buffer: luma = (r + g + b) / 3 / 255
    let mut lum = vec![0.0f32; total_pixels];
    for i in 0..total_pixels {
        let offset = i * bytes_per_pixel;
        let r = buf[offset] as f32;
        let g = buf[offset + 1] as f32;
        let b = buf[offset + 2] as f32;
        lum[i] = (r + g + b) / 3.0 / 255.0;
    }

    // Build silhouette mask
    let mask = build_silhouette_mask(&lum, width, height, threshold, invert);

    // Downsample
    let (downsampled, out_w, out_h) = downsample_mask(&mask, width, height, max_dim);

    // Compute signed distance
    let distances = compute_signed_distance(&downsampled, out_w, out_h);

    // Count inside pixels
    let inside_count = downsampled.iter().filter(|&&b| b).count();

    // Create and save FlameSdfImage
    let image = FlameSdfImage {
        width: out_w as u32,
        height: out_h as u32,
        data: distances,
    };

    save_flame_sdf(output_path.to_str().unwrap(), &image)?;

    Ok(json!({
        "width": out_w,
        "height": out_h,
        "inside_pixel_count": inside_count,
    })
    .to_string())
}
