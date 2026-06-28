use std::path::Path;

use anyhow::{anyhow, Result};

use thyllore_animation::headless::{render_usd_to_png, render_usd_to_png_gpu, CameraConfig};

fn parse_vec3(text: &str) -> Result<[f32; 3]> {
    let parts: Vec<f32> = text
        .split(',')
        .map(|v| v.trim().parse::<f32>())
        .collect::<std::result::Result<_, _>>()?;
    match parts.as_slice() {
        [x, y, z] => Ok([*x, *y, *z]),
        _ => Err(anyhow!("expected 'x,y,z', got '{}'", text)),
    }
}

struct Args {
    usd_path: String,
    out_path: String,
    use_gpu: bool,
    resolution: u32,
    camera: CameraConfig,
}

fn parse_args() -> Result<Args> {
    let mut usd_path = None;
    let mut out_path = None;
    let mut position = [0.0, -3.0, 0.9];
    let mut target = [0.0, 0.0, 0.7];
    let mut fov_deg = 40.0;
    let mut resolution = 256;
    let mut frame = 1;
    let mut use_gpu = false;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| anyhow!("missing value for {}", flag))
        };
        match flag.as_str() {
            "--render-usd" => usd_path = Some(value()?),
            "--out" => out_path = Some(value()?),
            "--camera-pos" => position = parse_vec3(&value()?)?,
            "--camera-target" => target = parse_vec3(&value()?)?,
            "--fov" => fov_deg = value()?.parse()?,
            "--resolution" => resolution = value()?.parse()?,
            "--frame" => frame = value()?.parse()?,
            "--gpu" => use_gpu = true,
            "--exit" => {}
            other => return Err(anyhow!("unknown argument: {}", other)),
        }
    }

    Ok(Args {
        usd_path: usd_path.ok_or_else(|| anyhow!("--render-usd <file> is required"))?,
        out_path: out_path.ok_or_else(|| anyhow!("--out <png> is required"))?,
        use_gpu,
        resolution,
        camera: CameraConfig {
            position,
            target,
            fov_deg,
            resolution,
            frame,
        },
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;

    if args.use_gpu {
        unsafe {
            render_usd_to_png_gpu(
                &args.usd_path,
                args.resolution,
                args.resolution,
                Path::new(&args.out_path),
            )?;
        }
    } else {
        render_usd_to_png(&args.usd_path, &args.camera, Path::new(&args.out_path))?;
    }

    println!("Render written to: {}", args.out_path);
    Ok(())
}
