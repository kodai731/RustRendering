use cgmath::Vector3;
use serde::Deserialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::ecs::systems::{
    apply_skinning, compute_bone_local_offsets, compute_display_transforms_with_skeleton,
    compute_pose_global_transforms, create_pose_from_rest, sample_clip_to_pose,
};
use crate::loader::ModelLoadResult;

const USD_ASSET: &str = "blender/20260624_ellie_animation/ellie_animation.usdc";

#[derive(Deserialize)]
struct BlenderUsdVertices {
    fps: f32,
    frame_start: i32,
    frames: Vec<BlenderFrameVertices>,
}

#[derive(Deserialize)]
struct BlenderFrameVertices {
    frame: i32,
    vertices: Vec<[f32; 3]>,
}

fn read_blender_path() -> Option<String> {
    let content = std::fs::read_to_string(Path::new(".claude/local/paths.md")).ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("- BlenderPath = ") {
            let path = rest.trim().to_string();
            if Path::new(&path).exists() {
                return Some(path);
            }
        }
    }
    None
}

fn run_blender_vertices(blender_path: &str, model_path: &str) -> BlenderUsdVertices {
    let script = Path::new("scripts/blender_usd_vertices.py");
    let model_abs = Path::new(model_path)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(model_path));
    let output_file = std::env::temp_dir().join(format!(
        "blender_usd_vertices_{:?}.json",
        std::thread::current().id()
    ));

    let output = Command::new(blender_path)
        .args([
            "--background",
            "--python",
            &script.to_string_lossy(),
            "--",
            &model_abs.to_string_lossy(),
            &output_file.to_string_lossy(),
        ])
        .output()
        .expect("Failed to run Blender");

    if !output.status.success() {
        panic!(
            "Blender script failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let json = std::fs::read_to_string(&output_file).expect("Failed to read Blender output JSON");
    std::fs::remove_file(&output_file).ok();
    serde_json::from_str(&json).expect("Failed to parse Blender vertices JSON")
}

fn blender_point(point: &[f32; 3]) -> Vector3<f32> {
    Vector3::new(point[0], point[1], point[2])
}

fn compute_thyllore_skinned_cloud(load_result: &ModelLoadResult, time: f32) -> Vec<Vector3<f32>> {
    let skeleton = load_result.skeletons.first().expect("No skeleton");
    let clip = load_result.clips.first().expect("No animation clip");

    let mut pose = create_pose_from_rest(skeleton);
    sample_clip_to_pose(clip, time, skeleton, &mut pose, false);
    let global_transforms = compute_pose_global_transforms(skeleton, &pose);

    let mut cloud = Vec::new();
    for mesh in &load_result.meshes {
        let Some(skin_data) = &mesh.skin_data else {
            continue;
        };

        let vertex_count = skin_data.base_positions.len();
        let mut positions = vec![Vector3::new(0.0, 0.0, 0.0); vertex_count];
        let mut normals = vec![Vector3::new(0.0, 0.0, 0.0); vertex_count];
        apply_skinning(
            skin_data,
            &global_transforms,
            skeleton,
            &mut positions,
            &mut normals,
        );
        cloud.extend(positions);
    }
    cloud
}

struct SpatialGrid {
    cell_size: f32,
    cells: std::collections::HashMap<(i32, i32, i32), Vec<Vector3<f32>>>,
}

impl SpatialGrid {
    fn build(points: &[Vector3<f32>], cell_size: f32) -> Self {
        let cell_size = cell_size.max(1e-6);
        let mut cells: std::collections::HashMap<(i32, i32, i32), Vec<Vector3<f32>>> =
            std::collections::HashMap::new();
        for &p in points {
            cells.entry(Self::key(p, cell_size)).or_default().push(p);
        }
        Self { cell_size, cells }
    }

    fn key(p: Vector3<f32>, cell_size: f32) -> (i32, i32, i32) {
        (
            (p.x / cell_size).floor() as i32,
            (p.y / cell_size).floor() as i32,
            (p.z / cell_size).floor() as i32,
        )
    }

    fn nearest_distance(&self, query: Vector3<f32>) -> f32 {
        let (cx, cy, cz) = Self::key(query, self.cell_size);
        let mut best = f32::MAX;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(points) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &p in points {
                            let diff = p - query;
                            best = best.min(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
                        }
                    }
                }
            }
        }
        best.sqrt()
    }
}

fn cloud_diagonal(points: &[Vector3<f32>]) -> f32 {
    let mut min = Vector3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max = Vector3::new(f32::MIN, f32::MIN, f32::MIN);
    for p in points {
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        min.z = min.z.min(p.z);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
        max.z = max.z.max(p.z);
    }
    let diff = max - min;
    (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z).sqrt()
}

fn assert_clouds_match(label: &str, thyllore: &[Vector3<f32>], blender: &[[f32; 3]]) {
    let blender_cloud: Vec<Vector3<f32>> = blender.iter().map(blender_point).collect();
    let diagonal = cloud_diagonal(&blender_cloud);
    let tolerance = diagonal * 0.02;
    let grid = SpatialGrid::build(thyllore, tolerance);

    let mut distances: Vec<f32> = blender_cloud
        .iter()
        .map(|&p| grid.nearest_distance(p))
        .collect();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile =
        |q: f32| distances[((distances.len() as f32 * q) as usize).min(distances.len() - 1)];
    let p50 = percentile(0.50);
    let p90 = percentile(0.90);
    let p99 = percentile(0.99);

    eprintln!(
        "[{}] samples={} diagonal={:.4} tol={:.4} | nn p50={:.5} p90={:.5} p99={:.5}",
        label,
        distances.len(),
        diagonal,
        tolerance,
        p50,
        p90,
        p99,
    );

    assert!(
        p90 <= tolerance,
        "[{}] 90th-percentile nearest-neighbor distance {:.5} exceeds tolerance {:.5}",
        label,
        p90,
        tolerance,
    );
}

#[test]
fn test_usd_vertices_match_blender() {
    let Some(blender_path) = read_blender_path() else {
        eprintln!("Skipping: BlenderPath not configured in .claude/local/paths.md");
        return;
    };
    if !Path::new(USD_ASSET).exists() {
        eprintln!("Skipping: {} not found", USD_ASSET);
        return;
    }

    let usd_result = crate::loader::usd::load_usd_file(USD_ASSET).expect("Failed to load USD");
    let load_result = ModelLoadResult::from_usd(usd_result);
    assert!(
        load_result.meshes.iter().any(|m| m.skin_data.is_some()),
        "expected skinned meshes in USD asset"
    );

    let blender = run_blender_vertices(&blender_path, USD_ASSET);

    for frame_data in &blender.frames {
        if frame_data.vertices.is_empty() {
            continue;
        }
        let time = (frame_data.frame - blender.frame_start) as f32 / blender.fps;
        let cloud = compute_thyllore_skinned_cloud(&load_result, time);
        assert_clouds_match(
            &format!("USD frame {}", frame_data.frame),
            &cloud,
            &frame_data.vertices,
        );
    }
}

const RENDER_RESOLUTION: u32 = 256;
const RENDER_FRAME: i32 = 1;
const RENDER_CAMERA_POS: [f32; 3] = [0.0, -3.0, 0.9];
const RENDER_CAMERA_TARGET: [f32; 3] = [0.0, 0.0, 0.7];
const RENDER_FOV_DEG: f32 = 40.0;

fn render_camera_config() -> String {
    format!(
        "{{\"camera_pos\":[{},{},{}],\"camera_target\":[{},{},{}],\"fov_deg\":{},\"resolution\":{},\"frame\":{}}}",
        RENDER_CAMERA_POS[0],
        RENDER_CAMERA_POS[1],
        RENDER_CAMERA_POS[2],
        RENDER_CAMERA_TARGET[0],
        RENDER_CAMERA_TARGET[1],
        RENDER_CAMERA_TARGET[2],
        RENDER_FOV_DEG,
        RENDER_RESOLUTION,
        RENDER_FRAME,
    )
}

fn run_blender_render(blender_path: &str, model_path: &str, output_png: &Path) {
    let script = Path::new("scripts/blender_render_usd.py");
    let model_abs = Path::new(model_path)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(model_path));

    let output = Command::new(blender_path)
        .args([
            "--background",
            "--python",
            &script.to_string_lossy(),
            "--",
            &model_abs.to_string_lossy(),
            &output_png.to_string_lossy(),
            &render_camera_config(),
        ])
        .output()
        .expect("Failed to run Blender render");

    if !output.status.success() {
        panic!(
            "Blender render failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn decode_png_rgba(path: &Path) -> (u32, u32, Vec<u8>) {
    let decoder = png::Decoder::new(std::fs::File::open(path).expect("open png"));
    let mut reader = decoder.read_info().expect("read png header");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode png frame");

    let channels = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        other => panic!("unsupported png color type: {:?}", other),
    };

    let mut rgba = vec![0u8; (info.width * info.height * 4) as usize];
    for pixel in 0..(info.width * info.height) as usize {
        for c in 0..3 {
            rgba[pixel * 4 + c] = buf[pixel * channels + c];
        }
        rgba[pixel * 4 + 3] = if channels == 4 {
            buf[pixel * 4 + 3]
        } else {
            255
        };
    }
    (info.width, info.height, rgba)
}

fn is_foreground(rgba: &[u8], pixel: usize) -> bool {
    rgba[pixel * 4 + 3] > 16
}

fn silhouette_iou(a: &[u8], b: &[u8], pixel_count: usize) -> f32 {
    let mut intersection = 0u32;
    let mut union = 0u32;
    for pixel in 0..pixel_count {
        let fa = is_foreground(a, pixel);
        let fb = is_foreground(b, pixel);
        if fa && fb {
            intersection += 1;
        }
        if fa || fb {
            union += 1;
        }
    }
    if union == 0 {
        return 1.0;
    }
    intersection as f32 / union as f32
}

#[test]
fn test_usd_render_color_matches_blender() {
    let Some(blender_path) = read_blender_path() else {
        eprintln!("Skipping: BlenderPath not configured in .claude/local/paths.md");
        return;
    };
    if !Path::new(USD_ASSET).exists() {
        eprintln!("Skipping: {} not found", USD_ASSET);
        return;
    }

    let thyllore_png = std::env::temp_dir().join(format!(
        "thyllore_usd_render_{:?}.png",
        std::thread::current().id()
    ));
    crate::headless::render_usd_to_png(
        USD_ASSET,
        &crate::headless::CameraConfig {
            position: RENDER_CAMERA_POS,
            target: RENDER_CAMERA_TARGET,
            fov_deg: RENDER_FOV_DEG,
            resolution: RENDER_RESOLUTION,
            frame: RENDER_FRAME,
        },
        &thyllore_png,
    )
    .expect("Thyllore headless render failed");

    let blender_png = std::env::temp_dir().join(format!(
        "blender_usd_render_{:?}.png",
        std::thread::current().id()
    ));
    run_blender_render(&blender_path, USD_ASSET, &blender_png);

    let (bw, bh, blender) = decode_png_rgba(&blender_png);
    let (tw, th, thyllore) = decode_png_rgba(&thyllore_png);
    std::fs::remove_file(&blender_png).ok();
    std::fs::remove_file(&thyllore_png).ok();

    assert_eq!((bw, bh), (tw, th), "render resolutions must match");
    let pixel_count = (bw * bh) as usize;

    let iou = silhouette_iou(&blender, &thyllore, pixel_count);
    eprintln!("[USD render] silhouette IoU = {:.4}", iou);

    assert!(
        iou >= 0.8,
        "silhouette IoU {:.4} below 0.8 — Thyllore render diverges from Blender",
        iou
    );
}

#[derive(Deserialize)]
struct BlenderBones {
    fps: f32,
    frame_start: i32,
    bone_names: Vec<String>,
    frames: Vec<BlenderBoneFrame>,
}

#[derive(Deserialize)]
struct BlenderBoneFrame {
    frame: i32,
    heads: Vec<[f32; 3]>,
}

fn run_blender_bones(blender_path: &str, model_path: &str) -> BlenderBones {
    let script = Path::new("scripts/blender_usd_bones.py");
    let model_abs = Path::new(model_path)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(model_path));
    let output_file = std::env::temp_dir().join(format!(
        "blender_usd_bones_{:?}.json",
        std::thread::current().id()
    ));

    let output = Command::new(blender_path)
        .args([
            "--background",
            "--python",
            &script.to_string_lossy(),
            "--",
            &model_abs.to_string_lossy(),
            &output_file.to_string_lossy(),
        ])
        .output()
        .expect("Failed to run Blender");

    if !output.status.success() {
        panic!(
            "Blender bone script failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let json = std::fs::read_to_string(&output_file).expect("Failed to read Blender bones JSON");
    std::fs::remove_file(&output_file).ok();
    serde_json::from_str(&json).expect("Failed to parse Blender bones JSON")
}

/// Engine bone world positions via the exact bone-gizmo display path, keyed by
/// bone name so they can be compared 1:1 with Blender's pose bones.
fn compute_thyllore_bone_positions(
    load_result: &ModelLoadResult,
    time: f32,
) -> Vec<(String, [f32; 3])> {
    let skeleton = load_result.skeletons.first().expect("No skeleton");

    let mut pose = create_pose_from_rest(skeleton);
    if let Some(clip) = load_result.clips.first() {
        sample_clip_to_pose(clip, time, skeleton, &mut pose, false);
    }
    let globals = compute_pose_global_transforms(skeleton, &pose);
    let offsets = compute_bone_local_offsets(skeleton, &globals);
    let display = compute_display_transforms_with_skeleton(&globals, &offsets, 1.0, Some(skeleton));

    skeleton
        .bones
        .iter()
        .map(|b| (b.name.clone(), display[b.id as usize]))
        .collect()
}

#[test]
fn test_usd_bone_positions_match_blender() {
    let Some(blender_path) = read_blender_path() else {
        eprintln!("Skipping: BlenderPath not configured in .claude/local/paths.md");
        return;
    };
    if !Path::new(USD_ASSET).exists() {
        eprintln!("Skipping: {} not found", USD_ASSET);
        return;
    }

    let usd_result = crate::loader::usd::load_usd_file(USD_ASSET).expect("Failed to load USD");
    let load_result = ModelLoadResult::from_usd(usd_result);
    let blender = run_blender_bones(&blender_path, USD_ASSET);

    let mut log = String::new();
    log.push_str("# USD bone position comparison (engine display path vs Blender)\n");
    log.push_str(&format!(
        "armature_bones={} blender_bones={} fps={}\n",
        load_result.skeletons.first().map_or(0, |s| s.bones.len()),
        blender.bone_names.len(),
        blender.fps,
    ));

    let mut worst_overall = 0.0f32;
    for frame_data in &blender.frames {
        let time = (frame_data.frame - blender.frame_start) as f32 / blender.fps;
        let engine = compute_thyllore_bone_positions(&load_result, time);
        let engine_by_name: std::collections::HashMap<&str, [f32; 3]> =
            engine.iter().map(|(n, p)| (n.as_str(), *p)).collect();

        let mut worst = (0.0f32, String::new(), [0.0; 3], [0.0; 3]);
        let mut matched = 0;
        for (bi, name) in blender.bone_names.iter().enumerate() {
            let Some(&e) = engine_by_name.get(name.as_str()) else {
                continue;
            };
            let b = frame_data.heads[bi];
            matched += 1;
            let d = ((e[0] - b[0]).powi(2) + (e[1] - b[1]).powi(2) + (e[2] - b[2]).powi(2)).sqrt();
            if d > worst.0 {
                worst = (d, name.clone(), e, b);
            }
        }
        worst_overall = worst_overall.max(worst.0);

        log.push_str(&format!(
            "frame {} time={:.4} matched={} worst_dist={:.5} bone={:?}\n  engine=[{:.4},{:.4},{:.4}] blender=[{:.4},{:.4},{:.4}]\n",
            frame_data.frame, time, matched, worst.0, worst.1,
            worst.2[0], worst.2[1], worst.2[2], worst.3[0], worst.3[1], worst.3[2],
        ));
        // sample of first 8 bones for inspection
        for (bi, name) in blender.bone_names.iter().enumerate().take(8) {
            if let Some(&e) = engine_by_name.get(name.as_str()) {
                let b = frame_data.heads[bi];
                log.push_str(&format!(
                    "    {:<24} engine=[{:.4},{:.4},{:.4}] blender=[{:.4},{:.4},{:.4}]\n",
                    name, e[0], e[1], e[2], b[0], b[1], b[2]
                ));
            }
        }
    }

    let log_path = Path::new("log/log_bone_compare.txt");
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Ok(mut f) = std::fs::File::create(log_path) {
        let _ = f.write_all(log.as_bytes());
    }
    eprintln!("{}", log);
    eprintln!(
        "[USD bones] worst divergence across frames = {:.5}",
        worst_overall
    );

    assert!(
        worst_overall <= 0.01,
        "bone positions diverge from Blender by {:.5} (> 0.01) — see log/log_bone_compare.txt",
        worst_overall
    );
}
