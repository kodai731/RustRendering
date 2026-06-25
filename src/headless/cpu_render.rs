use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::Result;
use cgmath::{perspective, Deg, Matrix4, Point3, Vector3, Vector4};

use crate::ecs::systems::{
    apply_skinning, compute_pose_global_transforms, create_pose_from_rest, sample_clip_to_pose,
};
use crate::loader::{usd, ModelLoadResult};

pub struct CameraConfig {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub fov_deg: f32,
    pub resolution: u32,
    pub frame: i32,
}

struct Triangle {
    positions: [Vector3<f32>; 3],
    color: [f32; 3],
}

pub fn render_usd_to_png(usd_path: &str, camera: &CameraConfig, out_path: &Path) -> Result<()> {
    let usd_result = usd::load_usd_file(usd_path)?;
    let start = usd_result.start_time_code;
    let fps = usd_result.time_codes_per_second.max(1.0);
    let time = ((camera.frame as f32 - start) / fps).max(0.0);

    let load_result = ModelLoadResult::from_usd(usd_result);
    let triangles = collect_world_triangles(&load_result, time);

    let pixels = rasterize(&triangles, camera);
    write_png(out_path, camera.resolution, &pixels)
}

fn collect_world_triangles(load_result: &ModelLoadResult, time: f32) -> Vec<Triangle> {
    let mut triangles = Vec::new();

    let skinned = skinned_world_positions(load_result, time);
    for (mesh_index, mesh) in load_result.meshes.iter().enumerate() {
        let Some(positions) = &skinned[mesh_index] else {
            continue;
        };
        append_mesh_triangles(&mut triangles, mesh, positions);
    }

    triangles
}

fn skinned_world_positions(
    load_result: &ModelLoadResult,
    time: f32,
) -> Vec<Option<Vec<Vector3<f32>>>> {
    let Some(skeleton) = load_result.skeletons.first() else {
        return vec![None; load_result.meshes.len()];
    };
    let Some(clip) = load_result.clips.first() else {
        return vec![None; load_result.meshes.len()];
    };

    let mut pose = create_pose_from_rest(skeleton);
    sample_clip_to_pose(clip, time, skeleton, &mut pose, false);
    let global_transforms = compute_pose_global_transforms(skeleton, &pose);

    load_result
        .meshes
        .iter()
        .map(|mesh| {
            let skin_data = mesh.skin_data.as_ref()?;
            let count = skin_data.base_positions.len();
            let mut positions = vec![Vector3::new(0.0, 0.0, 0.0); count];
            let mut normals = vec![Vector3::new(0.0, 0.0, 0.0); count];
            apply_skinning(
                skin_data,
                &global_transforms,
                skeleton,
                &mut positions,
                &mut normals,
            );
            Some(positions)
        })
        .collect()
}

fn append_mesh_triangles(
    triangles: &mut Vec<Triangle>,
    mesh: &crate::loader::LoadedMesh,
    positions: &[Vector3<f32>],
) {
    let indices = &mesh.vertex_data.indices;
    let vertices = &mesh.vertex_data.vertices;

    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        if a >= positions.len() || b >= positions.len() || c >= positions.len() {
            continue;
        }

        let color = vertices
            .get(a)
            .map(|v| [v.color.x, v.color.y, v.color.z])
            .unwrap_or([1.0, 1.0, 1.0]);

        triangles.push(Triangle {
            positions: [positions[a], positions[b], positions[c]],
            color,
        });
    }
}

fn rasterize(triangles: &[Triangle], camera: &CameraConfig) -> Vec<u8> {
    let size = camera.resolution as usize;
    let view = Matrix4::look_at_rh(
        Point3::new(camera.position[0], camera.position[1], camera.position[2]),
        Point3::new(camera.target[0], camera.target[1], camera.target[2]),
        Vector3::new(0.0, 0.0, 1.0),
    );
    let proj = perspective(Deg(camera.fov_deg), 1.0, 0.01, 100.0);
    let view_proj = proj * view;

    let mut color = vec![0u8; size * size * 4];
    let mut depth = vec![f32::INFINITY; size * size];

    for triangle in triangles {
        rasterize_triangle(triangle, &view_proj, size, &mut color, &mut depth);
    }

    color
}

fn project(view_proj: &Matrix4<f32>, point: Vector3<f32>, size: f32) -> Option<[f32; 3]> {
    let clip = view_proj * Vector4::new(point.x, point.y, point.z, 1.0);
    if clip.w <= 1e-5 {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    let ndc_z = clip.z / clip.w;
    Some([
        (ndc_x * 0.5 + 0.5) * size,
        (1.0 - (ndc_y * 0.5 + 0.5)) * size,
        ndc_z,
    ])
}

fn rasterize_triangle(
    triangle: &Triangle,
    view_proj: &Matrix4<f32>,
    size: usize,
    color: &mut [u8],
    depth: &mut [f32],
) {
    let fsize = size as f32;
    let screen: Vec<[f32; 3]> = triangle
        .positions
        .iter()
        .filter_map(|&p| project(view_proj, p, fsize))
        .collect();
    if screen.len() != 3 {
        return;
    }

    let min_x = screen
        .iter()
        .map(|s| s[0])
        .fold(fsize, f32::min)
        .floor()
        .max(0.0) as usize;
    let max_x = screen
        .iter()
        .map(|s| s[0])
        .fold(0.0, f32::max)
        .ceil()
        .min(fsize) as usize;
    let min_y = screen
        .iter()
        .map(|s| s[1])
        .fold(fsize, f32::min)
        .floor()
        .max(0.0) as usize;
    let max_y = screen
        .iter()
        .map(|s| s[1])
        .fold(0.0, f32::max)
        .ceil()
        .min(fsize) as usize;

    let area = edge(&screen[0], &screen[1], &screen[2]);
    if area.abs() < 1e-7 {
        return;
    }

    let rgb = [
        (triangle.color[0].clamp(0.0, 1.0) * 255.0) as u8,
        (triangle.color[1].clamp(0.0, 1.0) * 255.0) as u8,
        (triangle.color[2].clamp(0.0, 1.0) * 255.0) as u8,
    ];

    for y in min_y..max_y {
        for x in min_x..max_x {
            let p = [x as f32 + 0.5, y as f32 + 0.5, 0.0];
            let w0 = edge(&screen[1], &screen[2], &p) / area;
            let w1 = edge(&screen[2], &screen[0], &p) / area;
            let w2 = edge(&screen[0], &screen[1], &p) / area;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }

            let z = w0 * screen[0][2] + w1 * screen[1][2] + w2 * screen[2][2];
            let index = y * size + x;
            if z >= depth[index] {
                continue;
            }
            depth[index] = z;
            color[index * 4] = rgb[0];
            color[index * 4 + 1] = rgb[1];
            color[index * 4 + 2] = rgb[2];
            color[index * 4 + 3] = 255;
        }
    }
}

fn edge(a: &[f32; 3], b: &[f32; 3], c: &[f32; 3]) -> f32 {
    (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
}

fn write_png(out_path: &Path, resolution: u32, pixels: &[u8]) -> Result<()> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(out_path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, resolution, resolution);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(pixels)?;
    Ok(())
}
