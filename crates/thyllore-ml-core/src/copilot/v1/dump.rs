use std::fs::{create_dir_all, File};
use std::path::{Path, PathBuf};

use anyhow::Result;
use ndarray::Array1;
use ndarray_npy::NpzWriter;

pub struct V1CurveCopilotInferenceDump<'a> {
    pub context: &'a [f32],
    pub future: &'a [f32],
    pub reveal_mask: &'a [bool],
    pub mean_curve: &'a [f32],
    pub fps: f32,
    pub anchor_time: f32,
}

pub fn dump_v1_curve_copilot_inference(
    dump: &V1CurveCopilotInferenceDump<'_>,
    output_dir: &Path,
) -> Result<PathBuf> {
    create_dir_all(output_dir)?;
    let timestamp = chrono::Local::now().format("%Y%m%d-%H%M%S-%3f").to_string();
    let path = output_dir.join(format!("{}-v1-curve-copilot-inference.npz", timestamp));

    let mut npz = NpzWriter::new(File::create(&path)?);

    npz.add_array("context", &Array1::from_vec(dump.context.to_vec()))?;
    npz.add_array("future", &Array1::from_vec(dump.future.to_vec()))?;
    let reveal_u8: Vec<u8> = dump.reveal_mask.iter().map(|b| *b as u8).collect();
    npz.add_array("reveal_mask", &Array1::from_vec(reveal_u8))?;
    npz.add_array("mean_curve", &Array1::from_vec(dump.mean_curve.to_vec()))?;
    npz.add_array("fps", &Array1::from_vec(vec![dump.fps]))?;
    npz.add_array("anchor_time", &Array1::from_vec(vec![dump.anchor_time]))?;

    npz.finish()?;
    Ok(path)
}
