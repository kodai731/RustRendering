use std::borrow::Cow;
use std::fs;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{anyhow, Result};
use gltf::binary::Glb;
use gltf::json;

pub(crate) fn write_glb(root: &json::Root, bin: Vec<u8>, output_path: &Path) -> Result<()> {
    let json_bytes = root
        .to_vec()
        .map_err(|e| anyhow!("Failed to serialize glTF JSON: {:?}", e))?;

    let output_glb = Glb {
        header: gltf::binary::Header {
            magic: *b"glTF",
            version: 2,
            length: 0,
        },
        json: Cow::Owned(json_bytes),
        bin: if bin.is_empty() {
            None
        } else {
            Some(Cow::Owned(bin))
        },
    };

    let file = fs::File::create(output_path)?;
    let writer = BufWriter::new(file);
    output_glb
        .to_writer(writer)
        .map_err(|e| anyhow!("Failed to write GLB: {:?}", e))?;

    Ok(())
}
