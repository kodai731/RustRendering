use anyhow::Result;
use std::fs::File;

pub fn load_png_image(path: &str) -> Result<(Vec<u8>, u32, u32)> {
    let image_file = File::open(path)?;
    let decoder = png::Decoder::new(image_file);
    let mut reader = decoder.read_info()?;
    let mut pixels = vec![0; reader.info().raw_bytes()];
    reader.next_frame(&mut pixels)?;
    let (width, height) = reader.info().size();

    Ok((pixels, width, height))
}
