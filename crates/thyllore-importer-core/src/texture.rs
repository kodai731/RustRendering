use anyhow::{anyhow, Result};
use std::fs::File;

/// Load a PNG and always return tightly-packed 8-bit RGBA pixels. Callers upload
/// into an `R8G8B8A8` texture, so RGB / grayscale / palette sources are expanded
/// to four channels here rather than at every call site.
pub fn load_png_image(path: &str) -> Result<(Vec<u8>, u32, u32)> {
    let image_file = File::open(path)?;
    let mut decoder = png::Decoder::new(image_file);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);

    let mut reader = decoder.read_info()?;
    let mut raw = vec![0; reader.info().raw_bytes()];
    let frame = reader.next_frame(&mut raw)?;
    let raw = &raw[..frame.buffer_size()];

    let (width, height) = reader.info().size();
    let (color_type, _) = reader.output_color_type();
    let rgba = expand_to_rgba8(raw, color_type)?;

    Ok((rgba, width, height))
}

fn expand_to_rgba8(raw: &[u8], color_type: png::ColorType) -> Result<Vec<u8>> {
    let pixels = match color_type {
        png::ColorType::Rgba => raw.to_vec(),
        png::ColorType::Rgb => raw
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => raw
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        png::ColorType::Grayscale => raw.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::Indexed => {
            return Err(anyhow!(
                "indexed PNG not expanded; EXPAND transformation failed"
            ))
        }
    };
    Ok(pixels)
}
