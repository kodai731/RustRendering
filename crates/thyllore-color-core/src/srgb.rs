/// Standard sRGB component to linear light conversion (piecewise).
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Relative luminance from linear RGB (Rec. 709 weights).
pub fn luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// True if any sRGB8 channel is >= 250 (near-clipped to white).
pub fn is_saturated(srgb8: [u8; 3]) -> bool {
    srgb8[0] >= 250 || srgb8[1] >= 250 || srgb8[2] >= 250
}
