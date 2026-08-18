use crate::flame::*;

/// Erosion edge window and the tip silhouette of the medium.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameEdge {
    pub low: f32,
    pub high: f32,
    pub white_boost: f32,
    pub radius_tip_ratio: f32,
    pub outer_sharpen: f32,
}

impl Default for FlameEdge {
    fn default() -> Self {
        Self {
            low: 0.27,
            high: 0.33,
            white_boost: 4.0,
            radius_tip_ratio: 0.10,
            outer_sharpen: 0.0,
        }
    }
}

/// Contrast-scaled base edge window: center is fixed, half-width divides by
/// noise contrast (higher contrast = narrower window = harder carving).
/// Exactly 1.0 returns the authored low/high bytes untouched.
pub fn contrast_scaled_edges(edge: &FlameEdge, noise: &FlameNoise) -> (f32, f32) {
    let contrast = noise.contrast.clamp(0.25, 4.0);
    if contrast == 1.0 {
        return (edge.low, edge.high);
    }
    let center = 0.5 * (edge.low + edge.high);
    let half_width = 0.5 * (edge.high - edge.low) / contrast;
    (center - half_width, center + half_width)
}

/// Effective edge window (low, high): the center is fixed and the half-width
/// scales with |noise amplitude| / NOISE_AMPLITUDE_REF raised to EDGE_WIDTH_GAMMA,
/// clamped to [0.25, 4.0] times the contrast-scaled half-width.
pub fn effective_edge_window(edge: &FlameEdge, noise: &FlameNoise) -> (f32, f32) {
    let (edge_lo, edge_hi) = contrast_scaled_edges(edge, noise);
    let center = 0.5 * (edge_lo + edge_hi);
    let half_width0 = 0.5 * (edge_hi - edge_lo);
    let half_width =
        half_width0 * (noise.amplitude.abs() / NOISE_AMPLITUDE_REF).powf(EDGE_WIDTH_GAMMA);
    let half_width = half_width.clamp(0.25 * half_width0, 4.0 * half_width0);
    (center - half_width, center + half_width)
}

pub fn build_edge_style(edge: &FlameEdge, noise: &FlameNoise) -> FlameEdgeStyle {
    let (edge_low, edge_high) = contrast_scaled_edges(edge, noise);
    FlameEdgeStyle {
        radius_tip_ratio: edge.radius_tip_ratio,
        edge_low,
        edge_high,
        white_boost: edge.white_boost,
    }
}
