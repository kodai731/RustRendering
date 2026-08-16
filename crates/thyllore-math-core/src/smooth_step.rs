/// Smooth step function S(x) = x*x*(3-2x), clamped to [0, 1].
pub fn smooth_step(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}
