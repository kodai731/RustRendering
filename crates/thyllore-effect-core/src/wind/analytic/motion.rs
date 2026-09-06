pub fn h_top(t: f32, h0: f32, rise_duration: f32) -> f32 {
    let x = (t / rise_duration.max(1e-3)).clamp(0.0, 1.0);
    let smoothstep = 3.0 * x * x - 2.0 * x * x * x;
    h0 + (1.0 - h0) * smoothstep
}

pub fn spread_offset(t: f32, spread_start: f32, spread_rate: f32) -> f32 {
    2.0 * spread_rate * (t - spread_start).max(0.0)
}

pub fn wall_amp(t: f32, wall_strength: f32, dissipate_start: f32, dissipate_time: f32) -> f32 {
    if dissipate_time <= 0.0 {
        return wall_strength;
    }
    let elapsed = (t - dissipate_start).max(0.0);
    wall_strength * (-elapsed / dissipate_time).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h_top_rises_from_initial_height_to_one() {
        assert!((h_top(0.0, 0.2, 2.0) - 0.2).abs() < 1e-6);
        assert!((h_top(1.0, 0.2, 2.0) - 0.6).abs() < 1e-6);
        assert!((h_top(2.0, 0.2, 2.0) - 1.0).abs() < 1e-6);
        assert!((h_top(5.0, 0.2, 2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_spread_offset_starts_at_spread_start() {
        assert_eq!(spread_offset(0.4, 0.5, 0.1), 0.0);
        assert!((spread_offset(1.5, 0.5, 0.1) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_wall_amp_decays_only_after_dissipate_start() {
        assert_eq!(wall_amp(3.0, 2.0, 1.0, 0.0), 2.0);
        assert_eq!(wall_amp(0.5, 2.0, 1.0, 1.0), 2.0);
        assert!((wall_amp(2.0, 2.0, 1.0, 1.0) - 2.0 * (-1.0f32).exp()).abs() < 1e-6);
    }
}
