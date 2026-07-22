use thyllore_anim_core::{Interpolation, Keyframe};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlameParam {
    Height,
    Radius,
    Intensity,
    SigmaT,
    TemperatureBaseK,
    TemperatureTipK,
    WarpAmp,
    WarpFreq,
    RiseSpeed,
    NoiseAmplitude,
    WhiteBoost,
    BendAmount,
    WindX,
    WindZ,
    EdgeLow,
    EdgeHigh,
}

#[derive(Clone, Debug)]
pub struct FlameChannel {
    pub param: FlameParam,
    pub keys: Vec<Keyframe<f32>>,
}

#[derive(Clone, Debug, Default)]
pub struct FlameTrack {
    pub channels: Vec<FlameChannel>,
}

/// Sample a channel at `time`. Returns `None` if keys are empty.
/// Clamps `time` to the range of first and last keyframes.
/// Uses binary search to find the interval, then interpolates linearly
/// (treats CubicSpline as Linear).
pub fn sample_channel(keys: &[Keyframe<f32>], time: f32) -> Option<f32> {
    if keys.is_empty() {
        return None;
    }

    let clamped_time = time.max(keys.first().unwrap().time).min(keys.last().unwrap().time);

    // Binary search for the interval [keys[i], keys[i+1]] containing clamped_time
    let mut lo: usize = 0;
    let mut hi = keys.len() - 1;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if keys[mid].time < clamped_time {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // lo is now the index of the first keyframe with time >= clamped_time
    if lo == 0 || (keys[lo].time - clamped_time).abs() < 1e-9 {
        return Some(keys[lo].value);
    }

    let prev = &keys[lo - 1];
    let curr = &keys[lo];

    // Determine interpolation mode from the previous keyframe
    let interp = &prev.interpolation;

    match interp {
        Interpolation::Step => Some(prev.value),
        Interpolation::Linear | Interpolation::CubicSpline => {
            let t = (clamped_time - prev.time) / (curr.time - prev.time);
            let clamped_t = t.max(0.0).min(1.0);
            Some(prev.value + (curr.value - prev.value) * clamped_t)
        }
    }
}

/// Apply a `FlameTrack` at `time` to the mutable `FlameEffect`.
/// Iterates over channels and updates corresponding fields in `FlameEffect`.
/// For `WindX` and `WindZ`, updates the components of `wind_direction`.
pub fn apply_flame_track(
    track: &FlameTrack,
    time: f32,
    effect: &mut crate::ecs::resource::FlameEffect,
) {
    for channel in &track.channels {
        if let Some(value) = sample_channel(&channel.keys, time) {
            match channel.param {
                FlameParam::Height => effect.height = value,
                FlameParam::Radius => effect.radius = value,
                FlameParam::Intensity => effect.intensity = value,
                FlameParam::SigmaT => effect.sigma_t = value,
                FlameParam::TemperatureBaseK => effect.temperature_base_k = value,
                FlameParam::TemperatureTipK => effect.temperature_tip_k = value,
                FlameParam::WarpAmp => effect.warp_amp = value,
                FlameParam::WarpFreq => effect.warp_freq = value,
                FlameParam::RiseSpeed => effect.rise_speed = value,
                FlameParam::NoiseAmplitude => effect.noise_amplitude = value,
                FlameParam::WhiteBoost => effect.white_boost = value,
                FlameParam::BendAmount => effect.bend_amount = value,
                FlameParam::WindX => {
                    effect.wind_direction.x = value;
                }
                FlameParam::WindZ => {
                    effect.wind_direction.y = value;
                }
                FlameParam::EdgeLow => effect.edge_low = value,
                FlameParam::EdgeHigh => effect.edge_high = value,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_key(time: f32, value: f32) -> Keyframe<f32> {
        Keyframe::with_interpolation(time, value, Interpolation::Linear)
    }

    fn step_key(time: f32, value: f32) -> Keyframe<f32> {
        Keyframe::with_interpolation(time, value, Interpolation::Step)
    }

    #[test]
    fn test_sample_empty() {
        let keys: Vec<Keyframe<f32>> = vec![];
        assert_eq!(sample_channel(&keys, 0.0), None);
    }

    #[test]
    fn test_sample_single_key() {
        let keys = vec![linear_key(1.0, 5.0)];
        assert_eq!(sample_channel(&keys, 1.0), Some(5.0));
        // Clamped to the single key
        assert_eq!(sample_channel(&keys, 0.0), Some(5.0));
        assert_eq!(sample_channel(&keys, 2.0), Some(5.0));
    }

    #[test]
    fn test_sample_two_keys_linear_midpoint() {
        let keys = vec![linear_key(0.0, 1.0), linear_key(2.0, 3.0)];
        assert_eq!(sample_channel(&keys, 1.0), Some(2.0));
    }

    #[test]
    fn test_sample_two_keys_linear_step_hold() {
        let keys = vec![step_key(0.0, 1.0), step_key(2.0, 3.0)];
        // At time 0, should be first key's value
        assert_eq!(sample_channel(&keys, 0.0), Some(1.0));
        // Between keys, step holds the previous value
        assert_eq!(sample_channel(&keys, 1.0), Some(1.0));
        // At last key, should be last key's value
        assert_eq!(sample_channel(&keys, 2.0), Some(3.0));
    }

    #[test]
    fn test_sample_clamp_before_first() {
        let keys = vec![linear_key(1.0, 1.0), linear_key(2.0, 2.0)];
        assert_eq!(sample_channel(&keys, 0.0), Some(1.0));
    }

    #[test]
    fn test_sample_clamp_after_last() {
        let keys = vec![linear_key(1.0, 1.0), linear_key(2.0, 2.0)];
        assert_eq!(sample_channel(&keys, 3.0), Some(2.0));
    }

    #[test]
    fn test_apply_flame_track_integration() {
        let mut effect = crate::ecs::resource::FlameEffect::default();
        let original_height = effect.height;

        let track = FlameTrack {
            channels: vec![FlameChannel {
                param: FlameParam::Height,
                keys: vec![linear_key(0.0, 1.0), linear_key(2.0, 2.0)],
            }],
        };

        apply_flame_track(&track, 1.0, &mut effect);
        assert!((effect.height - 1.5).abs() < 1e-6, "expected height 1.5, got {}", effect.height);

        // Other fields should be unchanged
        assert_eq!(effect.radius, crate::ecs::resource::FlameEffect::default().radius);
    }
}
