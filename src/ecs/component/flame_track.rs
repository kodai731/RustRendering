use thyllore_anim_core::editable::{EditableKeyframe, InterpolationType, KeyframeId};
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
    pub keys: Vec<EditableKeyframe>,
    pub next_keyframe_id: KeyframeId,
}

#[derive(Clone, Debug, Default)]
pub struct FlameTrack {
    pub channels: Vec<FlameChannel>,
}

/// Sample a channel at `time`. Returns `None` if keys are empty.
/// Delegates to the editable curve_ops keyframes_sample (clamped to first/last key).
pub fn sample_channel(keys: &[EditableKeyframe], time: f32) -> Option<f32> {
    thyllore_anim_core::editable::keyframes_sample(keys, time)
}

/// Insert a key into the channel. Allocates an id from `next_keyframe_id`, inserts
/// in time-sorted order, and returns the new keyframe's id.
pub fn channel_insert_key(
    channel: &mut FlameChannel,
    time: f32,
    value: f32,
    interpolation: InterpolationType,
) -> KeyframeId {
    let id = channel.next_keyframe_id;
    channel.next_keyframe_id += 1;
    let mut kf = EditableKeyframe::new(id, time, value);
    kf.interpolation = interpolation;
    channel.keys.push(kf);
    channel
        .keys
        .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    id
}

/// Convert an old `Keyframe<f32>` (legacy curve format) to the editable keyframe fields.
/// Interpolation::Linear and CubicSpline -> InterpolationType::Linear (CubicSpline treated
/// as Linear is the current behavior), Step -> Stepped.
pub fn convert_legacy_key(key: &Keyframe<f32>) -> (f32, f32, InterpolationType) {
    let interpolation = match key.interpolation {
        Interpolation::Linear | Interpolation::CubicSpline => InterpolationType::Linear,
        Interpolation::Step => InterpolationType::Stepped,
    };
    (key.time, key.value, interpolation)
}

/// Apply a `FlameTrack` at `time` to the mutable `FlameEffect`.
/// Iterates over channels and updates corresponding fields in `FlameEffect`.
/// For `WindX` and `WindZ`, updates the components of `wind_direction`.
pub fn apply_flame_track(
    track: &FlameTrack,
    time: f32,
    effect: &mut crate::ecs::component::FlameEffect,
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

    fn cubic_key(time: f32, value: f32) -> Keyframe<f32> {
        Keyframe::with_interpolation(time, value, Interpolation::CubicSpline)
    }

    #[test]
    fn test_sample_empty() {
        let keys: Vec<EditableKeyframe> = vec![];
        assert_eq!(sample_channel(&keys, 0.0), None);
    }

    #[test]
    fn test_sample_single_key() {
        let mut channel = FlameChannel {
            param: FlameParam::Height,
            keys: vec![],
            next_keyframe_id: 1,
        };
        channel_insert_key(&mut channel, 1.0, 5.0, InterpolationType::Linear);
        assert_eq!(sample_channel(&channel.keys, 1.0), Some(5.0));
        // Clamped to the single key
        assert_eq!(sample_channel(&channel.keys, 0.0), Some(5.0));
        assert_eq!(sample_channel(&channel.keys, 2.0), Some(5.0));
    }

    #[test]
    fn test_sample_two_keys_linear_midpoint() {
        let mut channel = FlameChannel {
            param: FlameParam::Height,
            keys: vec![],
            next_keyframe_id: 1,
        };
        channel_insert_key(&mut channel, 0.0, 1.0, InterpolationType::Linear);
        channel_insert_key(&mut channel, 2.0, 3.0, InterpolationType::Linear);
        assert_eq!(sample_channel(&channel.keys, 1.0), Some(2.0));
    }

    #[test]
    fn test_sample_two_keys_linear_step_hold() {
        let mut channel = FlameChannel {
            param: FlameParam::Height,
            keys: vec![],
            next_keyframe_id: 1,
        };
        channel_insert_key(&mut channel, 0.0, 1.0, InterpolationType::Stepped);
        channel_insert_key(&mut channel, 2.0, 3.0, InterpolationType::Linear);
        // At time 0, should be first key's value
        assert_eq!(sample_channel(&channel.keys, 0.0), Some(1.0));
        // Between keys, step holds the previous value
        assert_eq!(sample_channel(&channel.keys, 1.0), Some(1.0));
        // At last key, should be last key's value
        assert_eq!(sample_channel(&channel.keys, 2.0), Some(3.0));
    }

    #[test]
    fn test_sample_clamp_before_first() {
        let mut channel = FlameChannel {
            param: FlameParam::Height,
            keys: vec![],
            next_keyframe_id: 1,
        };
        channel_insert_key(&mut channel, 1.0, 1.0, InterpolationType::Linear);
        channel_insert_key(&mut channel, 2.0, 2.0, InterpolationType::Linear);
        assert_eq!(sample_channel(&channel.keys, 0.0), Some(1.0));
    }

    #[test]
    fn test_sample_clamp_after_last() {
        let mut channel = FlameChannel {
            param: FlameParam::Height,
            keys: vec![],
            next_keyframe_id: 1,
        };
        channel_insert_key(&mut channel, 1.0, 1.0, InterpolationType::Linear);
        channel_insert_key(&mut channel, 2.0, 2.0, InterpolationType::Linear);
        assert_eq!(sample_channel(&channel.keys, 3.0), Some(2.0));
    }

    #[test]
    fn test_apply_flame_track_integration() {
        let mut effect = crate::ecs::component::FlameEffect::default();

        let mut channel = FlameChannel {
            param: FlameParam::Height,
            keys: vec![],
            next_keyframe_id: 1,
        };
        channel_insert_key(&mut channel, 0.0, 1.0, InterpolationType::Linear);
        channel_insert_key(&mut channel, 2.0, 2.0, InterpolationType::Linear);

        let track = FlameTrack {
            channels: vec![channel],
        };

        apply_flame_track(&track, 1.0, &mut effect);
        assert!(
            (effect.height - 1.5).abs() < 1e-6,
            "expected height 1.5, got {}",
            effect.height
        );

        // Other fields should be unchanged
        assert_eq!(
            effect.radius,
            crate::ecs::component::FlameEffect::default().radius
        );
    }

    /// Behavior-preserving test: Linear legacy keys converted and evaluated match old semantics.
    #[test]
    fn test_legacy_linear_behavior_preserved() {
        let legacy_keys = vec![linear_key(0.0, 1.0), linear_key(2.0, 3.0)];
        let duration = 2.0;
        let sample_count = 100;

        for i in 0..sample_count {
            let t = duration * (i as f32) / (sample_count - 1) as f32;

            // Old semantics: linear interpolation, clamped to first/last
            let expected = {
                let clamped_t = t.max(0.0).min(2.0);
                if clamped_t <= 0.0 {
                    1.0
                } else if clamped_t >= 2.0 {
                    3.0
                } else {
                    let frac = (clamped_t - 0.0) / (2.0 - 0.0);
                    1.0 + (3.0 - 1.0) * frac
                }
            };

            // New evaluation via convert_legacy_key + keyframes_sample
            let mut channel = FlameChannel {
                param: FlameParam::Height,
                keys: vec![],
                next_keyframe_id: 1,
            };
            for k in &legacy_keys {
                let (time, value, interp) = convert_legacy_key(k);
                channel_insert_key(&mut channel, time, value, interp);
            }
            let actual = sample_channel(&channel.keys, t).unwrap();

            assert!(
                (actual - expected).abs() < 1e-5,
                "Linear: at time {} expected {} got {}",
                t,
                expected,
                actual
            );
        }
    }

    /// Behavior-preserving test: Step legacy keys converted and evaluated match old semantics.
    #[test]
    fn test_legacy_step_behavior_preserved() {
        let legacy_keys = vec![step_key(0.0, 1.0), step_key(2.0, 3.0)];
        let duration = 2.0;
        let sample_count = 100;

        for i in 0..sample_count {
            let t = duration * (i as f32) / (sample_count - 1) as f32;

            // Old semantics: step holds previous value, clamped to first/last
            let expected = {
                let clamped_t = t.max(0.0).min(2.0);
                if clamped_t < 2.0 {
                    1.0
                } else {
                    3.0
                }
            };

            // New evaluation via convert_legacy_key + keyframes_sample
            let mut channel = FlameChannel {
                param: FlameParam::Height,
                keys: vec![],
                next_keyframe_id: 1,
            };
            for k in &legacy_keys {
                let (time, value, interp) = convert_legacy_key(k);
                channel_insert_key(&mut channel, time, value, interp);
            }
            let actual = sample_channel(&channel.keys, t).unwrap();

            assert!(
                (actual - expected).abs() < 1e-5,
                "Step: at time {} expected {} got {}",
                t,
                expected,
                actual
            );
        }
    }

    /// Behavior-preserving test: CubicSpline legacy keys converted and evaluated match old semantics.
    /// (CubicSpline is treated as Linear in both old and new code.)
    #[test]
    fn test_legacy_cubic_spline_behavior_preserved() {
        let legacy_keys = vec![cubic_key(0.0, 1.0), cubic_key(2.0, 3.0)];
        let duration = 2.0;
        let sample_count = 100;

        for i in 0..sample_count {
            let t = duration * (i as f32) / (sample_count - 1) as f32;

            // Old semantics: CubicSpline treated as Linear interpolation, clamped to first/last
            let expected = {
                let clamped_t = t.max(0.0).min(2.0);
                if clamped_t <= 0.0 {
                    1.0
                } else if clamped_t >= 2.0 {
                    3.0
                } else {
                    let frac = (clamped_t - 0.0) / (2.0 - 0.0);
                    1.0 + (3.0 - 1.0) * frac
                }
            };

            // New evaluation via convert_legacy_key + keyframes_sample
            let mut channel = FlameChannel {
                param: FlameParam::Height,
                keys: vec![],
                next_keyframe_id: 1,
            };
            for k in &legacy_keys {
                let (time, value, interp) = convert_legacy_key(k);
                channel_insert_key(&mut channel, time, value, interp);
            }
            let actual = sample_channel(&channel.keys, t).unwrap();

            assert!(
                (actual - expected).abs() < 1e-5,
                "CubicSpline: at time {} expected {} got {}",
                t,
                expected,
                actual
            );
        }
    }
}
