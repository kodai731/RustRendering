pub const FLAME_TRAIL_CAPACITY: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct FlameTrailSample {
    pub position: [f32; 3],
    pub age_seconds: f32,
}

#[derive(Clone, Debug)]
pub struct FlameTrailState {
    pub samples: Vec<FlameTrailSample>,
    pub fade_seconds: f32,
    pub enabled: bool,
}

impl Default for FlameTrailState {
    fn default() -> Self {
        Self {
            samples: Vec::new(),
            fade_seconds: 0.8,
            enabled: false,
        }
    }
}

pub fn advance_flame_trail(
    state: &mut FlameTrailState,
    emitter_position: [f32; 3],
    delta_seconds: f32,
) {
    if delta_seconds < 0.0 {
        state.samples.clear();
        state.samples.push(FlameTrailSample {
            position: emitter_position,
            age_seconds: 0.0,
        });
        return;
    }

    for sample in &mut state.samples {
        sample.age_seconds += delta_seconds;
    }

    state.samples.retain(|s| s.age_seconds < state.fade_seconds);

    let should_insert = if state.samples.is_empty() {
        true
    } else {
        state.samples[0].age_seconds >= state.fade_seconds / FLAME_TRAIL_CAPACITY as f32
    };

    if should_insert {
        state.samples.insert(
            0,
            FlameTrailSample {
                position: emitter_position,
                age_seconds: 0.0,
            },
        );

        while state.samples.len() > FLAME_TRAIL_CAPACITY {
            state.samples.pop();
        }
    }
}

pub fn reset_flame_trail(state: &mut FlameTrailState) {
    state.samples.clear();
}

pub fn flame_trail_fade_weight(sample: &FlameTrailSample, fade_seconds: f32) -> f32 {
    if fade_seconds <= 0.0 {
        return 0.0;
    }
    (1.0 - (sample.age_seconds / fade_seconds)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples_removed_when_exceeding_fade() {
        let mut state = FlameTrailState {
            samples: vec![
                FlameTrailSample {
                    position: [0.0, 0.0, 0.0],
                    age_seconds: 0.5,
                },
                FlameTrailSample {
                    position: [1.0, 1.0, 1.0],
                    age_seconds: 0.9,
                },
            ],
            fade_seconds: 0.8,
            enabled: false,
        };

        advance_flame_trail(&mut state, [2.0, 2.0, 2.0], 0.1);

        assert_eq!(state.samples.len(), 2);
        assert_eq!(state.samples[0].position, [2.0, 2.0, 2.0]);
        assert_eq!(state.samples[0].age_seconds, 0.0);
        assert_eq!(state.samples[1].position, [0.0, 0.0, 0.0]);
        assert_eq!(state.samples[1].age_seconds, 0.6);
    }

  #[test]
    fn test_oldest_dropped_when_capacity_exceeded() {
        let mut state = FlameTrailState::default();
        // fade large enough that no sample is removed during the test
        state.fade_seconds = 1000.0;

        // delta=63.0 >= threshold(1000/16=62.5) so every frame inserts
        for i in 0..FLAME_TRAIL_CAPACITY + 2 {
            advance_flame_trail(
                &mut state,
                [i as f32, 0.0, 0.0],
                63.0,
            );
        }

        assert_eq!(state.samples.len(), FLAME_TRAIL_CAPACITY);
        assert_eq!(
            state.samples[0].position,
            [(FLAME_TRAIL_CAPACITY + 1) as f32, 0.0, 0.0]
        );
        assert_eq!(state.samples[FLAME_TRAIL_CAPACITY - 1].position, [2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_negative_delta_clears_and_keeps_current() {
        let mut state = FlameTrailState {
            samples: vec![
                FlameTrailSample {
                    position: [0.0, 0.0, 0.0],
                    age_seconds: 0.1,
                },
                FlameTrailSample {
                    position: [1.0, 1.0, 1.0],
                    age_seconds: 0.2,
                },
            ],
            fade_seconds: 0.8,
            enabled: false,
        };

        advance_flame_trail(&mut state, [5.0, 5.0, 5.0], -1.0);

        assert_eq!(state.samples.len(), 1);
        assert_eq!(state.samples[0].position, [5.0, 5.0, 5.0]);
        assert_eq!(state.samples[0].age_seconds, 0.0);
    }

    #[test]
    fn test_fade_weight_boundaries() {
        let sample_age_0 = FlameTrailSample {
            position: [0.0, 0.0, 0.0],
            age_seconds: 0.0,
        };
        let sample_age_equal = FlameTrailSample {
            position: [0.0, 0.0, 0.0],
            age_seconds: 0.8,
        };

        assert_eq!(flame_trail_fade_weight(&sample_age_0, 0.8), 1.0);
        assert_eq!(flame_trail_fade_weight(&sample_age_equal, 0.8), 0.0);
    }

    #[test]
    fn test_deterministic_results() {
        let inputs: [[f32; 3]; 5] = [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ];

        let mut state1 = FlameTrailState::default();
        let mut state2 = FlameTrailState::default();

        for pos in &inputs {
            advance_flame_trail(&mut state1, *pos, 0.1);
            advance_flame_trail(&mut state2, *pos, 0.1);
        }

        assert_eq!(state1.samples.len(), state2.samples.len());
        for (s1, s2) in state1.samples.iter().zip(state2.samples.iter()) {
            assert_eq!(s1.position, s2.position);
            assert_eq!(s1.age_seconds, s2.age_seconds);
        }
    }

  #[test]
    fn test_time_decimation_covers_full_fade_window() {
        let mut state = FlameTrailState::default();
        state.fade_seconds = 2.4;

        let delta: f32 = 1.0 / 60.0;

        for _ in 0..240 {
            advance_flame_trail(&mut state, [0.0, 0.0, 0.0], delta);
        }

        assert_eq!(state.samples.len(), FLAME_TRAIL_CAPACITY);
        let oldest_age = state.samples.last().map(|s| s.age_seconds).unwrap_or(0.0);
        assert!(
            oldest_age >= 1.75,
            "oldest sample age {:.3} < 1.75, trail does not cover the fade window",
            oldest_age
        );
    }
}
