pub fn generate_query_times(
    times: &[f32],
    current_time: f32,
    clip_duration: f32,
    max_steps: usize,
) -> Vec<f32> {
    let duration = clip_duration.max(0.001);

    let future: Vec<f32> = times
        .iter()
        .filter(|&&t| t > current_time + 1e-6)
        .take(max_steps)
        .map(|&t| t / duration)
        .collect();

    if !future.is_empty() {
        return future;
    }

    let remaining = duration - current_time;
    if remaining <= 0.0 {
        return vec![current_time / duration];
    }

    let mut result = Vec::with_capacity(max_steps);
    for i in 0..max_steps {
        let t = current_time + remaining * (i as f32 + 1.0) / max_steps as f32;
        result.push(t / duration);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn future_keyframes_take_priority() {
        let times = vec![0.0_f32, 1.0, 2.0, 3.0];
        let result = generate_query_times(&times, 0.5, 4.0, 4);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn no_future_falls_back_to_evenly_spaced() {
        let times = vec![0.0_f32, 0.5];
        let result = generate_query_times(&times, 1.0, 4.0, 4);
        assert_eq!(result.len(), 4);
        assert!(result[0] > 0.25);
        assert!((result[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn at_or_past_end_returns_single_normalized() {
        let times = vec![0.0_f32, 1.0];
        let result = generate_query_times(&times, 4.0, 4.0, 4);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn max_steps_caps_future_keyframes() {
        let times: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let result = generate_query_times(&times, 0.0, 4.0, 4);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn zero_duration_handled() {
        let result = generate_query_times(&[], 0.0, 0.0, 4);
        assert!(!result.is_empty());
    }
}
