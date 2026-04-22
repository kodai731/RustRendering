use crate::Keyframe;

pub fn find_keyframe_segment<T>(keyframes: &[Keyframe<T>], time: f32) -> usize {
    let idx = keyframes.partition_point(|kf| kf.time <= time);
    if idx == 0 {
        0
    } else {
        (idx - 1).min(keyframes.len().saturating_sub(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::Vector3;

    fn make_keyframes(times: &[f32]) -> Vec<Keyframe<Vector3<f32>>> {
        times
            .iter()
            .map(|&t| Keyframe::new(t, Vector3::new(t, 0.0, 0.0)))
            .collect()
    }

    #[test]
    fn test_before_first_keyframe() {
        let kfs = make_keyframes(&[1.0, 2.0, 3.0]);
        assert_eq!(find_keyframe_segment(&kfs, 0.5), 0);
    }

    #[test]
    fn test_on_first_keyframe() {
        let kfs = make_keyframes(&[1.0, 2.0, 3.0]);
        assert_eq!(find_keyframe_segment(&kfs, 1.0), 0);
    }

    #[test]
    fn test_between_keyframes() {
        let kfs = make_keyframes(&[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(find_keyframe_segment(&kfs, 0.5), 0);
        assert_eq!(find_keyframe_segment(&kfs, 1.5), 1);
        assert_eq!(find_keyframe_segment(&kfs, 2.5), 2);
    }

    #[test]
    fn test_on_last_keyframe() {
        let kfs = make_keyframes(&[0.0, 1.0, 2.0]);
        assert_eq!(find_keyframe_segment(&kfs, 2.0), 1);
    }

    #[test]
    fn test_after_last_keyframe() {
        let kfs = make_keyframes(&[0.0, 1.0, 2.0]);
        assert_eq!(find_keyframe_segment(&kfs, 5.0), 1);
    }

    #[test]
    fn test_two_keyframes() {
        let kfs = make_keyframes(&[0.0, 1.0]);
        assert_eq!(find_keyframe_segment(&kfs, -1.0), 0);
        assert_eq!(find_keyframe_segment(&kfs, 0.0), 0);
        assert_eq!(find_keyframe_segment(&kfs, 0.5), 0);
        assert_eq!(find_keyframe_segment(&kfs, 1.0), 0);
        assert_eq!(find_keyframe_segment(&kfs, 2.0), 0);
    }

    #[test]
    fn test_many_keyframes() {
        let times: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let kfs = make_keyframes(&times);

        assert_eq!(find_keyframe_segment(&kfs, 0.05), 0);
        assert_eq!(find_keyframe_segment(&kfs, 5.05), 50);
        assert_eq!(find_keyframe_segment(&kfs, 9.85), 98);
    }
}
