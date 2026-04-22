use crate::editable::components::snap_settings::SnapSettings;

pub fn snap_time(
    raw_time: f32,
    snap_settings: &SnapSettings,
    nearby_times: &[f32],
    snap_threshold_time: f32,
) -> f32 {
    if snap_settings.snap_to_key {
        if let Some(snapped) = find_nearest_within(raw_time, nearby_times, snap_threshold_time) {
            return snapped;
        }
    }

    if snap_settings.snap_to_frame {
        return snap_to_frame(raw_time, snap_settings.frame_rate);
    }

    raw_time
}

fn find_nearest_within(time: f32, candidates: &[f32], threshold: f32) -> Option<f32> {
    candidates
        .iter()
        .copied()
        .min_by(|a, b| {
            let da = (a - time).abs();
            let db = (b - time).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .filter(|nearest| (nearest - time).abs() <= threshold)
}

fn snap_to_frame(time: f32, frame_rate: f32) -> f32 {
    if frame_rate <= 0.0 {
        return time;
    }
    (time * frame_rate).round() / frame_rate
}

pub fn compute_snap_threshold_time(threshold_px: f32, pixels_per_second: f32) -> f32 {
    if pixels_per_second <= 0.0 {
        return 0.0;
    }
    threshold_px / pixels_per_second
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snap_to_frame_30fps() {
        let settings = SnapSettings {
            snap_to_frame: true,
            snap_to_key: false,
            frame_rate: 30.0,
            snap_threshold_px: 8.0,
        };

        let result = snap_time(0.51, &settings, &[], 0.0);
        let expected = (0.51_f32 * 30.0).round() / 30.0;
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_snap_to_key_nearest() {
        let settings = SnapSettings {
            snap_to_frame: false,
            snap_to_key: true,
            frame_rate: 30.0,
            snap_threshold_px: 8.0,
        };

        let nearby = vec![0.0, 0.5, 1.0, 1.5];
        let result = snap_time(0.48, &settings, &nearby, 0.1);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_no_snap_when_disabled() {
        let settings = SnapSettings {
            snap_to_frame: false,
            snap_to_key: false,
            frame_rate: 30.0,
            snap_threshold_px: 8.0,
        };

        let nearby = vec![0.0, 0.5, 1.0];
        let result = snap_time(0.48, &settings, &nearby, 0.1);
        assert!((result - 0.48).abs() < 1e-6);
    }

    #[test]
    fn test_snap_to_key_outside_threshold() {
        let settings = SnapSettings {
            snap_to_frame: false,
            snap_to_key: true,
            frame_rate: 30.0,
            snap_threshold_px: 8.0,
        };

        let nearby = vec![0.0, 1.0];
        let result = snap_time(0.5, &settings, &nearby, 0.1);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_compute_snap_threshold_time() {
        let threshold = compute_snap_threshold_time(8.0, 200.0);
        assert!((threshold - 0.04).abs() < 1e-6);
    }
}
