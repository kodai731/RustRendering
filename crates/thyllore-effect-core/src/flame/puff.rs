use super::*;
use crate::flame::branch::hash01;

/// Live puffs at `time`, newest first, as (center height, lateral radius,
/// density, vertical radius):
/// the characteristic solution of the axial advection equation. Derived from
/// (parameters, time) only. Radii are in trunk-local radial units.
pub fn active_puffs(puff: &FlamePuff, base_trunk_radius: f32, time: f32) -> Vec<[f32; 4]> {
    if puff.gain == 0.0 || puff.period <= 0.0 || puff.rise <= 0.0 {
        return Vec::new();
    }
    let spawn_radius = puff.radius.max(1e-3) * base_trunk_radius;
    let exit_height = 1.0 + spawn_radius * (1.0 + puff.spread.max(0.0));
    let travel_time = (exit_height - puff.spawn_height) / puff.rise;

    let first = ((time - travel_time) / puff.period).floor() as i64 - 1;
    let last = (time / puff.period).ceil() as i64 + 1;
    let mut puffs: Vec<(f32, [f32; 4])> = (first..=last)
        .filter_map(|index| {
            let jitter = PUFF_SPAWN_JITTER * (hash01(0, index, 11) - 0.5);
            let spawn_time = (index as f32 + jitter) * puff.period;
            let age = time - spawn_time;
            let height = puff.spawn_height + puff.rise * age;
            if age < 0.0 || height >= exit_height {
                return None;
            }
            let radius = spawn_radius * (1.0 + puff.spread.max(0.0) * height);
            let density = if puff.decay > 0.0 {
                (-height / puff.decay).exp()
            } else {
                1.0
            };
            Some((
                spawn_time,
                [height, radius, density, radius * puff.aspect.max(1e-3)],
            ))
        })
        .collect();
    puffs.sort_by(|a, b| b.0.total_cmp(&a.0));
    puffs.truncate(PUFF_MAX_COUNT);

    let mut result: Vec<[f32; 4]> = puffs.into_iter().map(|(_, entry)| entry).collect();

    if puff.root_gain > 0.0 {
        let root_radius = spawn_radius * (1.0 + puff.spread.max(0.0) * puff.root_height);
        let root_entry: [f32; 4] = [
            puff.root_height,
            root_radius,
            puff.root_gain,
            root_radius * puff.aspect.max(1e-3),
        ];
        if result.len() >= PUFF_MAX_COUNT {
            result.pop();
        }
        result.insert(0, root_entry);
    }

    result
}

pub fn build_puff_field(effect: &FlameEffect, baked: &FlameBaked) -> FlamePuffField {
    let base_trunk_radius = branch_trunk_radius_at(effect, baked, 0.0);
    let active = active_puffs(&effect.puff, base_trunk_radius, effect.time);
    let mut puffs = [[0.0f32; 4]; PUFF_MAX_COUNT];
    for (slot, entry) in puffs.iter_mut().zip(&active) {
        *slot = *entry;
    }
    FlamePuffField {
        count: active.len() as f32,
        gain: effect.puff.gain,
        aspect: branch_aspect(effect),
        _padding: 0.0,
        puffs,
    }
}

/// Mirror of flamePuffDensityFactor at trunk-local `ps`.
pub fn puff_density_factor(field: &FlamePuffField, ps: [f32; 3], u: f32) -> f32 {
    let count = (field.count as usize).min(PUFF_MAX_COUNT);
    if count == 0 {
        return 1.0;
    }
    let sum: f32 = field.puffs[..count]
        .iter()
        .map(|puff| {
            let dy = (ps[1] - puff[0]) * field.aspect;
            let r2 = (ps[0] * ps[0] + ps[2] * ps[2]) / (puff[1] * puff[1]).max(1e-6)
                + dy * dy / (puff[3] * puff[3]).max(1e-6);
            puff[2] * (-r2).exp()
        })
        .sum();
    let interior = (1.0 - field.gain) + field.gain * sum.min(1.0);
    let t = ((u - 0.6) / 0.35).clamp(0.0, 1.0);
    let edge_fade = t * t * (3.0 - 2.0 * t);
    interior + (1.0 - interior) * edge_fade
}

#[cfg(test)]
mod tests {
    use super::*;

    fn puff_on() -> FlamePuff {
        FlamePuff {
            gain: 1.0,
            period: 0.5,
            rise: 0.25,
            radius: 0.6,
            spread: 0.5,
            decay: 0.8,
            aspect: 1.0,
            spawn_height: 0.0,
            root_gain: 0.0,
            root_height: 0.0,
        }
    }

    #[test]
    fn gain_zero_is_identity() {
        let mut effect = FlameEffect::default();
        effect.time = 3.0;
        let field = build_puff_field(&effect, &FlameBaked::default());
        assert_eq!(field.count, 0.0);
        assert_eq!(puff_density_factor(&field, [0.0, 0.5, 0.0], 0.0), 1.0);
    }

    #[test]
    fn puffs_rise_with_time_and_stay_inside_the_column() {
        let puff = puff_on();
        let early = active_puffs(&puff, 1.0, 2.0);
        let later = active_puffs(&puff, 1.0, 2.1);
        assert!(!early.is_empty() && early.len() <= PUFF_MAX_COUNT);
        let exit_height = 1.0 + 0.6 * 1.5;
        for a in &early {
            let risen = a[0] + 0.025;
            let followed = later.iter().any(|b| (b[0] - risen).abs() < 1e-5);
            assert!(
                followed || risen >= exit_height,
                "puff at {} rose by rise * dt",
                a[0]
            );
        }
        for entry in &early {
            assert!(entry[0] >= 0.0 && entry[0] < 1.0 + 0.6 * 1.5);
            assert!(entry[2] <= 1.0 && entry[2] > 0.0);
        }
    }

    #[test]
    fn density_factor_peaks_at_the_puff_center() {
        let mut effect = FlameEffect::default();
        effect.puff = puff_on();
        effect.time = 4.0;
        let field = build_puff_field(&effect, &FlameBaked::default());
        let center = field.puffs[0];
        let at_center = puff_density_factor(&field, [0.0, center[0], 0.0], 0.0);
        let beside = puff_density_factor(&field, [2.0 * center[1], center[0], 0.0], 0.0);
        assert!(at_center > 1.0 - 0.1 * field.gain - (1.0 - center[2]) * field.gain);
        assert!(beside < at_center);
        assert!(beside >= 1.0 - field.gain);
    }

    #[test]
    fn spawn_height_raises_all_puff_heights() {
        let mut puff = puff_on();
        puff.spawn_height = 0.3;
        let base = puff_on();
        let shifted = active_puffs(&puff, 1.0, 4.0);
        let baseline = active_puffs(&base, 1.0, 4.0);
        assert!(!shifted.is_empty());
        for entry in &shifted {
            assert!(
                entry[0] >= 0.3 - 1e-6,
                "height {} < spawn_height 0.3",
                entry[0]
            );
        }
        // With higher spawn_height, puffs reach exit_height faster, so fewer are active.
        // Match by index: each shifted puff corresponds to a baseline puff at the same
        // spawn time (same index in the filtered range), offset by 0.3 in height.
        assert!(
            shifted.len() <= baseline.len(),
            "shifted count {} > baseline count {}",
            shifted.len(),
            baseline.len()
        );
        for (i, s) in shifted.iter().enumerate() {
            let b = &baseline[i];
            let diff = s[0] - b[0];
            assert!(
                (diff - 0.3).abs() < 1e-5,
                "puff[{}] height offset {:.4} != 0.3",
                i,
                diff
            );
        }
    }

    #[test]
    fn root_puff_is_static_at_front() {
        let mut puff = puff_on();
        puff.root_gain = 0.6;
        puff.root_height = 0.1;
        let spawn_radius = puff.radius * 1.0;
        let expected_radius = spawn_radius * (1.0 + puff.spread * puff.root_height);
        let expected_entry: [f32; 4] = [
            puff.root_height,
            expected_radius,
            puff.root_gain,
            expected_radius * puff.aspect,
        ];

        let puffs_t0 = active_puffs(&puff, 1.0, 4.0);
        let puffs_t1 = active_puffs(&puff, 1.0, 5.0);
        assert!(!puffs_t0.is_empty());
        for i in 0..4 {
            assert!(
                (puffs_t0[0][i] - expected_entry[i]).abs() < 1e-6,
                "root entry[{}] {:.6} != {:.6}",
                i,
                puffs_t0[0][i],
                expected_entry[i]
            );
        }
        assert!(
            (puffs_t1[0][0] - puffs_t0[0][0]).abs() < 1e-6,
            "root entry height changed over time"
        );

        let base = puff_on();
        let baseline = active_puffs(&base, 1.0, 4.0);
        let expected_len = (baseline.len() + 1).min(PUFF_MAX_COUNT);
        assert_eq!(puffs_t0.len(), expected_len);
        for i in 0..expected_len - 1 {
            for j in 0..4 {
                assert!(
                    (puffs_t0[i + 1][j] - baseline[i][j]).abs() < 1e-6,
                    "moving puff[{}][{}] differs from baseline",
                    i,
                    j
                );
            }
        }
    }
}
