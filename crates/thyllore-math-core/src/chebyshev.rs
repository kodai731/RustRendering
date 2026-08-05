use std::f64::consts::PI;

#[derive(Debug, Clone, PartialEq)]
pub struct ChebyshevSeries {
    coefficients: Vec<f32>,
    domain: (f32, f32),
}

impl ChebyshevSeries {
    pub fn new(coefficients: Vec<f32>, domain: (f32, f32)) -> Self {
        Self {
            coefficients,
            domain,
        }
    }

    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }

    pub fn domain(&self) -> (f32, f32) {
        self.domain
    }
}

pub fn fit_chebyshev(
    target: impl Fn(f64) -> f64,
    domain: (f64, f64),
    coefficient_count: usize,
) -> ChebyshevSeries {
    assert!(coefficient_count > 0, "coefficient_count must be positive");
    assert!(domain.1 > domain.0, "domain must satisfy min < max");

    let node_count = coefficient_count;
    let center = 0.5 * (domain.0 + domain.1);
    let half_width = 0.5 * (domain.1 - domain.0);
    let node_angles: Vec<f64> = (0..node_count)
        .map(|k| (2 * k + 1) as f64 * PI / (2 * node_count) as f64)
        .collect();
    let samples: Vec<f64> = node_angles
        .iter()
        .map(|theta| target(center + half_width * theta.cos()))
        .collect();

    let coefficients = (0..coefficient_count)
        .map(|j| {
            let weighted_sum: f64 = node_angles
                .iter()
                .zip(&samples)
                .map(|(theta, sample)| sample * (j as f64 * theta).cos())
                .sum();
            let scale = if j == 0 { 1.0 } else { 2.0 } / node_count as f64;
            (scale * weighted_sum) as f32
        })
        .collect();

    ChebyshevSeries {
        coefficients,
        domain: (domain.0 as f32, domain.1 as f32),
    }
}

/// Interpolation at Gauss-Lobatto nodes (endpoints included), so the series value at
/// the domain ends equals the sampled target there — the property C0 piecewise joints
/// rely on. Coefficients follow the same convention as `fit_chebyshev` (c0 unhalved).
pub fn fit_chebyshev_lobatto(
    target: impl Fn(f64) -> f64,
    domain: (f64, f64),
    coefficient_count: usize,
) -> ChebyshevSeries {
    assert!(
        coefficient_count >= 2,
        "Lobatto interpolation needs both endpoints"
    );
    assert!(domain.1 > domain.0, "domain must satisfy min < max");

    let order = coefficient_count - 1;
    let center = 0.5 * (domain.0 + domain.1);
    let half_width = 0.5 * (domain.1 - domain.0);
    let samples: Vec<f64> = (0..=order)
        .map(|j| target(center + half_width * (PI * j as f64 / order as f64).cos()))
        .collect();

    let coefficients = (0..=order)
        .map(|i| {
            let weighted_sum: f64 = samples
                .iter()
                .enumerate()
                .map(|(j, sample)| {
                    let endpoint_weight = if j == 0 || j == order { 0.5 } else { 1.0 };
                    endpoint_weight * sample * (PI * (i * j) as f64 / order as f64).cos()
                })
                .sum();
            let scale = if i == 0 || i == order { 1.0 } else { 2.0 } / order as f64;
            (scale * weighted_sum) as f32
        })
        .collect();

    ChebyshevSeries {
        coefficients,
        domain: (domain.0 as f32, domain.1 as f32),
    }
}

/// Series values at the domain ends in closed form: `(sum (-1)^i c_i, sum c_i)`.
pub fn chebyshev_endpoint_values(series: &ChebyshevSeries) -> (f32, f32) {
    let mut at_min = 0.0f64;
    let mut at_max = 0.0f64;
    for (i, &coefficient) in series.coefficients.iter().enumerate() {
        at_max += coefficient as f64;
        at_min += if i % 2 == 0 {
            coefficient as f64
        } else {
            -coefficient as f64
        };
    }
    (at_min as f32, at_max as f32)
}

pub fn evaluate_chebyshev(series: &ChebyshevSeries, x: f32) -> f32 {
    let (min, max) = series.domain;
    let u = (2.0 * x - min - max) / (max - min);

    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    for &c in series.coefficients.iter().skip(1).rev() {
        let b0 = 2.0 * u * b1 - b2 + c;
        b2 = b1;
        b1 = b0;
    }
    u * b1 - b2 + series.coefficients[0]
}

pub fn integrate_chebyshev(series: &ChebyshevSeries) -> ChebyshevSeries {
    let source = &series.coefficients;
    let source_count = source.len();
    let (min, max) = series.domain;
    let half_width = 0.5 * (max - min);

    let mut primitive = vec![0.0f32; source_count + 1];
    for k in 1..=source_count {
        let c_prev = if k == 1 {
            2.0 * source[0]
        } else {
            source[k - 1]
        };
        let c_next = if k + 1 < source_count {
            source[k + 1]
        } else {
            0.0
        };
        primitive[k] = half_width * (c_prev - c_next) / (2.0 * k as f32);
    }

    let value_at_min: f32 = primitive
        .iter()
        .enumerate()
        .skip(1)
        .map(|(k, &coefficient)| {
            if k % 2 == 0 {
                coefficient
            } else {
                -coefficient
            }
        })
        .sum();
    primitive[0] = -value_at_min;

    ChebyshevSeries {
        coefficients: primitive,
        domain: series.domain,
    }
}

/// Monomial coefficients in the normalized variable `u = (2x - min - max) / (max - min)`
/// of a Chebyshev series: `sum c_k T_k(u) = sum m_j u^j`. The `T_k` monomial
/// coefficients are exact integers, accumulated in f64 so the conversion adds no
/// error beyond the final f32 rounding.
pub fn chebyshev_monomial_coefficients(series: &ChebyshevSeries) -> Vec<f32> {
    let count = series.coefficients.len();
    let mut monomial = vec![0.0f64; count];
    let mut t_prev = vec![0.0f64; count];
    let mut t_cur = vec![0.0f64; count];
    t_prev[0] = 1.0;
    monomial[0] = series.coefficients[0] as f64;
    if count > 1 {
        t_cur[1] = 1.0;
        monomial[1] = series.coefficients[1] as f64;
    }
    for k in 2..count {
        let mut t_next = vec![0.0f64; count];
        for j in 1..count {
            t_next[j] = 2.0 * t_cur[j - 1];
        }
        for j in 0..count {
            t_next[j] -= t_prev[j];
        }
        for j in 0..count {
            monomial[j] += series.coefficients[k] as f64 * t_next[j];
        }
        t_prev = t_cur;
        t_cur = t_next;
    }
    monomial.into_iter().map(|m| m as f32).collect()
}

pub fn pack_coefficients_vec4(series: &ChebyshevSeries, slot_count: usize) -> Vec<[f32; 4]> {
    let coefficient_count = series.coefficients.len();
    assert!(
        slot_count * 4 >= coefficient_count,
        "slot_count * 4 ({}) must hold all {} coefficients",
        slot_count * 4,
        coefficient_count
    );

    let mut slots = vec![[0.0f32; 4]; slot_count];
    for (i, &coefficient) in series.coefficients.iter().enumerate() {
        slots[i / 4][i % 4] = coefficient;
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compute_max_fit_error(series: &ChebyshevSeries, target: impl Fn(f64) -> f64) -> f64 {
        let (min, max) = series.domain;
        (0..256)
            .map(|i| {
                let x = min as f64 + (max - min) as f64 * i as f64 / 255.0;
                let approximation = evaluate_chebyshev(series, x as f32) as f64;
                (approximation - target(x)).abs()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn test_fit_chebyshev_gaussian_within_tolerance() {
        let target = |x: f64| (-4.0 * x * x).exp();
        let series = fit_chebyshev(target, (0.0, 1.0), 8);
        assert!(compute_max_fit_error(&series, target) < 1e-3);
    }

    #[test]
    fn test_fit_chebyshev_smoothstep_within_tolerance() {
        let target = |x: f64| x * x * (3.0 - 2.0 * x);
        let series = fit_chebyshev(target, (0.0, 1.0), 8);
        assert!(compute_max_fit_error(&series, target) < 1e-3);
    }

    #[test]
    fn test_monomial_coefficients_match_clenshaw() {
        let target = |x: f64| (-4.0 * x * x).exp() + 0.3 * (5.0 * x).sin();
        let series = fit_chebyshev(target, (0.0, 1.0), 8);
        let monomial = chebyshev_monomial_coefficients(&series);
        for i in 0..=256 {
            let x = i as f32 / 256.0;
            let u = 2.0 * x - 1.0;
            let mut horner = 0.0f64;
            for &m in monomial.iter().rev() {
                horner = horner * u as f64 + m as f64;
            }
            let clenshaw = evaluate_chebyshev(&series, x) as f64;
            assert!(
                (horner - clenshaw).abs() < 1e-5,
                "x={x}: monomial {horner} vs clenshaw {clenshaw}"
            );
        }
    }

    #[test]
    fn test_fit_chebyshev_lobatto_interpolates_endpoints() {
        let target = |x: f64| (-4.0 * x * x).exp() + 0.3 * (3.0 * x).sin();
        let series = fit_chebyshev_lobatto(target, (0.2, 1.3), 10);
        let (at_min, at_max) = chebyshev_endpoint_values(&series);
        assert!((at_min as f64 - target(0.2)).abs() < 1e-6);
        assert!((at_max as f64 - target(1.3)).abs() < 1e-6);
        // Clenshaw evaluation at the ends agrees with the closed-form sums.
        assert!((evaluate_chebyshev(&series, 0.2) - at_min).abs() < 1e-6);
        assert!((evaluate_chebyshev(&series, 1.3) - at_max).abs() < 1e-6);
    }

    #[test]
    fn test_fit_chebyshev_lobatto_accuracy_matches_gauss_nodes() {
        let target = |x: f64| (-4.0 * x * x).exp();
        let series = fit_chebyshev_lobatto(target, (0.0, 1.0), 10);
        assert!(compute_max_fit_error(&series, target) < 1e-4);
    }

    #[test]
    fn test_evaluate_chebyshev_matches_direct_sum() {
        let series = ChebyshevSeries {
            coefficients: vec![0.3, -0.5, 0.2, 0.7, -0.1, 0.05],
            domain: (-1.0, 1.0),
        };
        for i in 0..=100 {
            let x = (-1.0 + 2.0 * i as f64 / 100.0) as f32;
            let direct: f64 = series
                .coefficients
                .iter()
                .enumerate()
                .map(|(j, &c)| c as f64 * (j as f64 * (x as f64).acos()).cos())
                .sum();
            let clenshaw = evaluate_chebyshev(&series, x) as f64;
            assert!(
                (clenshaw - direct).abs() < 1e-6,
                "x = {x}: clenshaw = {clenshaw}, direct = {direct}"
            );
        }
    }

    #[test]
    fn test_integrate_chebyshev_derivative_matches_source() {
        let target = |x: f64| (-4.0 * x * x).exp();
        let series = fit_chebyshev(target, (0.0, 1.0), 12);
        let primitive = integrate_chebyshev(&series);

        let h = 0.02f32;
        for i in 0..=50 {
            let x = 2.0 * h + (1.0 - 4.0 * h) * i as f32 / 50.0;
            let derivative = (-evaluate_chebyshev(&primitive, x + 2.0 * h)
                + 8.0 * evaluate_chebyshev(&primitive, x + h)
                - 8.0 * evaluate_chebyshev(&primitive, x - h)
                + evaluate_chebyshev(&primitive, x - 2.0 * h))
                / (12.0 * h);
            let expected = evaluate_chebyshev(&series, x);
            assert!(
                (derivative - expected).abs() < 1e-4,
                "x = {x}: derivative = {derivative}, expected = {expected}"
            );
        }
    }

    #[test]
    fn test_integrate_chebyshev_is_zero_at_domain_min() {
        let target = |x: f64| (-4.0 * x * x).exp();
        let primitive = integrate_chebyshev(&fit_chebyshev(target, (0.0, 1.0), 12));
        assert!(evaluate_chebyshev(&primitive, 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_pack_coefficients_vec4_pads_with_zero() {
        let series = ChebyshevSeries {
            coefficients: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            domain: (0.0, 1.0),
        };
        let slots = pack_coefficients_vec4(&series, 2);
        assert_eq!(slots, vec![[1.0, 2.0, 3.0, 4.0], [5.0, 0.0, 0.0, 0.0]]);
    }

    #[test]
    #[should_panic(expected = "must hold all")]
    fn test_pack_coefficients_vec4_panics_when_slots_insufficient() {
        let series = ChebyshevSeries {
            coefficients: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            domain: (0.0, 1.0),
        };
        pack_coefficients_vec4(&series, 1);
    }

    #[test]
    fn test_fit_chebyshev_is_deterministic() {
        let target = |x: f64| (-4.0 * x * x).exp();
        let first = fit_chebyshev(target, (0.0, 1.0), 8);
        let second = fit_chebyshev(target, (0.0, 1.0), 8);
        let first_bits: Vec<u32> = first.coefficients().iter().map(|c| c.to_bits()).collect();
        let second_bits: Vec<u32> = second.coefficients().iter().map(|c| c.to_bits()).collect();
        assert_eq!(first_bits, second_bits);
    }
}
