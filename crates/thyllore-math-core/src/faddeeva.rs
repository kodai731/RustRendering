type C = (f32, f32);

#[inline]
fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
#[inline]
fn cadd(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}
#[inline]
fn csub(a: C, b: C) -> C {
    (a.0 - b.0, a.1 - b.1)
}
#[inline]
fn cscale(a: C, s: f32) -> C {
    (a.0 * s, a.1 * s)
}
#[inline]
fn cdiv(a: C, b: C) -> C {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}
#[inline]
fn cexp(a: C) -> C {
    let e = a.0.exp();
    (e * a.1.cos(), e * a.1.sin())
}

/// Faddeeva function w(z) = exp(-z^2) * erfc(-i z), z = x + i y, y >= 0.
/// Humlicek (1979) 4-region rational approximation.
fn faddeeva_w4(x: f32, y: f32) -> C {
    let t: C = (y, -x);
    let s = x.abs() + y;
    if s >= 15.0 {
        let tt = cmul(t, t);
        let denom = cadd((0.5, 0.0), tt);
        let numer = cscale(t, 0.5641896);
        cdiv(numer, denom)
    } else if s >= 5.5 {
        let u = cmul(t, t);
        let inner = cadd((1.410474, 0.0), cscale(u, 0.5641896));
        let numer = cmul(t, inner);
        let denom = cadd((0.75, 0.0), cmul(u, cadd((3.0, 0.0), u)));
        cdiv(numer, denom)
    } else if y >= 0.195 * x.abs() - 0.176 {
        let numer = cadd(
            (16.4955, 0.0),
            cmul(
                t,
                cadd(
                    (20.20933, 0.0),
                    cmul(
                        t,
                        cadd(
                            (11.96482, 0.0),
                            cmul(t, cadd((3.778987, 0.0), cscale(t, 0.5642236))),
                        ),
                    ),
                ),
            ),
        );
        let denom = cadd(
            (16.4955, 0.0),
            cmul(
                t,
                cadd(
                    (38.82363, 0.0),
                    cmul(
                        t,
                        cadd(
                            (39.27121, 0.0),
                            cmul(t, cadd((21.69274, 0.0), cmul(t, cadd((6.699398, 0.0), t)))),
                        ),
                    ),
                ),
            ),
        );
        cdiv(numer, denom)
    } else {
        let u = cmul(t, t);
        let n1 = csub((1.320522, 0.0), cscale(u, 0.56419));
        let n2 = csub((35.76683, 0.0), cmul(u, n1));
        let n3 = csub((219.0313, 0.0), cmul(u, n2));
        let n4 = csub((1540.787, 0.0), cmul(u, n3));
        let n5 = csub((3321.9905, 0.0), cmul(u, n4));
        let numer = csub((36183.31, 0.0), cmul(u, n5));
        let d1 = csub((1.841439, 0.0), u);
        let d2 = csub((61.57037, 0.0), cmul(u, d1));
        let d3 = csub((364.2191, 0.0), cmul(u, d2));
        let d4 = csub((2186.181, 0.0), cmul(u, d3));
        let d5 = csub((9022.228, 0.0), cmul(u, d4));
        let d6 = csub((24322.84, 0.0), cmul(u, d5));
        let denom = csub((32066.6, 0.0), cmul(u, d6));
        let ratio = cdiv(numer, denom);
        csub(cexp(u), cmul(t, ratio))
    }
}

/// F_stable(x, k) = e^{-k^2} * erf(x - i k), numerically stable form. Assumes k >= 0.
pub fn f_stable(x: f32, k: f32) -> (f32, f32) {
    if x >= 0.0 {
        let (wr, wi) = faddeeva_w4(k, x);
        let ek2 = (-k * k).exp();
        let ex2 = (-x * x).exp();
        let theta = 2.0 * x * k;
        let (cos_theta, sin_theta) = (theta.cos(), theta.sin());
        let term_r = ex2 * (cos_theta * wr - sin_theta * wi);
        let term_i = ex2 * (cos_theta * wi + sin_theta * wr);
        (ek2 - term_r, -term_i)
    } else {
        let (vr, vi) = f_stable(-x, k);
        (-vr, vi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: (f32, f32), expected: (f32, f32), tol: f32, label: &str) {
        let dr = actual.0 - expected.0;
        let di = actual.1 - expected.1;
        let err = (dr * dr + di * di).sqrt();
        assert!(
            err <= tol,
            "{}: got ({:.8}, {:.8}), expected ({:.8}, {:.8}), error={:.8}",
            label,
            actual.0,
            actual.1,
            expected.0,
            expected.1,
            err
        );
    }

    #[test]
    fn test_faddeeva_w4_known() {
        let (wr, wi) = faddeeva_w4(1.0, 0.5);
        assert!(
            (wr - 0.35489944).abs() < 1e-3,
            "w real part wrong: got {:.8}",
            wr
        );
        assert!(
            (wi - 0.34287214).abs() < 1e-3,
            "w imag part wrong: got {:.8}",
            wi
        );
    }

    #[test]
    fn test_f_stable_values() {
        let tol = 5e-3;
        assert_close(
            f_stable(0.5, 1.0),
            (0.44323865, -0.37685602),
            tol,
            "F_stable(0.5, 1.0)",
        );
        assert_close(
            f_stable(-0.5, 1.0),
            (-0.44323865, -0.37685602),
            tol,
            "F_stable(-0.5, 1.0)",
        );
        assert_close(
            f_stable(0.0, 0.5),
            (0.0, -0.47892517),
            tol,
            "F_stable(0.0, 0.5)",
        );
        assert_close(
            f_stable(2.0, 0.3),
            (0.91280073, -0.00450625),
            tol,
            "F_stable(2.0, 0.3)",
        );
        assert_close(
            f_stable(-2.0, 0.3),
            (-0.91280073, -0.00450625),
            tol,
            "F_stable(-2.0, 0.3)",
        );
        assert_close(
            f_stable(0.05, 2.0),
            (0.05599717, -0.33441012),
            tol,
            "F_stable(0.05, 2.0)",
        );
        assert_close(
            f_stable(1.5, 0.8),
            (0.55711813, -0.01079832),
            tol,
            "F_stable(1.5, 0.8)",
        );
        assert_close(
            f_stable(-1.5, 0.8),
            (-0.55711813, -0.01079832),
            tol,
            "F_stable(-1.5, 0.8)",
        );
    }
}
