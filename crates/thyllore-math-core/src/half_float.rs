pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if (bits & 0x8000) != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x3FF) as f32;

    if exp == 0 {
        if mantissa == 0.0 {
            return sign * 0.0;
        }
        sign * (mantissa / 1024.0) * 2f32.powi(-14)
    } else if exp == 31 {
        if mantissa != 0.0 {
            f32::NAN
        } else {
            sign * f32::INFINITY
        }
    } else {
        let e = exp as i32 - 15;
        sign * (1.0 + mantissa / 1024.0) * 2f32.powi(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        let val = f16_to_f32(0x3555);
        assert!((val - 0.33325).abs() / 0.33325 < 1e-4, "got {}", val);
        let val = f16_to_f32(0x0001);
        let expected = 5.960464e-8;
        assert!(
            (val - expected).abs() / expected < 1e-3,
            "subnormal: got {}, expected {}",
            val,
            expected
        );
        assert!(f16_to_f32(0x7C00).is_infinite());
    }
}
