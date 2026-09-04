/// Knuth MMIX linear congruential generator; identical seeds replay identical sequences.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearCongruentialGenerator {
    state: u64,
}

impl LinearCongruentialGenerator {
    pub fn from_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub fn next_unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn next_angle_f32(&mut self) -> f32 {
        (self.next_unit_f64() * 2.0 * std::f64::consts::PI) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_replays_same_sequence() {
        let mut first = LinearCongruentialGenerator::from_seed(12345);
        let mut second = LinearCongruentialGenerator::from_seed(12345);
        for _ in 0..16 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
    }

    #[test]
    fn unit_samples_stay_in_range() {
        let mut generator = LinearCongruentialGenerator::from_seed(7);
        for _ in 0..1000 {
            let value = generator.next_unit_f64();
            assert!((0.0..1.0).contains(&value), "value {value} out of range");
        }
    }
}
