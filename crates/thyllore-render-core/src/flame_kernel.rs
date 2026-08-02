//! Kernel-basis erosion noise: deterministic biweight blobs replacing the fbm
//! lattice as the erosion noise source when `turbulence_model == 1`. Stateless
//! in time; GLSL reads the same list via the `kernelBlobs` UBO
//! (flameKernelBlobDensityAt).

pub const KERNEL_BLOB_COUNT: usize = 96;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KernelBlob {
    pub center: [f32; 3],
    pub radius: f32,
    pub amplitude: f32,
}

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn hash01(slot: u32, stream: u32) -> f32 {
    (hash_u32(slot.wrapping_mul(0x9e37_79b9) ^ stream.wrapping_mul(0x85eb_ca6b)) >> 8) as f32
        / (1u32 << 24) as f32
}

#[derive(Clone, Copy, Debug)]
pub struct KernelBlobParams {
    /// 0 = cylinder column, 1 = ring circle (matches emitter_kind 0/1).
    pub emitter_kind: u32,
    /// Ring major radius in flame-local units (emitter_params.y of the UBO).
    pub ring_major_norm: f32,
    /// Base blob radius in flame-local units.
    pub blob_size: f32,
    /// Peak blob density amplitude.
    pub blob_amp: f32,
    /// Upward advection speed; one life spans the unit height.
    pub rise_speed: f32,
    pub time: f32,
}

/// Life-cycle amplitude envelope: zero at both ends of the period so the
/// wrap of `fract` never pops, peaking near the lower third of the rise.
fn life_envelope(u: f32) -> f32 {
    3.0 * u.sqrt() * (1.0 - u) * (1.0 - u).sqrt()
}

/// Octave split (few large, many small): multi-scale spectrum like fbm.
fn octave_scales(slot: usize) -> (f32, f32) {
    if slot < 18 {
        (1.6, 1.0)
    } else if slot < 48 {
        (0.8, 0.7)
    } else {
        (0.4, 0.5)
    }
}

pub fn generate_kernel_blobs(params: &KernelBlobParams) -> [KernelBlob; KERNEL_BLOB_COUNT] {
    let mut blobs = [KernelBlob::default(); KERNEL_BLOB_COUNT];
    let rise = params.rise_speed.max(0.1);
    for (slot, blob) in blobs.iter_mut().enumerate() {
        let k = slot as u32;
        let (size_scale, amp_scale) = octave_scales(slot);
        let lifetime = (0.8 + 0.5 * hash01(k, 0)) / rise;
        let phase = hash01(k, 1);
        let u = (params.time / lifetime + phase).fract();

        let angle =
            std::f32::consts::TAU * (slot as f32 / KERNEL_BLOB_COUNT as f32 + 0.37 * hash01(k, 2));
        let (sin_a, cos_a) = angle.sin_cos();
        let spawn_radius = if params.emitter_kind == 1 {
            let minor_jitter = (hash01(k, 3) - 0.5) * 0.8 * (1.0 - params.ring_major_norm);
            params.ring_major_norm * (1.0 - 0.15 * u) + minor_jitter
        } else {
            0.35 * hash01(k, 3).sqrt() * (1.0 - 0.6 * u)
        };

        let drift_phase = std::f32::consts::TAU * hash01(k, 4);
        let drift = 0.05 * (std::f32::consts::TAU * (2.0 * u) + drift_phase).sin();

        blob.center = [
            spawn_radius * cos_a - drift * sin_a,
            u,
            spawn_radius * sin_a + drift * cos_a,
        ];
        blob.radius = params.blob_size * size_scale * (0.7 + 0.6 * hash01(k, 5)) * (0.6 + 0.8 * u);
        blob.amplitude =
            params.blob_amp * amp_scale * (0.7 + 0.6 * hash01(k, 6)) * life_envelope(u);
    }
    blobs
}

pub fn evaluate_kernel_blob_density(blobs: &[KernelBlob], p: [f32; 3]) -> f32 {
    let mut total = 0.0;
    for blob in blobs {
        if blob.amplitude <= 0.0 || blob.radius <= 0.0 {
            continue;
        }
        let dx = p[0] - blob.center[0];
        let dy = p[1] - blob.center[1];
        let dz = p[2] - blob.center[2];
        let u2 = (dx * dx + dy * dy + dz * dz) / (blob.radius * blob.radius);
        let inside = (1.0 - u2).max(0.0);
        total += blob.amplitude * inside * inside;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params(emitter_kind: u32, time: f32) -> KernelBlobParams {
        KernelBlobParams {
            emitter_kind,
            ring_major_norm: 0.6,
            blob_size: 0.15,
            blob_amp: 1.0,
            rise_speed: 1.5,
            time,
        }
    }

    #[test]
    fn test_generation_is_deterministic() {
        let params = test_params(1, 3.7);
        assert_eq!(
            generate_kernel_blobs(&params),
            generate_kernel_blobs(&params)
        );
    }

    #[test]
    fn test_blobs_stay_in_local_bounds_and_wrap_without_pop() {
        for kind in [0u32, 1] {
            for step in 0..200 {
                let params = test_params(kind, step as f32 * 0.173);
                for blob in generate_kernel_blobs(&params) {
                    assert!((0.0..=1.0).contains(&blob.center[1]));
                    let r_xz =
                        (blob.center[0] * blob.center[0] + blob.center[2] * blob.center[2]).sqrt();
                    assert!(r_xz <= 0.9, "xz radius {r_xz} out of local bounds");
                    assert!(blob.radius > 0.0 && blob.amplitude >= 0.0);
                }
            }
        }
        assert_eq!(life_envelope(0.0), 0.0);
        assert_eq!(life_envelope(1.0), 0.0);
    }
}
