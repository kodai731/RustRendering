use crate::flame::*;

/// Emitter shape: 0 = axial column, 1 = ring of `ring_major_radius`, 2 = SDF billboard slab.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameEmitter {
    pub kind: u32,
    pub ring_major_radius: f32,
    pub ring_angular_speed: f32,
}

impl Default for FlameEmitter {
    fn default() -> Self {
        Self {
            kind: 0,
            ring_major_radius: 1.0,
            ring_angular_speed: 0.6,
        }
    }
}

const SDF_SLAB_DEPTH: f32 = 0.15;

pub fn emitter_bounding_radius(emitter: &FlameEmitter, radius: f32) -> f32 {
    if emitter.kind == 1 {
        emitter.ring_major_radius + radius
    } else {
        radius
    }
}

pub fn build_emitter_params(emitter: &FlameEmitter, radius: f32) -> FlameEmitterParams {
    FlameEmitterParams {
        kind: emitter.kind as f32,
        ring_major_ratio: if emitter.kind == 1 {
            emitter.ring_major_radius / emitter_bounding_radius(emitter, radius)
        } else {
            0.0
        },
        ring_angular_speed: emitter.ring_angular_speed,
        sdf_slab_depth: if emitter.kind == 2 {
            SDF_SLAB_DEPTH
        } else {
            0.0
        },
    }
}
