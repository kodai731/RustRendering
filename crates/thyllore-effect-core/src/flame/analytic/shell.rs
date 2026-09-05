// Analytic shell envelope of the flame proxy (cone x y-slab). The resolve pass
// derives the ray interval from it in closed form (clampToShellCone in
// shaders/flame/flameResolveFragment.frag + shaders/flame/include/flame_shell_profile.glsl);
// no rasterized proxy geometry exists. These constants and functions are the Rust
// mirror used by the scissor bound (pass_recording) and CPU picking (flame_pick).
pub const FLAME_SHELL_BASE_RADIUS: f32 = 0.5;
pub const FLAME_SHELL_TAPER_TIP_SCALE: f32 = 1.0;
// Historic width factor (1/cos(pi/16)) kept so the analytic envelope matches the
// former circumscribed 16-gon proxy exactly; it is now just part of the proxy width.
pub const FLAME_SHELL_CIRCUMSCRIBE: f32 = 1.0195911;
pub const FLAME_SHELL_SUPPORT_HEADROOM: f32 = 1.5;

/// Emitter-dependent widening of the proxy: a ring's tube (centerline at normalized
/// major radius rm, minor support 1.5 * (1 - rm)) reaches past the cylinder support
/// 0.75, and a proxy that stops there slices the torus flat. Mirrors
/// shaders/flame/include/flame_shell_support.glsl.
pub fn flame_shell_support_scale(
    emitter_kind: u32,
    ring_major_norm: f32,
    support_margin: f32,
) -> f32 {
    if emitter_kind == 1 {
        let rm = ring_major_norm;
        ((rm + FLAME_SHELL_SUPPORT_HEADROOM * support_margin * (1.0 - rm))
            / (FLAME_SHELL_BASE_RADIUS * FLAME_SHELL_SUPPORT_HEADROOM * support_margin))
            .max(1.0)
    } else {
        1.0
    }
}

/// Multiplier on the shell's base half-extent at a normalized height.
pub fn flame_shell_radius_scale(height01: f32, support_scale: f32, support_margin: f32) -> f32 {
    support_scale
        * FLAME_SHELL_SUPPORT_HEADROOM
        * support_margin
        * FLAME_SHELL_CIRCUMSCRIBE
        * (1.0 + (FLAME_SHELL_TAPER_TIP_SCALE - 1.0) * height01)
}

/// Outer radius of the shell in flame-local units. Linear in height, so callers that
/// need a bound over the whole shell take the maximum of the two endpoints.
pub fn flame_shell_outer_radius(height01: f32, support_scale: f32, support_margin: f32) -> f32 {
    FLAME_SHELL_BASE_RADIUS * flame_shell_radius_scale(height01, support_scale, support_margin)
}
