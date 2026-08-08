#ifndef FLAME_SHELL_PROFILE_GLSL
#define FLAME_SHELL_PROFILE_GLSL

// Single source of truth for the flame shell envelope dimensions, used by the resolve
// pass to derive the ray interval in closed form (clampToShellCone). The Rust mirror in
// thyllore-render-core/src/flame_shell.rs must stay identical: the render-pass scissor
// bound and CPU picking both derive from it.
//
// The envelope only bounds where the density field is sampled; it must stay wider than
// the field's own extent, otherwise the silhouette becomes the envelope instead of the
// density. The height taper lives in the density instead: it is concave, so no cone
// encloses it.

const float FLAME_SHELL_BASE_RADIUS = 0.5;
const float FLAME_SHELL_TAPER_TIP_SCALE = 1.0;
// Historic width factor (1/cos(pi/16)) kept so the analytic envelope matches the
// former circumscribed 16-gon proxy exactly; it is now just part of the envelope width.
const float FLAME_SHELL_CIRCUMSCRIBE = 1.0195911;
const float FLAME_SHELL_SUPPORT_HEADROOM = 1.5; // density support r̂max until envelope doesn't cut

// Multiplier on the base half-extent.
// supportScale is the emitter-dependent widening (flame_shell_support.glsl).
float flameShellRadiusScale(float height01, float supportScale) {
    return supportScale * FLAME_SHELL_SUPPORT_HEADROOM * FLAME_SHELL_CIRCUMSCRIBE
        * mix(1.0, FLAME_SHELL_TAPER_TIP_SCALE, height01);
}

// Outer radius of the shell in flame-local units.
float flameShellOuterRadius(float height01, float supportScale) {
    return FLAME_SHELL_BASE_RADIUS * flameShellRadiusScale(height01, supportScale);
}

#endif
