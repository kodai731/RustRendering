#ifndef FLAME_SHELL_PROFILE_GLSL
#define FLAME_SHELL_PROFILE_GLSL

// Single source of truth for the flame shell proxy dimensions, shared by the shell
// vertex/geometry shaders that rasterize it and the resolve pass that clamps rays to it.
// The Rust mirror in thyllore-render-core/src/flame_shell.rs must stay identical:
// the winding tests and the render-pass scissor bound both derive from it.
//
// The proxy only bounds where the density field is sampled; it must stay wider than the
// field's own extent, otherwise the silhouette becomes the proxy instead of the density.
// The height taper lives in the density instead: it is concave, so no cone encloses it.

const float FLAME_SHELL_BASE_RADIUS = 0.5;
const float FLAME_SHELL_TAPER_TIP_SCALE = 1.0;
const float FLAME_SHELL_CIRCUMSCRIBE = 1.0195911; // 1/cos(pi/16): circumscribe 16-gon over unit cylinder
const float FLAME_SHELL_SUPPORT_HEADROOM = 1.5; // density support r̂max until proxy doesn't cut

// Multiplier on the base half-extent. Includes the circumscribe factor, so a cone built
// from it encloses the rasterized octagon rather than cutting its corners.
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
