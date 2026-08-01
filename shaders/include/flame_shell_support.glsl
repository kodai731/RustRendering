#ifndef FLAME_SHELL_SUPPORT_GLSL
#define FLAME_SHELL_SUPPORT_GLSL

// Must be included after the FlameUBO declaration and flame_shell_profile.glsl.
// Emitter-dependent widening of the shell proxy: a ring's tube (centerline at
// normalized major radius rm, minor support 1.5 * (1 - rm)) reaches past the
// cylinder support 0.75, and a proxy that stops there slices the torus flat.
// Mirrored in thyllore-render-core/src/flame_shell.rs (flame_shell_support_scale).
float flameShellSupportScale() {
    if (flame.emitterParams.x >= 0.5 && flame.emitterParams.x < 1.5) {
        float rm = flame.emitterParams.y;
        return max(
            (rm + 1.5 * (1.0 - rm)) / (FLAME_SHELL_BASE_RADIUS * FLAME_SHELL_SUPPORT_HEADROOM),
            1.0);
    }
    return 1.0;
}

#endif
