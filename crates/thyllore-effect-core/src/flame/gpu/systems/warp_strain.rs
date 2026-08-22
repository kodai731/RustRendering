use crate::flame::*;

/// Form-matched strain norm: max a|k| for the sequential composition
/// (per-shear bound), RMS for the displacement sum (gradient-sum bound).
fn shear_strain_norm(modes: &[crate::flame_wave::WaveWarpMode]) -> f32 {
    if read_env_warp_form_displacement() {
        crate::flame_wave::warp_strain_norm_rms(modes)
    } else {
        crate::flame_wave::warp_strain_norm(modes)
    }
}

/// The shear table the warp map actually evaluates: cf layers when the
/// closed-form variant is active, the 16 warp modes otherwise.
fn active_shear_table(
    warp_modes: &[crate::flame_wave::WaveWarpMode],
) -> Vec<crate::flame_wave::WaveWarpMode> {
    if read_env_wave_cf() {
        crate::flame_wave::generate_wave_cf_shear_layers(
            warp_modes,
            read_env_wave_cf_layers(),
            read_env_wave_cf_shear(),
        )
        .to_vec()
    } else {
        warp_modes.to_vec()
    }
}

/// Medium swirl modes with amplitudes expressing swirl_gain as a share of the
/// active shear table's strain norm (motion_design L2). Public so the debug
/// harnesses replay the exact packed table.
pub fn build_medium_swirl_modes(
    effect: &FlameEffect,
    warp_modes: &[crate::flame_wave::WaveWarpMode],
) -> [crate::flame_wave::WaveWarpMode; crate::flame_wave::WAVE_MEDIUM_SWIRL_MODE_COUNT] {
    let base_norm = shear_strain_norm(&active_shear_table(warp_modes));
    crate::flame_wave::generate_medium_swirl_modes(
        read_env_swirl_gain(effect.swirl.gain),
        base_norm,
    )
}

/// Asymptotic warp-strain profile. The strain norm is taken over the combined
/// table the warp map evaluates — the active shear table plus the medium swirl
/// modes — so the swirl joins the fixed strain budget: raising the swirl gain
/// thins the carve warp instead of exceeding the cap.
pub fn build_warp_strain_params(effect: &FlameEffect) -> FlameWarpStrainParams {
    let warp_modes = crate::flame_wave::generate_wave_warp_modes();
    let mut table = active_shear_table(&warp_modes);
    table.extend_from_slice(&build_medium_swirl_modes(effect, &warp_modes));
    let [strain_base, strain_tip, inv_reach, inv_strain_norm] =
        crate::flame_wave::warp_strain_params(
            effect.warp.amp,
            effect.warp.reach,
            shear_strain_norm(&table),
        );
    FlameWarpStrainParams {
        strain_base,
        strain_tip,
        inv_reach,
        inv_strain_norm,
    }
}

pub(super) fn build_warp_form_params(effect: &FlameEffect) -> FlameWarpFormParams {
    FlameWarpFormParams {
        displacement_form: if read_env_warp_form_displacement() {
            1.0
        } else {
            0.0
        },
        burnout_gain: effect.carve.burnout_gain,
        _padding: [0.0; 2],
    }
}
