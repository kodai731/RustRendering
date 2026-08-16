use crate::flame::*;

/// Contour and radiative-transfer options of the erosion chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameContour {
    pub wiggle_amp: f32,
    /// 0 = world-Y anisotropy axis, 1 = advect direction axis.
    pub aniso_axis_advect: f32,
    /// RTE band count in mode 0: <= 1 legacy linear path, >= 2 per-band Beer-Lambert.
    pub rte_bands: f32,
    /// RTE absorption wavelength dispersion: 0 = grey body, 1 = Rayleigh 1/lambda.
    pub sigma_dispersion: f32,
}

impl Default for FlameContour {
    fn default() -> Self {
        Self {
            wiggle_amp: 0.3,
            aniso_axis_advect: 0.0,
            rte_bands: 4.0,
            sigma_dispersion: 1.0,
        }
    }
}

pub fn build_contour_params(contour: &FlameContour) -> FlameContourParams {
    FlameContourParams {
        wiggle_amp: contour.wiggle_amp,
        aniso_axis_advect: contour.aniso_axis_advect,
        rte_bands: contour.rte_bands,
        sigma_dispersion: contour.sigma_dispersion,
    }
}
