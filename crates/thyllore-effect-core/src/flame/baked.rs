/// Texture-fit output attached to a flame entity. Kept apart from
/// `FlameEffect` so the authoring parameters stay a pure closed-form
/// parameter block; the LUT payload here is a registered closed-form-guard
/// exception until the fit emits coefficients directly.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlameBaked {
    pub envelope: Option<[f32; 33]>,
    pub radius: Option<[f32; 33]>,
    pub color: Option<[[f32; 3]; 8]>,
    pub blend: f32,
}

impl FlameBaked {
    pub fn is_active(&self) -> bool {
        self.blend > 0.0
            && (self.envelope.is_some() || self.radius.is_some() || self.color.is_some())
    }
}
