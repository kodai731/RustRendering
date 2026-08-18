/// World-space ray through the clicked pixel, kept so picking can depth-order candidates that
/// never reach the object-id buffer.
#[derive(Clone, Copy, Debug)]
pub struct PickRay {
    pub origin: cgmath::Vector3<f32>,
    pub direction: cgmath::Vector3<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct ObjectIdReadback {
    pub pending_pixel: Option<(u32, u32)>,
    pub pick_ray: Option<PickRay>,
    pub copy_in_flight: bool,
    pub last_read_object_id: Option<u32>,
    /// G-buffer world position of the picked pixel, valid only when the object id is non-zero.
    pub last_read_world_position: Option<[f32; 3]>,
    pub is_shift: bool,
    pub is_ctrl: bool,
}
