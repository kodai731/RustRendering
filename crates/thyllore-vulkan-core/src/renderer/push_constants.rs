#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GBufferPushConstants {
    pub object_id: u32,
    pub heatmap_mode: u32,
}

impl GBufferPushConstants {
    pub fn new(object_id: u32, heatmap_mode: u32) -> Self {
        Self {
            object_id,
            heatmap_mode,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OnionSkinPushConstants {
    pub ghost_tint_r: f32,
    pub ghost_tint_g: f32,
    pub ghost_tint_b: f32,
    pub ghost_opacity: f32,
    pub debug_mode: i32,
    pub _pad: [f32; 3],
}

impl OnionSkinPushConstants {
    pub fn new(tint_color: [f32; 3], opacity: f32) -> Self {
        Self {
            ghost_tint_r: tint_color[0],
            ghost_tint_g: tint_color[1],
            ghost_tint_b: tint_color[2],
            ghost_opacity: opacity,
            debug_mode: 0,
            _pad: [0.0; 3],
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbuffer_push_constants_size() {
        assert_eq!(std::mem::size_of::<GBufferPushConstants>(), 8);
    }

    #[test]
    fn test_onion_skin_push_constants_size() {
        assert_eq!(std::mem::size_of::<OnionSkinPushConstants>(), 32);
    }
}
