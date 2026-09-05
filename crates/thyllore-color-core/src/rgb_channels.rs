pub const RGB_CHANNEL_SUFFIXES: [&str; 3] = ["_r", "_g", "_b"];

pub trait RgbField<Owner> {
    const GET: fn(&Owner) -> [f32; 3];
    const SET: fn(&mut Owner, [f32; 3]);
}

pub fn get_rgb_channel<Owner, Field: RgbField<Owner>, const CHANNEL: usize>(owner: &Owner) -> f32 {
    (Field::GET)(owner)[CHANNEL]
}

pub fn set_rgb_channel<Owner, Field: RgbField<Owner>, const CHANNEL: usize>(
    owner: &mut Owner,
    value: f32,
) {
    let mut rgb = (Field::GET)(owner);
    rgb[CHANNEL] = value;
    (Field::SET)(owner, rgb);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Paint {
        tint: [f32; 3],
    }

    struct Tint;

    impl RgbField<Paint> for Tint {
        const GET: fn(&Paint) -> [f32; 3] = |paint| paint.tint;
        const SET: fn(&mut Paint, [f32; 3]) = |paint, value| paint.tint = value;
    }

    #[test]
    fn test_channel_accessors_touch_only_their_channel() {
        let mut paint = Paint {
            tint: [0.1, 0.2, 0.3],
        };
        set_rgb_channel::<Paint, Tint, 1>(&mut paint, 0.9);
        assert_eq!(paint.tint, [0.1, 0.9, 0.3]);
        assert_eq!(get_rgb_channel::<Paint, Tint, 2>(&paint), 0.3);
    }
}
