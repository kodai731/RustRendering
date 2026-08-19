use cgmath::{Deg, Euler, Quaternion, Rad};

use super::camera::CameraComponent;
use super::scalar_channel::{ScalarChannel, ScalarChannelDomain};
use crate::ecs::resource::TimelineState;
use crate::ecs::world::{Entity, Transform, World};
use thyllore_anim_core::editable::PropertyType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraParam {
    TranslationX,
    TranslationY,
    TranslationZ,
    RotationX,
    RotationY,
    RotationZ,
    FovY,
}

impl CameraParam {
    pub const ALL: [CameraParam; 7] = [
        CameraParam::TranslationX,
        CameraParam::TranslationY,
        CameraParam::TranslationZ,
        CameraParam::RotationX,
        CameraParam::RotationY,
        CameraParam::RotationZ,
        CameraParam::FovY,
    ];

    pub const fn code(self) -> u16 {
        match self {
            CameraParam::TranslationX => 256,
            CameraParam::TranslationY => 257,
            CameraParam::TranslationZ => 258,
            CameraParam::RotationX => 259,
            CameraParam::RotationY => 260,
            CameraParam::RotationZ => 261,
            CameraParam::FovY => 262,
        }
    }

    pub fn from_code(code: u16) -> Option<CameraParam> {
        CameraParam::ALL.iter().copied().find(|p| p.code() == code)
    }

    pub const fn property_type(self) -> PropertyType {
        PropertyType::Custom(self.code())
    }

    pub fn from_property_type(property_type: PropertyType) -> Option<CameraParam> {
        match property_type {
            PropertyType::Custom(code) => CameraParam::from_code(code),
            _ => None,
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            CameraParam::TranslationX => "Translation X",
            CameraParam::TranslationY => "Translation Y",
            CameraParam::TranslationZ => "Translation Z",
            CameraParam::RotationX => "Rotation X",
            CameraParam::RotationY => "Rotation Y",
            CameraParam::RotationZ => "Rotation Z",
            CameraParam::FovY => "FOV Y",
        }
    }

    pub const fn cli_name(self) -> &'static str {
        match self {
            CameraParam::TranslationX => "camera_translation_x",
            CameraParam::TranslationY => "camera_translation_y",
            CameraParam::TranslationZ => "camera_translation_z",
            CameraParam::RotationX => "camera_rotation_x",
            CameraParam::RotationY => "camera_rotation_y",
            CameraParam::RotationZ => "camera_rotation_z",
            CameraParam::FovY => "camera_fov_y",
        }
    }

    pub const fn scene_name(self) -> &'static str {
        match self {
            CameraParam::TranslationX => "CameraTranslationX",
            CameraParam::TranslationY => "CameraTranslationY",
            CameraParam::TranslationZ => "CameraTranslationZ",
            CameraParam::RotationX => "CameraRotationX",
            CameraParam::RotationY => "CameraRotationY",
            CameraParam::RotationZ => "CameraRotationZ",
            CameraParam::FovY => "CameraFovY",
        }
    }

    pub const fn debug_value_range(self) -> (f32, f32) {
        match self {
            CameraParam::TranslationX | CameraParam::TranslationY | CameraParam::TranslationZ => {
                (-5.0, 5.0)
            }
            CameraParam::RotationX | CameraParam::RotationY | CameraParam::RotationZ => {
                (-90.0, 90.0)
            }
            CameraParam::FovY => (20.0, 90.0),
        }
    }

    const fn channel(self) -> ScalarChannel {
        ScalarChannel {
            code: self.code(),
            display_name: self.display_name(),
            cli_name: self.cli_name(),
            scene_name: self.scene_name(),
            debug_value_range: self.debug_value_range(),
        }
    }
}

pub static CAMERA_CHANNELS: [ScalarChannel; 7] = {
    let mut channels = [CameraParam::TranslationX.channel(); 7];
    let mut i = 0;
    while i < 7 {
        channels[i] = CameraParam::ALL[i].channel();
        i += 1;
    }
    channels
};

pub static CAMERA_DOMAIN: ScalarChannelDomain = ScalarChannelDomain {
    name: "Camera",
    channels: &CAMERA_CHANNELS,
    has_component: camera_has_component,
    entities: camera_entities,
    read: camera_channel_read,
    local_time: camera_local_time,
};

fn camera_has_component(world: &World, entity: Entity) -> bool {
    world.get_component::<CameraComponent>(entity).is_some()
}

fn camera_entities(world: &World) -> Vec<Entity> {
    world
        .iter_components::<CameraComponent>()
        .map(|(e, _)| e)
        .collect()
}

fn camera_channel_read(world: &World, entity: Entity, property_type: PropertyType) -> Option<f32> {
    let param = CameraParam::from_property_type(property_type)?;
    let transform = world.get_component::<Transform>(entity)?;
    let camera = world.get_component::<CameraComponent>(entity)?;
    Some(camera_channel_value(&transform, &camera, param))
}

fn camera_local_time(world: &World, _entity: Entity) -> Option<f32> {
    world
        .get_resource::<TimelineState>()
        .map(|ts| ts.current_time)
}

fn camera_channel_value(
    transform: &Transform,
    camera: &CameraComponent,
    param: CameraParam,
) -> f32 {
    match param {
        CameraParam::TranslationX => transform.translation.x,
        CameraParam::TranslationY => transform.translation.y,
        CameraParam::TranslationZ => transform.translation.z,
        CameraParam::RotationX | CameraParam::RotationY | CameraParam::RotationZ => {
            let euler: Euler<Rad<f32>> = Euler::from(transform.rotation);
            match param {
                CameraParam::RotationX => Deg::from(euler.x).0,
                CameraParam::RotationY => Deg::from(euler.y).0,
                _ => Deg::from(euler.z).0,
            }
        }
        CameraParam::FovY => camera.fov_y.0,
    }
}

pub fn apply_camera_param_value(
    transform: &mut Transform,
    camera: &mut CameraComponent,
    param: CameraParam,
    value: f32,
) {
    match param {
        CameraParam::TranslationX => transform.translation.x = value,
        CameraParam::TranslationY => transform.translation.y = value,
        CameraParam::TranslationZ => transform.translation.z = value,
        CameraParam::RotationX | CameraParam::RotationY | CameraParam::RotationZ => {
            let euler: Euler<Rad<f32>> = Euler::from(transform.rotation);
            let (mut x, mut y, mut z) = (
                Deg::from(euler.x).0,
                Deg::from(euler.y).0,
                Deg::from(euler.z).0,
            );
            match param {
                CameraParam::RotationX => x = value,
                CameraParam::RotationY => y = value,
                _ => z = value,
            }
            transform.rotation = Quaternion::from(Euler::new(
                Rad::from(Deg(x)),
                Rad::from(Deg(y)),
                Rad::from(Deg(z)),
            ));
        }
        CameraParam::FovY => camera.fov_y = Deg(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_from_property_type_roundtrip_all() {
        for param in CameraParam::ALL {
            let pt = param.property_type();
            let recovered = CameraParam::from_property_type(pt).unwrap();
            assert_eq!(recovered, param);
        }
    }

    #[test]
    fn test_read_apply_roundtrip() {
        let mut transform = Transform::default();
        let mut camera = CameraComponent::default();

        // Apply translation (1, 2, 3)
        apply_camera_param_value(&mut transform, &mut camera, CameraParam::TranslationX, 1.0);
        apply_camera_param_value(&mut transform, &mut camera, CameraParam::TranslationY, 2.0);
        apply_camera_param_value(&mut transform, &mut camera, CameraParam::TranslationZ, 3.0);

        // Apply rotation Y = 30 degrees
        apply_camera_param_value(&mut transform, &mut camera, CameraParam::RotationY, 30.0);

        // Read back and verify
        assert!(
            (camera_channel_value(&transform, &camera, CameraParam::TranslationX) - 1.0).abs()
                < 1e-4
        );
        assert!(
            (camera_channel_value(&transform, &camera, CameraParam::TranslationY) - 2.0).abs()
                < 1e-4
        );
        assert!(
            (camera_channel_value(&transform, &camera, CameraParam::TranslationZ) - 3.0).abs()
                < 1e-4
        );
        assert!(
            (camera_channel_value(&transform, &camera, CameraParam::RotationY) - 30.0).abs() < 1e-3,
            "RotationY read back {}",
            camera_channel_value(&transform, &camera, CameraParam::RotationY)
        );
    }
}
