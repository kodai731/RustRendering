use crate::ecs::component::WindTornadoEffect;
use crate::ecs::resource::PickRay;
use crate::ecs::world::{Entity, World};
use cgmath::SquareMatrix;
use thyllore_effect_core::{build_wind_model_matrix, pick_wind, WindShellParams};

/// Nearest wind whose envelope the ray enters, with the distance at which it enters.
pub fn find_wind_by_pick_ray(world: &World, ray: &PickRay) -> Option<(Entity, f32)> {
    world
        .query_winds()
        .into_iter()
        .filter_map(|entity| {
            let effect = world.get_component::<WindTornadoEffect>(entity)?;
            let inverse = build_wind_model_matrix(&effect).invert()?;
            let params = WindShellParams::from_effect(&effect);
            let distance = pick_wind(ray.origin, ray.direction, inverse, &params)?;
            Some((entity, distance))
        })
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
}
