use crate::ecs::component::WaterTorusEffect;
use crate::ecs::resource::PickRay;
use crate::ecs::world::{Entity, World};
use cgmath::SquareMatrix;
use thyllore_effect_core::{build_water_model_matrix, pick_torus};
/// Nearest water whose torus the ray enters, with the distance at which it enters.
pub fn find_water_by_pick_ray(world: &World, ray: &PickRay) -> Option<(Entity, f32)> {
    world
        .query_waters()
        .into_iter()
        .filter_map(|entity| {
            let effect = world.get_component::<WaterTorusEffect>(entity)?;
            let model = build_water_model_matrix(&effect);
            let inverse = model.invert()?;
            let distance = pick_torus(
                ray.origin,
                ray.direction,
                model,
                inverse,
                effect.major_radius,
                effect.minor_radius,
            )?;
            Some((entity, distance))
        })
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
}
