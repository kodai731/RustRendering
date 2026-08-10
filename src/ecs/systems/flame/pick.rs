use crate::ecs::component::FlameEffect;
use crate::ecs::resource::PickRay;
use crate::ecs::world::{Entity, World};
use thyllore_render_core::{build_flame_inverse_model_matrix, intersect_flame_proxy};

/// Nearest flame whose proxy the ray enters, with the distance at which it enters.
///
/// Flames are not drawn into the object-id buffer — they have no geometry — so a click has to
/// test them separately and then order the two candidates by distance.
pub fn find_flame_by_pick_ray(world: &World, ray: &PickRay) -> Option<(Entity, f32)> {
    world
        .query_flames()
        .into_iter()
        .filter_map(|entity| {
            let effect = world.get_component::<FlameEffect>(entity)?;
            let inverse_model = build_flame_inverse_model_matrix(&effect);
            let distance =
                intersect_flame_proxy(&effect, &inverse_model, ray.origin, ray.direction)?;
            Some((entity, distance))
        })
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
}
