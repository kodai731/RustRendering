use crate::app::FrameContext;
use crate::ecs::component::{apply_flame_param_value, FlameEffect, FlameParam};
use crate::ecs::resource::LightState;
use crate::ecs::systems::effect_time::{
    resolve_effect_time, EffectEntityInputs, EffectTimeSources,
};

pub fn flame_time_advance(ctx: &mut FrameContext) {
    let light_position = ctx
        .world
        .get_resource::<LightState>()
        .map(|ls| ls.light_position);
    let flame_entities = ctx.world.query_flames();
    let inputs = EffectEntityInputs::collect(ctx.world, &flame_entities);
    let time_sources = EffectTimeSources::collect(ctx.world, ctx.delta_time, None, false);

    for &entity in &flame_entities {
        let Some(mut effect) = ctx.world.get_component_mut::<FlameEffect>(entity) else {
            continue;
        };
        resolve_effect_time(
            &mut effect.time,
            effect.time_scale,
            effect.time_offset,
            time_sources,
        );
        if let Some(lp) = light_position {
            effect.light_position_world = lp;
        }
        if let Some(transform) = inputs.transform_of(entity) {
            effect.position = transform.translation;
            effect.rotation = transform.rotation;
        }

        for (property_type, value) in inputs.sampled_scalars_of(entity, effect.time) {
            if let Some(param) = FlameParam::from_property_type(property_type) {
                apply_flame_param_value(&mut effect, param, value);
            }
        }
    }
}
