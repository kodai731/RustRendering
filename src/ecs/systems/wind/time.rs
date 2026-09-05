use crate::app::FrameContext;
use crate::ecs::component::{apply_wind_param_value, WindParam, WindTornadoEffect};
use crate::ecs::resource::WindRenderSettings;
use crate::ecs::systems::effect_time::{
    resolve_effect_time, EffectEntityInputs, EffectTimeSources,
};

pub fn wind_time_advance(ctx: &mut FrameContext) {
    let wind_entities = ctx.world.query_winds();
    let inputs = EffectEntityInputs::collect(ctx.world, &wind_entities);
    let (batch_fixed_time, free_run_when_paused) = ctx
        .world
        .get_resource::<WindRenderSettings>()
        .map(|settings| (settings.batch_fixed_time, settings.free_run_when_paused))
        .unwrap_or((None, WindRenderSettings::default().free_run_when_paused));
    let time_sources = EffectTimeSources::collect(
        ctx.world,
        ctx.delta_time,
        batch_fixed_time,
        free_run_when_paused,
    );

    for &entity in &wind_entities {
        let Some(mut effect) = ctx.world.get_component_mut::<WindTornadoEffect>(entity) else {
            continue;
        };
        resolve_effect_time(
            &mut effect.time,
            effect.time_scale,
            effect.time_offset,
            time_sources,
        );
        if let Some(transform) = inputs.transform_of(entity) {
            effect.position = transform.translation;
            effect.rotation = transform.rotation;
        }

        for (property_type, value) in inputs.sampled_scalars_of(entity, effect.time) {
            if let Some(param) = WindParam::from_property_type(property_type) {
                apply_wind_param_value(&mut effect, param, value);
            }
        }
    }
}
