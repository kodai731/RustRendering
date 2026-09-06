use crate::ecs::systems::{FLAME_EFFECT_HOOK, WATER_EFFECT_HOOK};
use crate::hooks::effect::EffectHooks;

pub fn subscribe_effects(hooks: &mut EffectHooks) {
    hooks.register(WATER_EFFECT_HOOK);
    hooks.register(FLAME_EFFECT_HOOK);
}
