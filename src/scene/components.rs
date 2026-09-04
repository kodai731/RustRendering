use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::format::{
    apply_flame_state_to_world, apply_water_state_to_world, build_flame_scene_data,
    build_water_scene_data, FlameSceneData, WaterSceneData,
};
use crate::asset::AssetStorage;
use crate::ecs::world::World;

/// A named entity whose effect state is stored as component values keyed by type key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEntity {
    pub name: String,
    #[serde(default)]
    pub components: BTreeMap<String, serde_json::Value>,
}

pub struct SceneComponentEntry {
    pub type_key: &'static str,
    pub capture: fn(&World) -> Option<serde_json::Value>,
    pub apply: fn(&mut World, &mut AssetStorage, &serde_json::Value) -> anyhow::Result<()>,
}

pub fn scene_component_registry() -> &'static [SceneComponentEntry] {
    &[
        SceneComponentEntry {
            type_key: "flame",
            capture: capture_flame_component,
            apply: apply_flame_component,
        },
        SceneComponentEntry {
            type_key: "water_torus",
            capture: capture_water_component,
            apply: apply_water_component,
        },
    ]
}

fn encode_component<T: Serialize>(type_key: &str, data: T) -> Option<serde_json::Value> {
    match serde_json::to_value(data) {
        Ok(value) => Some(value),
        Err(error) => {
            log_warn!("Failed to encode scene component {}: {}", type_key, error);
            None
        }
    }
}

fn capture_flame_component(world: &World) -> Option<serde_json::Value> {
    encode_component("flame", build_flame_scene_data(world)?)
}

fn apply_flame_component(
    world: &mut World,
    assets: &mut AssetStorage,
    value: &serde_json::Value,
) -> anyhow::Result<()> {
    let flame: FlameSceneData = serde_json::from_value(value.clone())?;
    apply_flame_state_to_world(world, assets, &flame);
    Ok(())
}

fn capture_water_component(world: &World) -> Option<serde_json::Value> {
    encode_component("water_torus", build_water_scene_data(world)?)
}

fn apply_water_component(
    world: &mut World,
    assets: &mut AssetStorage,
    value: &serde_json::Value,
) -> anyhow::Result<()> {
    let water: WaterSceneData = serde_json::from_value(value.clone())?;
    apply_water_state_to_world(world, assets, &water);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::format::SceneFile;

    #[test]
    fn scene_file_entities_survive_a_ron_round_trip() {
        let flame = FlameSceneData {
            effect: thyllore_effect_core::FlameEffect::default(),
            channels: Vec::new(),
            clip_min_duration: 0.0,
            motion_path: None,
            style: None,
        };
        let mut components = BTreeMap::new();
        components.insert(
            "flame".to_string(),
            serde_json::to_value(&flame).expect("flame scene data encodes to json"),
        );

        let mut scene = SceneFile::new("round_trip", "models/mesh.glb");
        scene.entities.push(SceneEntity {
            name: "effects".to_string(),
            components: components.clone(),
        });

        let config = ron::ser::PrettyConfig::new()
            .depth_limit(8)
            .separate_tuple_members(true)
            .enumerate_arrays(false);
        let serialized =
            ron::ser::to_string_pretty(&scene, config).expect("scene file serializes to ron");
        let restored: SceneFile = ron::from_str(&serialized).expect("scene file parses back");

        assert_eq!(restored.entities.len(), 1);
        assert_eq!(restored.entities[0].name, "effects");
        assert_eq!(restored.entities[0].components, components);
    }
}
