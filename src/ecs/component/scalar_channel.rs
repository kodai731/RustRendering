use thyllore_anim_core::editable::PropertyType;

use crate::ecs::world::{Entity, World};

/// One animatable scalar channel exposed by a component domain. `code` is the
/// stable `PropertyType::Custom` payload persisted in clip files — never
/// reorder or reuse codes.
#[derive(Clone, Copy, Debug)]
pub struct ScalarChannel {
    pub code: u16,
    pub display_name: &'static str,
    /// Stable snake_case identifier used by batch CLI flags and anim dumps.
    /// Unique across every registered domain.
    pub cli_name: &'static str,
    /// Stable PascalCase identifier persisted in scene files.
    pub scene_name: &'static str,
    /// Conservative value range for generated debug keys.
    pub debug_value_range: (f32, f32),
}

impl ScalarChannel {
    pub const fn property_type(&self) -> PropertyType {
        PropertyType::Custom(self.code)
    }
}

/// A component domain whose scalar fields animate through clip scalar curves.
/// Registering a domain in `scalar_channel_domains` is all the curve editor,
/// timeline, batch CLI and scene serialization need to animate its channels;
/// flame is one such registration. Applying sampled curve values back to the
/// component stays inside the domain's own system.
///
/// Each domain owns a disjoint block of `Custom` codes: flame uses 0..=15,
/// the next domain should start at 256.
pub struct ScalarChannelDomain {
    /// Display name of the domain (also the name of the clip it creates).
    pub name: &'static str,
    pub channels: &'static [ScalarChannel],
    pub has_component: fn(&World, Entity) -> bool,
    pub entities: fn(&World) -> Vec<Entity>,
    /// Current component value of a channel (None when the entity lost the
    /// component or the property belongs to another domain).
    pub read: fn(&World, Entity, PropertyType) -> Option<f32>,
    /// Domain-local playback time used to sample curves.
    pub local_time: fn(&World, Entity) -> Option<f32>,
}

static SCALAR_CHANNEL_DOMAINS: [&ScalarChannelDomain; 2] = [
    &super::flame_param::FLAME_DOMAIN,
    &super::water_param::WATER_DOMAIN,
];

pub fn scalar_channel_domains() -> &'static [&'static ScalarChannelDomain] {
    &SCALAR_CHANNEL_DOMAINS
}

pub fn scalar_domain_for_entity(
    world: &World,
    entity: Entity,
) -> Option<&'static ScalarChannelDomain> {
    scalar_channel_domains()
        .iter()
        .copied()
        .find(|domain| (domain.has_component)(world, entity))
}

pub fn scalar_channel_for_property(
    property_type: PropertyType,
) -> Option<(&'static ScalarChannelDomain, &'static ScalarChannel)> {
    scalar_channel_domains().iter().find_map(|domain| {
        domain
            .channels
            .iter()
            .find(|c| c.property_type() == property_type)
            .map(|c| (*domain, c))
    })
}

pub fn scalar_channel_for_cli_name(
    name: &str,
) -> Option<(&'static ScalarChannelDomain, &'static ScalarChannel)> {
    scalar_channel_domains().iter().find_map(|domain| {
        domain
            .channels
            .iter()
            .find(|c| c.cli_name == name)
            .map(|c| (*domain, c))
    })
}

pub fn scalar_channel_for_scene_name(
    name: &str,
) -> Option<(&'static ScalarChannelDomain, &'static ScalarChannel)> {
    scalar_channel_domains().iter().find_map(|domain| {
        domain
            .channels
            .iter()
            .find(|c| c.scene_name == name)
            .map(|c| (*domain, c))
    })
}

pub fn scalar_cli_names_joined() -> String {
    scalar_channel_domains()
        .iter()
        .flat_map(|domain| domain.channels.iter().map(|c| c.cli_name))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_codes_and_names_are_unique_across_domains() {
        let mut codes = HashSet::new();
        let mut cli_names = HashSet::new();
        let mut scene_names = HashSet::new();
        for domain in scalar_channel_domains() {
            for channel in domain.channels {
                assert!(
                    codes.insert(channel.code),
                    "duplicate code {}",
                    channel.code
                );
                assert!(
                    cli_names.insert(channel.cli_name),
                    "duplicate cli name {}",
                    channel.cli_name
                );
                assert!(
                    scene_names.insert(channel.scene_name),
                    "duplicate scene name {}",
                    channel.scene_name
                );
            }
        }
        assert!(!codes.is_empty());
    }

    #[test]
    fn test_lookups_roundtrip_every_channel() {
        for domain in scalar_channel_domains() {
            for channel in domain.channels {
                let (d, c) = scalar_channel_for_property(channel.property_type()).unwrap();
                assert_eq!(d.name, domain.name);
                assert_eq!(c.code, channel.code);
                let (_, c) = scalar_channel_for_cli_name(channel.cli_name).unwrap();
                assert_eq!(c.code, channel.code);
                let (_, c) = scalar_channel_for_scene_name(channel.scene_name).unwrap();
                assert_eq!(c.code, channel.code);
            }
        }
        assert!(scalar_channel_for_cli_name("no_such_channel").is_none());
        assert!(scalar_channel_for_property(PropertyType::TranslationX).is_none());
    }
}
