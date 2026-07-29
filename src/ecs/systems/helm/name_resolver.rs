//! Resolves the object name a user spoke to a concrete `Entity`.
//!
//! The helm never sees `Entity` ids — it cannot know them — so every
//! object argument arrives as a name and is resolved here by deterministic
//! logic. Ambiguity is reported rather than guessed: acting on the wrong object
//! is worse than asking.

use crate::ecs::world::{Entity, Name, World};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NameResolution {
    Resolved(Entity),
    NotFound,
    Ambiguous(Vec<Entity>),
}

pub fn resolve_entity_by_name(world: &World, requested: &str) -> NameResolution {
    let query = requested.trim();
    if query.is_empty() {
        return NameResolution::NotFound;
    }

    let named: Vec<(Entity, String)> = world
        .iter_components::<Name>()
        .map(|(entity, name)| (entity, name.0.clone()))
        .collect();

    let exact = collect_matches(&named, |name| name == query);
    if !exact.is_empty() {
        return classify(exact);
    }

    let lowered = query.to_lowercase();
    let case_insensitive = collect_matches(&named, |name| name.to_lowercase() == lowered);
    if !case_insensitive.is_empty() {
        return classify(case_insensitive);
    }

    classify(collect_matches(&named, |name| {
        name.to_lowercase().contains(&lowered)
    }))
}

fn collect_matches(named: &[(Entity, String)], predicate: impl Fn(&str) -> bool) -> Vec<Entity> {
    let mut matches: Vec<Entity> = named
        .iter()
        .filter(|(_, name)| predicate(name))
        .map(|(entity, _)| *entity)
        .collect();
    matches.sort_unstable();
    matches
}

fn classify(matches: Vec<Entity>) -> NameResolution {
    match matches.len() {
        0 => NameResolution::NotFound,
        1 => NameResolution::Resolved(matches[0]),
        _ => NameResolution::Ambiguous(matches),
    }
}

pub fn read_entity_name(world: &World, entity: Entity) -> Option<String> {
    world
        .get_component::<Name>(entity)
        .map(|name| name.0.clone())
}

pub fn list_entity_names(world: &World) -> Vec<String> {
    let mut names: Vec<String> = world
        .iter_components::<Name>()
        .map(|(_, name)| name.0.clone())
        .collect();
    names.sort();
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spawn_named(world: &mut World, name: &str) -> Entity {
        let entity = world.spawn();
        world.insert_component(entity, Name(name.to_string()));
        entity
    }

    #[test]
    fn resolves_an_exact_name() {
        let mut world = World::new();
        spawn_named(&mut world, "Camera01");
        let hero = spawn_named(&mut world, "Hero");

        assert_eq!(
            resolve_entity_by_name(&world, "Hero"),
            NameResolution::Resolved(hero)
        );
    }

    #[test]
    fn resolves_ignoring_case_when_no_exact_match_exists() {
        let mut world = World::new();
        let hero = spawn_named(&mut world, "Hero");

        assert_eq!(
            resolve_entity_by_name(&world, "hero"),
            NameResolution::Resolved(hero)
        );
    }

    #[test]
    fn an_exact_match_wins_over_a_case_insensitive_one() {
        let mut world = World::new();
        let lower = spawn_named(&mut world, "hero");
        spawn_named(&mut world, "Hero");

        assert_eq!(
            resolve_entity_by_name(&world, "hero"),
            NameResolution::Resolved(lower)
        );
    }

    #[test]
    fn resolves_a_unique_substring() {
        let mut world = World::new();
        let floor = spawn_named(&mut world, "Floor_Ground");
        spawn_named(&mut world, "Hero");

        assert_eq!(
            resolve_entity_by_name(&world, "floor"),
            NameResolution::Resolved(floor)
        );
    }

    #[test]
    fn an_exact_match_wins_over_a_substring_match() {
        let mut world = World::new();
        spawn_named(&mut world, "Cube_Large");
        let cube = spawn_named(&mut world, "Cube");

        assert_eq!(
            resolve_entity_by_name(&world, "Cube"),
            NameResolution::Resolved(cube)
        );
    }

    #[test]
    fn reports_ambiguity_instead_of_guessing() {
        let mut world = World::new();
        let first = spawn_named(&mut world, "Light01");
        let second = spawn_named(&mut world, "Light02");

        assert_eq!(
            resolve_entity_by_name(&world, "light"),
            NameResolution::Ambiguous(vec![first, second])
        );
    }

    #[test]
    fn reports_ambiguity_for_duplicate_exact_names() {
        let mut world = World::new();
        let first = spawn_named(&mut world, "Cube");
        let second = spawn_named(&mut world, "Cube");

        assert_eq!(
            resolve_entity_by_name(&world, "Cube"),
            NameResolution::Ambiguous(vec![first, second])
        );
    }

    #[test]
    fn reports_not_found_for_an_unknown_name() {
        let mut world = World::new();
        spawn_named(&mut world, "Hero");

        assert_eq!(
            resolve_entity_by_name(&world, "Dragon"),
            NameResolution::NotFound
        );
    }

    #[test]
    fn reports_not_found_for_a_blank_name() {
        let mut world = World::new();
        spawn_named(&mut world, "Hero");

        assert_eq!(
            resolve_entity_by_name(&world, "   "),
            NameResolution::NotFound
        );
    }

    #[test]
    fn reports_not_found_in_an_empty_scene() {
        let world = World::new();
        assert_eq!(
            resolve_entity_by_name(&world, "Hero"),
            NameResolution::NotFound
        );
    }

    #[test]
    fn surrounding_whitespace_does_not_affect_resolution() {
        let mut world = World::new();
        let hero = spawn_named(&mut world, "Hero");

        assert_eq!(
            resolve_entity_by_name(&world, "  Hero  "),
            NameResolution::Resolved(hero)
        );
    }

    #[test]
    fn lists_every_name_in_sorted_order() {
        let mut world = World::new();
        spawn_named(&mut world, "Hero");
        spawn_named(&mut world, "Camera01");

        assert_eq!(list_entity_names(&world), vec!["Camera01", "Hero"]);
    }
}
