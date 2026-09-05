pub fn flame_group_param_names(group: &str) -> Vec<&'static str> {
    thyllore_effect_core::FLAME_UI_PARAMS
        .iter()
        .filter(|p| p.group == group)
        .map(|p| p.name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_effect_core::{FLAME_SCALAR_PARAMS, FLAME_UI_PARAMS};
    use thyllore_scene_core::{find_scalar_param, find_ui_param};

    const GROUPS: [&str; 6] = ["body", "noise", "mix", "motion", "branch", "footer"];

    #[test]
    fn test_every_group_is_non_empty_and_resolves_to_ui_and_scalar_param() {
        for group in GROUPS {
            let names = flame_group_param_names(group);
            assert!(!names.is_empty(), "{group}");

            for name in names {
                assert!(find_ui_param(FLAME_UI_PARAMS, name).is_some(), "{name}");
                assert!(
                    find_scalar_param(FLAME_SCALAR_PARAMS, name).is_some(),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn test_grouped_names_are_unique_and_cover_all_ui_params() {
        let mut names: Vec<&str> = GROUPS
            .iter()
            .flat_map(|group| flame_group_param_names(group))
            .collect();
        let collected = names.len();

        names.sort_unstable();
        names.dedup();

        assert_eq!(names.len(), collected);
        assert_eq!(names.len(), 35);
    }
}
