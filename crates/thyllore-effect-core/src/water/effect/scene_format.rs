use crate::water::ownership::WaterParameterOwner;
use crate::water::*;
use cgmath::{Quaternion, Vector3};
use thyllore_scene_core::declare_scene_format;

declare_scene_format! {
    component: WaterTorusEffect,
    record: WaterSceneRecord,
    tag: WaterParameterOwner,
    items {
        tags: WATER_PARAMETER_OWNERSHIP,
        snapshot: water_parameter_snapshot,
        scalars: WATER_SCALAR_PARAMS,
        ui: WATER_UI_PARAMS,
        overwrite: overwrite_water_persisted_fields,
    },
    persisted {
        position: [f32; 3] = Frame {
            get: |e| [e.position.x, e.position.y, e.position.z],
            set: |e, v| e.position = Vector3::new(v[0], v[1], v[2]),
        },
        rotation: [f32; 4] = Frame {
            get: |e| [e.rotation.s, e.rotation.v.x, e.rotation.v.y, e.rotation.v.z],
            set: |e, v| e.rotation = Quaternion::new(v[0], v[1], v[2], v[3]),
        },
        major_radius: f32 = Frame {
            get: |e| e.major_radius,
            set: |e, v| e.major_radius = v,
            ui {
                min: 0.01,
                max: 10.0,
                format: "%.2f",
                group: "shape",
            },
        },
        minor_radius: f32 = Frame {
            get: |e| e.minor_radius,
            set: |e, v| e.minor_radius = v,
            ui {
                min: 0.01,
                max: 5.0,
                format: "%.2f",
                group: "shape",
            },
        },
        ior: f32 = Frame {
            get: |e| e.ior,
            set: |e, v| e.ior = v,
            ui {
                min: 1.0,
                max: 2.5,
                format: "%.3f",
                group: "optics",
            },
        },
        absorption_r: f32 = Frame {
            get: |e| e.absorption[0],
            set: |e, v| e.absorption[0] = v,
            ui {
                min: 0.0,
                max: 10.0,
                format: "%.2f",
                group: "optics",
            },
        },
        absorption_g: f32 = Frame {
            get: |e| e.absorption[1],
            set: |e, v| e.absorption[1] = v,
            ui {
                min: 0.0,
                max: 10.0,
                format: "%.2f",
                group: "optics",
            },
        },
        absorption_b: f32 = Frame {
            get: |e| e.absorption[2],
            set: |e, v| e.absorption[2] = v,
            ui {
                min: 0.0,
                max: 10.0,
                format: "%.2f",
                group: "optics",
            },
        },
        flow_longitudinal: f32 = Frame {
            get: |e| e.flow_longitudinal,
            set: |e, v| e.flow_longitudinal = v,
            ui {
                min: -5.0,
                max: 5.0,
                format: "%.2f",
                group: "flow",
            },
        },
        flow_meridional: f32 = Frame {
            get: |e| e.flow_meridional,
            set: |e, v| e.flow_meridional = v,
            ui {
                min: -5.0,
                max: 5.0,
                format: "%.2f",
                group: "flow",
            },
        },
        wave_amplitude: f32 = Frame {
            get: |e| e.wave_amplitude,
            set: |e, v| e.wave_amplitude = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.3f",
                group: "wave",
            },
        },
        wave_frequency: f32 = Frame {
            get: |e| e.wave_frequency,
            set: |e, v| e.wave_frequency = v,
            ui {
                min: 0.0,
                max: 50.0,
                format: "%.1f",
                group: "wave",
            },
        },
        wave_speed: f32 = Frame {
            get: |e| e.wave_speed,
            set: |e, v| e.wave_speed = v,
            ui {
                min: 0.0,
                max: 10.0,
                format: "%.2f",
                group: "wave",
            },
        },
        wave_dispersion: f32 = Frame {
            get: |e| e.wave_dispersion,
            set: |e, v| e.wave_dispersion = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                group: "wave",
            },
        },
        wave_lb_blend: f32 = Frame {
            get: |e| e.wave_lb_blend,
            set: |e, v| e.wave_lb_blend = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                group: "wave",
            },
        },
        light_intensity: f32 = Frame {
            get: |e| e.light_intensity,
            set: |e, v| e.light_intensity = v,
            ui {
                min: 0.0,
                max: 20.0,
                format: "%.2f",
                group: "lighting",
            },
        },
        highlight_sharpness: f32 = Frame {
            get: |e| e.highlight_sharpness,
            set: |e, v| e.highlight_sharpness = v,
            ui {
                min: 1.0,
                max: 1024.0,
                format: "%.0f",
                group: "lighting",
            },
        },
        sky_brightness: f32 = Frame {
            get: |e| e.sky_brightness,
            set: |e, v| e.sky_brightness = v,
            ui {
                min: 0.0,
                max: 2.0,
                format: "%.2f",
                group: "lighting",
            },
        },
        scatter_strength: f32 = Frame {
            get: |e| e.scatter_strength,
            set: |e, v| e.scatter_strength = v,
            ui {
                min: 0.0,
                max: 10.0,
                format: "%.2f",
                group: "lighting",
            },
        },
        scatter_anisotropy: f32 = Frame {
            get: |e| e.scatter_anisotropy,
            set: |e, v| e.scatter_anisotropy = v,
            ui {
                min: -0.9,
                max: 0.9,
                format: "%.2f",
                group: "lighting",
            },
        },
        reflect_strength: f32 = Frame {
            get: |e| e.reflect_strength,
            set: |e, v| e.reflect_strength = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                group: "look",
            },
        },
        refract_strength: f32 = Frame {
            get: |e| e.refract_strength,
            set: |e, v| e.refract_strength = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                group: "look",
            },
        },
        caustic_strength: f32 = Frame {
            get: |e| e.caustic_strength,
            set: |e, v| e.caustic_strength = v,
            ui {
                min: 0.0,
                max: 2.0,
                format: "%.2f",
                group: "look",
            },
        },
        tint_r: f32 = Frame {
            get: |e| e.tint[0],
            set: |e, v| e.tint[0] = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                group: "look",
            },
        },
        tint_g: f32 = Frame {
            get: |e| e.tint[1],
            set: |e, v| e.tint[1] = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                group: "look",
            },
        },
        tint_b: f32 = Frame {
            get: |e| e.tint[2],
            set: |e, v| e.tint[2] = v,
            ui {
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                group: "look",
            },
        },
    },
    runtime {
        time: f32 {
            get: |e| e.time,
            set: |e, v| e.time = v,
            ui {
                min: 0.0,
                max: 100.0,
                format: "%.2f",
            },
        },
        time_scale: f32 {
            get: |e| e.time_scale,
            set: |e, v| e.time_scale = v,
            ui {
                min: 0.0,
                max: 4.0,
                format: "%.2f",
            },
        },
        time_offset: f32 {
            get: |e| e.time_offset,
            set: |e, v| e.time_offset = v,
            ui {
                min: -100.0,
                max: 100.0,
                format: "%.2f",
            },
        },
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_scene_core::find_scalar_param;

    #[test]
    fn test_ron_struct_syntax_roundtrip() {
        let mut effect = WaterTorusEffect::default();
        effect.major_radius = 2.5;
        effect.reflect_strength = 0.8;

        let text = ron::ser::to_string_pretty(&effect, ron::ser::PrettyConfig::new())
            .expect("ron serialize");
        let restored: WaterTorusEffect = ron::from_str(&text).expect("ron deserialize");
        assert_eq!(restored.major_radius, 2.5);
        assert_eq!(restored.reflect_strength, 0.8);
    }

    #[test]
    fn test_scalar_param_names_are_unique() {
        let mut names: Vec<&str> = WATER_SCALAR_PARAMS.iter().map(|p| p.name).collect();
        names.sort_unstable();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
    }

    #[test]
    fn test_scalar_param_set_then_get_reaches_a_fixpoint() {
        for (i, param) in WATER_SCALAR_PARAMS.iter().enumerate() {
            let mut effect = WaterTorusEffect::default();
            (param.set)(&mut effect, 3.0 + i as f32);
            let first = (param.get)(&effect);
            (param.set)(&mut effect, first);
            assert_eq!((param.get)(&effect), first, "{}", param.name);
        }
    }

    #[test]
    fn test_every_ui_param_has_a_scalar_accessor_and_unique_name() {
        let mut names: Vec<&str> = WATER_UI_PARAMS.iter().map(|p| p.name).collect();
        names.sort_unstable();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
        for param in WATER_UI_PARAMS {
            assert!(
                find_scalar_param(WATER_SCALAR_PARAMS, param.name).is_some(),
                "{}",
                param.name
            );
            assert!(param.min < param.max, "{}", param.name);
        }
    }

    #[test]
    fn test_ui_param_groups_cover_every_water_group_in_display_order() {
        let mut groups: Vec<&str> = Vec::new();
        for param in WATER_UI_PARAMS {
            if !param.group.is_empty() && !groups.contains(&param.group) {
                groups.push(param.group);
            }
        }
        assert_eq!(
            groups,
            ["shape", "optics", "flow", "wave", "lighting", "look"]
        );
    }

    #[test]
    fn test_ui_param_group_members_match_their_group() {
        let expected: &[(&str, &[&str])] = &[
            ("shape", &["major_radius", "minor_radius"]),
            (
                "optics",
                &["ior", "absorption_r", "absorption_g", "absorption_b"],
            ),
            ("flow", &["flow_longitudinal", "flow_meridional"]),
            (
                "wave",
                &[
                    "wave_amplitude",
                    "wave_frequency",
                    "wave_speed",
                    "wave_dispersion",
                    "wave_lb_blend",
                ],
            ),
            (
                "lighting",
                &[
                    "light_intensity",
                    "highlight_sharpness",
                    "sky_brightness",
                    "scatter_strength",
                    "scatter_anisotropy",
                ],
            ),
            (
                "look",
                &[
                    "reflect_strength",
                    "refract_strength",
                    "caustic_strength",
                    "tint_r",
                    "tint_g",
                    "tint_b",
                ],
            ),
        ];

        for (group, members) in expected {
            let declared: Vec<&str> = WATER_UI_PARAMS
                .iter()
                .filter(|param| param.group == *group)
                .map(|param| param.name)
                .collect();
            assert_eq!(declared, *members, "{group}");
            for name in *members {
                assert!(
                    find_scalar_param(WATER_SCALAR_PARAMS, name).is_some(),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn test_runtime_ui_params_are_flagged_as_not_persisted() {
        for param in WATER_UI_PARAMS {
            let is_runtime = matches!(param.name, "time" | "time_scale" | "time_offset");
            assert_eq!(param.persisted, !is_runtime, "{}", param.name);
        }
    }

    #[test]
    fn test_overwrite_persisted_fields_keeps_runtime_state() {
        let mut loaded = WaterTorusEffect::default();
        loaded.major_radius = 5.0;
        loaded.time = 5.0;

        let mut target = WaterTorusEffect::default();
        target.time = 2.5;
        target.time_scale = 3.0;

        overwrite_water_persisted_fields(&mut target, &loaded);
        assert_eq!(target.major_radius, 5.0);
        assert_eq!(target.time, 2.5);
        assert_eq!(target.time_scale, 3.0);
    }

    #[test]
    fn test_runtime_params_are_not_serialized() {
        let value = serde_json::to_value(WaterTorusEffect::default()).expect("serialize");
        let object = value.as_object().expect("flat object");
        for name in ["time", "time_scale", "time_offset"] {
            assert!(!object.contains_key(name), "{name} must stay runtime-only");
        }
        assert_eq!(object.len(), WATER_PARAMETER_OWNERSHIP.len());
    }
}
