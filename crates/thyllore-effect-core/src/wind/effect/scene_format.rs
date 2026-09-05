use crate::wind::ownership::WindParameterOwner;
use crate::wind::*;
use cgmath::{Quaternion, Vector3};
use thyllore_scene_core::declare_scene_format;

declare_scene_format! {
    component: WindTornadoEffect,
    record: WindSceneRecord,
    tag: WindParameterOwner,
    items {
        tags: WIND_PARAMETER_OWNERSHIP,
        snapshot: wind_parameter_snapshot,
        scalars: WIND_SCALAR_PARAMS,
        ui: WIND_UI_PARAMS,
        overwrite: overwrite_wind_persisted_fields,
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
        column_height: f32 = Frame {
            get: |e| e.column_height,
            set: |e, v| e.column_height = v,
            ui {
                min: 0.1,
                max: 20.0,
                format: "%.2f",
            },
        },
        core_radius: f32 = Frame {
            get: |e| e.core_radius,
            set: |e, v| e.core_radius = v,
            ui {
                min: 0.0,
                max: 5.0,
                format: "%.3f",
            },
        },
        core_strength: f32 = Frame {
            get: |e| e.core_strength,
            set: |e, v| e.core_strength = v,
            ui {
                min: 0.0,
                max: 4.0,
                format: "%.2f",
            },
        },
        wall_radius_base: f32 = Frame {
            get: |e| e.wall_radius_base,
            set: |e, v| e.wall_radius_base = v,
            ui {
                min: 0.01,
                max: 10.0,
                format: "%.3f",
            },
        },
        wall_radius_top: f32 = Frame {
            get: |e| e.wall_radius_top,
            set: |e, v| e.wall_radius_top = v,
            ui {
                min: 0.01,
                max: 10.0,
                format: "%.3f",
            },
        },
        wall_width_q: f32 = Frame {
            get: |e| e.wall_width_q,
            set: |e, v| e.wall_width_q = v,
            ui {
                min: 0.001,
                max: 5.0,
                format: "%.3f",
                tooltip: "Half width of the wall shell in squared-radius units; the radial thickness is about wall_width_q / (2 R)",
            },
        },
        wall_strength: f32 = Frame {
            get: |e| e.wall_strength,
            set: |e, v| e.wall_strength = v,
            ui {
                min: 0.0,
                max: 4.0,
                format: "%.2f",
            },
        },
        top_fade: f32 = Frame {
            get: |e| e.top_fade,
            set: |e, v| e.top_fade = v,
            ui {
                min: 0.01,
                max: 1.0,
                format: "%.2f",
                tooltip: "Fraction of the height over which the density fades to zero at the top",
            },
        },
        density: f32 = Frame {
            get: |e| e.density,
            set: |e, v| e.density = v,
            ui {
                min: 0.0,
                max: 50.0,
                format: "%.2f",
                tooltip: "Extinction coefficient per meter at unit shell density",
            },
        },
        albedo: [f32; 3] = Frame {
            get: |e| e.albedo,
            set: |e, v| e.albedo = v,
            scalars: rgb,
            ui {
                kind: Color,
                min: 0.0,
                max: 1.0,
                format: "%.2f",
                tooltip: "Single-scattering albedo of the dust",
            },
        },
        ambient_brightness: f32 = Frame {
            get: |e| e.ambient_brightness,
            set: |e, v| e.ambient_brightness = v,
            ui {
                min: 0.0,
                max: 5.0,
                format: "%.2f",
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
    use thyllore_scene_core::{find_scalar_param, find_ui_param, UiKind};

    #[test]
    fn test_ron_struct_syntax_roundtrip() {
        let mut effect = WindTornadoEffect::default();
        effect.column_height = 3.5;
        effect.wall_width_q = 0.2;

        let text = ron::ser::to_string_pretty(&effect, ron::ser::PrettyConfig::new())
            .expect("ron serialize");
        let restored: WindTornadoEffect = ron::from_str(&text).expect("ron deserialize");
        assert_eq!(restored.column_height, 3.5);
        assert_eq!(restored.wall_width_q, 0.2);
    }

    #[test]
    fn test_scalar_param_names_are_unique() {
        let mut names: Vec<&str> = WIND_SCALAR_PARAMS.iter().map(|p| p.name).collect();
        names.sort_unstable();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
    }

    #[test]
    fn test_every_ui_param_has_scalar_accessors_and_unique_name() {
        let mut names: Vec<&str> = WIND_UI_PARAMS.iter().map(|p| p.name).collect();
        names.sort_unstable();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
        for param in WIND_UI_PARAMS {
            for accessor_name in param.scalar_accessor_names() {
                assert!(
                    find_scalar_param(WIND_SCALAR_PARAMS, &accessor_name).is_some(),
                    "{accessor_name}"
                );
            }
            assert!(param.min < param.max, "{}", param.name);
        }
    }

    #[test]
    fn test_albedo_is_a_color_and_serializes_as_one_vector() {
        assert_eq!(
            find_ui_param(WIND_UI_PARAMS, "albedo").map(|p| p.kind),
            Some(UiKind::Color)
        );
        let value = serde_json::to_value(WindTornadoEffect::default()).expect("serialize");
        let object = value.as_object().expect("flat object");
        assert!(object["albedo"].is_array());
        assert!(!object.contains_key("albedo_r"));
    }

    #[test]
    fn test_runtime_params_are_not_serialized() {
        let value = serde_json::to_value(WindTornadoEffect::default()).expect("serialize");
        let object = value.as_object().expect("flat object");
        for name in ["time", "time_scale", "time_offset"] {
            assert!(!object.contains_key(name), "{name} must stay runtime-only");
        }
        assert_eq!(object.len(), WIND_PARAMETER_OWNERSHIP.len());
    }

    #[test]
    fn test_overwrite_persisted_fields_keeps_runtime_state() {
        let mut loaded = WindTornadoEffect::default();
        loaded.column_height = 5.0;
        loaded.time = 5.0;

        let mut target = WindTornadoEffect::default();
        target.time = 2.5;

        overwrite_wind_persisted_fields(&mut target, &loaded);
        assert_eq!(target.column_height, 5.0);
        assert_eq!(target.time, 2.5);
    }
}
