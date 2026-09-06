use crate::flame::{
    apply_flame_preset, build_flame_ubo, refresh_flame_coefficients, FlameBaked, FlameEffect,
    FlameTemporalAccum,
};
use cgmath::{Quaternion, Vector3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn test_round_trip_bit_identity() {
    Python::attach(|py| {
        let preset_dict: Bound<'_, PyDict> = super::flame_preset_params(py, "campfire").unwrap();

        let bytes: Vec<u8> = super::pack_flame_ubo(
            py,
            &preset_dict,
            1.5f32,
            [0.0f32, 1.0f32, 2.0f32],
            [1.0f32, 0.0f32, 0.0f32, 0.0f32],
            None,
            0,
        )
        .unwrap();

        let mut effect = FlameEffect::default();
        apply_flame_preset(&mut effect, "campfire");
        effect.time = 1.5;
        effect.position = Vector3::new(0.0, 1.0, 2.0);
        effect.rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);

        let baked = FlameBaked::default();
        refresh_flame_coefficients(&mut effect, &baked);

        let temporal = FlameTemporalAccum::default();
        let ubo = build_flame_ubo(&effect, &baked, &temporal);

        let expected_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                &ubo as *const crate::flame::FlameUBO as *const u8,
                std::mem::size_of::<crate::flame::FlameUBO>(),
            )
        }
        .to_vec();

        assert_eq!(bytes, expected_bytes, "round-trip bytes differ");
    });
}

#[test]
fn test_ui_params_subset_of_preset_keys() {
    Python::attach(|py| {
        let ui_list = super::flame_ui_params(py).unwrap();
        let ui_names: Vec<String> = ui_list
            .try_iter()
            .unwrap()
            .map(|item| {
                let dict: Bound<'_, PyDict> = item.unwrap().cast_into::<PyDict>().unwrap();
                let name_obj = dict.get_item("name").unwrap().unwrap();
                let name: String = name_obj.extract().unwrap();
                name
            })
            .collect();

        let preset_dict: Bound<'_, PyDict> = super::flame_preset_params(py, "campfire").unwrap();
        let all_keys: Vec<String> = preset_dict
            .keys()
            .into_iter()
            .map(|k| k.extract().unwrap())
            .collect();

        for name in &ui_names {
            assert!(
                all_keys.contains(name),
                "UI param '{}' not found in preset keys: {:?}",
                name,
                all_keys
            );
        }
    });
}

#[test]
fn test_unknown_key_raises_value_error() {
    Python::attach(|py| {
        let bad_dict = PyDict::new(py);
        bad_dict.set_item("nonexistent", 1.0f32).unwrap();

        let result = super::pack_flame_ubo(
            py,
            &bad_dict,
            1.5f32,
            [0.0f32, 1.0f32, 2.0f32],
            [1.0f32, 0.0f32, 0.0f32, 0.0f32],
            None,
            0,
        );

        assert!(result.is_err(), "expected ValueError for unknown key");
        let err = result.unwrap_err();
        assert!(
            err.is_instance_of::<pyo3::exceptions::PyValueError>(py),
            "expected PyValueError, got {:?}",
            err
        );
    });
}

#[test]
fn test_empty_params_equals_campfire() {
    Python::attach(|py| {
        let empty: Bound<'_, PyDict> = PyDict::new(py);
        let campfire = super::flame_preset_params(py, "campfire").unwrap();
        let bytes_empty = super::pack_flame_ubo(
            py,
            &empty,
            1.5f32,
            [0.0f32, 1.0f32, 2.0f32],
            [1.0f32, 0.0f32, 0.0f32, 0.0f32],
            None,
            0,
        )
        .unwrap();
        let bytes_campfire = super::pack_flame_ubo(
            py,
            &campfire,
            1.5f32,
            [0.0f32, 1.0f32, 2.0f32],
            [1.0f32, 0.0f32, 0.0f32, 0.0f32],
            None,
            0,
        )
        .unwrap();
        assert_eq!(
            bytes_empty, bytes_campfire,
            "empty params should equal campfire preset (which is the default)"
        );
    });
}

#[test]
fn test_light_position_bit_identity() {
    Python::attach(|py| {
        let preset_dict: Bound<'_, PyDict> = super::flame_preset_params(py, "campfire").unwrap();

        let bytes: Vec<u8> = super::pack_flame_ubo(
            py,
            &preset_dict,
            1.5f32,
            [0.0f32, 1.0f32, 2.0f32],
            [1.0f32, 0.0f32, 0.0f32, 0.0f32],
            Some([1.0f32, 1.0f32, 2.0f32]),
            0,
        )
        .unwrap();

        let mut effect = FlameEffect::default();
        apply_flame_preset(&mut effect, "campfire");
        effect.time = 1.5;
        effect.position = Vector3::new(0.0, 1.0, 2.0);
        effect.rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        effect.light_position_world = Vector3::new(1.0, 1.0, 2.0);

        let baked = FlameBaked::default();
        refresh_flame_coefficients(&mut effect, &baked);

        let temporal = FlameTemporalAccum::default();
        let ubo = build_flame_ubo(&effect, &baked, &temporal);

        let expected_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                &ubo as *const crate::flame::FlameUBO as *const u8,
                std::mem::size_of::<crate::flame::FlameUBO>(),
            )
        }
        .to_vec();

        assert_eq!(bytes, expected_bytes, "light_position bytes differ");
    });
}

#[test]
fn test_frame_index_value() {
    Python::attach(|py| {
        let preset_dict: Bound<'_, PyDict> = super::flame_preset_params(py, "campfire").unwrap();

        let bytes: Vec<u8> = super::pack_flame_ubo(
            py,
            &preset_dict,
            1.5f32,
            [0.0f32, 1.0f32, 2.0f32],
            [1.0f32, 0.0f32, 0.0f32, 0.0f32],
            None,
            1,
        )
        .unwrap();

        let frame_index_value =
            f32::from_le_bytes([bytes[324], bytes[325], bytes[326], bytes[327]]);
        assert_eq!(
            frame_index_value, 1.0,
            "frame_index at offset 324 should be 1.0"
        );
    });
}

#[test]
fn test_shader_specialization_matches_packed_ubo() {
    Python::attach(|py| {
        let preset_dict: Bound<'_, PyDict> = super::flame_preset_params(py, "campfire").unwrap();
        let spec = super::flame_shader_specialization(py, &preset_dict).unwrap();

        let mut effect = FlameEffect::default();
        apply_flame_preset(&mut effect, "campfire");
        let baked = FlameBaked::default();
        refresh_flame_coefficients(&mut effect, &baked);
        let ubo = build_flame_ubo(&effect, &baked, &FlameTemporalAccum::default());

        let get = |key: &str| -> f32 {
            spec.get_item(key)
                .unwrap()
                .unwrap()
                .extract::<f32>()
                .unwrap()
        };
        assert_eq!(get("flame.emitterParams.kind"), ubo.emitter_params.kind);
        assert_eq!(
            get("flame.contourParams.rteBands"),
            ubo.contour_params.rte_bands
        );
        assert_eq!(
            get("flame.trailMeta.sampleCount"),
            ubo.trail_meta.sample_count
        );
        assert_eq!(spec.len(), 3);
    });
}

#[test]
fn test_bounds_corners_follow_position_and_height() {
    Python::attach(|py| {
        let preset_dict: Bound<'_, PyDict> = super::flame_preset_params(py, "campfire").unwrap();
        let corners =
            super::flame_bounds_corners(py, &preset_dict, [1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
                .unwrap();

        let mut effect = FlameEffect::default();
        apply_flame_preset(&mut effect, "campfire");
        let min_y = corners.iter().map(|c| c[1]).fold(f32::MAX, f32::min);
        let max_y = corners.iter().map(|c| c[1]).fold(f32::MIN, f32::max);
        let min_x = corners.iter().map(|c| c[0]).fold(f32::MAX, f32::min);
        let max_x = corners.iter().map(|c| c[0]).fold(f32::MIN, f32::max);

        assert_eq!(corners.len(), 8);
        assert!(
            (min_y - 2.0).abs() < 1e-4,
            "base sits at the flame position, got {min_y}"
        );
        assert!(
            max_y > 2.0 + effect.height * 0.9,
            "top reaches the flame height, got {max_y}"
        );
        assert!(
            (min_x + max_x - 2.0).abs() < 1e-4,
            "box is centred on x, got {min_x}..{max_x}"
        );
    });
}

#[test]
fn test_effective_optical_depth_falls_back_to_sigma_t_times_radius() {
    Python::attach(|py| {
        let campfire = super::flame_preset_params(py, "campfire").unwrap();
        let depth = super::flame_effective_optical_depth(py, &campfire).unwrap();
        assert!(
            (depth - 0.6).abs() < 1e-5,
            "campfire sigma_t 1.0 * radius 0.6, got {depth}"
        );

        campfire.set_item("optical_depth", 3.0).unwrap();
        let depth = super::flame_effective_optical_depth(py, &campfire).unwrap();
        assert!(
            (depth - 3.0).abs() < 1e-5,
            "explicit optical_depth wins, got {depth}"
        );
    });
}

#[test]
fn test_wind_spread_offset_grows_after_spread_start() {
    Python::attach(|py| {
        let preset_dict: Bound<'_, PyDict> = super::wind_preset_params(py, "funnel").unwrap();
        let spread_start: f32 = preset_dict
            .get_item("spread_start")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        let read_spread_offset = |time: f32| -> f32 {
            let bytes = super::pack_wind_ubo(
                py,
                &preset_dict,
                time,
                [0.0f32, 0.0f32, 0.0f32],
                [1.0f32, 0.0f32, 0.0f32, 0.0f32],
            )
            .unwrap();
            let offset = std::mem::offset_of!(crate::wind::WindUBO, albedo) + 12;
            f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        };

        let at_start = read_spread_offset(0.0);
        let after_start = read_spread_offset(spread_start + 1.0);

        assert_eq!(at_start, 0.0, "no spread before spread_start");
        assert!(
            after_start > 0.0,
            "spread_offset should grow after spread_start, got {after_start}"
        );
    });
}

#[test]
fn test_water_ui_params_expose_kind_and_reference_distance() {
    Python::attach(|py| {
        let ui_list = super::water_ui_params(py).unwrap();
        let mut kinds = std::collections::HashMap::new();
        let mut reference_distance = None;
        for item in ui_list.try_iter().unwrap() {
            let dict: Bound<'_, PyDict> = item.unwrap().cast_into::<PyDict>().unwrap();
            let name: String = dict.get_item("name").unwrap().unwrap().extract().unwrap();
            let kind: String = dict.get_item("kind").unwrap().unwrap().extract().unwrap();
            if let Some(distance) = dict.get_item("reference_distance").unwrap() {
                reference_distance = Some((name.clone(), distance.extract::<f32>().unwrap()));
            }
            kinds.insert(name, kind);
        }

        assert_eq!(kinds["absorption"], "absorption");
        assert_eq!(kinds["tint"], "color");
        assert_eq!(kinds["ior"], "scalar");
        assert_eq!(
            reference_distance,
            Some((
                "absorption".to_string(),
                crate::water::ABSORPTION_REFERENCE_DISTANCE
            ))
        );
    });
}
