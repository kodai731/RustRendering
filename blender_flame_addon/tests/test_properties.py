"""Tests for pure Python functions in properties module (no bpy dependency)."""
from __future__ import annotations

import pytest

from blender_flame_addon.properties import (
    collect_params,
    is_exposed_param,
    merge_preset_params,
    precision_from_format,
    property_kind,
    select_exposed_params,
)


class TestPrecisionFromFormat:
    def test_two_decimal(self):
        assert precision_from_format("%.2f") == 2

    def test_three_decimal(self):
        assert precision_from_format("%.3f") == 3

    def test_integer(self):
        assert precision_from_format("%d") == 0

    def test_unknown(self):
        assert precision_from_format("unknown") == 3

    def test_zero_decimal(self):
        assert precision_from_format("%.0f") == 0


class TestPropertyKind:
    def test_string_bool(self):
        assert property_kind("bool") == "bool"

    def test_string_vector(self):
        assert property_kind("vector") == "vector"

    def test_string_float(self):
        assert property_kind("float") == "float"

    def test_actual_bool_true(self):
        assert property_kind(True) == "bool"

    def test_actual_bool_false(self):
        assert property_kind(False) == "bool"

    def test_list(self):
        assert property_kind([1.0, 2.0, 3.0]) == "vector"

    def test_tuple(self):
        assert property_kind((1.0, 2.0)) == "vector"

    def test_float_value(self):
        assert property_kind(1.5) == "float"

    def test_int_value(self):
        assert property_kind(42) == "float"


class TestCollectParams:
    def test_scalar_values(self):
        class FakeProps:
            height = 1.6
            width = 0.8
            is_on = True

        props = FakeProps()
        result = collect_params(props, ["height", "width", "is_on"])
        assert result == {"height": 1.6, "width": 0.8, "is_on": True}

    def test_vector_values(self):
        class FakeProps:
            color_inner = (1.0, 0.5, 0.0)

        props = FakeProps()
        result = collect_params(props, ["color_inner"])
        assert result == {"color_inner": [1.0, 0.5, 0.0]}

    def test_mixed(self):
        class FakeProps:
            height = 1.6
            color_inner = (1.0, 0.5, 0.0)
            is_on = True

        props = FakeProps()
        result = collect_params(props, ["height", "color_inner", "is_on"])
        assert result == {"height": 1.6, "color_inner": [1.0, 0.5, 0.0], "is_on": True}

    def test_empty_names(self):
        class FakeProps:
            height = 1.6

        props = FakeProps()
        result = collect_params(props, [])
        assert result == {}


class TestExposedParams:
    def test_named_params_are_exposed(self):
        assert is_exposed_param("height")
        assert is_exposed_param("radius")
        assert is_exposed_param("optical_depth")

    def test_noise_prefix_is_exposed(self):
        assert is_exposed_param("noise_amplitude")
        assert is_exposed_param("noise_aniso_y")

    def test_other_params_are_hidden(self):
        assert not is_exposed_param("intensity")
        assert not is_exposed_param("branch_gain")
        assert not is_exposed_param("swirl_gain")

    def test_select_keeps_order_and_filters(self):
        ui_params = [{"name": "intensity"}, {"name": "height"}, {"name": "noise_contrast"}, {"name": "mix_lo"}]
        assert [p["name"] for p in select_exposed_params(ui_params)] == ["height", "noise_contrast"]


class TestMergePresetParams:
    def test_exposed_overrides_preset(self):
        merged = merge_preset_params({"height": 1.6, "intensity": 2.0}, {"height": 0.3})
        assert merged == {"height": 0.3, "intensity": 2.0}

    def test_inputs_are_not_mutated(self):
        preset = {"height": 1.6}
        merge_preset_params(preset, {"height": 0.3})
        assert preset == {"height": 1.6}
