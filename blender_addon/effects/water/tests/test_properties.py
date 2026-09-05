"""Tests for pure Python functions in the water properties module (no bpy dependency)."""
from __future__ import annotations

import pytest

from blender_addon.effects.water.properties import (
    absorption_to_color,
    color_to_absorption,
    display_property_names,
)


class TestAbsorptionColorMapping:
    def test_roundtrip_within_tolerance(self):
        absorption = [0.35, 0.08, 0.02]
        restored = color_to_absorption(absorption_to_color(absorption, 1.0), 1.0)
        assert restored == pytest.approx(absorption, abs=1e-5)

    def test_zero_absorption_is_white(self):
        assert absorption_to_color([0.0, 0.0, 0.0], 1.0) == [1.0, 1.0, 1.0]

    def test_black_maps_to_finite_coefficient(self):
        coefficients = color_to_absorption([0.0, 0.0, 0.0], 1.0)
        assert all(c > 0.0 and c < 10.0 for c in coefficients)

    def test_reference_distance_scales_coefficient(self):
        color = absorption_to_color([0.5, 0.5, 0.5], 2.0)
        assert color_to_absorption(color, 2.0) == pytest.approx([0.5, 0.5, 0.5], abs=1e-6)


class TestDisplayPropertyNames:
    def test_only_absorption_gets_a_picker_alias(self):
        params = [
            {"name": "absorption", "kind": "absorption"},
            {"name": "tint", "kind": "color"},
            {"name": "ior", "kind": "scalar"},
            {"name": "legacy"},
        ]
        assert display_property_names(params) == {"absorption": "absorption_color"}
