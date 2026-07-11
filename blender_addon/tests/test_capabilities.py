"""Layer-1 boundary tests: capability derivation from BUILD_MODE.

Pure derivation checks against the boundary matrix in the design doc
(boundary-tests.md). No Blender required.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest

from blender_addon import capabilities as capabilities_module
from blender_addon.capabilities import BuildMode, Capabilities


def test_source_tree_defaults_to_official_mode():
    assert capabilities_module.BUILD_MODE == "A"
    assert capabilities_module.CAPS.mode is BuildMode.OFFICIAL


@pytest.mark.parametrize(
    ("mode", "telemetry_available", "license_activation"),
    [
        ("A", False, False),
        ("B", True, False),
        ("C", False, True),
    ],
)
def test_derivation_matrix(mode, telemetry_available, license_activation):
    caps = Capabilities(BuildMode(mode))
    assert caps.telemetry_available is telemetry_available
    assert caps.license_activation is license_activation


def test_invalid_build_mode_is_rejected():
    with pytest.raises(ValueError):
        BuildMode("X")


def test_generated_build_config_overrides_default(monkeypatch):
    build_config = types.ModuleType("blender_addon.build_config")
    build_config.BUILD_MODE = "B"
    monkeypatch.setitem(sys.modules, "blender_addon.build_config", build_config)
    try:
        reloaded = importlib.reload(capabilities_module)
        assert reloaded.BUILD_MODE == "B"
        assert reloaded.CAPS.telemetry_available is True
    finally:
        monkeypatch.delitem(sys.modules, "blender_addon.build_config", raising=False)
        importlib.reload(capabilities_module)
