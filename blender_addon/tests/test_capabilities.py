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
    assert capabilities_module.CAPS.message_available is False


@pytest.mark.parametrize(
    ("mode", "telemetry_available", "curve_copilot_mode"),
    [
        ("A", False, "degrade"),
        ("B", True, "full"),
        ("C", False, "private"),
    ],
)
def test_derivation_matrix(mode, telemetry_available, curve_copilot_mode):
    caps = Capabilities(BuildMode(mode))
    assert caps.telemetry_available is telemetry_available
    assert caps.curve_copilot_mode == curve_copilot_mode


def test_invalid_build_mode_is_rejected():
    with pytest.raises(ValueError):
        BuildMode("X")


def test_generated_build_config_overrides_default(monkeypatch):
    build_config = types.ModuleType("blender_addon.build_config")
    build_config.BUILD_MODE = "B"
    build_config.FEEDBACK_ENDPOINT = "https://example.invalid/v1/feedback"
    build_config.INGEST_TOKEN = "dummy-ingest-token"
    monkeypatch.setitem(sys.modules, "blender_addon.build_config", build_config)
    try:
        reloaded = importlib.reload(capabilities_module)
        assert reloaded.BUILD_MODE == "B"
        assert reloaded.CAPS.telemetry_available is True
        assert reloaded.CAPS.message_available is True
    finally:
        monkeypatch.delitem(sys.modules, "blender_addon.build_config", raising=False)
        importlib.reload(capabilities_module)


@pytest.mark.parametrize(
    ("mode", "message_available"),
    [("A", False), ("B", True), ("C", False)],
)
def test_message_available_only_in_mode_b(monkeypatch, mode, message_available):
    build_config = types.ModuleType("blender_addon.build_config")
    build_config.BUILD_MODE = mode
    build_config.FEEDBACK_ENDPOINT = "https://example.invalid/v1/feedback"
    build_config.INGEST_TOKEN = "dummy-ingest-token"
    monkeypatch.setitem(sys.modules, "blender_addon.build_config", build_config)
    try:
        reloaded = importlib.reload(capabilities_module)
        assert reloaded.CAPS.message_available is message_available
    finally:
        monkeypatch.delitem(sys.modules, "blender_addon.build_config", raising=False)
        importlib.reload(capabilities_module)
