from __future__ import annotations

import pathlib
import tomllib

MANIFEST_PATH = pathlib.Path(__file__).resolve().parent.parent / "blender_manifest.toml"


def test_id() -> None:
    data = tomllib.loads(MANIFEST_PATH.read_text())
    assert data["id"] == "thyllore_flame"


def test_type() -> None:
    data = tomllib.loads(MANIFEST_PATH.read_text())
    assert data["type"] == "add-on"


def test_schema_version_exists() -> None:
    data = tomllib.loads(MANIFEST_PATH.read_text())
    assert "schema_version" in data


def test_blender_version_min_exists() -> None:
    data = tomllib.loads(MANIFEST_PATH.read_text())
    assert "blender_version_min" in data


def test_wheels_count_and_placeholder() -> None:
    data = tomllib.loads(MANIFEST_PATH.read_text())
    wheels = data["wheels"]
    assert len(wheels) == 1
    assert "PLATFORM" in wheels[0]


def test_platforms_placeholder() -> None:
    data = tomllib.loads(MANIFEST_PATH.read_text())
    assert data["platforms"] == ["PLATFORM_BLENDER_NAME"]


def test_permissions_no_network() -> None:
    data = tomllib.loads(MANIFEST_PATH.read_text())
    assert "network" not in data["permissions"]


def test_paths_exclude_pattern_tests() -> None:
    data = tomllib.loads(MANIFEST_PATH.read_text())
    patterns = data["build"]["paths_exclude_pattern"]
    assert "tests/" in patterns


def test_paths_exclude_pattern_tools() -> None:
    data = tomllib.loads(MANIFEST_PATH.read_text())
    patterns = data["build"]["paths_exclude_pattern"]
    assert "tools/" in patterns
