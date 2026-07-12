"""Layer-2 boundary tests: built ZIP structure per build mode.

Runs scripts/build_blender_addon.sh with --build-mode A/B/C and asserts the
boundary matrix (boundary-tests.md): telemetry bundled only in B, secret keys
present per mode (key presence only -- values are never asserted or logged),
and every mode fails fast without the message-channel environment variables.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_blender_addon.sh"
OUTPUT_REL_DIR = "build/test_dist_build_mode_boundary"

MESSAGE_ENV = {
    "THYLLORE_FEEDBACK_ENDPOINT": "https://example.invalid/v1/feedback",
    "THYLLORE_INGEST_TOKEN": "dummy-ingest-token",
}
MODE_ENV = {
    "A": {**MESSAGE_ENV},
    "B": {
        **MESSAGE_ENV,
        "THYLLORE_FULL_TOKEN_PUBKEY_B64": "ZHVtbXktcHVia2V5",
    },
    "C": {
        **MESSAGE_ENV,
        "THYLLORE_LICENSE_ENDPOINT": "https://example.invalid/v1/license/refresh",
        "THYLLORE_FULL_TOKEN_PUBKEY_B64": "ZHVtbXktcHVia2V5",
    },
}

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("zip") is None,
    reason="build script requires linux + zip",
)


def _run_build(mode: str, extra_env: dict) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for key in (
        "THYLLORE_FEEDBACK_ENDPOINT",
        "THYLLORE_INGEST_TOKEN",
        "THYLLORE_FULL_TOKEN_PUBKEY_B64",
        "THYLLORE_LICENSE_ENDPOINT",
    ):
        env.pop(key, None)
    env.update(extra_env)
    return subprocess.run(
        [
            "bash",
            str(BUILD_SCRIPT),
            "--platform",
            "linux_x86_64",
            "--variant",
            "lite",
            "--build-mode",
            mode,
            "--output-dir",
            OUTPUT_REL_DIR,
            "--skip-blender-validate",
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


@pytest.fixture(scope="module")
def built_zips() -> dict[str, Path]:
    output_dir = REPO_ROOT / OUTPUT_REL_DIR
    shutil.rmtree(output_dir, ignore_errors=True)

    zips: dict[str, Path] = {}
    for mode, extra_env in MODE_ENV.items():
        result = _run_build(mode, extra_env)
        assert result.returncode == 0, f"mode {mode} build failed:\n{result.stderr}"
        suffix = "" if mode == "A" else f"_mode_{mode.lower()}"
        zips[mode] = (
            output_dir / f"thyllore_animation_lite{suffix}-0.0.1-linux_x86_64.zip"
        )
        assert zips[mode].exists(), f"expected ZIP missing for mode {mode}"

    yield zips
    shutil.rmtree(output_dir, ignore_errors=True)


def _zip_names(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        return archive.namelist()


def _build_config_keys(zip_path: Path) -> set[str]:
    with zipfile.ZipFile(zip_path) as archive:
        text = archive.read("build_config.py").decode("utf-8")
    return {line.split("=", 1)[0].strip() for line in text.splitlines() if "=" in line}


@pytest.mark.parametrize(
    ("mode", "telemetry_bundled"),
    [("A", False), ("B", True), ("C", False)],
)
def test_telemetry_bundled_only_in_mode_b(built_zips, mode, telemetry_bundled):
    has_telemetry = any(name.startswith("telemetry/") for name in _zip_names(built_zips[mode]))
    assert has_telemetry is telemetry_bundled


@pytest.mark.parametrize(
    ("mode", "expected_keys"),
    [
        ("A", {"BUILD_MODE", "FEEDBACK_ENDPOINT", "INGEST_TOKEN"}),
        ("B", {"BUILD_MODE", "FEEDBACK_ENDPOINT", "INGEST_TOKEN", "FULL_TOKEN_PUBKEY"}),
        (
            "C",
            {
                "BUILD_MODE",
                "FEEDBACK_ENDPOINT",
                "INGEST_TOKEN",
                "FULL_TOKEN_PUBKEY",
                "LICENSE_ENDPOINT",
            },
        ),
    ],
)
def test_build_config_has_exactly_the_mode_fields(built_zips, mode, expected_keys):
    assert _build_config_keys(built_zips[mode]) == expected_keys


@pytest.mark.parametrize(
    ("mode", "license_client_bundled"),
    [("A", False), ("B", False), ("C", True)],
)
def test_license_client_bundled_only_in_mode_c(built_zips, mode, license_client_bundled):
    has_license_client = any(
        name.startswith("license_client/") for name in _zip_names(built_zips[mode])
    )
    assert has_license_client is license_client_bundled


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_every_mode_fails_fast_without_message_env(mode):
    result = _run_build(mode, {})
    assert result.returncode != 0
    assert "THYLLORE_FEEDBACK_ENDPOINT" in result.stderr


def test_feedback_message_bundled_in_every_mode(built_zips):
    for mode, zip_path in built_zips.items():
        assert "feedback_message.py" in _zip_names(zip_path), f"missing in mode {mode}"
