"""Layer-3 boundary smoke: build-mode behaviour inside headless Blender.

Installs are done by the caller (``blender --command extension install-file``);
this script runs inside Blender and asserts the boundary matrix from
``boundary-tests.md`` against the *installed* addon:

- common: addon enables, ``build_config.BUILD_MODE`` matches, Discord hook
  present, ``thyllore.send_feedback`` registered in every mode,
  ``effective_ctx`` defaults to 32 (wheel-decided, no cached token)
- A/C: telemetry not importable
- B: telemetry bundled, ``should_send`` follows opt-in x online access

With ``--live-send`` (mode B builds pointed at the *test* Worker) it also
exercises the real data path end to end: run the curve_copilot operator so a
prediction is recorded, save the file to trigger the batch send, then assert
the Worker's full token flips ``effective_ctx`` to 64 and ``/v1/message``
accepts a free-text feedback message.

With ``--live-license`` + ``--license-key`` (mode C builds pointed at the
*test* Worker, key provisioned by the caller with max_seats=1) it exercises
the seat licensing path: activate -> ctx64, re-activate on the registered
device -> ctx64, switch device_id (a copied key) -> refused -> ctx32.

Invocation::

    blender --background --factory-startup --python <this script> -- \\
        --result result.json --expect-mode B [--onnx <path> --live-send] \\
        [--live-license --license-key <key>]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curve_copilot_operator_smoke as operator_smoke

MODE_EXPECTATIONS = {
    "A": {"telemetry": False, "license_client": False},
    "B": {"telemetry": True, "license_client": False},
    "C": {"telemetry": False, "license_client": True},
}
DEGRADED_CTX = 32
FULL_CTX = 64
LIVE_SEND_TIMEOUT_SECONDS = 20.0

checks: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--expect-mode", required=True, choices=sorted(MODE_EXPECTATIONS))
    parser.add_argument("--onnx", default="")
    parser.add_argument("--live-send", action="store_true")
    parser.add_argument("--live-license", action="store_true")
    parser.add_argument("--license-key", default="")
    return parser.parse_args(argv)


def module_bundled(addon_pkg: str, submodule: str) -> bool:
    return importlib.util.find_spec(f"{addon_pkg}.{submodule}") is not None


def send_operator_registered() -> bool:
    import bpy

    try:
        bpy.ops.thyllore.send_feedback.get_rna_type()
        return True
    except (AttributeError, KeyError):
        return False


def effective_ctx(addon_pkg: str) -> int | None:
    preferences = importlib.import_module(f"{addon_pkg}.preferences")
    return preferences._effective_context_length()


def run_common_checks(addon_pkg: str, expect_mode: str) -> None:
    build_config = importlib.import_module(f"{addon_pkg}.build_config")
    check(
        "build_config.BUILD_MODE",
        build_config.BUILD_MODE == expect_mode,
        f"got {build_config.BUILD_MODE}, expected {expect_mode}",
    )

    capabilities = importlib.import_module(f"{addon_pkg}.capabilities")
    expected = MODE_EXPECTATIONS[expect_mode]
    check(
        "capabilities.telemetry_available",
        capabilities.CAPS.telemetry_available == expected["telemetry"],
    )
    check(
        "capabilities.license_activation",
        capabilities.CAPS.license_activation == expected["license_client"],
    )
    check(
        "telemetry bundled iff mode B",
        module_bundled(addon_pkg, "telemetry") == expected["telemetry"],
    )
    check(
        "license_client bundled iff mode C",
        module_bundled(addon_pkg, "license_client") == expected["license_client"],
    )
    check(
        "send_feedback operator registered in every mode",
        send_operator_registered() is True,
    )
    check(
        "capabilities.message_available",
        capabilities.CAPS.message_available is True,
    )

    preferences = importlib.import_module(f"{addon_pkg}.preferences")
    check("Discord invite hook present", hasattr(preferences, "DISCORD_INVITE_URL"))

    if expected["telemetry"]:
        importlib.import_module(f"{addon_pkg}.telemetry").discard_full_token()
    if expected["license_client"]:
        importlib.import_module(f"{addon_pkg}.license_client").discard_full_token()
    ctx = effective_ctx(addon_pkg)
    check(
        "effective_ctx defaults to degraded",
        ctx == DEGRADED_CTX,
        f"got {ctx}, expected {DEGRADED_CTX}",
    )


def wait_for(predicate, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.5)
    return predicate()


def run_live_send_checks(addon_pkg: str, onnx_path: str) -> None:
    import bpy

    telemetry = importlib.import_module(f"{addon_pkg}.telemetry")
    records = importlib.import_module(f"{addon_pkg}.telemetry.records")
    preferences = importlib.import_module(f"{addon_pkg}.preferences")

    records._outbox_path().unlink(missing_ok=True)
    prefs = preferences.get_preferences()
    check("should_send off before opt-in", telemetry.should_send(prefs) is False)

    bpy.context.preferences.system.use_online_access = True
    prefs.telemetry_opt_in = True
    check("should_send on after opt-in + online access", telemetry.should_send(prefs) is True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    armature_object = operator_smoke.build_minimal_armature()
    operator_smoke.insert_initial_keyframes(armature_object)
    operator_smoke.select_first_location_x_fcurve(armature_object)
    operator_smoke.configure_preferences(addon_pkg, onnx_path)

    operator_status = bpy.ops.thyllore.curve_copilot("EXEC_DEFAULT")
    check("curve_copilot operator finished", "FINISHED" in operator_status)
    pending = telemetry.pending_count()
    check("prediction recorded for feedback", pending > 0, f"pending={pending}")

    blend_path = Path(tempfile.mkdtemp(prefix="thyllore_smoke_")) / "live_send.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    sent = wait_for(
        lambda: not records._outbox_path().exists()
        and telemetry.resolve_full_token() is not None,
        LIVE_SEND_TIMEOUT_SECONDS,
    )
    check("feedback batch sent and full token cached", sent)

    ctx = effective_ctx(addon_pkg)
    check("effective_ctx raised to full", ctx == FULL_CTX, f"got {ctx}, expected {FULL_CTX}")

    feedback_message = importlib.import_module(f"{addon_pkg}.feedback_message")
    check(
        "free-text message accepted by worker",
        feedback_message.send_message("layer3 boundary smoke", "0.0.1") is True,
    )

    prefs.telemetry_opt_in = False
    check("opt-out reverts to degraded ctx", effective_ctx(addon_pkg) == DEGRADED_CTX)


def run_live_license_checks(addon_pkg: str, license_key: str) -> None:
    import bpy

    preferences = importlib.import_module(f"{addon_pkg}.preferences")
    prefs = preferences.get_preferences()
    bpy.context.preferences.system.use_online_access = True
    original_device_id = prefs.device_id

    prefs.license_key = license_key
    status = bpy.ops.thyllore.refresh_license()
    check("license activation finished", "FINISHED" in status)
    ctx = effective_ctx(addon_pkg)
    check("effective_ctx raised by license", ctx == FULL_CTX, f"got {ctx}")

    status = bpy.ops.thyllore.refresh_license()
    check("registered device re-activates", "FINISHED" in status)

    prefs.device_id = "smoke-copied-device"
    try:
        status = bpy.ops.thyllore.refresh_license()
        refused = "CANCELLED" in status
    except RuntimeError as error:
        refused = "seats" in str(error)
    check("copied key (new device, 1 seat) refused", refused)
    ctx = effective_ctx(addon_pkg)
    check("refusal reverts to degraded ctx", ctx == DEGRADED_CTX, f"got {ctx}")

    prefs.device_id = original_device_id
    prefs.license_key = ""
    check("clearing key discards token", effective_ctx(addon_pkg) == DEGRADED_CTX)


def main() -> None:
    args = parse_args()

    addon_pkg = operator_smoke.enable_addon()
    if args.live_send and args.expect_mode == "B":
        if not args.onnx or not Path(args.onnx).exists():
            check("onnx model available for live send", False, args.onnx or "<missing>")
        else:
            run_common_checks(addon_pkg, args.expect_mode)
            run_live_send_checks(addon_pkg, str(Path(args.onnx).resolve()))
    elif args.live_license and args.expect_mode == "C":
        if not args.license_key:
            check("license key provided for live license", False, "<missing>")
        else:
            run_common_checks(addon_pkg, args.expect_mode)
            run_live_license_checks(addon_pkg, args.license_key)
    else:
        run_common_checks(addon_pkg, args.expect_mode)

    ok = all(entry["ok"] for entry in checks)
    payload = {"ok": ok, "expect_mode": args.expect_mode, "checks": checks}
    Path(args.result).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for entry in checks:
        marker = "PASS" if entry["ok"] else "FAIL"
        print(f"  {marker}  {entry['name']}  {entry['detail']}".rstrip())
    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
