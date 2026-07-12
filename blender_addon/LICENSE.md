# Thyllore Animation Add-on License

The add-on Python source files in this distribution that link to Blender's
``bpy`` API are licensed under the GNU General Public License, version 3 or
(at the user's option) any later version (**GPL-3.0-or-later**) -- the
license required by the extensions.blender.org add-on policy.

Compiled binaries (`.pyd` / `.whl`), trained models (`.onnx`), and
auto-generated gRPC stubs are bundled under separate licenses described in
`THIRD_PARTY_LICENSES.md`. Their use here falls under the FSF System Library
exception (https://www.gnu.org/licenses/gpl-faq.html#GPLIncompatibleLibs) and
the proprietary binaries are governed by the accompanying `EULA.md`.

## Variant scope

This repository builds two ZIP variants. Both ship under the same
GPL-3.0-or-later terms for their Python sources; the difference is which
proprietary binaries are bundled.

- **Lite (MVP)** -- `thyllore_animation_lite`: ships only the Tier B
  in-process ONNX operators (Curve Copilot, and once the rewrite lands,
  Text-to-Motion). No gRPC client, no license verification, no SaaS
  authentication. Distributed as a one-time purchase.
- **Full** -- `thyllore_animation`: adds Tier A gRPC operators (Auto Rig,
  Text-to-Mesh) and the Phase 6 Auth Backend client. Distributed alongside a
  Thyllore Cloud subscription.

## GPL-3.0-or-later notice

Copyright (C) 2026 kodai731 <kodai731@gmail.com>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, version 3 or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

A copy of the GPL is available at https://www.gnu.org/licenses/gpl-3.0.html .

## Files covered by GPL-3.0-or-later

- `__init__.py`
- `_bootstrap.py`
- `preferences.py`
- `operators/*.py`
- `panels/*.py`
- `modal/*.py`

## Files covered by other licenses

- `wheels/thyllore_ml_core-*.whl` (proprietary, see `EULA.md` and
  `THIRD_PARTY_LICENSES.md`)
- `wheels/grpcio-*.whl`, `wheels/grpcio_status-*.whl`,
  `wheels/protobuf-*.whl`, `wheels/certifi-*.whl` (Apache-2.0 / BSD / MPL,
  see `THIRD_PARTY_LICENSES.md`; full Variant only)
- `assets/curve_copilot.onnx`, lazily-downloaded `light_t2m.onnx`
  (proprietary, see `EULA.md`)
- `grpc_client/stubs/*.py` (Apache-2.0, auto-generated from
  `proto/animation_ml.proto`; full Variant only)

The proprietary binaries above are independent works and are not derivative
works of the GPL-licensed Python sources for the purposes of GPL section 0.
They are invoked via a documented public ABI (`thyllore-ml-api ABI_MARKER`)
and may be replaced or removed without affecting the GPL portion.

See `EULA.md` for the End User License Agreement governing the proprietary
binaries.

See `THIRD_PARTY_LICENSES.md` for the complete attribution of bundled
third-party packages.
