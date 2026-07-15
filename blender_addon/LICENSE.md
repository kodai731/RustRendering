# Thyllore Animation Add-on License

The add-on Python source files in this distribution that link to Blender's
``bpy`` API are licensed under the GNU General Public License, version 3 or
(at the user's option) any later version (**GPL-3.0-or-later**) -- the
license required by the extensions.blender.org add-on policy.

Compiled binaries (`.whl`) and the trained model (`.onnx`) are bundled under
separate licenses described in `THIRD_PARTY_LICENSES.md`. Their use here
falls under the FSF System Library exception
(https://www.gnu.org/licenses/gpl-faq.html#GPLIncompatibleLibs).
The Curve Copilot inference wheel (`thyllore_ml_core`) is licensed under
**Apache-2.0**, the same license as its public source repository. The
proprietary Curve Copilot ONNX model is governed by the accompanying
`EULA.md`.

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

All Python source files (`*.py`) in this distribution.

## Files covered by other licenses

- `wheels/thyllore_ml_core-*.whl` (**Apache-2.0**, built from the public
  source repository; see `THIRD_PARTY_LICENSES.md`)
- `models/curve_copilot.onnx` (proprietary, see `EULA.md`)
- `lib/` ONNX Runtime shared library (MIT, see `THIRD_PARTY_LICENSES.md`)

The binaries and the model above are independent works and are not
derivative works of the GPL-licensed Python sources for the purposes of GPL
section 0. They are invoked via a documented public ABI
(`thyllore-ml-api ABI_MARKER`) and may be replaced or removed without
affecting the GPL portion.

See `EULA.md` for the End User License Agreement governing the proprietary
Curve Copilot ONNX model.

See `THIRD_PARTY_LICENSES.md` for the complete attribution of bundled
third-party packages.
