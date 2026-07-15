# Third-Party Licenses

This add-on bundles the following components, each governed by its own
license. The bundle is provided under the FSF *System Library* exception,
which permits independently licensed binaries to be distributed alongside
GPL Python code.

## thyllore_ml_core (`wheels/thyllore_ml_core-*.whl`)

- License: Apache-2.0, Copyright © 2026 kodai731.
- Distribution: Bundled binary artifact, treated as Independent Work.
- Source repository: https://github.com/kodai731/Thyllore-Animation .
- Built from `crates/thyllore-ml-core/` via maturin.
- Apache-2.0 full text: https://www.apache.org/licenses/LICENSE-2.0 .
- Linked Rust crates (e.g. `ort`, `numpy`) carry their own MIT/Apache-2.0
  licenses recorded in their respective package metadata.

## ONNX Runtime (`lib/`)

- License: MIT, Copyright © Microsoft Corporation.
- Source: https://github.com/microsoft/onnxruntime .
- Distribution: Bundled shared library used by `thyllore_ml_core`.

## cryptography (Blender's bundled Python)

- License: Apache-2.0 / BSD-3-Clause (dual).
- Source: https://github.com/pyca/cryptography .

## curve_copilot.onnx (`models/curve_copilot.onnx`)

- License: Proprietary, Copyright © 2026 kodai731. Governed by `EULA.md`.
- Distribution: Bundled binary asset.
