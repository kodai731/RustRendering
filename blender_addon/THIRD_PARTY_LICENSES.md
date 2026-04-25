# Third-Party Licenses

This add-on bundles the following third-party components, each governed by
its own license. The bundle is provided under the FSF *System Library*
exception, which permits independently licensed binaries to be distributed
alongside GPL Python code.

## thyllore_ml_core (`wheels/thyllore_ml_core-*.whl`)

- License: Proprietary, Copyright © 2026 kodai731.
- Distribution: Bundled binary artifact, treated as Independent Work.
- Source repository: https://github.com/kodai731/Thyllore-Animation .
- Built from `crates/thyllore-ml-core/` via maturin.
- Linked Rust crates (e.g. `ort`, `numpy`) carry their own MIT/Apache-2.0
  licenses recorded in their respective package metadata.

## grpcio, grpcio-status, protobuf (`wheels/grpcio*.whl`, `wheels/protobuf*.whl`)

- License: Apache-2.0 (grpcio family) and BSD-3-Clause (protobuf).
- Sources:
  - https://github.com/grpc/grpc
  - https://github.com/protocolbuffers/protobuf
- Apache-2.0 full text: https://www.apache.org/licenses/LICENSE-2.0 .

## certifi (`wheels/certifi-*.whl`)

- License: MPL-2.0 (Mozilla Public License 2.0).
- Source: https://github.com/certifi/python-certifi .

## cryptography (Blender's bundled Python)

- License: Apache-2.0 / BSD-3-Clause (dual).
- Source: https://github.com/pyca/cryptography .

## Auto-generated gRPC stubs (`grpc_client/stubs/*.py`)

- License: Apache-2.0 (inherited from grpcio's protoc plugin).
- Source: Generated from `crates/thyllore-grpc-client/proto/animation_ml.proto`
  via grpcio-tools; regenerable by `scripts/gen_grpc_stubs.ps1`.

## curve_copilot.onnx (Tier B model, when bundled)

- License: Proprietary, Copyright © 2026 kodai731.
- Distribution: Optional bundled binary asset.
