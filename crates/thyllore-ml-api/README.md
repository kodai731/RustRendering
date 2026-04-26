# thyllore-ml-api

Layer 2 (stable contract) between [`thyllore-ml-core`] (Rust implementation) and the
PyO3 wheel that ships inside the Blender addon. This crate defines:

- The [`MlOps`] trait, which the implementation crate satisfies and the wheel
  consumes through a single owned instance.
- Request/response schemas (`CopilotRequest`, `CopilotResponse`,
  `CurvePredictRequest`, `CurvePredictResponse`, `TopologyResult`,
  `SkeletonSnapshot`). All schemas are `serde`-serializable so the same payload
  can travel through typed methods or through the generic `call_op` dispatch.
- The [`MlError`] enum used uniformly across the boundary.
- `ABI_MARKER`, a compile-time constant the wheel re-exports as
  `thyllore_ml_core.__abi_marker__`. The Blender addon compares it against its
  own `EXPECTED_ABI_MARKER` at startup to decide whether to activate.

The crate intentionally has no runtime dependencies beyond `serde` and
`thiserror`, so the contract has the smallest possible surface area. New
features should usually flow through `MlOps::call_op` (see the design doc) and
only graduate to typed methods once they are stable.
