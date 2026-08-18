---
paths:
  - "shaders/**"
  - "assets/shaders/**"
  - "build.rs"
---

# Shader System

## Compiling Shaders

Shaders are automatically compiled during `cargo build`. The build system compiles all shader files from `shaders/`
directory to `assets/shaders/` using glslc from VulkanSDK.

## Shader Source Files

Shader source files are located in `shaders/`:

- `vertex.vert` -> `assets/shaders/vert.spv`
- `fragment.frag` -> `assets/shaders/frag.spv`
- `gbufferVertex.vert` -> `assets/shaders/gbufferVert.spv`
- `gbufferFragment.frag` -> `assets/shaders/gbufferFrag.spv`
- `rayQueryShadow.comp` -> `assets/shaders/rayQueryShadow.spv`
- etc.

## Pass Manifest (`shaders/passes.toml`)

`shaders/passes.toml` is the only hand-written pass definition. Each `[pass.<name>]` lists its `stages`
(source file names; the stage is derived from the extension) and `sets` (set index -> role: `frame` = 0,
`material` = 1, `object` = 2, `local` = pass-owned). `crates/thyllore-vulkan-core/build.rs` validates the file
(missing source, orphan shader not referenced by any pass, bad stage composition, role/set convention) and
generates `PassId`, `PassShaders` constants and `ALL_PASSES` into `$OUT_DIR/pass_manifest.rs`.

- Create pipelines with `PipelineBuilder::from_pass(&FLAME_RESOLVE)` / `RRPipeline::new_compute_with_push_constants(.., &RAY_QUERY_SHADOW, ..)`.
- Declare layouts with `ReflectedLayoutSpec::shared(SetRole::Frame)` or `ReflectedLayoutSpec::local(&TONEMAP)`.
- A pass with a `local` set must be mapped in `descriptor/pass_layouts.rs` (`local_layout_spec`, exhaustive match).
- Adding a shader = GLSL + `passes.toml` entry + `*DescriptorSet` + one `pass_layouts.rs` line.

## Shader Modifications

After editing shaders in `shaders/`, the build system automatically compiles them to `assets/shaders/` directory during
`cargo build`. The application loads compiled shaders from `assets/shaders/` directory.

## Reference Documentation

The `memo.txt` file contains useful reference links for:

- Vulkan coordinate systems and layout qualifiers
- glTF mesh loading examples
- FBX property access patterns
- Animation and skinning techniques
