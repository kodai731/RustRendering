---
paths:
  - "shaders/**"
  - "assets/shaders/**"
  - "build.rs"
---

# Shader System

## Compiling Shaders

Shaders are compiled by `crates/thyllore-vulkan-core/build.rs` during `cargo build`: every shader file in `shaders/`
is compiled to `assets/shaders/` with glslc from VulkanSDK, verified against its GLSL declarations, and reflected.
Stale `.spv` files with no matching source are removed automatically.

## Shader Source Files

Shader source files are located in `shaders/`:

- `vertex.vert` -> `assets/shaders/vert.spv`
- `fragment.frag` -> `assets/shaders/frag.spv`
- `gbufferVertex.vert` -> `assets/shaders/gbufferVert.spv`
- `gbufferFragment.frag` -> `assets/shaders/gbufferFrag.spv`
- `rayQueryShadow.comp` -> `assets/shaders/rayQueryShadowComp.spv`
- etc.

## Pass Manifest (`shaders/passes.toml`)

`shaders/passes.toml` is the only hand-written pass definition. Each `[pass.<name>]` lists its `stages`
(source file names; the stage is derived from the extension) and `sets` (set index -> role: `frame` = 0,
`material` = 1, `object` = 2, `local` = pass-owned). `crates/thyllore-vulkan-core/build.rs` validates the file
(missing source, orphan shader not referenced by any pass, bad stage composition, role/set convention) and
generates `PassId`, `PassShaders` constants and `ALL_PASSES` into `$OUT_DIR/pass_manifest.rs`.

- Create pipelines with `PipelineBuilder::from_pass(&FLAME_RESOLVE)` / `RRPipeline::new_compute_with_push_constants(.., &RAY_QUERY_SHADOW, ..)`.
- Declare layouts with `ReflectedLayoutSpec::shared(SetRole::Frame)` or `ReflectedLayoutSpec::local(&TONEMAP)`.
- `ReflectedLayoutSpec::for_role(pass, role)` derives the spec of any pass set from the manifest alone; the golden
  test `descriptor_reflection_golden.rs` walks `ALL_PASSES` with it, so no Rust-side pass list exists.
- Adding a shader = GLSL + `passes.toml` entry + its `*DescriptorSet`.

## Generated Binding Constants (`shader_bindings`)

`thyllore-vulkan-core/build.rs` also generates `$OUT_DIR/shader_bindings.rs` from the SPIR-V reflection: one
`pub mod <pass_name>` per pass with one `pub const <NAME>: ShaderBinding { set, binding, kind, count }` per
descriptor, named from the GLSL identifier (`historySampler` -> `HISTORY_SAMPLER`). Two stages of one pass declaring
the same (set, binding) under different names is a build error. No hand-written `*_BINDING: u32` constants exist;
renaming a descriptor in GLSL breaks the Rust build at the referencing site.

- Write descriptors through `layout.writer(set).buffer(flame_resolve::FLAME, ..)` — the writer takes `ShaderBinding`
  and rejects a kind that the layout slot does not accept.
- `ReflectedLayoutSpec::with_override(shader_bindings::.., vk::DescriptorType::..)` overrides a descriptor type.
- `PipelineBuilder::descriptor_layouts(&[&ReflectedSetLayout])` / `RRPipeline::new_compute*` verify at pipeline
  creation that every (set, binding) used by the pass shaders exists in the given layouts with a matching type.

## Shader Modifications

After editing shaders in `shaders/`, the build system automatically compiles them to `assets/shaders/` directory during
`cargo build`. The application loads compiled shaders from `assets/shaders/` directory.

## Reference Documentation

The `memo.txt` file contains useful reference links for:

- Vulkan coordinate systems and layout qualifiers
- glTF mesh loading examples
- FBX property access patterns
- Animation and skinning techniques
