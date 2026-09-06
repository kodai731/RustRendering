---
paths:
  - "src/vulkanr/**"
  - "src/app/**"
  - "src/render/**"
  - "crates/thyllore-vulkan-core/**"
  - "crates/thyllore-render-core/**"
---

# Rendering Architecture

File placement is defined in `hierarchy.md`; this file describes how a frame is produced. Shader build and
naming rules are in `shaders.md`, ECS phases in `ecs-architecture.md`.

## Layers

| Layer | Location | Knows |
|---|---|---|
| Abstract render types | `crates/thyllore-render-core` | `RenderBackend` trait, `MeshId`, buffer handles, `FrameUBO` / `ObjectUBO` / `MaterialUBO`, post-process settings. No Vulkan, no ECS |
| Vulkan primitives | `crates/thyllore-vulkan-core` | `core/` (`RRDevice`, `RRSwapchain`, descriptor allocator), `command/`, `descriptor/` (reflected set layouts, one file per pass), `pipeline/` (builder from the pass manifest, cache, ray tracing), `raytracing/` (BLAS / TLAS), `render/` (`RRRender` render pass + framebuffers, depth), `resource/` (buffers, images, HDR / gbuffer / offscreen / effect buffers, `RenderTargetStorage`, `RenderTargetTransient`), `renderer/` (per-pass command helpers), `backend.rs` (`VulkanBackend: RenderBackend`). No ECS |
| App-side Vulkan glue | `src/vulkanr/` | ECS resources wrapping swapchain / sync / gbuffer (`context/resources.rs`), `pass_recording.rs` (reads `App`, calls crate helpers), `scene_renderer.rs`, `backend.rs` (`BillboardBackend` impl) |
| Frame driver | `src/app/` | `App` lifecycle, `AppData`, `ViewportState`, `begin_frame` / `update` / `render` / present |

Effect-specific pass recording, resize and descriptor updates are moving from `src/app/` and
`pass_recording.rs` into `src/ecs/systems/<effect>/` (#151) and a pass graph (#156). Do not add new
effect-specific code to `src/app/render.rs` or `pass_recording.rs`.

## Frame flow

`src/platform/events.rs::render_frame` calls three `App` methods in order:

1. `begin_frame` (`src/app/render.rs`): apply pending viewport resize (`device_wait_idle`, viewport and
   effect buffers rebuilt, descriptors rebound), wait the frame fence, `RenderTargetTransient::begin_frame`
   (recycle the frame-in-flight bucket, evict stale images), read back auto exposure and object id, acquire
   the swapchain image.
2. `update` (`src/app/update.rs`): builds `FrameContext` and runs the ECS phase pipeline (`run_frame`),
   then uploads imgui buffers.
3. `render` (`src/app/render.rs`): TLAS refresh, `prepare_post_process_targets` and
   `prepare_water_frame_targets` (acquire transient images, update the descriptor sets of this frame slot),
   `record_command_buffer`, submit, present, `FrameSync::advance`.

Descriptor sets that read a transient image exist once per frame slot (`MAX_FRAMES_IN_FLIGHT = 2`, in
`src/app/init/instance.rs`). Never update a single descriptor set that a pending command buffer may still
bind; either keep one set per frame slot or wait idle first (resize path).

## Pass order (`src/app/command_recording.rs`)

```
gbuffer → object id copy → ray query shadow → composite to HDR → onion skin
→ water (trace → caustic → scene color copy → resolve) → flame (shading + temporal)
→ bloom (downsample / upsample mips) → dof → auto exposure (histogram + average)
→ tonemap to offscreen → onion skin composite → imgui
```

Water and flame write into the HDR buffer and read it (water copies HDR to a transient scene-color image
first). Post-process passes read the previous stage through per-slot descriptors. Passes and their shader
stages / descriptor set roles are declared once in `shaders/passes.toml` and generated into
`thyllore-vulkan-core` by its `build.rs`; a new pass starts there.

## Render targets

- `RenderTargetStorage` (viewport-extent lifetime, keyed by `RenderTargetKey`): flame and water history,
  caustic accumulation. Reset on resize, destroyed with the viewport.
- `RenderTargetTransient` (pass lifetime inside a frame): dof output, bloom mips, water scene color copy
  and trace image. `acquire(TransientDesc)` returns a frame-stamped `TransientHandle`; the image layout is
  `UNDEFINED` right after acquire, so the first use must transition it. Framebuffers for transient
  attachments come from `RenderTargetTransient::framebuffer` (cached by render pass + views).
- Core attachments (HDR color, depth, gbuffer, offscreen) are owned by `ViewportState` / `RenderTargets`
  and are not pooled.
- Design and migration record: `Design/20260906_render_target_transient_design/` under
  `${RustRenderingDocPath}`.

## Contexts passed to render code

- `thyllore_vulkan_core::FrameRenderContext`: device, graphics resources, buffer registry, pipeline storage,
  image index. Immutable; what crate-level `record_*_pass` helpers take.
- `src/app/render_context.rs::RenderContext`: mutable GPU resources, builds `VulkanBackend`.
- `src/app/frame_context.rs::FrameContext`: `RenderContext` plus `World`, `AssetStorage`, time, frame slot,
  swapchain extent. What the ECS phase pipeline takes.

## Camera controls (`src/ecs/systems/camera_systems.rs`)

- Right drag: look (yaw / pitch); with WASD / QE while held: fly
- Alt + right drag: orbit around the target
- Middle (wheel) drag: pan
- Wheel: zoom toward the cursor

## Constants (`src/app/init/instance.rs`)

- `MAX_FRAMES_IN_FLIGHT = 2`
- `VALIDATION_ENABLED = cfg!(debug_assertions)`, layer `VK_LAYER_KHRONOS_validation`

## Model loading

Importers live in `crates/thyllore-importer-core` (glTF / FBX / PNG) and return model-core + anim-core
types; `src/loader/` only re-exports them. `src/app/model_loader.rs` turns the result into `AssetStorage`
entries, GPU meshes (`GraphicsResources`) and acceleration structures. Sample models live under
`assets/models/<name>/`.

## Common issues

- Validation errors: run with `RUST_LOG=debug`; messages are logged through `log_error!` and appear in
  `log/log_0.txt`. Batch runs (`--batch-scene ... --batch-screenshot ...`) are the quickest reproduction.
- Objects dropped without `destroy()`: every Vulkan wrapper logs a warning from `Drop`. Batch runs exit
  without teardown, so these warnings at the very end of a batch log are expected; in the GUI they are bugs.
- Shader compilation errors: see `shaders.md` (`glslc` from VulkanSDK, run by `thyllore-vulkan-core/build.rs`).
- Descriptor written while in use: symptom is `vkUpdateDescriptorSets` validation errors or flicker one frame
  behind; the fix is a per-frame-slot set, see "Frame flow".
