---
paths:
  - "src/**"
  - "crates/**"
---

# Directory Hierarchy and File Responsibilities

Update this file whenever the directory layout changes. Before creating a file, decide its home with the
checklist at the end and never leave it in `src/app/` "for now".

## Dependency direction

```
crates/*-core            pure domain + GPU primitives, no ECS (vulkan-core knows Vulkan, nothing knows World)
    ▲
src/ecs/                 World, components, resources, systems, phases (all engine logic lives here)
    ▲
src/app/ , src/platform/ App lifecycle, Vulkan object ownership, frame driver, window / imgui / input
```

Lower layers never import upper ones. `src/app/` may call into `src/ecs/` and `src/vulkanr/`; a crate
must never depend on `src/`.

## crates/

A crate holds code that is meaningful without the engine: pure math, domain types and their pure
operations, GPU primitives, importers and exporters, codegen used by build scripts.

- Every crate is named `thyllore-<topic>-core` (or `-api`, `-debug`, `-client`) and states its layer and
  what it must not depend on in `Cargo.toml` `description`.
- A crate never depends on `World`, `Entity`, `AssetStorage` or `App`.
- Only `thyllore-vulkan-core` (and the debug crate) may name `vk::*`. Everything else describes GPU work
  through the abstract types of `thyllore-render-core`.
- Effects (`thyllore-effect-core`) keep one directory per effect with the same split: `effect/` data,
  `analytic/` pure math mirrored by the shaders, `gpu/` UBO structs, `presets`, `settings`.
- Domain crates use the `components/` (data) and `systems/` (pure functions) split, see
  `ecs-architecture.md`.

Rule of thumb: if the code needs neither `World` nor `vk::*`, it belongs in a crate. If it needs `vk::*`
but not `World`, it belongs in `thyllore-vulkan-core`.

### Crates as Blender addon build units

Some crates are also shipped standalone as Python wheels for the Blender addons, built with maturin
(`crate-type = ["cdylib", "rlib"]`, `python` feature, abi3-py310):

- `thyllore-ml-core` → the curve copilot addon (`blender_addon/`, wheels collected by
  `scripts/collect_wheels.{sh,ps1}`, packaged by `scripts/build_blender_addon.{sh,ps1}`).
  `thyllore-ml-api` is the ABI marker shared between that wheel and the addon; changing it requires a
  coordinated wheel rebuild and end-user reinstall.
- `thyllore-effect-core` → the flame and water addons (`blender_addon/effects/<effect>/`, release workflows
  `blender_<effect>_addon_release.yml`). One crate serves every effect; the addon imports the analytic
  core and presets from it so the Blender preview and the engine share the same math.

Consequences for placement:

- Anything the addon needs (analytic math, presets, settings, scene format, UBO layouts, `pybindings/`)
  must live in one of these crates or their pure dependencies, never in `src/` or `thyllore-vulkan-core`.
- These crates must stay free of ECS, Vulkan and rendering-pipeline dependencies, or the wheel stops
  building.
- The `python` feature only adds the PyO3 facade; the same code compiles as an `rlib` for the engine, so
  the Rust API is the single source of truth for both.
- The Blender addon is the only consumer of `crate-type = "cdylib"`; do not add it to other crates.
- `thyllore-grpc-client` is exercised against Blender in CI (`blender_grpc_parity.yml`) but is not a
  wheel; it stays a plain library crate.

## src/ecs/

Everything that decides what the engine does: components and resources (data only), systems (logic, one
file per domain, one directory per effect), phases (execution order and event dispatch), and the ECS core
(world, storage, query, registry, events). Rules are in `ecs-architecture.md`.

## src/hooks/

Generic hook infrastructure that lets a subsystem plug into the app lifecycle without being named by
`src/app/`. `effect.rs` holds the effect hook (setup, prepare frame, viewport resize, destroy) and the list
that runs them in subscription order. A hook file describes a lifecycle contract only; it never names a
concrete effect.

## src/effect/

The one place that subscribes the effects (`subscription.rs`): it lists the hook constants of flame, water
and any future effect. `src/app/` runs the hooks generically and never names an effect; an effect's own
systems (`src/ecs/systems/<effect>/`) implement the hook and own the effect's GPU state as an ECS resource.
Adding an effect means adding its hook constant to `subscription.rs`, nothing in `src/app/`.

## src/platform/

Window, input, imgui orchestration and the UI windows. Reads resources, records `UIEvent`s, calls one
dispatch entry point. Contains no business logic and no Vulkan commands beyond imgui rendering.

## src/vulkanr/

App-side Vulkan glue: ECS resources that wrap swapchain, sync objects and core attachments, per-frame pass
recording that has to read `App`, and implementations of app-side backend traits. Anything here that turns
out to need only device and handles moves down into `thyllore-vulkan-core`.

## src/render/

The app-side extension of the abstract render backend: traits whose signatures must mention an ECS
resource, plus re-exports of `thyllore-render-core` handle types. It is not a renderer and must stay small.

## src/app/

The application shell. It owns the Vulkan instance and device, the `AppData` aggregate and the viewport,
and drives one frame. It is the only place that sees `App` as a whole.

Belongs here:
- `App` construction and teardown
- ownership containers whose lifetime is the app or the viewport (core attachments, the render target
  storage and transient pools)
- the frame driver: begin frame, update, record, submit, present, swapchain recreation, screenshot
- context structs that bundle `App` fields for callees
- wiring that must touch several subsystems at once (a resize fan-out, rebinding after a resize)
- calls into the hook infrastructure of `src/hooks/` (setup, prepare frame, viewport resize, destroy)
  without naming an effect
- core post-processing passes (tonemap, auto exposure, dof, bloom) as one concept under
  `src/app/post_process/`: pipeline creation, resize rebinding and per-frame target acquisition live
  together there. These are engine passes, not effects, so `App` calls them directly

Does not belong here:
- ECS domain logic (querying, mutating components, deciding what an effect does) → `src/ecs/systems/`
- Vulkan helpers that need only device and handles (barriers, render passes, copies) → `thyllore-vulkan-core`
- pure math or analytic code → `thyllore-math-core`, `thyllore-effect-core`
- per-effect pass recording, resize or descriptor updates → `src/ecs/systems/<effect>/`
- UI drawing → `src/platform/ui/`

A function that takes `&mut App` only to read a few fields is misplaced: pass those fields in and put it
where the checklist says.

Effects own their GPU state: the buffers and per-frame handles of an effect are an ECS resource
(`src/ecs/resource/<effect>_render_targets.rs`), and creation, resize, per-frame acquisition and destroy are
systems in `src/ecs/systems/<effect>/render_targets.rs` exposed as an effect hook subscribed in
`src/effect/subscription.rs`. `src/app/` never enumerates effects (this mirrors bevy's `TextureCache` + per-effect `prepare_*` systems and Unreal's RDG +
per-feature `AddPass`).

## Other src/ directories

- `src/scene/` — scene file format, load / save, clip io (serde + world apply, no rendering)
- `src/asset/` — CPU-side model asset storage
- `src/debugview/` — `App` methods that dump GPU images or buffers for a debugging session. CPU mirrors of
  shader math go to `thyllore-render-debug` instead
- `src/ml/` — inference thread, feedback, licensing worker
- `src/logger/` — logger and message buffer
- `src/loader/`, `src/exporter/`, `src/grpc/`, `src/math/`, `src/animation.rs` — thin `pub use` shims over
  the crates; add nothing else there
- `src/bin/` — standalone tools

## The three "render" places

- `crates/thyllore-render-core` — how systems talk about GPU work without naming Vulkan: the backend trait,
  handles, UBO layouts, post-process settings.
- `crates/thyllore-vulkan-core` (`backend.rs`, `renderer/`) — the Vulkan implementation of that trait and the
  per-pass command helpers that take an immutable frame render context.
- `src/render/` and `src/vulkanr/` — the app-side extension of the trait (needs ECS resource types) and its
  Vulkan implementation, plus pass recording that reads `App`.
- `src/app/render.rs` — the frame driver only: begin and end of a frame, swapchain-level concerns.

Frame contexts, innermost to outermost: the crate-level immutable render context (device, resources,
pipelines, image index) → the app render context (mutable GPU resources, builds the backend) → the app frame
context (adds `World`, assets, time, frame slot) used by the ECS phases.

## Render target ownership

- Lender (engine, in the viewport): Storage for images that must survive a frame (history, accumulation;
  viewport-extent lifetime, keyed by purpose, reset on resize) and Transient for images that live inside one
  frame (described by extent / format / usage, handed out as frame-stamped handles, recycled per
  frame-in-flight bucket).
- Borrower (the pass or effect): asks the lender each frame, keeps what it borrowed in its own state
  (post-process targets under `src/app/post_process/`, effect resources under `src/ecs/resource/`), and keeps
  one descriptor set per frame slot for anything transient.
- Core attachments (HDR, depth, gbuffer, offscreen) are owned by the viewport and never pooled.

## Where does a new file go?

1. Needs neither `World` nor `vk::*` → a crate.
2. Needs `vk::*` but not `World` or `App` → `thyllore-vulkan-core` (`resource/` for objects, `renderer/` for
   command helpers, `descriptor/` for sets).
3. Needs `World` (query, mutate, decide) → `src/ecs/systems/`, one file per domain or one directory per
   effect. Data types go to `src/ecs/component/` or `src/ecs/resource/`.
4. Needs `App` because it wires several subsystems or owns app-lifetime Vulkan objects → `src/app/`.
5. Needs `App` only to read a handful of fields → not `src/app/`; pass those fields in and apply rules 1–3.
6. Draws imgui → `src/platform/ui/`.
7. Dumps GPU data for debugging → `src/debugview/` (needs `App`) or `thyllore-render-debug` (CPU mirror).
