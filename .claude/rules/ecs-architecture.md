---
paths:
  - "src/**"
  - "crates/**"
---

# ECS Architecture

**IMPORTANT:** MUST follow these architecture rules when adding files, code, or planning new features.

This project uses an Entity-Component-System (ECS) architecture. The design follows these principles:

## Design Philosophy

1. **Data-Behavior Separation**: Data structures hold only data; all behavior is implemented as system functions
2. **Composition over Inheritance**: Build complex objects by combining simple components
3. **Single Responsibility**: Each component/system has one clear purpose
4. **Components/Systems Directory Separation**: Within a feature domain, data types and logic are separated
   into `components/` and `systems/` subdirectories (inspired by Unity DOTS and Flecs module patterns)
5. **Small, Atomic Components**: Prefer multiple small components over one large component.
   Components that are always accessed together may share a struct, but data accessed by different systems
   should be in separate components (Flecs design principle: reduces cache misses and unnecessary data loading)
6. **Module Independence**: Feature modules depend on shared component types, not on each other's system
   functions. Inter-module communication happens through components, resources, and events — never through
   direct system-to-system calls across module boundaries (Flecs module design)

## Directory Structure

### Core ECS Layer (`src/ecs/`)

```
src/ecs/
├── component/           # Component definitions (data attached to entities)
├── resource/            # Global dynamic state (changes per frame), one file per resource
├── systems/             # System functions (behavior/logic), one file per domain
│   ├── phases/          # Phase coordinators (execution order) and event dispatchers
│   ├── flame/, water/   # One directory per effect (spawn, time, preset, pick, history, ...)
│   ├── animation/       # Animation pipeline (collect, evaluate, apply, post_process)
│   └── frame_runner.rs  # run_frame: the phase sequence
├── events/              # UIEvent definitions and UIEventQueue
├── query/               # Query builder, filters, tuple fetch
├── storage/             # Sparse set component storage
├── registry/            # Component registry (type info)
├── context.rs           # EcsContext: World + assets + per-frame inputs for phases
├── world.rs             # World container for entities and resources
└── mod.rs
```

Component and resource types that an effect exposes as parameters (flame, water) are declared once in
`crates/thyllore-effect-core` with `declare_scene_format!`; `src/ecs/component/` only wraps them.

### Domain ECS Modules

Feature-specific domains that have their own data/logic separation follow the `components/` and `systems/`
pattern within their module directory. The editable animation domain lives in the pure crate
`crates/thyllore-anim-core` (re-exported as `src/animation.rs`); it has no ECS dependency.

```
crates/thyllore-anim-core/src/editable/
├── components/          # Data types (structs, enums, type aliases)
│   ├── keyframe.rs      # EditableKeyframe, BezierHandle, TangentType, etc.
│   ├── curve.rs         # PropertyCurve, PropertyType (data + accessors only)
│   ├── track.rs         # BoneTrack (data + accessors only)
│   ├── clip.rs          # EditableAnimationClip (data + accessors only)
│   ├── clip_file.rs     # Clip file record
│   ├── blend.rs         # BlendMode, EaseType
│   ├── clip_instance.rs # ClipInstance
│   ├── clip_group.rs    # ClipGroup, ClipGroupId
│   ├── source_clip.rs   # SourceClip
│   ├── keyframe_copy.rs # Clipboard record
│   ├── snap_settings.rs # Snap settings
│   ├── mirror.rs        # MirrorAxis, MirrorMapping (data types only)
│   └── mod.rs
│
├── systems/             # Logic and operations (pure functions)
│   ├── curve_ops.rs     # curve_sample, curve_add_keyframe, curve_sort_keyframes
│   ├── clip_ops.rs      # clip-level edits
│   ├── tangent.rs       # sample_bezier, apply_auto_tangent, apply_tangent_by_type
│   ├── clip_convert.rs  # from_animation_clip, to_animation_clip
│   ├── bake.rs          # bake constraints / spring bones into keyframes
│   ├── mirror.rs        # build_mirror_mapping, mirror_keyframes
│   ├── snap.rs          # snap_time, compute_snap_threshold_time
│   ├── manager.rs       # EditableClipManager (I/O and lifecycle management)
│   └── mod.rs
│
└── mod.rs
```

The same split is used by `crates/thyllore-effect-core` (`flame/`, `water/`: `effect/` data,
`analytic/` pure math, `gpu/` UBO structs, `presets.rs`, `settings.rs`).

### What Goes in `components/`

- Struct and enum definitions (data types)
- `new()`, `Default`, `Clone`, `Serialize`/`Deserialize` derives
- Simple accessors: `get_*()`, `set_*()`, field access helpers
- Computed properties that read but do not mutate (e.g., `effective_duration()`, `is_empty()`)
- Collection helpers on owned data (e.g., `contains_instance()`, `add_instance()` for Vec operations)

### What Goes in `systems/`

- Pure functions that compute, transform, or sample data (e.g., `curve_sample`, `sample_bezier`)
- Functions that mutate state across multiple data structures (e.g., `recalculate_tangent_at`)
- Format conversion functions (e.g., `from_animation_clip`, `to_animation_clip`)
- I/O operations (e.g., `save_to_file`, `load_from_file`)
- Functions that coordinate operations across multiple components

## Core Concepts

### Components

Data-only structs attached to entities. Located in `ecs/component/`.

### Resources

Global state that changes per frame. Located in `ecs/resource/`. **Only use for dynamic data.**

Resources correspond to **singleton components** in other ECS frameworks:
- Flecs: singletons (component added to its own entity)
- Unity DOTS: singleton components (`GetSingleton<T>()`)
- EnTT: context variables (`registry.ctx()`)

### Systems

Pure functions that operate on components and resources. Located in `ecs/systems/`.

```rust
pub fn camera_rotate(camera: &mut Camera, delta: Vector2<f32>);
pub fn animation_update(playback: &mut AnimationPlayback, registry: &mut AnimationRegistry, dt: f32);
```

### Spawning

There is no `bundle/` directory. Entities are spawned by a `spawn_*` system in the owning domain
(`src/ecs/systems/flame/spawn.rs`, `water/spawn.rs`, `mesh_systems.rs`, ...) which inserts the component set.

## Phase Pipeline

Systems execute in a fixed phase order, defined in `src/ecs/systems/phases/`. Each phase is a coordinator
function that calls the appropriate systems in sequence.

```
run_frame()  (src/ecs/systems/frame_runner.rs, called from App::update)
├── batch_run_tick()               # Batch mode state machine
├── run_input_phase()              # Input handling, gizmo interaction (EcsContext)
├── run_transform_phase_ecs()      # Transform propagation (EcsContext)
├── run_timeline_phase()           # Timeline / clip schedule advance
├── run_inference_actor_phase()    # ML inference actors
├── run_animation_phase_ecs()      # Animation evaluation, blending
├── run_animation_phase_gpu()      # Skinning / vertex upload
├── run_onion_skin_phase()         # Ghost frame generation
├── run_transform_phase_gpu()      # Object UBO upload
└── run_render_prep_phase()        # Gizmo mesh building, render data collection

run_event_dispatch_phase()         # UI event processing, deferred actions;
                                   # called from src/platform/events.rs after the UI is built
```

### Phase Design Principles (from Flecs, Unity DOTS, Bevy)

- **Phases are sequential**: Each phase completes before the next begins
- **Systems within a phase may have internal ordering**: Expressed through call sequence in the phase
  coordinator function
- **Animation completes before Transform propagation**: Standard in all major engines
  (Bevy: `.before(TransformSystems::Propagate)`, Unity DOTS: animation in SimulationSystemGroup
  before TransformSystemGroup)
- **Event dispatch is last**: Deferred structural changes (entity creation/destruction, component
  add/remove) are processed at the end of the frame, similar to Unity DOTS EntityCommandBuffer
  pattern and Flecs sync points

### Adding New Phases

When adding a new phase:
1. Create a coordinator function in `phases/`
2. The function receives `FrameContext` and calls system functions in order
3. Add the call in `run_frame()` at the correct position
4. Document ordering dependencies (what must complete before this phase)

## Query Pattern

Use query functions instead of storing entity IDs:

```rust
pub fn query_grid(world: &World) -> Option<Entity>;
pub fn query_selectable_entities(world: &World) -> Vec<Entity>;
```

For frequently called queries, consider caching results in a resource to avoid repeated iteration
(analogous to Flecs cached queries vs uncached queries).

## RefCell-Based Interior Mutability

```rust
let camera = app.resource::<Camera>();           // ResRef<Camera> (immutable)
let mut camera = app.resource_mut::<Camera>();   // ResMut<Camera> (mutable)
```

## Adding New Scene Objects

1. Define components in `ecs/component/` (effect parameters: declare them in `thyllore-effect-core`)
2. Add a marker component for queries
3. Implement system functions in `ecs/systems/<domain>/`
4. Add a `spawn_*` system that inserts the component set, and call it from the event dispatcher or
   initialization

## Adding New Domain Features

When a feature domain has enough data types and logic to warrant separation:

1. Create `<domain>/components/` for data types
2. Create `<domain>/systems/` for logic functions
3. Keep `<domain>/mod.rs` for re-exports only
4. Data types must not depend on system functions
5. System functions import and operate on data types

## Layer Boundary Rules

**IMPORTANT:** ECS business logic MUST live inside `src/ecs/systems/`. Code outside `src/ecs/` (especially
`src/platform/`, `src/app/`) must NOT contain ECS domain logic.

### Dependency Direction

Following Flecs module design, dependencies flow in one direction:

```
Platform Layer (src/platform/, src/app/)
    │  reads resources, sends events, calls phase entry points
    ▼
ECS Systems Layer (src/ecs/systems/)
    │  calls pure domain functions, operates on components/resources
    ▼
Domain Layer (crates/thyllore-anim-core, thyllore-effect-core, thyllore-model-core, ...)
    │  pure data types and pure functions, no World dependency
    ▼
Core Types (src/ecs/component/, src/ecs/resource/)
    data definitions only
```

**Allowed in platform layer** (`src/platform/`):
- Reading resources for UI display (immutable access)
- Sending events to `UIEventQueue`
- Calling a single ECS dispatch entry point (e.g., `run_event_dispatch_phase`)
- Platform-specific I/O (file dialogs, window management, imgui orchestration)
- Handling `DeferredAction` that requires `App`/`GUIData`

**NOT allowed in platform layer**:
- Directly calling multiple ECS system functions to process events
- Match-dispatching `UIEvent` variants to mutate `World`/`AssetStorage`
- Implementing event handler logic inline
- Using `resource_mut` or `get_component_mut` for business logic mutations

### Domain Layer Independence

Domain crates (`crates/thyllore-anim-core`, `thyllore-effect-core`, ...) must be **pure** — no dependency on
`World`, `Entity`, `AssetStorage`, or `GraphicsResources`. This enables:
- Unit testing without ECS infrastructure
- Reuse across different ECS contexts
- Clear separation between "what the data is" and "how the engine uses it"

## Event System

Events follow a **record-then-dispatch** pattern (similar to Unity DOTS EntityCommandBuffer
and Flecs deferred events):

1. **Platform layer** records events into `UIEventQueue` (immutable World access only)
2. **Event dispatch phase** processes all queued events (mutable World access)
3. **Structural changes** (entity creation/destruction) happen only during dispatch

This prevents mutation during iteration and ensures deterministic ordering.

## Bones are NOT Entities

- Bones are data within `Skeleton.bones: Vec<Bone>`
- Bones identified by `BoneId` (u32 index), not `Entity`
- Constraints reference bones via `BoneId`

This is a deliberate design choice: skeleton data is a small, structured hierarchy that benefits from
contiguous memory (Vec) rather than ECS entity overhead. Flecs also recognizes this pattern —
small structural hierarchies can use optimized storage rather than full entity relationships.

## Reference Projects

### Tier 1: Industry Standard (Production-Proven)

- **[Unity DOTS (Entities)](https://docs.unity3d.com/Packages/com.unity.entities@1.0/manual/)**
  - Archetype ECS with chunk-based memory layout
  - 3 root SystemGroups: Initialization → Simulation → Presentation
  - `UpdateBefore`/`UpdateAfter` attributes for system ordering within groups
  - EntityCommandBuffer for deferred structural changes (record → playback)
  - Singleton components for global state (equivalent to this project's Resources)
  - Baker/Authoring pattern: separates editor data from runtime ECS data
  - **Primary reference for**: phase ordering, deferred action pattern, directory layout

- **[Unreal Mass Entity](https://dev.epicgames.com/documentation/en-us/unreal-engine/mass-entity-in-unreal-engine)**
  - Plugin-per-feature (MassMovement, MassRepresentation)
  - Fragments (Components) and Processors (Systems) co-located per plugin
  - **Primary reference for**: feature-per-plugin organization

### Tier 2: High-Quality ECS Libraries

- **[Flecs](https://github.com/SanderMertens/flecs)** (C/C++, ~7k stars)
  - Archetype ECS with 8 built-in pipeline phases
  - `DependsOn` relationships for topological sort of phase ordering
  - Module design: components are shared types; systems depend only on components, not other systems
  - Small atomic components recommended (Position + Rotation + Scale, not Transform)
  - Relationship system (ChildOf, IsA) with cleanup traits
  - Observer system with lifecycle events (OnAdd, OnRemove, OnSet)
  - [Design with Flecs](https://www.flecs.dev/flecs/md_docs_2DesignWithFlecs.html) — essential reading
  - **Primary reference for**: module boundary design, phase pipeline, component granularity

- **[EnTT](https://github.com/skypjack/entt)** (C++, ~10k stars)
  - Sparse set ECS (vs archetype): O(1) component add/remove, fast single-component queries
  - Registry as coordinator (not monolithic container)
  - View (flexible, no ownership) vs Group (high-performance, owns component pools)
  - Signal system: on_construct, on_update, on_destroy per component type
  - Context variables for singleton/resource data
  - No prescribed project structure — library, not framework
  - **Primary reference for**: sparse set trade-offs, signal/event patterns, minimal API design

- **[Bevy Engine](https://github.com/bevyengine/bevy)** (Rust, ~44k stars)
  - Archetype ECS with Schedule + SystemSet for execution ordering
  - Feature-per-crate, data and systems co-located within each crate
  - Animation pipeline: 6 chained systems in PostUpdate, before TransformSystems::Propagate
  - Plugin architecture for modular feature registration
  - **Primary reference for**: Rust ECS patterns, animation pipeline phases, system ordering

### Tier 3: Specialized References

- **[Fyrox](https://github.com/FyroxEngine/Fyrox)** (Rust)
  - Scene graph + Generational Arena (Pool), not pure ECS
  - Animation Blending State Machine (ABSM): layers + states + transitions + blend nodes
  - Each layer has its own state machine; all layers blend to final pose
  - Plugin system with hot-reload support
  - **Primary reference for**: animation blending architecture (ABSM, layers)

- **[Hecs](https://github.com/Ralith/hecs)** (Rust, ~1.2k stars)
  - Minimal archetype ECS. Clean, small codebase
  - **Primary reference for**: understanding core ECS internals

### Architecture Comparison

| Feature | This Project | Flecs | Unity DOTS | EnTT | Bevy |
|---------|-------------|-------|------------|------|------|
| Storage | Sparse Set (`src/ecs/storage/`) | Archetype | Archetype (Chunk) | Sparse Set | Archetype |
| Data/Logic Split | Separate dirs | Separate modules | Flat per feature | Free-form | Mixed per crate |
| Organization Unit | Directory | Module | Directory | Free-form | Crate |
| Phase System | Coordinator fns | DependsOn pipeline | SystemGroup hierarchy | User-defined | Schedule + SystemSet |
| Global State | Resource | Singleton | Singleton Component | Context Variable | Resource |
| Events | UIEventQueue | Observer + emit | ECB + SystemGroup | Signal (sigh/sink) | Event\<T\> + EventReader |
| Deferred Changes | DeferredAction | Sync point flush | EntityCommandBuffer | - | Commands |

### Key Patterns Adopted from Each

| Pattern | Source | Implementation in This Project |
|---------|--------|-------------------------------|
| components/ + systems/ directory split | Flecs, Unity DOTS | `thyllore-anim-core/src/editable/components/` + `systems/` |
| Phase pipeline with explicit ordering | Flecs, Unity DOTS, Bevy | `phases/` directory with coordinator functions |
| Resources for global state | All (unanimous) | `ecs/resource/` |
| Record-then-dispatch events | Unity DOTS (ECB), Flecs (deferred) | `UIEventQueue` → `event_dispatch_phase` |
| Module depends on types only, not logic | Flecs | Platform layer reads resources, sends events only |
| Pure domain layer (no World dependency) | Bevy (per-crate), Flecs (module independence) | `thyllore-anim-core` / `thyllore-effect-core` have no ECS dependency |
| Contiguous memory for bulk data | Flecs, Bevy, Unreal | `Vec<Bone>`, `Vec<Keyframe>` for animation data |
| Animation before Transform propagation | Bevy, Unity DOTS | `run_animation_phase` before `run_transform_phase` |
