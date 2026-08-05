---
paths:
  - "src/**"
  - "crates/**"
  - "shaders/**"
---

# Robust Coding Guidelines

Design principles distilled from industry-standard coding guidelines for bug prevention
and robust software design. These rules apply to all source code in this project.

**Sources**:
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [C++ Core Guidelines (Stroustrup & Sutter)](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [Epic Games Unreal Engine Coding Standard](https://dev.epicgames.com/documentation/en-us/unreal-engine/epic-cplusplus-coding-standard-for-unreal-engine)
- [Apple Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Microsoft Rust Guidelines](https://microsoft.github.io/rust-guidelines/)

## 1. Make Invalid States Unrepresentable

Use the type system to ensure illegal combinations cannot exist at compile time.

- Use `enum` variants instead of boolean flags or optional fields that create
  impossible combinations.
- Wrap raw primitives (`f32`, `u32`, `String`) in named types when they carry
  domain meaning (e.g., `BoneId(u32)`, `KeyframeId(u32)`).
- Prefer `Option<T>` for "may not exist" and `Result<T, E>` for "may fail".
  Do not mix the two for the same failure mode.

```rust
// Good: enum prevents invalid combinations
enum AccelerationState {
    Ready(AccelerationStructure),
    NotAvailable,
}

// Bad: Option<T> + bool create 4 states, only 2 are valid
struct RayTracing {
    accel: Option<AccelerationStructure>,
    is_valid: bool,
}
```

*Source: C++ Core Guidelines I.4, Unreal Engine Coding Standard (enum class over bool)*

## 2. Validate at Boundaries, Trust the Interior

Validate inputs at system boundaries (file I/O, FFI, user input, external APIs).
Internal functions may assume valid data passed from validated boundaries.

- Boundary functions: validate and convert to strong types immediately.
- Internal functions: accept strong types, skip redundant validation.
- FFI boundaries (`unsafe`): validate every pointer, size, and invariant.

```rust
// Boundary: validate and convert
pub fn load_fbx(path: &str) -> Result<FbxModel, LoadError> {
    let data = std::fs::read(path)?;  // boundary validation
    parse_fbx_data(&data)             // internal: trusts validated data
}
```

*Source: C++ Core Guidelines I.6, Google C++ Style Guide (input validation)*

## 3. Consistent Error Handling

All fallible operations return `Result<T, E>`. Do not mix panics, `Option`, and
`Result` for similar failure modes within the same subsystem.

- Use `Result` for recoverable errors (I/O, parsing, Vulkan calls).
- Use `panic` / `unreachable!` only for programming errors (invariant violations).
- Use `#[must_use]` on types where ignoring the return value is a bug.

*Source: Google AIP-193, Rust API Guidelines C-MUST-USE*

## 4. Resource Lifetime Tied to Ownership (RAII)

Every resource (GPU handle, buffer, file handle) must be released through `Drop`.
Never scatter allocation and deallocation across unrelated functions.

- Encapsulate resource acquisition and release in a single struct with `Drop`.
- Do not store borrowed references (`&T`) in struct fields; use owned types or
  smart pointers (`Rc`, `Arc`).
- Keep `&mut` borrows as short as possible to prevent aliasing conflicts.

*Source: Google C++ Style Guide (RAII), C++ Core Guidelines R.1, Unreal Engine
Smart Pointer conventions*

## 5. Functions: Small, Focused, Clear Contract

- Keep functions to 80-120 lines. Split longer functions.
- Input parameters before output parameters.
- Prefer return values over mutable output parameters.
- Limit parameters to 3-4. Group related values into a struct when exceeding.

```rust
// Good: returns value, clear contract
fn compute_bone_transform(skeleton: &Skeleton, bone_id: BoneId) -> Matrix4<f32> { .. }

// Bad: mutable output parameter hides intent
fn compute_bone_transform(skeleton: &Skeleton, bone_id: BoneId, out: &mut Matrix4<f32>) { .. }
```

*Source: Google C++ Style Guide (function design), Apple Swift API Guidelines
(clarity at point of use)*

## 6. Exhaustive Matching

Use `match` with all variants explicitly handled. Avoid catch-all `_ =>` when
new variants could be added later, so the compiler catches missing cases.

```rust
// Good: compiler catches new variants
match interpolation {
    InterpolationType::Linear => { .. }
    InterpolationType::Bezier => { .. }
    InterpolationType::Stepped => { .. }
}
```

*Source: Rust API Guidelines C-SEALED, Safety-Critical Rust Coding Guidelines*

## 7. No Shadowed Variables in Nested Scopes

Do not redeclare a variable with the same name in an inner scope when the outer
variable is still logically relevant. Shadowing causes accidental use of the
wrong value.

*Source: Unreal Engine Coding Standard (shadowed variables), C++ Core Guidelines
ES.12*

## 8. Boolean Parameters Are Design Smells

Replace boolean parameters with enum types. Booleans at call sites are
unreadable and prone to transposition errors.

```rust
// Good: intent is clear at call site
render_mesh(mesh, RenderPass::GBuffer);

// Bad: what does `true` mean?
render_mesh(mesh, true);
```

*Source: Unreal Engine Coding Standard, Apple Swift API Guidelines*

## 9. Fail Fast, Do Not Corrupt State

When a precondition is violated, fail immediately rather than continuing with
corrupted state. Use `debug_assert!` for development-time checks and
`assert!` for invariants that must hold in release builds.

*Source: C++ Core Guidelines I.6, Google C++ Style Guide (assertions)*

## 10. Closure Captures and Deferred Execution

When closures are stored or executed later (callbacks, event handlers), use
explicit captures. Avoid capturing `&mut self` or large scopes implicitly,
as deferred execution can cause dangling references or unintended mutations.

*Source: Unreal Engine Coding Standard (lambda captures), C++ Core Guidelines
F.52*

## 11. Contiguous Memory for Performance-Critical Data

For data processed in bulk (vertices, keyframes, bone transforms), use
contiguous storage (`Vec<T>`) rather than scattered allocations. Cache-friendly
layout is critical for rendering and animation performance.

*Source: Unreal Engine (contiguous arrays 5x faster), Bevy Engine (archetype
storage)*

## 12. Test Public Contracts, Not Implementation Details

Tests should verify the public API behavior, not internal state. This makes
tests stable when implementation changes and ensures the contract is correct.

*Source: Google Testing Blog (Abseil TotW #135), C++ Core Guidelines T.2*

## 13. No Sentinel Extremes — Express "No Value Yet" in the Control Flow

Never seed an accumulator with a magic extreme (`1e30`, `f32::MAX`, `-1`,
`i32::MIN`) to make a later `min`/`max`/comparison work. The value can escape
the loop unchanged and reach code that treats it as real, and every reader has
to prove that it cannot. State "nothing seen yet" in the control flow instead:
a `bool` guard, the first iteration seeding the value, `Option<T>` in Rust, or
a meaningful fallback that is correct on its own.

```glsl
// Bad: 1e30 is not a radius; if no node is sampled it silently escapes
float tightScale = 1e30;
for (...) { tightScale = min(tightScale, nodeScale); }
tightScale *= 0.64;

// Good: the "no sample" case is a branch, and the initial value is meaningful
bool hasTight = false;
float tightScale = clipScale;          // fallback: the conservative bound
for (...) {
    tightScale = hasTight ? min(tightScale, nodeScale) : nodeScale;
    hasTight = true;
}
if (hasTight && tightScale > clipScale) { ... }
```

The same applies to "impossible" indices and IDs: use `Option<Id>` (Rust) or an
explicit `hasX` flag (GLSL), not `u32::MAX` / `-1`.

*Source: C++ Core Guidelines ES.20 / F.60 (no magic sentinels), Rust API
Guidelines (`Option<T>` over in-band signalling)*
