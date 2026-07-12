---
name: check-coding
description: Check if code follows the robust coding guidelines defined in .claude/rules/coding.md. Use when reviewing code quality, bug prevention patterns, and design robustness. Examples: "unwrap がないか確認", "コーディング規約チェック".
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Agent
argument-hint: "[file-or-directory]"
---

# Robust Coding Guidelines Compliance Check

Check the specified files or recent changes against the 12 robust coding rules
defined in `.claude/rules/coding.md`.

## Usage

```
/check-coding
/check-coding src/ecs/systems/animation_systems.rs
/check-coding src/exporter/
```

## Target

If `$ARGUMENTS` is provided, check those files or directories.

If no arguments are given:
1. Check recently modified `.rs` files under `src/` (git diff / git status)
2. **Additionally**, run mechanical pattern checks (Rules 3, 5, 6, 8) across
   **all** `.rs` files under `src/` to catch violations in unchanged files.
   This is essential because coding violations (e.g., oversized functions,
   `.unwrap()` calls) can exist in files that haven't been recently modified.

## Rules to Verify

MUST read `.claude/rules/coding.md` first to get the full rule definitions.

### Rule 1: Make Invalid States Unrepresentable

- [ ] Enums used instead of `Option<T>` + bool flag combinations
- [ ] Raw primitives (`f32`, `u32`, `String`) wrapped in named types when they
      carry domain meaning (e.g., `BoneId(u32)`, `KeyframeId(u32)`)
- [ ] No struct fields that create impossible state combinations

**Pattern to search:**
```rust
// Suspicious: Option field + bool field in same struct
struct Foo {
    value: Option<Bar>,
    is_valid: bool,  // ← can contradict Option state
}
```

### Rule 2: Validate at Boundaries, Trust the Interior

- [ ] Boundary functions (file I/O, FFI, user input) validate inputs
- [ ] Internal functions accept strong types without redundant validation
- [ ] FFI boundaries validate pointers, sizes, and invariants

### Rule 3: Consistent Error Handling

- [ ] Fallible operations return `Result<T, E>`
- [ ] `panic` / `unreachable!` used only for programming errors
- [ ] No mixing of `Option` and `Result` for the same failure mode
- [ ] `#[must_use]` on types where ignoring the return value is a bug

**Pattern to search:**
```rust
// Suspicious: unwrap() in non-test code
value.unwrap();
```

### Rule 4: Resource Lifetime (RAII)

- [ ] GPU handles, buffers, file handles released through `Drop`
- [ ] No scattered alloc/dealloc across unrelated functions
- [ ] No stored borrowed references (`&T`) in struct fields
- [ ] `&mut` borrows kept as short as possible

### Rule 5: Small Focused Functions

- [ ] Functions are 80-120 lines maximum
- [ ] Return values preferred over mutable output parameters
- [ ] Parameters limited to 3-4 (grouped into struct when exceeding)

### Rule 6: Exhaustive Matching

- [ ] `match` expressions handle all variants explicitly
- [ ] No catch-all `_ =>` when new variants could be added later

**Pattern to search:**
```rust
// Suspicious: catch-all on non-exhaustive enum
match value {
    Variant::A => { .. }
    _ => { .. }  // ← hides future variants
}
```

### Rule 7: No Shadowed Variables

- [ ] No redeclaration of a variable with the same name in an inner scope
      when the outer variable is still logically relevant

### Rule 8: No Boolean Parameters

- [ ] Boolean parameters replaced with enum types
- [ ] Call sites are readable without checking the function signature

**Pattern to search:**
```rust
// Suspicious: bool parameter
fn do_something(flag: bool) { .. }
fn do_something(x: i32, y: i32, verbose: bool) { .. }
```

### Rule 9: Fail Fast

- [ ] `debug_assert!` for development-time precondition checks
- [ ] `assert!` for invariants that must hold in release builds
- [ ] No silent continuation with corrupted state

### Rule 10: Explicit Closure Captures

- [ ] Closures stored or executed later use explicit captures
- [ ] No implicit capture of `&mut self` or large scopes
- [ ] No risk of dangling references from deferred execution

### Rule 11: Contiguous Memory

- [ ] `Vec<T>` used for bulk data (vertices, keyframes, bone transforms)
- [ ] No scattered allocations for performance-critical data paths

### Rule 12: Test Public Contracts

- [ ] Tests verify public API behavior, not internal state
- [ ] Tests remain stable when implementation changes

## Check Process

1. Read `.claude/rules/coding.md` for full rule definitions
2. Determine target files from `$ARGUMENTS` or recent git changes
3. Launch **two parallel agents** using the Agent tool:

### Agent A: Static Pattern Check (expert-explore)

Scan for mechanical violations that can be detected by pattern matching.
**Scope**: Search across ALL `.rs` files under `src/`, not just target files.
These rules are cheap to check mechanically and catch long-standing violations.

- Rule 1: Search for `Option<.*>` + `bool` field pairs in structs
- Rule 3: Search for `.unwrap()` in non-test code (`src/` excluding `tests`)
- Rule 5: Count function body lines for all `fn` definitions, flag > 120 lines
- Rule 6: Search for `_ =>` in match expressions
- Rule 7: Search for `let` bindings that shadow outer scope variables
- Rule 8: Search for function signatures with `bool` parameters
- Do NOT modify any code. Report findings only.

### Agent B: Design Pattern Check (expert-explore)

Review code for design-level violations requiring understanding of context.
**Scope**: Focus on target files (from arguments or recent changes), but also
spot-check high-risk areas (`src/platform/`, `src/app/`, `src/render/`) for
Rules 2 and 4.

- Rule 1: Identify structs with impossible state combinations
- Rule 2: Check if boundary functions validate inputs properly
- Rule 4: Identify resources not released through `Drop`
- Rule 9: Check if precondition violations are caught early
- Rule 10: Identify stored closures with implicit large-scope captures
- Rule 11: Check if bulk data uses contiguous storage
- Rule 12: Review test files for internal-state testing
- Do NOT modify any code. Report findings only.

4. Merge results from both agents
5. Report all violations with file paths and line numbers

## Output Format

```
## Coding Guidelines Check Results

### Target
- Files checked: [list of files]

### Summary
- Total violations: N
- Critical (Rules 1, 3, 4): N
- Warning (Rules 5, 6, 7, 8): N
- Info (Rules 2, 9, 10, 11, 12): N

### ❌ Violations

#### Rule 1: Make Invalid States Unrepresentable
- src/foo/bar.rs:45 — `Option<Handle>` + `is_valid: bool` creates 4 states, only 2 valid
  Suggestion: Use `enum HandleState { Valid(Handle), Invalid }`

#### Rule 5: Small Focused Functions
- src/foo/baz.rs:100-250 — `process_data()` is 150 lines
  Suggestion: Extract helper functions for each logical step

#### Rule 8: No Boolean Parameters
- src/foo/qux.rs:30 — `render(mesh: &Mesh, wireframe: bool)`
  Suggestion: Use `enum RenderMode { Solid, Wireframe }`

### ✅ Compliant
- Rule 2: All boundary functions validate inputs
- Rule 4: All GPU handles released through Drop
- Rule 11: Bulk data uses Vec<T>
- Rule 12: Tests verify public API behavior

### Recommendations
- [Priority fixes and suggested refactoring]
```

## Important Rules

- Always read `.claude/rules/coding.md` before starting the check.
- Report violations with concrete file paths and line numbers.
- Classify severity: Critical (correctness risk), Warning (maintainability risk), Info (style improvement).
- Do NOT modify any code. This skill is read-only analysis.
- Respond to the user in Japanese. Write the check results in English.
