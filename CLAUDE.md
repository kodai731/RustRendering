# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: This file must be written in English for Claude's optimal understanding.
Always respond to the user in user's Claude setting language, but keep this documentation in English.

## Goal

- this project targets Animation + Rendering Engine powered by ECS Architecture.

## Code Format

**IMPORTANT**: Must follow PROJECT_ROOT/rustfmt.toml on coding.
MUST Use ``cargo fmt`` to format after code written or before git commit.

## Project Overview

This is a Rust-based Vulkan rendering engine with support for:

- 3D model loading (glTF and FBX formats)
- Skeletal animation, node animation, and morph target animation
- Real-time rendering with Vulkan API
- ImGui integration for debugging UI

## Build and Run Commands

```bash
cargo build                                             # Standard build
cargo test                                              # Run all tests
$env:RUST_LOG="debug"; cargo run --bin thyllore-animation   # Run with debug logging
```

**Build with tests (recommended)**:

```powershell
.\build-with-tests.ps1            # Build and run tests, save results to log/log_test.txt
.\build-with-tests.ps1 -Release   # Release build
.\build-with-tests.ps1 -SkipTests # Skip tests
```

## Testing and Feature Flags

**IMPORTANT**: The `ort` crate (ONNX Runtime) included via the `ml` feature (enabled by default) has CRT initializers
that crash integration test binaries on Windows with `STATUS_ACCESS_VIOLATION`. This only affects integration tests
(`tests/*.rs`), not lib tests (`cargo test --lib`).

**Before running tests**, check `.cargo/config.toml` for test aliases and environment settings (e.g., `ORT_DYLIB_PATH`).

**How to run tests correctly**:

| Command | Description |
|---------|-------------|
| `.\build-with-tests.ps1` | Recommended. Runs lib tests (with ml) and integration tests (without ml) correctly |
| `cargo test --lib` | Lib tests only (144 tests, ml enabled, safe) |
| `cargo test --test ecs_tests --no-default-features` | Integration tests (59 tests, ml disabled, safe) |
| `cargo test --no-default-features` | All tests with ml disabled (reduces functionality but avoids crash) |

**Do NOT run**: `cargo test --test ecs_tests` (without `--no-default-features`) — this will crash.

**If a test crashes with `STATUS_ACCESS_VIOLATION`**: The cause is the `ort` (ONNX Runtime) dependency linked via the
`ml` feature. Add `--no-default-features` to exclude it. See `${IssueHistoryPath}/FbxExportReimportIssues.md`
Issue 4 for details.

## ECS Architecture

**IMPORTANT:** MUST follow the rules defined in `.claude/rules/ecs-architecture.md` for all ECS-related code.
This includes core ECS layer (`src/ecs/`), domain ECS modules (`src/animation/editable/`), and layer boundary rules.

## Single Source of Truth

- **IMPORTANT**: Always follow the Single Source of Truth principle when designing the system.
- For example, skeleton or mesh data in the timeline system, and animation curve data in the UI system, must each have a
  single authoritative source.

## Robust Coding Guidelines

- **IMPORTANT**: Follow the rules defined in `.claude/rules/coding.md` for all source code.
- These rules are derived from Google C++ Style Guide, C++ Core Guidelines, Unreal Engine Coding Standard, Apple Swift
  API Guidelines, Rust API Guidelines, and Microsoft Rust Guidelines.
- Key principles: make invalid states unrepresentable, validate at boundaries, consistent error handling, RAII, exhaustive
  matching, no boolean parameters, fail fast.

## Path

**IMPORTANT:** All `${...Path}` variables MUST be resolved by reading `.claude/local/paths.md`.
This file contains the absolute paths for this machine. Agents and subagents MUST read it before using any path variable.
Do NOT resolve relative paths manually — always use the absolute paths from `.claude/local/paths.md`.

## Document

**IMPORTANT:** All documents (research, design, issue history, explore history) MUST be saved under
`${DocumentPath}/Rust_Rendering/`. Never place documents directly under `${DocumentPath}/`.
MUST resolve `${DocumentPath}` by reading `.claude/local/paths.md` before writing any file.
Do NOT use relative paths like `../SharedData/` — agents may have different working directories, causing files to be
saved in wrong locations.

- You MUST name files with date prefix, like
```
20260315_new_file.md
```

## Issue History

**IMPORTANT:** If you encounter a complex issue and resolve it, you must document the issue and its solution in detail at
`${IssueHistoryPath}`.

File names must use CamelCase (e.g., ImageLayoutTransition.md).

Each issue must be documented in a separate file, but to avoid huge number of files, try to add issue in a existing file
and recap it.
At the top of each file, include a brief summary of the issue and its resolution to read shortly.

**IMPORTANT:** MUST write in English.

## Last Conversation

- last conversation is saved at .claude/local/last-conversation.md
- **IMPORTANT** MUST read the last conversation file and work continue.

## AnimationTraining
### Repository
repository is separated to ../AnimationModelTraining

### Large Model Storage

When downloading large ML models (HuggingFace weights, TripoSG, etc.), MUST save them to
`${LargeModelStoragePath}`. This path is symlinked to a high-capacity drive.
Do NOT download large models to the project directory or HuggingFace default cache.

### Trained Data

The trained data for the copilot curve is stored in `${SharedDataPath}/exports/`.

### Interaction Log

- If any issues occur (for example, the training collapses), report them using the Context Memory format so the training
repository can fully understand the situation.
- location at `${SharedDataPath}/log/Rendering` is rendering side, and `log/Training` is training side.