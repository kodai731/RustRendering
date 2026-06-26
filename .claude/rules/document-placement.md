# Document Placement Rules

Where to save documents (research, design, issue history) and how to name them. This rule is auto-surfaced
when writing any `.md` file (see the Write/Edit hook in `.claude/settings.json`).

## Resolve paths first

MUST resolve `${DocumentPath}`, `${ExploreHistoryPath}`, `${IssueHistoryPath}` by reading
`.claude/local/paths.md` before writing any file. Do NOT use relative paths like `../SharedData/` — agents may
have different working directories, causing files to be saved in the wrong location.

## Type → directory mapping (MUST follow)

All documents live under `${DocumentPath}/Rust_Rendering/`, one level deeper in the subdirectory that matches
the document TYPE. Never place a document directly under `${DocumentPath}/` or directly under
`${DocumentPath}/Rust_Rendering/`.

| Document type | Directory |
|---|---|
| Design / specification / decision record | `${DocumentPath}/Rust_Rendering/Design/` |
| Investigation / exploration notes | `${ExploreHistoryPath}` (`.../Rust_Rendering/ExploreHistory/`) |
| Issue history (bug + fix) | `${IssueHistoryPath}` (`.../Rust_Rendering/IssueHistory/`) |

**IMPORTANT:** Before writing ANY document, `ls ${DocumentPath}/Rust_Rendering/` first and pick the matching
type subdirectory. Do NOT infer placement from stray top-level files — they are legacy exceptions, not the convention.

## Naming

- Design / exploration docs: date prefix, like `20260315_new_file.md`.
- For a multi-file design, follow `.claude/rules/report-format.md`: create `Design/<YYYYMMDD>_<topic>/index.md`
  as the parent, with detail files in a same-named child directory
  (e.g. `Design/20260626_usd_hair_curves/index.md`).
- Issue history files: CamelCase (e.g. `ImageLayoutTransition.md`).

## Issue History specifics

- If you encounter a complex issue and resolve it, document the issue and its solution in detail at
  `${IssueHistoryPath}`.
- Prefer adding to an existing file (and recapping it) over creating many small files.
- At the top of each file, include a brief summary of the issue and its resolution for quick reading.
- MUST write in English.

## Report format

Research / design / decision md files MUST follow `.claude/rules/report-format.md` (200-char Summary,
≤300-line parent, MADR Context/Decision/Consequences for decisions).
