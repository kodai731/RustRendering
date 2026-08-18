---
paths:
  - "src/**"
  - "crates/**"
  - "shaders/**"
  - "build.rs"
---

# Comments

**IMPORTANT:** These rules apply to all source code, including shaders. They override any
habit of narrating code with comments.

## 1. Comments Are Not Required by Default

Write no comment unless the rules below demand one. A file with zero comments is a normal,
good outcome. Reviewers should never have to skip past prose to reach the code.

## 2. Make the Code Self-Explanatory Instead

Before writing a comment, ask whether the code can carry the meaning by itself. Almost always
it can:

- Name the variable or function after what it *is*, not after its type or its step number.
- Extract a named function instead of labelling a block with a comment.
- Group statements into paragraphs separated by a blank line, one paragraph per idea. The
  paragraph break is the section marker — never a `// ---` or `// =====` divider comment.

```rust
// Bad: the comment exists because the code does not say what it does
// convert to world space and clamp to the shell
let p = m * v;
let p = p.min(r);

// Good: names and a paragraph break carry the same information
let world_position = model_matrix * local_position;
let clamped_position = world_position.min(shell_radius);
```

## 3. Comment Only What the Code Cannot Say — One Line

A comment is justified only when the information genuinely does not exist in the code:

- The source of a formula or approximation (`// Abramowitz-Stegun 7.1.26.`)
- The mathematical model a block implements, when the code is the *evaluation* of a formula
  that is not recoverable from it
- A non-obvious numerical or physical constraint that would otherwise be violated by a
  plausible-looking edit (`// Monomial coefficients in h would grow as 1/d.y^2 and cancel away.`)
- An external contract the code must match (protocol, file format, hardware/API requirement)

Keep it to **one line, placed at the top of the block it explains**. If one line is not
enough, the explanation belongs in a design document under `${DocumentPath}/Rust_Rendering/`,
and the code may link to it by filename.

Doc comments (`///`) on public items follow the same limit: state what the item is, not how
it came to be.

## 4. Do Not Leave a Record of the Work in the Code

When fixing a bug or rewriting code — especially inside an AI session — do not leave comments
that describe the change, the previous behaviour, the reasoning, or the measurements that
justified it. A reader who was not present for that session cannot use them, and they rot as
soon as the code moves again.

Banned in source:

- `// Fixed: ...`, `// Changed from ... to ...`, `// Previously this used ...`
- `// This is needed because the old version broke when ...`
- Measured numbers that only justify a past decision (`// 0.85% -> 0.20% -> 0.18%`)
- Restating an alternative that was considered and rejected
- Commented-out code

That history belongs in the commit message, the design document, and
`${IssueHistoryPath}`. The code holds only the current truth.

## 5. Delete Stale Comments With the Code

When behaviour changes, the comments around it are part of the change. Update them or delete
them in the same edit. Leaving a comment that describes the previous implementation is worse
than having no comment at all.

## Checklist

- [ ] Could a better name or an extracted function remove this comment? → Remove it
- [ ] Is it a blank-line-worthy section label? → Use a blank line
- [ ] Does it describe *what changed* rather than *what is*? → Delete it
- [ ] Is it longer than one line? → Move it to a design document
- [ ] Does the code still do what the comment says? → If not, fix both
