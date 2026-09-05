"""GLSL include expansion shared by the Blender effect exporters."""

import os
import re


def resolve_layout_macros(line: str, defines: dict[str, str]) -> str:
    if not re.match(r'^\s*layout\s*\(', line):
        return line
    return re.sub(r'\b[A-Za-z_]\w*\b', lambda m: defines.get(m.group(0), m.group(0)), line)


def resolve_include(including_path: str, included: str, repo_root: str) -> str:
    """Mirror glslc lookup: relative to the including file first, then the shaders/ root (-I)."""
    relative = os.path.normpath(os.path.join(os.path.dirname(including_path), included))
    for candidate in (relative, os.path.normpath(included)):
        if os.path.isfile(os.path.join(repo_root, "shaders", candidate)):
            return candidate
    raise FileNotFoundError(f"{included} (included from {including_path}) not found under shaders/")


def expand_includes(
    source_path: str, repo_root: str, skip_includes: set[str] | None = None
) -> list[str]:
    """Expand every #include of source_path (relative to shaders/), dropping skip_includes."""
    skipped = skip_includes or set()
    seen: set[str] = set()
    result: list[str] = []
    defines: dict[str, str] = {}

    def _expand(path: str, text: str) -> None:
        if path in seen:
            return
        seen.add(path)
        for line in text.split("\n"):
            m_define = re.match(r'^\s*#\s*define\s+(\w+)\s+(\d+)\s*$', line)
            if m_define:
                defines[m_define.group(1)] = m_define.group(2)

            m = re.match(r'^\s*#\s*include\s+"([^"]+)"', line)
            if m:
                included = m.group(1)
                if included in skipped:
                    continue
                inc_path = resolve_include(path, included, repo_root)
                if inc_path.replace(os.sep, "/") in skipped:
                    continue
                with open(os.path.join(repo_root, "shaders", inc_path), "r") as f:
                    _expand(inc_path, f.read())
            else:
                result.append(resolve_layout_macros(line, defines))

    with open(os.path.join(repo_root, "shaders", source_path), "r") as f:
        _expand(source_path, f.read())
    return result


def strip_include_guards(lines: list[str]) -> list[str]:
    """Strip #ifndef/#define _GLSL guards and drop #ifdef WATER_RAY_QUERY blocks."""
    result: list[str] = []
    stack: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        m_ifndef = re.match(r'^#\s*ifndef\s+(\S+)', stripped)
        if m_ifndef:
            macro = m_ifndef.group(1)
            is_guard = macro.endswith("_GLSL")
            stack.append("guard" if is_guard else "other")
            i += 1
            if not is_guard:
                result.append(line)
            if is_guard and i < len(lines):
                next_stripped = lines[i].strip()
                m_define = re.match(r'^#\s*define\s+' + re.escape(macro), next_stripped)
                if m_define:
                    i += 1
            continue

        m_ifdef = re.match(r'^#\s*ifdef\s+(\S+)', stripped)
        if m_ifdef:
            if m_ifdef.group(1) == "WATER_RAY_QUERY":
                stack.append("water_ray_query")
                i += 1
                continue
            stack.append("other")
            result.append(line)
            i += 1
            continue

        m_if = re.match(r'^#\s*if\b', stripped)
        if m_if:
            stack.append("other")
            result.append(line)
            i += 1
            continue

        m_endif = re.match(r'^#\s*endif\b', stripped)
        if m_endif:
            if stack and stack[-1] in ("water_ray_query", "guard"):
                stack.pop()
                i += 1
                continue
            if stack:
                stack.pop()
            result.append(line)
            i += 1
            continue

        if stack and stack[-1] == "water_ray_query":
            i += 1
            continue

        result.append(line)
        i += 1
    return result
