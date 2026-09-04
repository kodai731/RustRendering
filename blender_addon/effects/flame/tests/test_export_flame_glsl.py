
import json
import os
import re
import subprocess
import sys
import tempfile

import pytest

import importlib.util

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "blender", "flame", "export_glsl.py")
_flame_spec = importlib.util.spec_from_file_location("flame_export_glsl", SCRIPT)
_flame_module = importlib.util.module_from_spec(_flame_spec)
_flame_spec.loader.exec_module(_flame_module)
expand_includes = _flame_module.expand_includes


def _run_exporter(tmp_path: str) -> tuple[str, dict]:
    out_dir = os.path.join(tmp_path, "shaders")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--repo-root", os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."), "--out", out_dir],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"exporter failed: {result.stderr}"
    glsl_path = os.path.join(out_dir, "flame_resolve.glsl")
    json_path = os.path.join(out_dir, "flame_resolve.bindings.json")
    with open(glsl_path) as f:
        glsl_text = f.read()
    with open(json_path) as f:
        bindings = json.load(f)
    return glsl_text, bindings


def _expanded_lines(repo_root: str) -> list[str]:
    return expand_includes("flameResolveFragment.frag", repo_root)


@pytest.fixture(scope="module")
def exported():
    with tempfile.TemporaryDirectory() as tmp:
        glsl_text, bindings = _run_exporter(tmp)
        yield glsl_text, bindings


class TestNoTransformedArtifacts:

    def test_no_include(self, exported):
        glsl_text, _ = exported
        assert "#include" not in glsl_text, "output still contains #include lines"

    def test_no_version(self, exported):
        glsl_text, _ = exported
        assert "#version" not in glsl_text, "output still contains #version line"

    def test_no_layout(self, exported):
        glsl_text, _ = exported
        assert "layout(" not in glsl_text, "output still contains layout( declarations"

    def test_no_include_guards(self, exported):
        glsl_text, _ = exported
        for line in glsl_text.splitlines():
            assert not re.match(r'^\s*#\s*ifndef\s+\S+_GLSL\b', line), (
                f"output still contains #ifndef *_GLSL guard: {line!r}"
            )
            assert not re.match(r'^\s*#\s*define\s+\S+_GLSL\b', line), (
                f"output still contains #define *_GLSL guard: {line!r}"
            )

    def test_non_guard_ifndef_preserved(self, exported):
        glsl_text, _ = exported
        assert "#ifndef FLAME_NOISE_ROT_DEG_OVERRIDE" in glsl_text, (
            "output is missing #ifndef FLAME_NOISE_ROT_DEG_OVERRIDE"
        )

    def test_preprocessor_guards_balanced(self, exported):
        glsl_text, _ = exported
        lines = glsl_text.splitlines()
        open_count = sum(
            1 for line in lines
            if re.match(r'^\s*#\s*(if|ifdef|ifndef)\b', line)
        )
        close_count = sum(
            1 for line in lines
            if re.match(r'^\s*#\s*endif\b', line)
        )
        assert open_count == close_count, (
            f"unbalanced preprocessor guards: {open_count} #if/#ifdef/#ifndef vs {close_count} #endif"
        )

    def test_sampler_count(self, exported):
        _, bindings = exported
        samplers = bindings["samplers"]
        assert len(samplers) == 3, f"expected 3 samplers, got {len(samplers)}: {samplers}"

    def test_sampler_names(self, exported):
        _, bindings = exported
        names = {s["name"] for s in bindings["samplers"]}
        expected = {"flameHistorySampler", "flameSdfSampler", "sceneDepthSampler"}
        assert names == expected, f"expected {expected}, got {names}"

    def test_output_count(self, exported):
        _, bindings = exported
        outputs = bindings["outputs"]
        assert len(outputs) == 2, f"expected 2 outputs, got {len(outputs)}: {outputs}"

    def test_output_names(self, exported):
        _, bindings = exported
        outputs = bindings["outputs"]
        expected = {"outColor", "outHistory"}
        assert set(outputs) == expected, f"expected {expected}, got {set(outputs)}"


class TestByteIdentical:

    def test_lines_match(self, exported):
        glsl_text, _ = exported
        repo_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        expanded = _expanded_lines(repo_root)
        stripped = []
        stack: list[bool] = []
        i = 0
        while i < len(expanded):
            line = expanded[i]
            m_ifndef = re.match(r'^#\s*ifndef\s+(\S+)', line.strip())
            if m_ifndef:
                macro = m_ifndef.group(1)
                is_guard = macro.endswith("_GLSL")
                stack.append(is_guard)
                i += 1
                if is_guard and i < len(expanded):
                    ns = expanded[i].strip()
                    if re.match(r'^#\s*define\s+' + re.escape(macro), ns):
                        i += 1
                if not is_guard:
                    stripped.append(line)
                continue
            m_ifdef = re.match(r'^#\s*ifdef\s+\S+', line.strip())
            m_if = re.match(r'^#\s*if\b', line.strip())
            if m_ifdef or m_if:
                stack.append(False)
                stripped.append(line)
                i += 1
                continue
            m_endif = re.match(r'^#\s*endif\b', line.strip())
            if m_endif:
                if stack and stack[-1]:
                    stack.pop()
                    i += 1
                    continue
                elif stack:
                    stack.pop()
                stripped.append(line)
                i += 1
                continue
            stripped.append(line)
            i += 1

        expected_lines = []
        i = 0
        while i < len(stripped):
            line = stripped[i]
            if re.match(r'^\s*#\s*version\b', line):
                i += 1
                continue
            layout_match = re.match(r'^\s*layout\s*\(', line)
            if layout_match:
                if re.match(r'^\s*layout\s*\(\s*location\b', line):
                    i += 1
                    continue
                if re.match(r'^\s*layout\s*\(\s*set\s*=\s*\d+\s*,\s*binding\s*=', line):
                    if "uniform sampler2D" in line:
                        i += 1
                        continue
                    ubo_match = re.match(
                        r'^\s*layout\s*\(\s*set\s*=\s*\d+\s*,\s*binding\s*=\s*\d+\s*\)\s+uniform\s+(\w+)\s*\{',
                        line,
                    )
                    if ubo_match:
                        struct_name = ubo_match.group(1)
                        expected_lines.append(f"struct {struct_name} {{")
                        i += 1
                        while i < len(stripped):
                            if re.match(r'^\s*\}\s*\w+\s*;', stripped[i]):
                                expected_lines.append("};")
                                i += 1
                                break
                            else:
                                expected_lines.append(stripped[i])
                                i += 1
                        continue
                push_match = re.match(
                    r'^\s*layout\s*\(\s*push_constant\s*\)\s+uniform\s+(\w+)\s*\{',
                    line,
                )
                if push_match:
                    j = i
                    while j < len(stripped):
                        if re.match(r'^\s*\}\s*\w+\s*;', stripped[j]):
                            break
                        j += 1
                    i = j + 1
                    continue
            expected_lines.append(line)
            i += 1

        actual_lines = glsl_text.splitlines()

        mismatches = []
        for idx, (exp, act) in enumerate(zip(expected_lines, actual_lines)):
            if exp != act:
                mismatches.append((idx, exp, act))

        if mismatches:
            first = mismatches[0]
            msg = f"Line {first[0]} differs:\n  expected: {first[1]!r}\n  actual:   {first[2]!r}"
            if len(mismatches) > 1:
                msg += f"\n... and {len(mismatches) - 1} more mismatches"
            assert False, msg

        assert len(actual_lines) == len(expected_lines), (
            f"line count mismatch: expected {len(expected_lines)}, got {len(actual_lines)}"
        )
