"""Prompt strategies for tool selection.

The first measurement showed the prompt, not the model, dominates the result:
zero-shot collapses onto whichever tool is listed first, while a handful of
demonstrations restores real selection. Strategies live here so the eval can
compare them, and so the Rust prompt builder has a decided reference.

FEW_SHOT_EXAMPLES demonstrate four tools. Cases whose expected tool is one of
those are reported separately, because the demonstration inflates them.
"""

from dataclasses import dataclass

from tool_schema import ToolSpec, build_tool_catalog

PAUSE_EXAMPLE_BY_TOOL_NAME = {
    "pause_animation": '{"tool": "pause_animation"}',
    "control_playback": '{"tool": "control_playback", "action": "pause"}',
}

FEW_SHOT_EXAMPLES = (
    ("pause the animation", None),
    ("select the cube", '{"tool": "select_object", "name": "cube"}'),
    ("frame the whole model", '{"tool": "focus_camera", "target": "model"}'),
    ("redo it", '{"tool": "redo"}'),
)

DEMONSTRATED_TOOLS = frozenset(
    {"pause_animation", "control_playback", "select_object", "focus_camera", "redo"}
)

ROLE_INSTRUCTION = (
    "You control a 3D animation editor. Choose the one tool that performs what "
    "the user asks."
)


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str


def resolve_pause_example(tools: tuple[ToolSpec, ...]) -> str:
    available = {tool.name for tool in tools}
    for tool_name, tool_call in PAUSE_EXAMPLE_BY_TOOL_NAME.items():
        if tool_name in available:
            return tool_call
    raise ValueError("no playback tool available to demonstrate")


def format_few_shot_block(tools: tuple[ToolSpec, ...]) -> str:
    lines = ["Examples:"]
    for request, tool_call in FEW_SHOT_EXAMPLES:
        lines.append(f"Request: {request}")
        lines.append(f"Tool call: {tool_call or resolve_pause_example(tools)}")
    return "\n".join(lines)


def build_zero_shot_prompt(tools: tuple[ToolSpec, ...], utterance: str) -> RenderedPrompt:
    system = (
        "You control a 3D animation editor. Translate the user's request into "
        "exactly one tool call, written as a single JSON object.\n\n"
        f"Available tools:\n{build_tool_catalog(tools)}\n\n"
        'Answer with JSON only, in the form {"tool": "<name>", ...arguments}. '
        "If the request is ambiguous call ask_clarification. "
        "If no tool fits the request call unsupported_request."
    )
    return RenderedPrompt(system, utterance)


def build_few_shot_prompt(tools: tuple[ToolSpec, ...], utterance: str) -> RenderedPrompt:
    system = (
        f"{ROLE_INSTRUCTION}\n\n"
        f"Available tools:\n{build_tool_catalog(tools)}\n\n"
        f"{format_few_shot_block(tools)}"
    )
    return RenderedPrompt(system, f"Request: {utterance}\nTool call:")


def build_request_framing_prompt(tools: tuple[ToolSpec, ...], utterance: str) -> RenderedPrompt:
    system = f"{ROLE_INSTRUCTION}\n\nAvailable tools:\n{build_tool_catalog(tools)}"
    return RenderedPrompt(system, f"Request: {utterance}\nTool call:")


PROMPT_STRATEGIES = {
    "zero_shot": build_zero_shot_prompt,
    "request_framing": build_request_framing_prompt,
    "few_shot": build_few_shot_prompt,
}
