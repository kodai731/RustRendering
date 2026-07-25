"""Tool schema SSoT for the orchestrator evaluation.

Mirrors the design in
${DocumentPath}/Rust_Rendering/Design/20260725_tiny_llm_command_orchestrator/
20260725_tiny_llm_command_orchestrator/thyllore_tool_schema.md

Both the JSON Schema handed to llguidance and the tool catalog embedded in the
prompt are generated from TOOL_VARIANTS, so the two never drift apart. This
module is the Python-side rehearsal of src/orchestrator/components/tool_schema.rs.
"""

from dataclasses import dataclass, field

QUERY = "query"
COMMAND = "command"
ESCAPE = "escape"


@dataclass(frozen=True)
class ArgSpec:
    name: str
    description: str
    values: tuple[str, ...] = ()

    def is_enum(self) -> bool:
        return len(self.values) > 0

    def to_json_schema(self) -> dict:
        if self.is_enum():
            return {"enum": list(self.values)}
        return {"type": "string"}

    def describe(self) -> str:
        if self.is_enum():
            return f"{self.name}: {'|'.join(self.values)}"
        return f"{self.name}: string"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: str
    description: str
    args: tuple[ArgSpec, ...] = field(default_factory=tuple)

    def to_json_schema(self) -> dict:
        properties = {"tool": {"const": self.name}}
        for arg in self.args:
            properties[arg.name] = arg.to_json_schema()
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
            "additionalProperties": False,
        }

    def describe(self) -> str:
        signature = ", ".join(arg.describe() for arg in self.args)
        return f"- {self.name}({signature}) — {self.description}"


SPEED_VALUES = ("slow", "normal", "fast")
SEEK_VALUES = ("start", "end", "next_key", "prev_key")
VISIBILITY_VALUES = ("show", "hide")
FOCUS_VALUES = ("selection", "model", "reset")
MOTION_CATEGORY_VALUES = ("walk", "run", "idle", "jump", "turn")
MOTION_VARIATION_VALUES = ("default", "alternate")

QUERY_TOOLS = (
    ToolSpec("list_objects", QUERY, "List the names of every object in the scene"),
    ToolSpec("describe_selection", QUERY, "Describe what is currently selected"),
    ToolSpec("get_playback_state", QUERY, "Report whether the animation is playing, its time and speed"),
    ToolSpec("take_screenshot", QUERY, "Capture a screenshot of the viewport"),
)

PLAYBACK_TOOLS = (
    ToolSpec("play_animation", COMMAND, "Start playing the animation"),
    ToolSpec("pause_animation", COMMAND, "Pause the animation at the current time"),
    ToolSpec("stop_animation", COMMAND, "Stop the animation and rewind to the start"),
)

AGGREGATED_PLAYBACK_TOOLS = (
    ToolSpec(
        "control_playback",
        COMMAND,
        "Start, pause or stop the animation",
        (ArgSpec("action", "What to do with playback", ("play", "pause", "stop")),),
    ),
)

SHARED_COMMAND_TOOLS = (
    ToolSpec(
        "set_playback_speed",
        COMMAND,
        "Change how fast the animation plays",
        (ArgSpec("speed", "Playback speed", SPEED_VALUES),),
    ),
    ToolSpec(
        "seek_time",
        COMMAND,
        "Jump to a position on the timeline",
        (ArgSpec("position", "Where to jump to", SEEK_VALUES),),
    ),
    ToolSpec("toggle_loop", COMMAND, "Turn looped playback on or off"),
    ToolSpec(
        "select_object",
        COMMAND,
        "Select a scene object by name",
        (ArgSpec("name", "Name of the object to select"),),
    ),
    ToolSpec(
        "set_object_visibility",
        COMMAND,
        "Show or hide a scene object",
        (
            ArgSpec("name", "Name of the object"),
            ArgSpec("state", "Whether to show or hide it", VISIBILITY_VALUES),
        ),
    ),
    ToolSpec(
        "focus_camera",
        COMMAND,
        "Point the camera at something",
        (ArgSpec("target", "What the camera should frame", FOCUS_VALUES),),
    ),
    ToolSpec("undo", COMMAND, "Undo the last edit"),
    ToolSpec("redo", COMMAND, "Redo the edit that was undone"),
    ToolSpec("save_scene", COMMAND, "Save the current scene to disk"),
    ToolSpec(
        "generate_motion",
        COMMAND,
        "Generate a new animation for the selected character",
        (
            ArgSpec("category", "Kind of motion", MOTION_CATEGORY_VALUES),
            ArgSpec("variation", "Which variant to generate", MOTION_VARIATION_VALUES),
            ArgSpec("speed", "How fast the motion should be", SPEED_VALUES),
        ),
    ),
)

ESCAPE_TOOLS = (
    ToolSpec(
        "ask_clarification",
        ESCAPE,
        "Ask the user what they meant when the request is ambiguous",
        (ArgSpec("reason", "What is unclear about the request"),),
    ),
    ToolSpec(
        "unsupported_request",
        ESCAPE,
        "Report that no available tool can satisfy the request",
    ),
)

TOOL_VARIANTS = {
    "baseline": QUERY_TOOLS + PLAYBACK_TOOLS + SHARED_COMMAND_TOOLS + ESCAPE_TOOLS,
    "aggregated_playback": QUERY_TOOLS + AGGREGATED_PLAYBACK_TOOLS + SHARED_COMMAND_TOOLS + ESCAPE_TOOLS,
}


def build_json_schema(tools: tuple[ToolSpec, ...]) -> dict:
    return {"anyOf": [tool.to_json_schema() for tool in tools]}


def build_tool_catalog(tools: tuple[ToolSpec, ...]) -> str:
    return "\n".join(tool.describe() for tool in tools)


def find_tool(tools: tuple[ToolSpec, ...], name: str) -> ToolSpec | None:
    return next((tool for tool in tools if tool.name == name), None)
