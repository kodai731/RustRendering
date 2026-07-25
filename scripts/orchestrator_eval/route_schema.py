"""Route expansion for the embedding router.

A route is a tool combined with the enum arguments that actually appear in an
utterance. Expanding those enums into the route identity turns almost every
route into a zero-argument target, which is the condition the generative
evaluation scored 100% on.

Mirrors src/orchestrator/components/route.rs; the route ids must match exactly
or the measured accuracy does not describe the engine build.
"""

from dataclasses import dataclass

from tool_schema import TOOL_VARIANTS, ESCAPE, ToolSpec

ESCAPE_ROUTE = "__escape__"

ROUTE_DIMENSIONS = {
    "set_playback_speed": ("speed",),
    "seek_time": ("position",),
    "set_object_visibility": ("state",),
    "focus_camera": ("target",),
    "generate_motion": ("category",),
}

MODIFIER_ARGS = {("generate_motion", "speed")}

DEFAULT_ARGS = {("generate_motion", "variation"): "default"}


@dataclass(frozen=True)
class RouteSpec:
    route_id: str
    tool_name: str
    kind: str
    enum_args: tuple[tuple[str, str], ...]
    slot_args: tuple[str, ...]

    def is_read_only(self) -> bool:
        return self.kind == "query"


def find_dimension_args(tool: ToolSpec) -> tuple[str, ...]:
    return ROUTE_DIMENSIONS.get(tool.name, ())


def find_slot_args(tool: ToolSpec) -> tuple[str, ...]:
    dimensions = find_dimension_args(tool)
    return tuple(
        arg.name
        for arg in tool.args
        if not arg.is_enum()
        and arg.name not in dimensions
        and (tool.name, arg.name) not in MODIFIER_ARGS
    )


def build_route_id(tool_name: str, enum_args: tuple[tuple[str, str], ...]) -> str:
    if not enum_args:
        return tool_name
    return tool_name + ":" + ":".join(value for _, value in enum_args)


def expand_tool_to_routes(tool: ToolSpec) -> list[RouteSpec]:
    slot_args = find_slot_args(tool)
    dimensions = find_dimension_args(tool)
    if not dimensions:
        return [RouteSpec(tool.name, tool.name, tool.kind, (), slot_args)]

    combinations: list[tuple[tuple[str, str], ...]] = [()]
    for dimension in dimensions:
        spec = next(arg for arg in tool.args if arg.name == dimension)
        combinations = [
            existing + ((dimension, value),)
            for existing in combinations
            for value in spec.values
        ]

    return [
        RouteSpec(build_route_id(tool.name, combo), tool.name, tool.kind, combo, slot_args)
        for combo in combinations
    ]


def build_routes(variant: str = "baseline") -> list[RouteSpec]:
    routes: list[RouteSpec] = []
    for tool in TOOL_VARIANTS[variant]:
        if tool.kind == ESCAPE:
            continue
        routes.extend(expand_tool_to_routes(tool))
    return routes


def resolve_expected_route(tool_name: str, args: dict, variant: str = "baseline") -> str:
    tool = next(
        (spec for spec in TOOL_VARIANTS[variant] if spec.name == tool_name), None
    )
    if tool is None or tool.kind == ESCAPE:
        return ESCAPE_ROUTE

    dimensions = find_dimension_args(tool)
    enum_args = tuple((name, args[name]) for name in dimensions)
    return build_route_id(tool_name, enum_args)
