from __future__ import annotations

from .stubs import pb2


_PROPERTY_TYPE_BY_INT: dict[int, str] = {
    pb2.TRANSLATION_X: "TRANSLATION_X",
    pb2.TRANSLATION_Y: "TRANSLATION_Y",
    pb2.TRANSLATION_Z: "TRANSLATION_Z",
    pb2.ROTATION_X: "ROTATION_X",
    pb2.ROTATION_Y: "ROTATION_Y",
    pb2.ROTATION_Z: "ROTATION_Z",
}

_INTERPOLATION_BY_INT: dict[int, str] = {
    pb2.LINEAR: "LINEAR",
    pb2.BEZIER: "BEZIER",
}


def property_type_str(value: int) -> str:
    return _PROPERTY_TYPE_BY_INT.get(value, "UNKNOWN")


def interpolation_str(value: int) -> str:
    return _INTERPOLATION_BY_INT.get(value, "LINEAR")


def mesh_input_mode_int(name: str) -> int:
    return getattr(pb2, name)


def mesh_model_type_int(name: str) -> int:
    return getattr(pb2, name)


def text_to_image_model_type_int(name: str) -> int:
    return getattr(pb2, name)


def skeleton_type_int(name: str) -> int:
    return getattr(pb2, name)
