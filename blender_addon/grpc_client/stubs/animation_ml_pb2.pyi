from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SkeletonType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SMPL_22: _ClassVar[SkeletonType]
    VRM_HUMANOID: _ClassVar[SkeletonType]

class PropertyType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRANSLATION_X: _ClassVar[PropertyType]
    TRANSLATION_Y: _ClassVar[PropertyType]
    TRANSLATION_Z: _ClassVar[PropertyType]
    ROTATION_X: _ClassVar[PropertyType]
    ROTATION_Y: _ClassVar[PropertyType]
    ROTATION_Z: _ClassVar[PropertyType]

class InterpolationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LINEAR: _ClassVar[InterpolationType]
    BEZIER: _ClassVar[InterpolationType]

class MeshInputMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TEXT_TO_MESH: _ClassVar[MeshInputMode]
    IMAGE_TO_MESH: _ClassVar[MeshInputMode]
    IMAGE_REFINED_TO_MESH: _ClassVar[MeshInputMode]

class MeshModelType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRELLIS: _ClassVar[MeshModelType]
    HUNYUAN3D: _ClassVar[MeshModelType]
    CHARACTER_GEN: _ClassVar[MeshModelType]
    STDGEN: _ClassVar[MeshModelType]
    ERA3D: _ClassVar[MeshModelType]

class TextToImageModelType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    T2I_SERVER_DEFAULT: _ClassVar[TextToImageModelType]
    T2I_SDXL: _ClassVar[TextToImageModelType]
    T2I_ANIMAGINE: _ClassVar[TextToImageModelType]

class RiggingModelType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RIGGING_UNIRIG: _ClassVar[RiggingModelType]
SMPL_22: SkeletonType
VRM_HUMANOID: SkeletonType
TRANSLATION_X: PropertyType
TRANSLATION_Y: PropertyType
TRANSLATION_Z: PropertyType
ROTATION_X: PropertyType
ROTATION_Y: PropertyType
ROTATION_Z: PropertyType
LINEAR: InterpolationType
BEZIER: InterpolationType
TEXT_TO_MESH: MeshInputMode
IMAGE_TO_MESH: MeshInputMode
IMAGE_REFINED_TO_MESH: MeshInputMode
TRELLIS: MeshModelType
HUNYUAN3D: MeshModelType
CHARACTER_GEN: MeshModelType
STDGEN: MeshModelType
ERA3D: MeshModelType
T2I_SERVER_DEFAULT: TextToImageModelType
T2I_SDXL: TextToImageModelType
T2I_ANIMAGINE: TextToImageModelType
RIGGING_UNIRIG: RiggingModelType

class MotionRequest(_message.Message):
    __slots__ = ("prompt", "duration_seconds", "target_fps", "skeleton_type", "bone_mappings", "glb_skeleton", "internal_use_only")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TARGET_FPS_FIELD_NUMBER: _ClassVar[int]
    SKELETON_TYPE_FIELD_NUMBER: _ClassVar[int]
    BONE_MAPPINGS_FIELD_NUMBER: _ClassVar[int]
    GLB_SKELETON_FIELD_NUMBER: _ClassVar[int]
    INTERNAL_USE_ONLY_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    duration_seconds: float
    target_fps: int
    skeleton_type: SkeletonType
    bone_mappings: _containers.RepeatedCompositeFieldContainer[BoneMapping]
    glb_skeleton: GlbSkeletonSpec
    internal_use_only: bool
    def __init__(self, prompt: _Optional[str] = ..., duration_seconds: _Optional[float] = ..., target_fps: _Optional[int] = ..., skeleton_type: _Optional[_Union[SkeletonType, str]] = ..., bone_mappings: _Optional[_Iterable[_Union[BoneMapping, _Mapping]]] = ..., glb_skeleton: _Optional[_Union[GlbSkeletonSpec, _Mapping]] = ..., internal_use_only: bool = ...) -> None: ...

class GlbSkeletonSpec(_message.Message):
    __slots__ = ("glb_data", "skeleton_cache_id")
    GLB_DATA_FIELD_NUMBER: _ClassVar[int]
    SKELETON_CACHE_ID_FIELD_NUMBER: _ClassVar[int]
    glb_data: bytes
    skeleton_cache_id: str
    def __init__(self, glb_data: _Optional[bytes] = ..., skeleton_cache_id: _Optional[str] = ...) -> None: ...

class BoneMapping(_message.Message):
    __slots__ = ("source_joint_index", "target_bone_name")
    SOURCE_JOINT_INDEX_FIELD_NUMBER: _ClassVar[int]
    TARGET_BONE_NAME_FIELD_NUMBER: _ClassVar[int]
    source_joint_index: int
    target_bone_name: str
    def __init__(self, source_joint_index: _Optional[int] = ..., target_bone_name: _Optional[str] = ...) -> None: ...

class MotionResponse(_message.Message):
    __slots__ = ("curves", "generation_time_ms", "model_used")
    CURVES_FIELD_NUMBER: _ClassVar[int]
    GENERATION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    MODEL_USED_FIELD_NUMBER: _ClassVar[int]
    curves: _containers.RepeatedCompositeFieldContainer[AnimationCurve]
    generation_time_ms: float
    model_used: str
    def __init__(self, curves: _Optional[_Iterable[_Union[AnimationCurve, _Mapping]]] = ..., generation_time_ms: _Optional[float] = ..., model_used: _Optional[str] = ...) -> None: ...

class AnimationCurve(_message.Message):
    __slots__ = ("bone_name", "property_type", "keyframes")
    BONE_NAME_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_TYPE_FIELD_NUMBER: _ClassVar[int]
    KEYFRAMES_FIELD_NUMBER: _ClassVar[int]
    bone_name: str
    property_type: PropertyType
    keyframes: _containers.RepeatedCompositeFieldContainer[CurveKeyframe]
    def __init__(self, bone_name: _Optional[str] = ..., property_type: _Optional[_Union[PropertyType, str]] = ..., keyframes: _Optional[_Iterable[_Union[CurveKeyframe, _Mapping]]] = ...) -> None: ...

class CurveKeyframe(_message.Message):
    __slots__ = ("time", "value", "tangent_in_dt", "tangent_in_dv", "tangent_out_dt", "tangent_out_dv", "interpolation")
    TIME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    TANGENT_IN_DT_FIELD_NUMBER: _ClassVar[int]
    TANGENT_IN_DV_FIELD_NUMBER: _ClassVar[int]
    TANGENT_OUT_DT_FIELD_NUMBER: _ClassVar[int]
    TANGENT_OUT_DV_FIELD_NUMBER: _ClassVar[int]
    INTERPOLATION_FIELD_NUMBER: _ClassVar[int]
    time: float
    value: float
    tangent_in_dt: float
    tangent_in_dv: float
    tangent_out_dt: float
    tangent_out_dv: float
    interpolation: InterpolationType
    def __init__(self, time: _Optional[float] = ..., value: _Optional[float] = ..., tangent_in_dt: _Optional[float] = ..., tangent_in_dv: _Optional[float] = ..., tangent_out_dt: _Optional[float] = ..., tangent_out_dv: _Optional[float] = ..., interpolation: _Optional[_Union[InterpolationType, str]] = ...) -> None: ...

class StatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("ready", "active_model", "gpu_memory_mb")
    READY_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_MODEL_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    active_model: str
    gpu_memory_mb: int
    def __init__(self, ready: bool = ..., active_model: _Optional[str] = ..., gpu_memory_mb: _Optional[int] = ...) -> None: ...

class MeshRequest(_message.Message):
    __slots__ = ("prompt", "params", "input_image_png", "input_mode", "model_type", "t2i_model_type")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    INPUT_IMAGE_PNG_FIELD_NUMBER: _ClassVar[int]
    INPUT_MODE_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    T2I_MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    params: MeshGenerationParams
    input_image_png: bytes
    input_mode: MeshInputMode
    model_type: MeshModelType
    t2i_model_type: TextToImageModelType
    def __init__(self, prompt: _Optional[str] = ..., params: _Optional[_Union[MeshGenerationParams, _Mapping]] = ..., input_image_png: _Optional[bytes] = ..., input_mode: _Optional[_Union[MeshInputMode, str]] = ..., model_type: _Optional[_Union[MeshModelType, str]] = ..., t2i_model_type: _Optional[_Union[TextToImageModelType, str]] = ...) -> None: ...

class MeshGenerationParams(_message.Message):
    __slots__ = ("target_faces", "seed", "image_size", "image_inference_steps")
    TARGET_FACES_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    IMAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    IMAGE_INFERENCE_STEPS_FIELD_NUMBER: _ClassVar[int]
    target_faces: int
    seed: int
    image_size: int
    image_inference_steps: int
    def __init__(self, target_faces: _Optional[int] = ..., seed: _Optional[int] = ..., image_size: _Optional[int] = ..., image_inference_steps: _Optional[int] = ...) -> None: ...

class MeshResponse(_message.Message):
    __slots__ = ("glb_data", "metadata")
    GLB_DATA_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    glb_data: bytes
    metadata: MeshMetadata
    def __init__(self, glb_data: _Optional[bytes] = ..., metadata: _Optional[_Union[MeshMetadata, _Mapping]] = ...) -> None: ...

class MeshMetadata(_message.Message):
    __slots__ = ("vertex_count", "face_count", "generation_time_ms", "intermediate_image_png")
    VERTEX_COUNT_FIELD_NUMBER: _ClassVar[int]
    FACE_COUNT_FIELD_NUMBER: _ClassVar[int]
    GENERATION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    INTERMEDIATE_IMAGE_PNG_FIELD_NUMBER: _ClassVar[int]
    vertex_count: int
    face_count: int
    generation_time_ms: float
    intermediate_image_png: bytes
    def __init__(self, vertex_count: _Optional[int] = ..., face_count: _Optional[int] = ..., generation_time_ms: _Optional[float] = ..., intermediate_image_png: _Optional[bytes] = ...) -> None: ...

class MeshStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class MeshStatusResponse(_message.Message):
    __slots__ = ("ready", "t2i_model", "i2m_model", "gpu_memory_mb")
    READY_FIELD_NUMBER: _ClassVar[int]
    T2I_MODEL_FIELD_NUMBER: _ClassVar[int]
    I2M_MODEL_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    t2i_model: str
    i2m_model: str
    gpu_memory_mb: int
    def __init__(self, ready: bool = ..., t2i_model: _Optional[str] = ..., i2m_model: _Optional[str] = ..., gpu_memory_mb: _Optional[int] = ...) -> None: ...

class RiggingRequest(_message.Message):
    __slots__ = ("glb_data", "params", "model_type")
    GLB_DATA_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    glb_data: bytes
    params: RiggingParams
    model_type: RiggingModelType
    def __init__(self, glb_data: _Optional[bytes] = ..., params: _Optional[_Union[RiggingParams, _Mapping]] = ..., model_type: _Optional[_Union[RiggingModelType, str]] = ...) -> None: ...

class RiggingParams(_message.Message):
    __slots__ = ("num_sample_points",)
    NUM_SAMPLE_POINTS_FIELD_NUMBER: _ClassVar[int]
    num_sample_points: int
    def __init__(self, num_sample_points: _Optional[int] = ...) -> None: ...

class SkeletonJoint(_message.Message):
    __slots__ = ("name", "x", "y", "z", "tail_x", "tail_y", "tail_z", "parent_index")
    NAME_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    TAIL_X_FIELD_NUMBER: _ClassVar[int]
    TAIL_Y_FIELD_NUMBER: _ClassVar[int]
    TAIL_Z_FIELD_NUMBER: _ClassVar[int]
    PARENT_INDEX_FIELD_NUMBER: _ClassVar[int]
    name: str
    x: float
    y: float
    z: float
    tail_x: float
    tail_y: float
    tail_z: float
    parent_index: int
    def __init__(self, name: _Optional[str] = ..., x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., tail_x: _Optional[float] = ..., tail_y: _Optional[float] = ..., tail_z: _Optional[float] = ..., parent_index: _Optional[int] = ...) -> None: ...

class RiggingResponse(_message.Message):
    __slots__ = ("rigged_glb_data", "metadata", "skeleton_joints")
    RIGGED_GLB_DATA_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    SKELETON_JOINTS_FIELD_NUMBER: _ClassVar[int]
    rigged_glb_data: bytes
    metadata: RiggingMetadata
    skeleton_joints: _containers.RepeatedCompositeFieldContainer[SkeletonJoint]
    def __init__(self, rigged_glb_data: _Optional[bytes] = ..., metadata: _Optional[_Union[RiggingMetadata, _Mapping]] = ..., skeleton_joints: _Optional[_Iterable[_Union[SkeletonJoint, _Mapping]]] = ...) -> None: ...

class RiggingMetadata(_message.Message):
    __slots__ = ("joint_count", "bone_count", "generation_time_ms")
    JOINT_COUNT_FIELD_NUMBER: _ClassVar[int]
    BONE_COUNT_FIELD_NUMBER: _ClassVar[int]
    GENERATION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    joint_count: int
    bone_count: int
    generation_time_ms: float
    def __init__(self, joint_count: _Optional[int] = ..., bone_count: _Optional[int] = ..., generation_time_ms: _Optional[float] = ...) -> None: ...

class RiggingStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RiggingStatusResponse(_message.Message):
    __slots__ = ("ready", "model_name", "gpu_memory_mb")
    READY_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    model_name: str
    gpu_memory_mb: int
    def __init__(self, ready: bool = ..., model_name: _Optional[str] = ..., gpu_memory_mb: _Optional[int] = ...) -> None: ...
