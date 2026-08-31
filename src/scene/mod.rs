mod clip_io;
mod error;
mod format;
mod motion_path_format;
mod scene_io;
mod transform_format;

pub use clip_io::{load_animation_clip, save_animation_clip};
pub use error::{SceneError, SceneResult};
pub use format::{
    apply_flame_state_to_world, apply_water_state_to_world, build_flame_scene_data,
    build_water_scene_data, AnimationClipFile, AnimationClipRef, CameraState as SavedCameraState,
    EditorState, ModelReference, SceneFile, SceneMetadata, TimelineConfig,
    ANIMATION_FORMAT_VERSION, SCENE_FORMAT_VERSION,
};
pub use motion_path_format::{
    motion_path_parameter_snapshot, overwrite_motion_path_persisted_fields,
    MOTION_PATH_SCALAR_PARAMS,
};
pub use scene_io::{
    apply_loaded_scene_to_world, find_default_scene, load_scene, save_scene, LoadedScene,
};
pub use transform_format::{
    overwrite_transform_persisted_fields, transform_parameter_snapshot, TRANSFORM_SCALAR_PARAMS,
};
