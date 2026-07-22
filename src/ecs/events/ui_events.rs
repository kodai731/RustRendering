use cgmath::{Quaternion, Vector3};

use crate::animation::editable::{
    BezierHandle, BlendMode, ClipGroupId, ClipInstanceId, InterpolationType, KeyframeId,
    PropertyType, SourceClipId, TangentType, TangentWeightMode,
};
use crate::animation::BoneId;
use crate::animation::{ConstraintId, ConstraintType};
use crate::app::data::LightMoveTarget;
use crate::ecs::component::{
    ColliderShape, FlameParam, SpringChain, SpringChainId, SpringColliderDef, SpringColliderGroup,
    SpringColliderGroupId, SpringColliderId, SpringJointParam,
};
use crate::ecs::resource::gizmo::BoneDisplayStyle;
use crate::ecs::resource::{
    AutoExposure, CoordinateSpace, DepthOfField, FlameEffect, FlameRenderSettings,
    HierarchyDisplayMode, OnionSkinningConfig, PhysicalCameraParameters, SelectedKeyframe,
    SelectionModifier, TransformGizmoMode, TransformGizmoState,
};
use crate::ecs::world::Entity;
use crate::ecs::world::Visibility;

#[cfg(feature = "auto-rig")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelLoadSource {
    UserFile,
    AutoRigOutput,
    TextToMeshOutput,
}

#[derive(Clone, Debug)]
pub enum UIEvent {
    LoadModel {
        path: String,
    },
    LoadModelAdditive {
        path: String,
    },

    ResetCamera,
    ResetCameraUp,
    MoveCameraToModel,
    MoveCameraToLightGizmo,

    SetLightPosition(Vector3<f32>),
    MoveLightToBounds(LightMoveTarget),

    TakeScreenshot,

    #[cfg(debug_assertions)]
    DebugShadowInfo,
    #[cfg(debug_assertions)]
    DebugBillboardDepth,
    DumpDebugInfo,
    DumpAnimationDebug,

    SelectEntity(Entity),
    DeselectAll,
    ToggleEntitySelection(Entity),
    DeleteSelectedEntities,
    ExpandEntity(Entity),
    CollapseEntity(Entity),
    SetSearchFilter(String),

    SetHierarchyDisplayMode(HierarchyDisplayMode),
    SelectBone(BoneId),
    DeselectBone,
    ExpandBone(BoneId),
    CollapseBone(BoneId),

    SetEntityVisible(Entity, Visibility),
    SetEntityTranslation(Entity, Vector3<f32>),
    SetEntityRotation(Entity, Quaternion<f32>),
    SetEntityScale(Entity, Vector3<f32>),
    RenameEntity(Entity, String),
    FocusOnEntity(Entity),

    TimelinePlay,
    TimelinePause,
    TimelineStop,
    TimelineSetTime(f32),
    TimelineSetSpeed(f32),
    TimelineToggleLoop,
    TimelineSelectClip(SourceClipId),
    TimelineToggleTrack(BoneId),
    TimelineExpandTrack(BoneId),
    TimelineCollapseTrack(BoneId),
    TimelineSelectKeyframe {
        bone_id: BoneId,
        property_type: PropertyType,
        keyframe_id: KeyframeId,
        modifier: SelectionModifier,
    },
    TimelineAddKeyframe {
        bone_id: BoneId,
        property_type: PropertyType,
        time: f32,
        value: f32,
    },
    TimelineDeleteSelectedKeyframes,
    DeleteFlameKeysAt {
        time: f32,
    },
    TimelineMoveSelectedKeyframes {
        time_delta: f32,
    },
    TimelineSetKeyframeSelection {
        keyframes: Vec<SelectedKeyframe>,
        modifier: SelectionModifier,
    },
    TimelineDeleteKeyframe {
        bone_id: BoneId,
        property_type: PropertyType,
        keyframe_id: KeyframeId,
    },
    TimelineMoveKeyframe {
        bone_id: BoneId,
        property_type: PropertyType,
        keyframe_id: KeyframeId,
        new_time: f32,
        new_value: f32,
    },
    TimelineSetKeyframeInterpolation {
        bone_id: BoneId,
        property_type: PropertyType,
        keyframe_id: KeyframeId,
        interpolation: InterpolationType,
    },
    TimelineSetKeyframeTangent {
        bone_id: BoneId,
        property_type: PropertyType,
        keyframe_id: KeyframeId,
        in_tangent: BezierHandle,
        out_tangent: BezierHandle,
    },
    TimelineSetTangentType {
        bone_id: BoneId,
        property_type: PropertyType,
        keyframe_id: KeyframeId,
        tangent_type: TangentType,
    },

    TimelineSetTangentWeightMode {
        bone_id: BoneId,
        property_type: PropertyType,
        keyframe_id: KeyframeId,
        weight_mode: TangentWeightMode,
    },

    TimelineSetSnapToFrame(bool),
    TimelineSetSnapToKey(bool),
    TimelineSetFrameRate(f32),

    TimelineCopyKeyframes,
    TimelinePasteKeyframes {
        paste_time: f32,
    },
    TimelineMirrorPaste {
        paste_time: f32,
    },

    TimelineCaptureBuffer,
    TimelineSwapBuffer,

    ClipInstanceSelect {
        entity: Entity,
        instance_id: ClipInstanceId,
    },
    ClipInstanceDeselect,
    ClipInstanceMove {
        entity: Entity,
        instance_id: ClipInstanceId,
        new_start_time: f32,
    },
    ClipInstanceTrimStart {
        entity: Entity,
        instance_id: ClipInstanceId,
        new_clip_in: f32,
    },
    ClipInstanceTrimEnd {
        entity: Entity,
        instance_id: ClipInstanceId,
        new_clip_out: f32,
    },
    ClipInstanceToggleMute {
        entity: Entity,
        instance_id: ClipInstanceId,
    },
    ClipInstanceDelete {
        entity: Entity,
        instance_id: ClipInstanceId,
    },
    ClipInstanceSetWeight {
        entity: Entity,
        instance_id: ClipInstanceId,
        weight: f32,
    },
    ClipInstanceSetBlendMode {
        entity: Entity,
        instance_id: ClipInstanceId,
        blend_mode: BlendMode,
    },

    ClipGroupCreate {
        entity: Entity,
        name: String,
    },
    ClipGroupDelete {
        entity: Entity,
        group_id: ClipGroupId,
    },
    ClipGroupAddInstance {
        entity: Entity,
        group_id: ClipGroupId,
        instance_id: ClipInstanceId,
    },
    ClipGroupRemoveInstance {
        entity: Entity,
        group_id: ClipGroupId,
        instance_id: ClipInstanceId,
    },
    ClipGroupToggleMute {
        entity: Entity,
        group_id: ClipGroupId,
    },
    ClipGroupSetWeight {
        entity: Entity,
        group_id: ClipGroupId,
        weight: f32,
    },

    Undo,
    Redo,

    ClipInstanceAdd {
        entity: Entity,
        source_id: SourceClipId,
        start_time: f32,
    },
    ClipBrowserCreateEmpty,
    ClipBrowserDuplicate(SourceClipId),
    ClipBrowserDelete(SourceClipId),
    ClipBrowserLoadFromFile,
    ClipBrowserSaveToFile(SourceClipId),
    ClipBrowserExportFbx(SourceClipId),
    ClipBrowserExportGltf(SourceClipId),
    ClipBrowserExportGltfAnimationOnly(SourceClipId),

    SaveScene,

    PoseLibrarySaveCurrent {
        name: String,
    },
    PoseLibraryApply(SourceClipId),
    PoseLibraryDelete(SourceClipId),

    CreateTestConstraints,
    ClearTestConstraints,

    AddTestSpringBones,
    ClearSpringBones,
    SpringBoneBake,
    SpringBoneDiscardBake,
    SpringBoneSaveBake,
    SpringBoneRebake,

    ConstraintAdd {
        entity: Entity,
        constraint_type_index: u8,
    },
    ConstraintRemove {
        entity: Entity,
        constraint_id: ConstraintId,
    },
    ConstraintUpdate {
        entity: Entity,
        constraint_id: ConstraintId,
        constraint: ConstraintType,
    },
    ConstraintBakeToKeyframes {
        entity: Entity,
        sample_fps: f32,
    },

    SpringChainAdd {
        entity: Entity,
        root_bone_id: BoneId,
        chain_length: u32,
    },
    SpringChainRemove {
        entity: Entity,
        chain_id: SpringChainId,
    },
    SpringChainUpdate {
        entity: Entity,
        chain_id: SpringChainId,
        chain: SpringChain,
    },
    SpringJointUpdate {
        entity: Entity,
        chain_id: SpringChainId,
        joint_index: usize,
        joint: SpringJointParam,
    },
    SpringColliderAdd {
        entity: Entity,
        bone_id: BoneId,
        shape: ColliderShape,
    },
    SpringColliderRemove {
        entity: Entity,
        collider_id: SpringColliderId,
    },
    SpringColliderUpdate {
        entity: Entity,
        collider_id: SpringColliderId,
        collider: SpringColliderDef,
    },
    SpringColliderGroupAdd {
        entity: Entity,
        name: String,
    },
    SpringColliderGroupRemove {
        entity: Entity,
        group_id: SpringColliderGroupId,
    },
    SpringColliderGroupUpdate {
        entity: Entity,
        group_id: SpringColliderGroupId,
        group: SpringColliderGroup,
    },
    SpringBoneToggleGizmo(bool),

    BoneSetKey,

    SetBoneDisplayStyle(BoneDisplayStyle),
    SetBoneInFront(bool),
    SetBoneDistanceScaling(bool),
    SetBoneDistanceScaleFactor(f32),

    #[cfg(feature = "ml")]
    CurveSuggestionRequest {
        bone_id: BoneId,
        property_type: PropertyType,
    },
    #[cfg(feature = "ml")]
    CurveSuggestionAccept,
    #[cfg(feature = "ml")]
    CurveSuggestionDismiss,

    #[cfg(feature = "auto-rig")]
    TextToAnimationGenerate {
        prompt: String,
        duration_seconds: f32,
    },
    #[cfg(feature = "auto-rig")]
    TextToAnimationCancel,
    #[cfg(feature = "auto-rig")]
    ModelLoadedFromMemory {
        source: ModelLoadSource,
    },

    #[cfg(feature = "auto-rig")]
    TextToMeshGenerate {
        prompt: String,
        target_faces: u32,
        seed: u32,
        input_mode: crate::grpc::MeshInputMode,
        input_image_png: Option<Vec<u8>>,
        model_type: crate::grpc::MeshModelType,
        t2i_model_type: crate::grpc::TextToImageModelType,
    },
    #[cfg(feature = "auto-rig")]
    TextToMeshApply,
    #[cfg(feature = "auto-rig")]
    TextToMeshCancel,

    #[cfg(feature = "auto-rig")]
    AutoRigGenerate {
        num_sample_points: u32,
    },
    #[cfg(feature = "auto-rig")]
    AutoRigApply,
    #[cfg(feature = "auto-rig")]
    AutoRigDiscard,

    ExportModelGltf,

    ResampleSelectedModelAnimations {
        fps: f32,
    },

    TimelineZoomIn {
        max_zoom: f32,
    },
    TimelineZoomOut {
        min_zoom: f32,
    },

    SetBoneGizmoVisible(bool),
    SetWeightHeatmapEnabled(bool),
    SetTransformGizmoMode(TransformGizmoMode),
    SetTransformGizmoSpace(CoordinateSpace),
    UpdateTransformGizmoState(Box<TransformGizmoState>),
    UpdateDepthOfField(DepthOfField),
    UpdatePhysicalCamera(PhysicalCameraParameters),
    UpdateAutoExposure(AutoExposure),
    UpdateOnionSkinning(OnionSkinningConfig),
    UpdateFlameEffect(Box<FlameEffect>),
    AddFlame,
    UpdateFlameRenderSettings(FlameRenderSettings),
    SetGridShowYAxis(bool),
    ClearMessageLog,
    InsertFlameKey { param: FlameParam, value: f32 },
    ClearFlameKeys,
    SelectFlameInstance(usize),
    ToggleFlameCurves,
    MoveFlameKey {
        param: FlameParam,
        old_time: f32,
        new_time: f32,
        new_value: f32,
    },
    DeleteFlameKeyExact {
        param: FlameParam,
        time: f32,
    },
    ToggleFlameCurveParam {
        param: FlameParam,
    },
}

#[derive(Default)]
pub struct UIEventQueue {
    events: Vec<UIEvent>,
}

impl UIEventQueue {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn send(&mut self, event: UIEvent) {
        self.events.push(event);
    }

    pub fn drain(&mut self) -> impl Iterator<Item = UIEvent> + '_ {
        self.events.drain(..)
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }
}
