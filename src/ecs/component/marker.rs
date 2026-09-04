#[derive(Clone, Copy, Debug, Default)]
pub struct GridMarker;

#[derive(Clone, Copy, Debug, Default)]
pub struct GridGizmoMarker;

#[derive(Clone, Copy, Debug, Default)]
pub struct LightGizmoMarker;

#[derive(Clone, Copy, Debug, Default)]
pub struct LightBillboardMarker;

#[derive(Clone, Copy, Debug, Default)]
pub struct Animated;

#[derive(Clone, Copy, Debug, Default)]
pub struct Skinned;

#[derive(Clone, Copy, Debug, Default)]
pub struct Model;

#[derive(Clone, Copy, Debug, Default)]
pub struct Constrained;

#[derive(Clone, Copy, Debug, Default)]
pub struct WithSpringBone;

#[derive(Clone, Copy, Debug)]
pub struct DebugPrimitiveTag {
    pub kind: crate::ecs::events::DebugPrimitiveKind,
}

#[cfg(feature = "ml")]
#[derive(Clone, Copy, Debug, Default)]
pub struct WithInferenceActor;
