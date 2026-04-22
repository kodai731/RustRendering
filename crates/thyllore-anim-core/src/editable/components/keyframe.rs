use serde::{Deserialize, Serialize};

pub type SourceClipId = u64;
pub type ClipInstanceId = u64;
pub type KeyframeId = u64;
pub type CurveId = u64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationType {
    #[default]
    Linear,
    Bezier,
    Stepped,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TangentWeightMode {
    #[default]
    NonWeighted,
    Weighted,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TangentType {
    #[default]
    Manual,
    Spline,
    Flat,
    Linear,
    Clamped,
    Plateau,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BezierHandle {
    pub time_offset: f32,
    pub value_offset: f32,
}

impl BezierHandle {
    pub fn new(time_offset: f32, value_offset: f32) -> Self {
        Self {
            time_offset,
            value_offset,
        }
    }

    pub fn linear() -> Self {
        Self {
            time_offset: 0.0,
            value_offset: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditableKeyframe {
    pub id: KeyframeId,
    pub time: f32,
    pub value: f32,
    pub in_tangent: BezierHandle,
    pub out_tangent: BezierHandle,
    #[serde(default)]
    pub interpolation: InterpolationType,
    #[serde(default)]
    pub weight_mode: TangentWeightMode,
    #[serde(default)]
    pub tangent_type: TangentType,
}

impl EditableKeyframe {
    pub fn new(id: KeyframeId, time: f32, value: f32) -> Self {
        Self {
            id,
            time,
            value,
            in_tangent: BezierHandle::linear(),
            out_tangent: BezierHandle::linear(),
            interpolation: InterpolationType::Linear,
            weight_mode: TangentWeightMode::NonWeighted,
            tangent_type: TangentType::Manual,
        }
    }

    pub fn with_tangents(
        id: KeyframeId,
        time: f32,
        value: f32,
        in_tangent: BezierHandle,
        out_tangent: BezierHandle,
    ) -> Self {
        Self {
            id,
            time,
            value,
            in_tangent,
            out_tangent,
            interpolation: InterpolationType::Linear,
            weight_mode: TangentWeightMode::NonWeighted,
            tangent_type: TangentType::Manual,
        }
    }
}

impl Default for EditableKeyframe {
    fn default() -> Self {
        Self {
            id: 0,
            time: 0.0,
            value: 0.0,
            in_tangent: BezierHandle::linear(),
            out_tangent: BezierHandle::linear(),
            interpolation: InterpolationType::Linear,
            weight_mode: TangentWeightMode::NonWeighted,
            tangent_type: TangentType::Manual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serde_backward_compat_no_tangent_type() {
        let json = r#"{
            "id": 1,
            "time": 0.5,
            "value": 3.0,
            "in_tangent": {"time_offset": -0.1, "value_offset": -0.5},
            "out_tangent": {"time_offset": 0.1, "value_offset": 0.5}
        }"#;

        let kf: EditableKeyframe = serde_json::from_str(json).unwrap();
        assert_eq!(kf.tangent_type, TangentType::Manual);
        assert_eq!(kf.interpolation, InterpolationType::Linear);
        assert_eq!(kf.id, 1);
        assert!((kf.time - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tangent_type_serde_roundtrip() {
        let variants = [
            TangentType::Manual,
            TangentType::Spline,
            TangentType::Flat,
            TangentType::Linear,
            TangentType::Clamped,
            TangentType::Plateau,
        ];

        for variant in &variants {
            let mut kf = EditableKeyframe::new(1, 1.0, 5.0);
            kf.tangent_type = *variant;

            let json = serde_json::to_string(&kf).unwrap();
            let deserialized: EditableKeyframe = serde_json::from_str(&json).unwrap();

            assert_eq!(
                deserialized.tangent_type, *variant,
                "Roundtrip failed for {:?}",
                variant
            );
        }
    }

    #[test]
    fn test_serde_backward_compat_no_weight_mode() {
        let json = r#"{
            "id": 1,
            "time": 0.5,
            "value": 1.0,
            "in_tangent": { "time_offset": -0.1, "value_offset": -0.2 },
            "out_tangent": { "time_offset": 0.1, "value_offset": 0.2 },
            "interpolation": "Bezier"
        }"#;

        let kf: EditableKeyframe = serde_json::from_str(json).expect("Should deserialize");
        assert_eq!(kf.id, 1);
        assert_eq!(kf.weight_mode, TangentWeightMode::NonWeighted);
    }
}
