//! Curve Copilot prediction quality on a real run-animation glTF.
//!
//! Loads `t2a_run_hard1.glb`, picks 10 random (bone, property, anchor_time) cases,
//! runs curve copilot inference, and asserts:
//! - tangent_x sign correctness = 100% (Phase 10 design: tangent_x is structurally
//!   constrained to in_x <= 0 and out_x >= 0)
//! - value direction trend match >= 90% (model should predict the same rising /
//!   falling direction as the actual curve in at least 9 out of 10 cases)
//!
//! Skipped automatically when the glb or curve_copilot ONNX is not present.

#![cfg(feature = "ml")]

use std::collections::HashMap;
use std::path::Path;

use thyllore_animation::animation::editable::{
    clip_from_animation, curve_sample, EditableAnimationClip, EditableKeyframe, PropertyType,
};
use thyllore_animation::animation::BoneId;
use thyllore_animation::ecs::resource::{
    build_rest_position_cache, BoneNameTokenCache, BoneRestPositionCache, BoneTopologyCache,
};
use thyllore_animation::ecs::systems::curve_suggestion_bone_context::{
    build_bone_context_arrays, BoneContextRequest,
};
use thyllore_animation::ml::resolve_curve_copilot_model_path;
use thyllore_importer_core::{load_gltf_file, ModelLoadResult};
use thyllore_ml_core::copilot::context::flatten_context;
use thyllore_ml_core::copilot::input::{BoneContextInput, CONTEXT_KEYFRAME_COUNT};
use thyllore_ml_core::copilot::property::{property_kind_to_id, PropertyKind};
use thyllore_ml_core::copilot::query::generate_query_times;
use thyllore_ml_core::copilot::session::{CurveCopilotRequest, Session};
use thyllore_ml_core::copilot::tangent_synth::{resolve_in_tangent, resolve_out_tangent};
use thyllore_ml_core::copilot::window::sample_window;
use thyllore_ml_core::{compute_bone_name_tokens, compute_bone_topology};
use thyllore_model_core::Skeleton;

const GLB_PATH: &str = "/home/kodai/Projects/AnimationModelTraining/tests/assets/models/unirig/charactergen/t2a_run/t2a_run_hard1.glb";
const NUM_CASES: usize = 10;
const MAX_PICK_ATTEMPTS: usize = 500;
const RNG_SEED: u64 = 0x5eed_1234_a5a5_a5a5;
const MIN_CURVE_STD: f32 = 0.01;
const MIN_KFS_BEFORE_ANCHOR: usize = 3;
const REAL_TREND_NOISE_TOLERANCE: f32 = 1.0e-4;
const REQUIRED_DIRECTION_MATCH_RATIO: f32 = 0.9;

const ALL_PROPERTY_TYPES: [PropertyType; 9] = [
    PropertyType::TranslationX,
    PropertyType::TranslationY,
    PropertyType::TranslationZ,
    PropertyType::RotationX,
    PropertyType::RotationY,
    PropertyType::RotationZ,
    PropertyType::ScaleX,
    PropertyType::ScaleY,
    PropertyType::ScaleZ,
];

struct XorshiftRng {
    state: u64,
}

impl XorshiftRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_range(&mut self, max_exclusive: usize) -> usize {
        if max_exclusive == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max_exclusive
    }
}

#[test]
fn curve_copilot_predictions_on_t2a_run_hard1() {
    if !Path::new(GLB_PATH).exists() {
        eprintln!("Skipping: glb not found at {GLB_PATH}");
        return;
    }
    let Some(onnx_path) = resolve_curve_copilot_model_path() else {
        eprintln!("Skipping: curve_copilot ONNX not found. Set THYLLORE_SHARED_DATA_DIR.");
        return;
    };

    let gltf_result = unsafe { load_gltf_file(GLB_PATH).expect("load glb") };
    let load_result = ModelLoadResult::from_gltf(gltf_result);

    assert!(!load_result.skeletons.is_empty(), "glb has no skeleton");
    assert!(!load_result.clips.is_empty(), "glb has no animation clip");

    let skeleton = &load_result.skeletons[0];
    let source_clip = &load_result.clips[0];

    let bone_names: HashMap<BoneId, String> = skeleton
        .bones
        .iter()
        .map(|b| (b.id, b.name.clone()))
        .collect();
    let editable_clip = clip_from_animation(0, source_clip, &bone_names);

    let topology_cache = build_topology_cache(skeleton);
    let name_token_cache = build_name_token_cache(skeleton);
    let rest_position_cache = build_rest_position_cache(skeleton);

    let mut session = Session::from_onnx_path(&onnx_path).expect("load onnx session");
    let max_steps = session
        .max_steps()
        .expect("ONNX missing max_steps metadata");

    let mut rng = XorshiftRng::new(RNG_SEED);
    let cases = pick_random_cases(&editable_clip, skeleton, max_steps, NUM_CASES, &mut rng);
    assert_eq!(
        cases.len(),
        NUM_CASES,
        "failed to pick {NUM_CASES} valid cases from clip"
    );

    let mut tangent_violations = 0usize;
    let mut value_direction_matches = 0usize;

    for (i, case) in cases.iter().enumerate() {
        let result = run_case(
            &editable_clip,
            skeleton,
            &topology_cache,
            &name_token_cache,
            &rest_position_cache,
            &mut session,
            max_steps,
            case,
        );

        let has_tangent_violation = result.predictions.iter().any(|p| {
            let tan_in_x_denorm = p.tangent_in.0 * editable_clip.duration;
            let tan_out_x_denorm = p.tangent_out.0 * editable_clip.duration;
            tan_in_x_denorm > 0.0 || tan_out_x_denorm < 0.0
        });
        if has_tangent_violation {
            tangent_violations += 1;
        }

        let real_first = result.real_samples.first().copied().unwrap_or(0.0);
        let real_last = result.real_samples.last().copied().unwrap_or(0.0);
        let pred_first = result.pred_values.first().copied().unwrap_or(0.0);
        let pred_last = result.pred_values.last().copied().unwrap_or(0.0);

        let real_trend = real_last - real_first;
        let pred_trend = pred_last - pred_first;

        let direction_matches = real_trend.abs() < REAL_TREND_NOISE_TOLERANCE
            || real_trend.signum() == pred_trend.signum();
        if direction_matches {
            value_direction_matches += 1;
        }

        eprintln!(
            "case {:2}: bone={:3} prop={:?} anchor={:.3} | real_trend={:+.4} pred_trend={:+.4} \
             dir_match={} tan_violation={}",
            i,
            case.bone_id,
            case.property_type,
            case.anchor_time,
            real_trend,
            pred_trend,
            direction_matches,
            has_tangent_violation,
        );
    }

    eprintln!(
        "Summary: tangent_violations={}/{} value_direction_matches={}/{} ({}%)",
        tangent_violations,
        NUM_CASES,
        value_direction_matches,
        NUM_CASES,
        (value_direction_matches as f32 / NUM_CASES as f32 * 100.0) as u32,
    );

    assert_eq!(
        tangent_violations, 0,
        "tangent_x sign must be 100% correct, got {tangent_violations} cases with violations"
    );

    let direction_match_ratio = value_direction_matches as f32 / NUM_CASES as f32;
    assert!(
        direction_match_ratio >= REQUIRED_DIRECTION_MATCH_RATIO,
        "value direction match should be >= {:.0}%, got {}/{} ({:.0}%)",
        REQUIRED_DIRECTION_MATCH_RATIO * 100.0,
        value_direction_matches,
        NUM_CASES,
        direction_match_ratio * 100.0,
    );
}

struct TestCase {
    bone_id: BoneId,
    property_type: PropertyType,
    anchor_time: f32,
}

struct CaseResult {
    predictions: Vec<thyllore_ml_core::copilot::CopilotStepPrediction>,
    real_samples: Vec<f32>,
    pred_values: Vec<f32>,
}

fn build_topology_cache(skeleton: &Skeleton) -> BoneTopologyCache {
    let mut cache = BoneTopologyCache::default();
    cache.features = compute_bone_topology(skeleton);
    cache
}

fn build_name_token_cache(skeleton: &Skeleton) -> BoneNameTokenCache {
    let mut cache = BoneNameTokenCache::default();
    cache.tokens = compute_bone_name_tokens(skeleton);
    cache
}

fn pick_random_cases(
    clip: &EditableAnimationClip,
    skeleton: &Skeleton,
    max_steps: usize,
    n: usize,
    rng: &mut XorshiftRng,
) -> Vec<TestCase> {
    let mut cases = Vec::with_capacity(n);
    let mut attempts = 0;
    while cases.len() < n && attempts < MAX_PICK_ATTEMPTS {
        attempts += 1;
        let Some(case) = try_pick_case(clip, skeleton, max_steps, rng) else {
            continue;
        };
        cases.push(case);
    }
    cases
}

fn try_pick_case(
    clip: &EditableAnimationClip,
    skeleton: &Skeleton,
    max_steps: usize,
    rng: &mut XorshiftRng,
) -> Option<TestCase> {
    let bone_idx = rng.next_range(skeleton.bones.len());
    let bone_id = skeleton.bones[bone_idx].id;
    let property_type = ALL_PROPERTY_TYPES[rng.next_range(ALL_PROPERTY_TYPES.len())];

    let track = clip.get_track(bone_id)?;
    let curve = track.get_curve(property_type);
    let total_kfs = curve.keyframes.len();
    if total_kfs < MIN_KFS_BEFORE_ANCHOR + max_steps {
        return None;
    }

    let mut sorted_times: Vec<f32> = curve.keyframes.iter().map(|kf| kf.time).collect();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let earliest_anchor_idx = MIN_KFS_BEFORE_ANCHOR - 1;
    let latest_anchor_idx = sorted_times.len() - 1 - max_steps;
    if earliest_anchor_idx > latest_anchor_idx {
        return None;
    }
    let anchor_choice_count = latest_anchor_idx - earliest_anchor_idx + 1;
    let anchor_idx = earliest_anchor_idx + rng.next_range(anchor_choice_count);
    let anchor_time = sorted_times[anchor_idx];

    let value_range = curve_value_range(&curve.keyframes, anchor_time);
    if value_range < MIN_CURVE_STD {
        return None;
    }

    Some(TestCase {
        bone_id,
        property_type,
        anchor_time,
    })
}

fn curve_value_range(keyframes: &[EditableKeyframe], anchor_time: f32) -> f32 {
    let values: Vec<f32> = keyframes
        .iter()
        .filter(|kf| kf.time <= anchor_time + 1.0e-6)
        .map(|kf| kf.value)
        .collect();
    if values.is_empty() {
        return 0.0;
    }
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    max - min
}

fn run_case(
    clip: &EditableAnimationClip,
    skeleton: &Skeleton,
    topology_cache: &BoneTopologyCache,
    name_token_cache: &BoneNameTokenCache,
    rest_position_cache: &BoneRestPositionCache,
    session: &mut Session,
    max_steps: usize,
    case: &TestCase,
) -> CaseResult {
    let clip_duration = clip.duration;
    let track = clip.get_track(case.bone_id).expect("track exists");
    let curve = track.get_curve(case.property_type);

    let mut sorted: Vec<&EditableKeyframe> = curve.keyframes.iter().collect();
    sorted.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let cutoff = sorted
        .iter()
        .position(|kf| kf.time > case.anchor_time + 1.0e-6)
        .unwrap_or(sorted.len());

    let mut times = Vec::with_capacity(cutoff);
    let mut values = Vec::with_capacity(cutoff);
    let mut in_dt = Vec::with_capacity(cutoff);
    let mut in_dv = Vec::with_capacity(cutoff);
    let mut out_dt = Vec::with_capacity(cutoff);
    let mut out_dv = Vec::with_capacity(cutoff);
    for i in 0..cutoff {
        let kf = sorted[i];
        let prev = sorted.get(i.wrapping_sub(1)).map(|p| (p.time, p.value));
        let next = sorted.get(i + 1).map(|n| (n.time, n.value));
        let (idt, idv) = resolve_in_tangent(
            kf.in_tangent.time_offset,
            kf.in_tangent.value_offset,
            prev,
            kf.time,
            kf.value,
        );
        let (odt, odv) = resolve_out_tangent(
            kf.out_tangent.time_offset,
            kf.out_tangent.value_offset,
            kf.time,
            kf.value,
            next,
        );
        times.push(kf.time);
        values.push(kf.value);
        in_dt.push(idt);
        in_dv.push(idv);
        out_dt.push(odt);
        out_dv.push(odv);
    }

    let context = flatten_context(
        &times,
        &values,
        &in_dt,
        &in_dv,
        &out_dt,
        &out_dv,
        CONTEXT_KEYFRAME_COUNT,
        clip_duration,
    );

    let property_type_id = property_kind_to_id(property_type_to_kind(case.property_type));
    let topology_features = topology_cache.get(case.bone_id).to_vec();
    let bone_name_tokens = name_token_cache.get(case.bone_id).to_vec();

    let all_times: Vec<f32> = curve.keyframes.iter().map(|kf| kf.time).collect();
    let query_times = generate_query_times(&all_times, case.anchor_time, clip_duration, max_steps);

    let context_window_kfs = times.len().min(CONTEXT_KEYFRAME_COUNT);
    let context_start_index = times.len().saturating_sub(context_window_kfs);
    let context_start_time = times
        .get(context_start_index)
        .copied()
        .unwrap_or(case.anchor_time);
    let curve_window = sample_window(
        &times,
        &values,
        context_start_time,
        case.anchor_time,
        context.curve_mean,
        context.curve_std,
    );

    let bone_context = build_bone_context_arrays(BoneContextRequest {
        clip,
        skeleton,
        topology_cache,
        rest_position_cache,
        property_type: case.property_type,
        anchor_time: case.anchor_time,
        clip_duration,
    });

    let predictions = session
        .run_curve_copilot(CurveCopilotRequest {
            context: &context.flat,
            property_type_id,
            topology_features: &topology_features,
            bone_name_tokens: &bone_name_tokens,
            query_times: &query_times,
            curve_window: &curve_window,
            bone_context: BoneContextInput {
                keyframes: &bone_context.keyframes,
                topology: &bone_context.topology,
                rest_positions: &bone_context.rest_positions,
                mask: &bone_context.mask,
            },
        })
        .expect("inference failed");

    let denorm_query_times: Vec<f32> = query_times
        .iter()
        .map(|t| t * clip_duration.max(0.001))
        .collect();
    let real_samples: Vec<f32> = denorm_query_times
        .iter()
        .map(|t| curve_sample(curve, *t).unwrap_or(f32::NAN))
        .collect();
    let pred_values: Vec<f32> = predictions
        .iter()
        .map(|p| p.value * context.curve_std + context.curve_mean)
        .collect();

    CaseResult {
        predictions,
        real_samples,
        pred_values,
    }
}

fn property_type_to_kind(pt: PropertyType) -> PropertyKind {
    match pt {
        PropertyType::TranslationX => PropertyKind::TranslationX,
        PropertyType::TranslationY => PropertyKind::TranslationY,
        PropertyType::TranslationZ => PropertyKind::TranslationZ,
        PropertyType::RotationX => PropertyKind::RotationX,
        PropertyType::RotationY => PropertyKind::RotationY,
        PropertyType::RotationZ => PropertyKind::RotationZ,
        PropertyType::ScaleX => PropertyKind::ScaleX,
        PropertyType::ScaleY => PropertyKind::ScaleY,
        PropertyType::ScaleZ => PropertyKind::ScaleZ,
    }
}
