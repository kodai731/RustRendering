use super::blend::{BlendMode, EaseType};
use super::keyframe::{ClipInstanceId, SourceClipId};

#[derive(Clone, Debug)]
pub struct ClipInstance {
    pub instance_id: ClipInstanceId,
    pub source_id: SourceClipId,
    pub start_time: f32,
    pub clip_in: f32,
    pub clip_out: f32,
    pub speed: f32,
    pub weight: f32,
    pub blend_mode: BlendMode,
    pub ease_in: EaseType,
    pub ease_out: EaseType,
    pub muted: bool,
    pub cycle_count: f32,
}

impl ClipInstance {
    pub fn new(instance_id: ClipInstanceId, source_id: SourceClipId, duration: f32) -> Self {
        Self {
            instance_id,
            source_id,
            start_time: 0.0,
            clip_in: 0.0,
            clip_out: duration,
            speed: 1.0,
            weight: 1.0,
            blend_mode: BlendMode::default(),
            ease_in: EaseType::default(),
            ease_out: EaseType::default(),
            muted: false,
            cycle_count: 1.0,
        }
    }

    pub fn effective_duration(&self) -> f32 {
        (self.clip_out - self.clip_in) / self.speed
    }

    pub fn end_time(&self) -> f32 {
        self.start_time + self.effective_duration() * self.cycle_count
    }

    pub fn is_active_at(&self, time: f32) -> bool {
        !self.muted && time >= self.start_time && time < self.end_time()
    }
}
