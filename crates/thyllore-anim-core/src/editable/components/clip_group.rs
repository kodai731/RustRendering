use super::keyframe::ClipInstanceId;

pub type ClipGroupId = u64;

#[derive(Clone, Debug)]
pub struct ClipGroup {
    pub id: ClipGroupId,
    pub name: String,
    pub instance_ids: Vec<ClipInstanceId>,
    pub muted: bool,
    pub weight: f32,
}

impl ClipGroup {
    pub fn new(id: ClipGroupId, name: String) -> Self {
        Self {
            id,
            name,
            instance_ids: Vec::new(),
            muted: false,
            weight: 1.0,
        }
    }

    pub fn contains_instance(&self, instance_id: ClipInstanceId) -> bool {
        self.instance_ids.contains(&instance_id)
    }

    pub fn add_instance(&mut self, instance_id: ClipInstanceId) {
        if !self.instance_ids.contains(&instance_id) {
            self.instance_ids.push(instance_id);
        }
    }

    pub fn remove_instance(&mut self, instance_id: ClipInstanceId) {
        self.instance_ids.retain(|&id| id != instance_id);
    }
}
