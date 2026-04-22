use cgmath::{Matrix4, SquareMatrix};
use std::collections::HashMap;

pub type BoneId = u32;
pub type SkeletonId = u32;

#[derive(Clone, Debug)]
pub struct Bone {
    pub id: BoneId,
    pub name: String,
    pub parent_id: Option<BoneId>,
    pub children: Vec<BoneId>,
    pub local_transform: Matrix4<f32>,
    pub inverse_bind_pose: Matrix4<f32>,
    pub node_index: Option<usize>,
}

impl Default for Bone {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            parent_id: None,
            children: Vec::new(),
            local_transform: Matrix4::identity(),
            inverse_bind_pose: Matrix4::identity(),
            node_index: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Skeleton {
    pub id: SkeletonId,
    pub name: String,
    pub bones: Vec<Bone>,
    pub bone_name_to_id: HashMap<String, BoneId>,
    pub root_bone_ids: Vec<BoneId>,
    pub root_transform: Matrix4<f32>,
}

impl Default for Skeleton {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            bones: Vec::new(),
            bone_name_to_id: HashMap::new(),
            root_bone_ids: Vec::new(),
            root_transform: Matrix4::identity(),
        }
    }
}

impl Skeleton {
    pub fn new(name: &str) -> Self {
        Self {
            id: 0,
            name: name.to_string(),
            bones: Vec::new(),
            bone_name_to_id: HashMap::new(),
            root_bone_ids: Vec::new(),
            root_transform: Matrix4::identity(),
        }
    }

    pub fn add_bone(&mut self, name: &str, parent_id: Option<BoneId>) -> BoneId {
        let id = self.bones.len() as BoneId;
        let bone = Bone {
            id,
            name: name.to_string(),
            parent_id,
            children: Vec::new(),
            local_transform: Matrix4::identity(),
            inverse_bind_pose: Matrix4::identity(),
            node_index: None,
        };

        self.bone_name_to_id.insert(name.to_string(), id);
        self.bones.push(bone);

        if let Some(parent) = parent_id {
            if let Some(parent_bone) = self.bones.get_mut(parent as usize) {
                parent_bone.children.push(id);
            }
        } else {
            self.root_bone_ids.push(id);
        }

        id
    }

    pub fn get_bone(&self, id: BoneId) -> Option<&Bone> {
        self.bones.get(id as usize)
    }

    pub fn get_bone_mut(&mut self, id: BoneId) -> Option<&mut Bone> {
        self.bones.get_mut(id as usize)
    }

    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    pub fn collect_descendants(&self, bone_id: BoneId) -> Vec<BoneId> {
        let mut result = Vec::new();
        self.collect_descendants_recursive(bone_id, &mut result);
        result
    }

    fn collect_descendants_recursive(&self, bone_id: BoneId, result: &mut Vec<BoneId>) {
        if let Some(bone) = self.get_bone(bone_id) {
            for &child_id in &bone.children {
                result.push(child_id);
                self.collect_descendants_recursive(child_id, result);
            }
        }
    }
}
