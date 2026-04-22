use std::collections::HashMap;

use crate::{BoneId, Skeleton};

#[derive(Clone, Debug)]
pub struct BoneTopologyFeatures {
    pub hierarchy_depth: f32,
    pub child_count: f32,
    pub sibling_index: f32,
    pub chain_length_to_leaf: f32,
    pub parent_child_count: f32,
    pub is_leaf: f32,
}

impl Default for BoneTopologyFeatures {
    fn default() -> Self {
        Self {
            hierarchy_depth: 0.0,
            child_count: 0.0,
            sibling_index: 0.0,
            chain_length_to_leaf: 0.0,
            parent_child_count: 0.0,
            is_leaf: 0.0,
        }
    }
}

impl BoneTopologyFeatures {
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.hierarchy_depth,
            self.child_count,
            self.sibling_index,
            self.chain_length_to_leaf,
            self.parent_child_count,
            self.is_leaf,
        ]
    }
}

pub fn compute_bone_topology(skeleton: &Skeleton) -> HashMap<BoneId, BoneTopologyFeatures> {
    let max_depth = compute_max_depth(skeleton) as f32;
    let max_children = compute_max_children(skeleton) as f32;
    let max_chain = compute_global_max_chain(skeleton) as f32;

    let depth_divisor = max_depth.max(1.0);
    let children_divisor = max_children.max(1.0);
    let chain_divisor = max_chain.max(1.0);

    let mut result = HashMap::new();

    for bone in &skeleton.bones {
        let depth = compute_depth(skeleton, bone.id) as f32;
        let child_count = bone.children.len() as f32;
        let chain_to_leaf = compute_chain_length_to_leaf(skeleton, bone.id) as f32;
        let sibling_index = compute_sibling_index(skeleton, bone.id) as f32;

        let parent_child_count = bone
            .parent_id
            .and_then(|pid| skeleton.get_bone(pid))
            .map(|parent| parent.children.len() as f32)
            .unwrap_or(0.0);

        let is_leaf = if bone.children.is_empty() { 1.0 } else { 0.0 };

        let sibling_divisor = parent_child_count.max(1.0);

        let features = BoneTopologyFeatures {
            hierarchy_depth: depth / depth_divisor,
            child_count: child_count / children_divisor,
            sibling_index: sibling_index / sibling_divisor,
            chain_length_to_leaf: chain_to_leaf / chain_divisor,
            parent_child_count: parent_child_count / children_divisor,
            is_leaf,
        };

        result.insert(bone.id, features);
    }

    result
}

fn compute_depth(skeleton: &Skeleton, bone_id: BoneId) -> u32 {
    let mut depth = 0u32;
    let mut current = bone_id;

    while let Some(bone) = skeleton.get_bone(current) {
        match bone.parent_id {
            Some(parent_id) => {
                depth += 1;
                current = parent_id;
            }
            None => break,
        }
    }

    depth
}

fn compute_max_depth(skeleton: &Skeleton) -> u32 {
    skeleton
        .bones
        .iter()
        .map(|bone| compute_depth(skeleton, bone.id))
        .max()
        .unwrap_or(0)
}

fn compute_max_children(skeleton: &Skeleton) -> u32 {
    skeleton
        .bones
        .iter()
        .map(|bone| bone.children.len() as u32)
        .max()
        .unwrap_or(0)
}

fn compute_chain_length_to_leaf(skeleton: &Skeleton, bone_id: BoneId) -> u32 {
    let bone = match skeleton.get_bone(bone_id) {
        Some(b) => b,
        None => return 0,
    };

    if bone.children.is_empty() {
        return 0;
    }

    bone.children
        .iter()
        .map(|&child_id| 1 + compute_chain_length_to_leaf(skeleton, child_id))
        .max()
        .unwrap_or(0)
}

fn compute_global_max_chain(skeleton: &Skeleton) -> u32 {
    skeleton
        .bones
        .iter()
        .map(|bone| compute_chain_length_to_leaf(skeleton, bone.id))
        .max()
        .unwrap_or(0)
}

fn compute_sibling_index(skeleton: &Skeleton, bone_id: BoneId) -> u32 {
    let bone = match skeleton.get_bone(bone_id) {
        Some(b) => b,
        None => return 0,
    };

    let parent_id = match bone.parent_id {
        Some(pid) => pid,
        None => return 0,
    };

    let parent = match skeleton.get_bone(parent_id) {
        Some(p) => p,
        None => return 0,
    };

    parent
        .children
        .iter()
        .position(|&cid| cid == bone_id)
        .unwrap_or(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_skeleton() -> Skeleton {
        let mut skeleton = Skeleton::new("test");
        skeleton.add_bone("root", None);
        skeleton.add_bone("child", Some(0));
        skeleton.add_bone("leaf", Some(1));
        skeleton
    }

    #[test]
    fn test_three_bone_chain() {
        let skeleton = create_test_skeleton();
        let features = compute_bone_topology(&skeleton);

        assert_eq!(features.len(), 3);

        let root = &features[&0];
        assert!((root.hierarchy_depth - 0.0).abs() < 1e-6);
        assert!((root.is_leaf - 0.0).abs() < 1e-6);

        let leaf = &features[&2];
        assert!((leaf.is_leaf - 1.0).abs() < 1e-6);
        assert!((leaf.chain_length_to_leaf - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalization_range() {
        let skeleton = create_test_skeleton();
        let features = compute_bone_topology(&skeleton);

        for (_id, feat) in &features {
            let values = feat.to_vec();
            for v in &values {
                assert!(*v >= 0.0 && *v <= 1.0, "value {} out of [0,1] range", v);
            }
        }
    }

    #[test]
    fn test_root_features() {
        let skeleton = create_test_skeleton();
        let features = compute_bone_topology(&skeleton);
        let root = &features[&0];

        assert!((root.hierarchy_depth - 0.0).abs() < 1e-6);
        assert!((root.is_leaf - 0.0).abs() < 1e-6);
        assert!(root.chain_length_to_leaf > 0.0);
    }

    #[test]
    fn test_leaf_features() {
        let skeleton = create_test_skeleton();
        let features = compute_bone_topology(&skeleton);
        let leaf = &features[&2];

        assert!((leaf.is_leaf - 1.0).abs() < 1e-6);
        assert!((leaf.chain_length_to_leaf - 0.0).abs() < 1e-6);
        assert!((leaf.hierarchy_depth - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_vec_length() {
        let features = BoneTopologyFeatures::default();
        assert_eq!(features.to_vec().len(), 6);
    }

    #[test]
    fn test_branching_skeleton() {
        let mut skeleton = Skeleton::new("branching");
        skeleton.add_bone("root", None);
        skeleton.add_bone("left", Some(0));
        skeleton.add_bone("right", Some(0));
        skeleton.add_bone("left_leaf", Some(1));

        let features = compute_bone_topology(&skeleton);

        let root = &features[&0];
        assert!(root.child_count > 0.0);

        let right = &features[&2];
        assert!(right.sibling_index > 0.0);
    }
}
