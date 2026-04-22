use crate::{BoneId, Skeleton};

pub fn build_evaluation_order(
    chains: &[(Vec<BoneId>,)],
    skeleton: &Skeleton,
) -> Vec<(usize, usize)> {
    let mut order: Vec<(usize, usize, u32)> = Vec::new();

    for (chain_idx, (joints,)) in chains.iter().enumerate() {
        for (joint_idx, &bone_id) in joints.iter().enumerate() {
            let depth = compute_bone_depth(skeleton, bone_id);
            order.push((chain_idx, joint_idx, depth));
        }
    }

    order.sort_by_key(|&(_, _, depth)| depth);
    order.iter().map(|&(ci, ji, _)| (ci, ji)).collect()
}

fn compute_bone_depth(skeleton: &Skeleton, bone_id: BoneId) -> u32 {
    let mut depth = 0;
    let mut current = bone_id;
    while let Some(bone) = skeleton.get_bone(current) {
        match bone.parent_id {
            Some(parent) => {
                depth += 1;
                current = parent;
            }
            None => break,
        }
    }
    depth
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_evaluation_order_parents_first() {
        let mut skeleton = Skeleton::new("test");
        let root = skeleton.add_bone("root", None);
        let child = skeleton.add_bone("child", Some(root));
        let grandchild = skeleton.add_bone("grandchild", Some(child));

        let chains = vec![(vec![grandchild, child, root],)];
        let order = build_evaluation_order(&chains, &skeleton);

        let first_bone = chains[order[0].0].0[order[0].1];
        let last_bone = chains[order[order.len() - 1].0].0[order[order.len() - 1].1];

        assert_eq!(first_bone, root);
        assert_eq!(last_bone, grandchild);
    }
}
