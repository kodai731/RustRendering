use serde::Deserialize;

use thyllore_anim_core::spring_bone::{
    ColliderShape, SpringBoneSetup, SpringChain, SpringColliderDef, SpringColliderGroup,
    SpringJointParam,
};

#[derive(Deserialize)]
struct VrmcSpringBone {
    colliders: Option<Vec<VrmcCollider>>,
    #[serde(rename = "colliderGroups")]
    collider_groups: Option<Vec<VrmcColliderGroup>>,
    springs: Option<Vec<VrmcSpring>>,
}

#[derive(Deserialize)]
struct VrmcCollider {
    node: u32,
    shape: VrmcColliderShape,
}

#[derive(Deserialize)]
struct VrmcColliderShape {
    sphere: Option<VrmcSphere>,
    capsule: Option<VrmcCapsule>,
}

#[derive(Deserialize)]
struct VrmcSphere {
    offset: Option<[f32; 3]>,
    radius: Option<f32>,
}

#[derive(Deserialize)]
struct VrmcCapsule {
    offset: Option<[f32; 3]>,
    radius: Option<f32>,
    tail: Option<[f32; 3]>,
}

#[derive(Deserialize)]
struct VrmcColliderGroup {
    name: Option<String>,
    colliders: Option<Vec<u32>>,
}

#[derive(Deserialize)]
struct VrmcSpring {
    name: Option<String>,
    joints: Vec<VrmcJoint>,
    #[serde(rename = "colliderGroups")]
    collider_groups: Option<Vec<u32>>,
    center: Option<u32>,
}

#[derive(Deserialize)]
struct VrmcJoint {
    node: u32,
    #[serde(rename = "hitRadius")]
    hit_radius: Option<f32>,
    stiffness: Option<f32>,
    #[serde(rename = "gravityPower")]
    gravity_power: Option<f32>,
    #[serde(rename = "gravityDir")]
    gravity_dir: Option<[f32; 3]>,
    #[serde(rename = "dragForce")]
    drag_force: Option<f32>,
}

pub fn parse_vrmc_spring_bone(
    extension_json: &serde_json::Value,
    resolve_bone_id: &dyn Fn(u32) -> Option<u32>,
) -> Option<SpringBoneSetup> {
    let vrmc: VrmcSpringBone = match serde_json::from_value(extension_json.clone()) {
        Ok(v) => v,
        Err(e) => {
            log!("Failed to parse VRMC_springBone: {}", e);
            return None;
        }
    };

    let colliders = convert_colliders(vrmc.colliders.as_deref().unwrap_or(&[]), resolve_bone_id);
    let collider_groups = convert_collider_groups(vrmc.collider_groups.as_deref().unwrap_or(&[]));
    let chains = convert_springs(vrmc.springs.as_deref().unwrap_or(&[]), resolve_bone_id);

    let next_collider_id = colliders.len() as u32;
    let next_group_id = collider_groups.len() as u32;
    let next_chain_id = chains.len() as u32;

    Some(SpringBoneSetup {
        chains,
        colliders,
        collider_groups,
        next_chain_id,
        next_collider_id,
        next_group_id,
    })
}

fn convert_colliders(
    vrmc_colliders: &[VrmcCollider],
    resolve_bone_id: &dyn Fn(u32) -> Option<u32>,
) -> Vec<SpringColliderDef> {
    let mut result = Vec::new();

    for (i, vc) in vrmc_colliders.iter().enumerate() {
        let Some(bone_id) = resolve_bone_id(vc.node) else {
            log!(
                "VRMC collider {}: node {} not found in skeleton, skipping",
                i,
                vc.node
            );
            continue;
        };

        let shape = if let Some(ref capsule) = vc.shape.capsule {
            let offset = capsule.offset.unwrap_or([0.0; 3]);
            let tail = capsule.tail.unwrap_or([0.0; 3]);
            let radius = capsule.radius.unwrap_or(0.0);
            ColliderShape::Capsule {
                radius,
                tail: cgmath::Vector3::new(
                    tail[0] - offset[0],
                    tail[1] - offset[1],
                    tail[2] - offset[2],
                ),
            }
        } else if let Some(ref sphere) = vc.shape.sphere {
            ColliderShape::Sphere {
                radius: sphere.radius.unwrap_or(0.0),
            }
        } else {
            ColliderShape::Sphere { radius: 0.0 }
        };

        let offset_arr = vc
            .shape
            .capsule
            .as_ref()
            .and_then(|c| c.offset)
            .or_else(|| vc.shape.sphere.as_ref().and_then(|s| s.offset))
            .unwrap_or([0.0; 3]);

        result.push(SpringColliderDef {
            id: i as u32,
            bone_id,
            offset: cgmath::Vector3::new(offset_arr[0], offset_arr[1], offset_arr[2]),
            shape,
        });
    }

    result
}

fn convert_collider_groups(vrmc_groups: &[VrmcColliderGroup]) -> Vec<SpringColliderGroup> {
    vrmc_groups
        .iter()
        .enumerate()
        .map(|(i, g)| SpringColliderGroup {
            id: i as u32,
            name: g.name.clone().unwrap_or_else(|| format!("group_{}", i)),
            collider_ids: g.colliders.clone().unwrap_or_default(),
        })
        .collect()
}

fn convert_springs(
    vrmc_springs: &[VrmcSpring],
    resolve_bone_id: &dyn Fn(u32) -> Option<u32>,
) -> Vec<SpringChain> {
    let mut result = Vec::new();

    for (i, vs) in vrmc_springs.iter().enumerate() {
        let joints: Vec<SpringJointParam> = vs
            .joints
            .iter()
            .filter_map(|vj| {
                let bone_id = resolve_bone_id(vj.node);
                if bone_id.is_none() {
                    log!(
                        "VRMC spring '{}' joint node {} not found, skipping",
                        vs.name.as_deref().unwrap_or("?"),
                        vj.node
                    );
                }
                bone_id.map(|bid| SpringJointParam {
                    bone_id: bid,
                    stiffness: vj.stiffness.unwrap_or(1.0),
                    drag_force: vj.drag_force.unwrap_or(0.5),
                    gravity_power: vj.gravity_power.unwrap_or(0.0),
                    gravity_dir: vj
                        .gravity_dir
                        .map(|d| cgmath::Vector3::new(d[0], d[1], d[2]))
                        .unwrap_or(cgmath::Vector3::new(0.0, -1.0, 0.0)),
                    hit_radius: vj.hit_radius.unwrap_or(0.0),
                })
            })
            .collect();

        if joints.is_empty() {
            log!(
                "VRMC spring '{}': no valid joints, skipping",
                vs.name.as_deref().unwrap_or("?")
            );
            continue;
        }

        let center_bone_id = vs.center.and_then(|node| resolve_bone_id(node));

        result.push(SpringChain {
            id: i as u32,
            name: vs.name.clone().unwrap_or_else(|| format!("spring_{}", i)),
            joints,
            collider_group_ids: vs.collider_groups.clone().unwrap_or_default(),
            center_bone_id,
            enabled: true,
        });
    }

    result
}
