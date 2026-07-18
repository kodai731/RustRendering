use crate::components::fbx::*;
use crate::fbx_animation::{FbxBoneExport, FbxConnection, FbxCurveNodeExport};

pub(crate) fn generate_bone_connections(bones: &[FbxBoneExport], connections: &mut Vec<FbxConnection>) {
    for bone in bones {
        let parent_uid = bone.parent_model_uid.unwrap_or(0);
        connections.push(FbxConnection::OO {
            child: bone.model_uid,
            parent: parent_uid,
        });

        if let Some(attr_uid) = bone.node_attribute_uid {
            connections.push(FbxConnection::OO {
                child: attr_uid,
                parent: bone.model_uid,
            });
        }
    }
}

pub(crate) fn generate_mesh_connections(
    mesh_models: &[FbxMeshModelExport],
    geometries: &[FbxGeometryExport],
    materials: &[FbxMaterialExport],
    textures: &[FbxTextureExport],
    skins: &[FbxSkinExport],
    connections: &mut Vec<FbxConnection>,
) {
    for (i, mesh_model) in mesh_models.iter().enumerate() {
        let parent_uid = mesh_model.parent_bone_uid.unwrap_or(0);
        connections.push(FbxConnection::OO {
            child: mesh_model.uid,
            parent: parent_uid,
        });

        if i < geometries.len() {
            connections.push(FbxConnection::OO {
                child: geometries[i].uid,
                parent: mesh_model.uid,
            });
        }
    }

    for material in materials {
        connections.push(FbxConnection::OO {
            child: material.uid,
            parent: material.mesh_model_uid,
        });
    }

    for texture in textures {
        connections.push(FbxConnection::OP {
            child: texture.texture_uid,
            parent: texture.material_uid,
            property: "DiffuseColor".to_string(),
        });

        connections.push(FbxConnection::OO {
            child: texture.video_uid,
            parent: texture.texture_uid,
        });
    }

    for skin in skins {
        connections.push(FbxConnection::OO {
            child: skin.skin_uid,
            parent: skin.geometry_uid,
        });

        for cluster in &skin.clusters {
            connections.push(FbxConnection::OO {
                child: cluster.uid,
                parent: skin.skin_uid,
            });

            connections.push(FbxConnection::OO {
                child: cluster.bone_model_uid,
                parent: cluster.uid,
            });
        }
    }
}

pub(crate) fn generate_animation_connections(
    stack_uid: i64,
    layer_uid: i64,
    curve_nodes: &[FbxCurveNodeExport],
    connections: &mut Vec<FbxConnection>,
) {
    connections.push(FbxConnection::OO {
        child: stack_uid,
        parent: 0,
    });

    connections.push(FbxConnection::OO {
        child: layer_uid,
        parent: stack_uid,
    });

    for cn in curve_nodes {
        connections.push(FbxConnection::OO {
            child: cn.uid,
            parent: layer_uid,
        });

        connections.push(FbxConnection::OP {
            child: cn.uid,
            parent: cn.bone_model_uid,
            property: cn.channel.property_name().to_string(),
        });

        let axis_names = ["d|X", "d|Y", "d|Z"];
        for (i, axis) in axis_names.iter().enumerate() {
            if let Some(curve_uid) = cn.curve_uids[i] {
                connections.push(FbxConnection::OP {
                    child: curve_uid,
                    parent: cn.uid,
                    property: axis.to_string(),
                });
            }
        }
    }
}
