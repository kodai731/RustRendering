use thyllore_anim_core::editable::EditableAnimationClip;

use crate::fbx_animation::{
    build_channel_exports, FbxChannel, FbxCurveExport, FbxCurveNodeExport, UidAllocator,
};

pub(crate) fn build_animation_curves(
    clip: Option<&EditableAnimationClip>,
    bone_name_to_model_uid: &std::collections::HashMap<String, i64>,
    uid_alloc: &mut UidAllocator,
    inv_unit_scale: f32,
) -> (Vec<FbxCurveNodeExport>, Vec<FbxCurveExport>) {
    let mut curve_nodes = Vec::new();
    let mut curves = Vec::new();

    let Some(clip) = clip else {
        return (curve_nodes, curves);
    };

    for track in clip.tracks.values() {
        let bone_model_uid = match bone_name_to_model_uid.get(track.bone_name.as_str()) {
            Some(&uid) => uid,
            None => continue,
        };

        if let Some((node, node_curves)) = build_channel_exports(
            [
                &track.translation_x,
                &track.translation_y,
                &track.translation_z,
            ],
            bone_model_uid,
            FbxChannel::Translation,
            uid_alloc,
            inv_unit_scale,
        ) {
            curve_nodes.push(node);
            curves.extend(node_curves);
        }

        if let Some((node, node_curves)) = build_channel_exports(
            [&track.rotation_x, &track.rotation_y, &track.rotation_z],
            bone_model_uid,
            FbxChannel::Rotation,
            uid_alloc,
            inv_unit_scale,
        ) {
            curve_nodes.push(node);
            curves.extend(node_curves);
        }

        if let Some((node, node_curves)) = build_channel_exports(
            [&track.scale_x, &track.scale_y, &track.scale_z],
            bone_model_uid,
            FbxChannel::Scale,
            uid_alloc,
            inv_unit_scale,
        ) {
            curve_nodes.push(node);
            curves.extend(node_curves);
        }
    }

    (curve_nodes, curves)
}
