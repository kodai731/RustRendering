use imgui::Condition;

use crate::animation::{BoneId, Skeleton};
use crate::asset::AssetStorage;
use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::gizmo::{BoneDisplayStyle, BoneGizmoData};
use crate::ecs::resource::{HierarchyDisplayMode, HierarchyState};
use crate::ecs::systems::{hierarchy_is_bone_expanded, query_hierarchy_tree};
use crate::ecs::world::World;

use super::layout_snapshot::LayoutSnapshot;

pub fn build_hierarchy_window(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    world: &World,
    state: &HierarchyState,
    assets: &AssetStorage,
    layout: &LayoutSnapshot,
) {
    ui.window("Hierarchy")
        .position([0.0, 0.0], Condition::Always)
        .size(
            [layout.hierarchy_width, layout.hierarchy_height],
            Condition::Always,
        )
        .resizable(false)
        .movable(false)
        .collapsible(false)
        .bring_to_front_on_focus(false)
        .build(|| {
            build_mode_tabs(ui, ui_events, state);
            build_search_bar(ui, ui_events, state);
            ui.separator();

            match state.display_mode {
                HierarchyDisplayMode::Entities => {
                    build_entity_tree(ui, ui_events, world, state);
                }
                HierarchyDisplayMode::Bones => {
                    build_bone_tree(ui, ui_events, world, state, assets);
                }
            }
        });
}

fn build_mode_tabs(ui: &imgui::Ui, ui_events: &mut UIEventQueue, state: &HierarchyState) {
    let tab_width = 80.0;

    let entities_selected = state.display_mode == HierarchyDisplayMode::Entities;
    ui.set_next_item_width(tab_width);
    if ui
        .selectable_config("Entities")
        .selected(entities_selected)
        .size([tab_width, 0.0])
        .build()
    {
        ui_events.send(UIEvent::SetHierarchyDisplayMode(
            HierarchyDisplayMode::Entities,
        ));
    }

    ui.same_line();

    let bones_selected = state.display_mode == HierarchyDisplayMode::Bones;
    ui.set_next_item_width(tab_width);
    if ui
        .selectable_config("Bones")
        .selected(bones_selected)
        .size([tab_width, 0.0])
        .build()
    {
        ui_events.send(UIEvent::SetHierarchyDisplayMode(
            HierarchyDisplayMode::Bones,
        ));
    }

    ui.separator();
}

fn build_search_bar(ui: &imgui::Ui, ui_events: &mut UIEventQueue, state: &HierarchyState) {
    let mut search_text = state.search_filter.clone();
    ui.set_next_item_width(-1.0);
    if ui
        .input_text("##search", &mut search_text)
        .hint("Search...")
        .build()
    {
        ui_events.send(UIEvent::SetSearchFilter(search_text));
    }
}

fn build_entity_tree(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    world: &World,
    state: &HierarchyState,
) {
    let entries = query_hierarchy_tree(world, state);

    for entry in entries {
        let indent = entry.depth as f32 * 16.0;
        let cursor_pos = ui.cursor_pos();
        ui.set_cursor_pos([cursor_pos[0] + indent, cursor_pos[1]]);

        let expand_button_width = 16.0;

        if entry.has_children {
            let expand_symbol = if entry.expanded { "v" } else { ">" };
            if ui.small_button(&format!("{}##{}", expand_symbol, entry.entity)) {
                if entry.expanded {
                    ui_events.send(UIEvent::CollapseEntity(entry.entity));
                } else {
                    ui_events.send(UIEvent::ExpandEntity(entry.entity));
                }
            }
            ui.same_line();
        } else {
            let cursor = ui.cursor_pos();
            ui.set_cursor_pos([cursor[0] + expand_button_width, cursor[1]]);
        }

        let icon_label = format!("[{}]", entry.icon_char);
        ui.text(&icon_label);
        ui.same_line();

        let label = format!("{}##{}", entry.name, entry.entity);
        let selected = entry.selected;

        if ui.selectable_config(&label).selected(selected).build() {
            if ui.io().key_ctrl {
                ui_events.send(UIEvent::ToggleEntitySelection(entry.entity));
            } else {
                ui_events.send(UIEvent::SelectEntity(entry.entity));
            }
        }

        if ui.is_item_hovered() && ui.is_mouse_double_clicked(imgui::MouseButton::Left) {
            ui_events.send(UIEvent::FocusOnEntity(entry.entity));
        }
    }

    if ui.is_key_pressed(imgui::Key::Delete) && state.selected_entity.is_some() {
        ui_events.send(UIEvent::DeleteSelectedEntities);
    }
}

fn build_bone_tree(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    world: &World,
    state: &HierarchyState,
    assets: &AssetStorage,
) {
    if let Some(bone_gizmo) = world.get_resource::<BoneGizmoData>() {
        build_bone_display_panel(ui, ui_events, &bone_gizmo);
        ui.separator();
    }

    let skeleton = match assets.skeletons.values().next() {
        Some(skel_asset) => &skel_asset.skeleton,
        None => {
            ui.text("No skeleton loaded");
            return;
        }
    };

    if skeleton.bones.is_empty() {
        ui.text("Skeleton has no bones");
        return;
    }

    ui.text(&format!(
        "Skeleton: {} ({} bones)",
        skeleton.name,
        skeleton.bones.len()
    ));
    ui.separator();

    for &root_id in &skeleton.root_bone_ids {
        build_bone_entry_recursive(ui, ui_events, state, skeleton, root_id, 0);
    }
}

fn build_bone_display_panel(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    bone_gizmo: &BoneGizmoData,
) {
    ui.text("Bone Display");

    let styles = [
        (BoneDisplayStyle::Stick, "Stick"),
        (BoneDisplayStyle::Octahedral, "Octa"),
        (BoneDisplayStyle::Box, "Box"),
        (BoneDisplayStyle::Sphere, "Sphere"),
    ];

    for (i, (style, label)) in styles.iter().enumerate() {
        if i > 0 {
            ui.same_line();
        }
        if ui.radio_button_bool(label, bone_gizmo.display_style == *style) {
            ui_events.send(UIEvent::SetBoneDisplayStyle(*style));
        }
    }

    let mut in_front = bone_gizmo.in_front;
    if ui.checkbox("In Front", &mut in_front) {
        ui_events.send(UIEvent::SetBoneInFront(in_front));
    }

    let mut dist_scaling = bone_gizmo.distance_scaling_enabled;
    if ui.checkbox("Distance Scaling", &mut dist_scaling) {
        ui_events.send(UIEvent::SetBoneDistanceScaling(dist_scaling));
    }

    if bone_gizmo.distance_scaling_enabled {
        let mut factor = bone_gizmo.distance_scaling_factor;
        ui.set_next_item_width(-1.0);
        if imgui::Slider::new(ui, "Factor", 0.01f32, 0.1f32)
            .display_format("%.3f")
            .build(&mut factor)
        {
            ui_events.send(UIEvent::SetBoneDistanceScaleFactor(factor));
        }
    }
}

fn build_bone_entry_recursive(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    state: &HierarchyState,
    skeleton: &Skeleton,
    bone_id: BoneId,
    depth: usize,
) {
    let bone = match skeleton.get_bone(bone_id) {
        Some(b) => b,
        None => return,
    };

    let has_children = !bone.children.is_empty();
    let expanded = hierarchy_is_bone_expanded(state, bone_id);
    let selected = state.selected_bone_id == Some(bone_id);

    let indent = depth as f32 * 16.0;
    let cursor_pos = ui.cursor_pos();
    ui.set_cursor_pos([cursor_pos[0] + indent, cursor_pos[1]]);

    let expand_button_width = 16.0;

    if has_children {
        let expand_symbol = if expanded { "v" } else { ">" };
        if ui.small_button(&format!("{}##bone_{}", expand_symbol, bone_id)) {
            if expanded {
                ui_events.send(UIEvent::CollapseBone(bone_id));
            } else {
                ui_events.send(UIEvent::ExpandBone(bone_id));
            }
        }
        ui.same_line();
    } else {
        let cursor = ui.cursor_pos();
        ui.set_cursor_pos([cursor[0] + expand_button_width, cursor[1]]);
    }

    ui.text("[B]");
    ui.same_line();

    let label = format!("{}##bone_{}", bone.name, bone_id);
    if ui.selectable_config(&label).selected(selected).build() {
        ui_events.send(UIEvent::SelectBone(bone_id));
    }

    if expanded && has_children {
        let children: Vec<BoneId> = bone.children.clone();
        for child_id in children {
            build_bone_entry_recursive(ui, ui_events, state, skeleton, child_id, depth + 1);
        }
    }
}
