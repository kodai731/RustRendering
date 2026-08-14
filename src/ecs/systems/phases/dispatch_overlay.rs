use crate::ecs::component::{FlameEffect, FlameTrail};
use crate::ecs::events::UIEvent;
use crate::ecs::resource::gizmo::BoneGizmoData;
use crate::ecs::resource::{
    AutoExposure, DepthOfField, FlameRenderSettings, GridMeshData, HierarchyState, MessageLog,
    OnionSkinningConfig, PhysicalCameraParameters, TransformGizmoState, WeightHeatmapState,
};
use crate::ecs::systems::{resolve_selected_flame, write_flame_transform};
use crate::ecs::world::{Animator, World};

pub fn dispatch_overlay_events(events: &[UIEvent], world: &mut World) {
    for event in events {
        match event {
            UIEvent::SetBoneGizmoVisible(visible) => {
                if let Some(mut gizmo) = world.get_resource_mut::<BoneGizmoData>() {
                    gizmo.visible = *visible;
                }
            }
            UIEvent::SetWeightHeatmapEnabled(enabled) => {
                log!("UIEvent::SetWeightHeatmapEnabled({})", enabled);
                if let Some(mut heatmap) = world.get_resource_mut::<WeightHeatmapState>() {
                    heatmap.enabled = *enabled;
                } else {
                    log_warn!("WeightHeatmapState resource missing when toggling heatmap");
                }
            }
            UIEvent::SetTransformGizmoMode(mode) => {
                if let Some(mut state) = world.get_resource_mut::<TransformGizmoState>() {
                    state.mode = *mode;
                }
            }
            UIEvent::SetTransformGizmoSpace(space) => {
                if let Some(mut state) = world.get_resource_mut::<TransformGizmoState>() {
                    state.coordinate_space = *space;
                }
            }
            UIEvent::UpdateTransformGizmoState(new_state) => {
                if let Some(mut state) = world.get_resource_mut::<TransformGizmoState>() {
                    *state = *new_state.clone();
                }
            }
            UIEvent::UpdateDepthOfField(new_dof) => {
                if let Some(mut dof) = world.get_resource_mut::<DepthOfField>() {
                    *dof = new_dof.clone();
                }
            }
            UIEvent::UpdatePhysicalCamera(new_params) => {
                if let Some(mut params) = world.get_resource_mut::<PhysicalCameraParameters>() {
                    *params = new_params.clone();
                }
            }
            UIEvent::UpdateAutoExposure(new_ae) => {
                if let Some(mut ae) = world.get_resource_mut::<AutoExposure>() {
                    *ae = new_ae.clone();
                }
            }
            UIEvent::UpdateOnionSkinning(new_config) => {
                if new_config.enabled {
                    auto_select_animator_entity(world);
                }
                if let Some(mut config) = world.get_resource_mut::<OnionSkinningConfig>() {
                    *config = new_config.clone();
                }
            }
            UIEvent::UpdateFlameEffect(effect) => {
                let Some(target) = resolve_selected_flame(world) else {
                    continue;
                };
                write_flame_transform(world, target, effect.position, effect.rotation);
                if let Some(current) = world.get_component_mut::<FlameEffect>(target) {
                    *current = effect.as_ref().clone();
                }
            }
            UIEvent::UpdateFlameBaked(baked) => {
                let Some(target) = resolve_selected_flame(world) else {
                    continue;
                };
                world.insert_component(target, *baked.as_ref());
            }
            UIEvent::ApplyFlamePreset(name) => {
                crate::ecs::systems::apply_flame_preset_to_selected(world, name);
            }
            UIEvent::ApplyFlameTextureFit {
                path,
                blend,
                groups,
                profile,
            } => {
                crate::ecs::systems::apply_flame_texture_fit_to_selected(
                    world,
                    path,
                    *blend,
                    thyllore_effect_core::TextureFitGroups {
                        silhouette: groups[0],
                        color: groups[1],
                        turbulence: groups[2],
                        tilt: groups[3],
                    },
                    *profile,
                    "ui",
                );
            }
            UIEvent::ApplyFlameStyle { path, groups } => {
                crate::ecs::systems::apply_flame_style_to_selected(
                    world,
                    path,
                    thyllore_effect_core::StyleGroups {
                        motion: groups[0],
                        texture: groups[1],
                        optics: groups[2],
                    },
                );
            }
            UIEvent::SaveFlameStyle { name } => {
                if crate::ecs::systems::save_flame_style_of_selected(world, name).is_some() {
                    if let Some(mut model_state) =
                        world.get_resource_mut::<crate::ecs::resource::ModelState>()
                    {
                        model_state.flame_style_scan_done = false;
                        model_state.flame_style_scan.clear();
                    }
                }
            }
            UIEvent::UpdateFlameTrailEnabled(enabled) => {
                let Some(target) = resolve_selected_flame(world) else {
                    continue;
                };
                if let Some(trail) = world.get_component_mut::<FlameTrail>(target) {
                    trail.state.enabled = *enabled;
                } else {
                    world.insert_component(
                        target,
                        FlameTrail {
                            state: thyllore_effect_core::FlameTrailState {
                                enabled: *enabled,
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                    );
                }
            }
            UIEvent::UpdateFlameTrailFade(fade) => {
                let Some(target) = resolve_selected_flame(world) else {
                    continue;
                };
                if let Some(trail) = world.get_component_mut::<FlameTrail>(target) {
                    trail.state.fade_seconds = *fade;
                } else {
                    world.insert_component(
                        target,
                        FlameTrail {
                            state: thyllore_effect_core::FlameTrailState {
                                fade_seconds: *fade,
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                    );
                }
            }
            UIEvent::SelectFlameInstance(index) => {
                let flames = world.query_flames();
                if flames.is_empty() {
                    continue;
                }
                let clamped = (*index as usize).min(flames.len() - 1);
                if let Some(mut hierarchy) = world.get_resource_mut::<HierarchyState>() {
                    hierarchy.selected_entity = Some(flames[clamped]);
                }
            }
            UIEvent::DumpFlameWallProbe { viewport_size } => {
                crate::ecs::systems::perform_flame_wall_probe_dump(world, *viewport_size);
            }
            UIEvent::UpdateFlameRenderSettings(new_settings) => {
                if let Some(mut settings) = world.get_resource_mut::<FlameRenderSettings>() {
                    *settings = new_settings.clone();
                }
            }
            UIEvent::SetGridShowYAxis(show) => {
                if let Some(mut grid) = world.get_resource_mut::<GridMeshData>() {
                    grid.show_y_axis_grid = *show;
                }
            }
            UIEvent::ClearMessageLog => {
                if let Some(mut log) = world.get_resource_mut::<MessageLog>() {
                    crate::ecs::systems::message_log_clear_buffer(&mut log);
                }
            }
            _ => {}
        }
    }
}

fn auto_select_animator_entity(world: &mut World) {
    let already_selected = world
        .get_resource::<HierarchyState>()
        .and_then(|h| h.selected_entity)
        .is_some();
    if already_selected {
        return;
    }

    let first_animator = world.iter_components::<Animator>().next().map(|(e, _)| e);
    if let Some(entity) = first_animator {
        let mut hierarchy = world.resource_mut::<HierarchyState>();
        crate::ecs::systems::hierarchy_select(&mut hierarchy, entity);
    }
}
