use anyhow::Result;
use cgmath::Vector3;

use cgmath::Matrix4;

use crate::animation::{BoneId, SkeletonId};
use crate::ecs::component::LineMesh;
use crate::ecs::context::EcsContext;
use crate::ecs::resource::gizmo::transform_gizmo::TransformGizmoHandle;
use crate::ecs::resource::gizmo::{BoneDisplayStyle, BoneGizmoData, TransformGizmoData};
use crate::ecs::resource::CurveEditorState;
use crate::ecs::resource::{
    BonePoseOverride, ClipLibrary, HierarchyDisplayMode, ImGuiInputCapture, KeyboardModifiers,
    MeshAssets, MouseInput, TimelineState, TransformGizmoMode, TransformGizmoState, ViewportInput,
};
use crate::ecs::systems::{
    compute_local_override_from_global_rotation, compute_local_override_from_global_scale,
    compute_local_override_from_global_translation, select_bone_by_mesh_ray, select_bone_by_ray,
    transform_gizmo_systems,
};
use crate::ecs::world::{Entity, GlobalTransform, Transform};
use crate::ecs::{
    compute_pose_global_transforms, create_pose_from_rest, sample_clip_to_pose, GizmoAxis,
};
use crate::ecs::{gizmo_try_select, gizmo_update_position_with_constraint};
use crate::math::screen_to_world_ray;

pub fn run_input_phase(ctx: &mut EcsContext) -> Result<()> {
    update_pointer_state(ctx);

    ctx.pointer_capture_mut().active = false;
    let wants_pointer = ctx.pointer_state().imgui_wants_pointer;
    let viewport_hovered = ctx.pointer_state().viewport_hovered;
    if wants_pointer && !viewport_hovered {
        ctx.pointer_capture_mut().active = true;
    }

    {
        let imgui_wants_mouse = ctx.world.resource::<ImGuiInputCapture>().wants_mouse;
        let viewport_hovered = ctx.world.resource::<ViewportInput>().hovered;
        ctx.world
            .resource_mut::<MouseInput>()
            .compute_drag_delta(imgui_wants_mouse, viewport_hovered);
    }

    let transform_gizmo_active = process_transform_gizmo_interaction(ctx)?;

    if !transform_gizmo_active {
        process_gizmo_interaction(ctx)?;
    }

    let capture_active = ctx.pointer_capture().active;
    let left_just_pressed = ctx.pointer_state().left.just_pressed();
    if left_just_pressed && !capture_active {
        let bone_hit = process_bone_selection(ctx)?;
        if !bone_hit {
            request_mesh_selection(ctx);
        }
    }

    sync_transform_gizmo_to_bone(ctx);

    let capture_active = ctx.pointer_capture().active;
    let viewport_hovered = ctx.pointer_state().viewport_hovered;
    if !capture_active && viewport_hovered {
        let mouse = ctx.world.resource::<MouseInput>();
        let is_right_clicked = mouse.right_pressed;
        let is_wheel_clicked = mouse.middle_pressed;
        let mouse_wheel = mouse.wheel;
        let mouse_diff = mouse.delta;
        let mouse_pos = mouse.position;
        drop(mouse);

        let viewport_pos = ctx.world.resource::<ViewportInput>().position;
        let local_mouse_pos = [
            mouse_pos[0] - viewport_pos[0],
            mouse_pos[1] - viewport_pos[1],
        ];
        let screen_size = [ctx.swapchain_extent.0 as f32, ctx.swapchain_extent.1 as f32];
        let mut camera = ctx.camera_mut();
        crate::ecs::camera_input_system_inner(
            &mut *camera,
            is_right_clicked,
            is_wheel_clicked,
            mouse_wheel,
            mouse_diff,
            local_mouse_pos,
            screen_size,
        );
    }

    Ok(())
}

fn update_pointer_state(ctx: &mut EcsContext) {
    let mouse = ctx.world.resource::<MouseInput>();
    let is_left = mouse.left_pressed;
    let is_right = mouse.right_pressed;
    let is_wheel = mouse.middle_pressed;
    let mouse_pos = mouse.position;
    let mouse_wheel = mouse.wheel;
    drop(mouse);

    let viewport_pos = ctx.world.resource::<ViewportInput>().position;
    let viewport_hovered = ctx.world.resource::<ViewportInput>().hovered;
    let imgui_wants_mouse = ctx.world.resource::<ImGuiInputCapture>().wants_mouse;

    use crate::ecs::resource::RawButtonInput;
    use crate::ecs::systems::button_state_advance;
    let to_input = |down: bool| {
        if down {
            RawButtonInput::Pressed
        } else {
            RawButtonInput::Released
        }
    };

    let mut pointer = ctx.pointer_state_mut();
    button_state_advance(&mut pointer.left, to_input(is_left));
    button_state_advance(&mut pointer.right, to_input(is_right));
    button_state_advance(&mut pointer.middle, to_input(is_wheel));
    pointer.position = mouse_pos;
    pointer.viewport_position = [
        mouse_pos[0] - viewport_pos[0],
        mouse_pos[1] - viewport_pos[1],
    ];
    pointer.wheel_delta = mouse_wheel;
    pointer.viewport_hovered = viewport_hovered;
    pointer.imgui_wants_pointer = imgui_wants_mouse;
}

fn compute_viewport_local_mouse_pos(ctx: &EcsContext) -> cgmath::Vector2<f32> {
    let mouse_pos = ctx.world.resource::<MouseInput>().position;
    let viewport_pos = ctx.world.resource::<ViewportInput>().position;
    cgmath::Vector2::new(
        mouse_pos[0] - viewport_pos[0],
        mouse_pos[1] - viewport_pos[1],
    )
}

fn process_gizmo_interaction(ctx: &mut EcsContext) -> Result<()> {
    let mouse_pos = compute_viewport_local_mouse_pos(ctx);
    let drag_active = ctx.light_gizmo().drag_active;

    if drag_active {
        ctx.pointer_capture_mut().active = true;

        if ctx.pointer_state().left.held() {
            update_light_gizmo_position(ctx, mouse_pos)?;
        }

        if ctx.pointer_state().left.just_released() {
            reset_light_gizmo_drag(ctx);
        }

        return Ok(());
    }

    let left_just_pressed = ctx.pointer_state().left.just_pressed();
    let capture_active = ctx.pointer_capture().active;
    if left_just_pressed && !capture_active && ctx.pointer_state().viewport_hovered {
        let camera = ctx.camera();
        let camera_pos = crate::ecs::compute_camera_position(&camera);
        let camera_dir = crate::ecs::compute_camera_direction(&camera);
        let camera_up = crate::ecs::compute_camera_up(&camera);
        let fov_y = camera.fov_y;
        let near_plane = camera.near_plane;
        drop(camera);

        let billboard_click_rect = ctx
            .world
            .get_resource::<crate::ecs::resource::DebugViewState>()
            .map(|s| s.billboard_click_rect)
            .unwrap_or(None);

        {
            let mut gizmo_ref = ctx.light_gizmo_mut();
            let light_gizmo = &mut *gizmo_ref;
            let position = light_gizmo.position.clone();
            gizmo_try_select(
                &position,
                &mut light_gizmo.selectable,
                &mut light_gizmo.draggable,
                mouse_pos,
                camera_pos,
                camera_dir,
                camera_up,
                ctx.swapchain_extent,
                billboard_click_rect,
                fov_y,
                near_plane,
            );
        }

        let hit = ctx.light_gizmo().selectable.is_selected;
        if hit {
            ctx.light_gizmo_mut().drag_active = true;
            ctx.pointer_capture_mut().active = true;
        }
    }

    Ok(())
}

fn reset_light_gizmo_drag(ctx: &mut EcsContext) {
    log!("Mouse released - resetting light gizmo state");
    let mut gizmo = ctx.light_gizmo_mut();
    gizmo.drag_active = false;
    gizmo.selectable.is_selected = false;
    gizmo.selectable.selected_axis = GizmoAxis::None;
    gizmo.draggable.drag_axis = GizmoAxis::None;
    gizmo.draggable.initial_position = Vector3::new(0.0, 0.0, 0.0);
}

fn update_light_gizmo_position(
    ctx: &mut EcsContext,
    mouse_pos: cgmath::Vector2<f32>,
) -> Result<()> {
    use crate::math::coordinate_system::perspective_infinite_reverse;
    use cgmath::InnerSpace;

    let camera = ctx.camera();
    let camera_pos = crate::ecs::compute_camera_position(&camera);
    let camera_dir = crate::ecs::compute_camera_direction(&camera);
    let camera_up = crate::ecs::compute_camera_up(&camera);
    let fov_y = camera.fov_y;
    let near_plane = camera.near_plane;
    drop(camera);

    let view = unsafe { crate::math::view(camera_pos, camera_dir, camera_up) };
    let aspect = ctx.swapchain_extent.0 as f32 / ctx.swapchain_extent.1 as f32;
    let proj = perspective_infinite_reverse(fov_y, aspect, near_plane);
    let screen_size =
        cgmath::Vector2::new(ctx.swapchain_extent.0 as f32, ctx.swapchain_extent.1 as f32);

    let (ray_origin, ray_direction) = screen_to_world_ray(mouse_pos, screen_size, view, proj);

    let light_pos = ctx.light_state().light_position;
    let plane_point = light_pos;
    let plane_normal = -camera_dir;

    let denom = plane_normal.dot(ray_direction);

    if denom.abs() > std::f32::EPSILON {
        let t = (plane_point - ray_origin).dot(plane_normal) / denom;

        if t >= 0.0 {
            let intersection = ray_origin + ray_direction * t;

            let is_ctrl = ctx.world.resource::<KeyboardModifiers>().ctrl;
            {
                let mut gizmo = ctx.light_gizmo_mut();
                let draggable = gizmo.draggable.clone();
                gizmo_update_position_with_constraint(
                    &mut gizmo.position,
                    intersection,
                    &draggable,
                    is_ctrl,
                );
            }

            ctx.light_state_mut().light_position = ctx.light_gizmo().position.position;
        }
    }

    Ok(())
}

fn process_bone_selection(ctx: &mut EcsContext) -> Result<bool> {
    if !ctx.world.contains_resource::<BoneGizmoData>() {
        return Ok(false);
    }

    let (visible, display_style) = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        (bone_gizmo.visible, bone_gizmo.display_style)
    };

    if !visible || display_style != BoneDisplayStyle::Octahedral {
        return Ok(false);
    }

    if ctx.light_gizmo().drag_active {
        return Ok(false);
    }

    let (ray_origin, ray_direction) = compute_bone_pick_ray(ctx);

    let (skeleton_id, transforms, offsets, mesh_scale) = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        (
            bone_gizmo.cached_skeleton_id,
            bone_gizmo.cached_global_transforms.clone(),
            bone_gizmo.bone_local_offsets.clone(),
            bone_gizmo.mesh_scale,
        )
    };

    let Some(skel_id) = skeleton_id else {
        return Ok(false);
    };

    let Some(skeleton_ref) = ctx.assets.get_skeleton_by_skeleton_id(skel_id) else {
        return Ok(false);
    };
    let skeleton = skeleton_ref.clone();

    let widget_hit = select_bone_by_ray(
        ray_origin,
        ray_direction,
        &skeleton,
        &transforms,
        &offsets,
        mesh_scale,
        None,
    );

    let hit = widget_hit.or_else(|| {
        let entity_transform = find_animated_entity_transform(ctx.world)?;
        let mesh_assets = ctx.world.resource::<MeshAssets>();
        select_bone_by_mesh_ray(ray_origin, ray_direction, &mesh_assets, entity_transform)
    });

    let bone_hit = hit.is_some();

    let is_shift = ctx.world.resource::<KeyboardModifiers>().shift;
    let new_active_bone = apply_bone_selection_result(ctx, &skeleton, hit, is_shift);

    sync_bone_selection_to_hierarchy(ctx, new_active_bone, is_shift);
    sync_curve_editor_bone(ctx, new_active_bone);

    Ok(bone_hit)
}

fn compute_bone_pick_ray(ctx: &EcsContext) -> (Vector3<f32>, Vector3<f32>) {
    let mouse_pos = compute_viewport_local_mouse_pos(ctx);
    let screen_size =
        cgmath::Vector2::new(ctx.swapchain_extent.0 as f32, ctx.swapchain_extent.1 as f32);

    let camera = ctx.camera();
    let camera_pos = crate::ecs::compute_camera_position(&camera);
    let camera_dir = crate::ecs::compute_camera_direction(&camera);
    let camera_up = crate::ecs::compute_camera_up(&camera);
    let fov_y = camera.fov_y;
    let near_plane = camera.near_plane;
    drop(camera);

    let view = unsafe { crate::math::view(camera_pos, camera_dir, camera_up) };
    let aspect = screen_size.x / screen_size.y;
    let proj =
        crate::math::coordinate_system::perspective_infinite_reverse(fov_y, aspect, near_plane);

    let (ray_origin, ray_direction) = screen_to_world_ray(mouse_pos, screen_size, view, proj);

    let mouse_raw = ctx.world.resource::<MouseInput>().position;
    let viewport_position = ctx.world.resource::<ViewportInput>().position;
    let viewport_size = ctx.world.resource::<ViewportInput>().size;
    log!(
        "bone_select: viewport_pos=({:.0},{:.0}) viewport_size=({:.0},{:.0}) mouse_raw=({:.0},{:.0}) mouse_local=({:.1},{:.1}) ray_origin=({:.2},{:.2},{:.2}) ray_dir=({:.3},{:.3},{:.3})",
        viewport_position[0],
        viewport_position[1],
        viewport_size[0],
        viewport_size[1],
        mouse_raw[0],
        mouse_raw[1],
        mouse_pos.x,
        mouse_pos.y,
        ray_origin.x, ray_origin.y, ray_origin.z,
        ray_direction.x, ray_direction.y, ray_direction.z,
    );

    (ray_origin, ray_direction)
}

fn apply_bone_selection_result(
    ctx: &mut EcsContext,
    skeleton: &crate::animation::Skeleton,
    hit: Option<(usize, f32)>,
    is_shift: bool,
) -> Option<BoneId> {
    let mut selection = ctx.bone_selection_mut();

    match hit {
        Some((bone_idx, _distance)) => {
            let bone_id = bone_idx as BoneId;
            let descendants = skeleton.collect_descendants(bone_id);

            let bone_name = skeleton
                .bones
                .iter()
                .find(|b| b.id as usize == bone_idx)
                .map(|b| b.name.as_str())
                .unwrap_or("unknown");

            if is_shift {
                if selection.selected_bone_indices.contains(&bone_idx) {
                    selection.selected_bone_indices.remove(&bone_idx);
                    for desc_id in &descendants {
                        selection.selected_bone_indices.remove(&(*desc_id as usize));
                    }
                    if selection.active_bone_index == Some(bone_idx) {
                        selection.active_bone_index =
                            selection.selected_bone_indices.iter().copied().next();
                    }
                } else {
                    selection.selected_bone_indices.insert(bone_idx);
                    for desc_id in &descendants {
                        selection.selected_bone_indices.insert(*desc_id as usize);
                    }
                    selection.active_bone_index = Some(bone_idx);
                }
            } else {
                selection.selected_bone_indices.clear();
                selection.selected_bone_indices.insert(bone_idx);
                for desc_id in &descendants {
                    selection.selected_bone_indices.insert(*desc_id as usize);
                }
                selection.active_bone_index = Some(bone_idx);
            }

            log!(
                "Bone selected: [{}] '{}' (active={:?}, total={}, descendants={})",
                bone_idx,
                bone_name,
                selection.active_bone_index,
                selection.selected_bone_indices.len(),
                descendants.len()
            );

            Some(bone_id)
        }
        None => {
            if !is_shift {
                selection.selected_bone_indices.clear();
                selection.active_bone_index = None;
            }
            None
        }
    }
}

fn request_mesh_selection(ctx: &mut EcsContext) {
    if !ctx
        .world
        .contains_resource::<crate::ecs::resource::ObjectIdReadback>()
    {
        return;
    }

    if ctx.light_gizmo().drag_active {
        return;
    }

    let readback = ctx.object_id_readback();
    if readback.copy_in_flight {
        return;
    }
    drop(readback);

    let mouse_pos = ctx.world.resource::<MouseInput>().position;
    let viewport = ctx.world.resource::<ViewportInput>();
    let viewport_pos = viewport.position;
    let viewport_size = viewport.size;
    drop(viewport);

    let local_x = mouse_pos[0] - viewport_pos[0];
    let local_y = mouse_pos[1] - viewport_pos[1];

    if local_x < 0.0 || local_y < 0.0 || local_x >= viewport_size[0] || local_y >= viewport_size[1]
    {
        return;
    }

    let gbuffer_w = ctx.swapchain_extent.0;
    let gbuffer_h = ctx.swapchain_extent.1;

    let px = ((local_x / viewport_size[0]) * gbuffer_w as f32) as u32;
    let py = ((local_y / viewport_size[1]) * gbuffer_h as f32) as u32;
    let px = px.min(gbuffer_w.saturating_sub(1));
    let py = py.min(gbuffer_h.saturating_sub(1));

    let modifiers = ctx.world.resource::<KeyboardModifiers>();
    let is_shift = modifiers.shift;
    let is_ctrl = modifiers.ctrl;
    drop(modifiers);

    let mut readback = ctx.object_id_readback_mut();
    readback.pending_pixel = Some((px, py));
    readback.is_shift = is_shift;
    readback.is_ctrl = is_ctrl;
}

fn process_transform_gizmo_interaction(ctx: &mut EcsContext) -> Result<bool> {
    if !ctx.world.contains_resource::<TransformGizmoData>() {
        return Ok(false);
    }
    if !ctx.world.contains_resource::<TransformGizmoState>() {
        return Ok(false);
    }

    let visible = ctx.transform_gizmo().visible;
    if !visible {
        return Ok(false);
    }

    let drag_active = ctx.transform_gizmo().drag_active;
    let mouse_pos = compute_viewport_local_mouse_pos(ctx);

    if drag_active {
        ctx.pointer_capture_mut().active = true;

        if ctx.pointer_state().left.held() {
            process_transform_gizmo_drag(ctx, mouse_pos)?;
        }

        if ctx.pointer_state().left.just_released() {
            let mut tg = ctx.transform_gizmo_mut();
            tg.drag_active = false;
            tg.selectable.is_selected = false;
            tg.active_handle = TransformGizmoHandle::None;
            log!("TransformGizmo released");
        }

        return Ok(true);
    }

    let left_just_pressed = ctx.pointer_state().left.just_pressed();
    let capture_active = ctx.pointer_capture().active;
    if left_just_pressed && !capture_active && ctx.pointer_state().viewport_hovered {
        let hit = try_select_transform_gizmo_handle(ctx, mouse_pos);
        if hit {
            ctx.transform_gizmo_mut().drag_active = true;
            ctx.pointer_capture_mut().active = true;
            return Ok(true);
        }
    }

    Ok(false)
}

fn try_select_transform_gizmo_handle(
    ctx: &mut EcsContext,
    mouse_pos: cgmath::Vector2<f32>,
) -> bool {
    let camera = ctx.camera();
    let camera_pos = crate::ecs::compute_camera_position(&camera);
    let camera_dir = crate::ecs::compute_camera_direction(&camera);
    let camera_up = crate::ecs::compute_camera_up(&camera);
    let fov_y = camera.fov_y;
    let near_plane = camera.near_plane;
    drop(camera);

    let state = ctx.transform_gizmo_state();
    let mode = state.mode;
    let gizmo_scale = state.gizmo_scale;
    drop(state);

    let tg = ctx.transform_gizmo();
    let tg_clone_for_select = TransformGizmoData {
        visible: tg.visible,
        position: tg.position.clone(),
        selectable: tg.selectable.clone(),
        draggable: tg.draggable.clone(),
        active_handle: tg.active_handle,
        drag_active: false,
        line_mesh: LineMesh::default(),
        solid_mesh: LineMesh::default(),
        line_render_info: tg.line_render_info,
        solid_render_info: tg.solid_render_info,
        drag_start_position: tg.drag_start_position,
        drag_start_rotation: tg.drag_start_rotation,
        drag_start_scale: tg.drag_start_scale,
        drag_plane_normal: tg.drag_plane_normal,
        drag_initial_hit: tg.drag_initial_hit,
        drag_initial_angle: tg.drag_initial_angle,
        target_bone_id: tg.target_bone_id,
        target_entity: tg.target_entity,
    };
    drop(tg);

    let handle = transform_gizmo_systems::transform_gizmo_try_select(
        &tg_clone_for_select,
        mode,
        mouse_pos,
        camera_pos,
        camera_dir,
        camera_up,
        ctx.swapchain_extent,
        fov_y,
        near_plane,
        gizmo_scale,
    );

    if handle == TransformGizmoHandle::None {
        return false;
    }

    let gizmo_pos = tg_clone_for_select.position.position;
    let (plane_point, plane_normal) =
        transform_gizmo_systems::transform_gizmo_compute_drag_plane(handle, gizmo_pos, camera_dir);

    let view_mat = unsafe { crate::math::view(camera_pos, camera_dir, camera_up) };
    let aspect = ctx.swapchain_extent.0 as f32 / ctx.swapchain_extent.1 as f32;
    let proj =
        crate::math::coordinate_system::perspective_infinite_reverse(fov_y, aspect, near_plane);
    let screen_size =
        cgmath::Vector2::new(ctx.swapchain_extent.0 as f32, ctx.swapchain_extent.1 as f32);
    let (ray_origin, ray_direction) = screen_to_world_ray(mouse_pos, screen_size, view_mat, proj);
    let initial_hit =
        crate::math::ray_plane_intersection(ray_origin, ray_direction, plane_point, plane_normal)
            .unwrap_or(gizmo_pos);

    let entity_scale = {
        let tg = ctx.transform_gizmo();
        tg.target_entity
            .and_then(|e| ctx.world.get_component::<Transform>(e).map(|t| t.scale))
    };

    let mut tg = ctx.transform_gizmo_mut();
    tg.selectable.is_selected = true;
    tg.active_handle = handle;
    tg.drag_start_position = tg.position.position;
    tg.drag_plane_normal = plane_normal;
    tg.drag_initial_hit = initial_hit;
    if let Some(scale) = entity_scale {
        tg.drag_start_scale = scale;
    }

    log!("TransformGizmo selected: handle={:?}", handle);
    true
}

fn process_transform_gizmo_drag(
    ctx: &mut EcsContext,
    mouse_pos: cgmath::Vector2<f32>,
) -> Result<()> {
    let camera = ctx.camera();
    let camera_pos = crate::ecs::compute_camera_position(&camera);
    let camera_dir = crate::ecs::compute_camera_direction(&camera);
    let camera_up = crate::ecs::compute_camera_up(&camera);
    let fov_y = camera.fov_y;
    let near_plane = camera.near_plane;
    drop(camera);

    let mode = ctx.transform_gizmo_state().mode;
    let state = ctx.transform_gizmo_state();
    let snap_translate = if state.snap_enabled {
        Some(state.translate_snap_value)
    } else {
        None
    };
    let snap_rotate = if state.snap_enabled {
        Some(state.rotate_snap_degrees)
    } else {
        None
    };
    let snap_scale = if state.snap_enabled {
        Some(state.scale_snap_value)
    } else {
        None
    };
    drop(state);

    match mode {
        TransformGizmoMode::Translate => {
            let tg = ctx.transform_gizmo();
            let tg_ref = &*tg;
            let new_pos = transform_gizmo_systems::transform_gizmo_process_translate_drag(
                tg_ref,
                mouse_pos,
                camera_pos,
                camera_dir,
                camera_up,
                ctx.swapchain_extent,
                fov_y,
                near_plane,
                snap_translate,
            );
            drop(tg);

            if let Some(pos) = new_pos {
                let target_bone = ctx.transform_gizmo().target_bone_id;
                let target_entity = ctx.transform_gizmo().target_entity;
                ctx.transform_gizmo_mut().position.position = pos;

                if let Some(bone_id) = target_bone {
                    apply_bone_translation(ctx, bone_id, pos);
                } else if let Some(entity) = target_entity {
                    apply_entity_translation(ctx, entity, pos);
                }
            }
        }
        TransformGizmoMode::Rotate => {
            let tg = ctx.transform_gizmo();
            let rotation = transform_gizmo_systems::transform_gizmo_process_rotate_drag(
                &tg,
                mouse_pos,
                camera_pos,
                camera_dir,
                camera_up,
                ctx.swapchain_extent,
                fov_y,
                near_plane,
                snap_rotate,
            );
            let target_bone = tg.target_bone_id;
            let target_entity = tg.target_entity;
            drop(tg);

            if let Some(rot) = rotation {
                if let Some(bone_id) = target_bone {
                    let gizmo_pos = ctx.transform_gizmo().position.position;
                    apply_bone_rotation(ctx, bone_id, gizmo_pos, rot);
                } else if let Some(entity) = target_entity {
                    apply_entity_rotation(ctx, entity, rot);
                }
            }
        }
        TransformGizmoMode::Scale => {
            let tg = ctx.transform_gizmo();
            let new_scale = transform_gizmo_systems::transform_gizmo_process_scale_drag(
                &tg,
                mouse_pos,
                camera_pos,
                camera_dir,
                camera_up,
                ctx.swapchain_extent,
                fov_y,
                near_plane,
                snap_scale,
            );
            let target_bone = tg.target_bone_id;
            let target_entity = tg.target_entity;
            drop(tg);

            if let Some(scale) = new_scale {
                if let Some(bone_id) = target_bone {
                    apply_bone_scale(ctx, bone_id, scale);
                } else if let Some(entity) = target_entity {
                    apply_entity_scale(ctx, entity, scale);
                }
            }
        }
    }

    Ok(())
}

fn apply_bone_translation(ctx: &mut EcsContext, bone_id: u32, new_pos: Vector3<f32>) {
    if !ctx.world.contains_resource::<BoneGizmoData>() {
        return;
    }

    let (skeleton_id, mesh_scale) = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        (bone_gizmo.cached_skeleton_id, bone_gizmo.mesh_scale)
    };

    let Some(skel_id) = skeleton_id else { return };
    let Some(globals) = compute_animation_globals(ctx, skel_id) else {
        return;
    };
    let Some(skeleton_ref) = ctx.assets.get_skeleton_by_skeleton_id(skel_id) else {
        return;
    };
    let skeleton = skeleton_ref.clone();

    let inv_scale = if mesh_scale.abs() > f32::EPSILON {
        1.0 / mesh_scale
    } else {
        1.0
    };
    let skeleton_pos = Vector3::new(
        new_pos.x * inv_scale,
        new_pos.y * inv_scale,
        new_pos.z * inv_scale,
    );

    let local_pose =
        compute_local_override_from_global_translation(&skeleton, &globals, bone_id, skeleton_pos);

    if let Some(pose) = local_pose {
        if let Some(mut overrides) = ctx.world.get_resource_mut::<BonePoseOverride>() {
            overrides.overrides.insert(bone_id, pose);
        }
    }
}

fn apply_bone_rotation(
    ctx: &mut EcsContext,
    bone_id: u32,
    gizmo_pos: Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
) {
    if !ctx.world.contains_resource::<BoneGizmoData>() {
        return;
    }

    let (skeleton_id, mesh_scale) = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        (bone_gizmo.cached_skeleton_id, bone_gizmo.mesh_scale)
    };

    let Some(skel_id) = skeleton_id else { return };
    let Some(globals) = compute_animation_globals(ctx, skel_id) else {
        return;
    };
    let Some(skeleton_ref) = ctx.assets.get_skeleton_by_skeleton_id(skel_id) else {
        return;
    };
    let skeleton = skeleton_ref.clone();

    let inv_scale = if mesh_scale.abs() > f32::EPSILON {
        1.0 / mesh_scale
    } else {
        1.0
    };
    let skeleton_gizmo_pos = Vector3::new(
        gizmo_pos.x * inv_scale,
        gizmo_pos.y * inv_scale,
        gizmo_pos.z * inv_scale,
    );

    let local_pose = compute_local_override_from_global_rotation(
        &skeleton,
        &globals,
        bone_id,
        skeleton_gizmo_pos,
        rotation,
    );

    if let Some(pose) = local_pose {
        if let Some(mut overrides) = ctx.world.get_resource_mut::<BonePoseOverride>() {
            overrides.overrides.insert(bone_id, pose);
        }
    }
}

fn apply_bone_scale(ctx: &mut EcsContext, bone_id: u32, scale: Vector3<f32>) {
    if !ctx.world.contains_resource::<BoneGizmoData>() {
        return;
    }

    let skeleton_id = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        bone_gizmo.cached_skeleton_id
    };

    let Some(skel_id) = skeleton_id else { return };
    let Some(globals) = compute_animation_globals(ctx, skel_id) else {
        return;
    };
    let Some(skeleton_ref) = ctx.assets.get_skeleton_by_skeleton_id(skel_id) else {
        return;
    };
    let skeleton = skeleton_ref.clone();

    let local_pose = compute_local_override_from_global_scale(&skeleton, &globals, bone_id, scale);

    if let Some(pose) = local_pose {
        if let Some(mut overrides) = ctx.world.get_resource_mut::<BonePoseOverride>() {
            overrides.overrides.insert(bone_id, pose);
        }
    }
}

fn apply_entity_translation(ctx: &mut EcsContext, entity: Entity, new_pos: Vector3<f32>) {
    if let Some(mut transform) = ctx.world.get_component_mut::<Transform>(entity) {
        transform.translation = new_pos;
    }
}

fn apply_entity_rotation(ctx: &mut EcsContext, entity: Entity, rotation: cgmath::Quaternion<f32>) {
    if let Some(mut transform) = ctx.world.get_component_mut::<Transform>(entity) {
        transform.rotation = rotation * transform.rotation;
    }
}

fn apply_entity_scale(ctx: &mut EcsContext, entity: Entity, scale: Vector3<f32>) {
    if let Some(mut transform) = ctx.world.get_component_mut::<Transform>(entity) {
        transform.scale = scale;
    }
}

fn sync_transform_gizmo_to_bone(ctx: &mut EcsContext) {
    if !ctx.world.contains_resource::<TransformGizmoData>() {
        return;
    }
    if !ctx.world.contains_resource::<BoneGizmoData>() {
        return;
    }

    if ctx.transform_gizmo().drag_active {
        return;
    }

    let (active_bone, transforms, offsets, mesh_scale) = {
        let selection = ctx.bone_selection();
        let active = selection.active_bone_index;
        drop(selection);

        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        if !bone_gizmo.visible {
            let mut tg = ctx.transform_gizmo_mut();
            tg.visible = false;
            return;
        }
        (
            active,
            bone_gizmo.cached_global_transforms.clone(),
            bone_gizmo.bone_local_offsets.clone(),
            bone_gizmo.mesh_scale,
        )
    };

    if active_bone.is_some() {
        let mut tg = ctx.transform_gizmo_mut();
        transform_gizmo_systems::transform_gizmo_sync_to_bone(
            &mut tg,
            active_bone,
            &transforms,
            &offsets,
            mesh_scale,
        );
        tg.target_entity = None;
        return;
    }

    sync_transform_gizmo_to_entity(ctx);
}

fn sync_transform_gizmo_to_entity(ctx: &mut EcsContext) {
    let selected_entity = ctx.hierarchy_state().selected_entity;

    let Some(entity) = selected_entity else {
        let mut tg = ctx.transform_gizmo_mut();
        tg.visible = false;
        tg.target_bone_id = None;
        tg.target_entity = None;
        return;
    };

    let transform_entity = crate::ecs::systems::resolve_transform_entity(ctx.world, entity);

    let world_pos = ctx
        .world
        .get_component::<GlobalTransform>(transform_entity)
        .map(|gt| {
            let m = gt.0;
            Vector3::new(m[3][0], m[3][1], m[3][2])
        });

    let Some(pos) = world_pos else {
        let mut tg = ctx.transform_gizmo_mut();
        tg.visible = false;
        tg.target_entity = None;
        return;
    };

    let mut tg = ctx.transform_gizmo_mut();
    transform_gizmo_systems::transform_gizmo_sync_to_entity(&mut tg, pos, transform_entity);
}

fn sync_bone_selection_to_hierarchy(
    ctx: &mut EcsContext,
    new_active_bone: Option<BoneId>,
    is_shift: bool,
) {
    if let Some(bone_id) = new_active_bone {
        let mut hierarchy = ctx.hierarchy_state_mut();
        hierarchy.selected_bone_id = Some(bone_id);
        hierarchy.display_mode = HierarchyDisplayMode::Bones;
    } else if !is_shift {
        ctx.hierarchy_state_mut().selected_bone_id = None;
    }
}

fn sync_curve_editor_bone(ctx: &mut EcsContext, new_active_bone: Option<BoneId>) {
    let Some(bone_id) = new_active_bone else {
        return;
    };

    let is_open = ctx
        .world
        .get_resource::<CurveEditorState>()
        .map(|s| s.is_open)
        .unwrap_or(false);
    if !is_open {
        return;
    }

    let has_track = {
        let clip_library = ctx.world.resource::<ClipLibrary>();
        let source_id = ctx.world.resource::<TimelineState>().current_clip_id;
        source_id
            .and_then(|id| clip_library.get(id))
            .map(|clip| clip.tracks.contains_key(&bone_id))
            .unwrap_or(false)
    };

    if has_track {
        let mut editor = ctx.world.resource_mut::<CurveEditorState>();
        editor.select_bone(bone_id);
    }
}

fn find_animated_entity_transform(world: &crate::ecs::World) -> Option<Matrix4<f32>> {
    use crate::ecs::world::{Animator, GlobalTransform};

    let (entity, _) = world.iter_components::<Animator>().next()?;
    world
        .get_component::<GlobalTransform>(entity)
        .map(|gt| gt.0)
}

fn compute_animation_globals(
    ctx: &EcsContext,
    skeleton_id: SkeletonId,
) -> Option<Vec<Matrix4<f32>>> {
    let skeleton = ctx.assets.get_skeleton_by_skeleton_id(skeleton_id)?;

    let timeline = ctx.world.resource::<TimelineState>();
    let current_time = timeline.current_time;
    let clip_id = timeline.current_clip_id?;
    drop(timeline);

    let clip_library = ctx.world.resource::<ClipLibrary>();
    let asset_id = clip_library.get_asset_id_for_source(clip_id)?;
    drop(clip_library);

    let clip_asset = ctx.assets.animation_clips.get(&asset_id)?;

    let mut pose = create_pose_from_rest(skeleton);
    sample_clip_to_pose(&clip_asset.clip, current_time, skeleton, &mut pose, false);

    Some(compute_pose_global_transforms(skeleton, &pose))
}
