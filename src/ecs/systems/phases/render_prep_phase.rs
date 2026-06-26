use anyhow::Result;
use cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector3};

use crate::app::FrameContext;
use crate::ecs::component::{ConstraintSet, LineMesh};
use crate::ecs::resource::gizmo::BoneSelectionState;
use crate::ecs::resource::gizmo::TransformGizmoData;
use crate::ecs::resource::gizmo::{
    BoneDisplayStyle, BoneGizmoData, ConstraintGizmoData, SpringBoneGizmoData,
};
use crate::ecs::resource::CurveMeshData;
use crate::ecs::resource::ProjectionData;
use crate::ecs::resource::{Camera, Exposure, TransformGizmoState};
use crate::ecs::systems::render_data_systems::{
    bone_gizmo_render_data, constraint_gizmo_render_data, curve_mesh_render_data,
    gizmo_mesh_render_data, gizmo_selectable_render_data, grid_mesh_render_data,
    spring_bone_gizmo_render_data, transform_gizmo_render_data,
};
use crate::ecs::{
    build_bone_line_mesh, build_box_bone_meshes_with_selection, build_constraint_gizmo_mesh,
    build_octahedral_bone_meshes_with_selection, build_sphere_bone_meshes_with_selection,
    build_spring_bone_gizmo_mesh, gizmo_update_vertex_buffer,
};
use crate::render::RenderBackend;
use crate::vulkanr::renderer::scene_renderer::update_object_ubo;

pub unsafe fn run_render_prep_phase(ctx: &mut FrameContext) -> Result<()> {
    let (view, proj, screen_size, aspect) = {
        let proj_data = ctx.world.resource::<ProjectionData>();
        (
            proj_data.view,
            proj_data.proj,
            proj_data.screen_size,
            proj_data.aspect,
        )
    };

    let camera_position = {
        use crate::ecs::systems::camera_systems::compute_camera_position;
        compute_camera_position(&ctx.camera())
    };

    update_frame_and_scene_uniforms(ctx, view, proj, screen_size, aspect, camera_position)?;

    let render_data_vec = collect_gizmo_render_data(ctx, camera_position);
    let render_data_refs: Vec<_> = render_data_vec.iter().collect();

    if let Err(e) = update_object_ubo(
        &render_data_refs,
        ctx.image_index,
        &ctx.graphics.objects,
        ctx.device,
    ) {
        eprintln!("Failed to update object UBOs: {}", e);
    }

    update_billboard_ubo(ctx, view, proj)?;

    update_grid_gizmo_buffers(ctx)?;
    update_transform_gizmo_mesh(ctx)?;
    update_curve_mesh(ctx)?;
    update_bone_gizmo_mesh(ctx)?;
    update_constraint_gizmo_mesh(ctx)?;
    update_spring_bone_gizmo_mesh(ctx)?;
    crate::ecs::systems::gizmo_systems::run_vertical_lines_update(ctx)?;

    Ok(())
}

unsafe fn update_mesh_entity_transforms(ctx: &mut FrameContext) -> Result<()> {
    use crate::ecs::world::{GlobalTransform, MeshRef};
    use crate::render::ObjectUBO;

    let transforms: Vec<(usize, Matrix4<f32>)> = ctx
        .world
        .iter_components::<MeshRef>()
        .map(|(entity, mesh_ref)| {
            let model_matrix = ctx
                .world
                .get_component::<GlobalTransform>(entity)
                .map(|gt| gt.0)
                .unwrap_or_else(Matrix4::identity);
            (mesh_ref.object_index, model_matrix)
        })
        .collect();

    for (object_index, model_matrix) in transforms {
        let ubo = ObjectUBO {
            model: model_matrix,
        };
        ctx.graphics
            .objects
            .update(ctx.device, ctx.image_index, object_index, &ubo)?;
    }

    Ok(())
}

unsafe fn update_frame_and_scene_uniforms(
    ctx: &mut FrameContext,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    screen_size: cgmath::Vector2<f32>,
    aspect: f32,
    camera_position: Vector3<f32>,
) -> Result<()> {
    let light_position = ctx.light_state().light_position;

    {
        let proj_data = ProjectionData {
            view,
            proj,
            screen_size,
            aspect,
        };
        let image_index = ctx.image_index;
        let mut backend = ctx.create_backend();
        backend.update_frame_ubo(
            &proj_data,
            camera_position,
            light_position,
            Vector3::new(1.0, 1.0, 1.0),
            image_index,
        )?;
    }

    update_mesh_entity_transforms(ctx)?;

    let light = ctx.light_state();
    let light_pos = light.light_position;
    let shadow_strength = light.shadow_strength;
    let distance_attenuation = light.distance_attenuation;
    drop(light);

    let debug_mode = ctx.debug_view_state().debug_view_mode.as_int();

    let exposure_value = ctx
        .world
        .get_resource::<Exposure>()
        .map(|e| e.exposure_value)
        .unwrap_or(1.0);

    let mut backend = ctx.create_backend();
    backend.update_scene_uniform(
        view,
        proj,
        light_pos,
        Vector3::new(1.0, 1.0, 1.0),
        debug_mode,
        shadow_strength,
        distance_attenuation,
        exposure_value,
    )?;

    Ok(())
}

fn collect_gizmo_render_data(
    ctx: &FrameContext,
    camera_position: Vector3<f32>,
) -> Vec<crate::ecs::component::RenderData> {
    let mut render_data_vec = vec![
        grid_mesh_render_data(&ctx.grid_mesh()),
        gizmo_mesh_render_data(&ctx.gizmo()),
        gizmo_selectable_render_data(&ctx.light_gizmo(), camera_position),
    ];

    if ctx.world.contains_resource::<BoneGizmoData>() {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        if bone_gizmo.visible {
            render_data_vec.extend(bone_gizmo_render_data(&bone_gizmo));
        }
    }

    if ctx.world.contains_resource::<CurveMeshData>() {
        let curves = ctx.world.resource::<CurveMeshData>();
        if curves.visible && !curves.mesh.indices.is_empty() {
            render_data_vec.push(curve_mesh_render_data(&curves));
        }
    }

    if ctx.world.contains_resource::<ConstraintGizmoData>() {
        let cg = ctx.world.resource::<ConstraintGizmoData>();
        if cg.visible {
            render_data_vec.extend(constraint_gizmo_render_data(&cg));
        }
    }

    if ctx.world.contains_resource::<SpringBoneGizmoData>() {
        let sg = ctx.world.resource::<SpringBoneGizmoData>();
        if sg.visible {
            render_data_vec.extend(spring_bone_gizmo_render_data(&sg));
        }
    }

    if ctx.world.contains_resource::<TransformGizmoData>() {
        let tg = ctx.world.resource::<TransformGizmoData>();
        let gizmo_scale = ctx
            .world
            .get_resource::<TransformGizmoState>()
            .map(|s| s.gizmo_scale)
            .unwrap_or(0.08);
        render_data_vec.extend(transform_gizmo_render_data(
            &tg,
            camera_position,
            gizmo_scale,
        ));
    }

    render_data_vec
}

unsafe fn update_billboard_ubo(
    ctx: &mut FrameContext,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
) -> Result<()> {
    let model_matrix = {
        let billboard = ctx.billboard();
        billboard
            .transform
            .as_ref()
            .map(|t| t.model_matrix)
            .unwrap_or(Matrix4::identity())
    };

    let image_index = ctx.image_index;
    ctx.update_billboard_ubo_internal(model_matrix, view, proj, image_index)?;

    Ok(())
}

unsafe fn update_grid_gizmo_buffers(ctx: &mut FrameContext) -> Result<()> {
    let mesh = ctx.gizmo().mesh.clone();
    let backend = ctx.create_backend();
    gizmo_update_vertex_buffer(&mesh, &backend)?;

    Ok(())
}

unsafe fn update_transform_gizmo_mesh(ctx: &mut FrameContext) -> Result<()> {
    if !ctx.world.contains_resource::<TransformGizmoData>() {
        return Ok(());
    }

    let visible = {
        let tg = ctx.world.resource::<TransformGizmoData>();
        tg.visible
    };

    if !visible {
        return Ok(());
    }

    let (mode, active_handle) = {
        let tg = ctx.world.resource::<TransformGizmoData>();
        let state = ctx
            .world
            .resource::<crate::ecs::resource::TransformGizmoState>();
        (state.mode, tg.active_handle)
    };

    let camera_dir = ctx.camera_direction();

    let mut line_mesh_clone = LineMesh::default();
    let mut solid_mesh_clone = LineMesh::default();

    match mode {
        crate::ecs::resource::TransformGizmoMode::Translate => {
            crate::ecs::systems::transform_gizmo_systems::build_translate_gizmo_meshes(
                active_handle,
                &mut line_mesh_clone,
                &mut solid_mesh_clone,
            );
        }
        crate::ecs::resource::TransformGizmoMode::Rotate => {
            crate::ecs::systems::transform_gizmo_systems::build_rotate_gizmo_meshes(
                active_handle,
                camera_dir,
                &mut line_mesh_clone,
                &mut solid_mesh_clone,
            );
        }
        crate::ecs::resource::TransformGizmoMode::Scale => {
            crate::ecs::systems::transform_gizmo_systems::build_scale_gizmo_meshes(
                active_handle,
                &mut line_mesh_clone,
                &mut solid_mesh_clone,
            );
        }
    }

    {
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut line_mesh_clone)?;
        backend.update_or_create_line_buffers(&mut solid_mesh_clone)?;
    }

    {
        let mut tg = ctx.world.resource_mut::<TransformGizmoData>();
        tg.line_mesh = line_mesh_clone;
        tg.solid_mesh = solid_mesh_clone;
    }

    Ok(())
}

unsafe fn update_curve_mesh(ctx: &mut FrameContext) -> Result<()> {
    if !ctx.world.contains_resource::<CurveMeshData>() {
        return Ok(());
    }

    let dirty = ctx.world.resource::<CurveMeshData>().dirty;
    if !dirty {
        return Ok(());
    }

    let mut mesh_clone = ctx.world.resource::<CurveMeshData>().mesh.clone();
    if !mesh_clone.indices.is_empty() {
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut mesh_clone)?;
    }

    let mut curves = ctx.world.resource_mut::<CurveMeshData>();
    curves.mesh.vertex_buffer_handle = mesh_clone.vertex_buffer_handle;
    curves.mesh.index_buffer_handle = mesh_clone.index_buffer_handle;
    curves.dirty = false;

    Ok(())
}

unsafe fn update_bone_gizmo_mesh(ctx: &mut FrameContext) -> Result<()> {
    if !ctx.world.contains_resource::<BoneGizmoData>() {
        return Ok(());
    }

    let (
        visible,
        display_style,
        skeleton_id,
        transforms,
        offsets,
        distance_scaling_enabled,
        distance_scaling_factor,
        mesh_scale,
    ) = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        (
            bone_gizmo.visible,
            bone_gizmo.display_style,
            bone_gizmo.cached_skeleton_id,
            bone_gizmo.cached_global_transforms.clone(),
            bone_gizmo.bone_local_offsets.clone(),
            bone_gizmo.distance_scaling_enabled,
            bone_gizmo.distance_scaling_factor,
            bone_gizmo.mesh_scale,
        )
    };

    if !visible {
        return Ok(());
    }

    let Some(skel_id) = skeleton_id else {
        return Ok(());
    };

    let Some(skeleton) = ctx.assets.get_skeleton_by_skeleton_id(skel_id) else {
        return Ok(());
    };
    let skeleton = skeleton.clone();

    let visual_scale = compute_visual_scale(
        ctx,
        &transforms,
        distance_scaling_enabled,
        distance_scaling_factor,
    );

    match display_style {
        BoneDisplayStyle::Stick => {
            update_stick_bone_mesh(ctx, &skeleton, &transforms, &offsets, mesh_scale)?;
        }
        BoneDisplayStyle::Octahedral => {
            update_octahedral_bone_mesh(
                ctx,
                &skeleton,
                &transforms,
                &offsets,
                visual_scale,
                mesh_scale,
            )?;
        }
        BoneDisplayStyle::Box => {
            update_box_bone_mesh(
                ctx,
                &skeleton,
                &transforms,
                &offsets,
                visual_scale,
                mesh_scale,
            )?;
        }
        BoneDisplayStyle::Sphere => {
            update_sphere_bone_mesh(
                ctx,
                &skeleton,
                &transforms,
                &offsets,
                visual_scale,
                mesh_scale,
            )?;
        }
    }

    Ok(())
}

fn compute_visual_scale(
    ctx: &FrameContext,
    transforms: &[Matrix4<f32>],
    distance_scaling_enabled: bool,
    distance_scaling_factor: f32,
) -> f32 {
    if !distance_scaling_enabled || transforms.is_empty() {
        return 1.0;
    }

    let camera_pos = {
        use crate::ecs::systems::camera_systems::compute_camera_position;
        compute_camera_position(&ctx.world.resource::<Camera>())
    };

    let mut center = Vector3::new(0.0f32, 0.0, 0.0);
    for t in transforms.iter() {
        center.x += t[3][0];
        center.y += t[3][1];
        center.z += t[3][2];
    }
    let count = transforms.len() as f32;
    center /= count;

    let distance = (center - camera_pos).magnitude();
    (distance * distance_scaling_factor).max(0.1)
}

unsafe fn update_stick_bone_mesh(
    ctx: &mut FrameContext,
    skeleton: &crate::animation::Skeleton,
    transforms: &[Matrix4<f32>],
    offsets: &[[f32; 3]],
    mesh_scale: f32,
) -> Result<()> {
    {
        let mut bone_gizmo = ctx.world.resource_mut::<BoneGizmoData>();
        build_bone_line_mesh(
            skeleton,
            transforms,
            offsets,
            mesh_scale,
            None,
            &mut bone_gizmo.stick_mesh,
        );
    }

    let mut mesh_clone = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        bone_gizmo.stick_mesh.clone()
    };

    {
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut mesh_clone)?;
    }

    {
        let mut bone_gizmo = ctx.world.resource_mut::<BoneGizmoData>();
        bone_gizmo.stick_mesh.vertex_buffer_handle = mesh_clone.vertex_buffer_handle;
        bone_gizmo.stick_mesh.index_buffer_handle = mesh_clone.index_buffer_handle;
    }

    Ok(())
}

unsafe fn update_octahedral_bone_mesh(
    ctx: &mut FrameContext,
    skeleton: &crate::animation::Skeleton,
    transforms: &[Matrix4<f32>],
    offsets: &[[f32; 3]],
    visual_scale: f32,
    mesh_scale: f32,
) -> Result<()> {
    let selection = ctx
        .world
        .get_resource::<BoneSelectionState>()
        .map(|s| (*s).clone())
        .unwrap_or_default();

    let mut solid_mesh = LineMesh::default();
    let mut wire_mesh = LineMesh::default();
    build_octahedral_bone_meshes_with_selection(
        skeleton,
        transforms,
        offsets,
        &selection,
        visual_scale,
        mesh_scale,
        None,
        &mut solid_mesh,
        &mut wire_mesh,
    );

    {
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut solid_mesh)?;
        backend.update_or_create_line_buffers(&mut wire_mesh)?;
    }

    {
        let mut bone_gizmo = ctx.world.resource_mut::<BoneGizmoData>();
        bone_gizmo.solid_mesh = solid_mesh;
        bone_gizmo.wire_mesh = wire_mesh;
    }

    Ok(())
}

unsafe fn update_box_bone_mesh(
    ctx: &mut FrameContext,
    skeleton: &crate::animation::Skeleton,
    transforms: &[Matrix4<f32>],
    offsets: &[[f32; 3]],
    visual_scale: f32,
    mesh_scale: f32,
) -> Result<()> {
    let selection = ctx
        .world
        .get_resource::<BoneSelectionState>()
        .map(|s| (*s).clone())
        .unwrap_or_default();

    let mut solid_mesh = LineMesh::default();
    let mut wire_mesh = LineMesh::default();
    build_box_bone_meshes_with_selection(
        skeleton,
        transforms,
        offsets,
        &selection,
        visual_scale,
        mesh_scale,
        None,
        &mut solid_mesh,
        &mut wire_mesh,
    );

    {
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut solid_mesh)?;
        backend.update_or_create_line_buffers(&mut wire_mesh)?;
    }

    {
        let mut bone_gizmo = ctx.world.resource_mut::<BoneGizmoData>();
        bone_gizmo.solid_mesh = solid_mesh;
        bone_gizmo.wire_mesh = wire_mesh;
    }

    Ok(())
}

unsafe fn update_sphere_bone_mesh(
    ctx: &mut FrameContext,
    skeleton: &crate::animation::Skeleton,
    transforms: &[Matrix4<f32>],
    offsets: &[[f32; 3]],
    visual_scale: f32,
    mesh_scale: f32,
) -> Result<()> {
    let selection = ctx
        .world
        .get_resource::<BoneSelectionState>()
        .map(|s| (*s).clone())
        .unwrap_or_default();

    let mut solid_mesh = LineMesh::default();
    let mut wire_mesh = LineMesh::default();
    build_sphere_bone_meshes_with_selection(
        skeleton,
        transforms,
        offsets,
        &selection,
        visual_scale,
        mesh_scale,
        None,
        &mut solid_mesh,
        &mut wire_mesh,
    );

    {
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut solid_mesh)?;
        backend.update_or_create_line_buffers(&mut wire_mesh)?;
    }

    {
        let mut bone_gizmo = ctx.world.resource_mut::<BoneGizmoData>();
        bone_gizmo.solid_mesh = solid_mesh;
        bone_gizmo.wire_mesh = wire_mesh;
    }

    Ok(())
}

unsafe fn update_constraint_gizmo_mesh(ctx: &mut FrameContext) -> Result<()> {
    if !ctx.world.contains_resource::<ConstraintGizmoData>() {
        return Ok(());
    }
    if !ctx.world.contains_resource::<BoneGizmoData>() {
        return Ok(());
    }

    let visible = {
        let cg = ctx.world.resource::<ConstraintGizmoData>();
        cg.visible
    };
    if !visible {
        return Ok(());
    }

    let (skeleton_id, transforms, offsets, constraint_mesh_scale) = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        (
            bone_gizmo.cached_skeleton_id,
            bone_gizmo.cached_global_transforms.clone(),
            bone_gizmo.bone_local_offsets.clone(),
            bone_gizmo.mesh_scale,
        )
    };

    let Some(skel_id) = skeleton_id else {
        return Ok(());
    };
    let Some(skeleton) = ctx.assets.get_skeleton_by_skeleton_id(skel_id) else {
        return Ok(());
    };
    let skeleton = skeleton.clone();

    let constraint_set = ctx
        .world
        .iter_constrained_entities()
        .next()
        .map(|(_, cs)| cs.clone());

    let Some(constraint_set) = constraint_set else {
        return Ok(());
    };

    let mut wire_mesh = LineMesh::default();
    build_constraint_gizmo_mesh(
        &constraint_set,
        &skeleton,
        &transforms,
        &offsets,
        constraint_mesh_scale,
        &mut wire_mesh,
    );

    {
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut wire_mesh)?;
    }

    {
        let mut cg = ctx.world.resource_mut::<ConstraintGizmoData>();
        cg.wire_mesh = wire_mesh;
    }

    Ok(())
}

unsafe fn update_spring_bone_gizmo_mesh(ctx: &mut FrameContext) -> Result<()> {
    if !ctx.world.contains_resource::<SpringBoneGizmoData>() {
        return Ok(());
    }
    if !ctx.world.contains_resource::<BoneGizmoData>() {
        return Ok(());
    }

    let visible = {
        let sg = ctx.world.resource::<SpringBoneGizmoData>();
        sg.visible
    };
    if !visible {
        return Ok(());
    }

    let (transforms, offsets, mesh_scale) = {
        let bone_gizmo = ctx.world.resource::<BoneGizmoData>();
        (
            bone_gizmo.cached_global_transforms.clone(),
            bone_gizmo.bone_local_offsets.clone(),
            bone_gizmo.mesh_scale,
        )
    };

    use crate::ecs::component::{SpringBoneSetup, WithSpringBone};
    let setup = ctx
        .world
        .iter_components::<WithSpringBone>()
        .next()
        .and_then(|(e, _)| ctx.world.get_component::<SpringBoneSetup>(e).cloned());

    let Some(setup) = setup else {
        return Ok(());
    };

    let mut wire_mesh = LineMesh::default();
    build_spring_bone_gizmo_mesh(&setup, &transforms, &offsets, mesh_scale, &mut wire_mesh);

    {
        let mut backend = ctx.create_backend();
        backend.update_or_create_line_buffers(&mut wire_mesh)?;
    }

    {
        let mut sg = ctx.world.resource_mut::<SpringBoneGizmoData>();
        sg.wire_mesh = wire_mesh;
    }

    Ok(())
}
