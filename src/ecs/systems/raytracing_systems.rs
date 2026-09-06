use anyhow::Result;
use cgmath::{Matrix4, SquareMatrix};
use thyllore_vulkan_core::raytracing::{RRAccelerationStructure, RRBLAS};

use crate::app::FrameContext;
use crate::asset::AssetStorage;
use crate::ecs::component::WaterTorusEffect;
use crate::ecs::world::{GlobalTransform, MeshRef, World};

pub fn collect_water_instances(world: &World) -> Vec<(Matrix4<f32>, f32, f32)> {
    world
        .query_waters()
        .iter()
        .filter_map(|&entity| {
            let effect = world.get_component::<WaterTorusEffect>(entity)?;
            let ubo = thyllore_effect_core::build_water_ubo(effect, 0);
            Some((ubo.model, effect.major_radius, effect.minor_radius))
        })
        .collect()
}

pub fn collect_mesh_transforms(world: &World, assets: &AssetStorage) -> Vec<Matrix4<f32>> {
    let indexed_transforms: Vec<(usize, Matrix4<f32>)> = world
        .iter_components::<MeshRef>()
        .filter_map(|(entity, mesh_ref)| {
            let mesh_asset = assets.get_mesh(mesh_ref.mesh_asset_id)?;
            let model_matrix = world
                .get_component::<GlobalTransform>(entity)
                .map(|global_transform| global_transform.0)
                .unwrap_or_else(Matrix4::identity);
            Some((mesh_asset.graphics_mesh_index, model_matrix))
        })
        .collect();

    let transform_count = indexed_transforms
        .iter()
        .map(|(index, _)| index + 1)
        .max()
        .unwrap_or(0);

    let mut transforms = vec![Matrix4::identity(); transform_count];
    for (index, model_matrix) in indexed_transforms {
        transforms[index] = model_matrix;
    }

    transforms
}

pub unsafe fn refresh_tlas_mesh_transforms(ctx: &mut FrameContext) -> Result<()> {
    if !ctx.raytracing.has_valid_tlas() {
        return Ok(());
    }

    let mesh_transforms = collect_mesh_transforms(ctx.world, ctx.assets);
    let water_instances = collect_water_instances(ctx.world);
    let gbuffer_mesh_indices: Vec<usize> = ctx
        .graphics
        .meshes
        .iter()
        .enumerate()
        .filter(|(_, mesh)| mesh.render_to_gbuffer)
        .map(|(mesh_index, _)| mesh_index)
        .collect();

    let Some(acceleration_structure) = ctx.raytracing.acceleration_structure.as_mut() else {
        return Ok(());
    };

    if acceleration_structure.blas_list.len() != gbuffer_mesh_indices.len()
        || acceleration_structure.water_blas.len() != water_instances.len()
    {
        return Ok(());
    }

    let mut needs_update = false;
    for (blas, &mesh_index) in acceleration_structure
        .blas_list
        .iter_mut()
        .zip(gbuffer_mesh_indices.iter())
    {
        let model = mesh_transforms
            .get(mesh_index)
            .copied()
            .unwrap_or_else(Matrix4::identity);
        needs_update |= apply_instance_transform(blas, &model);
    }
    for (blas, (model, _, _)) in acceleration_structure
        .water_blas
        .iter_mut()
        .zip(water_instances.iter())
    {
        needs_update |= apply_instance_transform(blas, model);
    }

    if !needs_update {
        return Ok(());
    }

    RRAccelerationStructure::update_tlas(
        ctx.instance,
        ctx.device,
        ctx.command_pool.as_ref(),
        &mut acceleration_structure.tlas,
        &acceleration_structure.blas_list,
        &acceleration_structure.water_blas,
    )
}

fn apply_instance_transform(blas: &mut RRBLAS, model: &Matrix4<f32>) -> bool {
    let matrix = [
        [model[0][0], model[1][0], model[2][0], model[3][0]],
        [model[0][1], model[1][1], model[2][1], model[3][1]],
        [model[0][2], model[1][2], model[2][2], model[3][2]],
    ];
    if blas.transform.matrix == matrix {
        return false;
    }
    blas.transform.matrix = matrix;
    true
}
