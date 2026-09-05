use vulkanalia::vk;

use thyllore_vulkan_core::raytracing::{BlasGeometry, GpuPrimitive};

use crate::ecs::component::WaterTorusEffect;
use crate::ecs::world::World;

pub trait GpuPrimitiveSource {
    fn gpu_primitive(&self) -> GpuPrimitive<'static>;
}

impl GpuPrimitiveSource for WaterTorusEffect {
    fn gpu_primitive(&self) -> GpuPrimitive<'static> {
        let model = thyllore_effect_core::build_water_ubo(self, 0).model;
        let extent = self.major_radius + self.minor_radius;

        GpuPrimitive {
            geometry: BlasGeometry::ProceduralAabb {
                aabb: vk::AabbPositionsKHR {
                    min_x: -extent,
                    min_y: -self.minor_radius,
                    min_z: -extent,
                    max_x: extent,
                    max_y: self.minor_radius,
                    max_z: extent,
                },
            },
            model,
            base_color: [1.0, 1.0, 1.0, 1.0],
            params: [1.0, self.major_radius, self.minor_radius, 0.0],
        }
    }
}

/// Entity order is the TLAS instance order, so it must stay stable across
/// frames and match the per-effect collectors used to refresh transforms.
pub fn collect_from<T>(world: &World) -> Vec<GpuPrimitive<'static>>
where
    T: crate::ecs::storage::Component + GpuPrimitiveSource,
{
    let mut sources: Vec<_> = world.iter_components::<T>().collect();
    sources.sort_by_key(|(entity, _)| *entity);

    sources
        .into_iter()
        .map(|(_, component)| component.gpu_primitive())
        .collect()
}

macro_rules! declare_gpu_primitive_collectors {
    ($($source_type:ty),* $(,)?) => {
        pub fn primitive_collectors() -> Vec<fn(&World) -> Vec<GpuPrimitive<'static>>> {
            vec![$(collect_from::<$source_type>),*]
        }
    };
}

declare_gpu_primitive_collectors!(WaterTorusEffect);
