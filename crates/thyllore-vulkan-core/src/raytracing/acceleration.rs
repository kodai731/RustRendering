use crate::command::*;
use crate::core::device::*;
use crate::resource::{HitShadingRecord, HitShadingTable};
use crate::vulkan::*;
use anyhow::Result;
use cgmath::Matrix4;
use thyllore_math_core::{AffineRows3x4, GpuMat4};
use vulkanalia::vk::KhrAccelerationStructureExtension;

#[derive(Clone, Debug, Default)]
pub struct RRBLAS {
    pub acceleration_structure: Option<vk::AccelerationStructureKHR>,
    pub buffer: Option<vk::Buffer>,
    pub buffer_memory: Option<vk::DeviceMemory>,
    pub device_address: vk::DeviceAddress,
    pub update_scratch: Option<DeviceBuffer>,
    pub transform: vk::TransformMatrixKHR,
}

#[derive(Clone, Debug, Default)]
pub struct RRTLAS {
    pub acceleration_structure: Option<vk::AccelerationStructureKHR>,
    pub buffer: Option<vk::Buffer>,
    pub buffer_memory: Option<vk::DeviceMemory>,
    pub device_address: vk::DeviceAddress,
    pub update_scratch: Option<DeviceBuffer>,
    pub instances_buf: Option<DeviceBuffer>,
}

#[derive(Clone, Debug, Default)]
struct DeviceBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    address: vk::DeviceAddress,
    size: vk::DeviceSize,
}

impl DeviceBuffer {
    fn buffer_size(&self) -> vk::DeviceSize {
        self.size
    }
}

#[derive(Clone, Debug)]
pub struct RRAccelerationStructure {
    pub blas_list: Vec<RRBLAS>,
    pub procedural_blas: Vec<RRBLAS>,
    pub tlas: RRTLAS,
    pub hit_shading_table: Option<HitShadingTable>,
}

unsafe fn allocate_device_buffer(
    instance: &Instance,
    rrdevice: &RRDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    memory_flags: vk::MemoryPropertyFlags,
) -> Result<DeviceBuffer> {
    let device = &rrdevice.device;

    let buffer_info = vk::BufferCreateInfo::builder().size(size).usage(usage);
    let buffer = device.create_buffer(&buffer_info, None)?;
    let memory_requirements = device.get_buffer_memory_requirements(buffer);

    let memory_type_index = get_memory_type_index(
        instance,
        rrdevice.physical_device,
        memory_flags,
        memory_requirements,
    )?;

    let mut allocate_flags_info =
        vk::MemoryAllocateFlagsInfo::builder().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(memory_requirements.size)
        .memory_type_index(memory_type_index)
        .push_next(&mut allocate_flags_info);

    let memory = device.allocate_memory(&memory_info, None)?;
    device.bind_buffer_memory(buffer, memory, 0)?;

    let address =
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::builder().buffer(buffer));

    Ok(DeviceBuffer {
        buffer,
        memory,
        address,
        size,
    })
}

unsafe fn destroy_device_buffer(device: &vulkanalia::Device, buf: &DeviceBuffer) {
    device.destroy_buffer(buf.buffer, None);
    device.free_memory(buf.memory, None);
}

unsafe fn execute_as_build(
    rrdevice: &RRDevice,
    rrcommand_pool: &RRCommandPool,
    build_info: &vk::AccelerationStructureBuildGeometryInfoKHRBuilder,
    primitive_count: u32,
) -> Result<()> {
    let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .primitive_count(primitive_count)
        .primitive_offset(0)
        .first_vertex(0)
        .transform_offset(0)
        .build();
    let build_range_infos = [build_range_info];

    let command_buffer = begin_single_time_commands(rrdevice, rrcommand_pool.command_pool)?;

    rrdevice.device.cmd_build_acceleration_structures_khr(
        command_buffer,
        std::slice::from_ref(build_info),
        &[&build_range_infos[0]],
    );

    end_single_time_commands(
        rrdevice,
        rrdevice.graphics_queue,
        rrcommand_pool.command_pool,
        command_buffer,
    )
}

unsafe fn create_acceleration_structure_with_buffer(
    instance: &Instance,
    rrdevice: &RRDevice,
    size: vk::DeviceSize,
    as_type: vk::AccelerationStructureTypeKHR,
) -> Result<(vk::AccelerationStructureKHR, vk::Buffer, vk::DeviceMemory)> {
    let as_buf = allocate_device_buffer(
        instance,
        rrdevice,
        size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let as_create_info = vk::AccelerationStructureCreateInfoKHR::builder()
        .buffer(as_buf.buffer)
        .size(size)
        .type_(as_type);

    let acceleration_structure = rrdevice
        .device
        .create_acceleration_structure_khr(&as_create_info, None)?;

    Ok((acceleration_structure, as_buf.buffer, as_buf.memory))
}

fn build_triangle_geometry(
    vertex_buffer_address: vk::DeviceAddress,
    vertex_count: u32,
    vertex_stride: u32,
    index_buffer_address: vk::DeviceAddress,
) -> vk::AccelerationStructureGeometryKHR {
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertex_buffer_address,
        })
        .vertex_stride(vertex_stride as vk::DeviceSize)
        .max_vertex(vertex_count - 1)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: index_buffer_address,
        })
        .build();

    vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .build()
}

unsafe fn fill_instances_buffer(
    rrdevice: &RRDevice,
    buf: &DeviceBuffer,
    instances_size: vk::DeviceSize,
    blas_list: &[RRBLAS],
    procedural_blas: &[RRBLAS],
) -> Result<()> {
    let ptr =
        rrdevice
            .device
            .map_memory(buf.memory, 0, instances_size, vk::MemoryMapFlags::empty())?
            as *mut vk::AccelerationStructureInstanceKHR;

    let mesh_count = blas_list.len();
    let total = mesh_count + procedural_blas.len();
    let mut instances: Vec<vk::AccelerationStructureInstanceKHR> = Vec::with_capacity(total);

    for (i, blas) in blas_list.iter().enumerate() {
        instances.push(vk::AccelerationStructureInstanceKHR {
            transform: blas.transform,
            instance_custom_index_and_mask: vk::Bitfield24_8::new(i as u32, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Bitfield24_8::new(0, 0),
            acceleration_structure_reference: blas.device_address,
        });
    }

    for (j, blas) in procedural_blas.iter().enumerate() {
        instances.push(vk::AccelerationStructureInstanceKHR {
            transform: blas.transform,
            instance_custom_index_and_mask: vk::Bitfield24_8::new((mesh_count + j) as u32, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Bitfield24_8::new(1, 0),
            acceleration_structure_reference: blas.device_address,
        });
    }

    std::ptr::copy_nonoverlapping(instances.as_ptr(), ptr, instances.len());
    rrdevice.device.unmap_memory(buf.memory);

    Ok(())
}

unsafe fn upload_instances_buffer(
    instance: &Instance,
    rrdevice: &RRDevice,
    blas_list: &[RRBLAS],
    procedural_blas: &[RRBLAS],
) -> Result<DeviceBuffer> {
    let total = blas_list.len() + procedural_blas.len();
    let instances_size =
        (std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() * total) as vk::DeviceSize;

    let buf = allocate_device_buffer(
        instance,
        rrdevice,
        instances_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    fill_instances_buffer(rrdevice, &buf, instances_size, blas_list, procedural_blas)?;

    Ok(buf)
}

const AS_BUILD_FLAGS: vk::BuildAccelerationStructureFlagsKHR =
    vk::BuildAccelerationStructureFlagsKHR::from_bits_truncate(
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE.bits()
            | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE.bits(),
    );

impl RRAccelerationStructure {
    pub fn new() -> Self {
        Self {
            blas_list: Vec::new(),
            procedural_blas: Vec::new(),
            tlas: RRTLAS::default(),
            hit_shading_table: None,
        }
    }

    pub unsafe fn create_blas(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &RRCommandPool,
        vertex_buffer: &vk::Buffer,
        vertex_count: u32,
        vertex_stride: u32,
        index_buffer: &vk::Buffer,
        index_count: u32,
    ) -> Result<RRBLAS> {
        let device = &rrdevice.device;

        let vertex_addr = device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::builder().buffer(*vertex_buffer),
        );
        let index_addr = device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::builder().buffer(*index_buffer),
        );

        let geometry =
            build_triangle_geometry(vertex_addr, vertex_count, vertex_stride, index_addr);
        let primitive_count = index_count / 3;

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .type_(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(AS_BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geometry));

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        device.get_acceleration_structure_build_sizes_khr(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[primitive_count],
            &mut size_info,
        );

        let (acceleration_structure, as_buffer, as_buffer_memory) =
            create_acceleration_structure_with_buffer(
                instance,
                rrdevice,
                size_info.acceleration_structure_size,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            )?;

        let scratch = allocate_device_buffer(
            instance,
            rrdevice,
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .type_(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(AS_BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(acceleration_structure)
            .geometries(std::slice::from_ref(&geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch.address,
            });

        execute_as_build(rrdevice, rrcommand_pool, &build_info, primitive_count)?;
        destroy_device_buffer(device, &scratch);

        let device_address = device.get_acceleration_structure_device_address_khr(
            &vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                .acceleration_structure(acceleration_structure),
        );

        Ok(RRBLAS {
            acceleration_structure: Some(acceleration_structure),
            buffer: Some(as_buffer),
            buffer_memory: Some(as_buffer_memory),
            device_address,
            update_scratch: None,
            transform: vk::TransformMatrixKHR {
                matrix: AffineRows3x4::IDENTITY.rows,
            },
        })
    }

    pub unsafe fn create_aabb_blas(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &RRCommandPool,
        aabb_buffer: &vk::Buffer,
        aabb_count: u32,
    ) -> Result<RRBLAS> {
        let device = &rrdevice.device;

        let aabb_addr = device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::builder().buffer(*aabb_buffer),
        );

        let aabbs_data = vk::AccelerationStructureGeometryAabbsDataKHR::builder()
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: aabb_addr,
            })
            .stride(24)
            .build();

        let geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::builder()
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: aabb_addr,
                    })
                    .stride(std::mem::size_of::<vk::AabbPositionsKHR>() as u64)
                    .build(),
            })
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .build();

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .type_(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(AS_BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geometry));

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        device.get_acceleration_structure_build_sizes_khr(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[aabb_count],
            &mut size_info,
        );

        let (acceleration_structure, as_buffer, as_buffer_memory) =
            create_acceleration_structure_with_buffer(
                instance,
                rrdevice,
                size_info.acceleration_structure_size,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            )?;

        let scratch = allocate_device_buffer(
            instance,
            rrdevice,
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .type_(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(AS_BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(acceleration_structure)
            .geometries(std::slice::from_ref(&geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch.address,
            });

        execute_as_build(rrdevice, rrcommand_pool, &build_info, aabb_count)?;
        destroy_device_buffer(device, &scratch);

        let device_address = device.get_acceleration_structure_device_address_khr(
            &vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                .acceleration_structure(acceleration_structure),
        );

        Ok(RRBLAS {
            acceleration_structure: Some(acceleration_structure),
            buffer: Some(as_buffer),
            buffer_memory: Some(as_buffer_memory),
            device_address,
            update_scratch: None,
            transform: vk::TransformMatrixKHR {
                matrix: AffineRows3x4::IDENTITY.rows,
            },
        })
    }

    pub unsafe fn create_procedural_blas(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &RRCommandPool,
        model: &Matrix4<f32>,
        aabb: vk::AabbPositionsKHR,
    ) -> Result<RRBLAS> {
        let size = std::mem::size_of::<vk::AabbPositionsKHR>() as vk::DeviceSize;
        let buf = allocate_device_buffer(
            instance,
            rrdevice,
            size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let ptr = rrdevice
            .device
            .map_memory(buf.memory, 0, size, vk::MemoryMapFlags::empty())?
            as *mut vk::AabbPositionsKHR;
        ptr.write(aabb);
        rrdevice.device.unmap_memory(buf.memory);
        let mut blas = Self::create_aabb_blas(instance, rrdevice, rrcommand_pool, &buf.buffer, 1)?;
        destroy_device_buffer(&rrdevice.device, &buf);
        blas.transform = vk::TransformMatrixKHR {
            matrix: AffineRows3x4::from_mat4(*model).rows,
        };
        Ok(blas)
    }

    pub unsafe fn create_tlas(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &RRCommandPool,
        blas_list: &[RRBLAS],
        procedural_blas: &[RRBLAS],
    ) -> Result<RRTLAS> {
        let device = &rrdevice.device;

        let instances_buf = if blas_list.is_empty() && procedural_blas.is_empty() {
            // Empty case: allocate buffer with 1 zero-initialized instance (mask=0, accelerationStructureReference=0)
            // This is an "inactive instance" per spec — the TLAS has 1 primitive but no hits.
            let instances_size =
                std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as vk::DeviceSize;

            let buf = allocate_device_buffer(
                instance,
                rrdevice,
                instances_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            // Zero-initialize: write a single zeroed instance
            let ptr = rrdevice.device.map_memory(
                buf.memory,
                0,
                instances_size,
                vk::MemoryMapFlags::empty(),
            )? as *mut vk::AccelerationStructureInstanceKHR;
            std::ptr::write(ptr, vk::AccelerationStructureInstanceKHR::default());
            rrdevice.device.unmap_memory(buf.memory);

            buf
        } else {
            upload_instances_buffer(instance, rrdevice, blas_list, procedural_blas)?
        };

        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::builder()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instances_buf.address,
            });

        let geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: *instances_data,
            })
            .flags(vk::GeometryFlagsKHR::OPAQUE);

        let primitive_count = if blas_list.is_empty() && procedural_blas.is_empty() {
            1
        } else {
            (blas_list.len() + procedural_blas.len()) as u32
        };

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .type_(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(AS_BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geometry));

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        device.get_acceleration_structure_build_sizes_khr(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[primitive_count],
            &mut size_info,
        );

        let (acceleration_structure, as_buffer, as_buffer_memory) =
            create_acceleration_structure_with_buffer(
                instance,
                rrdevice,
                size_info.acceleration_structure_size,
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            )?;

        let scratch = allocate_device_buffer(
            instance,
            rrdevice,
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .type_(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(AS_BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(acceleration_structure)
            .geometries(std::slice::from_ref(&geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch.address,
            });

        execute_as_build(rrdevice, rrcommand_pool, &build_info, primitive_count)?;

        destroy_device_buffer(device, &scratch);
        destroy_device_buffer(device, &instances_buf);

        let device_address = device.get_acceleration_structure_device_address_khr(
            &vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                .acceleration_structure(acceleration_structure),
        );

        Ok(RRTLAS {
            acceleration_structure: Some(acceleration_structure),
            buffer: Some(as_buffer),
            buffer_memory: Some(as_buffer_memory),
            device_address,
            instances_buf: None,
            update_scratch: None,
        })
    }

    pub unsafe fn update_blas(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &RRCommandPool,
        blas: &mut RRBLAS,
        vertex_buffer: &vk::Buffer,
        vertex_count: u32,
        vertex_stride: u32,
        index_buffer: &vk::Buffer,
        index_count: u32,
    ) -> Result<()> {
        let device = &rrdevice.device;

        let vertex_addr = device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::builder().buffer(*vertex_buffer),
        );
        let index_addr = device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::builder().buffer(*index_buffer),
        );

        let geometry =
            build_triangle_geometry(vertex_addr, vertex_count, vertex_stride, index_addr);
        let primitive_count = index_count / 3;

        let accel_structure = blas
            .acceleration_structure
            .ok_or_else(|| anyhow::anyhow!("BLAS acceleration structure not initialized"))?;

        let build_info_for_size = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .type_(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(AS_BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
            .geometries(std::slice::from_ref(&geometry));

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        device.get_acceleration_structure_build_sizes_khr(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info_for_size,
            &[primitive_count],
            &mut size_info,
        );

        let scratch = if let Some(ref existing) = blas.update_scratch {
            if size_info.update_scratch_size > existing.buffer_size() {
                destroy_device_buffer(device, &existing);
                let new = allocate_device_buffer(
                    instance,
                    rrdevice,
                    size_info.update_scratch_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?;
                blas.update_scratch = Some(new.clone());
                new
            } else {
                blas.update_scratch.clone().unwrap()
            }
        } else {
            let buf = allocate_device_buffer(
                instance,
                rrdevice,
                size_info.update_scratch_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            blas.update_scratch = Some(buf.clone());
            buf
        };

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .type_(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(AS_BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
            .src_acceleration_structure(accel_structure)
            .dst_acceleration_structure(accel_structure)
            .geometries(std::slice::from_ref(&geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch.address,
            });

        execute_as_build(rrdevice, rrcommand_pool, &build_info, primitive_count)?;

        Ok(())
    }

    pub unsafe fn update_tlas(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &RRCommandPool,
        tlas: &mut RRTLAS,
        blas_list: &[RRBLAS],
        procedural_blas: &[RRBLAS],
    ) -> Result<()> {
        let device = &rrdevice.device;

        if blas_list.is_empty() && procedural_blas.is_empty() {
            return Ok(());
        }

        let total = blas_list.len() + procedural_blas.len();
        let instances_size =
            (std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() * total) as vk::DeviceSize;

        let instances_buf = if let Some(ref existing) = tlas.instances_buf {
            if instances_size > existing.buffer_size() {
                destroy_device_buffer(device, &existing);
                let new = upload_instances_buffer(instance, rrdevice, blas_list, procedural_blas)?;
                tlas.instances_buf = Some(new.clone());
                new
            } else {
                let buf = tlas.instances_buf.clone().unwrap();
                fill_instances_buffer(rrdevice, &buf, instances_size, blas_list, procedural_blas)?;
                buf
            }
        } else {
            let buf = upload_instances_buffer(instance, rrdevice, blas_list, procedural_blas)?;
            tlas.instances_buf = Some(buf.clone());
            buf
        };

        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::builder()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instances_buf.address,
            });

        let geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: *instances_data,
            })
            .flags(vk::GeometryFlagsKHR::OPAQUE);

        let primitive_count = total as u32;

        let accel_structure = tlas
            .acceleration_structure
            .ok_or_else(|| anyhow::anyhow!("TLAS acceleration structure not initialized"))?;

        let build_info_for_size = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .type_(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(AS_BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
            .geometries(std::slice::from_ref(&geometry));

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        device.get_acceleration_structure_build_sizes_khr(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info_for_size,
            &[primitive_count],
            &mut size_info,
        );

        let scratch = if let Some(ref existing) = tlas.update_scratch {
            if size_info.update_scratch_size > existing.buffer_size() {
                destroy_device_buffer(device, &existing);
                let new = allocate_device_buffer(
                    instance,
                    rrdevice,
                    size_info.update_scratch_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?;
                tlas.update_scratch = Some(new.clone());
                new
            } else {
                tlas.update_scratch.clone().unwrap()
            }
        } else {
            let buf = allocate_device_buffer(
                instance,
                rrdevice,
                size_info.update_scratch_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            tlas.update_scratch = Some(buf.clone());
            buf
        };

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .type_(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(AS_BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
            .src_acceleration_structure(accel_structure)
            .dst_acceleration_structure(accel_structure)
            .geometries(std::slice::from_ref(&geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch.address,
            });

        execute_as_build(rrdevice, rrcommand_pool, &build_info, primitive_count)?;

        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if let Some(tlas_as) = self.tlas.acceleration_structure {
            device.destroy_acceleration_structure_khr(tlas_as, None);
        }
        if let Some(buffer) = self.tlas.buffer {
            device.destroy_buffer(buffer, None);
        }
        if let Some(memory) = self.tlas.buffer_memory {
            device.free_memory(memory, None);
        }
        if let Some(ref scratch) = self.tlas.update_scratch {
            destroy_device_buffer(device, scratch);
        }
        if let Some(ref instances) = self.tlas.instances_buf {
            destroy_device_buffer(device, instances);
        }

        for blas in &mut self.blas_list {
            if let Some(blas_as) = blas.acceleration_structure {
                device.destroy_acceleration_structure_khr(blas_as, None);
            }
            if let Some(buffer) = blas.buffer {
                device.destroy_buffer(buffer, None);
            }
            if let Some(memory) = blas.buffer_memory {
                device.free_memory(memory, None);
            }
            if let Some(ref scratch) = blas.update_scratch {
                destroy_device_buffer(device, scratch);
            }
        }

        for blas in &mut self.procedural_blas {
            if let Some(blas_as) = blas.acceleration_structure {
                device.destroy_acceleration_structure_khr(blas_as, None);
            }
            if let Some(buffer) = blas.buffer {
                device.destroy_buffer(buffer, None);
            }
            if let Some(memory) = blas.buffer_memory {
                device.free_memory(memory, None);
            }
            if let Some(ref scratch) = blas.update_scratch {
                destroy_device_buffer(device, scratch);
            }
        }

        if let Some(table) = self.hit_shading_table.take() {
            device.destroy_buffer(table.buffer, None);
            device.free_memory(table.memory, None);
        }
        self.blas_list.clear();
        self.procedural_blas.clear();
    }

    pub unsafe fn update_all(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &RRCommandPool,
        vertex_buffers: &[(&vk::Buffer, u32, u32, &vk::Buffer, u32)],
    ) -> Result<()> {
        for (i, (vertex_buffer, vertex_count, vertex_stride, index_buffer, index_count)) in
            vertex_buffers.iter().enumerate()
        {
            if i < self.blas_list.len() {
                Self::update_blas(
                    instance,
                    rrdevice,
                    rrcommand_pool,
                    &mut self.blas_list[i],
                    vertex_buffer,
                    *vertex_count,
                    *vertex_stride,
                    index_buffer,
                    *index_count,
                )?;
            }
        }

        Self::update_tlas(
            instance,
            rrdevice,
            rrcommand_pool,
            &mut self.tlas,
            &self.blas_list,
            &self.procedural_blas,
        )?;

        self.fill_hit_shading_table(instance, rrdevice, vertex_buffers, &[])?;

        Ok(())
    }

    pub unsafe fn fill_hit_shading_table(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        vertex_buffers: &[(&vk::Buffer, u32, u32, &vk::Buffer, u32)],
        procedurals: &[(Matrix4<f32>, [f32; 4])],
    ) -> Result<()> {
        let mut records: Vec<HitShadingRecord> =
            Vec::with_capacity(vertex_buffers.len() + procedurals.len());

        for (vertex_buffer, _, _, index_buffer, _) in vertex_buffers.iter() {
            let vertex_address = rrdevice.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder().buffer(**vertex_buffer),
            );
            let index_address = rrdevice.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder().buffer(**index_buffer),
            );

            records.push(HitShadingRecord {
                vertex_address,
                index_address,
                model: GpuMat4::IDENTITY,
                normal_matrix: GpuMat4::IDENTITY,
                base_color: [1.0, 1.0, 1.0, 1.0],
                params: [0.0; 4],
            });
        }

        for (model, params) in procedurals.iter() {
            records.push(HitShadingRecord {
                vertex_address: 0,
                index_address: 0,
                model: GpuMat4::from_mat4(*model),
                normal_matrix: GpuMat4::normal_matrix_of(*model),
                base_color: [1.0, 1.0, 1.0, 1.0],
                params: *params,
            });
        }

        if records.is_empty() {
            records.push(HitShadingRecord::default_record());
        }

        if !records.is_empty() {
            let table = match &mut self.hit_shading_table {
                Some(table) if table.capacity >= records.len() => table,
                _ => {
                    let new_table = HitShadingTable::new(instance, rrdevice, records.len())?;
                    if let Some(old_table) = self.hit_shading_table.take() {
                        old_table.destroy(rrdevice);
                    }
                    self.hit_shading_table.insert(new_table)
                }
            };
            table.upload(rrdevice, &records)?;
        }

        Ok(())
    }
}
