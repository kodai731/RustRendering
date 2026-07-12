use crate::command::*;
use crate::core::device::*;
use crate::vulkan::*;
use anyhow::Result;
use vulkanalia::vk::KhrAccelerationStructureExtension;

#[derive(Clone, Debug, Default)]
pub struct RRBLAS {
    pub acceleration_structure: Option<vk::AccelerationStructureKHR>,
    pub buffer: Option<vk::Buffer>,
    pub buffer_memory: Option<vk::DeviceMemory>,
    pub device_address: vk::DeviceAddress,
}

#[derive(Clone, Debug, Default)]
pub struct RRTLAS {
    pub acceleration_structure: Option<vk::AccelerationStructureKHR>,
    pub buffer: Option<vk::Buffer>,
    pub buffer_memory: Option<vk::DeviceMemory>,
    pub device_address: vk::DeviceAddress,
}

struct DeviceBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    address: vk::DeviceAddress,
}

#[derive(Clone, Debug)]
pub struct RRAccelerationStructure {
    pub blas_list: Vec<RRBLAS>,
    pub tlas: RRTLAS,
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

unsafe fn upload_instances_buffer(
    instance: &Instance,
    rrdevice: &RRDevice,
    blas_list: &[RRBLAS],
) -> Result<DeviceBuffer> {
    let instances: Vec<vk::AccelerationStructureInstanceKHR> = blas_list
        .iter()
        .enumerate()
        .map(|(i, blas)| vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
            },
            instance_custom_index_and_mask: vk::Bitfield24_8::new(i as u32, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Bitfield24_8::new(0, 0),
            acceleration_structure_reference: blas.device_address,
        })
        .collect();

    let instances_size = (std::mem::size_of::<vk::AccelerationStructureInstanceKHR>()
        * instances.len()) as vk::DeviceSize;

    let buf = allocate_device_buffer(
        instance,
        rrdevice,
        instances_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let ptr =
        rrdevice
            .device
            .map_memory(buf.memory, 0, instances_size, vk::MemoryMapFlags::empty())?
            as *mut vk::AccelerationStructureInstanceKHR;

    std::ptr::copy_nonoverlapping(instances.as_ptr(), ptr, instances.len());
    rrdevice.device.unmap_memory(buf.memory);

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
            tlas: RRTLAS::default(),
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
        })
    }

    pub unsafe fn create_tlas(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &RRCommandPool,
        blas_list: &[RRBLAS],
    ) -> Result<RRTLAS> {
        let device = &rrdevice.device;

        if blas_list.is_empty() {
            return Ok(RRTLAS::default());
        }

        let instances_buf = upload_instances_buffer(instance, rrdevice, blas_list)?;

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

        let primitive_count = blas_list.len() as u32;

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
        })
    }

    pub unsafe fn update_blas(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &RRCommandPool,
        blas: &RRBLAS,
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

        let scratch = allocate_device_buffer(
            instance,
            rrdevice,
            size_info.update_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

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
        destroy_device_buffer(device, &scratch);

        Ok(())
    }

    pub unsafe fn update_tlas(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &RRCommandPool,
        tlas: &RRTLAS,
        blas_list: &[RRBLAS],
    ) -> Result<()> {
        let device = &rrdevice.device;

        if blas_list.is_empty() {
            return Ok(());
        }

        let instances_buf = upload_instances_buffer(instance, rrdevice, blas_list)?;

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

        let primitive_count = blas_list.len() as u32;

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

        let scratch = allocate_device_buffer(
            instance,
            rrdevice,
            size_info.update_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

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

        destroy_device_buffer(device, &scratch);
        destroy_device_buffer(device, &instances_buf);

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
        }
        self.blas_list.clear();
    }

    pub unsafe fn update_all(
        &self,
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
                    &self.blas_list[i],
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
            &self.tlas,
            &self.blas_list,
        )?;

        Ok(())
    }
}
