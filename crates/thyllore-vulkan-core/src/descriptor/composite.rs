use crate::core::device::*;
use crate::descriptor::pass_manifest::COMPOSITE;
use crate::descriptor::reflected_layout::{ReflectedLayoutSpec, ReflectedSetLayout};
use crate::descriptor::shader_bindings::composite;
use crate::resource::buffer::create_buffer;
use crate::vulkan::*;

pub const MAX_SELECTED_OBJECTS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SelectionUBO {
    pub selected_ids: [[u32; 4]; MAX_SELECTED_OBJECTS],
    pub selected_count: u32,
    pub _padding: [u32; 3],
}

impl Default for SelectionUBO {
    fn default() -> Self {
        Self {
            selected_ids: [[0u32; 4]; MAX_SELECTED_OBJECTS],
            selected_count: 0,
            _padding: [0; 3],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CompositeGBufferViews {
    pub position_image_view: vk::ImageView,
    pub position_sampler: vk::Sampler,
    pub normal_image_view: vk::ImageView,
    pub normal_sampler: vk::Sampler,
    pub shadow_mask_image_view: vk::ImageView,
    pub shadow_mask_sampler: vk::Sampler,
    pub albedo_image_view: vk::ImageView,
    pub albedo_sampler: vk::Sampler,
    pub object_id_image_view: vk::ImageView,
    pub object_id_sampler: vk::Sampler,
}

#[derive(Clone, Debug, Default)]
pub struct RRCompositeDescriptorSet {
    pub layout: ReflectedSetLayout,
    pub descriptor_set: vk::DescriptorSet,
    pub selection_buffer: vk::Buffer,
    pub selection_buffer_memory: vk::DeviceMemory,
}

impl RRCompositeDescriptorSet {
    pub fn layout_spec() -> ReflectedLayoutSpec {
        ReflectedLayoutSpec::local(&COMPOSITE)
    }

    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let layout = ReflectedSetLayout::create(rrdevice, &Self::layout_spec())?;

        Ok(Self {
            layout,
            descriptor_set: vk::DescriptorSet::null(),
            selection_buffer: vk::Buffer::null(),
            selection_buffer_memory: vk::DeviceMemory::null(),
        })
    }

    pub unsafe fn allocate_and_update(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        gbuffer_views: CompositeGBufferViews,
        scene_uniform_buffer: vk::Buffer,
    ) -> Result<()> {
        self.descriptor_set = self.layout.allocate_set(rrdevice)?;

        let (selection_buffer, selection_buffer_memory) =
            Self::create_selection_buffer(instance, rrdevice)?;
        self.selection_buffer = selection_buffer;
        self.selection_buffer_memory = selection_buffer_memory;

        self.update_gbuffer_views(rrdevice, gbuffer_views)?;
        self.layout
            .writer(self.descriptor_set)
            .buffer(
                composite::SCENE_DATA,
                scene_uniform_buffer,
                0,
                std::mem::size_of::<crate::data::SceneUniformData>() as u64,
            )?
            .buffer(
                composite::SELECTION,
                selection_buffer,
                0,
                std::mem::size_of::<SelectionUBO>() as u64,
            )?
            .apply(rrdevice);
        Ok(())
    }

    pub unsafe fn update_gbuffer_views(
        &self,
        rrdevice: &RRDevice,
        views: CompositeGBufferViews,
    ) -> Result<()> {
        if self.descriptor_set == vk::DescriptorSet::null() {
            return Ok(());
        }

        self.layout
            .writer(self.descriptor_set)
            .image(
                composite::POSITION_SAMPLER,
                views.position_image_view,
                views.position_sampler,
                vk::ImageLayout::GENERAL,
            )?
            .image(
                composite::NORMAL_SAMPLER,
                views.normal_image_view,
                views.normal_sampler,
                vk::ImageLayout::GENERAL,
            )?
            .image(
                composite::SHADOW_MASK_SAMPLER,
                views.shadow_mask_image_view,
                views.shadow_mask_sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .image(
                composite::ALBEDO_SAMPLER,
                views.albedo_image_view,
                views.albedo_sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .image(
                composite::OBJECT_ID_SAMPLER,
                views.object_id_image_view,
                views.object_id_sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?
            .apply(rrdevice);
        Ok(())
    }

    unsafe fn create_selection_buffer(
        instance: &Instance,
        rrdevice: &RRDevice,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        create_buffer(
            instance,
            rrdevice,
            std::mem::size_of::<SelectionUBO>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    pub unsafe fn update_selection(
        &self,
        rrdevice: &RRDevice,
        selected_mesh_ids: &[u32],
    ) -> Result<()> {
        let mut ubo = SelectionUBO::default();
        let count = selected_mesh_ids.len().min(MAX_SELECTED_OBJECTS);

        for (i, &id) in selected_mesh_ids.iter().take(count).enumerate() {
            ubo.selected_ids[i] = [id, 0, 0, 0];
        }
        ubo.selected_count = count as u32;

        let memory = rrdevice.device.map_memory(
            self.selection_buffer_memory,
            0,
            std::mem::size_of::<SelectionUBO>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        std::ptr::copy_nonoverlapping(&ubo, memory as *mut SelectionUBO, 1);

        rrdevice.device.unmap_memory(self.selection_buffer_memory);

        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if self.selection_buffer != vk::Buffer::null() {
            device.destroy_buffer(self.selection_buffer, None);
            self.selection_buffer = vk::Buffer::null();
        }

        if self.selection_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(self.selection_buffer_memory, None);
            self.selection_buffer_memory = vk::DeviceMemory::null();
        }

        self.layout.destroy(device);
    }
}
