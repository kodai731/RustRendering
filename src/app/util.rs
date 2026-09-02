use crate::app::App;
use crate::vulkanr::command::RRCommandPool;
use crate::vulkanr::vulkan::*;

use anyhow::{anyhow, Result};

impl App {
    pub(crate) unsafe fn get_memory_type_index(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        let mem_properties = self
            .instance
            .get_physical_device_memory_properties(self.rrdevice.physical_device);

        for i in 0..mem_properties.memory_type_count {
            let has_type = (type_filter & (1 << i)) != 0;
            let has_properties = mem_properties.memory_types[i as usize]
                .property_flags
                .contains(properties);

            if has_type && has_properties {
                return Ok(i);
            }
        }

        Err(anyhow!("Failed to find suitable memory type"))
    }

    pub(crate) unsafe fn transition_image_layout_and_copy(
        device: &vulkanalia::Device,
        command_pool: &RRCommandPool,
        graphics_queue: &vk::Queue,
        image: vk::Image,
        buffer: vk::Buffer,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device.begin_command_buffer(command_buffer, &begin_info)?;

        // Transition to TRANSFER_DST_OPTIMAL
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        // Copy buffer to image
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        // Transition to SHADER_READ_ONLY_OPTIMAL
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        device.end_command_buffer(command_buffer)?;

        // Submit command buffer
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);

        device.queue_submit(*graphics_queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(*graphics_queue)?;

        device.free_command_buffers(command_pool.command_pool, &[command_buffer]);

        Ok(())
    }
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if (bits & 0x8000) != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x3FF) as f32;

    if exp == 0 {
        if mantissa == 0.0 {
            return sign * 0.0;
        }
        sign * (mantissa / 1024.0) * 2f32.powi(-14)
    } else if exp == 31 {
        if mantissa != 0.0 {
            f32::NAN
        } else {
            sign * f32::INFINITY
        }
    } else {
        let e = exp as i32 - 15;
        sign * (1.0 + mantissa / 1024.0) * 2f32.powi(e)
    }
}

pub fn write_npy_f32(path: &std::path::Path, shape: &[usize], data: &[f32]) -> anyhow::Result<()> {
    write_npy_le4(path, "<f4", shape, data.iter().map(|v| v.to_le_bytes()))
}

pub fn write_npy_u32(path: &std::path::Path, shape: &[usize], data: &[u32]) -> anyhow::Result<()> {
    write_npy_le4(path, "<u4", shape, data.iter().map(|v| v.to_le_bytes()))
}

fn write_npy_le4(
    path: &std::path::Path,
    dtype_descriptor: &str,
    shape: &[usize],
    elements: impl Iterator<Item = [u8; 4]>,
) -> anyhow::Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(b"\x93NUMPY")?;
    writer.write_all(&[1u8])?;
    writer.write_all(&[0u8])?;

    let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    let header_content = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({}) }}",
        dtype_descriptor,
        shape_str.join(", ")
    );

    // NPY v1.0 requires the data to start at a multiple of 64 bytes, header terminated by \n
    let terminated_len = header_content.len() + 1;
    let padding = (64 - ((10 + terminated_len) % 64)) % 64;
    let header_len = terminated_len + padding;

    writer.write_all(&(header_len as u16).to_le_bytes())?;
    writer.write_all(header_content.as_bytes())?;
    for _ in 0..padding {
        writer.write_all(b" ")?;
    }
    writer.write_all(b"\n")?;

    for bytes in elements {
        writer.write_all(&bytes)?;
    }
    writer.flush()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::{BufReader, Read};
    use std::path::PathBuf;

    #[test]
    fn test_write_npy_f32_format() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let shape = [2, 3, 4];

        let tmp_dir = std::env::temp_dir();
        let path = PathBuf::from(tmp_dir.join("test_npy_f32.npy"));

        write_npy_f32(&path, &shape, &data).unwrap();

        let file = File::open(&path).unwrap();
        let mut reader = BufReader::new(file);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).unwrap();

        std::fs::remove_file(&path).ok();

        // Verify magic: \x93NUMPY\x01\x00
        assert_eq!(&bytes[0..6], b"\x93NUMPY");
        assert_eq!(bytes[6], 1);
        assert_eq!(bytes[7], 0);

        // Read u16 LE header length at offset 8
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;

        // Extract header string (starts at byte 10, ends with \n)
        let header_start = 10;
        let header_end = header_start + header_len;
        let header_str = &bytes[header_start..header_end];

        // Verify total header (10 + header_len) is a multiple of 64
        assert_eq!(
            (10 + header_len) % 64,
            0,
            "total header length {} is not a multiple of 64",
            10 + header_len
        );

        // Verify no null bytes in header string
        for (i, &b) in header_str.iter().enumerate() {
            assert_ne!(b, 0u8, "null byte at header offset {}", i);
        }

        // Verify header ends with \n
        assert_eq!(
            header_str.last(),
            Some(&b'\n'),
            "header does not end with newline"
        );

        // Verify header content starts with expected dict
        let header_text = std::str::from_utf8(header_str).unwrap();
        assert!(
            header_text.contains("'descr': '<f4'"),
            "missing descr in header"
        );
        assert!(
            header_text.contains("'fortran_order': False"),
            "missing fortran_order in header"
        );
        assert!(
            header_text.contains("'shape': (2, 3, 4)"),
            "wrong shape in header"
        );

        // Verify data: 24 f32 values in LE after the header
        let data_start = header_end;
        assert_eq!(bytes.len(), data_start + 24 * 4, "unexpected total length");

        for i in 0..24 {
            let val = f32::from_le_bytes([
                bytes[data_start + i * 4],
                bytes[data_start + i * 4 + 1],
                bytes[data_start + i * 4 + 2],
                bytes[data_start + i * 4 + 3],
            ]);
            assert_eq!(val, i as f32, "data mismatch at index {}", i);
        }
    }

    #[test]
    fn test_write_npy_u32_preserves_all_bits() {
        let data: Vec<u32> = vec![0, 1, 0x00FF_FFFF, 0x0100_0001, 0x7FFF_FFFF, u32::MAX];
        let shape = [2, 3];

        let path = std::env::temp_dir().join("test_npy_u32.npy");
        write_npy_u32(&path, &shape, &data).unwrap();

        let mut bytes = Vec::new();
        BufReader::new(File::open(&path).unwrap())
            .read_to_end(&mut bytes)
            .unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(&bytes[0..6], b"\x93NUMPY");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);

        let header_end = 10 + header_len;
        let header_text = std::str::from_utf8(&bytes[10..header_end]).unwrap();
        assert!(header_text.contains("'descr': '<u4'"));
        assert!(header_text.contains("'shape': (2, 3)"));
        assert!(header_text.ends_with('\n'));

        assert_eq!(bytes.len(), header_end + data.len() * 4);
        for (index, &expected) in data.iter().enumerate() {
            let offset = header_end + index * 4;
            let val = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            assert_eq!(val, expected, "data mismatch at index {}", index);
        }
    }

    #[test]
    fn test_f16_to_f32() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        let val = f16_to_f32(0x3555);
        assert!((val - 0.33325).abs() / 0.33325 < 1e-4, "got {}", val);
        let val = f16_to_f32(0x0001);
        let expected = 5.960464e-8;
        assert!(
            (val - expected).abs() / expected < 1e-3,
            "subnormal: got {}, expected {}",
            val,
            expected
        );
        assert!(f16_to_f32(0x7C00).is_infinite());
    }
}
