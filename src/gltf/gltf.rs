use anyhow::{anyhow, Result};
use core::result::Result::Ok;
use glium::buffer::Content;
use gltf::animation::util::morph_target_weights;
use gltf::buffer::Data;
use gltf::{image, Document, Gltf, Node};

pub struct GltfData {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub tex_coords: Vec<[f32; 2]>,
    pub joint_indices: Vec<[u16; 4]>,
    pub joint_weights: Vec<[f32; 4]>,
    pub morph_positions: Vec<[f32; 3]>,
    pub morph_normals: Vec<[f32; 3]>,
    pub morph_tangents: Vec<[f32; 3]>,
    pub image_indices: Vec<[u16; 4]>,
    pub image_data: Vec<ImageData>,
}

impl GltfData {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            indices: Vec::new(),
            tex_coords: Vec::new(),
            joint_indices: Vec::new(),
            joint_weights: Vec::new(),
            morph_positions: Vec::new(),
            morph_normals: Vec::new(),
            morph_tangents: Vec::new(),
            image_indices: Vec::new(),
            image_data: Vec::new(),
        }
    }
}

pub struct ImageData {
    pub data: Vec<u8>,
    pub size: u64,
    pub width: u32,
    pub height: u32,
}

pub unsafe fn load_gltf(path: &str) -> Result<GltfData> {
    let (gltf, buffers, images) = gltf::import(format!("{}", path)).expect("Failed to load grass");
    let mut gltf_data = GltfData::new();
    for scene in gltf.scenes() {
        for node in scene.nodes() {
            process_node(&gltf, &buffers, &images, &node, &mut gltf_data)?;
        }
    }
    Ok(gltf_data)
}

unsafe fn process_node(
    gltf: &Document,
    buffers: &Vec<Data>,
    images: &Vec<gltf::image::Data>,
    node: &Node,
    gltf_data: &mut GltfData,
) -> Result<()> {
    println!("Node {} {}", node.index().to_string(), node.name().unwrap());
    if let Some(mesh) = node.mesh() {
        println!("mesh found");
        let primitives = mesh.primitives();
        let mut normals = Vec::new();
        let mut joint_indices = Vec::new();
        let mut joint_weights = Vec::new();
        primitives.for_each(|primitive| {
            println!("primitive found");
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            println!("Topology: {:?}", primitive.mode());

            let index_offset = gltf_data.indices.len();
            if let Some(gltf::mesh::util::ReadIndices::U16(gltf::accessor::Iter::Standard(iter))) =
                reader.read_indices()
            {
                for index in iter {
                    gltf_data
                        .indices
                        .push((index_offset + index as usize) as u32);
                }
            }

            if let Some(iter) = reader.read_positions() {
                for position in iter {
                    let mut position_converted = position;
                    position_converted[1] = 1.0 - position_converted[1];
                    gltf_data.positions.push(position_converted);
                }
            }

            if let Some(gltf::mesh::util::ReadTexCoords::F32(gltf::accessor::Iter::Standard(
                iter,
            ))) = reader.read_tex_coords(0)
            {
                for texture_coord in iter {
                    gltf_data.tex_coords.push(texture_coord);
                }
            }

            if let Some(iter) = reader.read_normals() {
                for normal in iter {
                    normals.push(normal);
                }
            }

            // texture
            if let Some(material) = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
            {
                let texture = material.texture();
                let image = &images[texture.source().index()];

                let size = (size_of::<u8>() * image.pixels.len()) as u64;
                let (width, height) = (image.width, image.height);
                let image_data = ImageData {
                    data: image.pixels.clone(),
                    size: size,
                    width: width,
                    height: height,
                };
                gltf_data.image_data.push(image_data);
            }

            // joint
            if let Some(iter) = reader.read_joints(0) {
                for joint in iter.into_u16() {
                    print!("Joint: {:?}", joint);
                    joint_indices.push(joint);
                    gltf_data.joint_indices.push(joint);
                }
            }

            if let Some(iter) = reader.read_weights(0) {
                for weight in iter.into_f32() {
                    println!("weight: {:?}", weight);
                    joint_weights.push(weight);
                    gltf_data.joint_weights.push(weight);
                }
            }

            // morph targets
            if let morph_targets = reader.read_morph_targets() {
                for target in morph_targets {
                    let (positions, normals, tangents) = target;
                    // positions
                    if let Some(position_iter) = positions {
                        for position in position_iter {
                            gltf_data.morph_positions.push(position);
                        }
                    }
                    // normals
                    if let Some(normal_iter) = normals {
                        for normal in normal_iter {
                            gltf_data.morph_normals.push(normal);
                        }
                    }
                    // tangents
                    if let Some(tangent_iter) = tangents {
                        for tangent in tangent_iter {
                            gltf_data.morph_tangents.push(tangent);
                        }
                    }
                }
            }
        });
    }
    for child in node.children() {
        process_node(gltf, buffers, images, &child, gltf_data)?;
    }

    // check
    println!("indices count {}", gltf_data.indices.len());
    println!("joint indices count {}", gltf_data.joint_indices.len());
    println!("joint weights count {}", gltf_data.joint_weights.len());
    println!("morph position count {}", gltf_data.morph_positions.len());
    println!("morph normal count {}", gltf_data.morph_normals.len());
    println!("morph tangent count {}", gltf_data.morph_tangents.len());

    Ok(())
}
