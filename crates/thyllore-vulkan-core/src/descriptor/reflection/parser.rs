use std::collections::HashMap;

use super::types::{
    DescriptorCount, DescriptorKind, ReflectError, ReflectedBinding, ShaderReflection, ShaderStage,
};

const SPIRV_MAGIC: u32 = 0x0723_0203;
const HEADER_WORD_COUNT: usize = 5;

const OP_NAME: u32 = 5;
const OP_ENTRY_POINT: u32 = 15;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_MATRIX: u32 = 24;
const OP_TYPE_IMAGE: u32 = 25;
const OP_TYPE_SAMPLER: u32 = 26;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const OP_TYPE_ACCELERATION_STRUCTURE_KHR: u32 = 5341;

const DECORATION_ROW_MAJOR: u32 = 4;
const DECORATION_BLOCK: u32 = 2;
const DECORATION_BUFFER_BLOCK: u32 = 3;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_MATRIX_STRIDE: u32 = 7;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;

const STORAGE_CLASS_UNIFORM_CONSTANT: u32 = 0;
const STORAGE_CLASS_UNIFORM: u32 = 2;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

const DIM_BUFFER: u32 = 5;
const DIM_SUBPASS_DATA: u32 = 6;
const IMAGE_SAMPLED_WITH_SAMPLER: u32 = 1;
const IMAGE_SAMPLED_STORAGE: u32 = 2;

pub fn reflect_shader_bytes(bytes: &[u8]) -> Result<ShaderReflection, ReflectError> {
    if bytes.len() % 4 != 0 {
        return Err(ReflectError::Truncated);
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    reflect_shader_words(&words)
}

pub fn reflect_shader_words(words: &[u32]) -> Result<ShaderReflection, ReflectError> {
    if words.len() < HEADER_WORD_COUNT {
        return Err(ReflectError::Truncated);
    }
    if words[0] != SPIRV_MAGIC {
        return Err(ReflectError::InvalidMagic);
    }

    let module = SpirvModule::decode(&words[HEADER_WORD_COUNT..])?;
    let mut bindings = Vec::new();
    for variable in &module.variables {
        if let Some(binding) = module.resolve_binding(variable)? {
            bindings.push(binding);
        }
    }
    bindings.sort_by_key(|binding| (binding.set, binding.binding));

    Ok(ShaderReflection {
        stages: module.stages,
        bindings,
    })
}

#[derive(Debug)]
enum TypeDef {
    Bool,
    Int { width: u32 },
    Float { width: u32 },
    Vector { component: u32, count: u32 },
    Matrix { column: u32, columns: u32 },
    Image { dim: u32, sampled: u32 },
    Sampler,
    SampledImage,
    Array { element: u32, length_id: u32 },
    RuntimeArray { element: u32 },
    Struct { members: Vec<u32> },
    Pointer { pointee: u32 },
    AccelerationStructure,
}

#[derive(Debug, Default, Clone)]
struct Decorations {
    descriptor_set: Option<u32>,
    binding: Option<u32>,
    array_stride: Option<u32>,
    is_block: bool,
    is_buffer_block: bool,
}

#[derive(Debug, Default, Clone)]
struct MemberDecorations {
    offset: Option<u32>,
    matrix_stride: Option<u32>,
    row_major: bool,
}

#[derive(Debug)]
struct Variable {
    id: u32,
    pointer_type: u32,
    storage_class: u32,
}

#[derive(Debug, Default)]
struct SpirvModule {
    names: HashMap<u32, String>,
    decorations: HashMap<u32, Decorations>,
    member_decorations: HashMap<(u32, u32), MemberDecorations>,
    types: HashMap<u32, TypeDef>,
    constants: HashMap<u32, u32>,
    variables: Vec<Variable>,
    stages: Vec<ShaderStage>,
}

impl SpirvModule {
    fn decode(instructions: &[u32]) -> Result<Self, ReflectError> {
        let mut module = Self::default();
        let mut cursor = 0;

        while cursor < instructions.len() {
            let word_count = (instructions[cursor] >> 16) as usize;
            let opcode = instructions[cursor] & 0xFFFF;
            if word_count == 0 || cursor + word_count > instructions.len() {
                return Err(ReflectError::Truncated);
            }
            let operands = &instructions[cursor + 1..cursor + word_count];
            module.record_instruction(opcode, operands)?;
            cursor += word_count;
        }

        Ok(module)
    }

    fn record_instruction(&mut self, opcode: u32, operands: &[u32]) -> Result<(), ReflectError> {
        let operand = |index: usize| operands.get(index).copied().ok_or(ReflectError::Truncated);

        match opcode {
            OP_ENTRY_POINT => {
                let model = operand(0)?;
                let stage = execution_model_to_stage(model)
                    .ok_or(ReflectError::UnknownExecutionModel(model))?;
                if !self.stages.contains(&stage) {
                    self.stages.push(stage);
                }
            }
            OP_NAME => {
                self.names
                    .insert(operand(0)?, decode_literal_string(&operands[1..]));
            }
            OP_DECORATE => self.record_decoration(operand(0)?, operand(1)?, operands.get(2)),
            OP_MEMBER_DECORATE => self.record_member_decoration(
                (operand(0)?, operand(1)?),
                operand(2)?,
                operands.get(3),
            ),
            OP_CONSTANT => {
                self.constants.insert(operand(1)?, operand(2)?);
            }
            OP_VARIABLE => self.variables.push(Variable {
                id: operand(1)?,
                pointer_type: operand(0)?,
                storage_class: operand(2)?,
            }),
            _ => {
                if let Some(type_def) = decode_type(opcode, operands)? {
                    self.types.insert(operand(0)?, type_def);
                }
            }
        }
        Ok(())
    }

    fn record_decoration(&mut self, target: u32, decoration: u32, argument: Option<&u32>) {
        let entry = self.decorations.entry(target).or_default();
        let argument = argument.copied();
        match decoration {
            DECORATION_DESCRIPTOR_SET => entry.descriptor_set = argument,
            DECORATION_BINDING => entry.binding = argument,
            DECORATION_ARRAY_STRIDE => entry.array_stride = argument,
            DECORATION_BLOCK => entry.is_block = true,
            DECORATION_BUFFER_BLOCK => entry.is_buffer_block = true,
            _ => {}
        }
    }

    fn record_member_decoration(
        &mut self,
        target: (u32, u32),
        decoration: u32,
        argument: Option<&u32>,
    ) {
        let entry = self.member_decorations.entry(target).or_default();
        let argument = argument.copied();
        match decoration {
            DECORATION_OFFSET => entry.offset = argument,
            DECORATION_MATRIX_STRIDE => entry.matrix_stride = argument,
            DECORATION_ROW_MAJOR => entry.row_major = true,
            _ => {}
        }
    }

    fn type_def(&self, type_id: u32) -> Result<&TypeDef, ReflectError> {
        self.types
            .get(&type_id)
            .ok_or(ReflectError::MissingType(type_id))
    }

    fn name_of(&self, id: u32) -> String {
        self.names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("%{id}"))
    }

    fn resolve_binding(
        &self,
        variable: &Variable,
    ) -> Result<Option<ReflectedBinding>, ReflectError> {
        let is_descriptor_class = matches!(
            variable.storage_class,
            STORAGE_CLASS_UNIFORM_CONSTANT | STORAGE_CLASS_UNIFORM | STORAGE_CLASS_STORAGE_BUFFER
        );
        if !is_descriptor_class {
            return Ok(None);
        }

        let name = self.name_of(variable.id);
        let TypeDef::Pointer { pointee } = self.type_def(variable.pointer_type)? else {
            return Err(ReflectError::UnsupportedVariableType(name));
        };
        let (resource_type, count) = self.strip_arrays(*pointee)?;
        let (kind, block_size) =
            self.classify_resource(variable.storage_class, resource_type, &name)?;

        let decorations = self
            .decorations
            .get(&variable.id)
            .cloned()
            .unwrap_or_default();
        let binding = decorations
            .binding
            .ok_or_else(|| ReflectError::MissingBinding(name.clone()))?;

        Ok(Some(ReflectedBinding {
            set: decorations.descriptor_set.unwrap_or(0),
            binding,
            name,
            kind,
            count,
            block_size,
        }))
    }

    fn strip_arrays(&self, mut type_id: u32) -> Result<(u32, DescriptorCount), ReflectError> {
        let mut count = DescriptorCount::Fixed(1);
        loop {
            match self.type_def(type_id)? {
                TypeDef::Array { element, length_id } => {
                    let length = self.array_length(*length_id)?;
                    if let DescriptorCount::Fixed(fixed) = count {
                        count = DescriptorCount::Fixed(fixed * length);
                    }
                    type_id = *element;
                }
                TypeDef::RuntimeArray { element } => {
                    count = DescriptorCount::Unbounded;
                    type_id = *element;
                }
                _ => return Ok((type_id, count)),
            }
        }
    }

    fn array_length(&self, length_id: u32) -> Result<u32, ReflectError> {
        self.constants
            .get(&length_id)
            .copied()
            .ok_or(ReflectError::NonConstantArrayLength(length_id))
    }

    fn classify_resource(
        &self,
        storage_class: u32,
        type_id: u32,
        name: &str,
    ) -> Result<(DescriptorKind, Option<u32>), ReflectError> {
        let unsupported = || ReflectError::UnsupportedVariableType(name.to_string());
        let type_def = self.type_def(type_id)?;

        match storage_class {
            STORAGE_CLASS_UNIFORM_CONSTANT => {
                let kind = match type_def {
                    TypeDef::SampledImage => DescriptorKind::CombinedImageSampler,
                    TypeDef::Sampler => DescriptorKind::Sampler,
                    TypeDef::AccelerationStructure => DescriptorKind::AccelerationStructure,
                    TypeDef::Image { dim, sampled } => {
                        classify_image(*dim, *sampled).ok_or_else(unsupported)?
                    }
                    _ => return Err(unsupported()),
                };
                Ok((kind, None))
            }
            STORAGE_CLASS_UNIFORM | STORAGE_CLASS_STORAGE_BUFFER => {
                let TypeDef::Struct { .. } = type_def else {
                    return Err(unsupported());
                };
                let is_buffer_block = self
                    .decorations
                    .get(&type_id)
                    .map(|decorations| decorations.is_buffer_block)
                    .unwrap_or(false);
                let kind = if storage_class == STORAGE_CLASS_STORAGE_BUFFER || is_buffer_block {
                    DescriptorKind::StorageBuffer
                } else {
                    DescriptorKind::UniformBuffer
                };
                Ok((kind, Some(self.struct_size(type_id)?)))
            }
            _ => Err(unsupported()),
        }
    }

    fn struct_size(&self, struct_id: u32) -> Result<u32, ReflectError> {
        let TypeDef::Struct { members } = self.type_def(struct_id)? else {
            return Err(ReflectError::MissingType(struct_id));
        };

        let mut size = 0;
        for (index, member_type) in members.iter().enumerate() {
            let member = self
                .member_decorations
                .get(&(struct_id, index as u32))
                .cloned()
                .unwrap_or_default();
            let offset = member
                .offset
                .ok_or_else(|| ReflectError::MissingMemberOffset(self.name_of(struct_id)))?;
            size = size.max(offset + self.type_size(*member_type, &member)?);
        }
        Ok(size)
    }

    fn type_size(&self, type_id: u32, member: &MemberDecorations) -> Result<u32, ReflectError> {
        match self.type_def(type_id)? {
            TypeDef::Bool => Ok(4),
            TypeDef::Int { width } | TypeDef::Float { width } => Ok(width / 8),
            TypeDef::Vector { component, count } => Ok(self.type_size(*component, member)? * count),
            TypeDef::Matrix { column, columns } => self.matrix_size(*column, *columns, member),
            TypeDef::Array { element, length_id } => {
                let length = self.array_length(*length_id)?;
                let stride = match self.decorations.get(&type_id).and_then(|d| d.array_stride) {
                    Some(stride) => stride,
                    None => self.type_size(*element, member)?,
                };
                Ok(stride * length)
            }
            TypeDef::RuntimeArray { .. } => Ok(0),
            TypeDef::Struct { .. } => self.struct_size(type_id),
            TypeDef::Image { .. }
            | TypeDef::Sampler
            | TypeDef::SampledImage
            | TypeDef::Pointer { .. }
            | TypeDef::AccelerationStructure => Err(ReflectError::MissingType(type_id)),
        }
    }

    fn matrix_size(
        &self,
        column: u32,
        columns: u32,
        member: &MemberDecorations,
    ) -> Result<u32, ReflectError> {
        let TypeDef::Vector { count: rows, .. } = self.type_def(column)? else {
            return Err(ReflectError::MissingType(column));
        };
        let Some(stride) = member.matrix_stride else {
            return Ok(self.type_size(column, member)? * columns);
        };
        let vector_count = if member.row_major { *rows } else { columns };
        Ok(stride * vector_count)
    }
}

fn decode_type(opcode: u32, operands: &[u32]) -> Result<Option<TypeDef>, ReflectError> {
    let operand = |index: usize| operands.get(index).copied().ok_or(ReflectError::Truncated);
    let type_def = match opcode {
        OP_TYPE_BOOL => TypeDef::Bool,
        OP_TYPE_INT => TypeDef::Int { width: operand(1)? },
        OP_TYPE_FLOAT => TypeDef::Float { width: operand(1)? },
        OP_TYPE_VECTOR => TypeDef::Vector {
            component: operand(1)?,
            count: operand(2)?,
        },
        OP_TYPE_MATRIX => TypeDef::Matrix {
            column: operand(1)?,
            columns: operand(2)?,
        },
        OP_TYPE_IMAGE => TypeDef::Image {
            dim: operand(2)?,
            sampled: operand(6)?,
        },
        OP_TYPE_SAMPLER => TypeDef::Sampler,
        OP_TYPE_SAMPLED_IMAGE => TypeDef::SampledImage,
        OP_TYPE_ARRAY => TypeDef::Array {
            element: operand(1)?,
            length_id: operand(2)?,
        },
        OP_TYPE_RUNTIME_ARRAY => TypeDef::RuntimeArray {
            element: operand(1)?,
        },
        OP_TYPE_STRUCT => TypeDef::Struct {
            members: operands[1..].to_vec(),
        },
        OP_TYPE_POINTER => TypeDef::Pointer {
            pointee: operand(2)?,
        },
        OP_TYPE_ACCELERATION_STRUCTURE_KHR => TypeDef::AccelerationStructure,
        _ => return Ok(None),
    };
    Ok(Some(type_def))
}

fn classify_image(dim: u32, sampled: u32) -> Option<DescriptorKind> {
    match (dim, sampled) {
        (DIM_SUBPASS_DATA, _) => Some(DescriptorKind::InputAttachment),
        (DIM_BUFFER, IMAGE_SAMPLED_WITH_SAMPLER) => Some(DescriptorKind::UniformTexelBuffer),
        (DIM_BUFFER, IMAGE_SAMPLED_STORAGE) => Some(DescriptorKind::StorageTexelBuffer),
        (_, IMAGE_SAMPLED_WITH_SAMPLER) => Some(DescriptorKind::SampledImage),
        (_, IMAGE_SAMPLED_STORAGE) => Some(DescriptorKind::StorageImage),
        _ => None,
    }
}

fn execution_model_to_stage(model: u32) -> Option<ShaderStage> {
    match model {
        0 => Some(ShaderStage::Vertex),
        1 => Some(ShaderStage::TessellationControl),
        2 => Some(ShaderStage::TessellationEvaluation),
        3 => Some(ShaderStage::Geometry),
        4 => Some(ShaderStage::Fragment),
        5 => Some(ShaderStage::Compute),
        5313 => Some(ShaderStage::RayGeneration),
        5314 => Some(ShaderStage::Intersection),
        5315 => Some(ShaderStage::AnyHit),
        5316 => Some(ShaderStage::ClosestHit),
        5317 => Some(ShaderStage::Miss),
        5318 => Some(ShaderStage::Callable),
        5267 | 5364 => Some(ShaderStage::Task),
        5268 | 5365 => Some(ShaderStage::Mesh),
        _ => None,
    }
}

fn decode_literal_string(words: &[u32]) -> String {
    let bytes: Vec<u8> = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .take_while(|byte| *byte != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instruction(opcode: u32, operands: &[u32]) -> Vec<u32> {
        let mut words = vec![((operands.len() as u32 + 1) << 16) | opcode];
        words.extend_from_slice(operands);
        words
    }

    fn literal_string(text: &str) -> Vec<u32> {
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn header() -> Vec<u32> {
        vec![SPIRV_MAGIC, 0x0001_0000, 0, 100, 0]
    }

    fn with_name(id: u32, name: &str) -> Vec<u32> {
        let mut operands = vec![id];
        operands.extend(literal_string(name));
        instruction(OP_NAME, &operands)
    }

    fn fragment_module_with_block_and_sampler() -> Vec<u32> {
        let mut entry_operands = vec![4, 1];
        entry_operands.extend(literal_string("main"));

        let module = [
            header(),
            instruction(OP_ENTRY_POINT, &entry_operands),
            with_name(20, "frame"),
            with_name(30, "texSampler"),
            with_name(31, "shadowMaps"),
            with_name(40, "particles"),
            instruction(OP_DECORATE, &[10, DECORATION_BLOCK]),
            instruction(OP_MEMBER_DECORATE, &[10, 0, DECORATION_OFFSET, 0]),
            instruction(OP_MEMBER_DECORATE, &[10, 0, DECORATION_MATRIX_STRIDE, 16]),
            instruction(OP_MEMBER_DECORATE, &[10, 1, DECORATION_OFFSET, 64]),
            instruction(OP_MEMBER_DECORATE, &[10, 2, DECORATION_OFFSET, 80]),
            instruction(OP_DECORATE, &[8, DECORATION_ARRAY_STRIDE, 16]),
            instruction(OP_DECORATE, &[20, DECORATION_DESCRIPTOR_SET, 1]),
            instruction(OP_DECORATE, &[20, DECORATION_BINDING, 2]),
            instruction(OP_DECORATE, &[30, DECORATION_DESCRIPTOR_SET, 0]),
            instruction(OP_DECORATE, &[30, DECORATION_BINDING, 0]),
            instruction(OP_DECORATE, &[31, DECORATION_DESCRIPTOR_SET, 0]),
            instruction(OP_DECORATE, &[31, DECORATION_BINDING, 1]),
            instruction(OP_DECORATE, &[12, DECORATION_BUFFER_BLOCK]),
            instruction(OP_MEMBER_DECORATE, &[12, 0, DECORATION_OFFSET, 0]),
            instruction(OP_DECORATE, &[40, DECORATION_DESCRIPTOR_SET, 2]),
            instruction(OP_DECORATE, &[40, DECORATION_BINDING, 0]),
            instruction(OP_TYPE_FLOAT, &[1, 32]),
            instruction(OP_TYPE_INT, &[2, 32, 0]),
            instruction(OP_TYPE_VECTOR, &[3, 1, 4]),
            instruction(OP_TYPE_MATRIX, &[4, 3, 4]),
            instruction(OP_CONSTANT, &[2, 5, 3]),
            instruction(OP_TYPE_ARRAY, &[8, 3, 5]),
            instruction(OP_TYPE_STRUCT, &[10, 4, 3, 8]),
            instruction(OP_TYPE_POINTER, &[11, STORAGE_CLASS_UNIFORM, 10]),
            instruction(OP_TYPE_RUNTIME_ARRAY, &[13, 3]),
            instruction(OP_TYPE_STRUCT, &[12, 13]),
            instruction(OP_TYPE_POINTER, &[14, STORAGE_CLASS_UNIFORM, 12]),
            instruction(OP_TYPE_IMAGE, &[6, 1, 1, 0, 0, 0, 1, 0]),
            instruction(OP_TYPE_SAMPLED_IMAGE, &[7, 6]),
            instruction(OP_TYPE_POINTER, &[9, STORAGE_CLASS_UNIFORM_CONSTANT, 7]),
            instruction(OP_CONSTANT, &[2, 15, 4]),
            instruction(OP_TYPE_ARRAY, &[16, 7, 15]),
            instruction(OP_TYPE_POINTER, &[17, STORAGE_CLASS_UNIFORM_CONSTANT, 16]),
            instruction(OP_VARIABLE, &[11, 20, STORAGE_CLASS_UNIFORM]),
            instruction(OP_VARIABLE, &[9, 30, STORAGE_CLASS_UNIFORM_CONSTANT]),
            instruction(OP_VARIABLE, &[17, 31, STORAGE_CLASS_UNIFORM_CONSTANT]),
            instruction(OP_VARIABLE, &[14, 40, STORAGE_CLASS_UNIFORM]),
        ];
        module.concat()
    }

    #[test]
    fn reflects_bindings_kinds_counts_and_block_sizes() {
        let reflection = reflect_shader_words(&fragment_module_with_block_and_sampler()).unwrap();

        assert_eq!(reflection.stages, vec![ShaderStage::Fragment]);
        assert_eq!(
            reflection.bindings,
            vec![
                ReflectedBinding {
                    set: 0,
                    binding: 0,
                    name: "texSampler".into(),
                    kind: DescriptorKind::CombinedImageSampler,
                    count: DescriptorCount::Fixed(1),
                    block_size: None,
                },
                ReflectedBinding {
                    set: 0,
                    binding: 1,
                    name: "shadowMaps".into(),
                    kind: DescriptorKind::CombinedImageSampler,
                    count: DescriptorCount::Fixed(4),
                    block_size: None,
                },
                ReflectedBinding {
                    set: 1,
                    binding: 2,
                    name: "frame".into(),
                    kind: DescriptorKind::UniformBuffer,
                    count: DescriptorCount::Fixed(1),
                    block_size: Some(128),
                },
                ReflectedBinding {
                    set: 2,
                    binding: 0,
                    name: "particles".into(),
                    kind: DescriptorKind::StorageBuffer,
                    count: DescriptorCount::Fixed(1),
                    block_size: Some(0),
                },
            ]
        );
    }

    #[test]
    fn rejects_wrong_magic_and_truncated_streams() {
        let mut words = fragment_module_with_block_and_sampler();
        words[0] = 0xdead_beef;
        assert_eq!(
            reflect_shader_words(&words),
            Err(ReflectError::InvalidMagic)
        );

        let mut truncated = fragment_module_with_block_and_sampler();
        truncated.pop();
        assert_eq!(
            reflect_shader_words(&truncated),
            Err(ReflectError::Truncated)
        );

        assert_eq!(
            reflect_shader_bytes(&[1, 2, 3]),
            Err(ReflectError::Truncated)
        );
    }

    #[test]
    fn requires_binding_decoration_on_descriptor_variables() {
        let mut entry_operands = vec![4, 1];
        entry_operands.extend(literal_string("main"));
        let module = [
            header(),
            instruction(OP_ENTRY_POINT, &entry_operands),
            with_name(30, "orphan"),
            instruction(OP_TYPE_FLOAT, &[1, 32]),
            instruction(OP_TYPE_IMAGE, &[6, 1, 1, 0, 0, 0, 1, 0]),
            instruction(OP_TYPE_SAMPLED_IMAGE, &[7, 6]),
            instruction(OP_TYPE_POINTER, &[9, STORAGE_CLASS_UNIFORM_CONSTANT, 7]),
            instruction(OP_VARIABLE, &[9, 30, STORAGE_CLASS_UNIFORM_CONSTANT]),
        ]
        .concat();

        assert_eq!(
            reflect_shader_words(&module),
            Err(ReflectError::MissingBinding("orphan".into()))
        );
    }
}
