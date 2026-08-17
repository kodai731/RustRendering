#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlslDescriptorClass {
    Opaque,
    UniformBlock,
    StorageBlock,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlslArrayCount {
    Fixed(u32),
    Unbounded,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlslDescriptorDeclaration {
    pub set: u32,
    pub binding: u32,
    pub name: String,
    pub class: GlslDescriptorClass,
    pub count: GlslArrayCount,
}

const STORAGE_MODIFIERS: [&str; 9] = [
    "readonly",
    "writeonly",
    "coherent",
    "volatile",
    "restrict",
    "precise",
    "highp",
    "mediump",
    "lowp",
];

pub fn scan_glsl_descriptor_declarations(preprocessed: &str) -> Vec<GlslDescriptorDeclaration> {
    let mut declarations = Vec::new();
    let mut rest = preprocessed;

    while let Some(start) = find_layout_keyword(rest) {
        let after_keyword = &rest[start + "layout".len()..];
        let Some((qualifiers, after_qualifiers)) = split_parenthesized(after_keyword) else {
            rest = after_keyword;
            continue;
        };
        if let Some(declaration) = parse_declaration(qualifiers, after_qualifiers) {
            declarations.push(declaration);
        }
        rest = after_qualifiers;
    }

    declarations
}

fn find_layout_keyword(text: &str) -> Option<usize> {
    let mut offset = 0;
    while let Some(found) = text[offset..].find("layout") {
        let start = offset + found;
        let end = start + "layout".len();
        let preceded_by_identifier = text[..start]
            .chars()
            .next_back()
            .is_some_and(is_identifier_char);
        let followed_by_identifier = text[end..].chars().next().is_some_and(is_identifier_char);
        if !preceded_by_identifier && !followed_by_identifier {
            return Some(start);
        }
        offset = end;
    }
    None
}

fn split_parenthesized(text: &str) -> Option<(&str, &str)> {
    let trimmed = text.trim_start();
    let body = trimmed.strip_prefix('(')?;
    let close = body.find(')')?;
    Some((&body[..close], &body[close + 1..]))
}

fn parse_declaration(qualifiers: &str, statement: &str) -> Option<GlslDescriptorDeclaration> {
    let binding = qualifier_value(qualifiers, "binding")?;
    let set = qualifier_value(qualifiers, "set").unwrap_or(0);

    let statement_end = statement.find(';')?;
    let mut words = statement[..statement_end.min(statement.len())]
        .split_whitespace()
        .skip_while(|word| STORAGE_MODIFIERS.contains(word));
    let storage = words.next()?;
    let block_class = match storage {
        "uniform" => GlslDescriptorClass::UniformBlock,
        "buffer" => GlslDescriptorClass::StorageBlock,
        _ => return None,
    };

    let type_or_block_name = words.next()?;
    let opens_block = statement[..statement_end]
        .find('{')
        .is_some_and(|brace| brace < statement_end);
    if opens_block {
        let close = find_matching_brace(statement)?;
        let (name, count) = parse_declarator(&statement[close + 1..]);
        let name = if name.is_empty() {
            type_or_block_name.to_string()
        } else {
            name
        };
        return Some(GlslDescriptorDeclaration {
            set,
            binding,
            name,
            class: block_class,
            count,
        });
    }

    let declarator: String = words.collect::<Vec<_>>().join(" ");
    let (name, count) = parse_declarator(&declarator);
    Some(GlslDescriptorDeclaration {
        set,
        binding,
        name,
        class: GlslDescriptorClass::Opaque,
        count,
    })
}

fn qualifier_value(qualifiers: &str, key: &str) -> Option<u32> {
    qualifiers.split(',').find_map(|qualifier| {
        let (name, value) = qualifier.split_once('=')?;
        if name.trim() != key {
            return None;
        }
        value.trim().parse().ok()
    })
}

fn find_matching_brace(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (index, character) in text.char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_declarator(text: &str) -> (String, GlslArrayCount) {
    let declarator = text.split(';').next().unwrap_or("").trim();
    let Some((name, array)) = declarator.split_once('[') else {
        return (declarator.to_string(), GlslArrayCount::Fixed(1));
    };
    let length = array.split(']').next().unwrap_or("").trim();
    let count = if length.is_empty() {
        GlslArrayCount::Unbounded
    } else {
        length
            .parse()
            .map(GlslArrayCount::Fixed)
            .unwrap_or(GlslArrayCount::Unknown)
    };
    (name.trim().to_string(), count)
}

fn is_identifier_char(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scans_blocks_opaque_types_and_arrays() {
        let source = r#"
            #version 450
            layout(location = 0) in vec2 fragTexCoord;
            layout(std140, set = 1, binding = 0) uniform FlameUBO { mat4 model; } flame;
            layout(set = 1, binding = 4) uniform sampler2D flameHistorySampler;
            layout(binding = 2, rgba32f) uniform image2D histogramImage;
            layout(std430, binding = 1) buffer HistogramBuffer { uint bins[]; };
            layout(binding = 3) uniform sampler2D shadowMaps[4];
            layout(binding = 5) uniform texture2D bindless[];
            layout(push_constant) uniform Push { float t; } push;
            const int layoutish = 1;
        "#;

        let declarations = scan_glsl_descriptor_declarations(source);

        assert_eq!(
            declarations,
            vec![
                GlslDescriptorDeclaration {
                    set: 1,
                    binding: 0,
                    name: "flame".into(),
                    class: GlslDescriptorClass::UniformBlock,
                    count: GlslArrayCount::Fixed(1),
                },
                GlslDescriptorDeclaration {
                    set: 1,
                    binding: 4,
                    name: "flameHistorySampler".into(),
                    class: GlslDescriptorClass::Opaque,
                    count: GlslArrayCount::Fixed(1),
                },
                GlslDescriptorDeclaration {
                    set: 0,
                    binding: 2,
                    name: "histogramImage".into(),
                    class: GlslDescriptorClass::Opaque,
                    count: GlslArrayCount::Fixed(1),
                },
                GlslDescriptorDeclaration {
                    set: 0,
                    binding: 1,
                    name: "HistogramBuffer".into(),
                    class: GlslDescriptorClass::StorageBlock,
                    count: GlslArrayCount::Fixed(1),
                },
                GlslDescriptorDeclaration {
                    set: 0,
                    binding: 3,
                    name: "shadowMaps".into(),
                    class: GlslDescriptorClass::Opaque,
                    count: GlslArrayCount::Fixed(4),
                },
                GlslDescriptorDeclaration {
                    set: 0,
                    binding: 5,
                    name: "bindless".into(),
                    class: GlslDescriptorClass::Opaque,
                    count: GlslArrayCount::Unbounded,
                },
            ]
        );
    }

    #[test]
    fn ignores_layouts_without_binding() {
        let source = "layout(location = 0) out vec4 outColor; layout(local_size_x = 16) in;";
        assert!(scan_glsl_descriptor_declarations(source).is_empty());
    }
}
