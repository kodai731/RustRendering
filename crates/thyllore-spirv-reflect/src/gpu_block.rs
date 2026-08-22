use crate::types::{ReflectedBlock, ReflectedMember};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMember {
    pub name: &'static str,
    pub offset: usize,
    pub size: usize,
    pub nested: &'static [GpuMember],
}

/// Implemented only through `declare_gpu_block!`, which forces `#[repr(C)]` and
/// plain-data fields so that `as_bytes` is the single sanctioned bytes view.
pub trait GpuBlock: Copy + 'static {
    const NAME: &'static str;
    const MEMBERS: &'static [GpuMember];
    const SIZE: usize = std::mem::size_of::<Self>();

    fn as_bytes(&self) -> &[u8] {
        let pointer = (self as *const Self).cast::<u8>();
        unsafe { std::slice::from_raw_parts(pointer, Self::SIZE) }
    }
}

#[macro_export]
macro_rules! declare_gpu_block {
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$field_attr:meta])*
                $field_vis:vis $field:ident : $field_ty:ty $(= nested $nested:ty)?
            ),* $(,)?
        }
    ) => {
        #[repr(C)]
        $(#[$attr])*
        $vis struct $name {
            $(
                $(#[$field_attr])*
                $field_vis $field: $field_ty,
            )*
        }

        impl $crate::GpuBlock for $name {
            const NAME: &'static str = stringify!($name);
            const MEMBERS: &'static [$crate::GpuMember] = &[
                $(
                    $crate::GpuMember {
                        name: stringify!($field),
                        offset: ::std::mem::offset_of!($name, $field),
                        size: ::std::mem::size_of::<$field_ty>(),
                        nested: $crate::declare_gpu_block!(@nested $($nested)?),
                    },
                )*
            ];
        }
    };
    (@nested) => { &[] };
    (@nested $nested:ty) => { <$nested as $crate::GpuBlock>::MEMBERS };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockCoverage {
    Exact,
    ShaderReadsPrefix,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutDifference {
    SizeDiffers {
        shader: u32,
        rust: usize,
    },
    MissingInRust {
        path: String,
        offset: u32,
    },
    OffsetDiffers {
        path: String,
        shader: u32,
        rust: usize,
    },
    SizeOfMemberDiffers {
        path: String,
        shader: u32,
        rust: usize,
    },
    ExtraInRust {
        path: String,
        offset: usize,
    },
}

impl std::fmt::Display for LayoutDifference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SizeDiffers { shader, rust } => {
                write!(f, "block size     : glsl={shader} rust={rust}")
            }
            Self::MissingInRust { path, offset } => {
                write!(f, "missing in Rust: {path} @ {offset}")
            }
            Self::OffsetDiffers { path, shader, rust } => {
                write!(f, "offset mismatch: {path} glsl={shader} rust={rust}")
            }
            Self::SizeOfMemberDiffers { path, shader, rust } => {
                write!(f, "size mismatch  : {path} glsl={shader} rust={rust}")
            }
            Self::ExtraInRust { path, offset } => {
                write!(f, "extra in Rust  : {path} @ {offset}")
            }
        }
    }
}

pub fn compare_block_layout<T: GpuBlock>(
    shader: &ReflectedBlock,
    coverage: BlockCoverage,
) -> Vec<LayoutDifference> {
    let mut differences = Vec::new();

    let padded_shader_size = (shader.size as usize).div_ceil(16) * 16;
    let size_covers = match coverage {
        BlockCoverage::Exact => T::SIZE >= shader.size as usize && T::SIZE <= padded_shader_size,
        BlockCoverage::ShaderReadsPrefix => T::SIZE >= shader.size as usize,
    };
    if !size_covers {
        differences.push(LayoutDifference::SizeDiffers {
            shader: shader.size,
            rust: T::SIZE,
        });
    }

    compare_members(
        &shader.members,
        T::MEMBERS,
        0,
        "",
        coverage,
        &mut differences,
    );
    differences
}

fn compare_members(
    shader: &[ReflectedMember],
    rust: &[GpuMember],
    rust_base: usize,
    path: &str,
    coverage: BlockCoverage,
    differences: &mut Vec<LayoutDifference>,
) {
    let rust_members: Vec<&GpuMember> = rust
        .iter()
        .filter(|member| !is_padding(member.name))
        .collect();
    let mut matched = vec![false; rust_members.len()];

    for shader_member in shader.iter().filter(|member| !is_padding(&member.name)) {
        let member_path = format!("{path}{}", shader_member.name);
        let wanted = normalize_name(&shader_member.name);
        let Some(index) = rust_members
            .iter()
            .position(|member| normalize_name(member.name) == wanted)
        else {
            differences.push(LayoutDifference::MissingInRust {
                path: member_path,
                offset: shader_member.offset,
            });
            continue;
        };
        matched[index] = true;
        let rust_member = rust_members[index];
        let rust_offset = rust_base + rust_member.offset;

        if rust_offset != shader_member.offset as usize {
            differences.push(LayoutDifference::OffsetDiffers {
                path: member_path.clone(),
                shader: shader_member.offset,
                rust: rust_offset,
            });
        }
        if rust_member.size != shader_member.size as usize {
            differences.push(LayoutDifference::SizeOfMemberDiffers {
                path: member_path.clone(),
                shader: shader_member.size,
                rust: rust_member.size,
            });
        }
        if !shader_member.members.is_empty() || !rust_member.nested.is_empty() {
            compare_members(
                &shader_member.members,
                rust_member.nested,
                rust_offset,
                &format!("{member_path}."),
                BlockCoverage::Exact,
                differences,
            );
        }
    }

    if coverage == BlockCoverage::Exact {
        for (member, _) in rust_members
            .iter()
            .zip(&matched)
            .filter(|(_, was_matched)| !**was_matched)
        {
            differences.push(LayoutDifference::ExtraInRust {
                path: format!("{path}{}", member.name),
                offset: rust_base + member.offset,
            });
        }
    }
}

fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|character| *character != '_')
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

fn is_padding(name: &str) -> bool {
    normalize_name(name).starts_with("pad")
}

#[cfg(test)]
mod tests {
    use super::*;

    declare_gpu_block! {
        #[derive(Clone, Copy, Debug)]
        struct Inner {
            rgb: [f32; 3],
            occlusion_lum_ref: f32,
        }
    }

    declare_gpu_block! {
        #[derive(Clone, Copy, Debug)]
        struct Outer {
            view: [[f32; 4]; 4],
            color_base: Inner = nested Inner,
            modes: [Inner; 2] = nested Inner,
            selected_ids: [u32; 4],
            _padding: [f32; 3],
            scale: f32,
        }
    }

    fn member(
        name: &str,
        offset: u32,
        size: u32,
        members: Vec<ReflectedMember>,
    ) -> ReflectedMember {
        ReflectedMember {
            name: name.into(),
            offset,
            size,
            type_name: String::new(),
            members,
        }
    }

    fn inner_members(base: u32) -> Vec<ReflectedMember> {
        vec![
            member("rgb", base, 12, vec![]),
            member("occlusionLumRef", base + 12, 4, vec![]),
        ]
    }

    fn matching_block() -> ReflectedBlock {
        ReflectedBlock {
            type_name: "Outer".into(),
            size: 144,
            members: vec![
                member("view", 0, 64, vec![]),
                member("colorBase", 64, 16, inner_members(64)),
                member("modes", 80, 32, inner_members(80)),
                member("selectedIDs", 112, 16, vec![]),
                member("pad0", 128, 4, vec![]),
                member("pad1", 132, 4, vec![]),
                member("pad2", 136, 4, vec![]),
                member("scale", 140, 4, vec![]),
            ],
        }
    }

    #[test]
    fn macro_generates_member_table_with_offsets() {
        assert_eq!(Outer::NAME, "Outer");
        assert_eq!(Outer::SIZE, 144);
        let modes = &Outer::MEMBERS[2];
        assert_eq!(modes.offset, 80);
        assert_eq!(modes.size, 32);
        assert_eq!(modes.nested, Inner::MEMBERS);
        assert_eq!(Inner::MEMBERS[1].offset, 12);
    }

    #[test]
    fn matching_layout_has_no_differences() {
        let differences = compare_block_layout::<Outer>(&matching_block(), BlockCoverage::Exact);
        assert_eq!(differences, vec![]);
    }

    #[test]
    fn reports_every_kind_of_drift_by_member_name() {
        let mut block = matching_block();
        block.members[3].name = "selectedIDsRenamed".into();
        block.members[1].members[1].offset = 80;
        block.members[7].size = 8;
        block.size = 160;

        let differences = compare_block_layout::<Outer>(&block, BlockCoverage::Exact);
        assert_eq!(
            differences,
            vec![
                LayoutDifference::SizeDiffers {
                    shader: 160,
                    rust: 144
                },
                LayoutDifference::OffsetDiffers {
                    path: "colorBase.occlusionLumRef".into(),
                    shader: 80,
                    rust: 76
                },
                LayoutDifference::MissingInRust {
                    path: "selectedIDsRenamed".into(),
                    offset: 112
                },
                LayoutDifference::SizeOfMemberDiffers {
                    path: "scale".into(),
                    shader: 8,
                    rust: 4
                },
                LayoutDifference::ExtraInRust {
                    path: "selected_ids".into(),
                    offset: 112
                },
            ]
        );
    }

    #[test]
    fn prefix_coverage_allows_trailing_rust_members() {
        let mut block = matching_block();
        block.members.truncate(4);
        block.size = 128;

        assert_eq!(
            compare_block_layout::<Outer>(&block, BlockCoverage::ShaderReadsPrefix),
            vec![]
        );
        assert_eq!(
            compare_block_layout::<Outer>(&block, BlockCoverage::Exact),
            vec![
                LayoutDifference::SizeDiffers {
                    shader: 128,
                    rust: 144
                },
                LayoutDifference::ExtraInRust {
                    path: "scale".into(),
                    offset: 140
                },
            ]
        );
    }

    #[test]
    fn as_bytes_views_the_whole_struct() {
        let inner = Inner {
            rgb: [1.0, 2.0, 3.0],
            occlusion_lum_ref: 4.0,
        };
        assert_eq!(inner.as_bytes().len(), 16);
        assert_eq!(&inner.as_bytes()[12..], &4.0f32.to_ne_bytes());
    }
}
