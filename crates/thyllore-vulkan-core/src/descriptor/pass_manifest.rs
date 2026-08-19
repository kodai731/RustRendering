use thyllore_spirv_reflect::ShaderStage;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SetRole {
    Frame,
    Material,
    Object,
    Local,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderFile {
    pub path: &'static str,
    pub stage: ShaderStage,
}

#[derive(Debug, PartialEq, Eq)]
pub struct PassShaders {
    pub id: PassId,
    pub stages: &'static [ShaderFile],
    pub set_roles: &'static [(u32, SetRole)],
}

impl PassShaders {
    pub fn name(&self) -> &'static str {
        self.id.name()
    }

    pub fn set_index(&self, role: SetRole) -> Option<u32> {
        self.set_roles
            .iter()
            .find(|(_, candidate)| *candidate == role)
            .map(|(set, _)| *set)
    }

    pub fn stage(&self, stage: ShaderStage) -> Option<&ShaderFile> {
        self.stages.iter().find(|file| file.stage == stage)
    }

    pub fn is_compute(&self) -> bool {
        self.stage(ShaderStage::Compute).is_some()
    }
}

pub fn passes_with_role(role: SetRole) -> Vec<&'static PassShaders> {
    ALL_PASSES
        .iter()
        .copied()
        .filter(|pass| pass.set_index(role).is_some())
        .collect()
}

include!(concat!(env!("OUT_DIR"), "/pass_manifest.rs"));
