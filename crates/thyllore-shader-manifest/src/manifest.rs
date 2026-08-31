use std::collections::BTreeSet;
use std::path::Path;

use thiserror::Error;
use toml::Value;

use crate::naming::is_shader_source;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageKind {
    Vertex,
    Fragment,
    Geometry,
    Compute,
    RayGeneration,
    Intersection,
    AnyHit,
    ClosestHit,
    Miss,
}

impl StageKind {
    pub fn from_source_file(file_name: &str) -> Option<Self> {
        let (_, extension) = file_name.rsplit_once('.')?;
        match extension {
            "vert" => Some(Self::Vertex),
            "frag" => Some(Self::Fragment),
            "geom" => Some(Self::Geometry),
            "comp" => Some(Self::Compute),
            "rgen" => Some(Self::RayGeneration),
            "rint" => Some(Self::Intersection),
            "rahit" => Some(Self::AnyHit),
            "rchit" => Some(Self::ClosestHit),
            "rmiss" => Some(Self::Miss),
            _ => None,
        }
    }

    pub fn reflect_variant(self) -> &'static str {
        match self {
            Self::Vertex => "Vertex",
            Self::Fragment => "Fragment",
            Self::Geometry => "Geometry",
            Self::Compute => "Compute",
            Self::RayGeneration => "RayGeneration",
            Self::Intersection => "Intersection",
            Self::AnyHit => "AnyHit",
            Self::ClosestHit => "ClosestHit",
            Self::Miss => "Miss",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SetRole {
    Frame,
    Material,
    Object,
    Local,
}

impl SetRole {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "frame" => Some(Self::Frame),
            "material" => Some(Self::Material),
            "object" => Some(Self::Object),
            "local" => Some(Self::Local),
            _ => None,
        }
    }

    pub fn fixed_set_index(self) -> Option<u32> {
        match self {
            Self::Frame => Some(0),
            Self::Material => Some(1),
            Self::Object => Some(2),
            Self::Local => None,
        }
    }

    pub fn variant_name(self) -> &'static str {
        match self {
            Self::Frame => "Frame",
            Self::Material => "Material",
            Self::Object => "Object",
            Self::Local => "Local",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageSource {
    pub source_file: String,
    pub stage: StageKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassDefinition {
    pub name: String,
    pub stages: Vec<StageSource>,
    pub sets: Vec<(u32, SetRole)>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PassManifest {
    pub passes: Vec<PassDefinition>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ManifestError {
    #[error("passes.toml is not valid TOML: {0}")]
    Toml(String),
    #[error("passes.toml has no [pass.<name>] table")]
    NoPasses,
    #[error("passes.toml: `{0}` has an unexpected shape (expected `[pass.<name>]` with `stages` and `sets`)")]
    Shape(String),
    #[error("pass `{0}` is not a valid pass name (use [a-z][a-z0-9_]*)")]
    InvalidPassName(String),
    #[error("pass `{pass}`: `{file}` has no shader extension (.vert/.frag/.geom/.comp/.rgen/.rint/.rahit/.rchit/.rmiss)")]
    UnknownStageExtension { pass: String, file: String },
    #[error("pass `{pass}`: {reason}")]
    StageComposition { pass: String, reason: String },
    #[error("pass `{pass}`: shader source `{file}` does not exist in shaders/")]
    MissingSource { pass: String, file: String },
    #[error("shader `{0}` exists in shaders/ but no pass in passes.toml references it")]
    OrphanShader(String),
    #[error("pass `{pass}`: unknown set role `{role}` (frame / material / object / local)")]
    UnknownSetRole { pass: String, role: String },
    #[error("pass `{pass}`: set key `{key}` is not a set index")]
    InvalidSetIndex { pass: String, key: String },
    #[error("pass `{pass}`: set {set} is declared twice")]
    DuplicateSetIndex { pass: String, set: u32 },
    #[error("pass `{pass}`: role {role:?} must be bound at set {expected}, not {set}")]
    RoleAtWrongSet {
        pass: String,
        role: SetRole,
        set: u32,
        expected: u32,
    },
    #[error("pass `{pass}`: role {role:?} is declared for more than one set")]
    DuplicateRole { pass: String, role: SetRole },
}

impl PassManifest {
    pub fn parse(toml_text: &str) -> Result<Self, ManifestError> {
        let root: Value = toml_text
            .parse()
            .map_err(|error: toml::de::Error| ManifestError::Toml(error.to_string()))?;
        let pass_table = root
            .get("pass")
            .and_then(Value::as_table)
            .ok_or(ManifestError::NoPasses)?;

        let mut passes = Vec::with_capacity(pass_table.len());
        for (name, definition) in pass_table {
            passes.push(parse_pass(name, definition)?);
        }
        if passes.is_empty() {
            return Err(ManifestError::NoPasses);
        }
        Ok(Self { passes })
    }

    pub fn validate_against_sources(&self, shader_dir: &Path) -> Result<(), ManifestError> {
        let mut referenced = BTreeSet::new();
        for pass in &self.passes {
            for stage in &pass.stages {
                if !shader_dir.join(&stage.source_file).is_file() {
                    return Err(ManifestError::MissingSource {
                        pass: pass.name.clone(),
                        file: stage.source_file.clone(),
                    });
                }
                referenced.insert(stage.source_file.clone());
            }
        }

        let mut orphans: Vec<String> = std::fs::read_dir(shader_dir)
            .map_err(|error| {
                ManifestError::Toml(format!("read {}: {error}", shader_dir.display()))
            })?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().is_file())
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|file_name| is_shader_source(file_name))
            .filter(|file_name| !referenced.contains(file_name))
            .collect();
        orphans.sort();
        match orphans.into_iter().next() {
            Some(orphan) => Err(ManifestError::OrphanShader(orphan)),
            None => Ok(()),
        }
    }
}

fn parse_pass(name: &str, definition: &Value) -> Result<PassDefinition, ManifestError> {
    validate_pass_name(name)?;
    let table = definition
        .as_table()
        .ok_or_else(|| ManifestError::Shape(format!("pass.{name}")))?;

    let stage_files = table
        .get("stages")
        .and_then(Value::as_array)
        .ok_or_else(|| ManifestError::Shape(format!("pass.{name}.stages")))?;
    let mut stages = Vec::with_capacity(stage_files.len());
    for file in stage_files {
        let file = file
            .as_str()
            .ok_or_else(|| ManifestError::Shape(format!("pass.{name}.stages")))?;
        let stage = StageKind::from_source_file(file).ok_or_else(|| {
            ManifestError::UnknownStageExtension {
                pass: name.to_string(),
                file: file.to_string(),
            }
        })?;
        stages.push(StageSource {
            source_file: file.to_string(),
            stage,
        });
    }
    validate_stage_composition(name, &stages)?;

    let set_table = table
        .get("sets")
        .and_then(Value::as_table)
        .ok_or_else(|| ManifestError::Shape(format!("pass.{name}.sets")))?;
    let mut sets = Vec::with_capacity(set_table.len());
    for (key, role) in set_table {
        let set = key
            .parse::<u32>()
            .map_err(|_| ManifestError::InvalidSetIndex {
                pass: name.to_string(),
                key: key.clone(),
            })?;
        let role_name = role
            .as_str()
            .ok_or_else(|| ManifestError::Shape(format!("pass.{name}.sets.{key}")))?;
        let role = SetRole::parse(role_name).ok_or_else(|| ManifestError::UnknownSetRole {
            pass: name.to_string(),
            role: role_name.to_string(),
        })?;
        sets.push((set, role));
    }
    sets.sort_by_key(|(set, _)| *set);
    validate_set_roles(name, &sets)?;

    Ok(PassDefinition {
        name: name.to_string(),
        stages,
        sets,
    })
}

fn validate_pass_name(name: &str) -> Result<(), ManifestError> {
    let mut chars = name.chars();
    let starts_lower = chars.next().is_some_and(|c| c.is_ascii_lowercase());
    let rest_valid = chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if starts_lower && rest_valid {
        Ok(())
    } else {
        Err(ManifestError::InvalidPassName(name.to_string()))
    }
}

fn validate_stage_composition(name: &str, stages: &[StageSource]) -> Result<(), ManifestError> {
    let count = |kind: StageKind| stages.iter().filter(|stage| stage.stage == kind).count();
    let (vertex, fragment, geometry, compute) = (
        count(StageKind::Vertex),
        count(StageKind::Fragment),
        count(StageKind::Geometry),
        count(StageKind::Compute),
    );

    let is_graphics = vertex == 1 && fragment == 1 && geometry <= 1 && compute == 0;
    let is_compute = compute == 1 && stages.len() == 1;
    if is_graphics || is_compute {
        return Ok(());
    }

    // RT pipeline: must have at least one RT stage (rgen/rint/rahit/rchit/rmiss) and no graphics/compute stages
    let rt_count = count(StageKind::RayGeneration)
        + count(StageKind::Intersection)
        + count(StageKind::AnyHit)
        + count(StageKind::ClosestHit)
        + count(StageKind::Miss);
    let is_rt = rt_count > 0 && vertex == 0 && fragment == 0 && geometry == 0 && compute == 0;
    if is_rt {
        return Ok(());
    }

    Err(ManifestError::StageComposition {
        pass: name.to_string(),
        reason: format!(
            "stages must be one .vert + one .frag (+ optional .geom) or exactly one .comp, got {} vert / {} frag / {} geom / {} comp",
            vertex, fragment, geometry, compute
        ),
    })
}

fn validate_set_roles(name: &str, sets: &[(u32, SetRole)]) -> Result<(), ManifestError> {
    let mut seen_sets = BTreeSet::new();
    let mut seen_roles = BTreeSet::new();
    for (set, role) in sets {
        if !seen_sets.insert(*set) {
            return Err(ManifestError::DuplicateSetIndex {
                pass: name.to_string(),
                set: *set,
            });
        }
        if !seen_roles.insert(role.variant_name()) {
            return Err(ManifestError::DuplicateRole {
                pass: name.to_string(),
                role: *role,
            });
        }
        if let Some(expected) = role.fixed_set_index() {
            if *set != expected {
                return Err(ManifestError::RoleAtWrongSet {
                    pass: name.to_string(),
                    role: *role,
                    set: *set,
                    expected,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"
[pass.model]
stages = ["vertex.vert", "fragment.frag"]
sets = { 0 = "frame", 1 = "material", 2 = "object" }

[pass.blur]
stages = ["blur.comp"]
sets = { 0 = "local" }
"#;

    #[test]
    fn parses_graphics_and_compute_passes() {
        let manifest = PassManifest::parse(VALID).unwrap();
        assert_eq!(manifest.passes.len(), 2);
        let blur = &manifest.passes[0];
        assert_eq!(blur.name, "blur");
        assert_eq!(blur.stages[0].stage, StageKind::Compute);
        assert_eq!(blur.sets, vec![(0, SetRole::Local)]);
        let model = &manifest.passes[1];
        assert_eq!(
            model.sets,
            vec![
                (0, SetRole::Frame),
                (1, SetRole::Material),
                (2, SetRole::Object)
            ]
        );
    }

    #[test]
    fn rejects_duplicate_pass_names() {
        let text =
            format!("{VALID}\n[pass.model]\nstages = [\"a.vert\", \"b.frag\"]\nsets = {{}}\n");
        assert!(matches!(
            PassManifest::parse(&text),
            Err(ManifestError::Toml(_))
        ));
    }

    #[test]
    fn rejects_bad_stage_composition() {
        let text = "[pass.p]\nstages = [\"a.vert\"]\nsets = {}\n";
        assert!(matches!(
            PassManifest::parse(text),
            Err(ManifestError::StageComposition { .. })
        ));
        let text = "[pass.p]\nstages = [\"a.comp\", \"b.comp\"]\nsets = {}\n";
        assert!(matches!(
            PassManifest::parse(text),
            Err(ManifestError::StageComposition { .. })
        ));
    }

    #[test]
    fn rejects_roles_at_wrong_set() {
        let text = "[pass.p]\nstages = [\"a.vert\", \"b.frag\"]\nsets = { 1 = \"frame\" }\n";
        assert_eq!(
            PassManifest::parse(text),
            Err(ManifestError::RoleAtWrongSet {
                pass: "p".into(),
                role: SetRole::Frame,
                set: 1,
                expected: 0
            })
        );
    }

    #[test]
    fn rejects_unknown_role_and_bad_pass_name() {
        let text = "[pass.p]\nstages = [\"a.vert\", \"b.frag\"]\nsets = { 0 = \"world\" }\n";
        assert!(matches!(
            PassManifest::parse(text),
            Err(ManifestError::UnknownSetRole { .. })
        ));
        let text = "[pass.BadName]\nstages = [\"a.vert\", \"b.frag\"]\nsets = {}\n";
        assert_eq!(
            PassManifest::parse(text),
            Err(ManifestError::InvalidPassName("BadName".into()))
        );
    }

    #[test]
    fn detects_missing_and_orphan_sources() {
        let dir =
            std::env::temp_dir().join(format!("thyllore_shader_manifest_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("vertex.vert"), "").unwrap();
        std::fs::write(dir.join("fragment.frag"), "").unwrap();
        std::fs::write(dir.join("blur.comp"), "").unwrap();
        std::fs::write(dir.join("common.glsl"), "").unwrap();

        let manifest = PassManifest::parse(VALID).unwrap();
        assert_eq!(manifest.validate_against_sources(&dir), Ok(()));

        std::fs::write(dir.join("orphan.frag"), "").unwrap();
        assert_eq!(
            manifest.validate_against_sources(&dir),
            Err(ManifestError::OrphanShader("orphan.frag".into()))
        );

        std::fs::remove_file(dir.join("blur.comp")).unwrap();
        assert!(matches!(
            manifest.validate_against_sources(&dir),
            Err(ManifestError::MissingSource { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
