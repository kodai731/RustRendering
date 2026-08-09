//! Declared map of every stochastic field source in the flame pipeline and the
//! look quantities it drives. Derived from the same values the UBO packer reads,
//! and pinned to the GLSL by the shader-audit test below (undeclared noise fails).

use crate::flame::FlameEffect;
use crate::flame_wave::read_env_wave_jitter;

/// One independent random table / noise evaluated anywhere in the flame pipeline.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum FieldSourceKind {
    /// 96-mode log-uniform broadband table (fringe-free alone; the unification target).
    ErosionWaveTable,
    /// 16-mode displacement-form warp table.
    WarpDisplacementTable,
    /// 64-mode detail table behind the contour wiggle.
    ContourWiggleTable,
    /// Two lattice fbm3 fields displacing the per-column envelope support.
    BoundaryFbm,
    /// Rank-3 shared phase-jitter fields.
    PhaseJitterFields,
}

impl FieldSourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ErosionWaveTable => "erosion-wave-table",
            Self::WarpDisplacementTable => "warp-displacement-table",
            Self::ContourWiggleTable => "contour-wiggle-table",
            Self::BoundaryFbm => "boundary-fbm",
            Self::PhaseJitterFields => "phase-jitter-fields",
        }
    }

    /// Scheduled for removal by the unified-field redesign; must not gain new consumers.
    pub fn is_unification_pending(self) -> bool {
        !matches!(self, Self::ErosionWaveTable | Self::WarpDisplacementTable)
    }

    pub fn all() -> [FieldSourceKind; 5] {
        [
            Self::ErosionWaveTable,
            Self::WarpDisplacementTable,
            Self::ContourWiggleTable,
            Self::BoundaryFbm,
            Self::PhaseJitterFields,
        ]
    }
}

/// A look quantity a stochastic source drives.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum FieldTargetKind {
    InteriorErosion,
    SampleCoordinates,
    SilhouetteRadius,
    SilhouetteHeight,
    CarrierPhase,
}

impl FieldTargetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InteriorErosion => "interior-erosion",
            Self::SampleCoordinates => "sample-coordinates",
            Self::SilhouetteRadius => "silhouette-radius",
            Self::SilhouetteHeight => "silhouette-height",
            Self::CarrierPhase => "carrier-phase",
        }
    }
}

/// One declared edge: source drives target, gated by lever.
#[derive(Clone, Debug)]
pub struct FieldInfluence {
    pub source: FieldSourceKind,
    pub target: FieldTargetKind,
    pub lever: &'static str,
    pub active: bool,
}

/// The declared composition for one flame.
#[derive(Clone, Debug, Default)]
pub struct FieldManifest {
    pub influences: Vec<FieldInfluence>,
}

impl FieldManifest {
    pub fn active(&self) -> impl Iterator<Item = &FieldInfluence> {
        self.influences.iter().filter(|i| i.active)
    }

    /// Distinct active sources, sorted.
    pub fn active_sources(&self) -> Vec<FieldSourceKind> {
        let mut sources: Vec<FieldSourceKind> = self.active().map(|i| i.source).collect();
        sources.sort();
        sources.dedup();
        sources
    }

    /// Active sources marked for removal by the unification.
    pub fn active_unification_pending(&self) -> Vec<FieldSourceKind> {
        self.active_sources()
            .into_iter()
            .filter(|s| s.is_unification_pending())
            .collect()
    }

    /// Stable one-line description for logs, dumps and change detection.
    pub fn summary(&self) -> String {
        let parts: Vec<String> = self
            .active()
            .map(|i| format!("{}->{}", i.source.as_str(), i.target.as_str()))
            .collect();
        if parts.is_empty() {
            "none".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Derives the manifest from the same values the UBO packer reads (jitter and
/// the unified switch via the shared levers). Under the unified field the old
/// boundary/wiggle/jitter sources are inert; their levers become spectral-tilt
/// edges of the one broadband table.
pub fn flame_field_manifest(effect: &FlameEffect) -> FieldManifest {
    flame_field_manifest_with(
        effect,
        crate::flame::read_env_wave_unified(),
        read_env_wave_jitter(),
    )
}

/// Pure derivation for a given unified switch and jitter lever value.
pub fn flame_field_manifest_with(
    effect: &FlameEffect,
    unified: bool,
    jitter_scale: f32,
) -> FieldManifest {
    let erosion = effect.noise_amplitude != 0.0;
    let jitter = erosion && !unified && jitter_scale > 0.0;
    FieldManifest {
        influences: vec![
            FieldInfluence {
                source: FieldSourceKind::ErosionWaveTable,
                target: FieldTargetKind::InteriorErosion,
                lever: "noise_amplitude",
                active: erosion,
            },
            FieldInfluence {
                source: FieldSourceKind::ErosionWaveTable,
                target: FieldTargetKind::SilhouetteHeight,
                lever: "boundary_amp (low-octave tilt)",
                active: unified && erosion && effect.boundary_amp != 0.0,
            },
            FieldInfluence {
                source: FieldSourceKind::ErosionWaveTable,
                target: FieldTargetKind::SilhouetteRadius,
                lever: "contour_wiggle_amp (mid-octave tilt)",
                active: unified && erosion && effect.contour_wiggle_amp != 0.0,
            },
            FieldInfluence {
                source: FieldSourceKind::WarpDisplacementTable,
                target: FieldTargetKind::SampleCoordinates,
                lever: "warp_amp",
                active: effect.warp_amp != 0.0,
            },
            FieldInfluence {
                source: FieldSourceKind::ContourWiggleTable,
                target: FieldTargetKind::SilhouetteRadius,
                lever: "contour_wiggle_amp",
                active: !unified && effect.contour_wiggle_amp != 0.0,
            },
            FieldInfluence {
                source: FieldSourceKind::BoundaryFbm,
                target: FieldTargetKind::SilhouetteHeight,
                lever: "boundary_amp",
                active: !unified && effect.boundary_amp != 0.0,
            },
            FieldInfluence {
                source: FieldSourceKind::BoundaryFbm,
                target: FieldTargetKind::SilhouetteRadius,
                lever: "boundary_amp",
                active: !unified && effect.boundary_amp != 0.0,
            },
            FieldInfluence {
                source: FieldSourceKind::PhaseJitterFields,
                target: FieldTargetKind::CarrierPhase,
                lever: "THYLLORE_FLAME_WAVE_JITTER",
                active: jitter,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn effect() -> FlameEffect {
        FlameEffect::default()
    }

    #[test]
    fn legacy_manifest_follows_the_levers() {
        let mut e = effect();
        e.noise_amplitude = 1.5;
        e.warp_amp = 1.4;
        e.contour_wiggle_amp = 0.3;
        e.boundary_amp = 0.2;
        let sources = flame_field_manifest_with(&e, false, 1.0).active_sources();
        assert!(sources.contains(&FieldSourceKind::ErosionWaveTable));
        assert!(sources.contains(&FieldSourceKind::WarpDisplacementTable));
        assert!(sources.contains(&FieldSourceKind::ContourWiggleTable));
        assert!(sources.contains(&FieldSourceKind::BoundaryFbm));
        assert!(sources.contains(&FieldSourceKind::PhaseJitterFields));

        e.boundary_amp = 0.0;
        e.contour_wiggle_amp = 0.0;
        let sources = flame_field_manifest_with(&e, false, 0.0).active_sources();
        assert!(!sources.contains(&FieldSourceKind::BoundaryFbm));
        assert!(!sources.contains(&FieldSourceKind::ContourWiggleTable));
        assert!(!sources.contains(&FieldSourceKind::PhaseJitterFields));
    }

    #[test]
    fn unified_manifest_has_no_pending_sources_and_absorbs_the_levers() {
        let mut e = effect();
        e.noise_amplitude = 1.5;
        e.warp_amp = 1.4;
        e.contour_wiggle_amp = 0.3;
        e.boundary_amp = 0.2;
        let m = flame_field_manifest_with(&e, true, 1.0);
        assert_eq!(
            m.active_sources(),
            vec![
                FieldSourceKind::ErosionWaveTable,
                FieldSourceKind::WarpDisplacementTable
            ]
        );
        assert!(m.active_unification_pending().is_empty());
        let s = m.summary();
        assert!(s.contains("erosion-wave-table->silhouette-height"));
        assert!(s.contains("erosion-wave-table->silhouette-radius"));
    }

    // Shader audit: every GLSL noise-primitive site must sit inside a declared anchor function.

    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn shader_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../shaders/include")
    }

    fn strip_line_comments(source: &str) -> String {
        source
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// (name, body) per top-level GLSL function; handles multi-line signatures and struct returns.
    fn parse_functions(source: &str) -> Vec<(String, String)> {
        const NON_TYPES: [&str; 9] = [
            "if", "for", "while", "return", "else", "switch", "const", "struct", "layout",
        ];
        let mut functions = Vec::new();
        let mut depth: i32 = 0;
        let mut current: Option<(String, String, bool)> = None;
        for line in source.lines() {
            if depth == 0 && current.is_none() {
                let trimmed = line.trim_start();
                let mut words = trimmed.split_whitespace();
                if let (Some(ty), Some(rest)) = (words.next(), words.next()) {
                    let is_type = ty.chars().all(|c| c.is_alphanumeric() || c == '_')
                        && !NON_TYPES.contains(&ty);
                    if is_type {
                        if let Some(paren) = rest.find('(') {
                            let name = &rest[..paren];
                            if paren > 0
                                && name.chars().all(|c| c.is_alphanumeric() || c == '_')
                            {
                                current = Some((name.to_string(), String::new(), false));
                            }
                        }
                    }
                }
            }
            if let Some((_, body, opened)) = current.as_mut() {
                body.push_str(line);
                body.push('\n');
                if line.contains('{') {
                    *opened = true;
                }
                if !*opened && line.contains(';') {
                    current = None;
                }
            }
            depth += line.matches('{').count() as i32;
            depth -= line.matches('}').count() as i32;
            if depth == 0 {
                if let Some((name, body, opened)) = current.take() {
                    if opened {
                        functions.push((name, body));
                    } else {
                        current = Some((name, body, opened));
                    }
                }
            }
        }
        functions
    }

    #[test]
    fn every_stochastic_evaluation_site_belongs_to_a_declared_source() {
        // primitive pattern -> allowed functions; extending this map is a design change.
        let mut allowed: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        allowed.insert(
            "fbm3(",
            vec![
                "fbm3",
                "flameBoundaryDisplacement",
            ],
        );
        allowed.insert(
            "waveModes[",
            vec![
                "flameWaveModeSum",
                "flameWaveCfPsiVectors",
                "flameWaveCfLoadCheb",
                "flameWaveWarpOffset",
                "flameWarpMapZ",
                "flameWarpMapJvp",
                "flameDetailNoise",
            ],
        );
        allowed.insert(
            "waveJitter[",
            vec!["flameWaveJitterKappaScale", "flameWaveModeSum"],
        );

        let dir = shader_dir();
        let mut audited_files = 0;
        let mut violations: Vec<String> = Vec::new();
        let mut found: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for entry in std::fs::read_dir(&dir).expect("shader include dir") {
            let path = entry.expect("dir entry").path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if !name.starts_with("flame") || !name.ends_with(".glsl") {
                continue;
            }
            audited_files += 1;
            let source = strip_line_comments(&std::fs::read_to_string(&path).unwrap());
            let functions = parse_functions(&source);
            for (pattern, funcs) in &allowed {
                for (fn_name, body) in &functions {
                    if body.contains(pattern) {
                        found.entry(pattern).or_default().push(fn_name.clone());
                        if !funcs.contains(&fn_name.as_str()) {
                            violations.push(format!("{name}: {pattern} inside {fn_name}"));
                        }
                    }
                }
            }
        }
        assert!(audited_files > 3, "shader dir not found or empty: {dir:?}");
        assert!(
            violations.is_empty(),
            "undeclared stochastic evaluation sites (declare them in \
             flame_field_manifest.rs or remove the noise):\n{}",
            violations.join("\n")
        );
        // Coverage guard: an empty audit would pass vacuously.
        for (pattern, expect_in) in [
            ("waveModes[", "flameWaveModeSum"),
            ("waveJitter[", "flameWaveModeSum"),
            ("fbm3(", "flameBoundaryDisplacement"),
        ] {
            assert!(
                found
                    .get(pattern)
                    .is_some_and(|fns| fns.iter().any(|f| f == expect_in)),
                "audit parser lost coverage: {pattern} not seen inside {expect_in} \
                 (found in: {:?})",
                found.get(pattern)
            );
        }
    }

    #[test]
    fn declared_anchor_functions_still_exist_in_the_shaders() {
        // Inverse direction: a removed anchor must shrink the declaration too.
        let dir = shader_dir();
        let mut all_source = String::new();
        for entry in std::fs::read_dir(&dir).expect("shader include dir") {
            let path = entry.expect("dir entry").path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if name.starts_with("flame") && name.ends_with(".glsl") {
                all_source.push_str(&std::fs::read_to_string(&path).unwrap());
            }
        }
        for anchor in [
            "flameWaveModeSum",
            "flameBoundaryDisplacement",
            "flameDetailNoise",
            "flameWarpMapZ",
            "flameWaveJitterKappaScale",
        ] {
            assert!(
                all_source.contains(anchor),
                "declared anchor {anchor} no longer exists — update flame_field_manifest.rs"
            );
        }
    }
}
