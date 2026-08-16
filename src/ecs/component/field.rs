use thyllore_effect_core::FieldManifest;

/// Attribute: this entity's look is field-driven; the manifest is attached by `field_manifest_sync`.
#[derive(Clone, Debug, Default)]
pub struct FieldAffected {
    pub manifest: FieldManifest,
}
