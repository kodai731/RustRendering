#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropertyKind {
    TranslationX,
    TranslationY,
    TranslationZ,
    RotationX,
    RotationY,
    RotationZ,
    ScaleX,
    ScaleY,
    ScaleZ,
}

pub fn property_kind_to_id(kind: PropertyKind) -> u32 {
    match kind {
        PropertyKind::TranslationX => 0,
        PropertyKind::TranslationY => 1,
        PropertyKind::TranslationZ => 2,
        PropertyKind::RotationX => 3,
        PropertyKind::RotationY => 4,
        PropertyKind::RotationZ => 5,
        PropertyKind::ScaleX => 6,
        PropertyKind::ScaleY => 7,
        PropertyKind::ScaleZ => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_mapping_is_stable() {
        assert_eq!(property_kind_to_id(PropertyKind::TranslationX), 0);
        assert_eq!(property_kind_to_id(PropertyKind::ScaleZ), 8);
    }

    #[test]
    fn all_variants_have_distinct_ids() {
        let kinds = [
            PropertyKind::TranslationX,
            PropertyKind::TranslationY,
            PropertyKind::TranslationZ,
            PropertyKind::RotationX,
            PropertyKind::RotationY,
            PropertyKind::RotationZ,
            PropertyKind::ScaleX,
            PropertyKind::ScaleY,
            PropertyKind::ScaleZ,
        ];
        let ids: Vec<u32> = kinds.iter().map(|&k| property_kind_to_id(k)).collect();
        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(unique.len(), kinds.len());
    }
}
