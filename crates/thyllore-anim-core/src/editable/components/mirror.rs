use crate::BoneId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MirrorAxis {
    X,
    Y,
    Z,
}

#[derive(Clone, Debug)]
pub struct MirrorMapping {
    pub pairs: Vec<(BoneId, BoneId)>,
    pub symmetry_axis: MirrorAxis,
}
