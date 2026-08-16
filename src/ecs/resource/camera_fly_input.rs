#[derive(Clone, Debug, Default)]
pub struct CameraFlyInput {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    pub boost: bool,
    pub delta_seconds: f32,
}
