use std::path::PathBuf;

#[derive(Debug)]
pub enum SceneError {
    Io(std::io::Error),
    Parse(ron::error::SpannedError),
    Serialize(ron::Error),
    Export(anyhow::Error),
    ModelNotFound(PathBuf),
    AnimationNotFound(PathBuf),
    VersionMismatch { expected: u32, found: u32 },
}

impl std::fmt::Display for SceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SceneError::Io(e) => write!(f, "IO error: {}", e),
            SceneError::Parse(e) => write!(f, "Parse error: {}", e),
            SceneError::Serialize(e) => write!(f, "Serialize error: {}", e),
            SceneError::Export(e) => write!(f, "Export error: {:#}", e),
            SceneError::ModelNotFound(p) => write!(f, "Model not found: {}", p.display()),
            SceneError::AnimationNotFound(p) => {
                write!(f, "Animation not found: {}", p.display())
            }
            SceneError::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "Version mismatch: expected {}, found {}",
                    expected, found
                )
            }
        }
    }
}

impl std::error::Error for SceneError {}

impl From<std::io::Error> for SceneError {
    fn from(e: std::io::Error) -> Self {
        SceneError::Io(e)
    }
}

impl From<ron::error::SpannedError> for SceneError {
    fn from(e: ron::error::SpannedError) -> Self {
        SceneError::Parse(e)
    }
}

impl From<ron::Error> for SceneError {
    fn from(e: ron::Error) -> Self {
        SceneError::Serialize(e)
    }
}

pub type SceneResult<T> = Result<T, SceneError>;
