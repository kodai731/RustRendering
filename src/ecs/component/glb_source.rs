#[derive(Clone)]
pub enum GlbSource {
    FilePath(String),
    InMemory(Vec<u8>),
}

impl GlbSource {
    pub fn read_bytes(&self) -> anyhow::Result<Vec<u8>> {
        match self {
            GlbSource::FilePath(path) => std::fs::read(path).map_err(|e| e.into()),
            GlbSource::InMemory(data) => Ok(data.clone()),
        }
    }
}
