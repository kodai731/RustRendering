use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

pub struct FlameDumpSink {
    pub path: PathBuf,
    pub writer: BufWriter<File>,
}

impl FlameDumpSink {
    pub fn new(path: PathBuf) -> Self {
        let file = File::create(&path).expect("failed to create flame dump file");
        let writer = BufWriter::new(file);
        Self { path, writer }
    }
}
