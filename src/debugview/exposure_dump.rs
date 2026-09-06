use crate::app::App;
use crate::ecs::resource::{Exposure, ExposureDumpSink};

impl App {
    pub(crate) fn record_exposure_dump(&self, frame: u64, adapted: Option<f32>) {
        let Some(sink) = self.data.ecs_world.get_resource::<ExposureDumpSink>() else {
            return;
        };
        let exposure_value = self
            .data
            .ecs_world
            .get_resource::<Exposure>()
            .map(|e| e.exposure_value)
            .unwrap_or(1.0);

        let adapted_text = match adapted {
            Some(value) => value.to_string(),
            None => "0.0".to_string(),
        };
        let line = format!(
            "{{\"frame\":{},\"adapted\":{},\"exposure_value\":{},\"ae_enabled\":{}}}\n",
            frame,
            adapted_text,
            exposure_value,
            adapted.is_some()
        );
        append_jsonl(&sink.path, &line);
    }
}

fn append_jsonl(path: &str, line: &str) {
    use std::fs::OpenOptions;
    use std::io::Write;
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(line.as_bytes());
    }
}
