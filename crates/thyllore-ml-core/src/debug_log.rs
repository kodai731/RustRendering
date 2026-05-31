//! File-based debug logging for the Blender addon, compiled only under the
//! `debug-log` feature. Production wheels build without this feature, so the
//! whole module — and the PyO3 functions that wrap it — are omitted from the
//! binary. The Python `_debuglog.py` caller discovers their absence and no-ops.

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

fn log_file() -> &'static Mutex<Option<std::fs::File>> {
    static LOG_FILE: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();
    LOG_FILE.get_or_init(|| Mutex::new(None))
}

/// Open (create/append) the log file. Subsequent calls replace the target.
pub fn init(path: &str) -> std::io::Result<()> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    *log_file().lock().expect("debug log mutex") = Some(file);
    Ok(())
}

/// Append one timestamped line. No-op until `init` has opened a file.
pub fn log_line(message: &str) {
    let mut guard = log_file().lock().expect("debug log mutex");
    if let Some(file) = guard.as_mut() {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(file, "{timestamp} INFO {message}");
        let _ = file.flush();
    }
}
