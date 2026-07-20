use chrono::Local;
use once_cell::sync::Lazy;
use std::fs::{File, OpenOptions};
use std::io::{Result, Write};
use std::path::Path;
use std::sync::Mutex;

pub static LOGGER: Lazy<Mutex<Option<Logger>>> = Lazy::new(|| {
    let logger = Logger::new("log/log");
    Mutex::new(logger.ok())
});

pub struct Logger {
    file: File,
    base_path: String,
    current_index: usize,
    max_file_size: u64,
}

impl Logger {
    pub fn new(base_path: &str) -> Result<Self> {
        let max_file_size = 10 * 1024 * 1024;

        let path = Path::new(base_path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        Self::delete_old_logs(base_path)?;

        let current_index = 0;
        let file_path = format!("{}_{}.txt", base_path, current_index);

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&file_path)?;

        Ok(Logger {
            file,
            base_path: base_path.to_string(),
            current_index,
            max_file_size,
        })
    }

    fn delete_old_logs(base_path: &str) -> Result<()> {
        let path = Path::new(base_path);
        let parent_dir = path.parent().unwrap_or(Path::new("."));
        let base_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("log");

        if let Ok(entries) = std::fs::read_dir(parent_dir) {
            for entry in entries.flatten() {
                if let Ok(file_name) = entry.file_name().into_string() {
                    if file_name.starts_with(base_name) && file_name.ends_with(".txt") {
                        let without_base = &file_name[base_name.len()..];
                        if without_base.starts_with('_') && without_base.ends_with(".txt") {
                            let _ = std::fs::remove_file(entry.path());
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn rotate_if_needed(&mut self) -> Result<()> {
        let metadata = self.file.metadata()?;
        let current_size = metadata.len();

        if current_size >= self.max_file_size {
            self.current_index += 1;
            let new_path = format!("{}_{}.txt", self.base_path, self.current_index);

            self.file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&new_path)?;

            let time = Local::now();
            writeln!(
                self.file,
                "[{}] === Log file rotated from {}_{}.txt ===",
                time.format("%Y-%m-%d %H:%M:%S"),
                self.base_path,
                self.current_index - 1
            )?;
        }

        Ok(())
    }

    pub fn log(&mut self, args: std::fmt::Arguments) -> Result<()> {
        self.rotate_if_needed()?;

        let time = Local::now();
        writeln!(self.file, "[{}] {}", time.format("%Y-%m-%d %H:%M:%S"), args)?;

        self.file.flush()
    }

    pub fn log_with_prefix(&mut self, prefix: &str, args: std::fmt::Arguments) -> Result<()> {
        self.rotate_if_needed()?;

        let time = Local::now();
        writeln!(
            self.file,
            "[{}] {} {}",
            time.format("%Y-%m-%d %H:%M:%S"),
            prefix,
            args
        )?;

        self.file.flush()
    }
}

/// Info level: compiled out in release builds
#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {{
        #[cfg(debug_assertions)]
        {
            if let Ok(mut logger) = $crate::logger::LOGGER.lock() {
                if let Some(ref mut l) = *logger {
                    if l.log(format_args!($($arg)*)).is_err() {
                        eprintln!("[LOG WRITE ERROR] {}", format_args!($($arg)*));
                    }
                }
            } else {
                eprintln!("[LOG LOCK ERROR] {}", format_args!($($arg)*));
            }
        }
    }};
}

/// Warning level: always active including release builds
#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {{
        if let Ok(mut logger) = $crate::logger::LOGGER.lock() {
            if let Some(ref mut l) = *logger {
                if l.log_with_prefix("[WARN]", format_args!($($arg)*)).is_err() {
                    eprintln!("[LOG WRITE ERROR] [WARN] {}", format_args!($($arg)*));
                }
            } else {
                eprintln!("[WARN] {}", format_args!($($arg)*));
            }
        } else {
            eprintln!("[LOG LOCK ERROR] [WARN] {}", format_args!($($arg)*));
        }
    }};
}

/// Error level: always active including release builds
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {{
        if let Ok(mut logger) = $crate::logger::LOGGER.lock() {
            if let Some(ref mut l) = *logger {
                if l.log_with_prefix("[ERROR]", format_args!($($arg)*)).is_err() {
                    eprintln!("[LOG WRITE ERROR] [ERROR] {}", format_args!($($arg)*));
                }
            } else {
                eprintln!("[ERROR] {}", format_args!($($arg)*));
            }
        } else {
            eprintln!("[LOG LOCK ERROR] [ERROR] {}", format_args!($($arg)*));
        }
    }};
}
