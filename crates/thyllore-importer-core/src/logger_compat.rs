#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {{
        let _ = format_args!($($arg)*);
    }};
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {{
        let _ = format_args!($($arg)*);
    }};
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {{
        let _ = format_args!($($arg)*);
    }};
}
