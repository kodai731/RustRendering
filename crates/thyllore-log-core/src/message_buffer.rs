use chrono::Local;
use once_cell::sync::Lazy;
use std::collections::VecDeque;
use std::sync::Mutex;

const MAX_MESSAGES: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MessageLevel {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug)]
pub struct Message {
    pub level: MessageLevel,
    pub text: String,
    pub timestamp: String,
}

pub struct MessageBuffer {
    messages: VecDeque<Message>,
}

pub static MESSAGE_BUFFER: Lazy<Mutex<MessageBuffer>> =
    Lazy::new(|| Mutex::new(MessageBuffer::new()));

impl MessageBuffer {
    pub fn new() -> Self {
        Self {
            messages: VecDeque::with_capacity(MAX_MESSAGES),
        }
    }

    pub fn push(&mut self, level: MessageLevel, text: String) {
        if self.messages.len() >= MAX_MESSAGES {
            self.messages.pop_front();
        }
        let timestamp = Local::now().format("%H:%M:%S").to_string();
        self.messages.push_back(Message {
            level,
            text,
            timestamp,
        });
    }

    pub fn snapshot(&self) -> Vec<Message> {
        self.messages.iter().cloned().collect()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub fn count_by_level(&self, level: MessageLevel) -> usize {
        self.messages.iter().filter(|m| m.level == level).count()
    }
}

pub fn push_message(level: MessageLevel, text: String) {
    MESSAGE_BUFFER
        .lock()
        .expect("Failed to lock message buffer")
        .push(level, text);
}

#[macro_export]
macro_rules! msg_info {
    ($($arg:tt)*) => {{
        let text = format!($($arg)*);
        $crate::log!("[INFO] {}", &text);
        $crate::message_buffer::push_message(
            $crate::message_buffer::MessageLevel::Info,
            text,
        );
    }};
}

#[macro_export]
macro_rules! msg_warn {
    ($($arg:tt)*) => {{
        let text = format!($($arg)*);
        $crate::log_warn!("{}", &text);
        $crate::message_buffer::push_message(
            $crate::message_buffer::MessageLevel::Warning,
            text,
        );
    }};
}

#[macro_export]
macro_rules! msg_error {
    ($($arg:tt)*) => {{
        let text = format!($($arg)*);
        $crate::log_error!("{}", &text);
        $crate::message_buffer::push_message(
            $crate::message_buffer::MessageLevel::Error,
            text,
        );
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_snapshot() {
        let mut buf = MessageBuffer::new();
        buf.push(MessageLevel::Info, "hello".to_string());
        buf.push(MessageLevel::Warning, "warn msg".to_string());

        let snap = buf.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].level, MessageLevel::Info);
        assert_eq!(snap[0].text, "hello");
        assert_eq!(snap[1].level, MessageLevel::Warning);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut buf = MessageBuffer::new();
        for i in 0..MAX_MESSAGES + 10 {
            buf.push(MessageLevel::Info, format!("msg {}", i));
        }
        let snap = buf.snapshot();
        assert_eq!(snap.len(), MAX_MESSAGES);
        assert_eq!(snap[0].text, "msg 10");
        assert_eq!(
            snap[MAX_MESSAGES - 1].text,
            format!("msg {}", MAX_MESSAGES + 9)
        );
    }

    #[test]
    fn test_clear() {
        let mut buf = MessageBuffer::new();
        buf.push(MessageLevel::Error, "err".to_string());
        buf.clear();
        assert_eq!(buf.snapshot().len(), 0);
    }

    #[test]
    fn test_count_by_level() {
        let mut buf = MessageBuffer::new();
        buf.push(MessageLevel::Info, "a".to_string());
        buf.push(MessageLevel::Info, "b".to_string());
        buf.push(MessageLevel::Warning, "c".to_string());
        buf.push(MessageLevel::Error, "d".to_string());

        assert_eq!(buf.count_by_level(MessageLevel::Info), 2);
        assert_eq!(buf.count_by_level(MessageLevel::Warning), 1);
        assert_eq!(buf.count_by_level(MessageLevel::Error), 1);
    }

    #[test]
    fn test_timestamp_not_empty() {
        let mut buf = MessageBuffer::new();
        buf.push(MessageLevel::Info, "test".to_string());
        let snap = buf.snapshot();
        assert!(!snap[0].timestamp.is_empty());
    }
}
