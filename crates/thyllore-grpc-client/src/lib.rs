#[macro_use]
extern crate thyllore_log_core;

#[cfg(feature = "text-to-motion")]
mod grpc_thread;
mod request;
mod response_converter;
#[cfg(feature = "auto-rig")]
mod server_launcher;

#[cfg(feature = "text-to-motion")]
pub use grpc_thread::*;
pub use request::*;
pub use response_converter::*;
#[cfg(feature = "auto-rig")]
pub use server_launcher::*;
