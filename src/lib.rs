#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]

extern crate core;

#[macro_use]
extern crate thyllore_log_core;

pub mod logger;

pub mod animation;
pub mod app;
pub mod asset;
#[cfg(debug_assertions)]
pub mod debugview;
pub mod ecs;
pub mod exporter;
#[cfg(feature = "text-to-motion")]
pub mod grpc;
pub mod loader;
pub mod math;
#[cfg(feature = "ml")]
pub mod ml;
pub mod orchestrator;
pub mod paths;
pub mod platform;
pub mod render;
pub mod scene;
pub mod vulkanr;
