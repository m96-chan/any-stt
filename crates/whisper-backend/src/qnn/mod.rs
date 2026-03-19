//! QNN (Qualcomm AI Engine Direct) backend for Hexagon NPU acceleration.
//!
//! This module provides runtime-detected QNN support via dlopen.
//! It compiles on all platforms but only activates when QNN libraries
//! are present at runtime (e.g., on Snapdragon devices with QNN SDK installed).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     dlopen      ┌──────────────┐
//! │  Rust code   │ ──────────────► │ libQnnHtp.so │
//! │  (qnn mod)   │                 │  (QNN SDK)   │
//! └──────┬───────┘                 └──────┬───────┘
//!        │                                │ FastRPC
//!        │  QnnGraph_execute()            │
//!        │                         ┌──────▼───────┐
//!        │                         │  Hexagon HTP │
//!        │                         │  (NPU chip)  │
//!        │                         └──────────────┘
//! ```

pub mod context;
pub mod loader;
pub mod types;

pub use context::QnnContext;
pub use loader::{is_qnn_available, QnnLibrary};
