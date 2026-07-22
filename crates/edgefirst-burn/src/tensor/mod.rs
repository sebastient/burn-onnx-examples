//! Bridges between HAL tensors and Burn tensors.
//!
//! - [`foreign`] describes a platform-native zero-copy buffer handed over by
//!   HAL (`IOSurface` on macOS, `dma-buf` on Linux — `dma-buf` is reserved for
//!   a follow-up phase).
//! - [`import`] turns a [`foreign::ForeignBuffer`] into a Burn tensor with **no
//!   host copy** when the matching backend is enabled (`metal-iosurface` on
//!   macOS Metal). It falls back to a host upload otherwise.
//! - [`export`] pulls a Burn output tensor back to the host and wraps it as a
//!   HAL `TensorDyn` so HAL's decoder/NMS/render pipeline can consume it. This
//!   path is inherently host-side (HAL's decoder runs on ndarray).

pub mod export;
pub mod foreign;

#[cfg(all(feature = "metal-iosurface", target_os = "macos"))]
pub mod import;
