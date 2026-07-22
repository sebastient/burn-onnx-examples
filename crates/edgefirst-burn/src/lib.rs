//! Zero-copy integration between [edgefirst-hal](https://crates.io/crates/edgefirst-hal)
//! and the [Burn](https://github.com/tracel-ai/burn) deep-learning framework.
//!
//! This crate bridges HAL's hardware-accelerated image-preprocessing pipeline
//! (convert targets backed by `IOSurface` on macOS, `dma-buf` on Linux) and
//! Burn's compute backends, so that a HAL-converted image can feed a Burn model
//! with **no host-side copy** on the input path.
//!
//! # Design
//!
//! - **Input (HAL image → Burn): GPU zero-copy** when HAL produces a
//!   platform-native buffer (`TensorMemory::Dma` → `IOSurface` on macOS) and a
//!   Metal backend is enabled. The buffer is imported directly into Burn's
//!   device memory and wrapped as a `Tensor`. Otherwise a host fallback uploads
//!   the converted bytes via the standard `Tensor::from_data` path.
//! - **Output (Burn → HAL decoder): host-side.** HAL's YOLO decoder/NMS is an
//!   ndarray host kernel, so the Burn output tensor is pulled to host, wrapped
//!   as a HAL `TensorDyn`, and fed to `edgefirst_decoder::Decoder`.
//!
//! # Features
//!
//! - `metal-iosurface`: enables the zero-copy `IOSurface` → Burn Metal import.
//!   Requires the consumer to also enable `burn/metal` and to carry the
//!   EdgeFirst cubecl + burn forks (see the consuming project's `[patch]`
//!   table). Without this feature the crate always uses the host fallback.
//! - `test`: enables `burn/flex` so unit/integration tests can construct tensors.
//!
//! # Quick start — ONNX codegen path
//!
//! The codegen path compiles an ONNX model into a Rust `Model` struct at build
//! time via `burn-onnx`. At runtime, load the `.bpk` weights and call
//! `forward`:
//!
//! ```no_run
//! # use edgefirst_burn as eb;
//! # use burn::tensor::{Device, Tensor};
//! # use edgefirst_hal as hal;
//! # use edgefirst_burn::htensor::{CpuAccess, DType as HalDType, PixelFormat, TensorMemory};
//! # fn example(device: Device) -> eb::Result<()> {
//! // 1. Load a source image as a HAL TensorDyn.
//! let mut src = hal::tensor::TensorDyn::image(
//!     640, 480, PixelFormat::Rgb, HalDType::U8,
//!     Some(TensorMemory::Mem), CpuAccess::ReadWrite)?;
//! // 2. HAL converts + letterboxes; edgefirst-burn builds the burn input tensor.
//! let mut processor = hal::image::ImageProcessor::new()?;
//! let cfg = eb::PreprocessConfig::yolo(640, 640);
//! eb::with_image_tensor(&mut processor, &mut src, &cfg, &device, |input, letterbox| {
//!     // 3. Run your codegen-generated burn model:
//!     // let out: Tensor<3> = model.forward(input.clone());
//!     // 4. Decode + NMS via HAL's edgefirst-decoder.
//!     // let decoder = eb::ultralytics_decoder(
//!     //     hal::decoder::DecoderVersion::Yolov8, 84, 0.25, 0.7)?;
//!     // let (boxes_, masks) = eb::decode(&decoder, out, &[1, 84, 8400])?;
//!     # let _ = (input, letterbox);
//! })?;
//! # Ok(())
//! # }
//! ```
//!
//! # Quick start — ONNX runtime path
//!
//! The runtime path loads the `.onnx` file directly via
//! `burn_onnx_runtime::GraphExecutor` — no build-time codegen, no `.bpk`:
//!
//! ```text
//! # use edgefirst_burn as eb;
//! # use burn::tensor::{Device, Tensor};
//! # use edgefirst_hal as hal;
//! # use std::collections::HashMap;
//! # use burn_onnx_runtime::{GraphExecutor, Value, FloatTensor};
//! # fn example(device: Device) -> eb::Result<()> {
//! let mut src = hal::tensor::TensorDyn::image(
//!     640, 480, hal::tensor::PixelFormat::Rgb, hal::tensor::DType::U8,
//!     Some(hal::tensor::TensorMemory::Mem), hal::tensor::CpuAccess::ReadWrite)?;
//! let mut processor = hal::image::ImageProcessor::new()?;
//! let cfg = eb::PreprocessConfig::yolo(640, 640);
//! eb::with_image_tensor(&mut processor, &mut src, &cfg, &device, |input, _letterbox| {
//!     // Load the ONNX at runtime and run inference.
//!     let model = GraphExecutor::from_file("model.onnx", &device)
//!         .expect("load onnx");
//!     let input_name = model.input_names().next().unwrap().to_string();
//!     let output_name = model.output_names().next().unwrap().to_string();
//!     let mut inputs = HashMap::new();
//!     inputs.insert(input_name, Value::from(input.clone()));
//!     let outputs = model.forward(inputs).expect("forward");
//!     // Extract the output tensor and decode.
//!     if let Value::Float(FloatTensor::R3(out)) = outputs.get(&output_name).unwrap() {
//!         let decoder = eb::ultralytics_decoder(
//!             hal::decoder::DecoderVersion::Yolov8, 84, 0.25, 0.7).unwrap();
//!         let (boxes_, _) = eb::decode(&decoder, out.clone(), &[1, 84, 8400]).unwrap();
//!         for b in &boxes_ {
//!             println!("class={} score={:.3}", b.label, b.score);
//!         }
//!     }
//! })?;
//! # Ok(())
//! # }
//! ```
//!
//! # IOSurface format constraint (macOS Metal zero-copy)
//!
//! macOS IOSurface only supports a limited set of `(PixelFormat, DType)` pairs.
//! For planar float, only `PlanarRgb`/`F16` is available (ANGLE's
//! `iosurface_client_buffer` extension accepts only RGBA16F float bindings).
//! The zero-copy import path therefore always uses `PlanarRgb/F16` and casts to
//! the caller-requested dtype on-GPU (one cheap kernel — the import itself
//! stays zero-copy). F32 IOSurface is spec-rejected by ANGLE.
//!
//! # Autotune fallback
//!
//! The cubecl fork includes a patch to the autotune cache-hit path: if the
//! autotune-selected kernel fails to launch at runtime (e.g. a CMMA tile
//! availability check that was accepted during tuning but rejected on a
//! subsequent run), the runtime falls back to the next candidate in
//! registration order instead of panicking. This unblocks F16 input on the
//! Metal backend where the conv autotune may select a CMMA kernel that fails
//! on re-run.

pub mod postprocess;
pub mod preprocess;
pub mod render;
pub mod tensor;

pub use postprocess::{decode, ultralytics_decoder};
pub use preprocess::{with_image_tensor, LetterboxMeta, PreprocessConfig};
pub use render::{draw_detections, draw_detections_over};
pub use tensor::{foreign::ForeignBuffer, export::tensordyn_from_burn};

/// Convenience re-exports of the HAL types most callers will need alongside
/// this crate, so the integration can be written against a single namespace.
pub use edgefirst_decoder as decoder;
pub use edgefirst_hal as hal;
pub use edgefirst_image as image;
pub use edgefirst_tensor as htensor;

/// Errors returned by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An `edgefirst-tensor` operation failed.
    #[error(transparent)]
    HalTensor(#[from] edgefirst_tensor::Error),

    /// An `edgefirst-image` operation failed.
    #[error(transparent)]
    HalImage(#[from] edgefirst_image::Error),

    /// An `edgefirst-decoder` operation failed.
    #[error(transparent)]
    HalDecoder(#[from] edgefirst_decoder::DecoderError),

    /// An `edgefirst-codec` operation failed.
    #[error(transparent)]
    HalCodec(#[from] edgefirst_codec::CodecError),

    /// The host fallback path encountered a dtype it cannot convert.
    #[error("unsupported dtype for burn import: {0:?}")]
    UnsupportedDType(edgefirst_tensor::DType),

    /// Zero-copy IOSurface import was requested but is not available in this
    /// build (the `metal-iosurface` feature is off, or the platform is not
    /// macOS). Re-run with `--features edgefirst-burn/metal-iosurface` on macOS,
    /// or use the host fallback.
    #[error("IOSurface zero-copy import is not available in this build")]
    IoSurfaceUnavailable,

    /// Shape/dtype mismatch between a HAL buffer and the requested burn tensor.
    #[error("buffer of {got_bytes} bytes cannot hold a {shape:?} {dtype:?} tensor ({need_bytes} bytes required)")]
    BufferShapeMismatch {
        got_bytes: usize,
        need_bytes: usize,
        shape: Vec<usize>,
        dtype: edgefirst_tensor::DType,
    },
}

/// Shorthand `Result` for this crate.
pub type Result<T> = std::result::Result<T, Error>;
