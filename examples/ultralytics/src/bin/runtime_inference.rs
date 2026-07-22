//! End-to-end Ultralytics YOLO inference — **runtime path**.
//!
//! Loads the `.onnx` model file directly at runtime via
//! `burn-onnx-runtime::GraphExecutor` (no `build.rs`, no codegen, no `.bpk`),
//! preprocesses a real image via HAL, runs the model, and decodes + NMS the
//! output via HAL's `edgefirst-decoder`. Detections are printed mapped back to
//! the original image coordinates.
//!
//! This is the counterpart to `codegen_inference`; the only difference is how
//! the model is loaded. The HAL preprocessing and decoding pipeline is shared.
//!
//! Run (CPU):
//! ```text
//! cargo run --release -p ultralytics --bin runtime_inference -- yolo26n fp32 --input image.jpg
//! ```
//!
//! Run (macOS Metal, zero-copy):
//! ```text
//! cargo run --release -p ultralytics --features metal --bin runtime_inference -- yolo26n fp16 --input image.jpg
//! ```
//!
//! Benchmark mode (`--runs N`, N > 1): warmup + N timed `forward` passes on the
//! same preprocessed input, with latency stats + a decoded sample.

use burn::tensor::Device;
use clap::Parser;
use edgefirst_hal as hal;
use ultralytics::pipeline::{run_pipeline, Args, RuntimeModel};
use ultralytics::{
    YOLO11N_T_B8B_FP16_ONNX, YOLO11N_T_B8B_FP32_ONNX, YOLO26N_T_B8F_FP16_ONNX,
    YOLO26N_T_B8F_FP32_ONNX, YOLOV5N_T_B7B_FP16_ONNX, YOLOV5N_T_B7B_FP32_ONNX,
    YOLOV8N_T_B86_FP16_ONNX, YOLOV8N_T_B86_FP32_ONNX,
};
use ultralytics::pipeline::Arch;
use ultralytics::pipeline::Precision;

fn main() {
    let args = Args::parse();
    let device = Device::default();

    let onnx = match (args.model, args.dtype) {
        (Arch::Yolov5n, Precision::Fp16) => YOLOV5N_T_B7B_FP16_ONNX,
        (Arch::Yolov5n, Precision::Fp32) => YOLOV5N_T_B7B_FP32_ONNX,
        (Arch::Yolov8n, Precision::Fp16) => YOLOV8N_T_B86_FP16_ONNX,
        (Arch::Yolov8n, Precision::Fp32) => YOLOV8N_T_B86_FP32_ONNX,
        (Arch::Yolo11n, Precision::Fp16) => YOLO11N_T_B8B_FP16_ONNX,
        (Arch::Yolo11n, Precision::Fp32) => YOLO11N_T_B8B_FP32_ONNX,
        (Arch::Yolo26n, Precision::Fp16) => YOLO26N_T_B8F_FP16_ONNX,
        (Arch::Yolo26n, Precision::Fp32) => YOLO26N_T_B8F_FP32_ONNX,
    };

    let mut src = match ultralytics::pipeline::load_source_image(&args.input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to load {}: {e}", args.input);
            std::process::exit(1);
        }
    };
    println!(
        "loaded {} ({}x{})",
        args.input,
        src.width().unwrap_or(0),
        src.height().unwrap_or(0)
    );

    let mut processor = match hal::image::ImageProcessor::new() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ImageProcessor::new: {e}");
            std::process::exit(1);
        }
    };

    let model = match RuntimeModel::from_file(onnx, &device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to load onnx: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = run_pipeline(&model, &device, &mut processor, &mut src, &args) {
        eprintln!("inference failed: {e}");
        std::process::exit(1);
    }
}
