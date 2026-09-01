//! End-to-end Ultralytics YOLO inference — **codegen path**.
//!
//! Loads a build-time-compiled `Model` from its `.bpk` weights (per AGENTS.md,
//! never use `Model::new()` to run inference — constants would be zeroed),
//! preprocesses a real image via HAL, runs the model, and decodes + NMS the
//! output via HAL's `edgefirst-decoder`. Detections are printed mapped back to
//! the original image coordinates.
//!
//! On the CPU `flex` backend (the default) this exercises the host fallback
//! import path. With `--features metal` on macOS it uses the zero-copy
//! `IOSurface` import.
//!
//! Run (CPU):
//! ```text
//! cargo run --release -p ultralytics --bin codegen_inference -- yolo11n fp32 --input image.jpg
//! ```
//!
//! Run (macOS Metal, zero-copy):
//! ```text
//! cargo run --release -p ultralytics --features metal --bin codegen_inference -- yolo11n fp16 --input image.jpg
//! ```
//!
//! Benchmark mode (`--runs N`, N > 1): warmup + N timed `forward` passes on the
//! same preprocessed input, with latency stats + a decoded sample.
//!
//! ```text
//! cargo run --release -p ultralytics --bin codegen_inference -- yolo11n fp32 --input image.jpg --runs 100
//! ```

use burn::tensor::Device;
use clap::Parser;
use edgefirst_hal as hal;
use ultralytics::model::{
    yolo11n_t_b8b_fp16, yolo11n_t_b8b_fp32, yolo26n_t_b8f_fp16, yolo26n_t_b8f_fp32,
    yolov5n_t_b7b_fp16, yolov5n_t_b7b_fp32, yolov8n_t_b86_fp16, yolov8n_t_b86_fp32,
};
use ultralytics::pipeline::{Arch, Args, CodegenModel, Precision, run_pipeline};
use ultralytics::{
    YOLO11N_T_B8B_FP16_BPK, YOLO11N_T_B8B_FP32_BPK, YOLO26N_T_B8F_FP16_BPK, YOLO26N_T_B8F_FP32_BPK,
    YOLOV5N_T_B7B_FP16_BPK, YOLOV5N_T_B7B_FP32_BPK, YOLOV8N_T_B86_FP16_BPK, YOLOV8N_T_B86_FP32_BPK,
};

fn main() {
    let args = Args::parse();
    let device = Device::default();

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

    // Construct the codegen model for the (arch, precision) pair and run the
    // shared pipeline. The `run` helper is monomorphized over the generated
    // `Model` type via `CodegenModel`.
    macro_rules! dispatch {
        ($module:ident, $bpk:expr) => {
            run::<$module::Model>($bpk, &device, &mut processor, &mut src, &args)
        };
    }
    let result = match (args.model, args.dtype) {
        (Arch::Yolov5n, Precision::Fp16) => dispatch!(yolov5n_t_b7b_fp16, YOLOV5N_T_B7B_FP16_BPK),
        (Arch::Yolov5n, Precision::Fp32) => dispatch!(yolov5n_t_b7b_fp32, YOLOV5N_T_B7B_FP32_BPK),
        (Arch::Yolov8n, Precision::Fp16) => dispatch!(yolov8n_t_b86_fp16, YOLOV8N_T_B86_FP16_BPK),
        (Arch::Yolov8n, Precision::Fp32) => dispatch!(yolov8n_t_b86_fp32, YOLOV8N_T_B86_FP32_BPK),
        (Arch::Yolo11n, Precision::Fp16) => dispatch!(yolo11n_t_b8b_fp16, YOLO11N_T_B8B_FP16_BPK),
        (Arch::Yolo11n, Precision::Fp32) => dispatch!(yolo11n_t_b8b_fp32, YOLO11N_T_B8B_FP32_BPK),
        (Arch::Yolo26n, Precision::Fp16) => dispatch!(yolo26n_t_b8f_fp16, YOLO26N_T_B8F_FP16_BPK),
        (Arch::Yolo26n, Precision::Fp32) => dispatch!(yolo26n_t_b8f_fp32, YOLO26N_T_B8F_FP32_BPK),
    };
    if let Err(e) = result {
        eprintln!("inference failed: {e}");
        std::process::exit(1);
    }
}

/// Load the codegen model and run the shared pipeline. Exists only so the
/// generated `Model` type can flow through `CodegenModel` into
/// [`ultralytics::pipeline::run_pipeline`].
fn run<M: CodegenModel>(
    bpk: &str,
    device: &Device,
    processor: &mut hal::image::ImageProcessor,
    src: &mut edgefirst_tensor::TensorDyn,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = M::from_file(bpk, device);
    run_pipeline(&model, device, processor, src, args)
}
