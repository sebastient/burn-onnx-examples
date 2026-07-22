//! Ultralytics YOLO examples: run 8 YOLO detection models (`yolov5n`, `yolov8n`,
//! `yolo11n`, `yolo26n` × {fp16, fp32}) end-to-end through the HAL preprocessing
//! pipeline, on either the codegen path (build-time-compiled `.bpk`) or the
//! runtime path (`burn-onnx-runtime::GraphExecutor` loads the `.onnx` directly).
//!
//! The two example binaries share the HAL pipeline code in [`pipeline`]; the
//! only difference is how the model is loaded. See:
//!
//! - `src/bin/codegen_inference.rs`
//! - `src/bin/runtime_inference.rs`
//!
//! A codegen↔runtime parity test lives in `tests/parity.rs`.

pub mod model;
pub mod pipeline;

pub use model::{
    YOLO11N_T_B8B_FP16_BPK, YOLO11N_T_B8B_FP16_ONNX, YOLO11N_T_B8B_FP32_BPK,
    YOLO11N_T_B8B_FP32_ONNX, YOLO26N_T_B8F_FP16_BPK, YOLO26N_T_B8F_FP16_ONNX,
    YOLO26N_T_B8F_FP32_BPK, YOLO26N_T_B8F_FP32_ONNX, YOLOV5N_T_B7B_FP16_BPK,
    YOLOV5N_T_B7B_FP16_ONNX, YOLOV5N_T_B7B_FP32_BPK, YOLOV5N_T_B7B_FP32_ONNX,
    YOLOV8N_T_B86_FP16_BPK, YOLOV8N_T_B86_FP16_ONNX, YOLOV8N_T_B86_FP32_BPK,
    YOLOV8N_T_B86_FP32_ONNX,
};
