//! Generated YOLO model modules and their build-time paths.
//!
//! Each `pub mod` below wraps a codegen-generated `Model` struct (`include!`'d from
//! `OUT_DIR`). The fp16 and fp32 variants of an architecture share the same ONNX graph
//! name, so `build.rs` writes them into separate `OUT_DIR` subdirectories
//! (`model_fp16/`, `model_fp32/`) to avoid filename collisions. The `*_BPK` consts
//! point at the build-generated `.bpk` weight files; the `*_ONNX` consts point at the
//! committed `.onnx` source files (consumed by the runtime binary and parity test).

// ─── fp16 variants ────────────────────────────────────────────────────────────

/// Codegen-generated fp16 YOLO11n model (`yolo11n-t-b8b`).
pub mod yolo11n_t_b8b_fp16 {
    include!(concat!(env!("OUT_DIR"), "/model_fp16/yolo11n-t-b8b.fp16.rs"));
}
/// Codegen-generated fp16 YOLO26n model (`yolo26n-t-b8f`).
pub mod yolo26n_t_b8f_fp16 {
    include!(concat!(env!("OUT_DIR"), "/model_fp16/yolo26n-t-b8f.fp16.rs"));
}
/// Codegen-generated fp16 YOLOv5n model (`yolov5n-t-b7b`).
pub mod yolov5n_t_b7b_fp16 {
    include!(concat!(env!("OUT_DIR"), "/model_fp16/yolov5n-t-b7b.fp16.rs"));
}
/// Codegen-generated fp16 YOLOv8n model (`yolov8n-t-b86`).
pub mod yolov8n_t_b86_fp16 {
    include!(concat!(env!("OUT_DIR"), "/model_fp16/yolov8n-t-b86.fp16.rs"));
}

// ─── fp32 variants ────────────────────────────────────────────────────────────

/// Codegen-generated fp32 YOLO11n model (`yolo11n-t-b8b`).
pub mod yolo11n_t_b8b_fp32 {
    include!(concat!(env!("OUT_DIR"), "/model_fp32/yolo11n-t-b8b.fp32.rs"));
}
/// Codegen-generated fp32 YOLO26n model (`yolo26n-t-b8f`).
pub mod yolo26n_t_b8f_fp32 {
    include!(concat!(env!("OUT_DIR"), "/model_fp32/yolo26n-t-b8f.fp32.rs"));
}
/// Codegen-generated fp32 YOLOv5n model (`yolov5n-t-b7b`).
pub mod yolov5n_t_b7b_fp32 {
    include!(concat!(env!("OUT_DIR"), "/model_fp32/yolov5n-t-b7b.fp32.rs"));
}
/// Codegen-generated fp32 YOLOv8n model (`yolov8n-t-b86`).
pub mod yolov8n_t_b86_fp32 {
    include!(concat!(env!("OUT_DIR"), "/model_fp32/yolov8n-t-b86.fp32.rs"));
}

// ─── .bpk weight paths (build-generated, in OUT_DIR) ─────────────────────────
// Consumed by `Model::from_file` per AGENTS.md (never `Model::new` for inference).

/// Absolute path to the build-generated `yolo11n-t-b8b.fp16.bpk` weight file.
pub const YOLO11N_T_B8B_FP16_BPK: &str =
    concat!(env!("OUT_DIR"), "/model_fp16/yolo11n-t-b8b.fp16.bpk");
/// Absolute path to the build-generated `yolo11n-t-b8b.fp32.bpk` weight file.
pub const YOLO11N_T_B8B_FP32_BPK: &str =
    concat!(env!("OUT_DIR"), "/model_fp32/yolo11n-t-b8b.fp32.bpk");
/// Absolute path to the build-generated `yolo26n-t-b8f.fp16.bpk` weight file.
pub const YOLO26N_T_B8F_FP16_BPK: &str =
    concat!(env!("OUT_DIR"), "/model_fp16/yolo26n-t-b8f.fp16.bpk");
/// Absolute path to the build-generated `yolo26n-t-b8f.fp32.bpk` weight file.
pub const YOLO26N_T_B8F_FP32_BPK: &str =
    concat!(env!("OUT_DIR"), "/model_fp32/yolo26n-t-b8f.fp32.bpk");
/// Absolute path to the build-generated `yolov5n-t-b7b.fp16.bpk` weight file.
pub const YOLOV5N_T_B7B_FP16_BPK: &str =
    concat!(env!("OUT_DIR"), "/model_fp16/yolov5n-t-b7b.fp16.bpk");
/// Absolute path to the build-generated `yolov5n-t-b7b.fp32.bpk` weight file.
pub const YOLOV5N_T_B7B_FP32_BPK: &str =
    concat!(env!("OUT_DIR"), "/model_fp32/yolov5n-t-b7b.fp32.bpk");
/// Absolute path to the build-generated `yolov8n-t-b86.fp16.bpk` weight file.
pub const YOLOV8N_T_B86_FP16_BPK: &str =
    concat!(env!("OUT_DIR"), "/model_fp16/yolov8n-t-b86.fp16.bpk");
/// Absolute path to the build-generated `yolov8n-t-b86.fp32.bpk` weight file.
pub const YOLOV8N_T_B86_FP32_BPK: &str =
    concat!(env!("OUT_DIR"), "/model_fp32/yolov8n-t-b86.fp32.bpk");

// ─── .onnx source paths (committed, in this crate's src/model) ───────────────
// Consumed by the runtime binary (`GraphExecutor::from_file`) and the parity test.

/// Absolute path to the committed `yolo11n-t-b8b.fp16.onnx` source model.
pub const YOLO11N_T_B8B_FP16_ONNX: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/yolo11n-t-b8b.fp16.onnx");
/// Absolute path to the committed `yolo11n-t-b8b.fp32.onnx` source model.
pub const YOLO11N_T_B8B_FP32_ONNX: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/yolo11n-t-b8b.fp32.onnx");
/// Absolute path to the committed `yolo26n-t-b8f.fp16.onnx` source model.
pub const YOLO26N_T_B8F_FP16_ONNX: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/yolo26n-t-b8f.fp16.onnx");
/// Absolute path to the committed `yolo26n-t-b8f.fp32.onnx` source model.
pub const YOLO26N_T_B8F_FP32_ONNX: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/yolo26n-t-b8f.fp32.onnx");
/// Absolute path to the committed `yolov5n-t-b7b.fp16.onnx` source model.
pub const YOLOV5N_T_B7B_FP16_ONNX: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/yolov5n-t-b7b.fp16.onnx");
/// Absolute path to the committed `yolov5n-t-b7b.fp32.onnx` source model.
pub const YOLOV5N_T_B7B_FP32_ONNX: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/yolov5n-t-b7b.fp32.onnx");
/// Absolute path to the committed `yolov8n-t-b86.fp16.onnx` source model.
pub const YOLOV8N_T_B86_FP16_ONNX: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/yolov8n-t-b86.fp16.onnx");
/// Absolute path to the committed `yolov8n-t-b86.fp32.onnx` source model.
pub const YOLOV8N_T_B86_FP32_ONNX: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/yolov8n-t-b86.fp32.onnx");
