//! Parity test: codegen `Model` vs `burn-onnx-runtime::GraphExecutor` on each YOLO architecture.
//!
//! For each fp32 architecture, run a fixed deterministic input through both paths and assert the
//! outputs match within a tolerance. Conv/matmul accumulation drift between the codegen
//! (compiled, fused) and runtime (interpreted, per-op) paths is expected, so the tolerance is a
//! combination of relative and absolute thresholds.
//!
//! Run: `cargo test -p ultralytics --test parity`
//!
//! Rationale for the input: zeros would exercise every op but mask numerical divergence; a
//! deterministic ramp exercises every op with distinct values without needing a dataset.

use std::collections::HashMap;

use burn::tensor::{Device, Tensor};
use burn_onnx_runtime::{FloatTensor, GraphExecutor, Value};
use float_cmp::approx_eq;

use ultralytics::model::{
    yolo11n_t_b8b_fp32, yolo26n_t_b8f_fp32, yolov5n_t_b7b_fp32, yolov8n_t_b86_fp32,
};
use ultralytics::{
    YOLO11N_T_B8B_FP32_BPK, YOLO11N_T_B8B_FP32_ONNX, YOLO26N_T_B8F_FP32_BPK,
    YOLO26N_T_B8F_FP32_ONNX, YOLOV5N_T_B7B_FP32_BPK, YOLOV5N_T_B7B_FP32_ONNX,
    YOLOV8N_T_B86_FP32_BPK, YOLOV8N_T_B86_FP32_ONNX,
};

fn to_vec_f32(t: &Tensor<3>) -> Vec<f32> {
    t.clone().into_data().to_vec().expect("tensor to vec")
}

/// Element-wise parity check using float-cmp's approx_eq with both an absolute floor
/// (`epsilon = 1e-4`, accommodates near-zero box coords) and a relative ULP tolerance
/// (`ulps = 2`, accommodates conv/matmul accumulation-order drift between codegen's fused
/// ops and the runtime's per-op dispatch).
fn approx_elementwise(x: f32, y: f32) -> bool {
    approx_eq!(f32, x, y, ulps = 2, epsilon = 1e-4)
}

/// Per-element step for the deterministic ramp input. 1e-4 keeps values in f32's
/// high-precision [0, 1) range; values wrap every 10000 elements.
const RAMP_STEP: f32 = 0.0001;

/// Build a deterministic ramp input [1,3,640,640]. Distinct values per element exercise every op
/// without masking divergence, with no dataset dependency.
fn make_input(device: &Device) -> Tensor<4> {
    let n = 3 * 640 * 640;
    let vals: Vec<f32> = (0..n).map(|i| (i as f32 * RAMP_STEP) % 1.0).collect();
    Tensor::<1>::from_floats(vals.as_slice(), device).reshape([1, 3, 640, 640])
}

fn runtime_forward(model: &GraphExecutor, input: Tensor<4>) -> Tensor<3> {
    let input_name = model
        .input_names()
        .next()
        .expect("graph has an input")
        .to_string();
    let output_name = model
        .output_names()
        .next()
        .expect("graph has an output")
        .to_string();
    let mut inputs = HashMap::new();
    inputs.insert(input_name, Value::from(input));
    let outputs = model.forward(inputs).expect("runtime forward");
    match outputs.get(&output_name).expect("runtime output") {
        Value::Float(FloatTensor::R3(t)) => t.clone(),
        other => panic!("expected rank-3 output, got rank {}", other.rank()),
    }
}

fn assert_parity(codegen_out: &Tensor<3>, runtime_out: &Tensor<3>, model: &str) {
    let a = to_vec_f32(codegen_out);
    let b = to_vec_f32(runtime_out);
    assert_eq!(a.len(), b.len(), "{model}: output length mismatch");
    let mut max_diff = 0.0f32;
    let mut max_rel = 0.0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let abs = (x - y).abs();
        let rel = abs / x.abs().max(1e-6);
        max_diff = max_diff.max(abs);
        max_rel = max_rel.max(rel);
        assert!(
            approx_elementwise(*x, *y),
            "{model}: element {i} diverges: codegen={x} runtime={y} abs={abs} rel={rel}"
        );
    }
    eprintln!(
        "{model}: parity ok ({:.2e} max abs, {:.2e} max rel over {} elems)",
        max_diff,
        max_rel,
        a.len()
    );
}

#[test]
fn parity_yolov5n_fp32() {
    let device = Device::default();
    let codegen = yolov5n_t_b7b_fp32::Model::from_file(YOLOV5N_T_B7B_FP32_BPK, &device);
    let runtime = GraphExecutor::from_file(YOLOV5N_T_B7B_FP32_ONNX, &device).expect("parse");
    let input = make_input(&device);
    let codegen_out = codegen.forward(input.clone());
    let runtime_out = runtime_forward(&runtime, input);
    assert_parity(&codegen_out, &runtime_out, "yolov5n");
}

#[test]
fn parity_yolov8n_fp32() {
    let device = Device::default();
    let codegen = yolov8n_t_b86_fp32::Model::from_file(YOLOV8N_T_B86_FP32_BPK, &device);
    let runtime = GraphExecutor::from_file(YOLOV8N_T_B86_FP32_ONNX, &device).expect("parse");
    let input = make_input(&device);
    let codegen_out = codegen.forward(input.clone());
    let runtime_out = runtime_forward(&runtime, input);
    assert_parity(&codegen_out, &runtime_out, "yolov8n");
}

#[test]
fn parity_yolo11n_fp32() {
    let device = Device::default();
    let codegen = yolo11n_t_b8b_fp32::Model::from_file(YOLO11N_T_B8B_FP32_BPK, &device);
    let runtime = GraphExecutor::from_file(YOLO11N_T_B8B_FP32_ONNX, &device).expect("parse");
    let input = make_input(&device);
    let codegen_out = codegen.forward(input.clone());
    let runtime_out = runtime_forward(&runtime, input);
    assert_parity(&codegen_out, &runtime_out, "yolo11n");
}

#[test]
fn parity_yolo26n_fp32() {
    let device = Device::default();
    let codegen = yolo26n_t_b8f_fp32::Model::from_file(YOLO26N_T_B8F_FP32_BPK, &device);
    let runtime = GraphExecutor::from_file(YOLO26N_T_B8F_FP32_ONNX, &device).expect("parse");
    let input = make_input(&device);
    let codegen_out = codegen.forward(input.clone());
    let runtime_out = runtime_forward(&runtime, input);
    assert_parity(&codegen_out, &runtime_out, "yolo26n");
}
