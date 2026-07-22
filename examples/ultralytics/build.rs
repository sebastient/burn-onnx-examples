use burn_onnx::ModelGen;

fn main() {
    // Generate Rust source + .bpk weights for each of the 8 Ultralytics models.
    //
    // fp16 and fp32 variants share an architecture (same ONNX graph name), so ModelGen produces
    // identical output filenames for the pair. To avoid them overwriting each other we use a
    // separate out_dir per precision: `model_fp16/` and `model_fp32/`.
    for arch in [
        "yolov5n-t-b7b",
        "yolov8n-t-b86",
        "yolo11n-t-b8b",
        "yolo26n-t-b8f",
    ] {
        ModelGen::new()
            .input(&format!("src/model/{arch}.fp16.onnx"))
            .out_dir("model_fp16/")
            .run_from_script();
        ModelGen::new()
            .input(&format!("src/model/{arch}.fp32.onnx"))
            .out_dir("model_fp32/")
            .run_from_script();
    }
}
