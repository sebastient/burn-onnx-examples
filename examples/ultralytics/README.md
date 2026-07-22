# Ultralytics YOLO examples

Runs the full HAL-preprocessed pipeline — load image, preprocess, infer,
decode + NMS, print detections — across 8 Ultralytics YOLO detection models
(`yolov5n`, `yolov8n`, `yolo11n`, `yolo26n` × {fp16, fp32}), on the Burn
**Metal** backend (macOS) or the CPU **flex** backend (cross-platform).

There are **two example binaries** that share the HAL pipeline code in
`src/pipeline.rs` and differ only in how the model is loaded:

| Binary | Model loading | Build-time codegen? |
|---|---|---|
| `codegen_inference` | Loads a build-time-compiled `Model` from its `.bpk` weights | Yes — `build.rs` runs `burn-onnx` codegen |
| `runtime_inference` | Loads the `.onnx` file at runtime via `burn-onnx-runtime::GraphExecutor` | No — no `build.rs`, no `.bpk` |

Both accept the same arguments and backends.

## Models

All in `src/model/` — ONNX opset 13, input `images [1,3,640,640]`, output `[1,84,8400]`.

| File | Architecture | Precision |
|---|---|---|
| `yolov5n-t-b7b.{fp16,fp32}.onnx` | YOLOv5n | F16 / F32 |
| `yolov8n-t-b86.{fp16,fp32}.onnx` | YOLOv8n | F16 / F32 |
| `yolo11n-t-b8b.{fp16,fp32}.onnx` | YOLO11n | F16 / F32 |
| `yolo26n-t-b8f.{fp16,fp32}.onnx` | YOLO26n | F16 / F32 |

## Codegen path (build-time code generation)

`build.rs` generates a Rust `Model` struct + `.bpk` weights per ONNX file (fp16
and fp32 variants are codegenned into separate `OUT_DIR` subdirectories because
the ONNX graph name is shared between the pair). The binary loads weights via
`Model::from_file` (never `Model::new` — see AGENTS.md) and runs the full
end-to-end pipeline on the input image.

```bash
# Single detection run (CPU flex backend — default).
cargo run --release -p ultralytics --bin codegen_inference -- yolo11n fp32 --input image.jpg

# Benchmark mode: warmup + N timed forwards + a decoded sample.
cargo run --release -p ultralytics --bin codegen_inference -- yolo26n fp32 --input image.jpg --runs 100
```

## Runtime path (no codegen)

Loads the `.onnx` directly via `burn_onnx_runtime::GraphExecutor` — no `build.rs`,
no codegen, no `.bpk`. The runtime parses the graph and executes it node-by-node,
calling the same Burn ops the codegen path emits. Same end-to-end pipeline, same
CLI.

```bash
cargo run --release -p ultralytics --bin runtime_inference -- yolo26n fp32 --input image.jpg
```

> The runtime path requires `burn-onnx-runtime` op coverage for the YOLO graphs.
 coverage gaps surface as `UnsupportedOp` errors at the first uncovered node.

## Benchmark mode

Both binaries accept `--runs N` (N > 1) to switch from a single detection run to
a benchmark: `--warmup` discarded runs, then N timed `forward` passes on the same
preprocessed input, with latency stats (min/mean/median/p99) printed alongside a
decoded sample. The input image is preprocessed once and reused across runs.

```bash
cargo run --release -p ultralytics --bin codegen_inference -- \
    yolo11n fp32 --input image.jpg --runs 100 --warmup 5
```

## macOS Metal (zero-copy IOSurface import)

On macOS, enable the `metal` feature to use Burn's Metal backend and
`edgefirst-burn`'s zero-copy `IOSurface` import. HAL's `ImageProcessor::convert`
writes directly to an `IOSurface`-backed `PlanarRgb`/`F16` tensor;
`edgefirst-burn` imports that `IOSurface` into a Burn Metal tensor via
`newBufferWithBytesNoCopy` (no host copy), then casts F16→F32 on-GPU for fp32
models.

```bash
cargo run --release -p ultralytics --features metal --bin codegen_inference -- \
    yolo11n fp16 --input image.jpg
```

The `metal` feature enables `burn/metal` and `edgefirst-burn/metal-iosurface`.
Requires the EdgeFirst cubecl and burn forks (see the `[patch]` tables in the
workspace `Cargo.toml`, and [Required forks](../../ARCHITECTURE.md#required-forks-and-upstream-prs)).

## Parity test

```bash
cargo test -p ultralytics
```

Runs each fp32 model through both the codegen and runtime paths on the same
deterministic ramp input and asserts the outputs match within tolerance
(`ulps=2, epsilon=1e-4`).

### Known issue: yolov8n on the Metal backend

`yolov8n` produces incorrect detections (0 boxes at the default 0.25 score
threshold) when run on the burn **Metal** backend, while `yolo11n` and `yolov5n`
produce correct results. This is **not** a HAL or edgefirst-burn integration
issue — it reproduces identically with the pure-F32 host-fallback path (no
IOSurface, no F16) on Metal. It reproduces in both `codegen_inference` and
`runtime_inference`.

**Diagnosis:**

- The model output's *box coordinates* (the DFL-decoded `cx/cy/w/h` rows) are
  close between CPU and Metal, but the *class scores* diverge structurally: the
  top-confidence anchor is class 0 (person) at score 0.79 on CPU, but class 60
  (dining table) at score 0.21 on Metal.
- The codegen↔runtime parity test passes on Metal (the backend is internally
  consistent) — the divergence is specifically CPU-flex vs Metal.
- yolov8n's ONNX graph is unique among the four models: 57 SiLU activations
  (Sigmoid+Mul pairs), a DFL `Softmax` over a 4D transposed tensor
  (`perm=[0,2,1,3]`), and a `Sigmoid` on the class-logit branch. yolo11n uses
  `MatMul` (attention) and yolov5n is anchor-based; neither hits the failing
  path.
- The profiler confirms the same ONNX fp16 model + HAL decoder produce correct
  detections via ONNX Runtime / CoreML — so the model and decoder are sound; the
  bug is in burn/cubecl's Metal execution of yolov8n's specific op sequence.

**Follow-up:** investigate burn-cubecl's Metal implementation of the ops in
yolov8n's class-score branch (likely the `Softmax` over the transposed DFL
tensor, or a `Slice`/`Concat` axis handling, or conv accumulation in the
57-SiLU backbone). A good starting point is to compare the intermediate tensor
values at each graph node between CPU-flex and Metal for yolov8n specifically.

## Requirements

- The default build uses the CPU `flex` backend (cross-platform, CI-safe).
- For Metal on macOS, add `--features metal`.
- For CUDA on supported hardware, add `--features burn/cuda`.
