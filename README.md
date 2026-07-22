# edgefirst-burn

Zero-copy integration between [edgefirst-hal](https://crates.io/crates/edgefirst-hal)
and the [Burn](https://github.com/tracel-ai/burn) deep-learning framework.

`edgefirst-burn` bridges HAL's hardware-accelerated image-preprocessing pipeline
(convert targets backed by `IOSurface` on macOS, `dma-buf` on Linux) and Burn's
compute backends, so a HAL-converted image can feed a Burn model with **no
host-side copy** on the input path. It targets [Ultralytics](https://www.ultralytics.com/)
YOLO detection models end-to-end — preprocessing, inference, and decode + NMS.

> **Status — forked dependencies.** The zero-copy `IOSurface` import on macOS
> Metal currently requires EdgeFirst forks of `burn`, `cubecl`, and `burn-onnx`,
> carried as local path patches until the upstream PRs merge. The host fallback
> path (no `metal` feature) works with upstream crates alone. See
> [Required forks](ARCHITECTURE.md#required-forks-and-upstream-prs) in
> `ARCHITECTURE.md`.

## Repository layout

This is a Cargo workspace.

| Member | Description |
|---|---|
| [`crates/edgefirst-burn`](crates/edgefirst-burn/) | The published integration crate — the HAL↔Burn bridge. |
| [`examples/ultralytics`](examples/ultralytics/) | End-to-end YOLO examples (codegen + runtime paths) over 8 model variants. |

The crate-level API documentation lives at
[`crates/edgefirst-burn/README.md`](crates/edgefirst-burn/README.md); the
high-level design lives at [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Prerequisites

- **Rust toolchain** (edition 2021 for the crate, 2024 for the example).
- For the zero-copy Metal path: **macOS** with a Metal-capable GPU.
- The three sibling checkouts, on the branches below, as **siblings of this
  repo** so the workspace `[patch]` tables resolve to local paths:

  | Repo | Checked out at | Branch |
  |---|---|---|
  | [tracel-ai/burn](https://github.com/tracel-ai/burn) | `../burn` | `ef/iosurface-import` |
  | [cubecl](https://github.com/tracel-ai/cubecl) (via `sebastient/cubecl`) | `../cubecl` | `ef/iosurface-import` |
  | [tracel-ai/burn-onnx](https://github.com/tracel-ai/burn-onnx) | `../burn-onnx` | `feat/burn-onnx-runtime` |

  These branches carry the changes described in
  [Required forks](ARCHITECTURE.md#required-forks-and-upstream-prs). The branch
  names are read from the workspace `Cargo.toml` `[patch]` tables.

## Getting started

```bash
# Build the integration crate (CPU flex backend — cross-platform, CI-safe).
cargo build -p edgefirst-burn

# Run the crate's unit + integration tests.
cargo test -p edgefirst-burn --features test
```

## Using the ultralytics examples

The `examples/ultralytics` workspace member runs the full HAL pipeline — load
image, preprocess, infer, decode + NMS, print detections — across 8 YOLO model
variants. It provides **two binaries that share the same pipeline code** and
differ only in how the model is loaded:

| Binary | Model loading | Build-time codegen? |
|---|---|---|
| `codegen_inference` | Loads a build-time-compiled `Model` from its `.bpk` weights | Yes — `build.rs` runs `burn-onnx` codegen |
| `runtime_inference` | Loads the `.onnx` file at runtime via `burn-onnx-runtime::GraphExecutor` | No — no `build.rs`, no `.bpk` |

Both accept the same arguments and support the same backends.

### Models

All 8 variants ship as ONNX (opset 13, input `images [1,3,640,640]`, output
`[1,84,8400]`) in `examples/ultralytics/src/model/`:

| Architecture | fp16 file | fp32 file |
|---|---|---|
| YOLOv5n | `yolov5n-t-b7b.fp16.onnx` | `yolov5n-t-b7b.fp32.onnx` |
| YOLOv8n | `yolov8n-t-b86.fp16.onnx` | `yolov8n-t-b86.fp32.onnx` |
| YOLO11n | `yolo11n-t-b8b.fp16.onnx` | `yolo11n-t-b8b.fp32.onnx` |
| YOLO26n | `yolo26n-t-b8f.fp16.onnx` | `yolo26n-t-b8f.fp32.onnx` |

### Run an inference

```bash
# Codegen path — single detection run on the CPU flex backend (default).
cargo run --release -p ultralytics --bin codegen_inference -- yolo11n fp32 --input image.jpg

# Runtime path — load the .onnx at runtime, no build-time codegen.
cargo run --release -p ultralytics --bin runtime_inference -- yolo26n fp16 --input image.jpg
```

### Benchmark mode

Pass `--runs N` (N > 1) to switch from a single detection run to a benchmark:
`--warmup` discarded runs, then N timed `forward` passes on the same
preprocessed input, followed by a decoded sample. Latency stats
(min/mean/median/p99) are printed alongside the detections.

```bash
cargo run --release -p ultralytics --bin codegen_inference -- \
    yolo11n fp32 --input image.jpg --runs 100 --warmup 5
```

### macOS Metal (zero-copy IOSurface import)

On macOS, enable the example's `metal` feature to use Burn's Metal backend and
`edgefirst-burn`'s zero-copy `IOSurface` import. HAL's `ImageProcessor::convert`
writes directly to an `IOSurface`-backed `PlanarRgb`/`F16` tensor;
`edgefirst-burn` imports that `IOSurface` into a Burn Metal tensor via
`newBufferWithBytesNoCopy` (no host copy), then casts F16→F32 on-GPU for fp32
models.

```bash
cargo run --release -p ultralytics --features metal --bin codegen_inference -- \
    yolo11n fp16 --input image.jpg
```

> `--features metal` resolves to the `ultralytics` package's own `metal` feature,
> which enables `burn/metal` + `edgefirst-burn/metal-iosurface`. Requires the
> EdgeFirst forks of `burn` and `cubecl` (see Prerequisites).

### Shared CLI

```
cargo run --release -p ultralytics --bin <BINARY> -- <MODEL> <DTYPE> --input <PATH> [OPTIONS]

Arguments:
  <MODEL>     yolov5n | yolov8n | yolo11n | yolo26n
  <DTYPE>     fp16 | fp32

Options:
  --input <PATH>     Input image (JPEG/PNG/etc. — anything HAL's codec decodes)  [required]
  --score <FLOAT>    Confidence threshold for the decoder                          [default: 0.25]
  --iou <FLOAT>      IoU threshold for NMS                                         [default: 0.7]
  --warmup <N>       Warmup runs before timing (benchmark mode only)              [default: 3]
  --runs <N>         1 = single detection run; N>1 = benchmark mode               [default: 1]
  --debug            Print input/output tensor statistics
```

### Parity test

The codegen and runtime paths are asserted to produce matching outputs (within
`ulps=2, epsilon=1e-4`) on a deterministic ramp input, for all four fp32
architectures:

```bash
cargo test -p ultralytics
```

### Known issue: yolov8n on the Metal backend

`yolov8n` produces no detections on the Burn **Metal** backend at the default
score threshold, while `yolo11n` and `yolov5n` are correct. This reproduces with
the pure-f32 host-fallback path on Metal (no IOSurface, no F16), so it is a
burn/cubecl Metal execution issue in yolov8n's specific op sequence — not a HAL
or `edgefirst-burn` bug. See
[`examples/ultralytics/README.md`](examples/ultralytics/README.md#known-issue-yolov8n-on-the-metal-backend)
for the full diagnosis.

## API at a glance

| Function | Purpose |
|---|---|
| [`with_image_tensor`] | HAL `ImageProcessor::convert` (letterbox + format) → Burn `Tensor<4>` via a scoped closure (lifetime-safe) |
| [`ultralytics_decoder`] | Build a HAL `Decoder` for `[1, 4+C, N]` (v8/11/5) or `[1, 6, N]` (Yolo26 end-to-end) |
| [`decode`] | Burn output `Tensor` → HAL `TensorDyn` → `Vec<DetectBox>` + `Vec<Segmentation>` (NMS in HAL) |
| [`draw_detections`] | Render boxes/masks onto a HAL destination image |
| [`draw_detections_over`] | Render boxes/masks composited over a background image |
| [`tensordyn_from_burn`] | Low-level Burn → HAL `TensorDyn` host bridge (any backend) |

See [`crates/edgefirst-burn/README.md`](crates/edgefirst-burn/README.md) and the
crate rustdoc (`cargo doc -p edgefirst-burn --open`) for full signatures,
constraints, and the IOSurface format / autotune-fallback notes.

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — high-level design, data flow, the
  zero-copy import mechanism, and the required forks + upstream PRs.
- [`crates/edgefirst-burn/README.md`](crates/edgefirst-burn/README.md) — the
  crate's detailed API doc (design table, features, quick starts, constraints).
- [`examples/ultralytics/README.md`](examples/ultralytics/README.md) — the
  examples' model table, usage, and the yolov8n-on-Metal diagnosis.
- Crate rustdoc: `cargo doc -p edgefirst-burn --open`.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

`SPDX-License-Identifier: Apache-2.0`
