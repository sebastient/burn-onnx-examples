# burn-onnx-examples

Models ported to the [Burn](https://github.com/tracel-ai/burn) ONNX **runtime**
([`burn-onnx-runtime`](https://github.com/sebastient/burn-onnx/tree/burn-onnx-runtime))
and **codegen** ([`burn-onnx`](https://github.com/tracel-ai/burn-onnx)) paths, using the
open-source [EdgeFirst HAL](https://github.com/EdgeFirstAI/hal) crates for
hardware-accelerated pre-processing and post-processing.

The `pipeline` crate bridges HAL's image-preprocessing pipeline (convert targets
backed by `IOSurface` on macOS, `dma-buf` on Linux) and Burn's compute backends,
so a HAL-converted image can feed a Burn model with **no host-side copy** on the
input path. The examples target [Ultralytics](https://www.ultralytics.com/) YOLO
detection models end-to-end — preprocessing, inference, and decode + NMS —
through two interchangeable execution paths: build-time codegen and runtime
graph execution.

> **Status.** The default build uses upstream `tracel-ai/burn` plus the
> `burn-onnx-runtime` proposal branch. Only the zero-copy `IOSurface` import on
> macOS Metal (`pipeline/metal-iosurface`) still requires forked `burn`/`cubecl`
> branches — swap the dependency blocks in `Cargo.toml` as commented there. See
> [Required forks](ARCHITECTURE.md#required-forks-and-upstream-prs) in
> `ARCHITECTURE.md`.

## Repository layout

This is a Cargo workspace.

| Member | Description |
|---|---|
| [`crates/pipeline`](crates/pipeline/) | The published integration crate — the HAL↔Burn bridge. |
| [`examples/ultralytics`](examples/ultralytics/) | End-to-end YOLO examples (codegen + runtime paths) over 8 model variants. |

The crate-level API documentation lives at
[`crates/pipeline/README.md`](crates/pipeline/README.md); the
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
cargo build -p pipeline

# Run the crate's unit + integration tests.
cargo test -p pipeline --features test
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

The examples use 8 variants as ONNX (opset 13, input `images [1,3,640,640]`,
output `[1,84,8400]`), expected in `examples/ultralytics/src/model/`. The weight
files are **not committed** (see Licensing below): download them from the
[EdgeFirst Model Zoo](https://huggingface.co/spaces/EdgeFirst/Models), or export
your own with `scripts/export-yolo.sh` (uses `pip install ultralytics` in a
local venv) and rename to match:

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
`pipeline`'s zero-copy `IOSurface` import. HAL's `ImageProcessor::convert`
writes directly to an `IOSurface`-backed `PlanarRgb`/`F16` tensor;
`pipeline` imports that `IOSurface` into a Burn Metal tensor via
`newBufferWithBytesNoCopy` (no host copy), then casts F16→F32 on-GPU for fp32
models.

```bash
cargo run --release -p ultralytics --features metal --bin codegen_inference -- \
    yolo11n fp16 --input image.jpg
```

> `--features metal` resolves to the `ultralytics` package's own `metal` feature,
> which enables `burn/metal` + `pipeline/metal-iosurface`. Requires the
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
or `pipeline` bug. See
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

See [`crates/pipeline/README.md`](crates/pipeline/README.md) and the
crate rustdoc (`cargo doc -p pipeline --open`) for full signatures,
constraints, and the IOSurface format / autotune-fallback notes.

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — high-level design, data flow, the
  zero-copy import mechanism, and the required forks + upstream PRs.
- [`crates/pipeline/README.md`](crates/pipeline/README.md) — the
  crate's detailed API doc (design table, features, quick starts, constraints).
- [`examples/ultralytics/README.md`](examples/ultralytics/README.md) — the
  examples' model table, usage, and the yolov8n-on-Metal diagnosis.
- Crate rustdoc: `cargo doc -p pipeline --open`.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

`SPDX-License-Identifier: Apache-2.0`

## Licensing

Everything in this repository — the examples, the `pipeline` HAL↔Burn bridge,
and the EdgeFirst stack it builds on — is **Apache-2.0**, and the stack runs
other models as well as Ultralytics.

The Ultralytics YOLO models themselves are **AGPL-3.0** (or Ultralytics'
commercial licence), and that holds regardless of where you get the weights:
exported locally with `pip install ultralytics`, or downloaded from the
[EdgeFirst Model Zoo](https://huggingface.co/spaces/EdgeFirst/Models), which
distributes the same weights under the same licence. This repository never
redistributes model weights; fetching them is on you, as is using them within
the Ultralytics licence terms.
