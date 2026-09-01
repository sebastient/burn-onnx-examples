# Architecture

This document describes the high-level architecture of `pipeline`: how it
bridges [edgefirst-hal](https://crates.io/crates/edgefirst-hal) (hardware-accelerated
image preprocessing) and [Burn](https://github.com/tracel-ai/burn) (deep-learning
compute), and the upstream forks that the zero-copy path requires.

For the crate API, see [`crates/pipeline/README.md`](crates/pipeline/README.md).
For usage, see the root [`README.md`](README.md).

## Overview

`pipeline` is the glue layer in this stack:

```
 ┌──────────┐   HAL ImageProcessor   ┌─────────────────┐   Burn model   ┌─────────────┐
 │  source  │ ─────────────────────► │  pipeline │ ─────────────► │   decode    │
 │  image   │   letterbox + convert  │  (this crate)   │   Tensor<4>    │  + NMS      │
 │ (JPEG/…) │                        │  import tensor  │                │ (HAL host)  │
 └──────────┘                        └─────────────────┘                └─────────────┘
        ▲                                       │                              │
        │                                       │                              │
   edgefirst-codec                    zero-copy on macOS Metal         Vec<DetectBox>
                                      (IOSurface), host fallback        mapped to src px
                                      otherwise
```

The crate does **three** things:

1. **Preprocess** — drive HAL's `ImageProcessor` to letterbox + format-convert a
   source image, and surface the result as a Burn `Tensor<4>` input.
2. **Bridge** — import the HAL-converted buffer into Burn's device memory with
   **no host-side copy** when the platform allows it (macOS Metal + `IOSurface`),
   or via a host-upload fallback otherwise.
3. **Postprocess** — pull the Burn model's output tensor to host, wrap it as a
   HAL `TensorDyn`, and run HAL's YOLO decode + NMS to get `DetectBox`es.

The application owns step 2's middle — the actual `model.forward(input)` — so
`pipeline` never depends on a specific model. The Ultralytics examples
([`examples/ultralytics/`](examples/ultralytics/)) wire that middle in.

## The asymmetric zero-copy design

The input and output paths are deliberately asymmetric:

| Path | Direction | Copy behaviour | Why |
|---|---|---|---|
| **Input** (HAL image → Burn) | HAL convert target → Burn tensor | **GPU zero-copy** on macOS Metal (`IOSurface` → `MTLBuffer`); host upload fallback otherwise | HAL already owns the GPU-resident converted image — it can be imported directly. |
| **Output** (Burn → HAL decoder) | Burn model output → HAL `TensorDyn` | **Host pull** — one device→host read | HAL's YOLO decoder + NMS is an `ndarray` host kernel; it needs the data on the CPU. |

The output pull is cheap: the fused YOLO output is `[1, 84, 8400]` ≈ 2.7 MB for
f32 (80 COCO classes), decoded once per frame. The asymmetry keeps the hot
input path copy-free while leaving the cold decode path on the host where HAL's
decoder already lives.

## Data flow

End-to-end, a single inference does (function names in `code`):

1. **Load** — `edgefirst_codec` decodes a JPEG/PNG into a HAL `TensorDyn` (`src`).
2. **Preprocess** — `with_image_tensor(&mut processor, &mut src, &cfg, &device, |input, letterbox| …)`:
   - HAL `ImageProcessor::convert` letterboxes `src` into a `cfg.width × cfg.height`
     `PlanarRgb` target and converts dtype + normalizes pixels to `[0,1]` (x/255).
   - `LetterboxMeta` records the content rectangle so decoded boxes can be
     mapped back to source pixels.
   - The converted buffer is imported into Burn as `Tensor<4>` (zero-copy or host).
3. **Infer** — the application calls `model.forward(input.clone())` → `Tensor<3>`
   shaped `[1, R, N]` (e.g. `[1, 84, 8400]`). The crate is model-agnostic here.
4. **Decode** — inside the closure:
   - `ultralytics_decoder(version, output_rows, score, iou)` builds a HAL
     `Decoder` whose config is selected from the **observed** row count `R`.
   - `decode(&decoder, out, &shape)` pulls the tensor to host, wraps it as a
     HAL `TensorDyn`, and runs HAL decode + NMS → `(Vec<DetectBox>, Vec<Segmentation>)`.
5. **Map back** — `letterbox.to_source_pixel(x, y)` converts model-normalized
   `[0,1]` box coords back to source-image pixels.

The HAL `TensorDyn` that owns the converted buffer is held alive inside
`with_image_tensor` for the closure's duration, so the Burn tensor — which
references that allocation on the zero-copy path — cannot outlive it.

## Module map

`crates/pipeline/src/`:

| Module | Responsibility | Key item |
|---|---|---|
| `preprocess.rs` | HAL convert + letterbox → Burn `Tensor<4>`; picks zero-copy vs host import. | `with_image_tensor`, `PreprocessConfig`, `LetterboxMeta` |
| `postprocess.rs` | Burn output → HAL decode + NMS. Auto-detects end-to-end vs pre-NMS layout from the row count. | `ultralytics_decoder`, `decode` |
| `render.rs` | Draw decoded boxes/masks onto a HAL destination image. | `draw_detections`, `draw_detections_over` |
| `tensor/import.rs` | **macOS Metal only.** Zero-copy `IOSurface` → Burn `Tensor`. | `image_tensor_from_iosurface` |
| `tensor/export.rs` | Burn → HAL `TensorDyn` host bridge (any backend). | `tensordyn_from_burn` |
| `tensor/foreign.rs` | Opaque platform-native buffer handle enum. | `ForeignBuffer` |
| `lib.rs` | Re-exports, `Error`, `Result`, HAL convenience re-exports. | `Error`, `Result` |

## Zero-copy import mechanism (macOS Metal)

Compiled only with `feature = "metal-iosurface"` on macOS. The path
(`tensor/import.rs::image_tensor_from_iosurface`) is five steps:

1. **Extract the wgpu device** — `Device::as_dispatch()` (made public by the
   burn fork) returns the concrete `DispatchDevice`; match `DispatchDevice::Metal(d)`
   to get the cubecl `WgpuDevice`.
2. **Get the typed compute client** —
   `<WgpuRuntime<MslCompiler> as Runtime>::client(&wgpu_device)`.
3. **Import the IOSurface** — `client.import_iosurface(surface, byte_size)`
   (cubecl fork) creates an `MTLBuffer` backed by the surface's storage via
   `newBufferWithBytesNoCopy`. The import CFRetains the surface; the buffer's
   deallocator CFReleases it, so the handle is self-contained.
4. **Wrap as a CubeTensor** — `CubeTensor::new_contiguous(client, device, shape, handle, dtype)`
   — Burn's `Metal` float-tensor primitive.
5. **Inject into Burn** — `Tensor::<4>::from_primitive::<Metal>(cube_tensor)`.

**IOSurface format constraint.** macOS IOSurface only supports a limited set of
`(PixelFormat, DType)` pairs. For planar float, only `PlanarRgb`/`F16` is
available (ANGLE's `iosurface_client_buffer` extension accepts only RGBA16F
float bindings). The zero-copy convert target is therefore always
`PlanarRgb`/`F16`, regardless of the model's dtype; after import, the tensor is
cast to the model's dtype (e.g. F32) with one cheap on-GPU kernel. F32
IOSurface is spec-rejected by ANGLE.

**Autotune fallback.** The cubecl fork patches the autotune cache-hit path: if
the autotune-selected kernel fails to launch at runtime (e.g. a CMMA tile
availability check accepted during tuning but rejected on a later run), the
runtime falls back to the next candidate in registration order instead of
panicking. This unblocks F16 input on Metal, where conv autotune may select a
CMMA kernel that fails on re-run.

## Host fallback path

Used when the `metal-iosurface` feature is off, the platform is not macOS, or
the convert target is not DMA-backed (`TensorMemory::Mem`/`Shm`).
`preprocess.rs::import_via_host` reads the converted buffer to an owned
`Vec<u8>` and constructs the Burn tensor via the standard
`Tensor::from_data` path — one device→host read (of the converted buffer)
followed by a host→device upload. Correctness is identical; only the input copy
is added.

## Lifetime safety

`with_image_tensor` takes a closure `f: impl FnOnce(&Tensor<4>, &LetterboxMeta) -> R`
and holds the HAL `TensorDyn` (the IOSurface owner on macOS) alive in its stack
frame until `f` returns. The Burn tensor references that allocation on the
zero-copy path, so this scoping makes "the tensor outlives its backing storage"
unrepresentable at the type level. (The cubecl import's CFRetain/CFRelease
makes the surface self-contained regardless; the scoping is defence-in-depth.)

## Required forks and upstream PRs

The zero-copy Metal path depends on changes to three upstream crates, carried as
local path patches in the workspace `Cargo.toml` `[patch]` tables. Each is
intended as an upstream PR. Until they merge, consumers must carry the same
`[patch]` tables pointing at local clones on the branches below.

| Fork | Repository | Branch | Upstream PR target | Changes |
|---|---|---|---|---|
| **cubecl** | `sebastient/cubecl` (fork of `tracel-ai/cubecl`) | `ef/iosurface-import` | `tracel-ai/cubecl` | `WgpuClientExt::import_iosurface` (foreign-buffer zero-copy import for wgpu Metal, in `cubecl-wgpu`); autotune fallback to next candidate on launch failure (in `cubecl-runtime`); `ComputeClient::submit_blocking`. ~12 files, +8 commits. |
| **burn** | `tracel-ai/burn` | `ef/iosurface-import` | `tracel-ai/burn` | `Device::as_dispatch()` made public (was `pub(crate)`); `FusionTensor::from_handle` constructor. 2 files, +2 commits. Small, additive, review-ready. |
| **burn-onnx** | `tracel-ai/burn-onnx` (origin) / `sebastient/burn-onnx` (fork) | `feat/burn-onnx-runtime` | `tracel-ai/burn-onnx` | New `burn-onnx-runtime` crate — runtime ONNX `GraphExecutor` that loads `.onnx` directly with no build-time codegen, plus op handlers (matmul, conv, maxpool, slice, split, resize, reduce, normalize, elementwise, constant, shape) and the ultralytics example. ~47 files, +16 commits. |

> **Branch names.** The crate-level README states "all `ef/iosurface-import`",
> which covers `cubecl` and `burn`. The `burn-onnx` fork is on
> `feat/burn-onnx-runtime` (the runtime crate is its headline change), not
> `ef/iosurface-import`. The workspace `[patch]` tables point at local path
> clones regardless of branch name.

### How the patches wire up

The workspace `Cargo.toml` carries:

- `[patch."https://github.com/tracel-ai/cubecl"]` — every cubecl sub-crate →
  `../cubecl/crates/*`.
- `[patch."https://github.com/tracel-ai/burn"]` — the burn sub-crates this
  workspace reaches under its `flex`/`metal`/`vision` feature set →
  `../burn/crates/*` (deliberately omits `burn-dataset`, `burn-train`,
  `burn-rl`, etc., which aren't reached).
- `burn-onnx` / `burn-onnx-runtime` are direct path dependencies on
  `../burn-onnx/crates/*` (not a `[patch]` entry — they aren't on crates.io at
  the required version yet).
- A commented-out `[patch.crates-io]` block is reserved for pointing the HAL
  crates at a local `../hal` fork when HAL changes are pending upstream.

## Feature flags

The crate (`crates/pipeline`):

| Feature | Effect |
|---|---|
| `metal-iosurface` | Enables the zero-copy `IOSurface` → Burn Metal import (macOS). Pulls in `burn/metal`, `burn/cubecl`, `burn-cubecl`. Requires the EdgeFirst cubecl + burn forks. Without it, the crate always uses the host fallback. |
| `test` | Enables `burn/flex` so unit/integration tests can construct tensors on a CPU backend. |

The `ultralytics` example package additionally defines a `metal` feature
(`burn/metal` + `pipeline/metal-iosurface`) as the user-facing toggle for
the Metal zero-copy path; see its
[`Cargo.toml`](examples/ultralytics/Cargo.toml).

## Build & test quick reference

```bash
# Integration crate (CPU flex backend).
cargo build -p pipeline
cargo test  -p pipeline --features test

# Examples — default (CPU flex).
cargo run --release -p ultralytics --bin codegen_inference -- yolo11n fp32 --input image.jpg
cargo run --release -p ultralytics --bin runtime_inference -- yolo26n fp16 --input image.jpg

# Examples — macOS Metal zero-copy.
cargo run --release -p ultralytics --features metal --bin codegen_inference -- yolo11n fp16 --input image.jpg

# Codegen ↔ runtime parity.
cargo test -p ultralytics
```
