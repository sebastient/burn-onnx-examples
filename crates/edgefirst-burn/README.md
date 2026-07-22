# edgefirst-burn

Zero-copy integration between [edgefirst-hal](https://crates.io/crates/edgefirst-hal) and the [Burn](https://github.com/tracel-ai/burn) deep-learning framework.

`edgefirst-burn` bridges HAL's hardware-accelerated image-preprocessing pipeline (convert targets backed by `IOSurface` on macOS, `dma-buf` on Linux) and Burn's compute backends, so that a HAL-converted image can feed a Burn model with **no host-side copy** on the input path.

## Design

| Path | Direction | Copy behaviour |
|---|---|---|
| **Input** (HAL image → Burn) | HAL convert target → Burn tensor | **GPU zero-copy** on macOS Metal (`IOSurface` → `MTLBuffer`); host upload fallback otherwise |
| **Output** (Burn → HAL decoder) | Burn model output → HAL `TensorDyn` | Host pull — HAL's YOLO decoder/NMS is an `ndarray` host kernel |

The asymmetry is by design: HAL already owns the GPU-resident converted image, so the input can be imported with no copy. The decoder runs on host `ndarray` views, so the output must come back to the host (a small `[1, 84, 8400]` tensor ≈ 2.7 MB for f32).

## Features

- `metal-iosurface` — zero-copy `IOSurface` → Burn Metal import (macOS). Requires the consumer to also enable `burn/metal` and to carry the EdgeFirst cubecl + burn forks (see the consuming project's `[patch]` table). Without this feature the crate always uses the host fallback.
- `test` — enables `burn/flex` so unit/integration tests can construct tensors.

## Quick start — ONNX codegen path

```rust,ignore
use edgefirst_burn as eb;
use burn::tensor::{Device, Tensor};
use edgefirst_hal as hal;

let device = Device::default();
let mut processor = hal::image::ImageProcessor::new()?;
// ... load a source image into a HAL TensorDyn `src` ...
let cfg = eb::PreprocessConfig::yolo(640, 640);

// with_image_tensor holds the HAL buffer owner alive for the closure's
// lifetime, so the burn tensor (which references the HAL allocation in the
// zero-copy path) can never outlive it.
eb::with_image_tensor(&mut processor, &mut src, &cfg, &device, |input, letterbox| {
    // Run your codegen-generated burn model:
    let out: Tensor<3> = model.forward(input.clone());
    // Decode + NMS via HAL's edgefirst-decoder.
    let decoder = eb::ultralytics_decoder(
        hal::decoder::DecoderVersion::Yolov8, 84, 0.25, 0.7)?;
    let (boxes_, masks) = eb::decode(&decoder, out, &[1, 84, 8400])?;
    for b in &boxes_ {
        let (px, py) = letterbox.to_source_pixel(b.bbox.xmin, b.bbox.ymin);
        println!("class={} score={:.3} @ ({:.0},{:.0})", b.label, b.score, px, py);
    }
})?;
```

## Quick start — ONNX runtime path

The runtime path loads the `.onnx` file directly via `burn_onnx_runtime::GraphExecutor` — no build-time codegen, no `.bpk`:

```rust,ignore
use burn_onnx_runtime::{GraphExecutor, Value, FloatTensor};
use std::collections::HashMap;

let model = GraphExecutor::from_file("model.onnx", &device)?;
let input_name = model.input_names().next().unwrap().to_string();
let output_name = model.output_names().next().unwrap().to_string();
let mut inputs = HashMap::new();
inputs.insert(input_name, Value::from(input.clone()));
let outputs = model.forward(inputs)?;
if let Value::Float(FloatTensor::R3(out)) = outputs.get(&output_name).unwrap() {
    let (boxes_, _) = eb::decode(&decoder, out.clone(), &[1, 84, 8400])?;
}
```

## API

| Function | Purpose |
|---|---|
| [`with_image_tensor`] | HAL `ImageProcessor::convert` (letterbox + format) → Burn `Tensor<4>` via a scoped closure (lifetime-safe) |
| [`ultralytics_decoder`] | Build a HAL `Decoder` for `[1, 4+C, N]` (v8/11/5) or `[1, 6, N]` (Yolo26 end-to-end) |
| [`decode`] | Burn output `Tensor` → HAL `TensorDyn` → `Vec<DetectBox>` + `Vec<Segmentation>` (NMS in HAL) |
| [`draw_detections`] | Render boxes/masks onto a HAL destination image |
| [`draw_detections_over`] | Render boxes/masks composited over a background image |
| [`tensordyn_from_burn`] | Low-level Burn → HAL `TensorDyn` host bridge (any backend) |

## Normalization

HAL owns normalization. Its U8→F32/F16 convert **already normalizes pixel values to `[0,1]`** (x/255), which is exactly what ultralytics models expect. `edgefirst-burn` applies no additional scaling. If a future model needs ImageNet-style mean/std, that support will be added directly to HAL so the whole pipeline stays zero-copy up to the model input.

## IOSurface format constraint (macOS Metal zero-copy)

macOS IOSurface only supports a limited set of `(PixelFormat, DType)` pairs. For planar float, only `PlanarRgb`/`F16` is available (ANGLE's `iosurface_client_buffer` extension accepts only RGBA16F float bindings). The zero-copy import path therefore always uses `PlanarRgb/F16` and casts to the caller-requested dtype on-GPU (one cheap kernel — the import itself stays zero-copy). F32 IOSurface is spec-rejected by ANGLE.

## Decoder layout detection

`ultralytics_decoder` takes the **observed** output row count (`dim[1]` of the model's `[1, R, N]` output) and selects the decode path from it:
- `R == 6` → Yolo26 end-to-end `(x1,y1,x2,y2,conf,class)`; HAL skips NMS.
- `R == 4 + num_classes` → pre-NMS layout (v8/11/5, and Yolo26 exports without embedded NMS); HAL runs NMS. A Yolo26 export that emits the pre-NMS layout is automatically decoded with the Yolo11 anchor-free DFL head.

## Autotune fallback

The cubecl fork includes a patch to the autotune cache-hit path: if the autotune-selected kernel fails to launch at runtime, the runtime falls back to the next candidate in registration order instead of panicking. This unblocks F16 input on the Metal backend where the conv autotune may select a CMMA kernel that fails on re-run.

## Required forks

The zero-copy path requires three fork branches (all `ef/iosurface-import`):

| Fork | Repository | Changes |
|---|---|---|
| **cubecl** | `sebastient/cubecl` | IOSurface import (`WgpuClientExt`), autotune fallback, `ComputeClient::submit_blocking` |
| **burn** | `tracel-ai/burn` | `Device::as_dispatch` public, `FusionTensor::from_handle` |
| **edgefirst-burn** | `EdgeFirstAI/edgefirst-burn` | This crate |

The consuming project's `Cargo.toml` carries `[patch]` tables pointing at the local clones. See the `codegen_inference` / `runtime_inference` examples for a working configuration.

## Examples

The `examples/ultralytics/` directory holds two end-to-end binaries that share the HAL pipeline and differ only in model loading: `codegen_inference` (build-time-compiled `.bpk`) and `runtime_inference` (loads `.onnx` at runtime via `burn-onnx-runtime`). Both load a JPEG via HAL, preprocess with `with_image_tensor`, run a YOLO model, decode via HAL's decoder, and print detections.

## License

Apache-2.0.
