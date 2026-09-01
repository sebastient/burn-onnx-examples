//! End-to-end host-pipeline integration test (no GPU zero-copy, no model).
//!
//! Exercises the full pipeline glue on CPU:
//! 1. Synthesize a HAL RGB source image.
//! 2. `with_image_tensor` converts it to PlanarRgb F32 and builds a burn tensor.
//! 3. Synthesize a `[1, 84, 8400]` burn output tensor (a fake model output).
//! 4. `decode` runs HAL's YOLOv8 decode + NMS.
//! 5. `draw_detections` renders onto an RGB destination.
//!
//! This validates that every HAL↔burn type bridge, the convert path, and the
//! decode/draw path compose correctly. It does NOT validate detection accuracy
//! (no real model) — that is the job of the ultralytics `codegen_inference` /
//! `runtime_inference` examples.

#![cfg(feature = "test")]

use burn::tensor::{Device, Tensor};
use pipeline::{
    decoder::DecoderVersion,
    htensor::{self, CpuAccess, DType as HalDType, PixelFormat, TensorMemory, TensorTrait},
    image::ImageProcessor,
    preprocess::PreprocessConfig,
    {decode, draw_detections, ultralytics_decoder, with_image_tensor},
};

/// Build a synthetic RGB u8 source image with a simple gradient.
fn make_src(width: usize, height: usize) -> htensor::TensorDyn {
    let mut src = htensor::TensorDyn::image(
        width,
        height,
        PixelFormat::Rgb,
        HalDType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("allocate src");
    {
        let t = src.as_u8_mut().expect("u8 tensor");
        let mut m = t.map_mut().expect("map mut");
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                m[i] = (x as u8).wrapping_mul(2); // R
                m[i + 1] = (y as u8).wrapping_mul(2); // G
                m[i + 2] = 128; // B
            }
        }
    }
    src
}

#[test]
fn full_host_pipeline_convert_decode_draw() {
    let device = Device::default();

    // 1. Source image.
    let mut src = make_src(320, 240);

    // 2. Preprocess: convert to 64×64 PlanarRgb F32 (small to keep the test
    //    fast), letterboxed. Host fallback builds a burn tensor.
    let mut processor = ImageProcessor::new().expect("ImageProcessor");
    let cfg = PreprocessConfig::yolo(64, 64);

    // Validate the input tensor shape inside the closure.
    let input_shape = with_image_tensor(
        &mut processor,
        &mut src,
        &cfg,
        &device,
        |input, _letterbox| {
            let data = input.clone().into_data();
            assert_eq!(data.shape.as_slice(), &[1, 3, 64, 64]);
            let slice: &[f32] = bytemuck::cast_slice(&data.bytes[..]);
            assert!(
                slice.iter().all(|v| v.is_finite()),
                "nan/inf in normalized input"
            );
            data.shape.as_slice().to_vec()
        },
    )
    .expect("preprocess");
    assert_eq!(input_shape, &[1, 3, 64, 64]);

    // 3. Synthesize a fake [1, 84, 8400] model output.
    let out_vals: Vec<f32> = (0..(84 * 8400))
        .map(|i| ((i as f32 * 0.0001).sin() * 0.5 + 0.5) * 0.01)
        .collect();
    let output: Tensor<3> =
        Tensor::<1>::from_floats(out_vals.as_slice(), &device).reshape([1, 84, 8400]);

    // 4. Decode with the YOLOv8 decoder.
    let decoder = ultralytics_decoder(DecoderVersion::Yolov8, 84, 0.5, 0.7).expect("decoder");
    let (boxes_, masks) = decode(&decoder, output, &[1, 84, 8400]).expect("decode");
    assert!(masks.is_empty(), "detection model emits no masks");

    // 5. Draw onto an RGB destination.
    let mut dst = htensor::TensorDyn::image(
        64,
        64,
        PixelFormat::Rgb,
        HalDType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("allocate dst");
    draw_detections(&mut processor, &mut dst, &boxes_, &masks, None).expect("draw");

    let dst_t = dst.as_u8().expect("u8 dst");
    let _m = dst_t.map_read().expect("read dst after draw");
}

#[test]
fn preprocess_shape_and_dtype() {
    let device = Device::default();
    let mut src = make_src(128, 96);
    let mut processor = ImageProcessor::new().expect("ImageProcessor");
    let cfg = PreprocessConfig::yolo_f16(32, 32);
    let shape = with_image_tensor(
        &mut processor,
        &mut src,
        &cfg,
        &device,
        |input, _| input.clone().into_data().shape.as_slice().to_vec(),
    )
    .expect("preprocess");
    assert_eq!(shape, &[1, 3, 32, 32]);
}
