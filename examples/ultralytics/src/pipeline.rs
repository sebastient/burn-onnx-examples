//! Shared end-to-end pipeline for the Ultralytics YOLO examples.
//!
//! Both example binaries (`codegen_inference`, `runtime_inference`) run the same
//! HAL-preprocessed pipeline:
//!
//! 1. Load a JPEG/PNG via `edgefirst-codec` into a HAL `TensorDyn`.
//! 2. HAL `ImageProcessor::convert` (letterbox + format conversion) → Burn input
//!    `Tensor<4>` via [`pipeline::with_image_tensor`] (zero-copy
//!    `IOSurface` import on macOS Metal, host upload otherwise).
//! 3. Run the model's `forward` (this is the only thing that differs between the
//!    two binaries — how the model is constructed).
//! 4. Decode + NMS via HAL's `edgefirst-decoder`, map boxes back to source pixels.
//!
//! The only difference between the binaries is step 3: the codegen binary loads
//! a build-time-compiled `Model` from its `.bpk` weights; the runtime binary
//! loads the `.onnx` at runtime via `burn-onnx-runtime::GraphExecutor`. Both
//! are surfaced to [`run_pipeline`] through the [`InferenceModel`] trait.
//!
//! With `--runs N` (N > 1) the pipeline switches to benchmark mode: it warms up
//! (discarded timings), then runs N timed `forward` passes on the same
//! preprocessed input, decoding the last run so correctness is visible alongside
//! the latency stats.

use std::collections::HashMap;
use std::time::Instant;

use burn::tensor::{DType, Device, Tensor};
use burn_onnx_runtime::{FloatTensor, GraphExecutor, Value};
use clap::{Parser, ValueEnum};
use edgefirst_codec::{ImageDecoder, ImageLoad, peek_info};
use edgefirst_hal as hal;
use pipeline as eb;

use crate::model::{
    yolo11n_t_b8b_fp16, yolo11n_t_b8b_fp32, yolo26n_t_b8f_fp16, yolo26n_t_b8f_fp32,
    yolov5n_t_b7b_fp16, yolov5n_t_b7b_fp32, yolov8n_t_b86_fp16, yolov8n_t_b86_fp32,
};

// ─── CLI ─────────────────────────────────────────────────────────────────────

/// Which YOLO architecture to run.
#[derive(Clone, Copy, ValueEnum)]
pub enum Arch {
    Yolov5n,
    Yolov8n,
    Yolo11n,
    Yolo26n,
}

impl Arch {
    /// The HAL decoder version matching this architecture.
    pub fn decoder_version(self) -> hal::decoder::DecoderVersion {
        match self {
            Arch::Yolov5n => hal::decoder::DecoderVersion::Yolov5,
            Arch::Yolov8n => hal::decoder::DecoderVersion::Yolov8,
            Arch::Yolo11n => hal::decoder::DecoderVersion::Yolo11,
            Arch::Yolo26n => hal::decoder::DecoderVersion::Yolo26,
        }
    }
}

/// Which weight dtype to load.
#[derive(Clone, Copy, ValueEnum)]
pub enum Precision {
    Fp16,
    Fp32,
}

impl Precision {
    /// The Burn dtype for this precision.
    pub fn dtype(self) -> DType {
        match self {
            Precision::Fp16 => DType::F16,
            Precision::Fp32 => DType::F32,
        }
    }
}

/// Shared CLI arguments for both example binaries.
#[derive(Parser)]
pub struct Args {
    /// Model architecture.
    #[arg(value_enum)]
    pub model: Arch,
    /// Weight precision (selects the fp16 or fp32 variant).
    #[arg(value_enum)]
    pub dtype: Precision,
    /// Path to the input image (JPEG/PNG/etc. — anything HAL's codec decodes).
    #[arg(long)]
    pub input: String,
    /// Confidence threshold for the decoder. Default 0.25.
    #[arg(long, default_value_t = 0.25)]
    pub score: f32,
    /// IoU threshold for NMS. Default 0.7.
    #[arg(long, default_value_t = 0.7)]
    pub iou: f32,
    /// Warmup runs before timing (benchmark mode only, not timed). Default 3.
    #[arg(long, default_value_t = 3)]
    pub warmup: usize,
    /// Number of timed forward passes. `1` (the default) runs a single
    /// detection pass and prints the results. `N > 1` switches to benchmark
    /// mode: `--warmup` discarded runs, then N timed runs with latency stats.
    #[arg(long, default_value_t = 1)]
    pub runs: usize,
    /// Print input/output tensor statistics for debugging.
    #[arg(long)]
    pub debug: bool,
    /// Save a copy of the input image with the detection overlays drawn by
    /// HAL (`draw_decoded_masks`), encoded as JPEG at the given path.
    #[arg(long)]
    pub output: Option<String>,
}

// ─── Latency statistics ──────────────────────────────────────────────────────

/// Summary statistics computed from a sample of per-run latencies (microseconds).
pub struct Stats {
    pub min: f64,
    pub mean: f64,
    pub median: f64,
    pub p99: f64,
}

/// Compute min/mean/median/p99 from a slice of per-run latencies in microseconds.
pub fn stats(samples: &[f64]) -> Stats {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("non-nan latency"));
    let n = s.len();
    let mean = s.iter().sum::<f64>() / n as f64;
    let p99_idx = ((n as f64) * 0.99) as usize;
    Stats {
        min: s[0],
        mean,
        median: s[n / 2],
        p99: s[p99_idx],
    }
}

/// Min/max/mean of an f32 slice (ignoring NaN/Inf for the mean only).
pub fn stats_f32(s: &[f32]) -> (f32, f32, f32) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for &v in s {
        if v.is_finite() {
            mn = mn.min(v);
            mx = mx.max(v);
            sum += v as f64;
            n += 1;
        }
    }
    (mn, mx, if n > 0 { (sum / n as f64) as f32 } else { 0.0 })
}

// ─── Image loading ───────────────────────────────────────────────────────────

/// Decode a JPEG/PNG via HAL's codec into a HAL `TensorDyn`. The codec emits the
/// source's native format (typically NV12 for a colour JPEG).
pub fn load_source_image(
    path: &str,
) -> Result<edgefirst_tensor::TensorDyn, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let info = peek_info(&bytes)?;
    let mut t = edgefirst_tensor::Tensor::<u8>::image(
        info.width,
        info.height,
        info.format,
        Some(edgefirst_tensor::TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )?;
    let mut dec = ImageDecoder::new();
    t.load_image(&mut dec, &bytes)?;
    Ok(edgefirst_tensor::TensorDyn::U8(t))
}

// ─── InferenceModel trait + adapters ─────────────────────────────────────────

/// Abstraction over a YOLO model that maps a `[1, 3, H, W]` input tensor to its
/// `[1, R, N]` detection output. Implemented by the codegen-generated `Model`
/// types (via [`CodegenModel`]) and by the runtime [`RuntimeModel`] wrapper.
pub trait InferenceModel {
    fn forward(&self, input: Tensor<4>) -> Tensor<3>;
}

/// Trait-bridge to the 8 codegen-generated `Model` types so they can be used
/// through [`InferenceModel`]. Each generated `Model` has
/// `from_file<P: AsRef<Path>>(P, &Device) -> Self` and `forward(&self, Tensor<4>)
/// -> Tensor<3>`.
pub trait CodegenModel {
    fn from_file(bpk: &str, device: &Device) -> Self;
    fn forward(&self, input: Tensor<4>) -> Tensor<3>;
}

macro_rules! impl_codegen_model {
    ($($m:ident),* $(,)?) => {
        $(
            impl CodegenModel for $m::Model {
                fn from_file(bpk: &str, device: &Device) -> Self {
                    <$m::Model>::from_file(bpk, device)
                }
                fn forward(&self, input: Tensor<4>) -> Tensor<3> {
                    <$m::Model>::forward(self, input)
                }
            }
        )*
    };
}
impl_codegen_model!(
    yolo11n_t_b8b_fp16,
    yolo11n_t_b8b_fp32,
    yolo26n_t_b8f_fp16,
    yolo26n_t_b8f_fp32,
    yolov5n_t_b7b_fp16,
    yolov5n_t_b7b_fp32,
    yolov8n_t_b86_fp16,
    yolov8n_t_b86_fp32,
);

impl<M: CodegenModel> InferenceModel for M {
    fn forward(&self, input: Tensor<4>) -> Tensor<3> {
        CodegenModel::forward(self, input)
    }
}

/// Adapter wrapping a `burn-onnx-runtime` `GraphExecutor` so the runtime path
/// can be driven through [`InferenceModel`]. Resolves the single input/output
/// names on construction.
pub struct RuntimeModel {
    executor: GraphExecutor,
    input_name: String,
    output_name: String,
}

impl RuntimeModel {
    /// Load an `.onnx` file at runtime. No build-time codegen.
    pub fn from_file(onnx: &str, device: &Device) -> Result<Self, burn_onnx_runtime::Error> {
        let executor = GraphExecutor::from_file(onnx, device)?;
        // A YOLO graph has exactly one input and one output; a graph missing
        // either is malformed and not something we can recover from.
        let input_name = executor
            .input_names()
            .next()
            .expect("graph has an input")
            .to_string();
        let output_name = executor
            .output_names()
            .next()
            .expect("graph has an output")
            .to_string();
        Ok(Self {
            executor,
            input_name,
            output_name,
        })
    }
}

impl InferenceModel for RuntimeModel {
    fn forward(&self, input: Tensor<4>) -> Tensor<3> {
        let mut inputs = HashMap::new();
        inputs.insert(self.input_name.clone(), Value::from(input));
        let outputs = self.executor.forward(inputs).expect("runtime forward");
        match outputs.get(&self.output_name).expect("runtime output") {
            Value::Float(FloatTensor::R3(t)) => t.clone(),
            other => panic!("expected rank-3 output, got rank {}", other.rank()),
        }
    }
}

// ─── Shared pipeline ─────────────────────────────────────────────────────────

/// Build the preprocessing config for the requested dtype. On macOS Metal (with
/// the `metal` feature) the IOSurface-backed convert target only supports
/// `PlanarRgb`/`F16`, so the F16 target is forced regardless of the model dtype
/// — `pipeline` casts back to the model's dtype on-GPU after the import.
pub fn preprocess_config(dtype: DType) -> eb::PreprocessConfig {
    let cfg = match dtype {
        DType::F16 => eb::PreprocessConfig::yolo_f16(640, 640),
        _ => eb::PreprocessConfig::yolo(640, 640),
    };
    // On macOS Metal, request an IOSurface-backed convert target (Dma) so the
    // pipeline import is zero-copy. macOS IOSurface only supports
    // PlanarRgb at F16, so the F16 convert target is forced for fp32 models —
    // pipeline casts back to the model's dtype on-GPU after the import.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let cfg = cfg.with_memory(edgefirst_tensor::TensorMemory::Dma);
    cfg
}

/// Run the shared end-to-end pipeline on `model`: preprocess `src`, run the
/// model `args.runs` times (benchmark mode when `args.runs > 1`), decode + NMS
/// via HAL, and print detections (and latency stats in benchmark mode).
///
/// The HAL `TensorDyn` owner is held alive for the closure's lifetime inside
/// [`eb::with_image_tensor`], guaranteeing the Burn input tensor — which
/// references that allocation on the zero-copy path — cannot outlive it.
pub fn run_pipeline<M: InferenceModel>(
    model: &M,
    device: &Device,
    processor: &mut hal::image::ImageProcessor,
    src: &mut edgefirst_tensor::TensorDyn,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = preprocess_config(args.dtype.dtype());
    let version = args.model.decoder_version();

    // `with_image_tensor` holds the HAL buffer owner alive for the closure's
    // lifetime, so the Burn tensor (which references the HAL allocation on the
    // zero-copy path) cannot outlive it.
    let result = eb::with_image_tensor(processor, src, &cfg, device, |input, letterbox| {
        if args.debug {
            // Cast to F32 for stats so F16 model inputs display correctly.
            let in_f32 = input.clone().cast(DType::F32);
            let in_data = in_f32.into_data();
            let in_slice: &[f32] = bytemuck::cast_slice(&in_data.bytes[..]);
            let (mn, mx, mean) = stats_f32(in_slice);
            eprintln!(
                "input [{} f32] min={mn:.4} max={mx:.4} mean={mean:.4}",
                in_slice.len()
            );
        }

        let out: Tensor<3> = if args.runs > 1 {
            // Benchmark mode: warmup + N timed forwards on the same input.
            for _ in 0..args.warmup {
                let _ = model.forward(input.clone());
            }
            let mut samples: Vec<f64> = Vec::with_capacity(args.runs);
            for _ in 0..args.runs {
                let start = Instant::now();
                let timed = model.forward(input.clone());
                // Force device sync so the timing reflects the full op, not
                // just kernel dispatch.
                let _ = timed.into_data();
                samples.push(start.elapsed().as_secs_f64() * 1_000_000.0); // us
            }
            let s = stats(&samples);
            println!("benchmark mode: {} timed runs", args.runs);
            println!("  min:     {:.1} us", s.min);
            println!("  mean:    {:.1} us", s.mean);
            println!("  median:  {:.1} us", s.median);
            println!("  p99:     {:.1} us", s.p99);
            // One more forward for the decoded sample shown below.
            model.forward(input.clone())
        } else {
            model.forward(input.clone())
        };

        let dims = out.dims();

        if args.debug {
            let out_f32 = out.clone().cast(DType::F32);
            let out_data = out_f32.into_data();
            let out_slice: &[f32] = bytemuck::cast_slice(&out_data.bytes[..]);
            let (mn, mx, mean) = stats_f32(out_slice);
            let above = out_slice.iter().filter(|v| **v > args.score).count();
            eprintln!(
                "output [{dims:?}] min={mn:.4} max={mx:.4} mean={mean:.4} vals>{:.2}={above}",
                args.score
            );
        }

        // Build the decoder AFTER inference so the config shape matches the
        // model's actual output. A 6-row output is the Yolo26 end-to-end layout
        // (x1,y1,x2,y2,conf,class); a (4+num_classes)-row output is the pre-NMS
        // layout used by v8/11/5 (and by some Yolo26 exports that don't embed
        // NMS). `ultralytics_decoder` picks the right config from the version +
        // the observed row count.
        let output_rows = dims[1];
        let decoder = eb::ultralytics_decoder(version, output_rows, args.score, args.iou)?;

        let out_shape: Vec<usize> = dims.to_vec();
        let (boxes_, masks) = eb::decode(&decoder, out, &out_shape)?;
        println!("detections: {} (masks: {})", boxes_.len(), masks.len());
        for b in &boxes_ {
            let (x1, y1) = letterbox.to_source_pixel(b.bbox.xmin, b.bbox.ymin);
            let (x2, y2) = letterbox.to_source_pixel(b.bbox.xmax, b.bbox.ymax);
            println!(
                "  class={:<3} score={:.3}  [({:.0},{:.0})-({:.0},{:.0})]",
                b.label, b.score, x1, y1, x2, y2
            );
        }
        Ok::<_, Box<dyn std::error::Error>>((boxes_, masks, *letterbox))
    });

    let (boxes_, masks, letterbox) = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Err(e),
        Err(e) => return Err(Box::new(e) as Box<dyn std::error::Error>),
    };

    if let Some(path) = &args.output {
        // Same render recipe as the ara2-rs / tflite-rs HAL examples: an RGBA
        // canvas at source resolution, the source image as the overlay
        // background, and the letterbox transform mapping detections back to
        // source coordinates; the canvas is then encoded as JPEG.
        use eb::htensor::{CpuAccess, DType as HalDType, PixelFormat};
        use eb::image::ImageProcessorTrait as _;
        let mut canvas = processor.create_image(
            letterbox.src_w,
            letterbox.src_h,
            PixelFormat::Rgba,
            HalDType::U8,
            None,
            CpuAccess::Read,
        )?;
        // Convert the native-format source (typically NV12 from the JPEG
        // decoder) to an RGBA background matching the canvas shape/format.
        let mut background = processor.create_image(
            letterbox.src_w,
            letterbox.src_h,
            PixelFormat::Rgba,
            HalDType::U8,
            None,
            CpuAccess::Read,
        )?;
        processor.convert(
            src,
            &mut background,
            eb::image::Rotation::None,
            eb::image::Flip::None,
            eb::image::Crop::default(),
        )?;
        eb::draw_detections_over(
            processor,
            &mut canvas,
            &background,
            &boxes_,
            &masks,
            0.5,
            eb::image::ColorMode::Class,
            Some(&letterbox),
        )?;
        eb::image::save_jpeg(&canvas, path, 95)?;
        println!("wrote {path}");
    }

    Ok(())
}
