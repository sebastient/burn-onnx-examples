//! Preprocessing: convert a source image via HAL's `ImageProcessor` into a Burn
//! input tensor.
//!
//! The HAL side performs decode-to-target-format conversion, resize, letterbox
//! (aspect-preserving) padding, rotation, and flip — all on the GPU when a DMA
//! convert target is requested. The Burn side receives the converted buffer
//! (zero-copy on macOS Metal, host upload otherwise).
//!
//! HAL owns normalization: its U8→F32/F16 convert already normalizes pixel
//! values to `[0,1]` (x/255), which is exactly what ultralytics models expect.
//! No mean/std or scale work is done on the Burn side — if a future model needs
//! ImageNet-style mean/std, that support will be added directly to HAL so the
//! whole pipeline stays zero-copy up to the model input.

use burn::tensor::{Device, DType as BurnDType, Tensor, TensorData};

use crate::{
    image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation},
    htensor::{CpuAccess, DType as HalDType, PixelFormat, TensorMemory, TensorDyn, TensorTrait},
    Error, Result,
};

/// The YOLO-standard letterbox pad colour (grey, RGB 114, opaque).
pub const YOLO_PAD: [u8; 4] = [114, 114, 114, 255];

/// Configuration for preparing a Burn model-input tensor from a HAL source
/// image.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Model input width.
    pub width: usize,
    /// Model input height.
    pub height: usize,
    /// Destination pixel format. `PlanarRgb` matches Burn's NCHW layout with no
    /// relayout (channels are already the leading dimension).
    pub format: PixelFormat,
    /// Destination dtype. `F32` for an fp32 model, `F16` for an fp16 model.
    /// HAL's U8→F32/F16 convert normalizes pixels to `[0,1]` (x/255).
    pub dtype: HalDType,
    /// Destination memory backend. `None` lets HAL pick (the safe default);
    /// `Some(TensorMemory::Dma)` requests a platform-native GPU buffer
    /// (`IOSurface` on macOS) for zero-copy import.
    pub memory: Option<TensorMemory>,
    /// Letterbox pad colour (RGBA). Applied by HAL when fitting the source into
    /// the target while preserving aspect ratio.
    pub pad: [u8; 4],
}

impl PreprocessConfig {
    /// Standard YOLO 640×640 fp32 config: `PlanarRgb`, `F32`, letterbox. HAL's
    /// convert normalizes pixels to `[0,1]` — no additional scaling is applied.
    pub fn yolo(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            format: PixelFormat::PlanarRgb,
            dtype: HalDType::F32,
            memory: None,
            pad: YOLO_PAD,
        }
    }

    /// Standard YOLO 640×640 fp16 config (same as [`Self::yolo`] but `F16`).
    pub fn yolo_f16(width: usize, height: usize) -> Self {
        let mut c = Self::yolo(width, height);
        c.dtype = HalDType::F16;
        c
    }

    /// Request a platform-native DMA/IOSurface convert target (zero-copy import
    /// on macOS Metal). The default (`None`) lets HAL pick the backend.
    pub fn with_memory(mut self, memory: TensorMemory) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Override the destination pixel format.
    pub fn with_format(mut self, format: PixelFormat) -> Self {
        self.format = format;
        self
    }

    /// Override the destination dtype.
    pub fn with_dtype(mut self, dtype: HalDType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the letterbox pad colour.
    pub fn with_pad(mut self, pad: [u8; 4]) -> Self {
        self.pad = pad;
        self
    }

    /// Number of channels for the configured format (3 for RGB/PlanarRgb, etc.).
    pub fn channels(&self) -> usize {
        self.format.channels()
    }
}

/// Geometry metadata captured during preprocessing so that decoded boxes (which
/// are in model-input normalized `[0,1]` space) can be mapped back to the
/// original source-image pixel coordinates.
///
/// `letterbox` is the normalized region `[lx0, ly0, lx1, ly1]` of the model
/// input that contains actual image content (the rest is padding). This matches
/// HAL's `MaskOverlay::letterbox` field and `MaskOverlay::with_letterbox_crop`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LetterboxMeta {
    /// Normalized content region of the model input `[xmin, ymin, xmax, ymax]`.
    pub letterbox: [f32; 4],
    /// Original source image width in pixels.
    pub src_w: usize,
    /// Original source image height in pixels.
    pub src_h: usize,
}

impl LetterboxMeta {
    /// Map a model-input normalized coordinate to the original source-image
    /// pixel coordinate, inverting the letterbox placement.
    ///
    /// `(nx, ny)` are in `[0,1]` model-input space. Returns `(px, py)` in
    /// source pixels.
    pub fn to_source_pixel(&self, nx: f32, ny: f32) -> (f32, f32) {
        let [lx0, ly0, lx1, ly1] = self.letterbox;
        let inv_w = if lx1 > lx0 { 1.0 / (lx1 - lx0) } else { 1.0 };
        let inv_h = if ly1 > ly0 { 1.0 / (ly1 - ly0) } else { 1.0 };
        let px = ((nx - lx0) * inv_w).clamp(0.0, 1.0) * self.src_w as f32;
        let py = ((ny - ly0) * inv_h).clamp(0.0, 1.0) * self.src_h as f32;
        (px, py)
    }
}

/// Compute the letterbox placement rectangle within a `dst_w`×`dst_h` target,
/// preserving the source aspect ratio and centring the content. Mirrors HAL's
/// internal `letterbox_rect`. Returns the content rect as normalized `[0,1]`
/// coordinates within the destination.
fn letterbox_region(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> [f32; 4] {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return [0.0, 0.0, 1.0, 1.0];
    }
    let src_aspect = src_w as f64 / src_h as f64;
    let dst_aspect = dst_w as f64 / dst_h as f64;
    let (content_w, content_h) = if src_aspect > dst_aspect {
        (dst_w as f64, dst_w as f64 / src_aspect)
    } else {
        (dst_h as f64 * src_aspect, dst_h as f64)
    };
    let left = (dst_w as f64 - content_w) / 2.0;
    let top = (dst_h as f64 - content_h) / 2.0;
    [
        left as f32 / dst_w as f32,
        top as f32 / dst_h as f32,
        (left + content_w) as f32 / dst_w as f32,
        (top + content_h) as f32 / dst_h as f32,
    ]
}

/// Preprocess a HAL source image and run a closure with the resulting Burn
/// tensor, **guaranteeing the HAL buffer owner outlives the tensor**.
///
/// Performs, via HAL's `ImageProcessor`:
/// 1. Letterbox resize of `src` to `cfg.width`×`cfg.height` (aspect-preserving,
///    padded with `cfg.pad`).
/// 2. Format conversion to `cfg.format` / `cfg.dtype` (e.g. `PlanarRgb`/`F32`,
///    with pixels normalized to `[0,1]`).
///
/// Then constructs a Burn `Tensor<4>` from the converted buffer. On macOS Metal
/// with `metal-iosurface` enabled and a DMA convert target, this is a **GPU
/// zero-copy** import. Otherwise the converted bytes are uploaded to the device
/// via the standard `Tensor::from_data` path (one host→device copy).
///
/// The HAL `TensorDyn` that owns the converted buffer (the IOSurface on macOS,
/// or a host allocation otherwise) is held inside this function for the
/// duration of the closure `f`. This guarantees the Burn tensor's device buffer
/// — which references that allocation in the zero-copy path — can never outlive
/// its owner. The closure receives `(&Tensor<4>, &LetterboxMeta)`.
///
/// # Example
///
/// ```no_run
/// # use pipeline as eb;
/// # use burn::tensor::Tensor;
/// # fn example(processor: &mut eb::image::ImageProcessor, src: &mut eb::htensor::TensorDyn, device: &burn::tensor::Device) -> eb::Result<()> {
/// let cfg = eb::PreprocessConfig::yolo(640, 640);
/// let result = eb::with_image_tensor(processor, src, &cfg, device, |input, letterbox| {
///     // `input` is a burn Tensor<4>; the HAL buffer owner is alive here.
///     // let out: Tensor<3> = model.forward(input.clone());
///     // eb::decode(&decoder, out, &[1, 84, 8400])
/// });
/// # Ok(())
/// # }
/// ```
pub fn with_image_tensor<R>(
    processor: &mut ImageProcessor,
    src: &mut TensorDyn,
    cfg: &PreprocessConfig,
    device: &Device,
    f: impl FnOnce(&Tensor<4>, &LetterboxMeta) -> R,
) -> Result<R> {
    let src_w = src.width().unwrap_or(cfg.width);
    let src_h = src.height().unwrap_or(cfg.height);

    // Allocate the convert destination in the requested format/dtype/memory.
    let mut dst = processor.create_image(
        cfg.width,
        cfg.height,
        cfg.format,
        cfg.dtype,
        cfg.memory,
        CpuAccess::Read,
    )?;

    let crop = Crop::letterbox(cfg.pad);
    processor.convert(src, &mut dst, Rotation::None, Flip::None, crop)?;

    let letterbox = letterbox_region(src_w, src_h, cfg.width, cfg.height);
    let meta = LetterboxMeta {
        letterbox,
        src_w,
        src_h,
    };

    // Build the burn tensor. When the convert target is IOSurface-backed (Dma
    // memory on macOS) and the metal-iosurface feature is enabled, import it
    // zero-copy via the wgpu Metal runtime. Otherwise fall back to a host
    // upload (one device→host read + host→device upload).
    let tensor = import_dst(&dst, cfg, device)?;

    // The HAL `dst` owner stays alive in this stack frame until `f` returns —
    // the Burn tensor (which references `dst`'s allocation in the zero-copy
    // path) cannot outlive it.
    Ok(f(&tensor, &meta))
}

/// Choose the zero-copy or host-fallback import path based on the convert
/// target's memory backend and the build configuration.
fn import_dst(dst: &TensorDyn, cfg: &PreprocessConfig, device: &Device) -> Result<Tensor<4>> {
    // Zero-copy path: only when the dst is IOSurface-backed (TensorMemory::Dma
    // on macOS) and the metal-iosurface feature is enabled.
    #[cfg(all(feature = "metal-iosurface", target_os = "macos"))]
    {
        if dst.memory() == TensorMemory::Dma {
            return import_zero_copy(dst, cfg, device);
        }
    }
    // Host fallback for all other cases (Mem/Shm, or non-macOS / no feature).
    import_via_host(dst, cfg, device)
}

/// Zero-copy import: extract the IOSurface handle from the HAL convert target
/// and import it into the wgpu Metal runtime. Only compiled with
/// `metal-iosurface` on macOS.
///
/// **IOSurface format constraint:** macOS IOSurface-backed tensors only support
/// a limited set of (format, dtype) pairs — `PlanarRgb` is supported at `F16`,
/// NOT `F32`. So the IOSurface convert target must be `PlanarRgb`/`F16`
/// regardless of the model's requested dtype. After the zero-copy import, the
/// tensor is cast on-GPU to the model's dtype (one cheap kernel).
#[cfg(all(feature = "metal-iosurface", target_os = "macos"))]
fn import_zero_copy(dst: &TensorDyn, cfg: &PreprocessConfig, device: &Device) -> Result<Tensor<4>> {
    // The IOSurface is always PlanarRgb/F16 (ANGLE's only float IOSurface
    // binding). Validate the format matches what the import expects.
    if cfg.format != PixelFormat::PlanarRgb {
        return Err(Error::UnsupportedDType(cfg.dtype));
    }

    let surface = dst
        .iosurface_ref()
        .ok_or(Error::IoSurfaceUnavailable)?;
    let c = cfg.channels();
    let elem_size = HalDType::F16.size();
    let byte_size = c * cfg.height * cfg.width * elem_size;
    // SAFETY: the cubecl import CFRetains the IOSurface and the MTLBuffer
    // deallocator CFReleases it when the buffer is dropped. The surface is
    // therefore self-contained — the HAL TensorDyn owner does not need to
    // outlive the burn tensor. The `with_image_tensor` scoping is a
    // convenience, not a hard lifetime requirement.
    let mut tensor = unsafe {
        crate::tensor::import::image_tensor_from_iosurface(
            surface,
            byte_size,
            [1, c, cfg.height, cfg.width],
            BurnDType::F16,
            device,
        )
    }?;
    // Cast from the IOSurface's F16 to F32. The IOSurface is always F16
    // (ANGLE's only float format), and the cast is one cheap on-GPU kernel
    // — the import itself stays zero-copy. F32 is the validated model input
    // dtype for the fp32 ultralytics models on Metal.
    //
    // Note: `cfg.dtype` is F16 here (the IOSurface format), NOT the model's
    // expected input dtype. The model's forward() handles dtype internally.
    tensor = tensor.cast(BurnDType::F32);
    Ok(tensor)
}

/// Phase-A host fallback: read the HAL convert target to host, wrap as a Burn
/// `Tensor<4>`. HAL already normalized the pixels to `[0,1]` during the convert,
/// so no further scaling is applied.
///
/// This is one device→host read (of the converted buffer) followed by a
/// host→device upload (the standard Burn path). Phase C replaces this with a
/// GPU zero-copy import on macOS Metal.
fn import_via_host(dst: &TensorDyn, cfg: &PreprocessConfig, device: &Device) -> Result<Tensor<4>> {
    let c = cfg.channels();
    let bytes_owned = read_dst_bytes(dst)?;
    let burn_dtype = match cfg.dtype {
        HalDType::F32 => BurnDType::F32,
        HalDType::F16 => BurnDType::F16,
        other => return Err(Error::UnsupportedDType(other)),
    };
    let data = TensorData::from_bytes_vec(bytes_owned, [1usize, c, cfg.height, cfg.width], burn_dtype);
    let t: Tensor<4> = Tensor::from_data(data, (device, burn_dtype));
    Ok(t)
}

/// Read a HAL convert target's bytes to an owned `Vec<u8>` in its native dtype.
fn read_dst_bytes(dst: &TensorDyn) -> Result<Vec<u8>> {
    match dst {
        TensorDyn::F32(t) => {
            let m = t.map_read()?;
            Ok(bytemuck::cast_slice::<f32, u8>(&m[..]).to_vec())
        }
        TensorDyn::F16(t) => {
            let m = t.map_read()?;
            Ok(bytemuck::cast_slice::<half::f16, u8>(&m[..]).to_vec())
        }
        TensorDyn::U8(t) => {
            let m = t.map_read()?;
            Ok(m[..].to_vec())
        }
        other => Err(Error::UnsupportedDType(other.dtype())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_region_wide_source() {
        // Source 1920x1080 into 640x640: width-limited, height shrinks.
        let r = letterbox_region(1920, 1080, 640, 640);
        assert!((r[0] - 0.0).abs() < 1e-4);
        assert!((r[2] - 1.0).abs() < 1e-4);
        assert!((r[1] - (1.0 - r[3])).abs() < 1e-3, "top/bottom not centred");
    }

    #[test]
    fn letterbox_region_tall_source() {
        // Source 1080x1920 into 640x640: height-limited, width shrinks.
        let r = letterbox_region(1080, 1920, 640, 640);
        assert!((r[1] - 0.0).abs() < 1e-4);
        assert!((r[3] - 1.0).abs() < 1e-4);
        assert!((r[0] - (1.0 - r[2])).abs() < 1e-3);
    }

    #[test]
    fn letterbox_region_square() {
        // Square source into same-shape target: no padding.
        let r = letterbox_region(640, 640, 640, 640);
        assert!((r[0] - 0.0).abs() < 1e-4);
        assert!((r[1] - 0.0).abs() < 1e-4);
        assert!((r[2] - 1.0).abs() < 1e-4);
        assert!((r[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn to_source_pixel_roundtrip() {
        let meta = LetterboxMeta {
            letterbox: letterbox_region(1920, 1080, 640, 640),
            src_w: 1920,
            src_h: 1080,
        };
        let (px, py) = meta.to_source_pixel(meta.letterbox[0], meta.letterbox[1]);
        assert!(px.abs() < 1.0, "px={px}");
        assert!(py.abs() < 1.0, "py={py}");
        let (px, py) = meta.to_source_pixel(meta.letterbox[2], meta.letterbox[3]);
        assert!((px - 1920.0).abs() < 1.0, "px={px}");
        assert!((py - 1080.0).abs() < 1.0, "py={py}");
    }
}
