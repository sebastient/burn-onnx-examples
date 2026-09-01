//! Rendering: draw decoded detections (boxes + segmentation masks) onto a HAL
//! destination image using `ImageProcessor::draw_decoded_masks`.
//!
//! This is the display/debug path. For best latency on segmentation models the
//! fused `ImageProcessor::draw_masks(decoder, outputs, dst)` path avoids
//! materializing intermediate masks; this module uses the explicit
//! decode-then-draw flow so the caller keeps access to the decoded `DetectBox`
//! list for downstream use.

use edgefirst_decoder::{DetectBox, Segmentation};

use crate::{
    image::{ColorMode, ImageProcessor, ImageProcessorTrait, MaskOverlay},
    htensor::TensorDyn,
    preprocess::LetterboxMeta,
    Error, Result,
};

/// Draw decoded detections onto `dst`.
///
/// `dst` is fully overwritten (its prior contents are not preserved). To
/// composite over a background image, build the overlay with
/// [`MaskOverlay::with_background`].
///
/// When `letterbox` is `Some`, the inverse letterbox transform is applied so
/// boxes/masks are mapped from model-input normalized space back to the
/// destination image's coordinate space.
///
/// # Format requirements
///
/// `dst` must be `RGB` or `RGBA` (`RGBA`/`BGRA`) on the CPU/OpenGL backends.
/// Allocate it with [`ImageProcessor::create_image`] in one of those formats.
pub fn draw_detections(
    processor: &mut ImageProcessor,
    dst: &mut TensorDyn,
    boxes_: &[DetectBox],
    masks: &[Segmentation],
    letterbox: Option<&LetterboxMeta>,
) -> Result<()> {
    let mut overlay = MaskOverlay::default();
    if let Some(meta) = letterbox {
        // Reconstruct the letterbox transform from the captured metadata. The
        // overlay wants the normalized region directly.
        overlay.letterbox = Some(meta.letterbox);
    }
    processor
        .draw_decoded_masks(dst, boxes_, masks, overlay)
        .map_err(Error::from)?;
    Ok(())
}

/// Draw decoded detections composited over a background image, with the given
/// opacity and color mode. Thin wrapper around [`draw_detections`] for the
/// common "overlay masks on the original frame" case.
pub fn draw_detections_over(
    processor: &mut ImageProcessor,
    dst: &mut TensorDyn,
    background: &TensorDyn,
    boxes_: &[DetectBox],
    masks: &[Segmentation],
    opacity: f32,
    color_mode: ColorMode,
    letterbox: Option<&LetterboxMeta>,
) -> Result<()> {
    let mut overlay = MaskOverlay::default()
        .with_background(background)
        .with_opacity(opacity)
        .with_color_mode(color_mode);
    if let Some(meta) = letterbox {
        overlay.letterbox = Some(meta.letterbox);
    }
    processor
        .draw_decoded_masks(dst, boxes_, masks, overlay)
        .map_err(Error::from)?;
    Ok(())
}
