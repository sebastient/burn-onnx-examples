//! Postprocessing: turn a Burn ultralytics output tensor into decoded
//! detections using HAL's `edgefirst_decoder`.
//!
//! The Burn model emits a `[1, 4+num_classes, num_anchors]` tensor (e.g.
//! `[1, 84, 8400]` for COCO 80-class). HAL's decoder splits boxes from class
//! scores, filters by confidence, runs NMS, and emits normalized `[0,1]` XYXY
//! boxes. This is **host-side** — the burn output is pulled to host and wrapped
//! as a HAL `TensorDyn`, then handed to `Decoder::decode`. A future effort could
//! push the decode onto the burn/cubecl GPU.

use burn::tensor::Tensor;
use edgefirst_decoder::{
    configs, Decoder, DecoderBuilder, DecoderVersion, DetectBox, Nms, Segmentation,
};

use crate::{
    htensor,
    tensor::export::{burn_dtype_to_hal, tensordyn_from_burn_data},
    Error, Result,
};

/// Build a HAL decoder for a fused-output ultralytics detection model.
///
/// `version` selects the YOLO decode head. `output_rows` is the number of rows
/// in the model's fused output (dim 1 of `[1, R, N]`):
/// - `6` → the Yolo26 end-to-end layout `(x1,y1,x2,y2,conf,class)`; HAL skips
///   NMS (the model already applied it).
/// - `4 + num_classes` → the pre-NMS layout used by v8/11/5 (and by some Yolo26
///   exports that don't embed NMS); HAL runs NMS. The class count is derived
///   from `output_rows - 4`.
///
/// Pass the **observed** output row count (from the burn tensor's dims) so the
/// decoder config matches the actual tensor regardless of how the model was
/// exported.
///
/// `score` / `iou` are the confidence and IoU thresholds.
pub fn ultralytics_decoder(
    version: DecoderVersion,
    output_rows: usize,
    score: f32,
    iou: f32,
) -> Result<Decoder> {
    // Detect end-to-end from the row count: 6 features means the model already
    // applied NMS. Anything else is the pre-NMS [4+C, N] layout.
    let is_end_to_end = output_rows == 6;
    let anchors = if is_end_to_end { 300 } else { 8400 };
    // HAL ties the end-to-end verification to the decoder *version*: a Yolo26
    // version requires the 6-feature layout. A Yolo26 export that did NOT embed
    // NMS (still emits [1, 4+C, 8400]) must be decoded with a pre-NMS head.
    // Yolo26's pre-NMS head is the same anchor-free DFL head as Yolo11, so use
    // that when the observed layout is pre-NMS.
    let decode_version = if is_end_to_end {
        version
    } else if version == DecoderVersion::Yolo26 {
        DecoderVersion::Yolo11
    } else {
        version
    };
    let detection = configs::Detection {
        anchors: None,
        decoder: configs::DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, output_rows, anchors],
        dshape: Vec::new(),
        normalized: Some(true),
    };
    let mut builder = DecoderBuilder::new()
        .with_config_yolo_det(detection, Some(decode_version))
        .with_score_threshold(score)
        .with_iou_threshold(iou)
        .with_max_det(300);
    // NMS is only meaningful for the pre-NMS heads; end-to-end models have NMS
    // baked into the graph.
    if !is_end_to_end {
        builder = builder.with_nms(Some(Nms::ClassAgnostic)).with_pre_nms_top_k(300);
    }
    let decoder = builder.build().map_err(Error::from)?;
    Ok(decoder)
}

/// Decode a single burn output tensor into NMS-filtered detections.
///
/// `output_shape` is the shape the decoder should see (e.g. `[1, 84, 8400]`).
/// It overrides the burn tensor's own shape metadata so the decoder receives a
/// well-formed `[1, 4+C, N]` view regardless of how the model code reshaped it.
///
/// Returns `(boxes, masks)`. For a pure detection model `masks` is empty; for
/// segmentation models it carries per-detection instance masks.
pub fn decode<const D: usize>(
    decoder: &Decoder,
    output: Tensor<D>,
    output_shape: &[usize],
) -> Result<(Vec<DetectBox>, Vec<Segmentation>)> {
    // Infer the HAL dtype from the burn tensor's actual dtype so F16 model
    // outputs are handled correctly (not just F32).
    let data = output.into_data();
    let dtype = burn_dtype_to_hal(data.dtype);
    let td = tensordyn_from_burn_data(data, output_shape, dtype)?;
    let mut boxes = Vec::new();
    let mut masks = Vec::new();
    decoder
        .decode(&[&td], &mut boxes, &mut masks)
        .map_err(Error::from)?;
    Ok((boxes, masks))
}

/// Decode multiple burn output tensors (e.g. for split-output models: separate
/// boxes `[1,4,N]` and scores `[1,C,N]`). The decoder must have been built with
/// the matching split config (`with_config_yolo_split_det`).
pub fn decode_many<const D: usize>(
    decoder: &Decoder,
    outputs: Vec<Tensor<D>>,
    shapes: &[&[usize]],
) -> Result<(Vec<DetectBox>, Vec<Segmentation>)> {
    let tds: Vec<htensor::TensorDyn> = outputs
        .into_iter()
        .zip(shapes.iter())
        .map(|(t, s)| {
            let data = t.into_data();
            let dtype = burn_dtype_to_hal(data.dtype);
            tensordyn_from_burn_data(data, s, dtype)
        })
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&htensor::TensorDyn> = tds.iter().collect();
    let mut boxes = Vec::new();
    let mut masks = Vec::new();
    decoder.decode(&refs, &mut boxes, &mut masks).map_err(Error::from)?;
    Ok((boxes, masks))
}

// Re-export common decoder types so callers can use this crate without pulling
// in the HAL `decoder` namespace directly.
pub use edgefirst_decoder::{BoundingBox, DecoderError as HalDecoderError};

#[cfg(all(test, feature = "test"))]
mod tests {
    use super::*;

    #[test]
    fn decoder_builds_for_yolov8_coco() {
        // [1, 84, 8400] pre-NMS output.
        let d = ultralytics_decoder(DecoderVersion::Yolov8, 84, 0.25, 0.7);
        assert!(d.is_ok(), "{:?}", d.err());
    }

    #[test]
    fn decoder_builds_for_yolo11_coco() {
        let d = ultralytics_decoder(DecoderVersion::Yolo11, 84, 0.25, 0.7);
        assert!(d.is_ok(), "{:?}", d.err());
    }

    #[test]
    fn decoder_builds_for_yolo26_end_to_end() {
        // True end-to-end: [1, 6, N].
        let d = ultralytics_decoder(DecoderVersion::Yolo26, 6, 0.25, 0.7);
        assert!(d.is_ok(), "{:?}", d.err());
    }

    #[test]
    fn decoder_builds_for_yolo26_pre_nms_export() {
        // A Yolo26 export that did NOT embed NMS still emits [1, 84, 8400].
        let d = ultralytics_decoder(DecoderVersion::Yolo26, 84, 0.25, 0.7);
        assert!(d.is_ok(), "{:?}", d.err());
    }
}
