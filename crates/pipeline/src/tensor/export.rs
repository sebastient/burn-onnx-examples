//! Pull a Burn output tensor to the host and wrap it as a HAL `TensorDyn` so
//! HAL's decoder/NMS/render pipeline can consume it.
//!
//! This path is **inherently host-side**: HAL's YOLO decoder runs on `ndarray`
//! views over host memory (see `edgefirst_decoder::TensorBridge`). A future
//! effort could push the decode onto the Burn/cubecl GPU; until then, this is
//! the one device→host copy in the pipeline, and it happens on the *output*
//! (small: `[1,84,8400]` ≈ 2.7 MB for f32), not on the input image.

use burn::tensor::{DType as BurnDType, Tensor, TensorData};

use crate::{htensor, Error, Result};

/// Convert a Burn dtype to the matching HAL dtype, or `None` if there is no
/// direct mapping (e.g. `Bool`, `QFloat`, `BF16`).
pub(crate) fn burn_dtype_to_hal(dtype: BurnDType) -> htensor::DType {
    match dtype {
        BurnDType::F32 => htensor::DType::F32,
        BurnDType::F16 => htensor::DType::F16,
        BurnDType::F64 => htensor::DType::F64,
        BurnDType::I8 => htensor::DType::I8,
        BurnDType::U8 => htensor::DType::U8,
        BurnDType::I16 => htensor::DType::I16,
        BurnDType::U16 => htensor::DType::U16,
        BurnDType::I32 => htensor::DType::I32,
        BurnDType::U32 => htensor::DType::U32,
        BurnDType::I64 => htensor::DType::I64,
        BurnDType::U64 => htensor::DType::U64,
        // BF16, Bool, QFloat have no direct HAL equivalent — fall back to F32.
        _ => htensor::DType::F32,
    }
}

/// Convert a Burn tensor to a HAL `TensorDyn` by pulling it to host memory.
///
/// `shape` overrides the shape recorded in the burn tensor (useful when burn's
/// own shape metadata differs from what the HAL decoder expects, e.g. a
/// reshaped view). The HAL dtype is inferred from the burn tensor's dtype.
///
/// The returned `TensorDyn` owns a fresh `TensorMemory::Mem` allocation holding
/// a copy of the burn tensor's bytes.
///
/// # Supported dtypes
///
/// `F32`, `F16`, `F64`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`.
/// `BF16`/`Bool`/`QFloat` fall back to F32 (the caller should cast first).
pub fn tensordyn_from_burn<const D: usize>(
    tensor: Tensor<D>,
    shape: &[usize],
) -> Result<htensor::TensorDyn> {
    let data = tensor.into_data();
    let dtype = burn_dtype_to_hal(data.dtype);
    tensordyn_from_burn_data(data, shape, dtype)
}

/// Low-level: convert already-materialized [`TensorData`] to a HAL `TensorDyn`.
/// The HAL `dtype` must match the burn `TensorData`'s dtype (validated
/// internally via [`dtypes_compatible`]).
pub(crate) fn tensordyn_from_burn_data(
    data: TensorData,
    shape: &[usize],
    dtype: htensor::DType,
) -> Result<htensor::TensorDyn> {
    // Validate the dtype pairing and the byte budget.
    let need_bytes = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .unwrap_or(usize::MAX)
        .checked_mul(dtype.size())
        .ok_or_else(|| Error::BufferShapeMismatch {
            got_bytes: data.bytes.len(),
            need_bytes: 0,
            shape: shape.to_vec(),
            dtype,
        })?;
    if data.bytes.len() < need_bytes {
        return Err(Error::BufferShapeMismatch {
            got_bytes: data.bytes.len(),
            need_bytes,
            shape: shape.to_vec(),
            dtype,
        });
    }

    let burn_dtype = data.dtype;
    if !dtypes_compatible(burn_dtype, dtype) {
        return Err(Error::UnsupportedDType(dtype));
    }

    macro_rules! build {
        ($hal_ty:ty, $variant:ident) => {{
            let slice: &[$hal_ty] = bytemuck::checked::try_cast_slice(&data.bytes[..])
                .map_err(|_| Error::UnsupportedDType(dtype))?;
            let tensor = htensor::Tensor::<$hal_ty>::from_slice(slice, shape)
                .map_err(crate::Error::from)?;
            htensor::TensorDyn::$variant(tensor)
        }};
    }

    let out = match dtype {
        htensor::DType::F32 => build!(f32, F32),
        htensor::DType::F16 => build!(half::f16, F16),
        htensor::DType::F64 => build!(f64, F64),
        htensor::DType::I8 => build!(i8, I8),
        htensor::DType::U8 => build!(u8, U8),
        htensor::DType::I16 => build!(i16, I16),
        htensor::DType::U16 => build!(u16, U16),
        htensor::DType::I32 => build!(i32, I32),
        htensor::DType::U32 => build!(u32, U32),
        htensor::DType::I64 => build!(i64, I64),
        htensor::DType::U64 => build!(u64, U64),
        _ => return Err(Error::UnsupportedDType(dtype)),
    };
    Ok(out)
}

/// Returns `true` when reinterpreting burn's `burn_dtype` bytes as HAL's
/// `hal_dtype` is byte-for-byte sound (same element width and signedness).
fn dtypes_compatible(burn_dtype: BurnDType, hal_dtype: htensor::DType) -> bool {
    match hal_dtype {
        htensor::DType::F32 => matches!(burn_dtype, BurnDType::F32),
        htensor::DType::F16 => matches!(burn_dtype, BurnDType::F16),
        htensor::DType::F64 => matches!(burn_dtype, BurnDType::F64),
        htensor::DType::I8 => matches!(burn_dtype, BurnDType::I8),
        htensor::DType::U8 => matches!(burn_dtype, BurnDType::U8 | BurnDType::Bool(_)),
        htensor::DType::I16 => matches!(burn_dtype, BurnDType::I16),
        htensor::DType::U16 => matches!(burn_dtype, BurnDType::U16),
        htensor::DType::I32 => matches!(burn_dtype, BurnDType::I32),
        htensor::DType::U32 => matches!(burn_dtype, BurnDType::U32),
        htensor::DType::I64 => matches!(burn_dtype, BurnDType::I64),
        htensor::DType::U64 => matches!(burn_dtype, BurnDType::U64),
        _ => false,
    }
}

#[cfg(all(test, feature = "test"))]
mod tests {
    use super::*;
    use burn::tensor::Device;
    use htensor::TensorTrait;

    #[test]
    fn f32_roundtrip() {
        let device = Device::default();
        let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::<1>::from_floats(vals.clone().as_slice(), &device).reshape([2, 3]);
        let td = tensordyn_from_burn(t, &[2, 3]).unwrap();
        assert_eq!(td.dtype(), htensor::DType::F32);
        let t32 = td.as_f32().unwrap();
        let mapped = t32.map_read().unwrap();
        assert_eq!(&mapped[..], &vals);
    }
}
