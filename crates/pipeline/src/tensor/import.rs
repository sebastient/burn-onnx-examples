//! Zero-copy foreign-buffer import: turn a HAL `IOSurface` into a Burn `Tensor`
//! with **no host copy**, by importing the surface's storage into the wgpu
//! Metal runtime and surfacing it as a Burn tensor.
//!
//! This module is compiled only when the `metal-iosurface` feature is enabled
//! on macOS. It depends on:
//! - the EdgeFirst cubecl fork (`WgpuClientExt::import_iosurface`),
//! - the patched burn fork (`Device::as_dispatch` public).
//!
//! # Path
//!
//! `IOSurfaceRef` → `WgpuClientExt::import_iosurface` (cubecl Handle) →
//! `CubeTensor::new_contiguous` → `Tensor::from_primitive::<Metal>`.
//!
//! The cubecl import CFRetains the IOSurface and locks it for the lifetime of
//! the resulting `MTLBuffer`. When the Burn tensor is dropped, the MTLBuffer's
//! deallocator unlocks and CFReleases the surface. The surface is therefore
//! self-contained — the HAL `TensorDyn` owner does not need to outlive the Burn
//! tensor.

use core::ffi::c_void;

use burn::tensor::{Device, Shape, Tensor};

use crate::{Error, Result};

/// Build a Burn `Tensor<4>` from an `IOSurface`-backed HAL buffer with **no
/// host copy**.
///
/// Imports the `IOSurface` into the wgpu Metal runtime via the cubecl
/// `WgpuClientExt::import_iosurface`, wraps the resulting cubecl `Handle` as a
/// `CubeTensor`, and surfaces it as a Burn `Tensor<4>` via
/// `Tensor::from_primitive::<Metal>`.
///
/// # Safety
///
/// `surface` must be a valid `IOSurfaceRef`. The cubecl import CFRetains the
/// surface and the MTLBuffer deallocator CFReleases it when the buffer is
/// dropped, so the surface is self-contained — no external lifetime management
/// is required.
///
/// `device` must be a Metal-backed burn `Device` (constructed via
/// `Device::metal(...)` or resolved to `DispatchDevice::Metal`).
pub unsafe fn image_tensor_from_iosurface(
    surface: *mut c_void,
    byte_size: usize,
    shape: [usize; 4],
    dtype: burn::tensor::DType,
    device: &Device,
) -> Result<Tensor<4>> {
    use burn::backend::{DispatchDevice, Metal};
    use burn::cubecl::wgpu::{MslCompiler, WgpuClientExt, WgpuDevice, WgpuRuntime};
    use burn::cubecl::Runtime;
    use burn_cubecl::tensor::CubeTensor;

    // 1. Extract the wgpu Metal device from the burn dispatch device.
    let wgpu_device: WgpuDevice = match device.as_dispatch() {
        DispatchDevice::Metal(d) => d.clone(),
        _ => return Err(Error::IoSurfaceUnavailable),
    };

    // 2. Get the typed cubecl compute client for the Metal runtime.
    let client = <WgpuRuntime<MslCompiler> as Runtime>::client(&wgpu_device);

    // 3. Import the IOSurface into the wgpu Metal runtime (zero copy). The
    //    surface is CFRetain'd + locked inside cubecl; the MTLBuffer
    //    deallocator unlocks + releases it when the buffer is dropped.
    let handle = unsafe { client.import_iosurface(surface, byte_size as u64) }
        .map_err(|_| Error::IoSurfaceUnavailable)?;

    // 4. Build the CubeTensor — burn::Metal's FloatTensorPrimitive.
    let cube_tensor: CubeTensor<WgpuRuntime<MslCompiler>> = CubeTensor::new_contiguous(
        client,
        wgpu_device,
        Shape::from(shape),
        handle,
        dtype,
    );

    // 5. Inject into the dispatch tensor via from_primitive::<Metal>.
    let tensor: Tensor<4> = Tensor::<4>::from_primitive::<Metal>(cube_tensor);
    Ok(tensor)
}
