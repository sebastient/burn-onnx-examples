//! Platform-native zero-copy buffer handles that HAL can hand to Burn.
//!
//! A `ForeignBuffer` is the *opaque* side of the zero-copy contract: HAL owns
//! the storage and Burn borrows it for the lifetime of the resulting tensor.
//! Each variant is platform-specific and is only constructed on the matching
//! OS, so the enum is `#[non_exhaustive]` to leave room for `DmaBuf` (Linux)
//! and `Host` (pinned host memory) in later phases.

/// A platform-native GPU/system buffer owned by HAL, borrowed by Burn.
///
/// # Lifetime
///
/// The handle is **borrowed**: the HAL tensor that produced it must outlive the
/// Burn tensor built from it. [`crate::preprocess::with_image_tensor`] scopes
/// tensor usage inside a closure that holds the owning HAL tensor alive.
#[non_exhaustive]
pub enum ForeignBuffer {
    /// A borrowed `IOSurfaceRef` (macOS / iOS). HAL retains ownership of the
    /// surface; Burn imports it via an `MTLBuffer` backed by the surface's
    /// allocation. Available when the `metal-iosurface` feature is enabled.
    #[cfg(any(target_os = "macos", target_os = "ios", docsrs))]
    IoSurface(*mut std::ffi::c_void),

    /// A Linux dma-buf file descriptor. HAL owns the underlying allocation;
    /// Burn imports it into a Vulkan/CUDA buffer via
    /// `wgpu::hal::vulkan::Buffer::from_raw_managed` (mirroring the existing
    /// cubecl-wgpu Vulkan foreign-buffer import at `backend/vulkan.rs`).
    ///
    /// On the HAL side, `Tensor::from_fd` produces a `DmaTensor` wrapping the
    /// dma-buf `OwnedFd`; on the burn side, `WgpuClientExt::import_foreign_buffer`
    /// (already implemented for a pre-built `wgpu::Buffer`) would be called after
    /// the Vulkan hal import creates the `wgpu::Buffer` from the fd.
    ///
    /// **Not yet implemented** — reserved for the Linux DMA-BUF → Vulkan/CUDA
    /// follow-up phase.
    #[cfg(target_os = "linux")]
    DmaBuf(std::os::fd::OwnedFd),

    /// Page-locked host memory suitable for a pinned `MTLBuffer`/`vkBuffer`.
    /// Used when the HAL convert target is `TensorMemory::Shm` (POSIX shared
    /// memory) and the backend supports importing pinned host memory directly
    /// (e.g. `MTLResourceOptions::StorageModeShared` on macOS, or
    /// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT` on Vulkan).
    ///
    /// **Not yet implemented** — reserved for a future pinned-memory path that
    /// avoids the `TensorData::from_bytes_vec` copy in the host fallback.
    #[allow(dead_code)]
    Host(*mut std::ffi::c_void, usize),
}

#[cfg(any(target_os = "macos", target_os = "ios", docsrs))]
impl ForeignBuffer {
    /// Returns the raw `IOSurfaceRef` if this buffer is an `IoSurface`.
    pub fn as_iosurface(&self) -> Option<*mut std::ffi::c_void> {
        match self {
            ForeignBuffer::IoSurface(p) => Some(*p),
            _ => None,
        }
    }

    /// Returns `true` if this buffer is backed by an `IOSurface`.
    pub fn is_iosurface(&self) -> bool {
        matches!(self, ForeignBuffer::IoSurface(_))
    }
}
