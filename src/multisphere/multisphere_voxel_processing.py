import numpy as np

from .multisphere_datatypes import VoxelGrid


def _apply_kernel_to_grid(
    grid: VoxelGrid,
    center,
    kernel: np.ndarray,
    mode: str = "add",
    scale: float = 1.0,
    ) -> np.ndarray:
    """
    Apply a kernel to a VoxelGrid at a specified center.

    This function supports 2D and 3D kernels and handles boundary clipping.

    Parameters
    ----------
    grid : VoxelGrid
        Target voxel grid whose data will be modified in-place.
    center : array-like of float
        Center position where the kernel should be applied. For odd-sized
        kernels, center must be integer coordinates. For even-sized kernels,
        center coordinates must end in .5 (between voxels).
    kernel : np.ndarray
        2D or 3D kernel array (binary or float).
    mode : {"add", "subtract", "zero"}, optional
        How to apply the kernel:
        - "add"      : data += kernel * scale
        - "subtract" : data -= kernel * scale
        - "zero"     : data[...] = 0 where kernel != 0
    scale : float, optional
        Scale factor for "add" / "subtract" modes.

    Returns
    -------
    np.ndarray
        The modified underlying ndarray (same as grid.data).

    Raises
    ------
    ValueError
        On dimension mismatch, invalid center vs kernel shape, or
        unsupported kernel dimensionality.
    """
    volume = grid.data

    kernel = np.asarray(kernel)
    center = np.asarray(center, dtype=float).ravel()

    kernel_dim = kernel.ndim
    volume_dim = volume.ndim

    if kernel_dim not in (2, 3):
        raise ValueError(
            f"Kernel must be 2D or 3D, got {kernel_dim}D."
        )
    if volume_dim != kernel_dim:
        raise ValueError(
            f"Kernel dimension ({kernel_dim}D) and volume dimension "
            f"({volume_dim}D) do not match."
        )

    if center.size != kernel_dim:
        raise ValueError(
            f"Center must have length {kernel_dim}, got {center.size}."
        )

    kernel_shape = np.array(kernel.shape, dtype=int)
    volume_shape = np.array(volume.shape, dtype=int)

    # even kernel → center coords must end with .5
    if np.all(kernel_shape % 2 == 0):
        if np.any(np.mod(center, 1.0) != 0.5):
            raise ValueError(
                "Center coordinates must end with .5 for even-sized kernels. "
                f"Got center={center}, kernel_shape={kernel_shape}."
            )
    # odd kernel → center coords must be integers
    elif np.all(kernel_shape % 2 == 1):
        if np.any(center != np.round(center)):
            raise ValueError(
                "Center coordinates must be integer for odd-sized kernels. "
                f"Got center={center}, kernel_shape={kernel_shape}."
            )
    else:
        raise ValueError(
            f"Asymmetric kernels are not supported. kernel_shape={kernel_shape}"
        )

    # lower-left-front and top-right-rear indices in volume
    llf_volume = np.zeros(kernel_dim, dtype=int)
    trr_volume = volume_shape - 1

    # desired kernel placement in volume indices
    half = (kernel_shape - 1) / 2.0
    llf_kernel = (center - half).astype(int)
    trr_kernel = (center + half).astype(int)

    # how far outside the volume are we?
    llf_exc = (llf_kernel - llf_volume).astype(int)
    trr_exc = (trr_volume - trr_kernel).astype(int)

    # clip kernel and adjust corners (handle boundaries)
    k = kernel  # alias

    if llf_exc[0] < 0:
        k = k[-llf_exc[0]:, ...]
        llf_kernel[0] = llf_volume[0]
    if llf_exc[1] < 0:
        k = k[:, -llf_exc[1]:, ...]
        llf_kernel[1] = llf_volume[1]
    if kernel_dim == 3 and llf_exc[2] < 0:
        k = k[:, :, -llf_exc[2]:]
        llf_kernel[2] = llf_volume[2]

    if trr_exc[0] < 0:
        k = k[: kernel_shape[0] + trr_exc[0], ...]
        trr_kernel[0] = trr_volume[0]
    if trr_exc[1] < 0:
        k = k[:, : kernel_shape[1] + trr_exc[1], ...]
        trr_kernel[1] = trr_volume[1]
    if kernel_dim == 3 and trr_exc[2] < 0:
        k = k[:, :, : kernel_shape[2] + trr_exc[2]]
        trr_kernel[2] = trr_volume[2]

    # now apply kernel to the overlapping region
    if kernel_dim == 3:
        vol_slice = volume[
            llf_kernel[0] : trr_kernel[0] + 1,
            llf_kernel[1] : trr_kernel[1] + 1,
            llf_kernel[2] : trr_kernel[2] + 1,
        ]
    else:
        vol_slice = volume[
            llf_kernel[0] : trr_kernel[0] + 1,
            llf_kernel[1] : trr_kernel[1] + 1,
        ]

    if mode == "add":
        vol_slice += k * float(scale)
    elif mode == "subtract":
        vol_slice -= k * float(scale)
    elif mode == "zero":
        vol_slice[k != 0] = 0
    else:
        raise ValueError(
            f"Invalid mode '{mode}'. Use 'add', 'subtract', or 'zero'."
        )

    return volume




