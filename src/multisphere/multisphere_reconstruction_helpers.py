import numpy as np

from .multisphere_datatypes import VoxelGrid
from .multisphere_voxel_processing import _apply_kernel_to_grid


def _append_sphere_table(
    sphere_table: np.ndarray,
    distance_field: np.ndarray,
    peaks: np.ndarray,
    min_radius_vox: int | None,
    max_spheres: int | None,
    ) -> np.ndarray:
    """
    Append spheres to the sphere table based on peak coordinates in a
    distance field.

    This function works both for the initial seeding step
    (sphere_table empty, peaks from the original distance field) and for
    later refinement steps (sphere_table non-empty, peaks from a
    modified/residual distance field).

    Parameters
    ----------
    sphere_table : np.ndarray
        Existing sphere table of shape (N, 4) with rows
        [x_vox, y_vox, z_vox, diameter_vox]. May be empty.
    distance_field : np.ndarray
        Distance field of the original voxel mask. Used to obtain
        radii at the peak positions.
    peaks : np.ndarray
        Peak coordinates of shape (M, 3), typically produced by
        ``peak_local_max``.
    min_radius_vox : int or None
        Minimum sphere radius in voxels. Peaks with smaller radii are
        ignored if specified.
    max_spheres : int or None
        Optional hard cap on total number of spheres. If None, no cap is
        applied here.

    Returns
    -------
    np.ndarray
        Updated sphere table (K, 4) with rows
        [x_vox, y_vox, z_vox, diameter_vox], sorted by diameter in
        descending order.
    """
    if peaks.size == 0:
        return sphere_table

    if distance_field.ndim != 3:
        raise ValueError(
            f"distance_field must be 3D, got shape {distance_field.shape}."
        )

    # Start from existing spheres (if any)
    if sphere_table.size == 0:
        spheres: list[list[float]] = []
    else:
        if sphere_table.shape[1] != 4:
            raise ValueError(
                "sphere_table must have shape (N, 4): [x, y, z, diameter_vox]."
            )
        spheres = sphere_table.tolist()

    initial_count = len(spheres)

    for coord in peaks:
        if max_spheres is not None and len(spheres) >= max_spheres:
            break

        coord_int = coord.astype(int)
        dist_val = float(distance_field[tuple(coord_int)])  # radius in voxels
        radius_vox = int(round(dist_val))

        if min_radius_vox is not None and radius_vox < min_radius_vox:
            # too small – skip this peak
            continue

        diameter_vox = 2 * radius_vox

        # Subvoxel-like center refinement: same logic as original
        offset = _shift_voxel_center(
            distance_field,
            coord_int,
            tolerance=0.25,
        )

        if np.any(offset == 0):
            # cannot consistently shift into an 8-voxel center → enforce odd diameter
            if diameter_vox % 2 == 0:
                diameter_vox += 1
            center = coord_int.astype(float)
        else:
            # shift by 0.5 along directions where neighbors match → enforce even diameter
            center = coord_int.astype(float) + 0.5 * offset
            if diameter_vox % 2 != 0:
                diameter_vox += 1

        spheres.append(
            [center[0], center[1], center[2], float(diameter_vox)]
        )

    if len(spheres) == initial_count:
        # no new spheres added
        return sphere_table

    sphere_table = np.asarray(spheres, dtype=float)
    # sort by diameter descending
    sphere_table = sphere_table[sphere_table[:, 3].argsort()[::-1]]
    return sphere_table



def _shift_voxel_center(
    distance_field: np.ndarray,
    center_vox: np.ndarray,
    tolerance: float = 0.25,
    ) -> np.ndarray:
    """
    Determine whether the center of a local distance-field maximum lies
    exactly on a voxel center or in between voxels (half-voxel shift).

    The function inspects the six axial neighbors (±x, ±y, ±z) and
    compares their distance-field values to the central value. If the
    difference is within `tolerance`, the center is considered to lie
    between voxels along that axis.

    Parameters
    ----------
    distance_field : np.ndarray
        3D distance field.
    center_vox : np.ndarray
        Integer voxel coordinate (x, y, z).
    tolerance : float, optional
        Maximum allowed deviation to accept a neighbor as equivalent.

    Returns
    -------
    np.ndarray
        Shift vector of shape (3,), with values in {-1, 0, +1}.
        A non-zero entry indicates a half-voxel shift along that axis.
    """

    if distance_field.ndim != 3:
        raise ValueError("distance_field must be a 3D array.")

    center_vox = np.asarray(center_vox, dtype=int)
    if center_vox.size != 3:
        raise ValueError("center_vox must be a 3-element coordinate.")

    cx, cy, cz = center_vox
    if not (
        0 <= cx < distance_field.shape[0]
        and 0 <= cy < distance_field.shape[1]
        and 0 <= cz < distance_field.shape[2]
    ):
        raise IndexError(
            f"center_vox {center_vox.tolist()} is outside the distance field."
        )

    central_value = distance_field[cx, cy, cz]
    shift = np.zeros(3, dtype=float)

    # axial neighbor checks
    neighbors = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]

    for dx, dy, dz in neighbors:
        nx, ny, nz = cx + dx, cy + dy, cz + dz
        if (
            0 <= nx < distance_field.shape[0]
            and 0 <= ny < distance_field.shape[1]
            and 0 <= nz < distance_field.shape[2]
        ):
            diff = abs(distance_field[nx, ny, nz] - central_value)
            if diff <= tolerance:
                if dx != 0:
                    shift[0] = dx
                elif dy != 0:
                    shift[1] = dy
                elif dz != 0:
                    shift[2] = dz

    return shift



def _compute_voxel_precision(
    target: VoxelGrid,
    reconstruction: VoxelGrid,
    ) -> float:
    """
    Compute coverage precision of the reconstructed voxel grid relative
    to the target voxel grid.

    Precision is defined as:
        precision = 1 - mismatch_fraction

    where mismatch_fraction is the fraction of target voxels that differ
    between target and reconstruction.

    Parameters
    ----------
    target : VoxelGrid
        Ground-truth voxel grid.
    reconstruction : VoxelGrid
        Reconstructed voxel grid.

    Returns
    -------
    float
        Precision in [0.0, 1.0].
    """
    target_data = target.data.astype(bool)
    recon_data = reconstruction.data.astype(bool)

    if target_data.shape != recon_data.shape:
        raise ValueError(
            "target and reconstruction must have the same shape, "
            f"got {target_data.shape} and {recon_data.shape}."
        )

    total_target = int(np.sum(target_data))
    if total_target == 0:
        # No target voxels: either perfectly empty or completely wrong.
        return 1.0 if int(np.sum(recon_data)) == 0 else 0.0

    # mismatch: XOR via abs difference
    mismatch = np.abs(target_data.astype(int) - recon_data.astype(int))
    mismatch_fraction = float(np.sum(mismatch)) / float(total_target)

    precision = 1.0 - mismatch_fraction
    # numerical safety
    if precision < 0.0:
        precision = 0.0
    elif precision > 1.0:
        precision = 1.0

    return precision



def _spheres_to_grid(
    sphere_table: np.ndarray,
    grid_shape: tuple[int, int, int],
    dtype: type = float,
    ) -> VoxelGrid:
    """
    Rasterize a sphere table into a VoxelGrid.

    Each row of sphere_table is interpreted as
        [x_vox, y_vox, z_vox, diameter_vox]

    and rendered as a solid sphere into a volume of the given shape.

    Parameters
    ----------
    sphere_table : np.ndarray
        Array of shape (N, 4) with voxel-space sphere parameters.
    grid_shape : tuple of int
        Shape (Nx, Ny, Nz) of the target volume.
    dtype : type, optional
        Data type of the voxel values (default: float).

    Returns
    -------
    VoxelGrid
        VoxelGrid with the rendered spheres. Overlapping spheres add up.
    """
    if len(grid_shape) != 3:
        raise ValueError(
            f"grid_shape must be a 3-tuple, got {grid_shape!r}."
        )

    if any(s <= 0 for s in grid_shape):
        raise ValueError(
            f"All entries of grid_shape must be positive, got {grid_shape!r}."
        )

    sphere_table = np.asarray(sphere_table, dtype=float)
    if sphere_table.size == 0:
        return VoxelGrid.empty(grid_shape, dtype=dtype)

    if sphere_table.ndim != 2 or sphere_table.shape[1] != 4:
        raise ValueError(
            "sphere_table must have shape (N, 4) with columns "
            "[x, y, z, diameter_vox], got shape "
            f"{sphere_table.shape}."
        )

    grid = VoxelGrid.empty(grid_shape, dtype=dtype)

    for x, y, z, diameter_vox in sphere_table:
        diameter_int = int(round(diameter_vox))
        if diameter_int <= 0:
            continue

        kernel_grid = VoxelGrid.sphere_kernel(diameter_int)
        kernel = kernel_grid.data
        center = np.array([x, y, z], dtype=float)

        _apply_kernel_to_grid(
            grid=grid,
            center=center,
            kernel=kernel,
            mode="add",
            scale=1.0,
        )

    return grid



def _residual_distance_field(
    original_distance: VoxelGrid,
    spheres_distance: VoxelGrid,
    ) -> VoxelGrid:
    """
    Compute the residual distance field from two distance transforms.

    D_res = max(D_orig - D_spheres, 0)

    Both inputs must be distance fields (NOT boolean masks). This
    function does NOT perform any distance transform internally.

    Parameters
    ----------
    original_distance : VoxelGrid
        Distance field of the original voxel mask.
    spheres_distance : VoxelGrid
        Distance field of the current sphere-based reconstruction.

    Returns
    -------
    VoxelGrid
        Residual distance field, same shape, voxel_size and origin as
        original_distance.

    Raises
    ------
    ValueError
        If shapes, voxel_size, or origin do not match, or if any input
        has boolean dtype.
    """
    orig_data = original_distance.data
    sph_data = spheres_distance.data

    # shape & metadata checks
    if orig_data.shape != sph_data.shape:
        raise ValueError(
            "original_distance and spheres_distance must have the same shape, "
            f"got {orig_data.shape} and {sph_data.shape}."
        )

    if original_distance.voxel_size != spheres_distance.voxel_size:
        raise ValueError(
            "original_distance and spheres_distance must have the same "
            f"voxel_size, got {original_distance.voxel_size!r} and "
            f"{spheres_distance.voxel_size!r}."
        )

    if not np.allclose(original_distance.origin, spheres_distance.origin):
        raise ValueError(
        "original_distance and spheres_distance must have the same origin, "
        f"got {original_distance.origin!r} and {spheres_distance.origin!r}."
    )

    # dtype sanity: reject boolean "distance fields"
    if orig_data.dtype == bool:
        raise ValueError(
            "original_distance.data has boolean dtype; expected a distance field, "
            "not a mask. Compute a distance transform first."
        )
    if sph_data.dtype == bool:
        raise ValueError(
            "spheres_distance.data has boolean dtype; expected a distance field, "
            "not a mask. Compute a distance transform first."
        )

    # residual = max(D_orig - D_spheres, 0)
    residual = orig_data - sph_data
    residual[residual < 0.0] = 0.0

    return VoxelGrid(
        data=residual,
        voxel_size=original_distance.voxel_size,
        origin=original_distance.origin,
    )




def _filter_peaks(
    peaks: np.ndarray,
    sphere_table: np.ndarray,
    min_center_distance_vox: int,
    template_grid: VoxelGrid,
    ) -> np.ndarray:
    """
    Remove peak candidates that are too close to existing sphere centers.

    - build a mask volume with a spherical kernel of diameter
      (min_center_distance_vox - 1) written at every existing center
      (rounded to voxel indices),
    - remove peaks that fall inside this mask.

    Parameters
    ----------
    peaks : np.ndarray
        Array of candidate peak coordinates of shape (M, 3). Typically
        produced by `peak_local_max(...)` on some distance field.
    sphere_table : np.ndarray
        Sphere table of shape (N, 4) with rows
        [x_vox, y_vox, z_vox, diameter_vox]. May be empty.
    min_center_distance_vox : int
        Minimum allowed distance between new centers and existing
        centers, in voxels.
    template_grid : VoxelGrid
        VoxelGrid providing shape, voxel_size and origin. Only the shape
        is actually used to build the mask, but metadata is preserved.

    Returns
    -------
    np.ndarray
        Filtered peaks of shape (K, 3), where K <= M. If there are no
        existing spheres or min_center_distance_vox <= 1, the input
        peaks are returned unchanged.
    """
    peaks = np.asarray(peaks, dtype=int)

    if peaks.size == 0:
        return peaks

    if sphere_table.size == 0:
        # no existing centers → no additional filtering
        return peaks

    volume_shape = template_grid.data.shape
    if len(volume_shape) != 3:
        raise ValueError(
            f"template_grid must be 3D, got shape {volume_shape}."
        )

    sphere_table = np.asarray(sphere_table, dtype=float)
    if sphere_table.ndim != 2 or sphere_table.shape[1] != 4:
        raise ValueError(
            "sphere_table must have shape (N, 4) with columns "
            "[x_vox, y_vox, z_vox, diameter_vox], got "
            f"{sphere_table.shape}."
        )

    # kernel diameter for exclusion region
    kernel_diameter = min_center_distance_vox - 1
    if kernel_diameter <= 0:
        return peaks

    # build mask grid
    mask_grid = VoxelGrid.empty(
        shape=volume_shape,
        dtype=float,
        voxel_size=template_grid.voxel_size,
        origin=template_grid.origin,
    )

    kernel_grid = VoxelGrid.sphere_kernel(kernel_diameter)
    kernel = kernel_grid.data

    # write exclusion kernel at each existing center
    for x, y, z, _ in sphere_table:
        center = np.round([x, y, z]).astype(float)
        _apply_kernel_to_grid(
            grid=mask_grid,
            center=center,
            kernel=kernel,
            mode="add",
            scale=1.0,
        )

    mask = mask_grid.data > 0

    # filter peaks that fall inside the mask
    filtered: list[list[int]] = []
    nx, ny, nz = volume_shape

    for coord in peaks:
        cx, cy, cz = coord
        if not (0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz):
            # out of bounds → drop
            continue
        if not mask[cx, cy, cz]:
            filtered.append([cx, cy, cz])

    if not filtered:
        return np.empty((0, 3), dtype=int)

    return np.asarray(filtered, dtype=int)
