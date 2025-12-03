import numpy as np
from skimage.feature import peak_local_max
import trimesh

from .multisphere_datatypes import VoxelGrid, SpherePack
from .multisphere_reconstruction_helpers import (  
    _append_sphere_table,
    _spheres_to_grid,
    _compute_voxel_precision,
    _residual_distance_field,
    _filter_peaks,
)
from .multisphere_utils import (print_progress_bar, 
    adjust_spheres_to_stl_boundary)
from .multisphere_io import mesh_to_voxel_grid

def multisphere_from_mesh(
    mesh: trimesh.Trimesh,
    div: int = 100,
    padding: int = 2,
    min_radius_vox: int | None = None,
    precision: float | None = None,
    min_center_distance_vox: int = 4,
    max_spheres: int | None = None,
    show_progress: bool = True,
    confine_mesh: bool = False,
    ) -> SpherePack:
    """
    Construct a multisphere representation directly from a triangle mesh.

    This is a thin convenience wrapper around:

    1. mesh_to_voxel_grid(...)
    2. multisphere_from_voxels(...)

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to approximate with spheres.
    div : int
        Resolution parameter for voxelization. The voxel size is
        min_AABB_length / div, where min_AABB_length is the shortest
        edge of the mesh AABB.
    padding : int, optional
        Number of empty voxels to pad on each side of the grid before
        reconstruction.
    min_radius_vox : int, optional
        Minimum sphere radius in voxels. See multisphere_from_voxels.
    precision : float, optional
        Target voxel precision in (0, 1]. See multisphere_from_voxels.
    min_center_distance_vox : int, optional
        Minimum allowed distance between sphere centers in voxels.
    max_spheres : int, optional
        Maximum number of spheres. See multisphere_from_voxels.
    show_progress : bool, optional
        Whether to print a progress bar during reconstruction.
    confine_mesh : bool, optional
        Whether to confine the multisphere representation to the mesh surface

    Returns
    -------
    SpherePack
        Sphere centers and radii in physical units, in the same
        coordinate system as the input mesh.
    """
    voxel_grid: VoxelGrid = mesh_to_voxel_grid(
        mesh=mesh,
        div=div,
        padding=padding,
    )

    sphere_pack = multisphere_from_voxels(
        voxel_grid=voxel_grid,
        min_radius_vox=min_radius_vox,
        precision=precision,
        min_center_distance_vox=min_center_distance_vox,
        max_spheres=max_spheres,
        show_progress=show_progress,
    )

    if confine_mesh:
        return adjust_spheres_to_stl_boundary(sphere_pack, mesh)
    else:
        return sphere_pack
        

def multisphere_from_voxels(
    voxel_grid: VoxelGrid,
    min_radius_vox: int | None = None,
    precision: float | None = None,
    min_center_distance_vox: int = 4,
    max_spheres: int | None = None,
    show_progress: bool = True,
) -> SpherePack:
    """
    Construct a multisphere representation from a 3D voxel grid.

    Parameters
    ----------
    voxel_grid : VoxelGrid
        Binary voxel grid describing the target geometry (True inside).
    min_radius_vox : int, optional
        Minimum sphere radius in voxels. If given, the algorithm stops
        once no new spheres with radius >= min_radius_vox can be added.
    precision : float, optional
        Target coverage precision in (0, 1]. If given, the algorithm
        stops once the voxel precision reaches or exceeds this value.
    min_center_distance_vox : int, optional
        Minimum allowed distance between sphere centers in voxels.
        Default is 4.
    max_spheres : int, optional
        Maximum number of spheres. If given, the algorithm stops once
        this number is reached.
    show_progress : bool, optional
        If True and max_spheres is not None, print a progress bar based
        on the current number of spheres.

    Returns
    -------
    SpherePack
        Sphere centers and radii in physical units (same coordinate
        system as `voxel_grid`).
    """
    # -----------------------------------------------------------
    # validate inputs
    # -----------------------------------------------------------
    if not isinstance(voxel_grid, VoxelGrid):
        raise TypeError(
            f"voxel_grid must be a VoxelGrid instance, got {type(voxel_grid)}."
        )

    if voxel_grid.data.ndim != 3:
        raise ValueError(
            f"voxel_grid.data must be 3D, got shape {voxel_grid.data.shape}."
        )

    if min_radius_vox is None and precision is None and max_spheres is None:
        raise ValueError(
            "At least one termination criterion must be specified: "
            "min_radius_vox, precision, or max_spheres."
        )

    if min_radius_vox is not None:
        if not isinstance(min_radius_vox, int) or min_radius_vox < 0:
            raise ValueError(
                f"min_radius_vox must be a non-negative integer, "
                f"got {min_radius_vox!r}."
            )

    if precision is not None:
        if not isinstance(precision, (int, float)) or not (0.0 < precision <= 1.0):
            raise ValueError(
                f"precision must be in the interval (0, 1], got {precision!r}."
            )

    if max_spheres is not None:
        if not isinstance(max_spheres, int) or max_spheres <= 0:
            raise ValueError(
                f"max_spheres must be a positive integer, got {max_spheres!r}."
            )

    if not isinstance(min_center_distance_vox, int) or min_center_distance_vox <= 0:
        raise ValueError(
            f"min_center_distance_vox must be a positive integer, "
            f"got {min_center_distance_vox!r}."
        )

    # -----------------------------------------------------------
    # basic data & early exit if voxel grid is empty
    # -----------------------------------------------------------
    target_mask = voxel_grid.data.astype(bool)

    if not np.any(target_mask):
        # empty target → no spheres
        return SpherePack(
            centers=np.zeros((0, 3), dtype=float),
            radii=np.zeros((0,), dtype=float),
        )

    # original distance field 
    original_distance = voxel_grid.distance_transform()
    distance_field = original_distance.data

    # sphere table: [x_vox, y_vox, z_vox, diameter_vox]
    sphere_table = np.empty((0, 4), dtype=float)

    # -----------------------------------------------------------
    # main loop
    # -----------------------------------------------------------
    while True:
        # -------------------------------------------------------
        # 1) hard cap on sphere count
        # -------------------------------------------------------
        if max_spheres is not None and sphere_table.shape[0] >= max_spheres:
            print(
                f"Stopping: reached maximum number of spheres "
                f"(max_spheres = {max_spheres})."
            )
            break

        # -------------------------------------------------------
        # 2) reconstruct current spheres, compute precision,
        #    compute residual distance
        # -------------------------------------------------------
        if sphere_table.shape[0] > 0:
            # rasterize spheres into voxel grid
            recon_counts = _spheres_to_grid(
                sphere_table=sphere_table,
                grid_shape=voxel_grid.shape,
                dtype=float,
            )

            recon_mask = VoxelGrid(
                data=(recon_counts.data > 0),
                voxel_size=voxel_grid.voxel_size,
                origin=voxel_grid.origin,
            )

            # precision termination (if requested)
            if precision is not None:
                voxel_precision = _compute_voxel_precision(
                    target=voxel_grid,
                    reconstruction=recon_mask,
                )
                if voxel_precision >= precision:
                    print(
                        "Stopping: desired precision reached "
                        f"(current = {voxel_precision:.4f}, "
                        f"target = {precision:.4f})."
                    )
                    break

            # distance transform of reconstruction
            spheres_distance = recon_mask.distance_transform()

            # residual distance field
            residual_distance = _residual_distance_field(
                original_distance=original_distance,
                spheres_distance=spheres_distance,
            )

            summed_field = distance_field + residual_distance.data
        else:
            # initial iteration: no spheres yet → use original distance field
            summed_field = distance_field

        if np.sum(summed_field) == 0.0:
            print(
                "Stopping: no remaining distance signal in summed field "
                "(residual field is zero everywhere)."
            )
            break

        # -------------------------------------------------------
        # 3) detect peaks in summed_field
        # -------------------------------------------------------
        peaks = peak_local_max(
            summed_field,
            min_distance=min_center_distance_vox,
        )

        if peaks.size == 0:
            print("Stopping: no local maxima detected in distance field.")
            break

        # enforce min distance to existing spheres (no-op if none exist)
        peaks = _filter_peaks(
            peaks=peaks,
            sphere_table=sphere_table,
            min_center_distance_vox=min_center_distance_vox,
            template_grid=voxel_grid,
        )

        if peaks.size == 0:
            print(
                "Stopping: no remaining peak positions after enforcing "
                f"min_center_distance_vox = {min_center_distance_vox}."
            )
            break

        # -------------------------------------------------------
        # 4) extend sphere table from peaks
        # -------------------------------------------------------
        previous_count = sphere_table.shape[0]

        sphere_table = _append_sphere_table(
            sphere_table=sphere_table,
            distance_field=distance_field,
            peaks=peaks,
            min_radius_vox=min_radius_vox,
            max_spheres=max_spheres,
        )

        # min-radius termination: if nothing new was added and we have a
        # radius cutoff, then all remaining peaks are too small
        if min_radius_vox is not None and sphere_table.shape[0] == previous_count:
            print(
                "Stopping: minimum radius in voxels reached; "
                "no further peaks with radius >= "
                f"min_radius_vox = {min_radius_vox}."
            )
            break

        # progress bar (only meaningful if max_spheres is given)
        if show_progress and max_spheres is not None:
            print_progress_bar(
                iteration=sphere_table.shape[0],
                total=max_spheres,
                prefix="Spheres",
                suffix="",
            )

    # -----------------------------------------------------------
    # convert sphere_table (voxel units) → SpherePack (physical units)
    # -----------------------------------------------------------
    if sphere_table.size == 0:
        return SpherePack(
            centers=np.zeros((0, 3), dtype=float),
            radii=np.zeros((0,), dtype=float),
        )

    centers_vox = sphere_table[:, :3]
    radii_vox = sphere_table[:, 3] / 2.0  # diameter → radius

    # handle voxel_size as scalar or length-3 array
    voxel_size = np.asarray(voxel_grid.voxel_size, dtype=float)
    if voxel_size.size == 1:
        voxel_size = np.repeat(voxel_size, 3)

    if not np.allclose(voxel_size, voxel_size[0]):
        raise ValueError(
            "Anisotropic voxel_size is not supported for spherical radii. "
            f"Got voxel_size = {voxel_size!r}."
        )

    # physical coordinates: origin + index * voxel_size
    centers_phys = voxel_grid.origin + centers_vox * voxel_size
    radii_phys = radii_vox * float(voxel_size[0])

    return SpherePack(
        centers=centers_phys,
        radii=radii_phys,
    )

    

