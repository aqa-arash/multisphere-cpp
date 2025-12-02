from __future__ import annotations

import numpy as np
import trimesh
import os

from .multisphere_datatypes import SpherePack, VoxelGrid
from .multisphere_utils import create_multisphere_mesh


# ---------- mesh input ----------

def load_mesh_from_stl(path: str) -> trimesh.Trimesh:
    """
    Load an STL mesh using trimesh 

    Parameters
    ----------
    path : str
        Path to the STL file.

    Returns
    -------
    trimesh.Trimesh
        Loaded triangular mesh.

    Raises
    ------
    TypeError
        If `path` is not a string.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not an STL or cannot be read as a valid mesh.
    RuntimeError
        If the loaded mesh has zero volume or invalid geometry.
    """

    # --- existence check ---
    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Mesh file not found: '{path}'")

    # --- extension check ---
    if not path.lower().endswith(".stl"):
        raise ValueError(
            f"File does not have .stl extension: '{path}'. "
            "Only STL input is supported here."
        )

    # --- attempt to load ---
    try:
        mesh = trimesh.load_mesh(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read STL file '{path}'. "
            f"Original error: {exc}"
        ) from exc

    # --- trimesh sometimes returns a Scene; handle that ---
    if isinstance(mesh, trimesh.Scene):
        # Try to convert to a single mesh
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            raise ValueError(
                f"STL file '{path}' contains multiple bodies and "
                "could not be converted to a single mesh."
            )

    # --- validate mesh contents ---
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"Loaded object is not a Trimesh instance: {type(mesh).__name__}"
        )

    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise RuntimeError(
            f"Mesh '{path}' contains no vertices or faces."
        )

    if mesh.volume is None or mesh.volume <= 0.0:
        raise RuntimeError(
            f"Mesh '{path}' has zero or undefined volume. "
            "This usually indicates a non-watertight STL."
        )

    return mesh

# ---------- voxel input (boolean matrices) ----------

def load_voxels_from_npy(
    path: str,
    voxel_size: float = 1.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> VoxelGrid:
    """
    Load a 3D voxel volume from a .npy file and return it as a VoxelGrid.

    The file must contain a 3D numpy array which is either:
    - of dtype bool, or
    - numeric with values only in {0, 1}.

    Parameters
    ----------
    path : str
        Path to the .npy file.
    voxel_size : float, optional
        Physical size of one voxel edge.
    origin : tuple of float, optional
        World-space origin of the voxel grid.

    Returns
    -------
    np.ndarray
        Boolean array of shape (Nx, Ny, Nz).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not .npy, or the loaded array is not 3D,
        or contains values other than 0 and 1.
    """

    # --- existence check ---
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Voxel file not found: '{path}'")

    # --- extension check ---
    if not path.lower().endswith(".npy"):
        raise ValueError(
            f"File does not have .npy extension: '{path}'. "
            "This function only loads .npy voxel volumes."
        )

    # --- load array ---
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"Failed to load .npy file '{path}'. Error: {exc}"
        ) from exc

    if not isinstance(arr, np.ndarray):
        raise ValueError(
            f"Loaded object from '{path}' is not a numpy array."
        )

    # --- ensure boolean or 0/1 only, and wrap in VoxelGrid ---
    if arr.dtype == bool:
        arr_out = arr
    else:
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(
                f"Voxel array in '{path}' has non-numeric dtype "
                f"{arr.dtype}; expected bool or numeric 0/1."
            )

        unique_vals = np.unique(arr)
        allowed = {0, 1, 0.0, 1.0}
        if not set(unique_vals.tolist()).issubset(allowed):
            raise ValueError(
                "Voxel array in '{path}' contains values other than "
                f"0 and 1: {unique_vals}."
            )

        arr_out = arr.astype(bool)

    return VoxelGrid(
        data=arr_out,
        voxel_size=float(voxel_size),
        origin=np.asarray(origin, dtype=float),
    )



# ---------- mesh -> voxel grid ----------

def mesh_to_voxel_grid(
    mesh: trimesh.Trimesh,
    div: int,
    padding: int = 2,
    ) -> VoxelGrid:
    """
    Convert a triangle mesh to a padded boolean voxel grid.

    The mesh is first recentered to the centroid of its axis-aligned
    bounding box (AABB). The voxel size is chosen as

        voxel_size = min_AABB_length / div

    where min_AABB_length is the shortest edge of the AABB. The mesh is
    then voxelized using trimesh's ``mesh.voxelized(...).fill()``.
    Finally, the voxel matrix is padded with a constant number of empty
    voxels on all sides.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to voxelize. The mesh is not modified in-place; a
        copy is used internally.
    div : int
        Resolution parameter. Larger values lead to smaller voxels
        (finer grid). The voxel size is ``min_AABB_length / div``.
    padding : int, optional
        Number of empty voxels to pad on each side of the grid.

    Returns
    -------
    VoxelGrid
        VoxelGrid with boolean ``data``, the computed ``voxel_size`` and
        ``origin`` equal to the original AABB centroid of the mesh
        (before recentering).

    Raises
    ------
    TypeError
        If ``mesh`` is not a ``trimesh.Trimesh``.
    ValueError
        If the mesh has no vertices, the AABB is degenerate, or
        ``div <= 0`` or ``padding < 0``.
    RuntimeError
        If voxelization fails.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            f"mesh must be a trimesh.Trimesh, got {type(mesh).__name__}."
        )

    if mesh.vertices.size == 0:
        raise ValueError("Mesh has no vertices; cannot voxelize.")

    if div <= 0:
        raise ValueError(f"div must be a positive integer, got {div!r}.")

    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding!r}.")

    # Work on a copy so we don't mutate the caller's mesh
    mesh_copy = mesh.copy()
    vertices = mesh_copy.vertices

    # AABB extremes
    min_x, min_y, min_z = np.min(vertices[:, 0]), np.min(vertices[:, 1]), np.min(vertices[:, 2])
    max_x, max_y, max_z = np.max(vertices[:, 0]), np.max(vertices[:, 1]), np.max(vertices[:, 2])


    # Shortest AABB edge for voxel size
    extents = np.array(
        [abs(max_x - min_x), abs(max_y - min_y), abs(max_z - min_z)],
        dtype=float,
    )
    min_aabb = np.min(extents)
    if min_aabb <= 0.0:
        raise ValueError(
            "Degenerate AABB (zero extent along at least one axis); "
            "cannot determine voxel size."
        )

    voxel_size = float(min_aabb) / float(div)

    # Voxelization
    try:
        voxelized = mesh_copy.voxelized(
            pitch=voxel_size,
            method="subdivide",
            max_iter=30,
        ).fill()
    except Exception as exc:
        raise RuntimeError(
            "Voxelization failed. This can happen for very thin or "
            "degenerate meshes, or for extreme resolution settings."
        ) from exc

    matrix = np.array(voxelized.matrix, dtype=bool)

    # Trimesh's VoxelGrid has a transform mapping index -> world coordinates.
    # world_coord = transform @ [i, j, k, 1]^T
    T = np.asarray(voxelized.transform, dtype=float)
    if T.shape != (4, 4):
        raise RuntimeError(
            f"Unexpected voxelized.transform shape: {T.shape}, "
            "expected (4, 4)."
        )

    # World-space coordinate of voxel index (0, 0, 0)
    origin_world = (T @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]

    # Sanity: assume axis-aligned, isotropic grid
    # (this is what trimesh does for standard voxelization)
    # We don't enforce it here, but if someone feeds rotated meshes /
    # transforms, your reconstruction algorithm will *not* handle that.

    # Padding: if we add 'padding' empty voxels in front of each axis,
    # index (0, 0, 0) of the *padded* grid corresponds to index
    # (−padding, −padding, −padding) of the original grid.
    if padding > 0:
        matrix = np.pad(
            matrix,
            pad_width=((padding, padding),
                       (padding, padding),
                       (padding, padding)),
            mode="constant",
            constant_values=False,
        )
        origin_world = origin_world - padding * voxel_size * np.array(
            [1.0, 1.0, 1.0], dtype=float
        )

    return VoxelGrid(
        data=matrix,
        voxel_size=voxel_size,
        origin=origin_world,
    )


# ---------- multisphere -> mesh ----------

def export_sphere_pack_to_stl(
    sphere_pack: SpherePack,
    path: str,
    resolution: int = 4,
    engine: str = "manifold",
) -> None:
    """
    Export a multisphere SpherePack as an STL mesh file.

    This is a convenience wrapper that:
      1. Calls ``create_multisphere_mesh`` to perform a Boolean union of
         all spheres in the SpherePack.
      2. Exports the resulting mesh to an STL file using trimesh.

    Parameters
    ----------
    sphere_pack : SpherePack
        Collection of spheres to be converted into a mesh.
    path : str
        Output file path for the STL mesh. Must end with ``.stl``.
        The parent directory must already exist.
    resolution : int, optional
        Subdivision level for the icosphere used to approximate each
        sphere. Higher values -> smoother but heavier meshes.
    engine : str, optional
        Boolean engine passed through to ``create_multisphere_mesh``.
        Typical values: ``"manifold"``, ``"blender"``, ``"scad"``,
        ``"auto"``.
    """

    

    if not str(path).lower().endswith(".stl"):
        raise ValueError(
            f"Output path '{path}' must end with '.stl'."
        )

    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Directory for STL export does not exist: '{directory}'"
        )

    # No further checks since create_multisphere_mesh already validates input
    # and handles errors
    mesh = create_multisphere_mesh(
        sphere_pack,
        resolution=resolution,
        engine=engine,
    )

    # This is a simple export. No wrapping needed.
    mesh.export(path)


# ---------- multisphere -> CSV / VTK ----------

def export_sphere_pack_to_csv(
    sphere_pack: SpherePack,
    path: str,) -> None:
    """
    Export a SpherePack to a CSV file with one row per sphere.

    The CSV layout is:

        x,y,z,radius

    where x, y, z are the center coordinates and radius is the sphere
    radius, all in the same units as stored in the SpherePack.

    Parameters
    ----------
    sphere_pack : SpherePack
        Collection of spheres to export.
    path : str
        Output CSV file path. Must end with '.csv'. The parent directory
        must already exist.

    Raises
    ------
    ValueError
        If the path does not end with '.csv'.
    FileNotFoundError
        If the parent directory does not exist.
    """
    # --- extension check ---
    if not str(path).lower().endswith(".csv"):
        raise ValueError(
            f"Output path '{path}' must end with '.csv' for CSV export."
        )

    # --- directory check ---
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Directory for CSV export does not exist: '{directory}'"
        )

    # --- build data array (N, 4): x, y, z, radius ---
    centers = sphere_pack.centers  # already validated in SpherePack
    radii = sphere_pack.radii      # shape (N,)

    data = np.column_stack((centers, radii))

    # --- write CSV ---
    # header without '#' because comments='' (pure CSV)
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="x,y,z,radius",
        comments="",
    )


def export_sphere_pack_to_vtk(
    sphere_pack: SpherePack,
    path: str
) -> None:
    """
    Export a SpherePack as a legacy VTK POLYDATA file containing sphere
    centers as points and radii as a point scalar.

    The file contains:
      - POINTS: sphere centers
      - VERTICES: one vertex cell per point
      - POINT_DATA: scalar array 'radius'

    This is *not* a triangulated sphere mesh; it is a point cloud with
    radius values. Use a glyph filter (e.g. in ParaView) to render
    spheres.

    Parameters
    ----------
    sphere_pack : SpherePack
        Collection of spheres to export.
    path : str
        Output VTK file path. Must end with '.vtk'. The parent directory
        must already exist.

    Raises
    ------
    ValueError
        If the path does not end with '.vtk'.
    FileNotFoundError
        If the parent directory does not exist.
    """
    # --- extension check ---
    if not str(path).lower().endswith(".vtk"):
        raise ValueError(
            f"Output path '{path}' must end with '.vtk' for VTK export."
        )

    # --- directory check ---
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Directory for VTK export does not exist: '{directory}'"
        )

    centers = sphere_pack.centers  # (N, 3)
    radii = sphere_pack.radii      # (N,)

    n = centers.shape[0]

    # safety: if there are no spheres, still write a valid VTK with 0 points
    with open(path, "w", encoding="utf-8") as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Multisphere sphere centers\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # POINTS section
        f.write(f"POINTS {n} float\n")
        if n > 0:
            for x, y, z in centers:
                f.write(f"{float(x)} {float(y)} {float(z)}\n")

        # VERTICES section: one vertex cell per point
        # Format: VERTICES N <N * (1 + 1)>  1 i0  1 i1  ...
        f.write(f"VERTICES {n} {n * 2}\n")
        for i in range(n):
            f.write(f"1 {i}\n")

        # POINT_DATA section: radii
        f.write(f"POINT_DATA {n}\n")
        f.write("SCALARS radius float 1\n")
        f.write("LOOKUP_TABLE default\n")
        if n > 0:
            for r in radii:
                f.write(f"{float(r)}\n")
