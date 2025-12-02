import warnings
import trimesh
import numpy as np
from scipy.spatial import cKDTree

from .multisphere_datatypes import SpherePack

def compute_dice_coefficient(
    mesh_1: trimesh.Trimesh,
    mesh_2: trimesh.Trimesh,
    engine: str = "manifold",
) -> float:
    """
    Compute the Dice coefficient between two meshes by constructing their
    boolean intersection internally.

    Dice = 2 * V_intersection / (V1 + V2)

    Parameters
    ----------
    mesh_1 : trimesh.Trimesh
        First input mesh.
    mesh_2 : trimesh.Trimesh
        Second input mesh.
    engine : str, optional
        Boolean engine passed to ``trimesh.boolean.intersection``, e.g.
        ``"manifold"``, ``"blender"``, or ``"auto"``.

    Returns
    -------
    float
        Dice similarity in percent, in the range [0, 100].

    Notes
    -----
    - This function assumes that meshes were already validated elsewhere
      (e.g. on load or creation). It does *not* re-check watertightness.
    - If the intersection fails or returns no volume, the function raises
      or returns 0.0, respectively.
    """

    if not isinstance(mesh_1, trimesh.Trimesh):
        raise TypeError(
            f"mesh_1 must be a trimesh.Trimesh, got {type(mesh_1).__name__}."
        )
    if not isinstance(mesh_2, trimesh.Trimesh):
        raise TypeError(
            f"mesh_2 must be a trimesh.Trimesh, got {type(mesh_2).__name__}."
        )

    # --- compute intersection internally ---
    try:
        mesh_inter = trimesh.boolean.intersection([mesh_1, mesh_2],
                                                  engine=engine)
    except Exception as exc:
        raise RuntimeError(
            f"Boolean intersection failed using engine='{engine}'. "
            "This is often caused by missing boolean backends "
            "(Manifold / Blender not installed or not on PATH) or by "
            "highly complex / degenerate meshes."
        ) from exc

    if mesh_inter is None:
        raise RuntimeError(
            f"Boolean intersection returned None using engine='{engine}'. "
            "This usually indicates that the backend failed internally."
        )

    # --- volumes (no heavy validation, just warnings) ---
    V1 = mesh_1.volume
    V2 = mesh_2.volume
    V_inter = mesh_inter.volume

    # Treat None as zero, but warn – meshes should already be validated upstream.
    if V1 is None:
        warnings.warn("Volume of mesh_1 is undefined (None). Treating as 0.",
                      RuntimeWarning)
        V1 = 0.0
    if V2 is None:
        warnings.warn("Volume of mesh_2 is undefined (None). Treating as 0.",
                      RuntimeWarning)
        V2 = 0.0
    if V_inter is None:
        warnings.warn(
            "Volume of intersection mesh is undefined (None). Treating as 0.",
            RuntimeWarning
        )
        V_inter = 0.0

    if V1 == 0:
        warnings.warn("Volume of mesh_1 is zero.", RuntimeWarning)
    if V2 == 0:
        warnings.warn("Volume of mesh_2 is zero.", RuntimeWarning)
    if V_inter == 0:
        warnings.warn("Volume of intersection is zero.", RuntimeWarning)

    if V1 + V2 == 0:
        # Nothing to compare – both are empty or invalid.
        return 0.0

    dice = (2.0 * V_inter) / (V1 + V2)
    return float(dice * 100.0)



def create_multisphere_mesh(
    positions: SpherePack | np.ndarray,
    radii: np.ndarray | None = None,
    resolution: int = 4,
    engine: str = "manifold",
) -> trimesh.Trimesh:
    """
    Create a triangle mesh from a multisphere representation using a
    Boolean union of individual spheres. Useful for visualization, 
    rendering and to compare the sphere representation with the original 
    mesh.

    This function uses ``trimesh.boolean.union`` under the hood. Boolean
    CSG is powerful but numerically fragile and depends on external
    backends:

    - ``engine="manifold"``:
        requires the Manifold C++ library / binary installed and
        configured so trimesh can find it.
    - ``engine="blender"``:
        requires Blender installed and available on your PATH.
    - ``engine="auto"``:
        lets trimesh pick an engine automatically.

    For large numbers of spheres, extremely overlapping configurations,
    or missing/misconfigured engines, the boolean union may fail or
    return invalid geometry.

    Parameters
    ----------
    positions : SpherePack or np.ndarray
        Either:
          * a SpherePack instance, or
          * a numpy array of shape (N, 3) with sphere centers.
    radii : np.ndarray, optional
        1D array of shape (N,) with sphere radii. Must be provided if
        `positions` is not a SpherePack. Ignored when `positions` is a
        SpherePack.
    resolution : int, optional
        Subdivision level for the icosphere used to approximate each
        sphere. Higher values give smoother surfaces but increase
        polygon count and make boolean operations slower and more
        fragile. Typical range: 2–5.
    engine : str, optional
        Name of the trimesh boolean engine to use. Common options:
        ``"manifold"``, ``"blender"``, ``"auto"``.

    Returns
    -------
    trimesh.Trimesh
        A single mesh approximating the union of all spheres.

    Raises
    ------
    ValueError
        If inputs are inconsistent (shapes, radii, resolution).
    RuntimeError
        If the boolean union fails or produces an invalid mesh.

    Notes
    -----
    If you see errors about boolean union failure or an empty/invalid
    mesh, try:
    - Reducing ``resolution``.
    - Reducing the number of spheres (e.g. remove very small ones).
    - Trying a different ``engine`` (``"auto"``, ``"blender"``, etc.).
    - Verifying that the chosen engine (Manifold, Blender, OpenSCAD)
      is installed and on your PATH.
    """

    # --- unpack SpherePack or (positions, radii) pair ---
    if isinstance(positions, SpherePack):
        centers = positions.centers
        radii_arr = positions.radii
    else:
        centers = np.asarray(positions, dtype=float)
        if radii is None:
            raise ValueError(
                "radii must be provided when 'positions' is not a SpherePack."
            )
        radii_arr = np.asarray(radii, dtype=float).ravel()

    # --- basic shape checks ---
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(
            f"'positions' must have shape (N, 3), got {centers.shape}."
        )

    if radii_arr.ndim != 1:
        raise ValueError(
            f"'radii' must be a 1D array, got shape {radii_arr.shape}."
        )

    if centers.shape[0] != radii_arr.size:
        raise ValueError(
            f"Number of centers ({centers.shape[0]}) and radii "
            f"({radii_arr.size}) does not match."
        )

    if centers.shape[0] == 0:
        raise ValueError("No spheres provided (N == 0).")

    if np.any(radii_arr <= 0.0):
        raise ValueError("All radii must be strictly positive.")

    if not isinstance(resolution, int) or resolution < 0:
        raise ValueError(
            f"'resolution' must be a non-negative integer, got {resolution!r}."
        )

    # --- create individual sphere meshes ---
    sphere_meshes: list[trimesh.Trimesh] = []
    for center, radius in zip(centers, radii_arr):
        sphere = trimesh.creation.icosphere(
            subdivisions=resolution,
            radius=float(radius),
        )
        sphere.apply_translation(center)
        sphere_meshes.append(sphere)

    # --- perform boolean union ---
    try:
        merged = trimesh.boolean.union(sphere_meshes, engine=engine)
    except Exception as exc:
        raise RuntimeError(
            f"Boolean union failed using engine='{engine}'. "
            "This is common with missing boolean backends (e.g. Manifold, "
            "Blender not installed or not on PATH) or with very "
            "large / heavily overlapping sphere packs.\n"
            "Try reducing 'resolution', using fewer spheres, or switching "
            "to engine='auto' or another engine."
        ) from exc

    if merged is None:
        raise RuntimeError(
            f"Boolean union returned None using engine='{engine}'. "
            "This usually indicates that the backend failed internally. "
            "Check that the engine is installed and configured correctly."
        )

    # --- handle Scenes and validate mesh ---
    if isinstance(merged, trimesh.Scene):
        try:
            merged = merged.dump(concatenate=True)
        except Exception as exc:
            raise RuntimeError(
                "Boolean union produced a Scene that could not be merged into "
                "a single Trimesh."
            ) from exc

    if not isinstance(merged, trimesh.Trimesh):
        raise RuntimeError(
            f"Unexpected result type from boolean union: "
            f"{type(merged).__name__}."
        )

    if merged.vertices.size == 0 or merged.faces.size == 0:
        raise RuntimeError(
            "Merged mesh is empty (no vertices or faces). "
            "Boolean operation likely failed."
        )

    # Optional cleanup – we *try* to improve watertightness, but we don't
    # guarantee it; caller can perform further validation if needed.
    if not merged.is_watertight:
        merged.fill_holes()
        merged.update_faces(merged.nondegenerate_faces())
        merged.remove_unreferenced_vertices()
        # merged.process(validate=True)  # can be very slow; left to the user

    return merged



def adjust_spheres_to_stl_boundary(
    sphere_pack: SpherePack,
    mesh: trimesh.Trimesh
) -> SpherePack:
    """
    Shrink spheres whose centers are closer to the mesh surface than
    their radius, so that they lie fully inside the STL mesh.

    This is a geometric post-processing step intended to fix spheres
    that slightly protrude through the boundary of the target mesh.
    The adjustment uses the distance to the nearest mesh vertex as a
    crude proxy for the distance to the surface.

    Parameters
    ----------
    sphere_pack : SpherePack
        Spheres to be adjusted. This object is NOT modified in-place;
        a new SpherePack is returned.
    mesh : trimesh.Trimesh
        Target mesh that defines the boundary. The sphere radii will
        be reduced so that sphere centers are at least their radius
        away from the nearest vertex of this mesh.

    Returns
    -------
    SpherePack
        New SpherePack with radii adjusted where necessary.

    Notes
    -----
    - This is an approximation: distance is computed to the nearest
      mesh vertex, not the true continuous surface.
    - For highly irregular meshes or very coarse tessellations, this
      may under- or over-shrink some spheres.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            f"mesh must be a trimesh.Trimesh, got {type(mesh).__name__}."
        )

    if mesh.vertices.size == 0:
        raise ValueError("Mesh has no vertices; cannot adjust spheres.")

    centers = sphere_pack.centers
    radii = sphere_pack.radii.copy()

    if centers.shape[0] == 0:
        return SpherePack(centers=centers.copy(), radii=radii)

    # Build KDTree on mesh vertices
    mesh_tree = cKDTree(mesh.vertices)

    # Query nearest vertex distance for each sphere center
    distances, _ = mesh_tree.query(centers)

    # If distance from center to nearest vertex is smaller than radius,
    # shrink the radius so it does not extend beyond that vertex distance.
    mask = distances < radii
    if np.any(mask):
        radii[mask] = distances[mask]

    return SpherePack(centers=centers.copy(), radii=radii)




def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 40,
    fill: str = "█",
    print_end: str = "\r",
    ) -> None:
    """
    Print a terminal progress bar.

    Parameters
    ----------
    iteration : int
        Current iteration count.
    total : int
        Total number of iterations.
    prefix : str, optional
        Prefix string.
    suffix : str, optional
        Suffix string.
    decimals : int, optional
        Number of decimals for the percentage.
    length : int, optional
        Length of the progress bar in characters.
    fill : str, optional
        Bar fill character.
    print_end : str, optional
        End character (e.g. '\\r' or '\\n').
    """
    if total <= 0:
        raise ValueError("total must be a positive integer.")

    if iteration < 0:
        iteration = 0
    if iteration > total:
        iteration = total

    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)

    print(
        f"\r{prefix} |{bar}| {percent}% {suffix}",
        end=print_end,
        flush=True,
    )

    if iteration == total:
        print()


