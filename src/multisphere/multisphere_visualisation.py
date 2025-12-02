import numpy as np
import trimesh
import pyvista as pv

from .multisphere_datatypes import SpherePack

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _require_pyvista():
    """
    Import pyvista lazily and raise a clear error if it is not installed.
    """
    try:
        import pyvista as pv  
    except ImportError as exc:
        raise ImportError(
            "PyVista is required for visualization but is not installed.\n"
            "Install it with:\n"
            "    pip install pyvista\n"
            "or install the visualization extra for this package, e.g.:\n"
            "    pip install multisphere[viz]\n"
        ) from exc
    return pv


def _get_colormap(cmap):
    """
    Return a callable colormap function f(t) -> RGBA, where t in [0, 1].

    Parameters
    ----------
    cmap : str or callable or None
        - str: name of a Matplotlib colormap (e.g. 'viridis', 'jet').
        - callable: function mapping float in [0, 1] to RGBA or RGB.
        - None: uses a reasonable default colormap.

    Returns
    -------
    callable
        Function mapping scalar in [0, 1] to RGBA.
    """
    if cmap is None:
        cmap = "viridis"

    if callable(cmap):
        return cmap

    if isinstance(cmap, str):
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Matplotlib is required to use colormap names in "
                "visualization functions.\n"
                "Install it with:\n"
                "    pip install matplotlib\n"
                "or pass a custom colormap function instead of a string."
            ) from exc

        return plt.get_cmap(cmap)

    raise TypeError(
        "cmap must be a colormap name (str), a callable, or None."
    )

# ---------------------------------------------------------------------------
# SpherePack visualization
# ---------------------------------------------------------------------------

def plot_sphere_pack(
    sphere_pack: SpherePack,
    opacity: float = 1.0,
    phi_res: int = 50,
    theta_res: int = 50,
    background: str = "white",
    cmap=None,
    show_axes: bool = True,
):
    """
    Visualize a SpherePack using PyVista.

    Each sphere is rendered as an individual PyVista Sphere mesh.
    Colors are assigned from a user-provided colormap.

    Parameters
    ----------
    sphere_pack : SpherePack
        Collection of spheres to visualize.
    opacity : float, optional
        Sphere opacity in [0, 1].
    phi_res : int, optional
        Longitudinal resolution of sphere meshes.
    theta_res : int, optional
        Latitudinal resolution of sphere meshes.
    background : str, optional
        Background color of the rendering window.
    cmap : str, callable, or None, optional
        - None: use default colormap ('viridis')
        - str: name of matplotlib colormap (requires matplotlib)
        - callable: f(t)->RGBA
    show_axes : bool, optional
        If True, display axes in the scene.

    Raises
    ------
    ValueError
        For invalid parameters.
    ImportError
        If PyVista (or Matplotlib for named colormaps) is missing.
    """
    pv = _require_pyvista()
    colormap = _get_colormap(cmap)

    centers = sphere_pack.centers
    radii = sphere_pack.radii
    n = sphere_pack.num_spheres

    if n == 0:
        raise ValueError("SpherePack is empty: no spheres to visualize.")

    if not (0.0 <= opacity <= 1.0):
        raise ValueError("opacity must be within [0, 1].")

    if phi_res < 3 or theta_res < 3:
        raise ValueError("phi_res and theta_res must be >= 3.")

    # Setup PyVista scene
    plotter = pv.Plotter()
    plotter.set_background(background)

    # Distribute colors: t in [0, 1]
    denom = max(n - 1, 1)

    for i, (center, radius) in enumerate(zip(centers, radii)):
        if radius <= 0:
            raise ValueError("Sphere radius must be > 0.")

        center = np.asarray(center, dtype=float).ravel()
        if center.size != 3:
            raise ValueError("Sphere center must be a 3D coordinate.")

        t = i / denom
        rgba = colormap(t)
        color = rgba[:3] if len(rgba) >= 3 else rgba

        sphere = pv.Sphere(
            radius=float(radius),
            center=center,
            phi_resolution=int(phi_res),
            theta_resolution=int(theta_res),
        )
        plotter.add_mesh(
            sphere,
            color=color,
            opacity=float(opacity),
        )

    if show_axes:
        plotter.show_axes()

    plotter.show(interactive=True)


# ---------------------------------------------------------------------------
# Mesh visualization
# ---------------------------------------------------------------------------

def trimesh_to_pv(mesh: trimesh.Trimesh):
    """
    Convert a trimesh.Trimesh to pyvista.PolyData.

    PyVista expects faces as a flat array with leading counts per face:
        [3, i, j, k, 3, i, j, k, ...]

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Triangle mesh to convert.

    Returns
    -------
    pyvista.PolyData
        PolyData object suitable for visualization with PyVista.

    Raises
    ------
    ValueError
        If the mesh has no faces.
    """

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            f"mesh must be a trimesh.Trimesh, got {type(mesh).__name__}."
        )

    if mesh.faces is None or mesh.faces.size == 0:
        raise ValueError("Input mesh has no triangular faces.")

    faces = mesh.faces
    counts = np.full((faces.shape[0], 1), 3, dtype=np.int64)
    faces_pv = np.hstack((counts, faces)).ravel()

    return pv.PolyData(mesh.vertices, faces_pv)


def plot_mesh(
    mesh,
    opacity: float = 1.0,
    color: str = "lightgray",
    show_edges: bool = False,
    edge_color: str = "black",
    smooth_shading: bool = True,
    background: str = "white",
    show_axes: bool = True,
):
    """
    Visualize a triangle mesh using PyVista.

    Parameters
    ----------
    mesh : trimesh.Trimesh or str
        Either a trimesh.Trimesh instance or a path to a mesh file that
        ``trimesh.load_mesh`` can read.
    opacity : float, optional
        Mesh opacity in [0, 1].
    color : str or tuple, optional
        Mesh color.
    show_edges : bool, optional
        If True, draw mesh edges.
    edge_color : str or tuple, optional
        Color of the mesh edges (only used if show_edges is True).
    smooth_shading : bool, optional
        If True, enable smooth shading.
    background : str, optional
        Plotter background color.
    show_axes : bool, optional
        If True, show coordinate axes.

    Returns
    -------
    pyvista.Plotter
        The plotter instance, in case the caller wants to further
        customize the view.

    Raises
    ------
    ImportError
        If pyvista is not installed.
    """
    pv = _require_pyvista()

    if isinstance(mesh, str):
        # Lazy import here to avoid forcing trimesh as a hard runtime
        # dependency for users who never call this.
        loaded = trimesh.load_mesh(mesh)
        if isinstance(loaded, trimesh.Scene):
            # Try to combine into a single mesh
            loaded = loaded.dump(concatenate=True)
        mesh = loaded

    poly = trimesh_to_pv(mesh)

    plotter = pv.Plotter()
    plotter.set_background(background)

    plotter.add_mesh(
        poly,
        color=color,
        opacity=opacity,
        show_edges=show_edges,
        edge_color=edge_color if show_edges else None,
        smooth_shading=smooth_shading,
    )

    if show_axes:
        plotter.show_axes()

    plotter.show(interactive=True)
    return plotter




