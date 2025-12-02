"""
multisphere
===========

Approximate 3D geometries with overlapping spheres.

Public API
----------
Types
~~~~~
- SpherePack
    Container for sphere centers and radii.
- VoxelGrid
    3D voxel grid with distance-transform and kernel helpers.

Reconstruction
~~~~~~~~~~~~~~
- multisphere_from_mesh
    Build a multisphere representation from a triangle mesh.
- multisphere_from_voxels
    Build a multisphere representation from a voxel grid.

Geometry utilities
~~~~~~~~~~~~~~~~~~
- create_multisphere_mesh
    Convert a SpherePack into a triangle mesh via boolean union.
- compute_dice_coefficient
    Compute Dice similarity between two meshes.
- adjust_spheres_to_stl_boundary
    Adjust sphere radii to prevent them from extending beyond the STL surface

I/O
~~~
- load_mesh_from_stl
- load_voxels_from_npy
- mesh_to_voxel_grid
- export_sphere_pack_to_stl
- export_sphere_pack_to_csv
- export_sphere_pack_to_vtk

Visualization
~~~~~~~~~~~~~
- plot_sphere_pack
- plot_mesh
"""

from .multisphere_datatypes import SpherePack, VoxelGrid
from .multisphere_reconstruction import (
    multisphere_from_mesh,
    multisphere_from_voxels,
)
from .multisphere_utils import (
    create_multisphere_mesh,
    compute_dice_coefficient,
    adjust_spheres_to_stl_boundary,
)
from .multisphere_io import (
    load_mesh_from_stl,
    load_voxels_from_npy,
    mesh_to_voxel_grid,
    export_sphere_pack_to_stl,
    export_sphere_pack_to_csv,
    export_sphere_pack_to_vtk,
)
from .multisphere_visualisation import (
    plot_sphere_pack,
    plot_mesh,
)

__version__ = "1.0.0"

__all__ = [
    # types
    "SpherePack",
    "VoxelGrid",
    # reconstruction
    "multisphere_from_mesh",
    "multisphere_from_voxels",
    # geometry utilities
    "create_multisphere_mesh",
    "compute_dice_coefficient",
    "adjust_spheres_to_stl_boundary",
    # IO
    "load_mesh_from_stl",
    "load_voxels_from_npy",
    "mesh_to_voxel_grid",
    "export_sphere_pack_to_stl",
    "export_sphere_pack_to_csv",
    "export_sphere_pack_to_vtk",
    # visualization
    "plot_sphere_pack",
    "plot_mesh",
    # meta
    "__version__",
]
