# multisphere

*multisphere* computes high-quality overlapping-sphere representations of
arbitrary 3D geometries based on voxelized Euclidean distance transforms
(EDT) and feature-enhanced distance fields (FEDT). The algorithm is
designed for Discrete Element Method (DEM) simulations, where accurate
yet computationally efficient particle shape representations are essential.

The implemented method follows the MSS algorithm introduced by Buchele
et al. and achieves high shape accuracy at low computational cost


The package supports reconstruction from:
	Triangle surface meshes (STL)
	Binary voxel volumes (NumPy)

## Scientific Background

The multisphere algorithm is based on:
	Voxelization of the target geometry
	Exact Euclidean Distance Transform (EDT)
	Peak refinement of EDT maxima
	Iterative residual correction using the
	Feature-Enhanced Distance Tensor (FEDT)
	Termination by shape accuracy, minimum radius, or maximum sphere count

This approach avoids the major drawbacks of greedy sphere removal methods, 
such as spurious small spheres, symmetry violations, and excessive
runtime. 

---

## Features

multisphere reconstruction from:
	Triangle meshes (STL)
	Binary voxel grids (NumPy)
Exact EDT-driven sphere placement
Iterative residual correction using FEDT
Multiple termination criteria:
	Shape precision
	Maximum number of spheres
	Minimum allowed sphere radius
Export formats:
	CSV (sphere centers & radii)
	VTK (visualization)
	STL (boolean union of spheres)
Optional visualization with PyVista
Optional Dice similarity coefficient for mesh-based validation
Optional boundary correction to enforce strict STL containment

---

## Installation

Requires Python ≥ 3.9. Linux and macOS are fully supported. 
Windows is supported but boolean backends may require additional setup.

### Core installation (no visualization, no STL boolean unions)

pip install multisphere

### Full installation

pip install multisphere[full]

The full installation adds:
- PyVista + Matplotlib for visualization
- manifold3d for boolean STL reconstruction and Dice computation

---

## Basic usage

### Mesh based reconstruction

import multisphere as ms

mesh = ms.load_mesh_from_stl("geometry.stl")

sphere_pack = ms.multisphere_from_mesh(
    mesh=mesh,
    div=150,
    padding=2,
    precision=0.90,
    min_center_distance_vox=4,
    max_spheres=100,
)

ms.export_sphere_pack_to_csv(sphere_pack, "spheres.csv")
ms.export_sphere_pack_to_vtk(sphere_pack, "spheres.vtk")
ms.export_sphere_pack_to_stl(sphere_pack, "spheres.stl")

Note: STL export requires a boolean backend
(manifold3d or Blender).

### Voxel based reconstruction

import multisphere as ms

voxel_grid = ms.load_voxels_from_npy(
    "volume.npy",
    voxel_size=1.0,
    origin=(0.0, 0.0, 0.0),
)

sphere_pack = ms.multisphere_from_voxels(
    voxel_grid=voxel_grid,
    precision=0.95,
    min_center_distance_vox=4,
)

ms.export_sphere_pack_to_csv(sphere_pack, "spheres.csv")
ms.export_sphere_pack_to_vtk(sphere_pack, "spheres.vtk")

### Examples

Complete working examples are located in
examples/

The examples directory contains:
- example_from_mesh.py — full mesh-to-multisphere pipeline
- example_from_voxels.py — voxel-to-multisphere reconstruction

---

## Limitations

Boolean STL reconstruction is numerically fragile for:
	Very large sphere counts
	Extreme overlaps
Performance scales with voxel resolution and peak density.
Reconstruction quality depends heavily on the chosen voxel resolution.

---

## License

This project is licensed under the GNU General Public License v3.0.

You are free to use, modify, and redistribute the software under the
terms of the GPL. Any derivative work must also be released under the GPL.

See the LICENSE file for full details.

This software optionally links against the Manifold3D library
(Apache License 2.0), which is compatible with GPL-3.0.

---

## Author

Felix Buchele
Friedrich-Alexander-Universität Erlangen–Nürnberg (FAU)

---

## Citation

If you use this software in academic work, you are expected to cite the
corresponding publication describing the multisphere reconstruction
method.

Felix Buchele, *Multi-Sphere-Shape generator for DEM simulations using the multi-sphere approach*
manuscript in preparation

A DOI will be added upon publication

