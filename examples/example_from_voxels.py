"""
Voxel-based example for the multisphere package.

Pipeline:
1. Load a 3D voxel mask from a .npy file.
2. Build a multisphere representation from the voxel grid.
3. Visualize the sphere pack.
4. Export the sphere pack as CSV, VTK, and STL.
"""

from __future__ import annotations

import os
from pathlib import Path

from multisphere import (
    load_voxels_from_npy,
    multisphere_from_voxels,
    export_sphere_pack_to_stl,
    export_sphere_pack_to_csv,
    export_sphere_pack_to_vtk,
    plot_sphere_pack,
    SpherePack,
    VoxelGrid,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Load voxel grid
    # ------------------------------------------------------------------
    examples_dir = Path(__file__).resolve().parent

    voxel_path = examples_dir / "example_voxels.npy"
    if not voxel_path.is_file():
        raise FileNotFoundError(
            f"Example voxel file not found at: {voxel_path}\n"
            "Place a 3D numpy array named 'example_voxels.npy' in the "
            "examples/ directory or adjust voxel_path in example_from_voxels.py."
        )

    output_dir = examples_dir / "output_voxel_example"
    os.makedirs(output_dir, exist_ok=True)


    print(f"Loading voxel grid from: {voxel_path}")
    voxel_grid: VoxelGrid = load_voxels_from_npy(
        path=str(voxel_path),
        voxel_size=1.0,
        origin=(0.0, 0.0, 0.0),
    )

    print(f"Voxel grid shape: {voxel_grid.shape}")
    print(f"Voxel size: {voxel_grid.voxel_size}")

    # ------------------------------------------------------------------
    # 2) Build multisphere from voxels
    # ------------------------------------------------------------------
    print("Reconstructing multisphere representation from voxel grid...")

    sphere_pack: SpherePack = multisphere_from_voxels(
        voxel_grid=voxel_grid,
        min_radius_vox=None,      # rely on precision / max_spheres
        precision=0.90,           # stop once ~90% voxel coverage is reached
        min_center_distance_vox=4,
        max_spheres=200,
        show_progress=False,
    )

    print(f"Number of spheres: {sphere_pack.num_spheres}")
    print(
        f"Radius range: [{sphere_pack.min_radius:.4g}, "
        f"{sphere_pack.max_radius:.4g}]"
    )

    # ------------------------------------------------------------------
    # 3) Visualization
    # ------------------------------------------------------------------
    print("Plotting sphere pack...")
    plot_sphere_pack(sphere_pack)

    # ------------------------------------------------------------------
    # 4) Create multisphere mesh and export CSV, VTK, STL
    # ------------------------------------------------------------------

    csv_path = output_dir / "multisphere_spheres.csv"
    vtk_path = output_dir / "multisphere_spheres.vtk"
    stl_path = output_dir / "multisphere_mesh.stl"

    print(f"Exporting sphere pack to CSV: {csv_path}")
    export_sphere_pack_to_csv(sphere_pack, str(csv_path))

    print(f"Exporting sphere pack to VTK: {vtk_path}")
    export_sphere_pack_to_vtk(sphere_pack, str(vtk_path))

    print(f"Exporting multisphere mesh to STL: {stl_path}")
    export_sphere_pack_to_stl(
        sphere_pack=sphere_pack,
        path=str(stl_path),
        resolution=4,
        engine="manifold",
    )

    print("Voxel example finished.")


if __name__ == "__main__":
    main()
