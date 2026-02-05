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

    voxel_path = examples_dir / "ice_cream_cone.npy"
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

    second_voxel_path = examples_dir / "example_voxels.npy"
    if second_voxel_path.is_file():
        print(f"Loading second voxel grid from: {second_voxel_path}")
        voxel_grid_2: VoxelGrid = load_voxels_from_npy(
            path=str(second_voxel_path),
            voxel_size=1.0,
            origin=(0.0, 0.0, 0.0),
        )
        print(f"Second voxel grid shape: {voxel_grid_2.shape}")
        print(f"Second voxel size: {voxel_grid_2.voxel_size}")
    else:
        print(f"No second voxel file found at: {second_voxel_path}, skipping.")

    # compare the two voxel grids if both are loaded, in terms of data type
    if 'voxel_grid_2' in locals():
        print(
            f"First voxel grid data type: {voxel_grid.data.dtype}, "
            f"Second voxel grid data type: {voxel_grid_2.data.dtype}"
        )

    

    # ------------------------------------------------------------------
    # 2) Build multisphere from voxels
    # ------------------------------------------------------------------
    print("Reconstructing multisphere representation from voxel grid...")

    sphere_pack: SpherePack = multisphere_from_voxels(
        voxel_grid=voxel_grid,
        min_radius_vox=2,      # rely on precision / max_spheres
        precision=0.99,           # stop once ~90% voxel coverage is reached
        min_center_distance_vox=13,
        max_spheres=20,
        show_progress=False,
    )

    print(f"Number of spheres: {sphere_pack.num_spheres}")
    print(
        f"Radius range: [{sphere_pack.min_radius:.4g}, "
        f"{sphere_pack.max_radius:.4g}]"
    )


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

    # ------------------------------------------------------------------
    # 3) Visualization
    # ------------------------------------------------------------------
    print("Plotting sphere pack...")
    plot_sphere_pack(sphere_pack)


if __name__ == "__main__":
    main()
