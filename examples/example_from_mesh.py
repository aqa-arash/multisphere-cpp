"""
Mesh-based example for the multisphere package.

Pipeline:
1. Load an STL mesh from the examples directory.
2. Build a multisphere representation from the mesh.
3. OPTIONAL: Visualize the original mesh and the sphere pack. 
   Requires pyvista and matplotlib
4. OPTIONAL: Create a mesh from the sphere pack via boolean union.
   Requires a boolean backend. Either blender or manifold3D
5. OPTIONAL: Compute dice coefficient between original mesh and multisphere
   Requires a boolean backend. Either blender or manifold3D
6. OPTIONAL: adjust sphere pack boundaries to STL boundaries
7. Export the sphere pack as CSV, VTK, and STL.
   The STL export requires a boolean backend, CSV and VTK export work without 
   additional dependencies.
   
To be able to perform all of the optional operations, install multisphere as
pip install multisphere[full]
"""

from __future__ import annotations

import os
from pathlib import Path

from multisphere import (
    load_mesh_from_stl,
    multisphere_from_mesh,
    create_multisphere_mesh,
    compute_dice_coefficient,
    export_sphere_pack_to_stl,
    export_sphere_pack_to_csv,
    export_sphere_pack_to_vtk,
    plot_mesh,
    plot_sphere_pack,
    SpherePack,
    adjust_spheres_to_stl_boundary,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Resolve paths inside the examples directory
    # ------------------------------------------------------------------
    examples_dir = Path(__file__).resolve().parent

    mesh_path = examples_dir / "example_mesh.stl"
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Example mesh not found at: {mesh_path}\n"
            "Place an STL file named 'example_mesh.stl' in the examples/"
            " directory or adjust mesh_path in mesh_example.py."
        )

    output_dir = examples_dir / "output_mesh_example"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Load mesh
    # ------------------------------------------------------------------
    print(f"Loading mesh from: {mesh_path}")
    mesh = load_mesh_from_stl(str(mesh_path))

    # ------------------------------------------------------------------
    # 3) Build multisphere from mesh
    # ------------------------------------------------------------------
    print("Reconstructing multisphere representation...")

    sphere_pack: SpherePack = multisphere_from_mesh(
        mesh=mesh,
        div=150,                  # voxel resolution (min AABB edge / div)
        padding=2,
        min_radius_vox=None,      # rely on precision / max_spheres here
        precision=0.90,           # stop once ~90% voxel agreement is reached
        min_center_distance_vox=4,
        max_spheres=100,
        show_progress=False,
    )

    print(f"Number of spheres: {sphere_pack.num_spheres}")
    print(
        f"Radius range: [{sphere_pack.min_radius:.4g}, "
        f"{sphere_pack.max_radius:.4g}]"
    )

    # ------------------------------------------------------------------
    # 4) Visualization
    # ------------------------------------------------------------------
    
    try:
        print("Plotting original mesh...")
        plot_mesh(mesh)
    
        print("Plotting sphere pack...")
        plot_sphere_pack(sphere_pack)
    except ImportError as exc:
        print(str(exc))
        print("Visualization not possible, PyVista and/or Matplotlib missing!")
        print("Install additional dependencies with pip install "
              "multisphere[full]")

    # ------------------------------------------------------------------
    # 5) Create mesh from spheres and compute Dice coefficient
    # ------------------------------------------------------------------
    try:
        print("Creating multisphere mesh via boolean union...")
        sphere_mesh = create_multisphere_mesh(
            sphere_pack,
            resolution=4,     # icosphere subdivisions (2–5 is typical)
            engine="manifold"     # or "auto"/"blender" depending on your setup
        )
    
        print("Computing Dice coefficient between original and multisphere "
              "mesh...")
        dice = compute_dice_coefficient(
            mesh_1=mesh,
            mesh_2=sphere_mesh,
            engine="manifold",
        )
        print(f"Dice coefficient: {dice:.2f}%")
    except (ImportError, RuntimeError) as exc:
        print(str(exc))
        print("Boolean backend (manifold3D or blender) missing!")
        print("Install additional dependencies with pip install "
              "multisphere[full]")
    
    # ------------------------------------------------------------------
    # 6) Adjust multisphere boundaries to STL boundaries
    # ------------------------------------------------------------------
    
    # Optional: clip spheres that protrude outside the STL surface
    sphere_pack =  adjust_spheres_to_stl_boundary(sphere_pack, mesh)

    # ------------------------------------------------------------------
    # 7) Export CSV, VTK, STL
    # ------------------------------------------------------------------
    csv_path = output_dir / "multisphere_spheres.csv"
    vtk_path = output_dir / "multisphere_spheres.vtk"
    stl_path = output_dir / "multisphere_mesh.stl"

    print(f"Exporting sphere pack to CSV: {csv_path}")
    export_sphere_pack_to_csv(sphere_pack, str(csv_path))

    print(f"Exporting sphere pack to VTK: {vtk_path}")
    export_sphere_pack_to_vtk(sphere_pack, str(vtk_path))
    
    try:
        print(f"Exporting multisphere mesh to STL: {stl_path}")
        export_sphere_pack_to_stl(
            sphere_pack=sphere_pack,
            path=str(stl_path),
            resolution=4,
            engine="manifold",
        )
    except (ImportError, RuntimeError) as exc:
        print(str(exc))
        print("Boolean backend (manifold3D or blender) missing!")
        print("Install additional dependencies with pip install "
              "multisphere[full]")

    print("Mesh example finished.")


if __name__ == "__main__":
    main()

