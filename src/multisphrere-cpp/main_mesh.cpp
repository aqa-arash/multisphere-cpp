#include <iostream>
#include <Eigen/Dense>
#include "multisphere_datatypes.hpp"
#include "multisphere_io.hpp"
#include "multisphere_reconstruction.hpp" // Contains the main algorithm logic
#include "multisphere_visualization.hpp"

int main() {
    std::cout << "--- Multisphere Reconstruction Test ---" << std::endl;

    // 1. load a mesh from file
    FastMesh example_mesh = load_mesh_fast("example_mesh.stl");
    // --- DEBUG SNIPPET START ---
    // Save the mesh immediately to check if parsing was correct.
    // If "debug_output.stl" looks scrambled or "exploded" in MeshLab/Paraview,
    // then the load_mesh_fast struct padding fix is still needed.
    save_mesh_to_stl(example_mesh, "debug_output.stl");
    //std::cout << "[DEBUG] Check 'debug_output.stl' against 'example_mesh.stl' now." << std::endl;
    // --- DEBUG SNIPPET END ---


    VoxelGrid<bool> voxel_grid = mesh_to_binary_grid(example_mesh, 100, 2);
    // show the corner and middle point values for debugging
   
    std::cout << "[1/3] Synthetic voxel grid created." << std::endl;

    // 2. Run the Reconstruction Algorithm
    // We target 10 spheres or 95% precision
    std::cout << "[2/3] Running reconstruction..." << std::endl;

    SpherePack single_sp = multisphere_from_mesh(
        example_mesh,
        150,    // div
        2,    // padding
        8,     // min_radius_vox
        0.99,  // precision_target
        8,    // min_center_distance_vox
        100,   // max_spheres
        1, // max_iter
        true, // show_progress
        false // confine_mesh
    ) ;

        SpherePack boosted_sp = multisphere_from_mesh(
        example_mesh,
        150,    // div
        2,    // padding
        8,     // min_radius_vox
        0.99,  // precision_target
        8,    // min_center_distance_vox
        100,   // max_spheres
        10, // max_iter
        true, // show_progress
        false // confine_mesh
    ) ;

    std::cout << "\nReconstruction Complete!" << std::endl;
    std::cout << "--Single Sphere : \n Spheres found: " << single_sp.num_spheres() << std::endl;
    std::cout << "Max radius: " << single_sp.max_radius() << " units" << std::endl;

    std::cout << "--Boosted Sphere : \n Spheres found: " << boosted_sp.num_spheres() << std::endl;
    std::cout << "Max radius: " << boosted_sp.max_radius() << " units" << std::endl;




    // 3. Visualization (Optional based on CMake/HAVE_VTK)
#ifdef HAVE_VTK
    std::cout << "[3/3] Opening VTK Visualization..." << std::endl;
    plot_sphere_pack(single_sp);
#else
    std::cout << "[3/3] VTK not enabled. Skipping visualization." << std::endl;
    export_to_csv(single_sp, "from_mesh_spheres.csv");
    export_to_vtk(single_sp, "from_mesh_spheres.vtk");

    export_to_csv(boosted_sp, "from_mesh_boosted_spheres.csv");
    export_to_vtk(boosted_sp, "from_mesh_boosted_spheres.vtk");

    std::cout << "To see the result, export to CSV or recompile with VTK fixed." << std::endl;
#endif

    return 0;
}