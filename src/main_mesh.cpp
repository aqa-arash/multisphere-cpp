/**
 * @file main_mesh.cpp
 * @brief Multisphere reconstruction from mesh files using multisphere-cpp.
 *
 * Loads mesh files, reconstructs sphere packs, and exports results.
 * Visualization is enabled if VTK is available.
 *
 * @author Arash Moradian
 * @date 2026-03-09
 */

#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

// Project headers
#include "GEMSS/GEMSS-interface.h"

using namespace GEMSS;



/**
 * @brief Entry point for mesh-based multisphere reconstruction.
 *
 * Loads mesh files, reconstructs sphere packs, and exports/visualizes results.
 *
 * @return int Exit code.
 */
int main() {
    std::cout << "--- Multisphere Reconstruction Test ---" << std::endl;

    // List of mesh files to process
    std::vector<std::string> models = { "example_mesh.stl" };

    for (const auto& model_name : models) {
        // 1. Load mesh from file
        STLMesh example_mesh = load_mesh(model_name);

        // --- DEBUG SNIPPET ---
        // Uncomment to save mesh for debugging
        // save_mesh_to_stl(example_mesh, "debug_output.stl");
        // std::cout << "[DEBUG] Check 'debug_output.stl' against 'input_mesh.stl' now." << std::endl;

        // 2. Run the Reconstruction Algorithm
        GEMSS::MultisphereConfig config;
        config.div = 400; // Voxel grid resolution
        config.padding = 2; // Grid padding
        config.search_window = 10; // Search window size
        config.min_center_distance_rel = 0.5f; // Minimum center distance relative to radius
        config.min_radius_vox = 12; // Minimum radius in voxels
        config.precision_target = 0.99f; // Target precision
        config.max_spheres = 10000; // Maximum number of spheres
        config.show_progress = true; // Show progress output
        config.confine_mesh = false; // Do not confine spheres to mesh boundary
        config.initial_sphere_table = Eigen::MatrixXf(0,4); // No initial sphere table
        config.compute_physics = 1; // Compute physical properties
        config.prune_isolated_spheres = true; // Do not prune isolated spheres

        SpherePack single_sp = multisphere_from_mesh(
            example_mesh,
            config
        );
        
        export_to_csv(single_sp, model_name + "_recon.csv");
        export_to_vtk(single_sp, model_name + "_recon.vtk");

        print_sphere_pack_info(single_sp);

        auto binary_grid = mesh_to_binary_grid(example_mesh, config); // Update config with min_radius_vox if minimum_radius_real is set
        compute_multisphere_physics(single_sp, binary_grid); // Recompute physics based on original mesh properties

    
        print_sphere_pack_info(single_sp);


    }

    return 0;
}