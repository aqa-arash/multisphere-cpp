/**
 * @file main_cube.cpp
 * @brief Synthetic cube test for multisphere-cpp reconstruction.
 *
 * Generates a synthetic voxel cube in a larger domain, runs multisphere reconstruction, and exports results.
 *
 * @author Arash Moradian
 * @date 2026-03-24
 */

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include "GEMSS/GEMSS-interface.h"

using namespace GEMSS;


int main() {
    std::cout << "--- Multisphere Cube Test ---" << std::endl;

    std::string meshname = "Hamburg_Sand_particle_00058";
    std::string meshdir = "samples/" + meshname +".stl";

    STLMesh mesh = load_mesh(meshdir);
    
    // 2. Run multisphere reconstruction
    std::cout << "[2/3] Running reconstruction..." << std::endl;
    GEMSS::MultisphereConfig config;
    config.div = 300;
    config.min_center_distance_rel = 10.0f;
    config.search_window = 4;
    config.min_radius_vox = 4;
    config.precision_target = 0.9999f;
    config.max_spheres = 2000;
    config.radius_offset_vox = 0.0f;
    config.show_progress = true;
    config.confine_mesh = false;
    config.compute_physics = 1;
    config.persistence = 20;

    SpherePack final_cube_sp;

    // parameter study for cube reconstruction
    #pragma omp parallel for schedule(dynamic) 
    for (int i = 0; i < 2; ++i) {
        int max_spheres_list[2] = {  1,   2};
        float md_list[2] =        {  3.0,    2.0};
        int max_spheres = max_spheres_list[i];
        float md = md_list[i];
        std::string case_name = std::to_string(max_spheres)+ meshname +".vtk";
        GEMSS::MultisphereConfig local_config = config; // thread-local copy
        local_config.max_spheres = max_spheres;
        local_config.min_center_distance_rel = md;
        SpherePack cube_sp = multisphere_from_mesh(mesh, local_config);
        if (max_spheres == 2560) {
            #pragma omp critical
            {
                final_cube_sp = cube_sp; // save for final export
            }
        }
        #pragma omp critical
        {
            std::cout << "\n--- Limit: " << max_spheres << " ---" << std::endl;
            print_sphere_pack_info(cube_sp);
            export_to_vtk(cube_sp, case_name);

        }
    }

    VoxelGrid<uint8_t> target_grid = mesh_to_binary_grid(mesh, config );

    
    
    compute_multisphere_physics(final_cube_sp, target_grid); // Recompute physics based on original mesh properties
    std::cout << "\n--- Ground Truth Parameters ---" << std::endl;
    print_sphere_pack_info(final_cube_sp);

    // 3. Export results
    //save_mesh_to_stl(grid_to_mesh(cube), "original_cube.stl");
    std::cout << "To see the result, export to CSV or recompile with VTK enabled." << std::endl;

    return 0;
}
