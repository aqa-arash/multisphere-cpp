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
#include "GEMSS/GEMSS-interface.h"

using namespace GEMSS;

int main() {
    std::cout << "--- Multisphere Cube Test ---" << std::endl;

    // 1. Create a 300^3 voxel cube in a 400^3 domain
    double v_size = 0.1;
    int nx = 400, ny = 400, nz = 400;
    VoxelGrid<uint8_t> cube(nx, ny, nz, v_size);
    int min_x = 5, max_x = 395;
    int min_y = 50, max_y = 350;
    int min_z = 100, max_z = 300;
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z)
                if (x >= min_x && x < max_x && y >= min_y && y < max_y && z >= min_z && z < max_z)
                    cube(x, y, z) = true;

    // 2. Run multisphere reconstruction
    std::cout << "[2/3] Running reconstruction..." << std::endl;
    GEMSS::MultisphereConfig config;
    config.min_center_distance_rel = 6.0f;
    config.search_window = 4;
    config.min_radius_vox = 4;
    config.precision_target = 0.99f;
    config.max_spheres = 2000;
    config.radius_offset_vox = 0.0f;
    config.show_progress = false;
    config.confine_mesh = false;
    config.compute_physics = 1;
    config.persistence = 20;

    SpherePack final_cube_sp;

    // parameter study for cube reconstruction
    #pragma omp parallel for schedule(dynamic) 
    for (int i = 0; i < 10; ++i) {
        int max_spheres_list[10] = {5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560};
        int max_spheres = max_spheres_list[i];
        GEMSS::MultisphereConfig local_config = config; // thread-local copy
        local_config.max_spheres = max_spheres;
        SpherePack cube_sp = multisphere_from_voxels(cube, local_config);
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
        }
    }

    
    compute_multisphere_physics(final_cube_sp, cube); // Recompute physics based on original mesh properties
    std::cout << "\n--- Ground Truth Parameters ---" << std::endl;
    print_sphere_pack_info(final_cube_sp);

    // 3. Export results
    export_to_csv(final_cube_sp, "reconstructed_cube.csv");
    save_mesh_to_stl(grid_to_mesh(cube), "original_cube.stl");
    export_to_vtk(final_cube_sp, "reconstructed_cube.vtk");
    std::cout << "To see the result, export to CSV or recompile with VTK enabled." << std::endl;

    return 0;
}
