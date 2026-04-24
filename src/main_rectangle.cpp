/**
 * @file main_rectangle.cpp
 * @brief Synthetic rectangular box test for multisphere-cpp reconstruction.
 *
 * Generates a synthetic voxel rectangular box with three distinct side lengths in a larger domain, runs multisphere reconstruction, and exports results.
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
    std::cout << "--- Multisphere Rectangle Test ---" << std::endl;

    // 1. Create a rectangular box (e.g., 300x200x100 voxels) in a 400^3 domain
    double v_size = 0.1;
    int nx = 400, ny = 400, nz = 400;
    VoxelGrid<uint8_t> rect(nx, ny, nz, v_size);
    int min_x = 50, max_x = 350;   // 300 voxels (x)
    int min_y = 100, max_y = 300;  // 200 voxels (y)
    int min_z = 150, max_z = 250;  // 100 voxels (z)
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z)
                if (x >= min_x && x < max_x && y >= min_y && y < max_y && z >= min_z && z < max_z)
                    rect(x, y, z) = true;

    // 2. Run multisphere reconstruction
    std::cout << "[2/3] Running reconstruction..." << std::endl;
    GEMSS::MultisphereConfig config;
    config.search_window = 3;
    config.min_radius_vox = 3;
    config.precision_target = 0.99f;
    config.max_spheres = 2000;
    config.show_progress = true;
    config.confine_mesh = false;
    config.compute_physics = 1;

    SpherePack rect_sp = multisphere_from_voxels(rect, config);

    std::cout << "\nReconstruction Complete!" << std::endl;
    print_sphere_pack_info(rect_sp);

    // 3. Export results
    export_to_csv(rect_sp, "reconstructed_rectangle.csv");
    save_mesh_to_stl(grid_to_mesh(rect), "original_rectangle.stl");
    export_to_vtk(rect_sp, "reconstructed_rectangle.vtk");
    std::cout << "To see the result, export to CSV or recompile with VTK enabled." << std::endl;

    return 0;
}
