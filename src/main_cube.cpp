/**
 * @file main_cube.cpp
 * @brief Synthetic cube test for multisphere-cpp reconstruction.
 *
 * Generates a synthetic voxel cube in a larger domain, runs multisphere reconstruction, and exports results.
 *
 * @author (auto-generated)
 * @date 2026-03-24
 */

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "GEMSS/GEMSS-interface.h"

using namespace GEMSS;

void print_sphere_pack_info(const SpherePack& sp) {
    std::cout << "Sphere Pack Info:" << std::endl;
    std::cout << "      Number of spheres: " << sp.num_spheres() << std::endl;
    std::cout << "      Max radius: " << sp.max_radius() << " units" << std::endl;
    std::cout << "      Min radius: " << sp.min_radius() << " units" << std::endl;
    std::cout << "      Volume of union: " << sp.volume << " units^3" << std::endl;
    std::cout << "      Center of mass: " << sp.center_of_mass.transpose() << " units" << std::endl;
    std::cout << "      Principal moments: " << sp.principal_moments.transpose() << " units^5" << std::endl;
    std::cout << "      Principal axes:\n" << sp.principal_axes << std::endl;
}

int main() {
    std::cout << "--- Multisphere Cube Test ---" << std::endl;

    // 1. Create a 300^3 voxel cube in a 400^3 domain
    double v_size = 0.1;
    int nx = 400, ny = 400, nz = 400;
    VoxelGrid<uint8_t> cube(nx, ny, nz, v_size);
    int min_x = 50, max_x = 350;
    int min_y = 50, max_y = 350;
    int min_z = 50, max_z = 350;
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z)
                if (x >= min_x && x < max_x && y >= min_y && y < max_y && z >= min_z && z < max_z)
                    cube(x, y, z) = true;

    // 2. Run multisphere reconstruction
    std::cout << "[2/3] Running reconstruction..." << std::endl;
    GEMSS::MultisphereConfig config;
    config.search_window = 3;
    config.min_radius_vox = 3;
    config.precision_target = 0.99f;
    config.max_spheres = 2000;
    config.show_progress = true;
    config.confine_mesh = false;
    config.compute_physics = 2;

    SpherePack cube_sp = multisphere_from_voxels(cube, config);

    std::cout << "\nReconstruction Complete!" << std::endl;
    print_sphere_pack_info(cube_sp);

    // 3. Export results
    export_to_csv(cube_sp, "reconstructed_cube.csv");
    save_mesh_to_stl(grid_to_mesh(cube), "original_cube.stl");
    export_to_vtk(cube_sp, "reconstructed_cube.vtk");
    std::cout << "To see the result, export to CSV or recompile with VTK enabled." << std::endl;

    return 0;
}
