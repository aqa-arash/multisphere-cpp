#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem> 
#include <Eigen/Dense>
#include "multisphere_datatypes.hpp"
#include "multisphere_reconstruction.hpp"
#include "multisphere_visualization.hpp"
#include "cnpy.h" 

namespace fs = std::filesystem;

int main() {
    // 1. Directory Setup
    std::string base_dir = "Boost_study_output";
    std::string csv_dir  = base_dir + "/csv_data";
    std::string bool_dir = base_dir + "/geometries/boolean";
    std::string edt_dir  = base_dir + "/geometries/distance_transform";

    fs::create_directories(csv_dir);
    fs::create_directories(bool_dir);
    fs::create_directories(edt_dir);

    // 2. Simulation Parameters
    const int nx = 256, ny = 256, nz = 256;
    const float a = 120.0f;
    const float b = 60.0f;
    const float c = 60.0f;
    // make an ellipsoid with radii a,b,c and then scale it to fit in the grid
    const float v_size = 1.0f; // voxel size
    
    std::ofstream master_csv(csv_dir + "/summary_results.csv");
    master_csv << "step,center_distance,found_sphere_idx,recon_x,recon_y,recon_z,recon_r\n";

    // Initialize Boolean Grid
    VoxelGrid<bool> grid(nx, ny, nz, v_size);
    std::vector<uint8_t> bool_buffer(nx * ny * nz, 0);

    Eigen::Vector3f center_point = Eigen::Vector3f(nx/2.0f, ny/2.0f, nz/2.0f);

    // 3. Generate Ellipsoid Geometry
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                float dx = (x - center_point.x()) * v_size;
                float dy = (y - center_point.y()) * v_size;
                float dz = (z - center_point.z()) * v_size;
                float val = (dx*dx)/(a*a) + (dy*dy)/(b*b) + (dz*dz)/(c*c);
                if (val <= 1.0f) {
                    grid(x, y, z) = true;
                }
                else {
                    grid(x, y, z) = false;
                }
            }
        }
    }


    // Fill the uint8 buffer for cnpy
    for(size_t i = 0; i < grid.data.size(); ++i) {
        if (grid.data[i]) bool_buffer[i] = 1;
    }

    // 3. Distance Transform & Export
    VoxelGrid<float> edt_grid = grid.distance_transform(); 

    cnpy::npy_save(bool_dir + "/bool_step_"  + ".npy", 
                    bool_buffer.data(), {(size_t)nx, (size_t)ny, (size_t)nz}, "w");

    cnpy::npy_save(edt_dir + "/edt_step_" + ".npy", 
                    edt_grid.data.data(), {(size_t)nx, (size_t)ny, (size_t)nz}, "w");

    Eigen::MatrixX4f sphere_table = Eigen::MatrixX4f(2, 4); // Empty table for initial reconstruction
    // Add two large spheres close to the center to kickstart the reconstruction

    float center_distance = 20.0f; // Distance from center for initial spheres

    float radius0 = edt_grid((int)(center_point.x() - center_distance), (int)center_point.y(), (int)center_point.z());
    float radius1 = edt_grid((int)(center_point.x() + center_distance), (int)center_point.y(), (int)center_point.z());

    std::cout << "Initial Sphere 0: Center = (" << center_point.x() - center_distance << ", " << center_point.y() << ", " << center_point.z() << "), Radius = " << radius0 << std::endl;
    std::cout << "Initial Sphere 1: Center = (" << center_point.x() + center_distance << ", " << center_point.y() << ", " << center_point.z() << "), Radius = " << radius1 << std::endl;

    sphere_table.row(0) = Eigen::Vector4f(center_point.x() - center_distance, center_point.y(), center_point.z(), radius0);
    sphere_table.row(1) = Eigen::Vector4f(center_point.x() + center_distance, center_point.y(), center_point.z(), radius1);


    // 4. Reconstruction
    SpherePack pack = multisphere_from_voxels(grid,
                                    10, // min_center_distance_vox
                                    50, // max iter
                                    20, // min_radius_vox
                                    0.99, // precision_target
                                    100000, // max_spheres
                                    false,
                                    sphere_table // initial sphere table
                                    );



    // 5. FIXED: Accessing Sphere Data via Eigen Matrices
    for (size_t i = 0; i < pack.num_spheres(); ++i) {
        // pack.centers is MatrixX3f (Rows = spheres, Cols = X,Y,Z)
        // pack.radii is VectorXf
        master_csv  << "final," 
                    << center_distance << "," 
                    << i << "," 
                    << pack.centers(i, 0) << "," // X
                    << pack.centers(i, 1) << "," // Y
                    << pack.centers(i, 2) << "," // Z
                    << pack.radii(i) << "\n";    // Radius
    }

    export_to_vtk(pack, 
        csv_dir + "/spheres_step_" +  ".vtk");

    std::cout  << " (Dist: " << center_distance << ") processed." << std::endl;


    master_csv.close();
    return 0;
}