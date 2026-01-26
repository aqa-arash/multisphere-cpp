#ifndef MULTISPHERE_RECONSTRUCTION_HPP
#define MULTISPHERE_RECONSTRUCTION_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <Eigen/Dense>
#ifdef HAVE_OPENMP
    #include <omp.h>
#endif

#include "multisphere_datatypes.hpp"
#include "multisphere_reconstruction_helpers.hpp"
#include "multisphere_utils.hpp"
#include "multisphere_io.hpp"
#include "multisphere_mesh_handler.hpp"


/**
 * Main Algorithm: Construct multisphere from voxel grid.
 */
SpherePack multisphere_from_voxels(
    const VoxelGrid<bool>& input_grid,
    std::optional<int> min_radius_vox = std::nullopt,
    std::optional<float> precision_target = std::nullopt,
    int min_center_distance_vox = 2,
    std::optional<int> max_spheres = std::nullopt,
    int max_iter = 10,
    bool show_progress = true
) {
    VoxelGrid<float> original_distance(input_grid.nx(), input_grid.ny(), input_grid.nz(), input_grid.voxel_size, input_grid.origin);
    VoxelGrid<bool> voxel_grid(input_grid.nx(), input_grid.ny(), input_grid.nz(), input_grid.voxel_size, input_grid.origin);
    
    bool check_min_center_distance = true;
    if (min_center_distance_vox <= 1) {
        std::cout<<"[WARNING] min_center_distance_vox <= 1."<<std::endl;
        min_center_distance_vox = 2;
    }
    // check type T and compute distance field if needed
    original_distance = input_grid.distance_transform();
    voxel_grid = input_grid;
    


    // [DEBUG] to verify field range
    float min_d = 1e9, max_d = -1e9;
    for(float v : original_distance.data) {
        if(v < min_d) min_d = v;
        if(v > max_d) max_d = v;
    }
    std::cout << "[DEBUG] Distance Field Range: [" << min_d << ", " << max_d << "]" << std::endl;
    if (max_d <= 0.0f) {
        std::cerr << "[ERROR] Distance field is zero/negative. Check EDT logic!" << std::endl;
    }
    // [END DEBUG]

    Eigen::MatrixX4f sphere_table(0, 4);
    int iter = 1;

    VoxelGrid<uint8_t> recon_mask (voxel_grid.nx(),voxel_grid.ny(),voxel_grid.nz(), voxel_grid.voxel_size, voxel_grid.origin);

    VoxelGrid<float> min_sphere_distance_field(voxel_grid.nx(), voxel_grid.ny(), voxel_grid.nz(), voxel_grid.voxel_size, voxel_grid.origin);
    
    VoxelGrid<float> residual(voxel_grid.nx(), voxel_grid.ny(), voxel_grid.nz(), voxel_grid.voxel_size, voxel_grid.origin);

    int prev_count = 0;
    bool peaks_found = true;
    while (true) {
        std::cout << "Iteration " << iter << ": Current spheres = " << sphere_table.rows() << std::endl;
        // Termination: Max Spheres
        if (max_spheres.has_value() && sphere_table.rows() >= *max_spheres) break;

        VoxelGrid<float> summed_field(voxel_grid.nx(), voxel_grid.ny(), voxel_grid.nz());

        if (sphere_table.rows() > 0) {

            if (peaks_found)  { 

                // uint8_t is thread-safe and acts as a boolean (0 vs 1).
                spheres_to_grid<uint8_t>( recon_mask, sphere_table.block(prev_count, 0, sphere_table.rows() - prev_count, 4) // new spheres only
                );

                // Termination: Precision
                if (precision_target.has_value()) {
                    float current_precision = compute_voxel_precision(voxel_grid, recon_mask);
                    std::cout<< " Iteration " << iter-1 << ": Current Precision = " << current_precision << std::endl;
                    if (current_precision >= *precision_target) {
                        std::cout << " Termination: Reached target precision of " << *precision_target << std::endl;
                        break;}
                    }
            
                // Min Center Distance Field
                Eigen::MatrixX4f min_sphere_distance_table = sphere_table.block(prev_count, 0, sphere_table.rows() - prev_count, 4);
                min_sphere_distance_table.col(3).array() = min_center_distance_vox;
                spheres_to_grid<float>(min_sphere_distance_field, min_sphere_distance_table, std::numeric_limits<float>::max() 

                );

        
            VoxelGrid<float> spheres_distance = recon_mask.distance_transform();
            residual = residual_distance_field(original_distance, spheres_distance, min_sphere_distance_field);
            }
            
                // summed_field = distance_field + residual
                #pragma omp parallel for
                for (size_t i = 0; i < summed_field.data.size(); ++i) {
                    summed_field.data[i] = original_distance.data[i] + residual.data[i] * iter ;
                }
        } else {
            summed_field.data = original_distance.data;
        }

        // 2. Peak Detection
        Eigen::MatrixX4f peaks = peak_local_max_3d(summed_field, original_distance, min_center_distance_vox, min_radius_vox.value_or(1));
        
        /* [DEBUG]
        std::cout << " Detected Peaks: " << peaks.rows() << std::endl;
        std::cout<< " Sample Peak Radii: ";
        for (int i = 0; i < std::min(5, int(peaks.rows())); ++i) {
            std::cout <<"x = " << peaks(i, 0) << ", y = " << peaks(i, 1) << ", z = " << peaks(i, 2) << ", r = " << peaks(i, 3) << "\n";
        }
        std::cout << std::endl;
        */

        // 3. Filter Peaks
        peaks = filter_and_shift_peaks(peaks, sphere_table,original_distance, min_center_distance_vox);
        if (peaks.rows() == 0) {
            std::cout << " No peaks found after filtering, for iter = " << iter << std::endl;
            if (iter >= max_iter) {
                std::cout << " Terminating." << std::endl;
                break;
              }
            else {
                peaks_found = false;
                ++iter;
                std::cout << " Continuing to next iteration." << std::endl;
                continue;
            }
        }
        peaks_found = true;

        // [DEBUG]
        std::cout << " Peaks after Filtering: " << peaks.rows() << std::endl;
        std::cout<< " Sample Peak Radii after Filtering: ";
        for (int i = 0; i < std::min(5, int(peaks.rows())); ++i) {
            std::cout <<"x = " << peaks(i, 0) << ", y = " << peaks(i, 1) << ", z = " << peaks(i, 2) << ", r = " << peaks(i, 3) << "\n";
        }
        std::cout << std::endl;
        

        // 4. Append Spheres
        prev_count = sphere_table.rows();
        append_sphere_table(sphere_table, peaks, max_spheres);

        if (min_radius_vox.has_value() && sphere_table.rows() == prev_count) {
            std::cout << " No new spheres added that meet min_radius_vox. Terminating." << std::endl;
            break;}

    }


        // debug 
    std::cout << "Final Spheres: " << sphere_table.rows() << std::endl;
    std::cout<< " Sample Sphere Radii: ";
    for (int i = 0; i < std::min(5, int(sphere_table.rows())); ++i) {
        std::cout <<"x = " << sphere_table(i, 0) << ", y = " << sphere_table(i, 1) << ", z = " << sphere_table(i, 2) << ", r = " << sphere_table(i, 3) << "\n";
    }
    std::cout << std::endl;

    //sort spheres by radius descending
    std::vector<std::pair<float, int>> radius_index_pairs;
    for (int i = 0; i < sphere_table.rows(); ++i) {
        radius_index_pairs.emplace_back(sphere_table(i, 3), i);
    }
    std::sort(radius_index_pairs.begin(), radius_index_pairs.end(), std::greater<std::pair<float, int>>());
    Eigen::MatrixX4f sorted_sphere_table(sphere_table.rows(), 4);
    for (int i = 0; i < radius_index_pairs.size(); ++i) {
        sorted_sphere_table.row(i) = sphere_table.row(radius_index_pairs[i].second);;
    }



    // Convert Table to SpherePack (Physical units)
    Eigen::MatrixX3f centers_phys(sorted_sphere_table.rows(), 3);
    Eigen::VectorXf radii_phys(sorted_sphere_table.rows());

    for (int i = 0; i < sorted_sphere_table.rows(); ++i) {
        Eigen::Vector3f pos_vox = sorted_sphere_table.block<1, 3>(i, 0).transpose();
        centers_phys.row(i) = voxel_grid.origin + (pos_vox.array()+0.5f).matrix() * voxel_grid.voxel_size;
        radii_phys(i) = (sorted_sphere_table(i, 3)) * voxel_grid.voxel_size;
    }

    // [DEBUG] Verify final sphere positions and radii
    std::cout << "[DEBUG] Final Sphere Sample (Physical Units):" << std::endl;
    for (int i = 0; i < std::min(5, int(sorted_sphere_table.rows())); ++i) {
        std::cout <<"x = " << centers_phys(i, 0) << ", y = " << centers_phys(i, 1) << ", z = " << centers_phys(i, 2) << ", r = " << radii_phys(i) << "\n";
    }
    std::cout << std::endl;

    return SpherePack(centers_phys, radii_phys);
}


/**
 * Construct a multisphere representation directly from a triangle mesh.
 * Logically equivalent to the Python version, optimized for C++.
 */
SpherePack multisphere_from_mesh(
    const FastMesh& mesh,
    int div = 100,
    int padding = 2,
    std::optional<int> min_radius_vox = std::nullopt,
    std::optional<float> precision = std::nullopt,
    int min_center_distance_vox = 4,
    std::optional<int> max_spheres = std::nullopt,
    float boost = 1,
    bool show_progress = true,
    bool confine_mesh = false ) {
    if (mesh.is_empty()) {
        throw std::runtime_error("Cannot reconstruct from an empty mesh.");
    }


    // 1. Convert to VoxelGrid (This would call your mesh voxelizer)
    // For parity with your logic, we assume the voxelization results in a grid
    // centered on the mesh centroid with the requested padding.
    VoxelGrid<bool> voxel_grid = mesh_to_binary_grid(mesh, div, padding);

    // 3. Perform Reconstruction
    SpherePack sp = multisphere_from_voxels(
        voxel_grid,
        min_radius_vox,
        precision,
        min_center_distance_vox,
        max_spheres,
        boost,
        show_progress
    );

    // 4. Boundary Adjustment (Optional)
    if (confine_mesh) {
#ifdef HAVE_MANIFOLD
        // We only include this if the Manifold/Nanoflann logic is enabled
        return utils::adjust_spheres_to_stl_boundary(sp, mesh.vertices);
#else
        constrain_radii_to_sdf(sp, mesh);
        return sp;
#endif
    }

    return sp;
}


#endif