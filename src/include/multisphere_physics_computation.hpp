#ifndef MULTISPHERE_PHYSICS_COMPUTATION_HPP
#define MULTISPHERE_PHYSICS_COMPUTATION_HPP

/**
 * @file multisphere_physics_computation.hpp
 * @brief Physical property calculations for multisphere-cpp.
 *
 * Computes volume, center of mass, and inertia tensor from a voxel grid mask
 * using a highly-optimized, single-pass Parallel Axis Theorem reduction.
 *
 * @author Arash Moradian
 * @date 2026-03-12
 */

#include <cmath>
#include <iostream>
#include <Eigen/Dense>

#ifdef HAVE_OPENMP
    #include <omp.h>
#endif

#include "multisphere_datatypes.hpp"

namespace MSS {

/**
 * @brief Computes volume, Center of Mass (CoM), and inertia tensor from the final voxel mask in a single pass.
 * Updates the physical properties directly inside the provided SpherePack.
 * @param pack The SpherePack object to update.
 * @param final_mask The binary voxel grid representing the multisphere union.
 */
inline void compute_multisphere_physics(SpherePack& pack, const VoxelGrid<uint8_t>& final_mask) {
    // 1st order moments (for CoM)
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    // 2nd order moments (for Inertia)
    double sum_xx = 0.0, sum_yy = 0.0, sum_zz = 0.0;
    double sum_xy = 0.0, sum_xz = 0.0, sum_yz = 0.0;
    size_t occupied_count = 0;
    
    size_t ny = final_mask.ny();
    size_t nz = final_mask.nz();
    size_t total_voxels = final_mask.data.size();
    float vs = final_mask.voxel_size;

    // =========================================================================
    // Single Pass: Accumulate Geometric Moments (Relative to Grid Origin)
    // =========================================================================
    #pragma omp parallel for reduction(+:sum_x, sum_y, sum_z, sum_xx, sum_yy, sum_zz, sum_xy, sum_xz, sum_yz, occupied_count)
    for (size_t i = 0; i < total_voxels; ++i) {
        if (final_mask.data[i] > 0) {
            size_t x = i / (ny * nz);
            size_t rem = i % (ny * nz);
            size_t y = rem / nz;
            size_t z = rem % nz;

            // Use local coordinates (relative to grid origin) inside the loop 
            // to prevent catastrophic floating-point cancellation on large grids.
            double lx = (x + 0.5) * vs;
            double ly = (y + 0.5) * vs;
            double lz = (z + 0.5) * vs;

            // Accumulate 1st order (CoM)
            sum_x += lx;
            sum_y += ly;
            sum_z += lz;

            // Accumulate 2nd order diagonals
            sum_xx += ly * ly + lz * lz;
            sum_yy += lx * lx + lz * lz;
            sum_zz += lx * lx + ly * ly;
            
            // Accumulate 2nd order off-diagonals (products of inertia)
            sum_xy += lx * ly;
            sum_xz += lx * lz;
            sum_yz += ly * lz;

            occupied_count++;
        }
    }

    if (occupied_count == 0) return;

    // =========================================================================
    // Post-Loop: Shift to Absolute Coordinates and Apply Parallel Axis Theorem
    // =========================================================================
    
    double voxel_vol = vs * vs * vs;
    pack.volume = occupied_count * voxel_vol;

    // Compute Local CoM
    double cx_local = sum_x / occupied_count;
    double cy_local = sum_y / occupied_count;
    double cz_local = sum_z / occupied_count;

    // Set Absolute CoM
    pack.center_of_mass = final_mask.origin + Eigen::Vector3f(cx_local, cy_local, cz_local);

    // Shift Inertia Tensor to CoM using Parallel Axis Theorem
    // I_com = I_origin - Mass * (Distance_to_CoM)^2
    double Ixx = (sum_xx * voxel_vol) - (pack.volume * (cy_local * cy_local + cz_local * cz_local));
    double Iyy = (sum_yy * voxel_vol) - (pack.volume * (cx_local * cx_local + cz_local * cz_local));
    double Izz = (sum_zz * voxel_vol) - (pack.volume * (cx_local * cx_local + cy_local * cy_local));

    // For products of inertia, the shift is + Mass * cx * cy
    double Ixy = -(sum_xy * voxel_vol) + (pack.volume * cx_local * cy_local);
    double Ixz = -(sum_xz * voxel_vol) + (pack.volume * cx_local * cz_local);
    double Iyz = -(sum_yz * voxel_vol) + (pack.volume * cy_local * cz_local);

    // Add self-inertia of the voxels (uniform cubes)
    double self_inertia_total = pack.volume * (vs * vs) / 6.0;
    Ixx += self_inertia_total;
    Iyy += self_inertia_total;
    Izz += self_inertia_total;

    pack.inertia_tensor << Ixx, Ixy, Ixz,
                           Ixy, Iyy, Iyz,
                           Ixz, Iyz, Izz;

    // =========================================================================
    // Principal Axes via Eigendecomposition
    // =========================================================================
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(pack.inertia_tensor);
    if (eigensolver.info() == Eigen::Success) {
        pack.principal_moments = eigensolver.eigenvalues();
        pack.principal_axes = eigensolver.eigenvectors();
    } else {
        std::cerr << "[WARNING] Eigendecomposition for inertia tensor failed." << std::endl;
    }
}

} // namespace MSS

#endif // MULTISPHERE_PHYSICS_COMPUTATION_HPP