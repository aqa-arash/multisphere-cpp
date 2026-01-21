// multisphere_reconstruction_helpers.hpp
#ifndef MULTISPHERE_RECONSTRUCTION_HELPERS_HPP
#define MULTISPHERE_RECONSTRUCTION_HELPERS_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <optional>
#include "multisphere_datatypes.hpp"
#include "multisphere_io.hpp"
#include "multisphere_voxel_processing.hpp"

// --- Helper: Distance Squared ---
inline double get_dist_sq(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return (a - b).squaredNorm();
}

// -- Helper packed struct for cache-friendly sorting
struct PeakEntry {
    float val;      // Pre-fetched distance value
    int x, y, z;    // Coordinates
    
    // Original index (optional, if you needed to map back to original list)
};

/**
 * Detect local maxima in a 3D volume.
 * Equivalent to skimage.feature.peak_local_max.
 */
Eigen::MatrixX3i peak_local_max_3d(const VoxelGrid<float>& field, int min_distance, int threshold_abs = 1) {
    std::vector<Eigen::Vector3i> peaks;
    int nx = (int)field.nx(); 
    int ny = (int)field.ny();
    int nz = (int)field.nz();

    #pragma omp parallel
    {
        std::vector<Eigen::Vector3i> local_peaks;
        #pragma omp for collapse(2)
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                for (int z = 0; z < nz; ++z) {
                    float val = field(x, y, z);
                    if (val <= 0.0f) continue;

                    bool is_max = true;
                    // Check local neighborhood
                    for (int dx = -min_distance; dx <= min_distance && is_max; ++dx) {
                        for (int dy = -min_distance; dy <= min_distance && is_max; ++dy) {
                            for (int dz = -min_distance; dz <= min_distance && is_max; ++dz) {
                                int nx_idx = x + dx, ny_idx = y + dy, nz_idx = z + dz;
                                if (nx_idx >= 0 && nx_idx < nx && ny_idx >= 0 && ny_idx < ny && nz_idx >= 0 && nz_idx < nz) {
                                    if (field(nx_idx, ny_idx, nz_idx) > val) is_max = false;
                                }
                            }
                        }
                    }
                    if (is_max) local_peaks.push_back({x, y, z});
                }
            }
        }
        #pragma omp critical
        peaks.insert(peaks.end(), local_peaks.begin(), local_peaks.end());
    }

  // 2. Sort Peaks: Primary = Intensity (Descending), Secondary = Coordinate (Ascending)
    std::sort(peaks.begin(), peaks.end(), [&](const Eigen::Vector3i& a, const Eigen::Vector3i& b) {
        float val_a = field(a.x(), a.y(), a.z());
        float val_b = field(b.x(), b.y(), b.z());

        // Define "super close" (floating point tolerance)
        const float epsilon = 1e-5f;

        // 1. Primary Sort: Intensity (Descending)
        // If the difference is significant (outside epsilon), return the larger intensity
        if (std::abs(val_a - val_b) > epsilon) {
            return val_a > val_b;
        }

        // 2. Tie-Breaker: X Coordinate (Ascending)
        if (a.x() != b.x()) {
            return a.x() < b.x();
        }

        // 3. Tie-Breaker: Y Coordinate (Ascending)
        if (a.y() != b.y()) {
            return a.y() < b.y();
        }

        // 4. Tie-Breaker: Z Coordinate (Ascending)
        return a.z() < b.z();
    });

    /* [DEBUG] Print top peak
    if (!peaks.empty()) {
        Eigen::Vector3i top = peaks[0];
        std::cout << "[DEBUG] Top Peak at (" << top.x() << "," << top.y() << "," << top.z() 
                  << ") Val: " << field(top.x(), top.y(), top.z()) << std::endl;
    }
    */

    Eigen::MatrixX3i result(peaks.size(), 3);
    for (size_t i = 0; i < peaks.size(); ++i) result.row(i) = peaks[i];
    return result;
}

Eigen::MatrixX3i filter_peaks(
    const Eigen::MatrixX3i& peaks,
    const Eigen::MatrixX4d& sphere_table,
    int min_center_distance_vox) 
{
    // If no candidates, return immediately
    if (peaks.rows() == 0) return peaks;

    // Squared threshold for unified center-to-center distance
    const double min_dist_sq = min_center_distance_vox * min_center_distance_vox;

    // Indices of peaks we decide to keep
    std::vector<int> keep_indices;
    keep_indices.reserve(peaks.rows());

    // --- The Scan ---
    for (int i = 0; i < peaks.rows(); ++i) {
        double px = (double)peaks(i, 0);
        double py = (double)peaks(i, 1);
        double pz = (double)peaks(i, 2);
        
        bool collision = false;

        // 1. Check against ALL existing spheres (The History)
        for (int j = 0; j < sphere_table.rows(); ++j) {
            double dx = px - sphere_table(j, 0);
            double dy = py - sphere_table(j, 1);
            double dz = pz - sphere_table(j, 2);
            
            if ((dx*dx + dy*dy + dz*dz) < min_dist_sq) {
                collision = true;
                break;
            }
        }
        if (collision) continue;

        // 2. Check against ALREADY ACCEPTED peaks in this batch (Self-NMS)
        for (int idx : keep_indices) {
            double dx = px - (double)peaks(idx, 0);
            double dy = py - (double)peaks(idx, 1);
            double dz = pz - (double)peaks(idx, 2);

            if ((dx*dx + dy*dy + dz*dz) < min_dist_sq) {
                collision = true;
                break;
            }
        }

        if (!collision) {
            keep_indices.push_back(i);
        }
    }

    // --- Pack Result ---
    Eigen::MatrixX3i result(keep_indices.size(), 3);
    for (size_t k = 0; k < keep_indices.size(); ++k) {
        result.row(k) = peaks.row(keep_indices[k]);
    }
    return result;
}


// --- Append Sphere Table ---
Eigen::MatrixX4d append_sphere_table(
    const Eigen::MatrixX4d& existing_table,
    const VoxelGrid<float>& distance_field,
    const Eigen::MatrixX3i& peaks,          
    std::optional<int> min_radius_vox,
    std::optional<int> max_spheres) 
{
    if (peaks.rows() == 0) return existing_table;

    std::vector<Eigen::Vector4d> all_spheres;
    for (int i = 0; i < existing_table.rows(); ++i) {
        all_spheres.push_back(existing_table.row(i));
    }

    for (int i = 0; i < peaks.rows(); ++i) {
        if (max_spheres.has_value() && all_spheres.size() >= static_cast<size_t>(*max_spheres)) {
            std::cout<< "Max sphere count reached"<<std::endl;
            break;
        }

        Eigen::Vector3i coord_int = peaks.row(i);
        float dist_val = distance_field(coord_int.x(), coord_int.y(), coord_int.z());
        int radius_vox = static_cast<int>(std::round(dist_val));

        if (min_radius_vox.has_value() && radius_vox < *min_radius_vox) continue;

        int diameter_vox = 2 * radius_vox;
        Eigen::Vector3d direction = shift_voxel_center(distance_field, coord_int, 0.25f);
        Eigen::Vector3d shift_vec = 0.5 * direction;

        Eigen::Vector3d true_center = coord_int.cast<double>() + shift_vec;

        double true_radius = dist_val + shift_vec.norm();
        double true_diameter = true_radius * 2.0;


        all_spheres.push_back({true_center.x(), true_center.y(), true_center.z(), static_cast<double>(true_diameter)});
    }
    
    std::sort(all_spheres.begin(), all_spheres.end(), [](const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
        return a[3] > b[3];
    });

    Eigen::MatrixX4d updated_table(all_spheres.size(), 4);
    for (size_t i = 0; i < all_spheres.size(); ++i) updated_table.row(i) = all_spheres[i];
    return updated_table;
}

// --- Residual Distance Field ---
VoxelGrid<float> residual_distance_field(
    const VoxelGrid<float>& original_distance,
    const VoxelGrid<float>& spheres_distance) 
{
    if (original_distance.nx() != spheres_distance.nx() || 
        original_distance.ny() != spheres_distance.ny() || 
        original_distance.nz() != spheres_distance.nz()) {
        throw std::invalid_argument("Shapes must match.");
    }

    VoxelGrid<float> residual(original_distance.nx(), original_distance.ny(), original_distance.nz(), 
                              original_distance.voxel_size, original_distance.origin);

    #pragma omp parallel for
    for (size_t i = 0; i < residual.data.size(); ++i) {
        // R = D_orig - D_spheres
        float val = original_distance.data[i] - spheres_distance.data[i];
        residual.data[i] = (val > 0.0f) ? val : 0.0f;
    }
    return residual;
}

#endif