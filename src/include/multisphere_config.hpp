#ifndef MULTISPHERE_CONFIG_HPP
#define MULTISPHERE_CONFIG_HPP

#include <optional>
#include <Eigen/Dense>

namespace MSS {

/**
 * @brief Configuration parameters for multisphere reconstruction.
 */
struct MultisphereConfig {
    // --- Grid Generation (Used only by multisphere_from_mesh) ---
    int div = 100;               ///< Voxel grid division (resolution)
    int padding = 2;             ///< Voxel grid padding
    bool confine_mesh = false;   ///< Confine spheres strictly to the mesh boundary SDF

    // --- Peak Detection & Filtering ---
    float min_center_distance_rel = 0.5f;              ///< Minimum center distance relative to radius (e.g., 0.5 means centers must be at least half a radius apart)
    std::optional<int> min_radius_vox = std::nullopt; ///< Minimum sphere radius in voxels

    // --- Convergence & Limits ---
    std::optional<float> precision_target = std::nullopt; ///< Target voxel overlap precision [0, 1]
    std::optional<int> max_spheres = std::nullopt;        ///< Maximum allowed spheres

    // --- Utilities & Prior State ---
    int compute_physics = 0; ///< Compute volume, CoM, and inertia tensor 0 = false, 1 = Compute based on reconstruction, 2 = compute based on original mesh (if available)
    bool prune_isolated_spheres = false; ///< Remove spheres that are not touching the biggest network of spheres
    bool show_progress = true;    ///< Print console progress
    std::optional<Eigen::MatrixX4f> initial_sphere_table = std::nullopt; ///< Prior solver state


    // --- Magic numbers (Don't touch unless you know what you're doing) ---
    float radius_offset_vox = 0.87f; ///< Correction factor added to radius during peak detection to better fit the original shape. Default is approximately the half-diagonal of a unit voxel sqrt(3)/2, which empirically improves reconstruction quality. Adjust with caution. 
    int search_window = 2; ///< Minimum distance of neighbors in voxels to consider during peak detection. This is a hard threshold that can be used to smooth out small fluctuation in the distance field and reduce the number of spheres, but setting it too high may cause missed detections in thin structures. Default is 2 voxels, to avoid discretization noise while still allowing for close peaks in thin structures. Adjust based on the voxel resolution and expected feature size in the input data.
    int persistence = 2; ///< Maximum number of iterations for the solver to increase the weight before giving up. Used only when precision_target or max_sphere is not reached. Can cause excessive runtime, set to 1 to turn off. Default is 2, which allows for increasing the weight if no peak is found for 1 iteration, but can be increased for more challenging reconstructions at the cost of runtime. Adjust based on the complexity of your input data and desired precision.

};

} // namespace MSS

#endif // MULTISPHERE_CONFIG_HPP