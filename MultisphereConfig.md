# MultisphereConfig Structure Documentation

This document explains the `MultisphereConfig` structure found in `src/GEMSS/GEMSS_config.hpp`. This structure contains configuration parameters for multisphere reconstruction in the GEMSS library.

## Structure Definition

```
struct MultisphereConfig {
    // --- Required for mesh reconstruction ---
    int div = 100;               // grid resolution
    int padding = 2;             // grid padding
    float minimum_radius_real = 0.0f; // min sphere radius (real units)
    bool confine_mesh = false;   // confine to mesh boundary

    // --- Peak detection and filtering ---
    float min_center_distance_rel = 0.5f; // min center distance (relative)
    int search_window = 2;       // neighbor distance for peaks
    float radius_offset_vox = 0.5f; // radius correction


    // --- Convergence & limits ---
    int min_radius_vox = 2;      // min sphere radius (voxels)
    float precision_target = 1.0f; // target precision
    int max_spheres = 0;         // max spheres (0 = unlimited)
    int persistence = 2;         // solver persistence

    // --- Utilities & prior state ---
    int compute_physics = 2;     // compute physics properties
    bool prune_isolated_spheres = false; // remove isolated spheres
    bool show_progress = true;   // print progress
    Eigen::MatrixX4f initial_sphere_table = Eigen::MatrixX4f(0, 4); // initial spheres
    
};

```

## Field Descriptions

### Required parameters for reconstruction from mesh
- **div**: Voxel grid division (resolution).
- **padding**: Voxel grid padding.
- **minimum_radius_real**: Minimum sphere radius in real units (e.g., millimeters). If not set, voxel units will be used and the `min_radius_vox` parameter will be required.
- **confine_mesh**: Confine spheres strictly to the mesh boundary SDF.

### Peak Detection & Filtering
- **min_center_distance_rel**: Minimum center distance relative to radius (e.g., 0.5 means centers must be at least half a radius apart).

### Convergence & Limits
- **min_radius_vox**: Minimum sphere radius in voxels. Default is 2 voxels, which allows for capturing small features while avoiding excessive numbers of tiny spheres due to discretization error. Adjust based on the expected feature size in your input data and desired level of detail.
- **precision_target**: Target voxel overlap precision [0, 1]. Note: 1 is impossible to reach due to discretization; set to 0.99 or lower for practical purposes. Adjust based on the desired balance between reconstruction quality and runtime.
- **max_spheres**: Maximum allowed spheres in the reconstruction (0 for unlimited).

### Utilities & Prior State
- **compute_physics**: Compute volume, center of mass, and inertia tensor. 0 = false, 1 = compute based on reconstruction, 2 = compute based on original mesh (if available).
- **prune_isolated_spheres**: Remove spheres that are not touching the biggest network of spheres.
- **show_progress**: Print console progress.
- **initial_sphere_table**: Prior solver state, can be in voxel units (if passed to `multisphere_from_voxels`) or physical units (if passed to `multisphere_from_mesh`). This is a matrix of shape (N, 4) where each row is (center_x, center_y, center_z, radius). If provided, the solver will start with this initial configuration instead of an empty table. Adjust based on whether you have a good initial guess for the sphere configuration that can speed up convergence.

### Magic numbers (Advanced)
- **radius_offset_vox**: Correction factor added to radius during peak detection to better fit the original shape. Default is approximately half of a unit voxel, which empirically improves reconstruction quality. Adjust with caution.
- **search_window**: Minimum distance of neighbors in voxels to consider during peak detection. This is a hard threshold that can be used to smooth out small fluctuation in the distance field and reduce the number of spheres, but setting it too high may cause missed detections in thin structures. Default is 2 voxels, to avoid discretization noise while still allowing for close peaks in thin structures. Adjust based on the voxel resolution and expected feature size in the input data.
- **persistence**: Maximum number of iterations for the solver to increase the weight before giving up. Used only when `precision_target` or `max_spheres` is not reached. Can cause excessive runtime; set to 1 to turn off. Default is 2, which allows for increasing the weight if no peak is found for 1 iteration, but can be increased for more challenging reconstructions at the cost of runtime. Adjust based on the complexity of your input data and desired precision.

---

For further details, see the source code in `src/GEMSS/GEMSS_config.hpp`.
