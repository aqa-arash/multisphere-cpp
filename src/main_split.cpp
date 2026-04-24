#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

// Project headers
#include "GEMSS/GEMSS-interface.h"
#include "GEMSS/GEMSS_split.hpp"

using namespace GEMSS;

/**
 * @brief Entry point for mesh-based multisphere splitting and fragment reconstruction.
 *
 * Loads a mesh, reconstructs a SpherePack, splits it by a plane through its center of mass, reconstructs fragments, and exports all to VTK.
 *
 * @return int Exit code.
 */
int main() {
	std::cout << "--- Multisphere Split & Fragment Reconstruction Test ---" << std::endl;

	// Mesh file to process
	std::string model_name = "ice_cream_cone.stl";

	// 1. Load mesh from file
	STLMesh example_mesh = load_mesh(model_name);

	// 2. Run the Reconstruction Algorithm for the whole mesh
	GEMSS::MultisphereConfig config;
	config.div = 400;
	config.padding = 2;
	config.search_window = 10;
	config.min_center_distance_rel = 0.5f;
	config.min_radius_vox = 20;
	config.precision_target = 0.99f;
	config.max_spheres = 10000;
	config.show_progress = true;
	config.confine_mesh = false;
	config.initial_sphere_table = Eigen::MatrixXf(0,4);
	config.compute_physics = 1;
	config.prune_isolated_spheres = true;

	SpherePack sp = multisphere_from_mesh(example_mesh, config);

	// Export original SpherePack
	export_to_vtk(sp, model_name + "_original.vtk");
	std::cout << "Original SpherePack exported to: " << model_name + "_original.vtk" << std::endl;

	// 3. Split by a plane through center of mass, normal along z (xy plane)
	Eigen::Vector3f com = sp.center_of_mass;
	Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
	std::cout << "Splitting at center of mass: " << com.transpose() << ", normal: " << normal.transpose() << std::endl;

	auto split_result = split_sp(sp, normal, com, config);
	const std::vector<SpherePack>& fragments = split_result.first;
	const VoxelGrid<uint8_t>& labeled_grid = split_result.second;

	// Export labeled voxel grid (optional)
	// save_voxel_grid_to_vtk(labeled_grid, model_name + "_labeled_grid.vtk"); // Uncomment if function exists

	// 4. Export all fragments to VTK
	for (size_t i = 0; i < fragments.size(); ++i) {
		std::string frag_name = model_name + "_fragment_" + std::to_string(i+1) + ".vtk";
		export_to_vtk(fragments[i], frag_name);
		std::cout << "Fragment " << (i+1) << " exported to: " << frag_name << std::endl;
		std::cout << "  Spheres: " << fragments[i].num_spheres() << ", Volume: " << fragments[i].volume << std::endl;
	}

	std::cout << "All fragments and original SpherePack exported to VTK." << std::endl;
	return 0;
}
