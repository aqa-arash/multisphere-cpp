/**
 * @file main_shape_validation.cpp
 * @brief Validates multisphere-cpp physics computation for common 3D shapes.
 *
 * Generates synthetic voxel representations of sphere, box, cylinder, cone, and hemisphere,
 * runs multisphere reconstruction with compute_physics=1 and 2, and compares results to analytical values.
 *
 * Usage: Compile and run. Results are printed to stdout.
 *
 * Author: Arash Moradian
 * Date: 2026-04-10
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "GEMSS/GEMSS-interface.h"

using namespace GEMSS;
using std::cout;
using std::endl;

struct AnalyticalResult {
    double volume;
    Eigen::Vector3d center_of_mass;
    Eigen::Vector3d principal_moments;
};

// Utility: Print comparison and write CSV row (with principal axes)
void print_comparison_and_write_csv(const std::string& shape, int compute_physics, const SpherePack& sp, const AnalyticalResult& ref, std::ostream& csv) {
    cout << "\n==== " << shape << " (compute_physics=" << compute_physics << ") ====" << endl;
    cout << "Number of spheres: " << sp.num_spheres() << endl;
    cout << "Precision: " << sp.precision << endl;
    cout << "Analytical Volume: " << ref.volume << ", Reconstructed: " << sp.volume << endl;
    cout << "Analytical COM: " << ref.center_of_mass.transpose() << ", Reconstructed: " << sp.center_of_mass.transpose() << endl;
    cout << "Analytical Moments: " << ref.principal_moments.transpose() << ", Reconstructed: " << sp.principal_moments.transpose() << endl;
    cout << "Analytical Principal Axes (columns):\n";
    cout << Eigen::Matrix3d::Identity() << endl;
    cout << "Reconstructed Principal Axes (columns):\n" << sp.principal_axes << endl;

    csv << shape << "," << compute_physics << ","
        << ref.volume << "," << sp.volume << ","
        << ref.center_of_mass.transpose() << "," << sp.center_of_mass.transpose() << ","
        << ref.principal_moments.transpose() << "," << sp.principal_moments.transpose();
    // Analytical axes (identity, 9 values)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            csv << "," << (i == j ? 1 : 0);
    // Reconstructed axes (9 values)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            csv << "," << sp.principal_axes(i, j);
    csv << std::endl;
}

// Voxelize a sphere
VoxelGrid<uint8_t> make_voxel_sphere(int nx, int ny, int nz, double v_size, double cx, double cy, double cz, double r) {
    VoxelGrid<uint8_t> grid(nx, ny, nz, v_size);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z) {
                double X = x * v_size, Y = y * v_size, Z = z * v_size;
                double d2 = (X-cx)*(X-cx) + (Y-cy)*(Y-cy) + (Z-cz)*(Z-cz);
                if (d2 <= r*r) grid(x, y, z) = true;
            }
    return grid;
}

// Voxelize a box
VoxelGrid<uint8_t> make_voxel_box(int nx, int ny, int nz, double v_size, int min_x, int max_x, int min_y, int max_y, int min_z, int max_z) {
    VoxelGrid<uint8_t> grid(nx, ny, nz, v_size);
    for (int x = min_x; x < max_x; ++x)
        for (int y = min_y; y < max_y; ++y)
            for (int z = min_z; z < max_z; ++z)
                grid(x, y, z) = true;
    return grid;
}

// Voxelize a cylinder (aligned with z)
VoxelGrid<uint8_t> make_voxel_cylinder(int nx, int ny, int nz, double v_size, double cx, double cy, double r, int min_z, int max_z) {
    VoxelGrid<uint8_t> grid(nx, ny, nz, v_size);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = min_z; z < max_z; ++z) {
                double X = x * v_size, Y = y * v_size;
                double d2 = (X-cx)*(X-cx) + (Y-cy)*(Y-cy);
                if (d2 <= r*r) grid(x, y, z) = true;
            }
    return grid;
}

// Voxelize a cone (aligned with z, apex at min_z, base at max_z-1)
VoxelGrid<uint8_t> make_voxel_cone(int nx, int ny, int nz, double v_size, double cx, double cy, double r_base, int min_z, int max_z) {
    VoxelGrid<uint8_t> grid(nx, ny, nz, v_size);
    int h_vox = max_z - min_z;
    if (h_vox <= 1) return grid;
    double h = (h_vox - 1) * v_size;
    double z_apex = min_z * v_size;
    for (int z = min_z; z < max_z; ++z) {
        double Z = z * v_size;
        double frac = (h > 0) ? (Z - z_apex) / h : 0.0;
        if (frac < 0.0 || frac > 1.0) continue;
        double r = r_base * frac; // r=0 at apex, r_base at base
        for (int x = 0; x < nx; ++x)
            for (int y = 0; y < ny; ++y) {
                double X = x * v_size, Y = y * v_size;
                double d2 = (X-cx)*(X-cx) + (Y-cy)*(Y-cy);
                if (d2 <= r*r) grid(x, y, z) = true;
            }
    }
    return grid;
}

// Voxelize a hemisphere (z >= center)
VoxelGrid<uint8_t> make_voxel_hemisphere(int nx, int ny, int nz, double v_size, double cx, double cy, double cz, double r) {
    VoxelGrid<uint8_t> grid(nx, ny, nz, v_size);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z) {
                double X = x * v_size, Y = y * v_size, Z = z * v_size;
                double d2 = (X-cx)*(X-cx) + (Y-cy)*(Y-cy) + (Z-cz)*(Z-cz);
                if (d2 <= r*r && Z >= cz) grid(x, y, z) = true;
            }
    return grid;
}

// Analytical values for shapes (unit density)
AnalyticalResult analytical_sphere(double r, double cx, double cy, double cz) {
    double V = (4.0/3.0) * M_PI * pow(r,3);
    double I = (2.0/5.0) * V * r * r; // All axes
    return {V, Eigen::Vector3d(cx, cy, cz), Eigen::Vector3d(I, I, I)};
}
AnalyticalResult analytical_box(double lx, double ly, double lz, double cx, double cy, double cz) {
    double V = lx*ly*lz;
    double Ixx = (1.0/12.0) * V * (ly*ly + lz*lz);
    double Iyy = (1.0/12.0) * V * (lx*lx + lz*lz);
    double Izz = (1.0/12.0) * V * (lx*lx + ly*ly);
    return {V, Eigen::Vector3d(cx, cy, cz), Eigen::Vector3d(Ixx, Iyy, Izz)};
}
AnalyticalResult analytical_cylinder(double r, double h, double cx, double cy, double cz) {
    double V = M_PI * r * r * h;
    double Izz = 0.5 * V * r * r;
    double Ixx = (1.0/12.0) * V * (3*r*r + h*h);
    return {V, Eigen::Vector3d(cx, cy, cz), Eigen::Vector3d(Ixx, Ixx, Izz)};
}
// For cone: apex at z_apex, base at z_base, height h. Center of mass at z_apex + 3/4*h
AnalyticalResult analytical_cone(double r_base, double h, double cx, double cy, double z_apex) {
    double V = (1.0/3.0) * M_PI * r_base * r_base * h;
    double cz_cm = z_apex + 0.75 * h;
    // Moments about center of mass (from apex):
    // Izz = (3/10) * m * r^2
    // Ixx = (3/20) * m * r^2 + (3/5) * m * h^2
    double Izz = (3.0/10.0) * V * r_base * r_base;
    double Ixx = (3.0/20.0) * V * r_base * r_base + (3.0/80.0) * V * h * h;
    return {V, Eigen::Vector3d(cx, cy, cz_cm), Eigen::Vector3d(Ixx, Ixx, Izz)};
}

AnalyticalResult analytical_hemisphere(double r, double cx, double cy, double cz) {
    double V = (2.0/3.0) * M_PI * pow(r,3);
    double I = (83.0/320.0) * V * r * r; // About base, approx
    return {V, Eigen::Vector3d(cx, cy, cz + (3.0/8.0)*r), Eigen::Vector3d(I, I, (2.0/5.0)*V*r*r)};
}

int main() {
    cout << "--- Multisphere Shape Validation ---" << endl;
    std::string outdir = "../examples/shape_validation_results/";
    double v_size = 0.05;
    int nx = 200, ny = 200, nz = 200;
    double cx = nx*v_size/2, cy = ny*v_size/2, cz = nz*v_size/2;

    // Sphere
    double r_sphere = 4.0;
    double cx_sphere = cx, cy_sphere = cy, cz_sphere = cz;
    auto vox_sphere = make_voxel_sphere(nx, ny, nz, v_size, cx_sphere, cy_sphere, cz_sphere, r_sphere);
    auto ref_sphere = analytical_sphere(r_sphere, cx_sphere, cy_sphere, cz_sphere);

    // Box
    int min_b = 60, max_b = 140;
    double l_box = (max_b - min_b) * v_size;
    double cx_box = ((min_b + max_b - 1) / 2.0) * v_size;
    double cy_box = cx_box, cz_box = cx_box;
    auto vox_box = make_voxel_box(nx, ny, nz, v_size, min_b, max_b, min_b, max_b, min_b, max_b);
    auto ref_box = analytical_box(l_box, l_box, l_box, cx_box, cy_box, cz_box);

    // Cylinder (aligned with z)
    double r_cyl = 2.0;
    int min_z_cyl = 10, max_z_cyl = 190;
    double h_cyl = (max_z_cyl - min_z_cyl) * v_size;
    double cx_cyl = cx, cy_cyl = cy;
    double cz_cyl = ((min_z_cyl + max_z_cyl - 1) / 2.0) * v_size;
    auto vox_cyl = make_voxel_cylinder(nx, ny, nz, v_size, cx_cyl, cy_cyl, r_cyl, min_z_cyl, max_z_cyl);
    auto ref_cyl = analytical_cylinder(r_cyl, h_cyl, cx_cyl, cy_cyl, cz_cyl);

    // --- Cone test (fully rewritten) ---
    double r_cone = 3.0;
    int min_z_cone = 50, max_z_cone = 150;
    double cx_cone = cx, cy_cone = cy;
    int h_vox_cone = max_z_cone - min_z_cone;
    double h_cone = (h_vox_cone - 1) * v_size;
    double z_apex_cone = min_z_cone * v_size;
    auto vox_cone = make_voxel_cone(nx, ny, nz, v_size, cx_cone, cy_cone, r_cone, min_z_cone, max_z_cone);
    auto ref_cone = analytical_cone(r_cone, h_cone, cx_cone, cy_cone, z_apex_cone);

    // Hemisphere (z >= center)
    double r_hemi = 4.0;
    double cx_hemi = cx, cy_hemi = cy, cz_hemi = cz;
    auto vox_hemi = make_voxel_hemisphere(nx, ny, nz, v_size, cx_hemi, cy_hemi, cz_hemi, r_hemi);
    auto ref_hemi = analytical_hemisphere(r_hemi, cx_hemi, cy_hemi, cz_hemi);

    // Config
    MultisphereConfig config;
    config.search_window = 5;
    config.min_radius_vox = 5;
    config.precision_target = 0.99f;
    config.max_spheres = 2000;
    config.radius_offset_vox = 0.0f;
    config.persistence = 3;
    config.show_progress = false;
    config.confine_mesh = false;

    // CSV output
    std::string csv_path = outdir + "shape_validation_results.csv";
    std::ofstream csv(csv_path);
    csv << "shape,compute_physics,analytical_volume,reconstructed_volume,analytical_com_x,analytical_com_y,analytical_com_z,reconstructed_com_x,reconstructed_com_y,reconstructed_com_z,analytical_Ix,analytical_Iy,analytical_Iz,reconstructed_Ix,reconstructed_Iy,reconstructed_Iz"
        ",analytical_axis_00,analytical_axis_01,analytical_axis_02,analytical_axis_10,analytical_axis_11,analytical_axis_12,analytical_axis_20,analytical_axis_21,analytical_axis_22"
        ",recon_axis_00,recon_axis_01,recon_axis_02,recon_axis_10,recon_axis_11,recon_axis_12,recon_axis_20,recon_axis_21,recon_axis_22" << std::endl;

    // For each shape, run with compute_physics=1 and 2
    struct ShapeCase {
        std::string name;
        VoxelGrid<uint8_t>* vox;
        AnalyticalResult* ref;
    } cases[] = {
        {"Sphere", &vox_sphere, &ref_sphere},
        {"Box", &vox_box, &ref_box},
        {"Cylinder", &vox_cyl, &ref_cyl},
        {"Cone", &vox_cone, &ref_cone},
        {"Hemisphere", &vox_hemi, &ref_hemi},
    };

    for (const auto& shape : cases) {
        // Save original mesh for each shape
        std::string mesh_path = outdir + shape.name + "_original.stl";
        save_mesh_to_stl(grid_to_mesh(*shape.vox), mesh_path);
        for (int cp : {1, 2}) {
            config.compute_physics = cp;
            SpherePack sp = multisphere_from_voxels(*shape.vox, config);
            print_comparison_and_write_csv(shape.name, cp, sp, *shape.ref, csv);
            // Save reconstructed mesh and CSV
            std::string recon_csv = outdir + shape.name + "_recon_cp" + std::to_string(cp) + ".csv";
            std::string recon_vtk = outdir + shape.name + "_recon_cp" + std::to_string(cp) + ".vtk";
            export_to_csv(sp, recon_csv);
            export_to_vtk(sp, recon_vtk);
        }
    }
    csv.close();
    cout << "\nValidation complete. Results saved to " << outdir << endl;
    return 0;
}
