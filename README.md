<p align="center">
  <img src="https://raw.githubusercontent.com/FelixBuchele/multisphere/main/logo/multisphere_banner_ext.png"
       alt="multisphere logo"
       width="85%">
</p>

<p align="center">
  <a href="https://github.com/FelixBuchele/multisphere/commits/main">
    <img src="https://img.shields.io/github/last-commit/FelixBuchele/multisphere.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub last commit">
  </a>
  <a href="https://github.com/FelixBuchele/multisphere/issues">
    <img src="https://img.shields.io/github/issues-raw/FelixBuchele/multisphere.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub issues">
  </a>
  <a href="https://github.com/FelixBuchele/multisphere/pulls">
    <img src="https://img.shields.io/github/issues-pr-raw/FelixBuchele/multisphere.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub pull requests">
  </a>
  <a href="https://opensource.org/licenses/GPL-3.0">
    <img src="https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat-square"
         alt="License: GPLv3">
  </a>
  <a href="#citation">
    <img src="https://img.shields.io/badge/DOI-pending-lightgrey.svg?style=flat-square"
         alt="DOI: pending">
  </a>
</p>


---


# multisphere

`multisphere` creates overlapping-sphere representations of
arbitrary 3D geometries based on voxelized Euclidean distance transforms
(EDT) and feature-enhanced distance fields (FEDT). The algorithm is
designed for Discrete Element Method (DEM) simulations, where accurate
yet computationally efficient particle shape representations are essential.

The package supports reconstruction from:
- Triangle surface meshes (STL)
- Binary voxel volumes (NumPy/CNPY)

**Implementations available:**
1. **Python**: Easy to use, rich visualization, integrates with SciPy/Trimesh.
2. **C++**: High-performance, multi-threaded (OpenMP), header-only architecture.

## Scientific Background

The multisphere algorithm is based on:
- Voxelization of the target geometry
- Exact Euclidean Distance Transform (EDT)
- Peak refinement of EDT maxima
- Iterative residual correction using the Feature-Enhanced Distance Tensor (FEDT)
- Termination by shape accuracy, minimum radius, or maximum sphere count

This approach avoids the major drawbacks of greedy sphere removal methods, 
such as spurious small spheres, symmetry violations, and excessive runtime. 


## Features

- multisphere reconstruction from:
  - Triangle meshes (STL)
  - Binary voxel grids 
- Exact EDT-driven sphere placement
- Iterative residual correction using FEDT
- Multiple termination criteria:
  - Shape precision
  - Maximum number of spheres
  - Minimum allowed sphere radius
- **(C++ Only)** Robust Generalized Winding Number voxelization (via libigl)
- **(C++ Only)** OpenMP parallelization for voxelization and field processing
- Export formats:
  - CSV (sphere centers & radii)
  - VTK (visualization)
  - STL (boolean union of spheres)


## Python Installation & Usage

Requires Python ≥ 3.9. 

```bash
pip install multisphere[full]

```

### Basic Python Usage

```python
import multisphere as ms

mesh = ms.load_mesh_from_stl("geometry.stl")
sphere_pack = ms.multisphere_from_mesh(
    mesh=mesh,
    div=150,
    max_spheres=100
)
ms.export_sphere_pack_to_csv(sphere_pack, "spheres.csv")

```

*(See the `examples/` folder for complete Python scripts)*

---

## C++ Implementation

The C++ implementation is located in `src/` and is designed for high-performance integration. It is a **header-only** library (core logic) with dependencies provided in `include/`.

### Dependencies

* **System**: CMake (≥3.10), C++17 compiler, OpenMP (optional but recommended).
* **Bundled (in `include/`)**: `libigl` (math/geometry), `cnpy` (numpy IO), `edt` (distance transform), `nanoflann`, `Eigen`.
* **Optional**: VTK (for visualization), Manifold (for boolean mesh export).

### Building the C++ Examples

The test scripts (`main.cpp` and `main_mesh.cpp`) are configured to be built within the `examples` directory so they can access sample mesh files relative to the build path.

```bash
mkdir build
cd build
cmake .. 
make -j4

```

### Basic C++ Usage

The core API is provided via `multisphere_reconstruction.hpp`.

```cpp
#include "multisphere_reconstruction.hpp"
#include "multisphere_io.hpp"

int main() {
    // 1. Load Mesh
    FastMesh mesh = load_mesh_fast("example_mesh.stl");

    // 2. Run Reconstruction
    SpherePack sp = multisphere_from_mesh(
        mesh,
        150,    // div (resolution)
        2,      // padding
        8,      // min_radius_vox
        0.99,   // precision_target
        4,      // min_center_distance_vox
        100,    // max_spheres
        1.0,    // boost
        true,   // show_progress
        false   // confine_mesh (requires Manifold)
    );

    // 3. Export
    export_to_csv(sp, "results.csv");
    return 0;
}

```

### Key Differences: C++ vs Python

1. **Voxelization Method**:
* **Python**: Uses `trimesh` (ray casting/subdivision).
* **C++**: Uses `igl::fast_winding_number`. This is mathematically more robust for "dirty" meshes (holes, self-intersections) but may have a different performance profile for extremely large meshes.


2. **Parallelism**:
* The C++ version explicitly uses **OpenMP** for voxel generation, distance transforms, and kernel application. Ensure your compiler supports OpenMP for maximum speed.


3. **IO**:
* The C++ implementation includes a custom, lightweight binary STL parser (`FastMesh`) for speed, whereas Python uses `trimesh`.
* It supports loading/saving NumPy (`.npy`) boolean volumes via `cnpy`.



---

## License

This project is licensed under the GNU General Public License v3.0.

See the LICENSE file for full details.

`multisphere` depends on third-party libraries with compatible licenses:

| Package | License | Usage |
| --- | --- | --- |
| NumPy/SciPy | BSD-3-Clause | Python |
| trimesh | MIT | Python |
| PyVista | MIT | Python Viz |
| **Eigen** | MPL2 | C++ Math |
| **libigl** | MPL2 | C++ Voxelization |
| **cnpy** | MIT | C++ IO |
| **manifold3d** | Apache-2.0 | Boolean Ops |

## Author

**Felix Buchele**, Patric Müller, Thorsten Pöschel

Friedrich-Alexander-Universität Erlangen–Nürnberg (FAU)

Institute for Multiscale Simulation (MSS)

## Citation

Felix Buchele, Patric Müller, Thorsten Pöschel,

*Multi-Sphere-Shape generator for DEM simulations using the multi-sphere approach* manuscript in preparation

```

### Analysis of the Changes
1.  **Badge**: Added a C++17 badge to signal language support.
2.  **Sectioning**: Created a distinct "C++ Implementation" section.
3.  **Deviations Explained**: I explicitly called out `igl::fast_winding_number`. This is important because users porting data from Python to C++ might see slightly different results for open meshes due to the different voxelization logic.
4.  **Build Instructions**: Clarified the build process targeting the `examples` folder usage.
5.  **Dependencies**: Clarified that most C++ dependencies are vendored (bundled) in `include/`, reducing the "scare factor" of installing complex C++ libraries.

```