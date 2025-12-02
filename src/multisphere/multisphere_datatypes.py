from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.ndimage as ndi


@dataclass
class SpherePack:
    """Collection of spheres approximating a geometry.

    Attributes
    ----------
    centers : np.ndarray
        Array of shape (N, 3) with sphere center coordinates.
    radii : np.ndarray
        Array of shape (N,) with sphere radii.
    """

    centers: np.ndarray
    radii: np.ndarray

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=float)
        self.radii = np.asarray(self.radii, dtype=float).ravel()

        if self.centers.ndim != 2 or self.centers.shape[1] != 3:
            raise ValueError("centers must have shape (N, 3) where N is the number of spheres with their XYZ coordinates.")
        if self.radii.ndim != 1:
            raise ValueError("radii must be 1D.")
        if self.centers.shape[0] != self.radii.shape[0]:
            raise ValueError("centers and radii must have same length.")

    @property
    def num_spheres(self) -> int:
        return int(self.radii.size)

    @property
    def min_radius(self) -> float:
        return float(self.radii.min()) if self.radii.size else 0.0

    @property
    def max_radius(self) -> float:
        return float(self.radii.max()) if self.radii.size else 0.0




@dataclass
class VoxelGrid:
    """
    3D voxel grid used for distance transforms and reconstruction.

    Attributes
    ----------
    data : np.ndarray
        3D array. Boolean or numeric.
    voxel_size : float, optional
        Physical spacing per voxel (same in all directions).
    origin : np.ndarray, optional
        World-space coordinates of voxel (0, 0, 0).
    """
    data: np.ndarray
    voxel_size: float = 1.0
    origin: np.ndarray | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.data)
        if arr.ndim != 3:
            raise ValueError(
                f"VoxelGrid.data must be 3D, got shape {arr.shape}."
            )
        self.data = arr
        if self.origin is None:
            self.origin = np.zeros(3, dtype=float)
        else:
            self.origin = np.asarray(self.origin, dtype=float).ravel()
            if self.origin.size != 3:
                raise ValueError("origin must be length-3 if provided.")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    # Core operations
    def distance_transform(self) -> "VoxelGrid":
        """Return a new VoxelGrid containing the EDT of the binary mask."""
        # We assume >0 is "inside"
        kernel = (self.data > 0).astype(np.uint8)
        padded = np.pad(kernel, pad_width=1, mode="constant", constant_values=0)
        dist = ndi.distance_transform_edt(padded)
        # Remove padding
        dist = dist[1:-1, 1:-1, 1:-1]
        return VoxelGrid(dist, voxel_size=self.voxel_size, origin=self.origin)

    # Constructors / factories
    @classmethod
    def empty(
        cls,
        shape: tuple[int, int, int],
        voxel_size: float = 1.0,
        origin: np.ndarray | None = None,
        dtype=bool,
        ) -> "VoxelGrid":
        """
        Create an empty voxel grid with the given shape.

        Parameters
        ----------
        shape : tuple of int
            Grid shape (nx, ny, nz).
        voxel_size : float, optional
            Physical size of one voxel.
        origin : array-like, optional
            Origin of the grid in world coordinates.
        dtype : data-type, optional
            Data type of the grid (default: bool).

        Returns
        -------
        VoxelGrid
            Grid filled with zeros (False for bool).
        """
        if len(shape) != 3:
            raise ValueError(
                f"shape must be length 3, got {shape}."
            )
        data = np.zeros(shape, dtype=dtype)
        return cls(data=data, voxel_size=voxel_size, origin=origin)



    @classmethod
    def sphere_kernel(
        cls,
        diameter: int,
        voxel_size: float = 1.0,
        origin: np.ndarray | None = None,
        dtype=bool,
    ) -> "VoxelGrid":
        """
        Create a boolean cubic voxel grid containing a centered spherical kernel.

        Parameters
        ----------
        diameter : int
            Sphere diameter in voxels. Grid shape is (d, d, d).
        voxel_size : float, optional
            Physical voxel size.
        origin : array-like, optional
            Origin in world coordinates.
        dtype : data-type, optional
            Data type of the grid (bool for mask, float for EDT, etc.).

        Returns
        -------
        VoxelGrid
            Grid with a centered spherical region.
        """
        if diameter <= 0:
            raise ValueError("diameter must be positive.")

        size = int(diameter)
        shape = (size, size, size)
        data = np.zeros(shape, dtype=float)

        # center index (float) to allow exact sphere
        center = np.array(
            [((size + 1) / 2.0) - 1.0] * 3,
            dtype=float,
        )
        coords = np.ogrid[:size, :size, :size]
        dist = np.sqrt(
            (coords[0] - center[0]) ** 2
            + (coords[1] - center[1]) ** 2
            + (coords[2] - center[2]) ** 2
        )
        data[dist <= size / 2.0] = 1.0

        if dtype is bool:
            data = data.astype(bool)
        else:
            data = data.astype(dtype)

        return cls(data=data, voxel_size=voxel_size, origin=origin)
