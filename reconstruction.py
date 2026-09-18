"""ASTRA cone-beam reconstruction + ground-truth comparison.

The reconstruction volume and the ground-truth occupancy grid always share the exact
same physical grid (same voxel count, same physical extents) by construction, so SSIM
is a fair, correctly-aligned comparison rather than an accident of matching parameters.

Ground truth is voxelized with our own GPU ray-mesh occupancy test (raytrace_gpu.py),
not trimesh's CPU implementation -- trimesh.contains() on a few hundred thousand points
took >10 minutes in testing; the GPU version does the same query in ~0.03s and is
needed anyway for the batch/adaptive-search workflow, which calls this many times.
"""

from __future__ import annotations

import astra
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

ALGORITHMS = {
    "SIRT3D_CUDA": {"iterative": True, "default_iterations": 100},
    "CGLS3D_CUDA": {"iterative": True, "default_iterations": 50},
    "FDK_CUDA": {"iterative": False, "default_iterations": None},
}


def make_vol_geom(vol_half: float, n_voxels: int):
    return astra.create_vol_geom(
        n_voxels, n_voxels, n_voxels,
        -vol_half, vol_half, -vol_half, vol_half, -vol_half, vol_half,
    )


def voxel_grid_centers(vol_half: float, n_voxels: int, center: np.ndarray) -> np.ndarray:
    """Physical coordinates of every voxel center, flattened (n_voxels**3, 3)."""
    step = 2.0 * vol_half / n_voxels
    xs = np.linspace(-vol_half, vol_half, n_voxels, endpoint=False) + step / 2.0
    grid = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1).reshape(-1, 3)
    return (grid.astype(np.float32) + np.asarray(center, dtype=np.float32))


def voxelize_ground_truth(
    vertices: np.ndarray, faces: np.ndarray, vol_half: float, n_voxels: int,
    center: np.ndarray, device: str,
) -> np.ndarray:
    """Binary occupancy ground truth, on the same grid used by `reconstruct`.

    Imports raytrace_gpu (and therefore warp-lang) lazily, on first call, so that just
    importing this module -- e.g. from the lightweight reconstruction-viewer app, which
    only needs `reconstruct`/`compute_ssim` and never calls this -- doesn't pull in
    Warp's ~350MB JIT toolchain for nothing.
    """
    import raytrace_gpu as rt

    projector = rt.GpuProjector(vertices, faces, device)
    points = voxel_grid_centers(vol_half, n_voxels, center)
    occ = projector.occupancy(points)
    return occ.reshape(n_voxels, n_voxels, n_voxels)


def reconstruct(
    sinogram: np.ndarray,
    astra_vectors: np.ndarray,
    rows: int,
    cols: int,
    vol_half: float,
    n_voxels: int,
    algorithm: str = "SIRT3D_CUDA",
    iterations: int = 100,
) -> np.ndarray:
    """sinogram: (n_views, rows, cols). astra_vectors: (n_views, 12), centered on the
    rotation center. Returns the reconstructed volume, shape (n_voxels,)*3.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}, expected one of {list(ALGORITHMS)}")

    vol_geom = make_vol_geom(vol_half, n_voxels)
    proj_geom = astra.create_proj_geom("cone_vec", rows, cols, astra_vectors)
    sino_astra = np.ascontiguousarray(np.transpose(sinogram, (1, 0, 2)), dtype=np.float32)

    proj_id = astra.data3d.create("-sino", proj_geom, sino_astra)
    rec_id = astra.data3d.create("-vol", vol_geom)
    alg_id = None
    try:
        cfg = astra.astra_dict(algorithm)
        cfg["ReconstructionDataId"] = rec_id
        cfg["ProjectionDataId"] = proj_id
        alg_id = astra.algorithm.create(cfg)
        if ALGORITHMS[algorithm]["iterative"]:
            astra.algorithm.run(alg_id, iterations)
        else:
            astra.algorithm.run(alg_id)
        rec = astra.data3d.get(rec_id)
    finally:
        if alg_id is not None:
            astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)
    return rec


def compute_ssim(ground_truth: np.ndarray, reconstruction: np.ndarray, mu: float) -> float:
    """Both volumes compared on a common 0..1 scale: ground truth is binary occupancy;
    the reconstruction (in units of mu * path_length) is divided by the known material
    mu and clipped, so a perfect reconstruction of the solid reads back as ~1 inside.
    """
    rec_norm = np.clip(reconstruction / mu, 0.0, 1.0).astype(np.float32)
    gt = ground_truth.astype(np.float32)
    return float(ssim_fn(gt, rec_norm, data_range=1.0))


def default_vol_half(mesh_radius: float, margin: float = 1.3) -> float:
    return float(mesh_radius * margin)
