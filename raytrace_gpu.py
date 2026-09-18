"""GPU-accelerated cone-beam projection generator.

Computes X-ray path length through a triangle mesh by analytic ray-mesh intersection
(Warp's BVH-accelerated mesh_query_ray) -- no voxelization of the sample, so this stays
independent from the reconstruction's discretization (avoids the "inverse crime").
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.init()

MAX_HITS_DEFAULT = 128
EPS_DEFAULT = 1.0e-4


@wp.kernel
def _trace_batch_kernel(
    mesh_id: wp.uint64,
    src: wp.array(dtype=wp.vec3),
    det_center: wp.array(dtype=wp.vec3),
    u_vec: wp.array(dtype=wp.vec3),
    v_vec: wp.array(dtype=wp.vec3),
    rows: int,
    cols: int,
    max_hits: int,
    eps: float,
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    pix_per_view = rows * cols
    view = tid // pix_per_view
    local = tid % pix_per_view
    row = local // cols
    col = local % cols

    s = src[view]
    dc = det_center[view]
    u = u_vec[view]
    v = v_vec[view]

    col_off = float(col) - float(cols - 1) * 0.5
    row_off = float(row) - float(rows - 1) * 0.5
    px = dc + u * col_off + v * row_off

    diff = px - s
    dist = wp.length(diff)
    dir = diff / dist

    path_length = float(0.0)
    inside = int(0)
    entry_t = float(0.0)
    t_cursor = float(0.0)

    for _ in range(max_hits):
        remaining = dist - t_cursor
        if remaining <= eps:
            break
        hit = wp.mesh_query_ray(mesh_id, s + dir * t_cursor, dir, remaining)
        if not hit.result:
            break
        t_hit = t_cursor + hit.t
        if inside == 0:
            inside = int(1)
            entry_t = t_hit
        else:
            inside = int(0)
            path_length += t_hit - entry_t
        t_cursor = t_hit + eps

    out[tid] = path_length


@wp.kernel
def _occupancy_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    count = wp.mesh_query_ray_count_intersections(mesh_id, points[tid], wp.vec3(1.0, 0.0, 0.0))
    out[tid] = float(count % 2)


class GpuProjector:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, device: str):
        self.device = device
        points = wp.array(np.ascontiguousarray(vertices, dtype=np.float32), dtype=wp.vec3, device=device)
        indices = wp.array(np.ascontiguousarray(faces.ravel(), dtype=np.int32), dtype=wp.int32, device=device)
        self.mesh = wp.Mesh(points=points, indices=indices)

    def trace_views(
        self,
        views,
        rows: int,
        cols: int,
        pixel_pitch: float,
        max_hits: int = MAX_HITS_DEFAULT,
        eps: float = EPS_DEFAULT,
    ) -> np.ndarray:
        """views: list of (src, det_center, u_hat, v_hat) world-space tuples (unit u_hat/v_hat).

        Returns path-length array, shape (len(views), rows, cols), in the mesh's length units.
        """
        n_views = len(views)
        src_np = np.array([v[0] for v in views], dtype=np.float32)
        det_np = np.array([v[1] for v in views], dtype=np.float32)
        u_np = np.array([v[2] * pixel_pitch for v in views], dtype=np.float32)
        v_np = np.array([v[3] * pixel_pitch for v in views], dtype=np.float32)

        src_wp = wp.array(src_np, dtype=wp.vec3, device=self.device)
        det_wp = wp.array(det_np, dtype=wp.vec3, device=self.device)
        u_wp = wp.array(u_np, dtype=wp.vec3, device=self.device)
        v_wp = wp.array(v_np, dtype=wp.vec3, device=self.device)

        out = wp.zeros(n_views * rows * cols, dtype=wp.float32, device=self.device)

        wp.launch(
            _trace_batch_kernel,
            dim=n_views * rows * cols,
            inputs=[self.mesh.id, src_wp, det_wp, u_wp, v_wp, rows, cols, max_hits, eps],
            outputs=[out],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        return out.numpy().reshape(n_views, rows, cols)

    def occupancy(self, points: np.ndarray) -> np.ndarray:
        """GPU inside/outside test for arbitrary points (e.g. a reconstruction voxel
        grid), via ray-parity counting. Requires a watertight mesh. Returns a bool
        array, shape (len(points),).
        """
        pts_wp = wp.array(np.ascontiguousarray(points, dtype=np.float32), dtype=wp.vec3, device=self.device)
        out = wp.zeros(len(points), dtype=wp.float32, device=self.device)
        wp.launch(
            _occupancy_kernel,
            dim=len(points),
            inputs=[self.mesh.id, pts_wp],
            outputs=[out],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        return out.numpy().astype(bool)


def generate_sinogram(
    vertices: np.ndarray,
    faces: np.ndarray,
    views,
    rows: int,
    cols: int,
    pixel_pitch: float,
    mu: float,
    device: str,
    chunk_size: int = 25,
    max_hits: int = MAX_HITS_DEFAULT,
    eps: float = EPS_DEFAULT,
    progress_cb=None,
) -> np.ndarray:
    """Generate the attenuation sinogram (mu * path_length) for all views, chunked for
    memory/responsiveness. progress_cb(done, total) is called after each chunk if given.
    """
    projector = GpuProjector(vertices, faces, device)
    n_views = len(views)
    sinogram = np.empty((n_views, rows, cols), dtype=np.float32)

    for start in range(0, n_views, chunk_size):
        chunk = views[start : start + chunk_size]
        path_lengths = projector.trace_views(chunk, rows, cols, pixel_pitch, max_hits, eps)
        sinogram[start : start + len(chunk)] = path_lengths * mu
        if progress_cb is not None:
            progress_cb(min(start + len(chunk), n_views), n_views)

    return sinogram
