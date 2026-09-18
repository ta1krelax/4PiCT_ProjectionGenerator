"""Shape / morphology descriptors for the loaded STL.

These characterize sample geometric complexity, to be correlated later against how
many 4pi projections (N) are needed for a good reconstruction. Definitions follow
established literature rather than ad-hoc formulas; where no rigorous 3D definition
exists for an arbitrary solid, a clearly-labelled proxy is used instead (see notes on
each function). All linear units follow the STL file's own units (assumed mm).

References:
- L1/L2/L3 caliper axes: Krumbein (1941) "Measurement and geological significance of
  shape and roundness of sedimentary particles"; Sneed & Folk (1958) elongation/flatness.
- Sphericity: Wadell (1932), psi = pi^(1/3) * (6V)^(2/3) / A.
- Local thickness: Hildebrand & Ruegsegger (1997) "A new method for the model-independent
  assessment of thickness in three-dimensional images" -- diameter of the largest sphere
  fully inside the solid that contains the evaluated point (maximal-ball method).
- Solidity / convexity: ISO 9276-6 style ratios against the convex hull (volume and area).
- Fractal dimension: box-counting (Minkowski-Bouligand dimension) on the voxelized solid.
- Surface roughness Sa/Sq: areal-roughness analogues (ISO 25178 naming). There is no flat
  reference plane for an arbitrary solid, so this reports vertex displacement from a
  Taubin-smoothed copy of the same mesh as a proxy.
- Tortuosity: the standard porous-media definition (flow path length / straight-line
  distance) has no meaning without an actual flow channel. This reports a clearly-labelled
  proxy: surface geodesic distance between the two most distant points, divided by their
  straight-line distance.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

METRIC_ORDER = [
    {"rank": 1, "key": "projection_number", "label": "Projection number N", "default": True},
    {"rank": 2, "key": "L1_over_T5", "label": "L1 / T5", "default": True},
    {"rank": 3, "key": "equivalent_diameter", "label": "Equivalent diameter D_eq / Volume", "default": True},
    {"rank": 4, "key": "major_dimension_L1", "label": "Major dimension L1", "default": True},
    {"rank": 5, "key": "aspect_ratio_L2_L1", "label": "Aspect ratio L2/L1 (elongation)", "default": True},
    {"rank": 6, "key": "local_thickness_T5", "label": "Local thickness T5", "default": True},
    {"rank": 7, "key": "aspect_ratio_L3_L2", "label": "Aspect ratio L3/L2 (flatness)", "default": True},
    {"rank": 8, "key": "sphericity", "label": "Sphericity Ψ (Wadell)", "default": False},
    {"rank": 9, "key": "sa_over_v", "label": "Surface area / volume (SA/V)", "default": False},
    {"rank": 10, "key": "porosity", "label": "Porosity", "default": False},
    {"rank": 11, "key": "convexity_solidity", "label": "Convexity / Solidity", "default": False},
    {"rank": 12, "key": "euler_characteristic", "label": "Euler characteristic / connectivity", "default": False},
    {"rank": 13, "key": "surface_roughness", "label": "Surface roughness (proxy)", "default": False},
    {"rank": 14, "key": "fractal_dimension", "label": "Fractal dimension (box-counting)", "default": False},
    {"rank": 15, "key": "tortuosity", "label": "Tortuosity (surface-geodesic proxy)", "default": False},
    {"rank": 16, "key": "principal_axis_orientation", "label": "Principal-axis orientation", "default": False},
]

_NEEDS_CALIPERS = {2, 4, 5, 7, 15}
_NEEDS_THICKNESS = {2, 6}
_NEEDS_VOXEL = _NEEDS_THICKNESS | {14}


# --------------------------------------------------------------- base geometry
def caliper_dimensions(mesh: trimesh.Trimesh) -> dict:
    """L1/L2/L3 mutually-perpendicular caliper (Feret) axes, Krumbein/Sneed & Folk style.

    L1 = longest straight-line dimension (max distance between any two points on the
    convex hull). L2 = longest dimension perpendicular to L1. L3 = longest dimension
    perpendicular to both L1 and L2.
    """
    hull = mesh.convex_hull
    pts = hull.vertices

    d = squareform(pdist(pts))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    p1, p2 = pts[i], pts[j]
    L1 = float(d[i, j])
    u1 = (p2 - p1) / L1 if L1 > 1e-12 else np.array([1.0, 0.0, 0.0])

    rel = pts - p1
    proj_along = rel @ u1
    in_plane = rel - np.outer(proj_along, u1)

    seed = np.array([1.0, 0.0, 0.0]) if abs(u1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u1, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u1, e1)

    coords2d = np.stack([in_plane @ e1, in_plane @ e2], axis=1)
    d2 = squareform(pdist(coords2d))
    i2, j2 = np.unravel_index(np.argmax(d2), d2.shape)
    L2 = float(d2[i2, j2])
    if L2 > 1e-9:
        dir2d = (coords2d[j2] - coords2d[i2]) / L2
    else:
        dir2d = np.array([1.0, 0.0])
    u2 = dir2d[0] * e1 + dir2d[1] * e2
    u3 = np.cross(u1, u2)

    proj3 = pts @ u3
    L3 = float(proj3.max() - proj3.min())

    return {
        "L1": L1, "L2": L2, "L3": L3,
        "u1": u1.tolist(), "u2": u2.tolist(), "u3": u3.tolist(),
        "p1": p1.tolist(), "p2": p2.tolist(),
    }


def default_voxel_pitch(mesh: trimesh.Trimesh, target_voxels: int = 64) -> float:
    extent = float(mesh.extents.max())
    return extent / max(target_voxels, 4)


def voxelize_solid(mesh: trimesh.Trimesh, voxel_pitch: float, pad: int = 2):
    """Voxelize the solid interior, padded with an empty border.

    Without padding, a shape whose tight bounding box it fills exactly (e.g. a box)
    produces an all-True array with no background voxel anywhere in it, which breaks
    distance_transform_edt (there is nothing to measure distance to). The padding
    border guarantees a real background reference on every side.
    """
    vg = mesh.voxelized(pitch=voxel_pitch).fill()
    matrix = vg.matrix.astype(bool)
    padded = np.pad(matrix, pad_width=pad, mode="constant", constant_values=False)
    return padded, voxel_pitch


def local_thickness_field(solid_mask: np.ndarray, voxel_pitch: float) -> np.ndarray:
    """Hildebrand & Ruegsegger maximal-ball local thickness, in physical units.

    For every voxel, the diameter of the largest sphere fully inside the solid that
    contains that voxel. Implemented as: find distance-transform ridge points (candidate
    maximal-ball centers), then for each such ball, stamp 2*radius onto every voxel it
    covers, keeping the max value seen.
    """
    D = distance_transform_edt(solid_mask) * voxel_pitch
    local_max = maximum_filter(D, size=3, mode="constant", cval=0.0)
    ridge_coords = np.argwhere((D >= local_max) & (D > 0))

    LT = 2.0 * D  # lower bound: ball centered at the point itself
    shape = D.shape
    for zc, yc, xc in ridge_coords:
        r = D[zc, yc, xc]
        rvox = int(np.ceil(r / voxel_pitch))
        if rvox <= 0:
            continue
        z0, z1 = max(0, zc - rvox), min(shape[0], zc + rvox + 1)
        y0, y1 = max(0, yc - rvox), min(shape[1], yc + rvox + 1)
        x0, x1 = max(0, xc - rvox), min(shape[2], xc + rvox + 1)
        zz, yy, xx = np.meshgrid(
            (np.arange(z0, z1) - zc) * voxel_pitch,
            (np.arange(y0, y1) - yc) * voxel_pitch,
            (np.arange(x0, x1) - xc) * voxel_pitch,
            indexing="ij",
        )
        within = (zz * zz + yy * yy + xx * xx) <= (r * r)
        sub = LT[z0:z1, y0:y1, x0:x1]
        np.putmask(sub, within & (2.0 * r > sub), 2.0 * r)
        LT[z0:z1, y0:y1, x0:x1] = sub
    return LT


def local_thickness_percentile(solid_mask: np.ndarray, voxel_pitch: float, percentile: float = 5.0) -> dict:
    LT = local_thickness_field(solid_mask, voxel_pitch)
    values = LT[solid_mask]
    return {
        "T_percentile_mm": float(np.percentile(values, percentile)),
        "percentile": percentile,
        "mean_thickness_mm": float(values.mean()),
        "max_thickness_mm": float(values.max()),
        "voxel_pitch_mm": voxel_pitch,
        "n_voxels": int(solid_mask.sum()),
    }


# ------------------------------------------------------------------ metrics
def sphericity_wadell(mesh: trimesh.Trimesh) -> float:
    V, A = mesh.volume, mesh.area
    return float(np.pi ** (1.0 / 3.0) * (6.0 * V) ** (2.0 / 3.0) / A)


def sa_over_v(mesh: trimesh.Trimesh) -> float:
    return float(mesh.area / mesh.volume)


def porosity(mesh: trimesh.Trimesh) -> dict:
    parts = mesh.split(only_watertight=True)
    if len(parts) <= 1:
        return {"porosity": 0.0, "void_volume_mm3": 0.0, "outer_volume_mm3": float(abs(mesh.volume)), "n_shells": len(parts)}
    volumes = [float(abs(p.volume)) for p in parts]
    outer_idx = int(np.argmax(volumes))
    outer_volume = volumes[outer_idx]
    void_volume = float(sum(v for i, v in enumerate(volumes) if i != outer_idx))
    return {
        "porosity": void_volume / outer_volume if outer_volume > 0 else 0.0,
        "void_volume_mm3": void_volume,
        "outer_volume_mm3": outer_volume,
        "n_shells": len(parts),
        "note": "assumes the largest-volume shell is the outer boundary and all other shells are internal voids",
    }


def convexity_solidity(mesh: trimesh.Trimesh) -> dict:
    hull = mesh.convex_hull
    return {
        "solidity_volume": float(mesh.volume / hull.volume) if hull.volume > 0 else None,
        "convexity_area": float(hull.area / mesh.area) if mesh.area > 0 else None,
    }


def surface_roughness_proxy(mesh: trimesh.Trimesh, iterations: int = 10) -> dict:
    smoothed = mesh.copy()
    trimesh.smoothing.filter_taubin(smoothed, lamb=0.5, nu=-0.53, iterations=iterations)
    d = np.linalg.norm(mesh.vertices - smoothed.vertices, axis=1)
    return {
        "Sa_mm": float(np.mean(d)),
        "Sq_mm": float(np.sqrt(np.mean(d ** 2))),
        "note": "proxy: vertex displacement from a Taubin-smoothed copy of the same mesh, "
                "not an ISO-25178 profilometer measurement (no flat reference plane exists for an arbitrary solid)",
    }


def fractal_dimension_boxcount(solid_mask: np.ndarray) -> dict:
    dims = solid_mask.shape
    size = int(2 ** np.ceil(np.log2(max(dims))))
    padded = np.zeros((size, size, size), dtype=bool)
    padded[: dims[0], : dims[1], : dims[2]] = solid_mask

    sizes, counts = [], []
    factor = size
    while factor >= 2:
        n = size // factor
        reshaped = padded.reshape(n, factor, n, factor, n, factor)
        occ = reshaped.any(axis=(1, 3, 5))
        counts.append(int(occ.sum()))
        sizes.append(factor)
        factor //= 2

    sizes = np.array(sizes, dtype=float)
    counts = np.array(counts, dtype=float)
    valid = counts > 0
    log_inv_size = np.log(1.0 / sizes[valid])
    log_count = np.log(counts[valid])
    slope, intercept = np.polyfit(log_inv_size, log_count, 1)
    pred = slope * log_inv_size + intercept
    ss_res = float(np.sum((log_count - pred) ** 2))
    ss_tot = float(np.sum((log_count - log_count.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {"fractal_dimension": float(slope), "r_squared": float(r2), "n_scales": int(valid.sum())}


def principal_axis_orientation(mesh: trimesh.Trimesh) -> dict:
    axes = np.asarray(mesh.principal_inertia_vectors)
    global_axes = np.eye(3)
    angles = []
    for row in axes:
        row = row / np.linalg.norm(row)
        angs = np.degrees(np.arccos(np.clip(np.abs(row @ global_axes.T), -1.0, 1.0)))
        angles.append(angs.tolist())
    return {"axes": axes.tolist(), "angle_to_XYZ_deg": angles}


def tortuosity_surface_geodesic(mesh: trimesh.Trimesh, p1: np.ndarray, p2: np.ndarray, straight_length: float) -> dict:
    import networkx as nx

    tree = cKDTree(mesh.vertices)
    _, i1 = tree.query(p1)
    _, i2 = tree.query(p2)

    graph = nx.Graph()
    for (a, b), length in zip(mesh.edges_unique, mesh.edges_unique_length):
        graph.add_edge(int(a), int(b), weight=float(length))

    try:
        geodesic = nx.shortest_path_length(graph, source=int(i1), target=int(i2), weight="weight")
    except nx.NetworkXNoPath:
        return {"tortuosity_proxy": None, "note": "mesh surface graph is disconnected between the two extreme points"}

    return {
        "tortuosity_proxy": float(geodesic / straight_length) if straight_length > 0 else None,
        "geodesic_surface_length_mm": float(geodesic),
        "straight_line_length_mm": float(straight_length),
        "note": "proxy: surface-geodesic distance / straight-line distance between the two most distant "
                "points (L1 endpoints); NOT the standard porous-media flow-path tortuosity",
    }


# --------------------------------------------------------------- orchestrator
def compute_shape_metrics(
    mesh: trimesh.Trimesh,
    selected_ranks: set,
    n_views: int | None = None,
    voxel_target: int = 64,
    percentile: float = 5.0,
    progress_cb=None,
) -> dict:
    results: dict = {}
    errors: dict = {}
    total = len(selected_ranks)
    done = 0

    def tick(label: str):
        nonlocal done
        done += 1
        if progress_cb is not None:
            progress_cb(done, total, label)

    def guarded(rank: int, label: str, fn):
        try:
            results[label] = fn()
        except Exception as exc:  # noqa: BLE001 - report per-metric, keep going
            errors[label] = str(exc)
        tick(label)

    calipers = None
    if selected_ranks & _NEEDS_CALIPERS:
        try:
            calipers = caliper_dimensions(mesh)
        except Exception as exc:
            errors["caliper_dimensions"] = str(exc)

    solid_mask, voxel_pitch = None, None
    if selected_ranks & _NEEDS_VOXEL:
        try:
            voxel_pitch = default_voxel_pitch(mesh, voxel_target)
            solid_mask, voxel_pitch = voxelize_solid(mesh, voxel_pitch)
        except Exception as exc:
            errors["voxelization"] = str(exc)

    thickness = None
    if selected_ranks & _NEEDS_THICKNESS and solid_mask is not None:
        try:
            thickness = local_thickness_percentile(solid_mask, voxel_pitch, percentile)
        except Exception as exc:
            errors["local_thickness"] = str(exc)

    if 1 in selected_ranks:
        guarded(1, "projection_number_N", lambda: n_views)

    if 4 in selected_ranks:
        guarded(4, "major_dimension_L1_mm", lambda: calipers["L1"])

    if 5 in selected_ranks:
        guarded(5, "aspect_ratio_L2_over_L1", lambda: calipers["L2"] / calipers["L1"] if calipers["L1"] else None)

    if 7 in selected_ranks:
        guarded(7, "aspect_ratio_L3_over_L2", lambda: calipers["L3"] / calipers["L2"] if calipers["L2"] else None)

    if 6 in selected_ranks:
        guarded(6, "local_thickness_T5", lambda: thickness)

    if 2 in selected_ranks:
        def _l1_over_t5():
            t5 = thickness["T_percentile_mm"]
            return calipers["L1"] / t5 if t5 else None
        guarded(2, "L1_over_T5", _l1_over_t5)

    if 3 in selected_ranks:
        def _deq():
            V = float(mesh.volume)
            return {"equivalent_diameter_mm": (6.0 * V / np.pi) ** (1.0 / 3.0), "volume_mm3": V}
        guarded(3, "equivalent_diameter_and_volume", _deq)

    if 8 in selected_ranks:
        guarded(8, "sphericity_wadell", lambda: sphericity_wadell(mesh))

    if 9 in selected_ranks:
        guarded(9, "sa_over_v_per_mm", lambda: sa_over_v(mesh))

    if 10 in selected_ranks:
        guarded(10, "porosity", lambda: porosity(mesh))

    if 11 in selected_ranks:
        guarded(11, "convexity_solidity", lambda: convexity_solidity(mesh))

    if 12 in selected_ranks:
        guarded(12, "euler_characteristic", lambda: int(mesh.euler_number))

    if 13 in selected_ranks:
        guarded(13, "surface_roughness", lambda: surface_roughness_proxy(mesh))

    if 14 in selected_ranks:
        guarded(14, "fractal_dimension", lambda: fractal_dimension_boxcount(solid_mask))

    if 15 in selected_ranks:
        def _tort():
            c = calipers or caliper_dimensions(mesh)
            return tortuosity_surface_geodesic(mesh, np.array(c["p1"]), np.array(c["p2"]), c["L1"])
        guarded(15, "tortuosity_proxy", _tort)

    if 16 in selected_ranks:
        guarded(16, "principal_axis_orientation", lambda: principal_axis_orientation(mesh))

    if errors:
        results["_errors"] = errors
    return results
