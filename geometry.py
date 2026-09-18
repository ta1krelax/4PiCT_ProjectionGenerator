"""4pi view direction sampling and cone-beam geometry construction."""

from __future__ import annotations

import numpy as np


def parse_n_sequence(segments: list[tuple[int, int, int]]) -> list[int]:
    """Merge piecewise (start, end, step) ranges into one sorted, de-duplicated list of N.

    Each segment's own start/end are always included even if `step` does not divide the
    span evenly, so an explicit boundary the user typed is never silently dropped.
    """
    values: set[int] = set()
    for start, end, step in segments:
        start, end, step = int(start), int(end), int(step)
        if step <= 0:
            raise ValueError(f"步长必须为正整数, 收到 {step}")
        lo, hi = min(start, end), max(start, end)
        if lo <= 0:
            raise ValueError(f"N 必须为正整数, 收到 {lo}")
        values.update(range(lo, hi + 1, step))
        values.add(lo)
        values.add(hi)
    return sorted(values)


def fibonacci_sphere_directions(n: int) -> np.ndarray:
    """Near-uniform directions over the full sphere (4pi). Returns (n,3) unit vectors."""
    i = np.arange(n)
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    theta = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    phi = (2.0 * np.pi * i / golden) % (2.0 * np.pi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)


def random_sphere_directions(n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def lonlat_sphere_directions(n_lon: int, n_lat: int) -> np.ndarray:
    """Regular longitude/latitude grid over the sphere, poles excluded to avoid duplicate points."""
    lons = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    lats = np.linspace(-np.pi / 2 + np.pi / (2 * n_lat), np.pi / 2 - np.pi / (2 * n_lat), n_lat)
    lo, la = np.meshgrid(lons, lats)
    x = np.cos(la) * np.cos(lo)
    y = np.cos(la) * np.sin(lo)
    z = np.sin(la)
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def circular_orbit_directions(n: int, axis: str = "z") -> np.ndarray:
    """Single-axis 360 degree orbit - conventional CT baseline, not full 4pi."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    c, s = np.cos(angles), np.sin(angles)
    zero = np.zeros(n)
    if axis == "z":
        v = np.stack([c, s, zero], axis=1)
    elif axis == "y":
        v = np.stack([c, zero, s], axis=1)
    else:
        v = np.stack([zero, c, s], axis=1)
    return v


SAMPLING_SCHEMES = {
    "circular_z": lambda n: circular_orbit_directions(n, axis="z"),
    "circular_y": lambda n: circular_orbit_directions(n, axis="y"),
    "circular_x": lambda n: circular_orbit_directions(n, axis="x"),
    "fibonacci_4pi": fibonacci_sphere_directions,
    "random_4pi": random_sphere_directions,
}


def build_view_frame(direction: np.ndarray, world_up: np.ndarray = np.array([0.0, 0.0, 1.0])):
    """Given unit direction (rotation-center -> source), build a consistent detector (u,v) frame.

    Falls back to an alternate up-vector near the poles to avoid a degenerate cross product.
    """
    d = direction / np.linalg.norm(direction)
    up = world_up
    if abs(np.dot(d, up)) > 0.98:
        up = np.array([1.0, 0.0, 0.0])
    u_hat = np.cross(up, d)
    u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(d, u_hat)
    v_hat /= np.linalg.norm(v_hat)
    return u_hat, v_hat


def build_view_geometry(direction: np.ndarray, rotation_center: np.ndarray, sod: float, sdd: float):
    """Build one cone-beam view in world space.

    source is at distance `sod` from rotation_center along `direction`; the detector center is
    on the opposite side of the rotation center, so that source-to-detector distance is `sdd`
    (magnification M = sdd / sod).
    """
    d = direction / np.linalg.norm(direction)
    src = rotation_center + d * sod
    det_center = rotation_center - d * (sdd - sod)
    u_hat, v_hat = build_view_frame(d)
    return src, det_center, u_hat, v_hat


def build_all_views(directions: np.ndarray, rotation_center: np.ndarray, sod: float, sdd: float):
    """Returns list of (src, det_center, u_hat, v_hat) world-space tuples, one per direction."""
    return [build_view_geometry(d, rotation_center, sod, sdd) for d in directions]


def to_astra_cone_vec(views, rotation_center: np.ndarray, pixel_pitch: float) -> np.ndarray:
    """Convert world-space views to ASTRA 'cone_vec' geometry rows, centered on rotation_center.

    Row layout: [srcX,srcY,srcZ, dX,dY,dZ, uX,uY,uZ, vX,vY,vZ]
    u,v are scaled by one pixel's physical size, per ASTRA convention.
    """
    rows = []
    for src, det_center, u_hat, v_hat in views:
        s = src - rotation_center
        d = det_center - rotation_center
        u = u_hat * pixel_pitch
        v = v_hat * pixel_pitch
        rows.append(np.concatenate([s, d, u, v]))
    return np.array(rows, dtype=np.float64)
