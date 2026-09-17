"""Dataset persistence: projections + ASTRA-ready geometry + metadata."""

from __future__ import annotations

import json
import os
import time

import numpy as np


def save_dataset(out_dir: str, sinogram: np.ndarray, astra_vectors: np.ndarray, meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "projections.npy"), sinogram.astype(np.float32))
    np.save(os.path.join(out_dir, "astra_vectors.npy"), astra_vectors.astype(np.float64))

    meta_full = dict(meta)
    meta_full["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta_full["projections_file"] = "projections.npy"
    meta_full["projections_shape"] = list(sinogram.shape)
    meta_full["projections_axes"] = ["view", "detector_row", "detector_col"]
    meta_full["projections_value"] = "mu * path_length (line integral, ready for FBP/SIRT/CGLS)"
    meta_full["astra_vectors_file"] = "astra_vectors.npy"
    meta_full["astra_vectors_shape"] = list(astra_vectors.shape)
    meta_full["astra_vectors_layout"] = "[srcX,srcY,srcZ, dX,dY,dZ, uX,uY,uZ, vX,vY,vZ] per row, centered on rotation_center"
    meta_full["astra_usage"] = (
        "import astra, numpy as np\n"
        "vectors = np.load('astra_vectors.npy')\n"
        "proj = np.load('projections.npy')  # (n_views, rows, cols)\n"
        "rows, cols = proj.shape[1], proj.shape[2]\n"
        "proj_geom = astra.create_proj_geom('cone_vec', rows, cols, vectors)\n"
        "sino = np.transpose(proj, (1, 0, 2))  # astra wants (row, view, col)\n"
        "proj_id = astra.data3d.create('-sino', proj_geom, sino)"
    )

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_full, f, indent=2, ensure_ascii=False)
