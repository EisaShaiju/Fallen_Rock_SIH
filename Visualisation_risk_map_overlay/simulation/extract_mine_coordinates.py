"""
Extract high-slope surface points from a GLB mine model.

Requires: trimesh

Example:
  python3 extract_mine_coordinates.py \
    --glb /Users/pookie/Desktop/rs2/dashboard/mine-dashboard/the-bingham-canyon-mine-utah/source/800004c9-0001-f500-b63f-84710c7967bb.glb \
    --n 100 \
    --slope-min 30 \
    --out ../outputs/mine_coordinates.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def compute_vertex_normals(mesh) -> np.ndarray:
    # Prefer trimesh-provided vertex normals if available
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        vn = np.asarray(mesh.vertex_normals)
        if vn.shape[0] == len(mesh.vertices):
            return vn
    # Fallback: compute via faces
    vn = np.zeros_like(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices)
    v0 = verts[faces[:, 1]] - verts[faces[:, 0]]
    v1 = verts[faces[:, 2]] - verts[faces[:, 0]]
    fn = np.cross(v0, v1)
    # accumulate to vertices
    for i, f in enumerate(faces):
        for vi in f:
            vn[vi] += fn[i]
    # normalize
    norms = np.linalg.norm(vn, axis=1)
    norms[norms == 0] = 1.0
    vn = vn / norms[:, None]
    return vn


def slope_from_normal(vertex_normals: np.ndarray) -> np.ndarray:
    # Slope angle relative to horizontal plane: 0°=flat, 90°=vertical
    # normal z-component magnitude near 1 -> slope ~ 0°; near 0 -> slope ~ 90°
    nz = np.clip(np.abs(vertex_normals[:, 2]), 1e-8, 1.0)
    slope_rad = np.arctan2(np.sqrt(1.0 - nz**2), nz)
    slope_deg = np.degrees(slope_rad)
    return slope_deg


def sample_high_slope_points(vertices: np.ndarray, slope_deg: np.ndarray, n_points: int, slope_min: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = slope_deg >= slope_min
    candidates = vertices[mask]
    if len(candidates) == 0:
        # fallback to all vertices
        candidates = vertices
    if len(candidates) <= n_points:
        return candidates
    idx = rng.choice(len(candidates), size=n_points, replace=False)
    return candidates[idx]


def parse_args():
    p = argparse.ArgumentParser(description='Extract high-slope surface points from GLB')
    p.add_argument('--glb', type=str, required=True, help='Path to GLB/GLTF file')
    p.add_argument('--n', type=int, default=100, help='Number of points to sample')
    p.add_argument('--slope-min', type=float, default=30.0, help='Minimum slope angle (degrees) to consider')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=str, default='../outputs/mine_coordinates.csv')
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import trimesh  # type: ignore
    except Exception as e:
        raise SystemExit('This script requires trimesh. Install with: pip install trimesh')

    scene_or_mesh = trimesh.load(args.glb, force='scene')
    if isinstance(scene_or_mesh, trimesh.Scene):
        # Merge geometry into a single mesh
        mesh = trimesh.util.concatenate(tuple(geom for geom in scene_or_mesh.dump()))
    else:
        mesh = scene_or_mesh

    vertices = np.asarray(mesh.vertices)
    vnormals = compute_vertex_normals(mesh)
    slope_deg = slope_from_normal(vnormals)

    sampled = sample_high_slope_points(vertices, slope_deg, args.n, args.slope_min, args.seed)

    df = pd.DataFrame({'x': sampled[:, 0], 'y': sampled[:, 1], 'z': sampled[:, 2]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'Saved {len(df)} surface points to {args.out}')


if __name__ == '__main__':
    main()




