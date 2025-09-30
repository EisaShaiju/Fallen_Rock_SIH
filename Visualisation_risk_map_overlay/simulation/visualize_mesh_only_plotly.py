"""
Render the original 3D mine model (GLB) as an interactive Plotly Mesh3d HTML.

Example:
  python3 visualize_mesh_only_plotly.py \
    --glb /Users/pookie/Desktop/rs2/dashboard/mine-dashboard/the-bingham-canyon-mine-utah/source/800004c9-0001-f500-b63f-84710c7967bb.glb \
    --out ../outputs/mine_mesh_only.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import plotly.graph_objects as go


def parse_args():
    p = argparse.ArgumentParser(description='View original 3D mine mesh as Plotly HTML')
    p.add_argument('--glb', type=str, required=True, help='Path to GLB/GLTF file')
    p.add_argument('--out', type=str, default='../outputs/mine_mesh_only.html')
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import trimesh  # type: ignore
    except Exception:
        raise SystemExit('This script requires trimesh. Install with: pip install trimesh')

    mesh = trimesh.load(args.glb, force='mesh')

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)

    fig = go.Figure(
        data=[go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=F[:, 0], j=F[:, 1], k=F[:, 2],
            color='lightsteelblue',
            opacity=1.0,
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1),
        )]
    )
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        scene_aspectmode='data'
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out)
    print(f'Saved original mesh HTML to {args.out}')


if __name__ == '__main__':
    main()


