"""
Render interactive Plotly visualizations for mine risk:
- Point cloud colored by predicted risk
- Volume/isosurface from interpolated grid

Outputs HTML files alongside data unless overridden.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def parse_args():
    p = argparse.ArgumentParser(description='Visualize mine risk points and grid')
    p.add_argument('--points', type=str, required=True, help='CSV with x,y,z and pred_* column')
    p.add_argument('--grid', type=str, required=True, help='NPZ with Xi,Yi,Zi,risk')
    p.add_argument('--out-points-html', type=str, default='')
    p.add_argument('--out-grid-html', type=str, default='')
    p.add_argument('--target', type=str, default='risk_severity')
    return p.parse_args()


def save_fig(fig: go.Figure, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    print(f'Saved {out_path}')


def main():
    args = parse_args()

    points = pd.read_csv(args.points)
    pred_col = f'pred_{args.target}'
    if pred_col not in points.columns:
        # fallback to first pred_* column
        pred_cols = [c for c in points.columns if c.startswith('pred_')]
        if not pred_cols:
            raise ValueError('No prediction column found (expected pred_*)')
        pred_col = pred_cols[0]

    # Points figure
    fig_points = go.Figure(
        data=[go.Scatter3d(
            x=points['x'], y=points['y'], z=points['z'],
            mode='markers',
            marker=dict(
                size=4,
                color=points[pred_col],
                colorscale='Jet',
                cmin=0.0, cmax=1.0,
                colorbar=dict(title=pred_col),
                opacity=0.9,
            ),
            text=[f"{pred_col}={v:.3f}" for v in points[pred_col].values],
            hoverinfo='text'
        )]
    )
    fig_points.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

    # Grid/volume figure
    grid = np.load(args.grid)
    Xi, Yi, Zi, risk = grid['Xi'], grid['Yi'], grid['Zi'], grid['risk']

    fig_grid = go.Figure(
        data=[go.Volume(
            x=Xi.flatten(), y=Yi.flatten(), z=Zi.flatten(),
            value=risk.flatten(),
            isomin=float(np.nanmin(risk)), isomax=float(np.nanmax(risk)),
            opacity=0.08,
            surface_count=15,
            colorscale='Jet',
            caps=dict(x_show=False, y_show=False, z_show=False),
        )]
    )
    fig_grid.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

    # Outputs
    out_points_html = args.out_points_html or str(Path(args.points).with_suffix('').as_posix() + f'_{args.target}.html')
    out_grid_html = args.out_grid_html or str(Path(args.grid).with_suffix('').as_posix() + f'_{args.target}.html')

    save_fig(fig_points, out_points_html)
    save_fig(fig_grid, out_grid_html)


if __name__ == '__main__':
    main()





