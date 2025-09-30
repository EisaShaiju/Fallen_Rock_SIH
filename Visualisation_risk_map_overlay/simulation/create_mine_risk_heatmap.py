"""
Create mine risk predictions and an interpolated 3D risk field from
coordinates + real features, using the trained model bundle.

Requires: scipy, optionally plotly or pyvista for visualization.

Example (full pipeline after prior steps):
  python3 create_mine_risk_heatmap.py \
    --merged ../outputs/mine_sensor_data.csv \
    --model ../outputs/advanced_risk_models.joblib \
    --target risk_prob_7d \
    --grid-nx 50 --grid-ny 50 --grid-nz 25 \
    --out-points ../outputs/mine_risk_points.csv \
    --out-grid ../outputs/mine_risk_grid.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib
from scipy.interpolate import griddata


def parse_args():
    p = argparse.ArgumentParser(description='Predict and interpolate mine risk field')
    p.add_argument('--merged', type=str, required=True, help='CSV with x,y,z and feature columns')
    p.add_argument('--model', type=str, default='../outputs/advanced_risk_models.joblib')
    p.add_argument('--target', type=str, default='risk_prob_7d')
    p.add_argument('--grid-nx', type=int, default=50)
    p.add_argument('--grid-ny', type=int, default=50)
    p.add_argument('--grid-nz', type=int, default=25)
    p.add_argument('--out-points', type=str, default='../outputs/mine_risk_points.csv')
    p.add_argument('--out-grid', type=str, default='../outputs/mine_risk_grid.npz')
    return p.parse_args()


def main():
    args = parse_args()

    merged = pd.read_csv(args.merged)
    if not set(['x', 'y', 'z']).issubset(merged.columns):
        raise ValueError('merged CSV must include x, y, z columns')

    bundle = joblib.load(args.model)
    feature_names: List[str] = list(bundle['feature_names'])
    model = bundle['artifacts'][args.target]['model']

    X = merged[feature_names].copy()
    preds = model.predict(X)
    preds = np.asarray(preds).reshape(-1)

    points = merged[['x', 'y', 'z']].copy()
    points[f'pred_{args.target}'] = preds

    # Save point predictions
    Path(args.out_points).parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.out_points, index=False)

    # Build grid over the coordinate bounds
    xi = np.linspace(points.x.min(), points.x.max(), args.grid_nx)
    yi = np.linspace(points.y.min(), points.y.max(), args.grid_ny)
    zi = np.linspace(points.z.min(), points.z.max(), args.grid_nz)
    Xi, Yi, Zi = np.meshgrid(xi, yi, zi, indexing='xy')

    risk_interp = griddata(
        (points.x.values, points.y.values, points.z.values),
        points[f'pred_{args.target}'].values,
        (Xi, Yi, Zi),
        method='linear'
    )
    # Fill NaNs (outside convex hull) with nearest neighbor
    mask_nan = np.isnan(risk_interp)
    if np.any(mask_nan):
        risk_interp_nn = griddata(
            (points.x.values, points.y.values, points.z.values),
            points[f'pred_{args.target}'].values,
            (Xi, Yi, Zi),
            method='nearest'
        )
        risk_interp[mask_nan] = risk_interp_nn[mask_nan]

    # Save grid npz
    np.savez_compressed(
        args.out_grid,
        Xi=Xi.astype(np.float32),
        Yi=Yi.astype(np.float32),
        Zi=Zi.astype(np.float32),
        risk=risk_interp.astype(np.float32),
        target=args.target,
    )
    print(f'Saved point predictions to {args.out_points}')
    print(f'Saved risk grid to {args.out_grid}')


if __name__ == '__main__':
    main()




