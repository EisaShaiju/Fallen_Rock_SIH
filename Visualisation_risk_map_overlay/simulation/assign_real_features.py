"""
Assign real feature rows (after first 5000) from advanced_rockfall_dataset.csv
to extracted mine coordinates. Produces a merged CSV with coordinates + features.

Example:
  python3 assign_real_features.py \
    --coords ../outputs/mine_coordinates.csv \
    --data ../data/advanced_rockfall_dataset.csv \
    --skip-first 5000 \
    --out ../outputs/mine_sensor_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description='Attach real features to mine coordinates')
    p.add_argument('--coords', type=str, required=True, help='CSV of mine coordinates (x,y,z)')
    p.add_argument('--data', type=str, default='../data/advanced_rockfall_dataset.csv', help='Path to dataset CSV')
    p.add_argument('--skip-first', type=int, default=5000, help='Skip first N rows (used for training/testing)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=str, default='../outputs/mine_sensor_data.csv')
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    coords = pd.read_csv(args.coords)
    if not set(['x', 'y', 'z']).issubset(coords.columns):
        raise ValueError('coords CSV must contain columns: x, y, z')

    df = pd.read_csv(args.data)
    if args.skip_first > 0:
        df = df.iloc[args.skip_first:, :]

    # Identify feature columns by excluding targets and simulation outputs
    target_cols = ['risk_prob_7d', 'risk_severity']
    drop_cols = target_cols + ['kinetic_energy', 'impact_position', 'runout_distance']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    if len(df) < len(coords):
        raise ValueError('Not enough rows in data after skipping to match number of coordinates')

    # Randomly sample rows equal to number of coordinates
    sampled = df.sample(n=len(coords), random_state=args.seed).reset_index(drop=True)

    merged = pd.concat([coords.reset_index(drop=True), sampled[feature_cols].reset_index(drop=True)], axis=1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f'Saved merged coordinates+features to {args.out} with {len(merged)} rows')


if __name__ == '__main__':
    main()





