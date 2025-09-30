"""
Predict risk using the saved advanced models bundle (CSV batch only).

Usage examples:

Batch prediction from CSV (columns must include the training feature names):
  python3 predict.py \
    --target risk_prob_7d \
    --csv /path/to/features.csv \
    --out /path/to/preds.csv

Predict on the remainder of the training dataset (skip first 5000 rows used for training):
  python3 predict.py \
    --target risk_prob_7d \
    --csv ../data/advanced_rockfall_dataset.csv \
    --skip-first 5000 \
    --out ../outputs/preds_remaining.csv

Optionally pass through ID/context columns to the output:
  --id-cols id,timestamp,x,y,z

Defaults assume running from this directory so the relative model path resolves.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = "../outputs/advanced_risk_models.joblib"


def load_bundle(model_path: str):
    bundle = joblib.load(model_path)
    required_keys = {"artifacts", "feature_names", "target_names"}
    missing = required_keys.difference(bundle.keys())
    if missing:
        raise ValueError(f"Model bundle missing keys: {sorted(missing)}")
    return bundle


def read_features_csv(
    csv_path: str,
    feature_names: List[str],
    id_cols: Optional[List[str]] = None,
    skip_first: int = 0,
    limit: int = 0,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Row window selection if requested
    if skip_first > 0:
        df = df.iloc[skip_first:, :]
    if limit and limit > 0:
        df = df.iloc[:limit, :]

    # Validate required features present
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required feature column(s): {', '.join(missing)}"
        )
    # Build output with optional passthrough id columns
    out_cols: List[str] = []
    if id_cols:
        out_cols.extend([c for c in id_cols if c in df.columns])
    out_cols.extend(feature_names)
    df = df[out_cols].copy()
    return df


def predict(
    bundle: Dict,
    target: str,
    X: pd.DataFrame,
) -> np.ndarray:
    artifacts: Dict[str, Dict] = bundle["artifacts"]
    if target not in artifacts:
        raise ValueError(
            f"Target '{target}' not found in model bundle. Available: {', '.join(artifacts.keys())}"
        )
    model = artifacts[target]["model"]
    preds = model.predict(X)
    # Ensure 1D numpy array
    preds = np.asarray(preds).reshape(-1)
    return preds


def save_predictions(
    X_full: pd.DataFrame,
    preds: np.ndarray,
    target: str,
    out_path: str,
):
    out_df = X_full.copy()
    out_df[f"pred_{target}"] = preds
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Predict risk using saved advanced models")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to joblib bundle")
    p.add_argument(
        "--target",
        type=str,
        default="risk_prob_7d",
        help="Which target to predict (e.g., risk_prob_7d or risk_severity)",
    )
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV with feature columns for batch prediction",
    )
    p.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Skip the first N rows from the CSV before predicting",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Predict on only the first N rows after skipping (0=all)",
    )
    p.add_argument(
        "--id-cols",
        type=str,
        default="",
        help="Comma-separated column names to pass through to output (e.g., id,timestamp,x,y,z)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output CSV path to save predictions (batch or single)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    bundle = load_bundle(args.model)
    feature_names: List[str] = list(bundle["feature_names"])  # ensure list
    id_cols: Optional[List[str]] = [c for c in args.id_cols.split(",") if c] if args.id_cols else None

    X_full = read_features_csv(
        args.csv,
        feature_names,
        id_cols=id_cols,
        skip_first=args.skip_first,
        limit=args.limit,
    )

    # Build the feature-only frame in correct order for the model
    X_features = X_full[feature_names].copy()

    preds = predict(bundle, args.target, X_features)

    # Output
    if args.out:
        save_predictions(X_full, preds, args.target, args.out)
        print(f"Saved predictions to {args.out}")
    else:
        # Print plain text preview of first few predictions
        preview = ", ".join(f"{v:.6f}" for v in preds[:10])
        print(f"target={args.target} n={len(preds)} preds_head=[{preview}]")


if __name__ == "__main__":
    main()


