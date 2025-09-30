"""
Train advanced risk models on the generated advanced_rockfall_dataset.csv.

Targets:
- risk_prob_7d (continuous probability 0-1)
- risk_severity (0-1 proxy)

Outputs:
- joblib file with best models and metadata
- simple plots for predictions vs actual and feature importance
"""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse


RANDOM_STATE = 42
DATA_PATH = '../data/advanced_rockfall_dataset.csv'
MODEL_OUT = '../outputs/advanced_risk_models.joblib'


def load_data(path: str, limit: int | None = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    if limit is not None and limit > 0 and limit < len(df):
        df = df.sample(n=limit, random_state=seed)
    df = df[:5000]
    return df


def prepare_X_y(df: pd.DataFrame):
    target_cols = ['risk_prob_7d', 'risk_severity']
    drop_cols = target_cols + ['kinetic_energy', 'impact_position', 'runout_distance']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    return X, y, feature_cols, target_cols


def create_models() -> Dict[str, object]:
    models = {
        'ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]),
        'random_forest': RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, max_depth=None, min_samples_split=4
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=300, random_state=RANDOM_STATE, max_depth=3, learning_rate=0.05
        ),
    }
    return models


def train_and_eval(X: pd.DataFrame, y: pd.Series, model_name: str, model) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    # Cross-val on train
    try:
        if isinstance(model, Pipeline):
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    except Exception:
        cv_scores = np.array([np.nan])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    return {
        'model': model,
        'metrics': {'mse': mse, 'mae': mae, 'r2': r2, 'rmse': rmse, 'cv_r2_mean': np.nanmean(cv_scores)},
        'splits': {'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred},
    }


def pick_best(results: Dict[str, Dict]) -> str:
    best_name = None
    best_r2 = -np.inf
    for name, res in results.items():
        if res['metrics']['r2'] > best_r2:
            best_r2 = res['metrics']['r2']
            best_name = name
    return best_name


def plot_predictions(y_test: pd.Series, y_pred: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, s=8, alpha=0.4)
    vmin = min(y_test.min(), np.min(y_pred))
    vmax = max(y_test.max(), np.max(y_pred))
    plt.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names, title: str, out_path: str):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        order = np.argsort(imp)[::-1][:20]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=imp[order], y=np.array(feature_names)[order], orient='h')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()


def parse_args():
    p = argparse.ArgumentParser(description='Train advanced risk models')
    p.add_argument('--data', type=str, default=DATA_PATH)
    p.add_argument('--limit', type=int, default=0, help='Subsample rows for quick runs (0=all)')
    p.add_argument('--seed', type=int, default=RANDOM_STATE)
    return p.parse_args()


def main():
    args = parse_args()
    df = load_data(args.data, limit=(args.limit if args.limit and args.limit > 0 else None), seed=args.seed)
    X, y, feature_cols, target_cols = prepare_X_y(df)
    models = create_models()

    artifacts = {}
    for target in target_cols:
        results = {}
        for name, model in models.items():
            res = train_and_eval(X, y[target], name, model)
            results[name] = res
            print(f"{target} | {name}: R2={res['metrics']['r2']:.4f} RMSE={res['metrics']['rmse']:.4f} MAE={res['metrics']['mae']:.4f}")

        best_name = pick_best(results)
        best = results[best_name]
        print(f"Best for {target}: {best_name} (R2={best['metrics']['r2']:.4f})")

        # Plots
        plot_predictions(best['splits']['y_test'], best['splits']['y_pred'],
                         title=f'{target}: {best_name} predictions',
                         out_path=f'adv_{target}_pred.png')
        plot_feature_importance(best['model'], feature_cols,
                                title=f'{target}: feature importance ({best_name})',
                                out_path=f'adv_{target}_featimp.png')

        artifacts[target] = {
            'model_name': best_name,
            'model': best['model'],
            'metrics': best['metrics'],
        }

    bundle = {
        'artifacts': artifacts,
        'feature_names': feature_cols,
        'target_names': target_cols,
        'random_state': RANDOM_STATE,
    }
    joblib.dump(bundle, MODEL_OUT)
    print(f"Saved models to {MODEL_OUT}")


if __name__ == '__main__':
    main()


