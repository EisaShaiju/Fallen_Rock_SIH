# step4_predict_and_visualize_hazard.py
"""
Predict hazard for a grid of (x, y, z) points and visualize as a 2D/3D hazard map.
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# --- Load trained model and scaler ---
model = joblib.load("extracted_data/xgb_hazard_model_improved.pkl")
scaler = joblib.load("extracted_data/feature_scaler.pkl")

# --- Feature engineering function (copied from training script) ---
def engineer_features(df):
    dx = df['seeder_x'].diff().fillna(0)
    dy = df['seeder_y'].diff().fillna(0)
    dz = df['seeder_z'].diff().fillna(0)
    dxy = np.sqrt(dx**2 + dy**2)
    df['slope'] = np.degrees(np.arctan2(dz, dxy.replace(0, np.nan)))  # degrees
    df['slope'] = df['slope'].fillna(0)
    df['x_normalized'] = (df['seeder_x'] - df['seeder_x'].min()) / (df['seeder_x'].max() - df['seeder_x'].min())
    df['y_normalized'] = (df['seeder_y'] - df['seeder_y'].min()) / (df['seeder_y'].max() - df['seeder_y'].min())
    df['log_travel_time'] = np.log1p(df['travel_time_s'])
    df['events_per_time'] = df['events'] / (df['travel_time_s'] + 1e-6)
    return df

# --- Define grid bounds (use min/max from your data) ---
df = pd.read_csv("extracted_data/rocfall_path_with_seeders.csv")
df = engineer_features(df)
x_min, x_max = df['seeder_x'].min(), df['seeder_x'].max()
y_min, y_max = df['seeder_y'].min(), df['seeder_y'].max()
z_min, z_max = df['seeder_z'].min(), df['seeder_z'].max()

# --- Create grid ---
grid_size = 50  # adjust for resolution
xg = np.linspace(x_min, x_max, grid_size)
yg = np.linspace(y_min, y_max, grid_size)
zg = np.linspace(z_min, z_max, grid_size)

# For 2D: use surface at mean z
grid = np.array(np.meshgrid(xg, yg)).reshape(2, -1).T
z_val = np.mean(df['seeder_z'])
grid_z = np.full((grid.shape[0],), z_val)

# --- Feature engineering for grid ---
grid_df = pd.DataFrame({
    'seeder_x': grid[:, 0],
    'seeder_y': grid[:, 1],
    'seeder_z': grid_z,
    'travel_time_s': np.mean(df['travel_time_s']),  # use mean as placeholder
    'events': np.mean(df['events']),                # use mean as placeholder
})

def engineer_features_grid(df, ref_df):
    # Slope: set to mean (since grid is flat)
    df['slope'] = np.mean(ref_df['slope'])
    df['x_normalized'] = (df['seeder_x'] - ref_df['seeder_x'].min()) / (ref_df['seeder_x'].max() - ref_df['seeder_x'].min())
    df['y_normalized'] = (df['seeder_y'] - ref_df['seeder_y'].min()) / (ref_df['seeder_y'].max() - ref_df['seeder_y'].min())
    df['log_travel_time'] = np.log1p(df['travel_time_s'])
    df['events_per_time'] = df['events'] / (df['travel_time_s'] + 1e-6)
    return df

grid_df = engineer_features_grid(grid_df, df)

features = ["seeder_z", "x_normalized", "y_normalized", "slope", "log_travel_time", "events_per_time"]
X_grid = scaler.transform(grid_df[features])

# --- Predict hazard probability ---
hazard_proba = model.predict_proba(X_grid)[:, 1]

# --- Visualize as 2D heatmap ---
plt.figure(figsize=(8, 6))
plt.scatter(grid_df['seeder_x'], grid_df['seeder_y'], c=hazard_proba, cmap='coolwarm', s=30)
plt.colorbar(label='Predicted Hazard Probability')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Predicted Hazard Map (2D)')
plt.tight_layout()
plt.show()

# --- Optional: 3D scatter plot ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(grid_df['seeder_x'], grid_df['seeder_y'], grid_df['seeder_z'], c=hazard_proba, cmap='coolwarm', s=15)
fig.colorbar(p, ax=ax, label='Predicted Hazard Probability')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Predicted Hazard Map (3D)')
plt.tight_layout()
plt.show()
