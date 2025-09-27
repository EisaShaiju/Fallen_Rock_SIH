# step3_improved_hazard_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV

# Load data
df = pd.read_csv("extracted_data/rocfall_path_with_seeders.csv")

# Feature Engineering - Create meaningful predictors
def engineer_features(df):
    # Slope geometry features
    # Only keep seeder_z (drop elevation, release_energy_potential)

    # Slope (approximate: dz/dxy between consecutive seeders)
    dx = df['seeder_x'].diff().fillna(0)
    dy = df['seeder_y'].diff().fillna(0)
    dz = df['seeder_z'].diff().fillna(0)
    dxy = np.sqrt(dx**2 + dy**2)
    df['slope'] = np.degrees(np.arctan2(dz, dxy.replace(0, np.nan)))  # degrees
    df['slope'] = df['slope'].fillna(0)

    # Only keep normalized coordinates (drop raw x/y)
    df['x_normalized'] = (df['seeder_x'] - df['seeder_x'].min()) / (df['seeder_x'].max() - df['seeder_x'].min())
    df['y_normalized'] = (df['seeder_y'] - df['seeder_y'].min()) / (df['seeder_y'].max() - df['seeder_y'].min())

    # Feature engineering from time and events
    df['log_travel_time'] = np.log1p(df['travel_time_s'])
    df['events_per_time'] = df['events'] / (df['travel_time_s'] + 1e-6)
    # Only keep log_travel_time and events_per_time (drop squared versions)

    return df


# Apply feature engineering
df = engineer_features(df)

# Define hazard based on runout (but don't use runout as feature)
RUNOUT_THRESHOLD = 150
df["hazard_label"] = (df["runout_m"] > RUNOUT_THRESHOLD).astype(int)


# Features: Only keep uncorrelated/important ones
features = [
    "seeder_z", "x_normalized", "y_normalized", "slope", "log_travel_time", "events_per_time"
]

# --- Correlation Heatmap ---
plt.figure(figsize=(12, 8))
corr = df[[*features, 'hazard_label']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

X = df[features].copy()
y = df["hazard_label"]

# Check class balance
print(f"Class distribution:")
print(y.value_counts())
print(f"Hazard rate: {y.mean():.3f}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)



# Train model with fixed parameters (change as needed)
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.5,
    reg_alpha=0.1,
    reg_lambda=1,
    min_child_weight=3,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFeature Importances:")
for feat, imp in zip(features, model.feature_importances_):
    print(f"{feat}: {imp:.4f}")

# # Save model and scaler
# joblib.dump(model, "extracted_data/xgb_hazard_model_improved.pkl")
# joblib.dump(scaler, "extracted_data/feature_scaler.pkl")