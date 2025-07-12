import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json

# Load config
with open("config/config.json") as f:
    config = json.load(f)

CSV_PATH = config.get("data_path", "data/processed/scored_posts_with_users.csv")
FEATURE_COLUMNS = config["feature_columns"]["lightgbm"]
TARGET_COLUMN = config.get("target_column", "target_label")
PARAM_GRID = config["hyperparameters"]["lightgbm"]
MODEL_PATH = config["model_path"]["lightgbm"]

# Load and preprocess training data
def load_training_data(csv_path):
    df = pd.read_csv(csv_path)

    df["user_follows_tag"] = df["user_follows_tag"].astype(int)
    df["is_buddy_post"] = df["is_buddy_post"].astype(int)

    # Encode content type
    content_type_map = {"Video": 0.08, "Image": 0.04, "Link": 0.02}
    df["Post Type"] = df["Post Type"].map(content_type_map).fillna(0).astype(float)

    # Encode Weekday Type
    weekday_type_map = {"Weekday": 0, "Weekend": 1}
    df["Weekday Type"] = df["Weekday Type"].map(weekday_type_map).fillna(0).astype(int)

    # Encode Time Periods
    time_period_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
    df["Time Periods"] = df["Time Periods"].map(time_period_map).fillna(0).astype(int)

    # Encode karma_bucket
    karma_bucket_map = {"low": 0, "medium": 1, "high": 2}
    df["karma_bucket"] = df["karma_bucket"].map(karma_bucket_map).fillna(0).astype(int)

    # Define features
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y, FEATURE_COLUMNS

# Load data
X, y, feature_cols = load_training_data(CSV_PATH)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create base model
base_model = LGBMRegressor(random_state=42, verbose = -1)

# Grid Search with lighter settings
grid_search = GridSearchCV(estimator=base_model,
                           param_grid=PARAM_GRID,
                           cv=3,
                           scoring='r2',
                           n_jobs=1,
                           verbose=0)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get the best estimator
model = grid_search.best_estimator_

print("✅ Best Parameters from Grid Search:", grid_search.best_params_)

# Predict and evaluate
y_pred = model.predict(X_test)

print("🔍 LightGBM Test Results:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# ========== 🔁 Cross-Validation Section ==========
print("\n🔁 Cross-Validation (3-fold):")
r2_scores = cross_val_score(model, X, y, cv=3, scoring="r2", n_jobs=1)
mse_scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_squared_error", n_jobs=1)
mae_scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error", n_jobs=1)

print("Average R^2 Score:      {:.4f}".format(np.mean(r2_scores)))
print("Average RMSE:           {:.4f}".format(np.mean(np.sqrt(-mse_scores))))
print("Average MAE:            {:.4f}".format(np.mean(-mae_scores)))

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values(by="mean_test_score", ascending=False)
print(cv_results[["mean_test_score", "params"]].head())

# Save model
joblib.dump((model, FEATURE_COLUMNS), MODEL_PATH)
