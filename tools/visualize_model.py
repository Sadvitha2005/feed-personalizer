import os
import joblib
import json
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns.set(style="whitegrid")


def load_config(config_path="config/config.json"):
    with open(config_path) as f:
        return json.load(f)


def preprocess_dataframe(df, feature_names):
    df = df.copy()

    df["user_follows_tag"] = df["user_follows_tag"].astype(int)
    df["is_buddy_post"] = df["is_buddy_post"].astype(int)

    content_type_map = {
        "Video": 0.08, "Image": 0.04, "Link": 0.02,
        "video": 0.08, "image": 0.04, "text": 0.02
    }
    weekday_type_map = {"Weekday": 0, "Weekend": 1}
    time_period_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
    karma_bucket_map = {"low": 0, "medium": 1, "high": 2}

    df["Post Type"] = df["Post Type"].map(content_type_map).fillna(0).astype(float)
    df["Weekday Type"] = df["Weekday Type"].map(weekday_type_map).fillna(0).astype(int)
    df["Time Periods"] = df["Time Periods"].map(time_period_map).fillna(0).astype(int)
    df["karma_bucket"] = df["karma_bucket"].map(karma_bucket_map).fillna(0).astype(int)

    return df[feature_names]


def visualize_shap(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)


def visualize_evaluation(y_true, y_pred, feature_names, model):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("Actual vs Predicted")

    plt.subplot(1, 3, 2)
    residuals = y_true - y_pred
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title("Residual Distribution")

    plt.subplot(1, 3, 3)
    importances = model.feature_importances_
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importances")
    plt.xlabel("Importance")

    plt.tight_layout()
    plt.show()


def visualize_correlation_matrix(df, feature_names):
    plt.figure(figsize=(10, 8))
    corr = df[feature_names].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()


def main(config):
    model_type = config["model_type"]
    model_path = config["model_path"][model_type]
    feature_names = config["feature_columns"][model_type]
    data_path = config["data_path"]
    target_col = config["target_column"]

    print(f"✅ Loading model ({model_type}) from {model_path}")
    model, _ = joblib.load(model_path)

    print("📂 Reading dataset...")
    df = pd.read_csv(data_path)

    print("🧼 Preprocessing features...")
    X_test = preprocess_dataframe(df, feature_names)
    y_test = df[target_col]

    print("🔮 Predicting scores...")
    y_pred = model.predict(X_test)

    if config.get("enable_eval", False):
        print("📊 Showing evaluation plots...")
        visualize_evaluation(y_test, y_pred, feature_names, model)

    if config.get("enable_corr", False):
        print("📈 Showing correlation matrix...")
        visualize_correlation_matrix(X_test, feature_names)

    if config.get("enable_shap", False):
        print("🧠 Showing SHAP summary plot...")
        visualize_shap(model, X_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained model")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to config.json")
    parser.add_argument("--shap", action="store_true", help="Enable SHAP even if config disables it")
    parser.add_argument("--eval", action="store_true", help="Enable evaluation plots even if config disables it")
    parser.add_argument("--corr", action="store_true", help="Enable correlation matrix even if config disables it")

    args = parser.parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.shap:
        config["enable_shap"] = True
    if args.eval:
        config["enable_eval"] = True
    if args.corr:
        config["enable_corr"] = True

    main(config)
