from __future__ import annotations

import json
import logging
import warnings
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def configure_logging() -> None:
    """Configure console logging for pipeline execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def ensure_output_dir() -> Path:
    """Create the output folder if it does not already exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the churn dataset from disk."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    return pd.read_csv(file_path)


def infer_target_column(df: pd.DataFrame) -> str:
    """Infer the most likely binary target column."""
    preferred_names = ["churn", "target", "label", "class", "exited", "attrition"]
    lower_map = {column.lower(): column for column in df.columns}
    for name in preferred_names:
        if name in lower_map:
            return lower_map[name]
    for column in df.columns:
        if any(keyword in column.lower() for keyword in preferred_names):
            return column
    return df.columns[-1]


def detect_identifier_columns(df: pd.DataFrame, target_column: str) -> List[str]:
    """Detect columns that look like identifiers and should be removed."""
    identifier_columns: List[str] = []
    row_count = len(df)
    for column in df.columns:
        if column == target_column:
            continue
        if "id" in column.lower():
            unique_ratio = df[column].nunique(dropna=False) / max(row_count, 1)
            if unique_ratio > 0.8:
                identifier_columns.append(column)
    return identifier_columns


def clean_dataset(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, List[str]]:
    """Clean column names, coerce numeric-like values, remove duplicates, and drop IDs."""
    cleaned_df = df.copy()
    cleaned_df.columns = [column.strip() for column in cleaned_df.columns]

    if target_column not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    if "TotalCharges" in cleaned_df.columns:
        cleaned_df["TotalCharges"] = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")

    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    identifier_columns = detect_identifier_columns(cleaned_df, target_column)
    if identifier_columns:
        cleaned_df = cleaned_df.drop(columns=identifier_columns)

    return cleaned_df, identifier_columns


def inspect_data(df: pd.DataFrame, title: str = "Dataset Overview") -> None:
    """Print descriptive dataset diagnostics."""
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\nSummary statistics:")
    print(df.describe(include="all").transpose().to_string())


def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Separate features and target and encode the target for binary classification."""
    features = df.drop(columns=[target_column])
    target_raw = df[target_column].astype(str)

    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target_raw)
    return features, target_encoded, label_encoder


def create_preprocessor(features: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build preprocessing pipelines for numeric and categorical features."""
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def get_model_definitions() -> Dict[str, object]:
    """Return the models required by the internship task."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def safe_predict_proba(pipeline: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    """Return probabilities for the positive class when available."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x_test)[:, 1]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(x_test)
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if max_score == min_score:
            return np.zeros_like(scores, dtype=float)
        return (scores - min_score) / (max_score - min_score)
    return np.zeros(len(x_test), dtype=float)


def train_and_evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, Dict[str, object]]]:
    """Train all requested models and compute evaluation metrics."""
    preprocessor, numeric_features, categorical_features = create_preprocessor(x_train)
    model_definitions = get_model_definitions()

    trained_models: Dict[str, Pipeline] = {}
    evaluation_details: Dict[str, Dict[str, object]] = {}
    metrics_rows: List[Dict[str, object]] = []
    confusion_filename_map = {
        "Logistic Regression": "confusion_matrix_logistic.png",
        "Decision Tree Classifier": "confusion_matrix_decision_tree.png",
        "Random Forest Classifier": "confusion_matrix_random_forest.png",
    }

    for model_name, estimator in model_definitions.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        y_prob = safe_predict_proba(pipeline, x_test)

        metrics_row = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, y_prob),
        }

        metrics_rows.append(metrics_row)
        trained_models[model_name] = pipeline
        evaluation_details[model_name] = {
            "y_pred": y_pred,
            "y_prob": y_prob,
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

        plot_confusion_matrix(
            evaluation_details[model_name]["confusion_matrix"],
            model_name,
            output_dir
            / confusion_filename_map.get(
                model_name,
                f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
            ),
        )

    comparison_df = pd.DataFrame(metrics_rows).sort_values(
        by=["F1 Score", "ROC AUC", "Accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    plot_model_accuracy_comparison(comparison_df, output_dir / "model_accuracy_comparison.png")

    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline = trained_models[best_model_name]
    plot_roc_curve(trained_models, x_test, y_test, output_dir / "roc_curve.png")
    plot_feature_importance(best_pipeline, output_dir / "feature_importance.png")

    return comparison_df, trained_models, evaluation_details


def plot_target_distribution(df: pd.DataFrame, target_column: str, output_path: Path) -> None:
    """Plot the class distribution of the target variable."""
    plt.figure(figsize=(8, 6))
    counts = df[target_column].value_counts().sort_index()
    ax = sns.barplot(x=counts.index.astype(str), y=counts.values, palette="viridis")
    plt.title("Target Distribution")
    plt.xlabel(target_column)
    plt.ylabel("Count")
    for index, value in enumerate(counts.values):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, target_column: str, output_path: Path) -> None:
    """Plot a correlation heatmap for numeric features and encoded target."""
    encoded_df = df.copy()
    encoded_df[target_column] = LabelEncoder().fit_transform(encoded_df[target_column].astype(str))
    numeric_df = encoded_df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return

    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_distribution(
    df: pd.DataFrame,
    target_column: str,
    numeric_features: List[str],
    categorical_features: List[str],
    output_path: Path,
) -> None:
    """Plot histograms for numeric features and count plots for categorical features."""
    plot_columns = numeric_features + categorical_features
    if not plot_columns:
        return

    n_plots = len(plot_columns)
    n_cols = 3
    n_rows = ceil(n_plots / n_cols)
    plt.figure(figsize=(18, 5 * n_rows))

    for index, column in enumerate(plot_columns, start=1):
        ax = plt.subplot(n_rows, n_cols, index)
        if column in numeric_features:
            sns.histplot(df[column].dropna(), kde=True, bins=30, color="#2a9d8f", ax=ax)
            ax.set_title(f"Histogram: {column}")
        else:
            sns.countplot(data=df, x=column, palette="Set2", ax=ax)
            ax.set_title(f"Count Plot: {column}")
            ax.tick_params(axis="x", rotation=35)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

    for index in range(n_plots + 1, n_rows * n_cols + 1):
        plt.subplot(n_rows, n_cols, index).axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplots(df: pd.DataFrame, target_column: str, numeric_features: List[str], output_path: Path) -> None:
    """Plot boxplots of numeric features against the target class."""
    if not numeric_features:
        return

    n_cols = 2
    n_rows = ceil(len(numeric_features) / n_cols)
    plt.figure(figsize=(14, 5 * n_rows))

    for index, column in enumerate(numeric_features, start=1):
        ax = plt.subplot(n_rows, n_cols, index)
        sns.boxplot(data=df, x=target_column, y=column, palette="Set3", ax=ax)
        ax.set_title(f"Boxplot of {column} by {target_column}")

    for index in range(len(numeric_features) + 1, n_rows * n_cols + 1):
        plt.subplot(n_rows, n_cols, index).axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(conf_matrix: np.ndarray, model_name: str, output_path: Path) -> None:
    """Plot and save a confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(trained_models: Dict[str, Pipeline], x_test: pd.DataFrame, y_test: np.ndarray, output_path: Path) -> None:
    """Plot ROC curves for every trained model."""
    plt.figure(figsize=(8, 7))
    for model_name, pipeline in trained_models.items():
        y_prob = safe_predict_proba(pipeline, x_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_accuracy_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Plot a model accuracy comparison bar chart."""
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=comparison_df, x="Model", y="Accuracy", palette="mako")
    plt.xticks(rotation=20, ha="right")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.3f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(best_pipeline: Pipeline, output_path: Path, top_n: int = 15) -> None:
    """Plot feature importance or absolute coefficients for the best model."""
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_).ravel()
    else:
        return

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importance_values})
        .sort_values(by="Importance", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="crest")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_results_summary(comparison_df: pd.DataFrame) -> str:
    """Generate a short explanation for the best performing model."""
    best_row = comparison_df.iloc[0]
    runner_up = comparison_df.iloc[1] if len(comparison_df) > 1 else None
    reason = [
        f"The best model was {best_row['Model']} because it achieved the strongest overall balance of F1 score ({best_row['F1 Score']:.3f}) and ROC AUC ({best_row['ROC AUC']:.3f}).",
    ]
    if runner_up is not None:
        reason.append(
            f"It outperformed the next best model by {best_row['F1 Score'] - runner_up['F1 Score']:.3f} F1 points, which suggests it captured the churn patterns more effectively."
        )
    if "Random Forest" in best_row["Model"]:
        reason.append("This is expected because random forests model non-linear relationships and feature interactions well while staying robust to noisy features.")
    elif "Decision Tree" in best_row["Model"]:
        reason.append("This indicates that interpretable rule-based splits matched the structure of the churn dataset well.")
    else:
        reason.append("This suggests that a linear decision boundary was sufficient for the processed feature space.")
    return " ".join(reason)


def run_pipeline(data_file: Path | None = None) -> Dict[str, object]:
    """Run the complete churn prediction workflow."""
    configure_logging()
    output_dir = ensure_output_dir()
    data_path = data_file or (BASE_DIR / "customer_churn_data.csv")

    logging.info("Loading dataset from %s", data_path)
    df = load_dataset(data_path)
    target_column = infer_target_column(df)

    inspect_data(df, title="Raw Dataset Overview")

    cleaned_df, dropped_identifier_columns = clean_dataset(df, target_column)
    logging.info("Dropped identifier columns: %s", dropped_identifier_columns if dropped_identifier_columns else "None")

    inspect_data(cleaned_df, title="Cleaned Dataset Overview")

    plot_target_distribution(cleaned_df, target_column, output_dir / "target_distribution.png")
    plot_correlation_heatmap(cleaned_df, target_column, output_dir / "correlation_heatmap.png")

    features, target_encoded, label_encoder = split_features_target(cleaned_df, target_column)
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()

    plot_feature_distribution(
        cleaned_df,
        target_column,
        numeric_features,
        categorical_features,
        output_dir / "feature_distribution.png",
    )
    plot_boxplots(cleaned_df, target_column, numeric_features, output_dir / "boxplots.png")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target_encoded,
    )

    comparison_df, trained_models, evaluation_details = train_and_evaluate_models(
        x_train,
        x_test,
        y_train,
        y_test,
        output_dir,
    )

    summary_text = build_results_summary(comparison_df)
    best_model_name = comparison_df.iloc[0]["Model"]

    logging.info("Best model: %s", best_model_name)
    logging.info("%s", summary_text)

    results = {
        "raw_data": df,
        "cleaned_data": cleaned_df,
        "target_column": target_column,
        "dropped_identifier_columns": dropped_identifier_columns,
        "label_encoder": label_encoder,
        "comparison_table": comparison_df,
        "trained_models": trained_models,
        "evaluation_details": evaluation_details,
        "best_model_name": best_model_name,
        "summary_text": summary_text,
        "output_dir": output_dir,
    }

    with open(output_dir / "results_summary.json", "w", encoding="utf-8") as file_handle:
        json.dump(
            {
                "best_model_name": best_model_name,
                "summary_text": summary_text,
                "comparison_table": comparison_df.to_dict(orient="records"),
            },
            file_handle,
            indent=2,
        )

    return results


def main() -> None:
    """Execute the churn prediction workflow end to end."""
    try:
        run_pipeline()
    except Exception as error:
        logging.exception("Pipeline execution failed: %s", error)
        raise


if __name__ == "__main__":
    main()
