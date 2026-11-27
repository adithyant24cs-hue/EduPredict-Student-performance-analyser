"""
Train the EduPredict dropout model on THIS computer and save it into the 'models' folder.

Run from the project root:
    python scripts/train_and_save_model.py

It looks for the dropout dataset in this order:
    1. data/processed/dropout_clean.csv   (from Phase 3)
    2. data/raw/data.csv                  (original UCI file, ';' separated)
    3. data/raw/dropout_raw.csv           (raw copy saved in Phase 2)
or you can give a path:  python scripts/train_and_save_model.py path/to/file.csv
"""
import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

PRIMARY = "sem1"
THRESHOLD = 0.40

# ----------------------------------------------------------------------------
# Feature sets (same as Phase 5 and Phase 6)
# ----------------------------------------------------------------------------
ENROLLMENT = ["Marital status", "Application mode", "Application order", "Course",
              "Daytime/evening attendance", "Previous qualification",
              "Previous qualification (grade)", "Admission grade", "Displaced",
              "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder",
              "Age at enrollment"]
SEM1 = [f"Curricular units 1st sem ({s})" for s in
        ["credited", "enrolled", "evaluations", "approved", "grade", "without evaluations"]]
SEM2 = [c.replace("1st", "2nd") for c in SEM1]
FEATURE_SETS = {"enrollment": ENROLLMENT, "sem1": ENROLLMENT + SEM1, "full": ENROLLMENT + SEM1 + SEM2}
CATEGORICAL = ["Marital status", "Application mode", "Application order", "Course", "Previous qualification"]
BINARY = ["Daytime/evening attendance", "Displaced", "Debtor",
          "Tuition fees up to date", "Gender", "Scholarship holder"]
NUMERIC = [c for c in FEATURE_SETS["full"] if c not in CATEGORICAL + BINARY]
FLOAT_COLUMNS = {"Previous qualification (grade)", "Admission grade", "Curricular units 1st sem (grade)"}


def find_data_file():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for rel in ["data/processed/dropout_clean.csv", "data/raw/data.csv", "data/raw/dropout_raw.csv"]:
        p = PROJECT_ROOT / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find the dataset. Put data.csv (UCI dropout file) in data/raw/ "
        "or dropout_clean.csv in data/processed/, or pass the file path as an argument."
    )


def load_and_clean(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip().str.replace("Nacionality", "Nationality", regex=False)
    df = df.drop_duplicates().reset_index(drop=True)
    df["Target"] = df["Target"].astype(str).str.strip().str.capitalize()
    assert set(df["Target"].unique()) == {"Dropout", "Enrolled", "Graduate"}, "Unexpected Target labels"
    missing = [c for c in FEATURE_SETS["full"] if c not in df.columns]
    assert not missing, f"Missing columns in the file: {missing}"
    assert df[FEATURE_SETS["full"]].isnull().sum().sum() == 0, "Dataset has missing values"
    for c in CATEGORICAL + BINARY:
        df[c] = df[c].astype("int64")
    df["Dropout_flag"] = (df["Target"] == "Dropout").astype(int)
    return df


def build_preprocessor(feature_list):
    cat = [c for c in feature_list if c in CATEGORICAL]
    binary = [c for c in feature_list if c in BINARY]
    num = [c for c in feature_list if c in NUMERIC]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("bin", "passthrough", binary),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=20, sparse_output=False), cat),
        ],
        remainder="drop", sparse_threshold=0, verbose_feature_names_out=False,
    )


def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                class_weight="balanced_subsample",
                                                random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
    }


def group_importance(series, cols):
    grouped = {}
    for key, value in series.items():
        base = next((c for c in cols if key == c or key.startswith(c + "_")), key)
        grouped[base] = grouped.get(base, 0) + float(value)
    return pd.Series(grouped).sort_values(ascending=False)


def main():
    data_path = find_data_file()
    print("Using data file:", data_path)
    df = load_and_clean(data_path)
    print("Rows:", len(df), "| dropout share:", f"{df['Dropout_flag'].mean() * 100:.1f}%")

    X = df[FEATURE_SETS["full"]]
    y = df["Dropout_flag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print("Train:", X_train.shape, "| Test:", X_test.shape)

    cols = FEATURE_SETS[PRIMARY]
    final_pipeline = Pipeline([
        ("prep", build_preprocessor(cols)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ]).fit(X_train[cols], y_train)
    prob = final_pipeline.predict_proba(X_test[cols])[:, 1]
    for th in (0.5, THRESHOLD):
        pred = (prob >= th).astype(int)
        print(f"Primary model at threshold {th}: recall {recall_score(y_test, pred):.3f} "
              f"| precision {precision_score(y_test, pred):.3f}")
    print(f"Primary model ROC-AUC: {roc_auc_score(y_test, prob):.3f}")

    print("Comparing all models (this can take a minute)...")
    comparison, rf_importance = [], None
    for fs, fcols in FEATURE_SETS.items():
        for name, model in make_models().items():
            pipe = Pipeline([("prep", build_preprocessor(fcols)), ("clf", model)]).fit(X_train[fcols], y_train)
            pr = pipe.predict(X_test[fcols])
            pp = pipe.predict_proba(X_test[fcols])[:, 1]
            comparison.append({
                "feature_set": fs, "model": name,
                "accuracy": round(float(accuracy_score(y_test, pr)), 3),
                "precision": round(float(precision_score(y_test, pr)), 3),
                "recall": round(float(recall_score(y_test, pr)), 3),
                "f1": round(float(f1_score(y_test, pr)), 3),
                "roc_auc": round(float(roc_auc_score(y_test, pp)), 3),
            })
            if fs == PRIMARY and name == "Random Forest":
                rf_importance = pd.Series(pipe[-1].feature_importances_,
                                          index=pipe[:-1].get_feature_names_out())
    rf_grouped = group_importance(rf_importance, cols)

    first = X_test.iloc[0]
    smoke_row = {c: (float(first[c]) if c in FLOAT_COLUMNS else int(first[c])) for c in cols}
    smoke_df = pd.DataFrame([smoke_row])[cols]

    metadata = {
        "model_name": "Logistic Regression",
        "feature_set": PRIMARY,
        "target": "Dropout_flag (1 = Dropout, 0 = Graduate or Enrolled)",
        "features": cols,
        "categorical": [c for c in cols if c in CATEGORICAL],
        "binary": [c for c in cols if c in BINARY],
        "numeric": [c for c in cols if c in NUMERIC],
        "alert_threshold": THRESHOLD,
        "allowed_values": {c: sorted(int(v) for v in X_train[c].unique())
                           for c in cols if c in CATEGORICAL + BINARY},
        "numeric_ranges": {c: [float(X_train[c].min()), float(X_train[c].max())]
                           for c in cols if c in NUMERIC},
        "training": {"n_train": int(len(X_train)), "n_test": int(len(X_test)),
                     "dropout_rate_train": round(float(y_train.mean()), 4), "random_state": 42},
        "test_metrics": {
            "threshold_0.50": {"recall": round(float(recall_score(y_test, prob >= 0.5)), 3),
                               "precision": round(float(precision_score(y_test, prob >= 0.5)), 3)},
            "threshold_0.40": {"recall": round(float(recall_score(y_test, prob >= THRESHOLD)), 3),
                               "precision": round(float(precision_score(y_test, prob >= THRESHOLD)), 3)},
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 3),
        },
        "model_comparison": comparison,
        "random_forest_top_features": {k: round(float(v), 3) for k, v in rf_grouped.head(10).items()},
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "smoke_test": {"row": smoke_row,
                       "expected_probability": float(final_pipeline.predict_proba(smoke_df)[:, 1][0])},
    }

    model_path = MODELS_DIR / "dropout_pipeline_sem1.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"
    joblib.dump(final_pipeline, model_path, compress=3)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    reloaded = joblib.load(model_path)
    same = np.allclose(reloaded.predict_proba(X_test[cols])[:, 1], prob)
    print("Reloaded model gives identical predictions:", same)
    assert same
    print("Saved:", model_path)
    print("Saved:", meta_path)
    print("scikit-learn used:", sklearn.__version__)


if __name__ == "__main__":
    main()