"""
predictor.py
------------
Dropout prediction engine for EduPredict.

Uses the pipeline saved in Phase 7 (models/dropout_pipeline_sem1.joblib), which holds the
preprocessing AND the Logistic Regression model together, so training and prediction always
use identical preprocessing. Facts about the model (target meaning, the alert threshold,
allowed codes, value ranges) are read from models/model_metadata.json, not guessed.

Rules this file enforces
  * No prediction unless ALL 20 required features are provided.
  * Category / yes-no codes must be codes the model was trained on. (The pipeline would
    otherwise silently ignore an unknown code and still return a number.)
  * Values outside the training range give a warning, not a silent pass.

The risk levels are rule-of-thumb bands on the model's probability. The model was trained with
class_weight="balanced", so its probabilities are NOT calibrated real-world chances.
"""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = "dropout_pipeline_sem1.joblib"
METADATA_FILE = "model_metadata.json"

# In your Phase 5/7 work: Dropout_flag  1 = Dropout, 0 = Graduate or Enrolled.
DROPOUT_LABEL = 1

LOW_RISK_BELOW = 0.20            # below this: Low
DEFAULT_ALERT_THRESHOLD = 0.40   # used only if the metadata file has no alert_threshold

DISCLAIMER = (
    "This is a statistical estimate from a model trained on a public dataset of students at one "
    "Portuguese higher-education institution. It is not a certain outcome and is meant only to "
    "support, not replace, the judgement of teachers and counsellors."
)


class ModelLoadError(Exception):
    """Raised when the saved model cannot be found or loaded."""


# ----------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------
def load_model(model_path=None):
    """Loads the saved pipeline. Only load model files you created yourself."""
    path = Path(model_path) if model_path else MODEL_DIR / MODEL_FILE
    if not path.exists():
        raise ModelLoadError(
            f"Model file not found: {path}. Copy {MODEL_FILE} from Colab into the models folder."
        )
    try:
        pipeline = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load {path.name}: {exc}. A common cause is a different scikit-learn "
            "version than the one used in Colab (see model_metadata.json -> versions)."
        ) from exc
    if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "feature_names_in_"):
        raise ModelLoadError(f"{path.name} is not the expected trained pipeline.")
    if DROPOUT_LABEL not in list(pipeline.classes_):
        raise ModelLoadError(
            f"DROPOUT_LABEL = {DROPOUT_LABEL} is not one of the model's classes {list(pipeline.classes_)}."
        )
    return pipeline


def load_metadata(metadata_path=None) -> dict:
    """Returns the metadata dictionary, or {} if the file is missing or unreadable."""
    path = Path(metadata_path) if metadata_path else MODEL_DIR / METADATA_FILE
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def get_required_features(pipeline) -> list:
    return [str(name) for name in pipeline.feature_names_in_]


def get_alert_threshold(metadata: dict) -> float:
    try:
        value = float(metadata.get("alert_threshold", DEFAULT_ALERT_THRESHOLD))
        return value if 0 < value < 1 else DEFAULT_ALERT_THRESHOLD
    except (TypeError, ValueError):
        return DEFAULT_ALERT_THRESHOLD


# ----------------------------------------------------------------------
# 2. Input checking
# ----------------------------------------------------------------------
def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_number(value):
    """True/False -> 1/0, numeric text -> float. Anything else is returned unchanged."""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    return value


def check_inputs(pipeline, inputs: dict, metadata: dict) -> dict:
    """
    Checks every required feature.
    Returns {"clean": {feature: value}, "missing": [...], "invalid": {feature: reason},
             "warnings": [...], "ignored": [...]}
    """
    required = get_required_features(pipeline)
    inputs = inputs or {}
    allowed = metadata.get("allowed_values", {})
    ranges = metadata.get("numeric_ranges", {})

    clean, missing, invalid, warnings = {}, [], {}, []
    if not metadata:
        warnings.append("model_metadata.json was not found, so the allowed-value checks were skipped.")

    for feature in required:
        if feature not in inputs or _is_missing(inputs[feature]):
            missing.append(feature)
            continue
        value = _to_number(inputs[feature])
        try:
            number = float(value)
        except (TypeError, ValueError):
            invalid[feature] = "must be a number"
            continue
        if not math.isfinite(number):
            invalid[feature] = "must be a finite number"
            continue

        if feature in allowed:                      # category codes and yes/no flags
            if number not in [float(a) for a in allowed[feature]]:
                invalid[feature] = (f"{number:g} is not a code the model was trained on. "
                                    f"Allowed: {allowed[feature]}")
            else:
                clean[feature] = int(number)
        else:                                       # numeric measurements
            if number < 0:
                invalid[feature] = "cannot be negative"
                continue
            clean[feature] = number
            if feature in ranges:
                low, high = ranges[feature]
                if not (low <= number <= high):
                    warnings.append(
                        f"{feature} = {number:g} is outside the range seen in training "
                        f"({low:g} to {high:g}); treat the result with extra caution."
                    )

    enrolled = clean.get("Curricular units 1st sem (enrolled)")
    approved = clean.get("Curricular units 1st sem (approved)")
    if enrolled is not None and approved is not None and approved > enrolled:
        warnings.append("Approved 1st-semester units are higher than enrolled units. "
                        "Please check these two values.")

    ignored = [k for k in inputs if k not in required]
    return {"clean": clean, "missing": missing, "invalid": invalid,
            "warnings": warnings, "ignored": ignored}


# ----------------------------------------------------------------------
# 3. Risk wording and explanations
# ----------------------------------------------------------------------
def risk_level(probability: float, alert_threshold: float) -> str:
    """Low below 0.20, High from the model's alert threshold, Medium in between."""
    if probability < LOW_RISK_BELOW:
        return "Low"
    if probability < alert_threshold:
        return "Medium"
    return "High"


def _original_feature(encoded_name: str, required: list) -> str:
    name = str(encoded_name)
    for prefix in ("num__", "bin__", "cat__", "remainder__"):
        name = name.replace(prefix, "")
    for feature in sorted(required, key=len, reverse=True):
        if name == feature or name.startswith(feature + "_"):
            return feature
    return name


def student_factors(pipeline, frame: pd.DataFrame, top_n: int = 3):
    """
    Approximate PER-STUDENT explanation for a linear model. Only the standardised numeric
    features are used: they are centred on the average training student, so a contribution means
    "compared with the average student". (Yes/no flags and one-hot codes have no such baseline,
    so they are handled by the rules in recommendations.py instead.)
    Returns (pushing_risk_up, pushing_risk_down) as lists of feature names, or ([], []).
    """
    try:
        prep = pipeline.named_steps["prep"]
        model = pipeline.steps[-1][1]
        if not hasattr(model, "coef_"):
            return [], []
        transformed = prep.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        contributions = np.asarray(transformed)[0] * model.coef_[0]
        required = get_required_features(pipeline)
        numeric = set()
        for step_name, _, columns in prep.transformers_:
            if step_name == "num":
                numeric = set(columns)
        totals = {}
        for name, value in zip(prep.get_feature_names_out(), contributions):
            feature = _original_feature(name, required)
            if feature in numeric:
                totals[feature] = totals.get(feature, 0.0) + float(value)
        ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        up = [f for f, v in ordered if v > 0.05][:top_n]
        down = [f for f, v in reversed(ordered) if v < -0.05][:top_n]
        return up, down
    except Exception:
        return [], []


def get_top_features(pipeline, top_n: int = 10) -> pd.DataFrame:
    """GLOBAL importance (what matters most overall). Not specific to one student."""
    empty = pd.DataFrame(columns=["Feature", "Importance"])
    try:
        model = pipeline.steps[-1][1]
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            importance = np.mean(np.abs(np.atleast_2d(model.coef_)), axis=0)
        else:
            return empty
        names = pipeline[:-1].get_feature_names_out()
        if len(names) != len(importance):
            return empty
        table = pd.DataFrame({"Feature": [str(n) for n in names], "Importance": importance})
        return table.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)
    except Exception:
        return empty


# ----------------------------------------------------------------------
# 4. Prediction
# ----------------------------------------------------------------------
def predict_student(pipeline, inputs: dict, metadata: dict = None) -> dict:
    """
    Predicts for ONE student. Never raises for bad user input: check result["ok"].

    Keys: ok, error, missing_features, invalid_features, warnings, ignored_inputs,
          dropout_probability, flagged, risk_level, alert_threshold, drivers_up,
          drivers_down, explanation, disclaimer
    """
    metadata = load_metadata() if metadata is None else metadata
    alert = get_alert_threshold(metadata)
    result = {
        "ok": False, "error": None, "missing_features": [], "invalid_features": {},
        "warnings": [], "ignored_inputs": [], "dropout_probability": None, "flagged": None,
        "risk_level": None, "alert_threshold": alert, "drivers_up": [], "drivers_down": [],
        "explanation": [], "disclaimer": DISCLAIMER,
    }
    try:
        checked = check_inputs(pipeline, inputs, metadata)
        result["missing_features"] = checked["missing"]
        result["invalid_features"] = checked["invalid"]
        result["warnings"] = checked["warnings"]
        result["ignored_inputs"] = checked["ignored"]

        if checked["missing"] or checked["invalid"]:
            parts = []
            if checked["missing"]:
                parts.append(f"{len(checked['missing'])} required feature(s) are missing")
            if checked["invalid"]:
                parts.append(f"{len(checked['invalid'])} value(s) are not valid")
            result["error"] = ("Prediction not made: " + " and ".join(parts) +
                               ". The model is never run on guessed or invalid values.")
            return result

        required = get_required_features(pipeline)
        frame = pd.DataFrame([checked["clean"]], columns=required)

        column = list(pipeline.classes_).index(DROPOUT_LABEL)
        probability = float(pipeline.predict_proba(frame)[0][column])
        flagged = probability >= alert
        level = risk_level(probability, alert)
        up, down = student_factors(pipeline, frame)

        explanation = [
            f"Estimated dropout score: {probability * 100:.1f}% -> {level} risk "
            f"(Low below {LOW_RISK_BELOW * 100:.0f}%, High from {alert * 100:.0f}%).",
        ]
        test = metadata.get("test_metrics", {}).get(f"threshold_{alert:.2f}", {})
        if test.get("recall") is not None and test.get("precision") is not None:
            explanation.append(
                f"At the {alert * 100:.0f}% alert threshold, on held-out test students the model found "
                f"{test['recall'] * 100:.0f}% of those who dropped out, and "
                f"{test['precision'] * 100:.0f}% of the students it flagged did drop out."
            )
        if up:
            explanation.append("Factors pushing this student's score up: " + ", ".join(up) + ".")
        if down:
            explanation.append("Factors pushing this student's score down: " + ", ".join(down) + ".")

        result.update({
            "ok": True, "dropout_probability": probability, "flagged": flagged,
            "risk_level": level, "drivers_up": up, "drivers_down": down,
            "explanation": explanation,
        })
        return result

    except Exception as exc:
        result["error"] = (f"The model could not process these inputs ({type(exc).__name__}). "
                           "Check each value and try again.")
        return result


# ----------------------------------------------------------------------
# 5. Self-test:   python app/predictor.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    try:
        pipe = load_model()
    except ModelLoadError as err:
        print("MODEL LOAD FAILED:", err)
        sys.exit(1)
    meta = load_metadata()
    print("Required features :", len(get_required_features(pipe)))
    print("Classes           :", list(pipe.classes_), "| dropout label =", DROPOUT_LABEL)
    print("Alert threshold   :", get_alert_threshold(meta))

    smoke = meta.get("smoke_test", {})
    if smoke:
        out = predict_student(pipe, smoke["row"], meta)
        expected = smoke["expected_probability"]
        print("\nSmoke test from Phase 7")
        print("  ok:", out["ok"], "| probability:", round(out["dropout_probability"], 6),
              "| expected:", round(expected, 6))
        print("  MATCH:", abs(out["dropout_probability"] - expected) < 1e-6)
        for line in out["explanation"]:
            print("  -", line)

    refused = predict_student(pipe, {}, meta)
    print("\nEmpty input refused:", not refused["ok"], "| missing:", len(refused["missing_features"]))
