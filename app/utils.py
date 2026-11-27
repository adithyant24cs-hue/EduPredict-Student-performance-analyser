"""Shared helpers for EduPredict: paths and model loading."""
import json
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "dropout_pipeline_sem1.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


class ModelNotAvailableError(Exception):
    """Raised when the saved model or its metadata cannot be loaded."""


@lru_cache(maxsize=1)
def load_metadata():
    """Read model_metadata.json (feature list, threshold, allowed values, metrics)."""
    if not METADATA_PATH.exists():
        raise ModelNotAvailableError(
            f"Metadata file not found: {METADATA_PATH}. "
            "Copy model_metadata.json into the 'models' folder (Phase 7)."
        )
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ModelNotAvailableError(f"model_metadata.json is not valid JSON: {exc}") from exc


def environment_warnings():
    """Return a list of warning messages if this PC's library versions differ from training."""
    problems = []
    try:
        trained = load_metadata().get("versions", {}).get("scikit_learn")
    except ModelNotAvailableError:
        return problems
    if trained and trained != sklearn.__version__:
        problems.append(
            f"scikit-learn version mismatch: model trained with {trained}, "
            f"this computer has {sklearn.__version__}. "
            f"Fix with: pip install scikit-learn=={trained}"
        )
    return problems


@lru_cache(maxsize=1)
def load_model():
    """Load the saved preprocessing + Logistic Regression pipeline."""
    if not MODEL_PATH.exists():
        raise ModelNotAvailableError(
            f"Model file not found: {MODEL_PATH}. "
            "Copy dropout_pipeline_sem1.joblib into the 'models' folder (Phase 7)."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # version warnings are reported by environment_warnings()
        try:
            return joblib.load(MODEL_PATH)
        except Exception as exc:  # corrupted file or incompatible library version
            raise ModelNotAvailableError(
                f"Could not load the model ({type(exc).__name__}: {exc}). "
                "This usually means scikit-learn on this computer differs from Colab's version."
            ) from exc