import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils import load_metadata, load_model, environment_warnings  # noqa: E402


@pytest.fixture(scope="module")
def meta():
    return load_metadata()


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_metadata_has_required_keys(meta):
    for key in ["features", "alert_threshold", "allowed_values", "numeric_ranges", "smoke_test", "versions"]:
        assert key in meta, f"missing key: {key}"


def test_smoke_prediction_matches_colab(meta, model):
    row = pd.DataFrame([meta["smoke_test"]["row"]])[meta["features"]]
    prob = float(model.predict_proba(row)[:, 1][0])
    assert prob == pytest.approx(meta["smoke_test"]["expected_probability"], abs=1e-6)


def test_probability_in_range(meta, model):
    row = pd.DataFrame([meta["smoke_test"]["row"]])[meta["features"]]
    assert 0.0 <= float(model.predict_proba(row)[:, 1][0]) <= 1.0


def test_missing_feature_raises_error(meta, model):
    row = pd.DataFrame([meta["smoke_test"]["row"]])[meta["features"]].drop(columns=["Admission grade"])
    with pytest.raises(Exception):
        model.predict_proba(row)


def test_unknown_category_does_not_crash(meta, model):
    row = pd.DataFrame([meta["smoke_test"]["row"]])[meta["features"]]
    row["Course"] = 99999
    prob = float(model.predict_proba(row)[:, 1][0])
    assert 0.0 <= prob <= 1.0


def test_environment_versions_match():
    assert environment_warnings() == [], environment_warnings()