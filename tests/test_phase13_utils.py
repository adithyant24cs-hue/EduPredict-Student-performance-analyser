"""
Tests for utils.py (Phase 13). They do not need Streamlit.
Run from the EduPredict folder:   python tests/test_phase13_utils.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd

import predictor as pr
import utils

passed, failed = 0, 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}  {detail}")


pipe, meta = pr.load_model(), pr.load_metadata()
required = pr.get_required_features(pipe)
spec = utils.build_form_spec(required, meta)
fields = [f for g in spec for f in g["fields"]]
by_name = {f["name"]: f for f in fields}

# --- the form is built from the saved model
check("form covers exactly the model's 20 features, once each",
      sorted(by_name) == sorted(required) and len(fields) == len(required) == 20)
check("dropdown options equal the model's allowed codes",
      all(f["options"] == meta["allowed_values"][f["name"]] for f in fields if f["kind"] == "select"))
check("number limits equal the training ranges",
      all([f["min"], f["max"]] == list(meta["numeric_ranges"][f["name"]]) for f in fields if f["kind"] == "number"))
check("whole-number fields use integers",
      isinstance(by_name["Age at enrollment"]["min"], int) and by_name["Age at enrollment"]["step"] == 1
      and isinstance(by_name["Admission grade"]["min"], float))
check("a feature unknown to utils still gets a field",
      "Brand new feature" in [f["name"] for g in utils.build_form_spec(required + ["Brand new feature"], meta) for f in g["fields"]])

# --- every code in the model has a readable label
missing = [(n, c) for n, vals in meta["allowed_values"].items() for c in vals
           if n in utils.CODE_LABELS and c not in utils.CODE_LABELS[n] and n != "Application order"]
check("every allowed code has a label", not missing, str(missing))
check("labels look right", utils.code_label("Debtor", 1) == "Yes" and utils.code_label("Gender", 0) == "Female"
      and utils.code_label("Course", 9670) == "9670 - Advertising and Marketing Management"
      and utils.code_label("Course", 123456) == "123456")

# --- values chosen through the form reach the predictor correctly
row = meta["smoke_test"]["row"]
out = pr.predict_student(pipe, {f["name"]: row[f["name"]] for f in fields}, meta)
check("form values -> prediction matches Phase 7", out["ok"]
      and abs(out["dropout_probability"] - meta["smoke_test"]["expected_probability"]) < 1e-6)

# --- student details
info, errors = utils.parse_info_fields({"student_name": "  Asha  ", "student_id": "", "sgpa": "8,4",
                                        "attendance": "72.5", "semester": "3", "backlogs": ""})
check("details: trims text, blanks become None, comma decimals accepted",
      info["student_name"] == "Asha" and info["student_id"] is None and info["sgpa"] == 8.4
      and info["backlogs"] is None and info["semester"] == 3 and not errors)
_, errors = utils.parse_info_fields({"sgpa": "abc", "semester": "3.5"})
check("details: bad numbers give messages", len(errors) == 2, str(errors))
text = utils.info_to_text({"student_name": "Asha", "semester": 3.0, "cgpa": 7.25, "attendance": None})
check("details: numbers shown cleanly in text boxes",
      text["semester"] == "3" and text["cgpa"] == "7.25" and text["attendance"] == "")

# --- marks table
frame = pd.DataFrame({"Subject": ["Maths", "", None, "Physics"], "Marks Obtained": [80, None, None, None],
                      "Max Marks": [100, None, None, 100]})
cleaned = utils.drop_blank_rows(frame)
check("drop_blank_rows removes only fully empty rows", list(cleaned["Subject"]) == ["Maths", "Physics"])

# --- Model Performance page data
summary = utils.metadata_summary(meta)
check("summary numbers come from the metadata",
      summary["roc_auc"] == meta["test_metrics"]["roc_auc"]
      and summary["recall"] == meta["test_metrics"]["threshold_0.40"]["recall"] and summary["n_features"] == 20)
table = utils.comparison_table(meta)
check("comparison table has every model", len(table) == len(meta["model_comparison"]) and "recall" in table.columns)
check("charts build", utils.comparison_chart(table, "recall") is not None
      and utils.top_features_chart(meta["random_forest_top_features"], "t") is not None
      and utils.comparison_chart(table, "not_a_metric") is None and utils.top_features_chart({}, "t") is None)
check("empty metadata does not crash", utils.metadata_summary({})["roc_auc"] is None and utils.comparison_table({}).empty)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)