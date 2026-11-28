"""
Tests for Phase 9 (performance_analyzer) and Phase 10 (predictor).
Run from the EduPredict folder:   python tests/test_phase9_10.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd

import performance_analyzer as pa
import predictor as pr

passed, failed = 0, 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}  {detail}")


# ---------------------------------------------------------------- Phase 9
# A table shaped exactly like the PDF extractor's output (Semester is a float,
# extra columns, one row with no Max Marks).
table = pd.DataFrame({
    "Semester": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    "Code": ["M1", "P1", "C1", "E2", "PR2", "EL2"],
    "Subject": ["Maths", "Physics", "Chemistry", "English", "Programming", "Electronics"],
    "Marks Obtained": [82, 35, 58, 71, 90, 44],
    "Max Marks": [100, 100, 100, 100, 100, None],
    "Grade": [None] * 6, "Result": [None] * 6, "Check": [""] * 6,
})
res = pa.analyze_performance(table, pass_percentage=40, weak_threshold=50,
                             cgpa=7.2, attendance=68, required_attendance=75)
check("analyzer: runs on extractor-shaped table", res["ok"])
check("analyzer: row without Max Marks is reported, not guessed",
      any("ignored" in w for w in res["warnings"]) and res["num_subjects"] == 5)
check("analyzer: failed subject found", res["failed_subjects"] == ["Physics"])
check("analyzer: trend improving", res["trend"] == "improving", str(res["trend_change"]))
check("analyzer: attendance below required", res["attendance"]["below_required"] is True)
check("analyzer: semester label is readable", pa._semester_label(2.0) == "Semester 2")
check("analyzer: all four charts build",
      all(f(res) is not None for f in (pa.subject_marks_chart, pa.performance_distribution_chart,
                                       pa.weak_subjects_chart, pa.semester_trend_chart)))
check("analyzer: empty table does not crash", pa.analyze_performance(pd.DataFrame())["ok"] is False)
try:
    pa.analyze_performance(table, pass_percentage=150)
    check("analyzer: invalid threshold rejected", False)
except ValueError:
    check("analyzer: invalid threshold rejected", True)

# ---------------------------------------------------------------- Phase 10
pipe = pr.load_model()
meta = pr.load_metadata()
row = meta["smoke_test"]["row"]
expected = meta["smoke_test"]["expected_probability"]

out = pr.predict_student(pipe, row, meta)
check("predictor: smoke test matches Phase 7 probability",
      out["ok"] and abs(out["dropout_probability"] - expected) < 1e-6)
check("predictor: low-risk student is not flagged", out["flagged"] is False and out["risk_level"] == "Low")
check("predictor: empty input refused", pr.predict_student(pipe, {}, meta)["ok"] is False)

bad = dict(row, Course=1234)
r = pr.predict_student(pipe, bad, meta)
check("predictor: unknown course code refused", (not r["ok"]) and "Course" in r["invalid_features"])
check("predictor: yes/no value of 2 refused", not pr.predict_student(pipe, dict(row, Gender=2), meta)["ok"])
check("predictor: text number accepted",
      pr.predict_student(pipe, {k: str(v) for k, v in row.items()}, meta)["ok"])
missing_one = {k: v for k, v in row.items() if k != "Admission grade"}
r = pr.predict_student(pipe, missing_one, meta)
check("predictor: one missing feature refused", (not r["ok"]) and r["missing_features"] == ["Admission grade"])
r = pr.predict_student(pipe, dict(row, **{"Age at enrollment": 70}), meta)
check("predictor: out-of-range age gives warning", r["ok"] and len(r["warnings"]) == 1)

risky = dict(row)
risky.update({"Curricular units 1st sem (approved)": 0, "Curricular units 1st sem (grade)": 0,
              "Debtor": 1, "Tuition fees up to date": 0, "Scholarship holder": 0,
              "Age at enrollment": 28, "Curricular units 1st sem (evaluations)": 4})
r = pr.predict_student(pipe, risky, meta)
check("predictor: high-risk student is flagged High", r["ok"] and r["flagged"] and r["risk_level"] == "High")
check("predictor: probability is the Dropout (class 1) column",
      abs(r["dropout_probability"] - pipe.predict_proba(pd.DataFrame([risky]))[0][1]) < 1e-9)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)