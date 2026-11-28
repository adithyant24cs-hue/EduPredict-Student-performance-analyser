"""
Tests for Phase 11 (recommendations).
Run from the EduPredict folder:   python tests/test_phase11.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd

import predictor as pr
import recommendations as rec
from performance_analyzer import analyze_performance

passed, failed = 0, 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}  {detail}")


def titles(result):
    return [i["title"] for i in result["items"]]


pipe, meta = pr.load_model(), pr.load_metadata()
smoke = dict(meta["smoke_test"]["row"])

# --- nothing provided
r = rec.generate_recommendations()
check("nothing provided: no items, clear message", r["items"] == [] and r["notes"])

# --- a good student
good = pd.DataFrame({"Subject": ["A", "B", "C"], "Marks Obtained": [85, 80, 90],
                     "Max Marks": [100] * 3, "Semester": [1, 1, 1]})
a_good = analyze_performance(good, attendance=90)
p_good = pr.predict_student(pipe, smoke, meta)
r = rec.generate_recommendations(a_good, p_good, smoke)
check("good student: no High items", not any(i["priority"] == "High" for i in r["items"]))
check("good student: positive advice given", "Keep up the good work" in titles(r))

# --- a struggling student with falling results
bad = pd.DataFrame({"Subject": ["A", "B", "C", "D"], "Marks Obtained": [80, 75, 30, 35],
                    "Max Marks": [100] * 4, "Semester": [1, 1, 2, 2]})
a_bad = analyze_performance(bad, attendance=55, required_attendance=75)
risky = dict(smoke)
risky.update({"Curricular units 1st sem (approved)": 0, "Curricular units 1st sem (grade)": 0,
              "Debtor": 1, "Tuition fees up to date": 0, "Scholarship holder": 0,
              "Age at enrollment": 28, "Curricular units 1st sem (evaluations)": 4})
p_bad = pr.predict_student(pipe, risky, meta)
r = rec.generate_recommendations(a_bad, p_bad, risky)
t = titles(r)
check("struggling: failed subjects rule", "Prepare to clear the failed subjects" in t)
check("struggling: attendance rule is High (20 points below)",
      any(i["title"] == "Improve attendance" and i["priority"] == "High" for i in r["items"]))
check("struggling: falling results rule", "Find out why results are falling" in t)
check("struggling: dropout-risk rule", "Arrange a support conversation soon" in t)
check("struggling: financial rule", "Seek financial support early" in t)
check("struggling: unit pass-rate rule", "Catch up on unpassed course units" in t)
check("items sorted High first",
      [rec.PRIORITY_ORDER[i["priority"]] for i in r["items"]] ==
      sorted(rec.PRIORITY_ORDER[i["priority"]] for i in r["items"]))
check("every item has a reason", all(i["reason"] for i in r["items"]))

# --- missing information is never guessed
r = rec.generate_recommendations(a_bad, None, {})
t = titles(r)
check("no prediction: note explains it", any("No dropout prediction" in n for n in r["notes"]))
check("no prediction: no dropout or financial advice",
      "Arrange a support conversation soon" not in t and "Seek financial support early" not in t)

failed_prediction = pr.predict_student(pipe, {}, meta)
r = rec.generate_recommendations(None, failed_prediction, None)
check("refused prediction is not treated as a risk",
      all(i["category"] != "Dropout risk" for i in r["items"]))

# --- study plan and text
plan = rec.weekly_study_plan(a_bad, hours_per_week=12)
check("study plan: hours add up to about 12", abs(plan["Suggested hours/week"].sum() - 12) < 0.3)
check("study plan: empty for a strong student", rec.weekly_study_plan(a_good).empty)
check("text version works", "[High]" in rec.recommendations_to_text(rec.generate_recommendations(a_bad, p_bad, risky)))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)