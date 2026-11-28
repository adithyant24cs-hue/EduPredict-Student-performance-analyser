"""
Tests for Phase 12 (PDF report generator).
Run from the EduPredict folder:   python tests/test_phase12.py
"""
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import pdfplumber

import predictor as pr
import recommendations as rec
import report_generator as rg
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


def pdf_text(pdf_bytes):
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages), len(pdf.pages)


pipe, meta = pr.load_model(), pr.load_metadata()
form = dict(meta["smoke_test"]["row"])
form.update({"Debtor": 1, "Tuition fees up to date": 0, "Curricular units 1st sem (approved)": 2})
prediction = pr.predict_student(pipe, form, meta)

marks = pd.DataFrame({
    "Subject": ["Maths", "Physics", "Chemistry", "English"],
    "Marks Obtained": [82, 35, 58, 71], "Max Marks": [100] * 4, "Semester": [1, 1, 2, 2]})
analysis = analyze_performance(marks, attendance=68)
recs = rec.generate_recommendations(analysis, prediction, form)
info = {"student_name": "José Müller", "student_id": "TEST001", "course": "B.Tech", "attendance": 68}

# --- full report
pdf = rg.generate_report(info, analysis, prediction, recs, rec.weekly_study_plan(analysis))
text, pages = pdf_text(pdf)
check("full report: valid PDF", pdf[:5] == b"%PDF-")
check("full report: has student name with accents", "José Müller" in text)
check("full report: has subject, risk level and a recommendation",
      "Physics" in text and "High" in text and "Prepare to clear the failed subjects" in text)
check("full report: has disclaimer and date", "not a certain outcome" in text and "Date of analysis" in text)
check("full report: no leftover 'nan' or 'None' text",
      re.search(r"\bnan\b", text, re.IGNORECASE) is None and "None" not in text)

# --- missing data is stated, never invented
text, _ = pdf_text(rg.generate_report())
check("empty report: still a valid PDF", len(text) > 0)
check("empty report: says data is missing",
      "Not provided" in text and "No subject marks" in text and "No dropout prediction was run" in text)

refused = pr.predict_student(pipe, {}, meta)
text, _ = pdf_text(rg.generate_report(info, analysis, refused, rec.generate_recommendations(analysis, refused, {})))
check("refused prediction: report says none was made", "No dropout prediction was made" in text)

# --- awkward text must not break the layout
tricky = pd.DataFrame({"Subject": ["R&D <Lab> & Co", "Math (Adv.)"], "Marks Obtained": [60, 70],
                       "Max Marks": [100, 100]})
tricky_analysis = analyze_performance(tricky)
try:
    text, _ = pdf_text(rg.generate_report({"student_name": "A & B <C>"}, tricky_analysis))
    check("special characters (& and <) are safe", "R&D" in text and "A & B" in text)
except Exception as exc:
    check("special characters (& and <) are safe", False, repr(exc))

# --- many subjects across pages
many = pd.DataFrame({"Subject": [f"Subject number {i}" for i in range(40)],
                     "Marks Obtained": [(i * 7) % 100 for i in range(40)], "Max Marks": [100] * 40,
                     "Semester": [1 + i // 10 for i in range(40)]})
many_analysis = analyze_performance(many)
pdf = rg.generate_report(info, many_analysis, None, rec.generate_recommendations(many_analysis))
text, pages = pdf_text(pdf)
check("40 subjects: report builds over several pages", pages >= 3 and "Subject number 39" in text, f"pages={pages}")

# --- file name is safe
name = rg.report_filename({"student_name": "A/B: C*?"})
check("file name has no unsafe characters", all(c not in name for c in '/\\:*?"<>|') and name.endswith(".pdf"), name)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)