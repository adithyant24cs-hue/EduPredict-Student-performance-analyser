"""
recommendations.py
------------------
Rule-based recommendation engine for EduPredict (Phase 11).

Inputs (all optional, so it works with whatever data is available):
  analysis       - the dictionary from performance_analyzer.analyze_performance()
  prediction     - the dictionary from predictor.predict_student()
  student_inputs - the dictionary of the 20 model features entered on the form
                   (used only for financial and unit-pass-rate rules)

Every recommendation carries a plain-language "reason" that quotes the exact value that
triggered it, so a teacher or student can see WHY it appears. These are suggestions based on
simple rules, not a diagnosis, and no rule guesses a value that was not provided.
"""

from typing import Optional

import pandas as pd

PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}

# Rule settings (change here if your institution works differently)
LOW_ATTENDANCE_SEVERE_GAP = 10     # points below the required attendance = High priority
STEEP_DECLINE_POINTS = 10          # semester average fell by this much or more = High priority
LOW_UNIT_PASS_RATE = 0.5           # passed fewer than half of enrolled 1st-semester units


def _num(value) -> Optional[float]:
    try:
        number = float(value)
        return None if pd.isna(number) else number
    except (TypeError, ValueError):
        return None


def _names(subjects, limit=6) -> str:
    subjects = [str(s) for s in subjects]
    text = ", ".join(subjects[:limit])
    return text + (f" and {len(subjects) - limit} more" if len(subjects) > limit else "")


def _add(items, priority, category, title, action, reason):
    items.append({"priority": priority, "category": category, "title": title,
                  "action": action, "reason": reason})


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def generate_recommendations(analysis: Optional[dict] = None,
                             prediction: Optional[dict] = None,
                             student_inputs: Optional[dict] = None) -> dict:
    """
    Returns {"items": [recommendation dicts, most important first],
             "summary": str, "notes": [str]}
    Each item: priority (High/Medium/Low), category, title, action, reason.
    """
    items, notes = [], []
    has_analysis = bool(analysis) and analysis.get("ok")
    has_prediction = bool(prediction) and prediction.get("ok")
    student_inputs = student_inputs or {}

    if not has_analysis and not has_prediction and not student_inputs:
        return {"items": [], "notes": ["No analysis, prediction or student details were provided."],
                "summary": "There is not enough information to give recommendations yet."}
    if not has_analysis:
        notes.append("No subject marks were analysed, so no subject-level advice is given.")
    if not has_prediction:
        notes.append("No dropout prediction is available, so these recommendations rely on the "
                     "academic data and details provided only.")

    # ---------------- Academic performance rules ----------------
    if has_analysis:
        pass_mark = analysis["pass_percentage"]
        weak_mark = analysis["weak_threshold"]
        failed = analysis["failed_subjects"]
        weak_only = [s for s in analysis["weak_subjects"] if s not in failed]

        if failed:
            _add(items, "High", "Failed subjects",
                 "Prepare to clear the failed subjects",
                 "Attend remedial or doubt-clearing classes and ask the department about the "
                 "re-examination or supplementary exam process for: " + _names(failed) + ".",
                 f"{len(failed)} subject(s) are below the pass mark of {pass_mark:g}%.")
        if weak_only:
            _add(items, "Medium", "Weak subjects",
                 "Focus extra study time on weaker subjects",
                 "Create a weekly study plan that gives more hours to: " + _names(weak_only) +
                 ". Revise the basics first and practise past questions.",
                 f"{len(weak_only)} subject(s) passed but scored below {weak_mark:g}%.")
        if analysis["average_percentage"] < weak_mark:
            _add(items, "High", "Overall performance",
                 "Meet an academic mentor",
                 "Book a meeting with a mentor or class advisor to review study methods and "
                 "set a realistic improvement target for next semester.",
                 f"The average across all subjects is {analysis['average_percentage']:.1f}%, "
                 f"below {weak_mark:g}%.")

        attendance = analysis.get("attendance")
        if attendance and attendance["below_required"]:
            gap = attendance["required"] - attendance["value"]
            level = "High" if gap >= LOW_ATTENDANCE_SEVERE_GAP else "Medium"
            _add(items, level, "Attendance", "Improve attendance",
                 "Aim to attend every class from now on, and speak to the class advisor early if "
                 "something (health, travel, family) is making this difficult. Low attendance "
                 "can lead to being barred from exams.",
                 f"Attendance is {attendance['value']:g}%, which is {gap:g} points below the "
                 f"required {attendance['required']:g}%.")

        if analysis.get("trend") == "declining" and analysis.get("trend_change") is not None:
            drop = abs(analysis["trend_change"])
            level = "High" if drop >= STEEP_DECLINE_POINTS else "Medium"
            _add(items, level, "Semester trend", "Find out why results are falling",
                 "Look at what changed since last semester (workload, attendance, health, "
                 "stress, part-time work) and discuss it with a mentor or counsellor.",
                 f"The semester average fell by {drop:.1f} percentage points compared with the "
                 "previous semester.")

        strongest = analysis["strongest_subject"]
        if strongest["percentage"] >= analysis["good_threshold"]:
            _add(items, "Low", "Strength", "Build on a strength",
                 f"Use the strong performance in {strongest['subject']} for peer tutoring, "
                 "projects or advanced electives. Explaining a topic to others also deepens "
                 "understanding.",
                 f"{strongest['subject']} scored {strongest['percentage']:.1f}%, above the "
                 f"{analysis['good_threshold']:g}% 'good' mark.")

    # ---------------- Dropout-risk rules (only if a valid prediction exists) ----------------
    if has_prediction:
        level = prediction["risk_level"]
        probability = prediction["dropout_probability"] * 100
        drivers = prediction.get("drivers_up") or []
        driver_text = (" Main numeric factors raising the score: " + _names(drivers) + "."
                       if drivers else "")
        if level == "High":
            _add(items, "High", "Dropout risk", "Arrange a support conversation soon",
                 "A teacher, mentor or counsellor should meet the student early this semester, "
                 "listen to the student's situation, and agree on concrete support (academic, "
                 "financial or personal).",
                 f"The model's dropout score is {probability:.0f}%, in the High range "
                 f"(at or above {prediction['alert_threshold'] * 100:.0f}%)." + driver_text +
                 " This is an estimate, not a certain outcome.")
        elif level == "Medium":
            _add(items, "Medium", "Dropout risk", "Keep a closer watch this semester",
                 "Check in with the student a few times during the semester and act early if "
                 "marks or attendance slip.",
                 f"The model's dropout score is {probability:.0f}%, in the Medium range." +
                 driver_text)

    # ---------------- Financial rules (from the form, not the model) ----------------
    debtor = _num(student_inputs.get("Debtor"))
    tuition_ok = _num(student_inputs.get("Tuition fees up to date"))
    scholarship = _num(student_inputs.get("Scholarship holder"))
    financial_flags = []
    if debtor == 1:
        financial_flags.append("the student has outstanding dues (Debtor = yes)")
    if tuition_ok == 0:
        financial_flags.append("tuition fees are not up to date")
    if financial_flags:
        extra = (" Also ask whether a scholarship or fee-waiver scheme could apply."
                 if scholarship == 0 else "")
        _add(items, "High", "Financial support", "Seek financial support early",
             "Contact the accounts or student-welfare office about instalments, fee concessions "
             "or institutional aid programmes before the situation affects studies." + extra,
             "Because " + " and ".join(financial_flags) + ".")

    # ---------------- First-semester unit pass rate (from the form) ----------------
    enrolled = _num(student_inputs.get("Curricular units 1st sem (enrolled)"))
    approved = _num(student_inputs.get("Curricular units 1st sem (approved)"))
    if enrolled and enrolled > 0 and approved is not None and approved / enrolled < LOW_UNIT_PASS_RATE:
        _add(items, "High", "Course progress", "Catch up on unpassed course units",
             "List the units not yet passed, plan when to retake them, and ask the department "
             "which support classes are available.",
             f"Only {approved:g} of {enrolled:g} enrolled first-semester units were passed "
             f"({approved / enrolled * 100:.0f}%).")

    # ---------------- Always-useful closing advice ----------------
    if items and any(i["priority"] != "Low" for i in items):
        _add(items, "Low", "Follow-up", "Monitor progress next semester",
             "Re-run this analysis after the next set of results and compare the numbers.",
             "Tracking results over time shows whether the actions above are working.")
    elif has_analysis or has_prediction:
        _add(items, "Low", "Follow-up", "Keep up the good work",
             "Keep the current study routine and attendance, and re-check after the next results.",
             "No warning signs were found in the data provided.")

    items.sort(key=lambda i: PRIORITY_ORDER[i["priority"]])
    high = sum(1 for i in items if i["priority"] == "High")
    medium = sum(1 for i in items if i["priority"] == "Medium")
    summary = (f"{len(items)} recommendation(s): {high} high priority, {medium} medium priority."
               if items else "No recommendations.")
    return {"items": items, "summary": summary, "notes": notes}


# ----------------------------------------------------------------------
# Weekly study plan (explainable: hours follow the size of the gap)
# ----------------------------------------------------------------------
def weekly_study_plan(analysis: Optional[dict], hours_per_week: float = 15.0) -> pd.DataFrame:
    """
    Splits the study hours the student can spare across the weaker subjects. A subject further
    below the 'weak' mark gets more hours. Returns an empty table if there are no weak subjects.
    """
    columns = ["Subject", "Current %", "Suggested hours/week"]
    if not analysis or not analysis.get("ok") or hours_per_week <= 0:
        return pd.DataFrame(columns=columns)
    data = analysis["clean_df"]
    weak = data[data["Percentage"] < analysis["weak_threshold"]].copy()
    if weak.empty:
        return pd.DataFrame(columns=columns)
    weights = (analysis["weak_threshold"] - weak["Percentage"]) + 5      # +5 so no subject gets 0
    weak["Suggested hours/week"] = (weights / weights.sum() * hours_per_week).round(1)
    plan = weak[["Subject", "Percentage", "Suggested hours/week"]].rename(
        columns={"Percentage": "Current %"})
    return plan.sort_values("Current %").reset_index(drop=True)


def recommendations_to_text(result: dict) -> str:
    """Plain-text version, used by the PDF report (Phase 12) and for quick printing."""
    lines = [result["summary"], ""]
    for number, item in enumerate(result["items"], start=1):
        lines.append(f"{number}. [{item['priority']}] {item['title']}")
        lines.append(f"   What to do: {item['action']}")
        lines.append(f"   Why: {item['reason']}")
        lines.append("")
    for note in result["notes"]:
        lines.append("Note: " + note)
    return "\n".join(lines).strip()


# ----------------------------------------------------------------------
# Self-test:   python app/recommendations.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import predictor
    from performance_analyzer import analyze_performance

    marks = pd.DataFrame({
        "Subject": ["Maths", "Physics", "Chemistry", "English", "Programming", "Electronics"],
        "Marks Obtained": [82, 35, 58, 71, 90, 44],
        "Max Marks": [100] * 6,
        "Semester": [1, 1, 1, 2, 2, 2],
    })
    analysis = analyze_performance(marks, attendance=68, required_attendance=75)

    pipe, meta = predictor.load_model(), predictor.load_metadata()
    form = dict(meta["smoke_test"]["row"])
    form.update({"Debtor": 1, "Tuition fees up to date": 0, "Scholarship holder": 0,
                 "Curricular units 1st sem (approved)": 2})
    prediction = predictor.predict_student(pipe, form, meta)

    result = generate_recommendations(analysis, prediction, form)
    print(recommendations_to_text(result))
    print()
    print(weekly_study_plan(analysis, hours_per_week=15).to_string(index=False))