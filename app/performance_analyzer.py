"""
performance_analyzer.py
-----------------------
Academic performance analysis for EduPredict.

Input : a pandas DataFrame with columns
        "Subject", "Marks Obtained", "Max Marks"  (+ optional "Semester")
Output: a dictionary of results + Plotly charts.

Nothing here assumes a fixed passing mark. The pass percentage,
weak-subject threshold, GPA scale and required attendance are all
parameters that the user can set in the app.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REQUIRED_COLUMNS = ["Subject", "Marks Obtained", "Max Marks"]
BAND_ORDER = ["Fail", "Weak", "Average", "Good"]
BAND_COLORS = {
    "Fail": "#D62728",
    "Weak": "#FF7F0E",
    "Average": "#1F77B4",
    "Good": "#2CA02C",
}


# ----------------------------------------------------------------------
# 1. Cleaning and validating the marks table
# ----------------------------------------------------------------------
def prepare_marks_dataframe(df: Optional[pd.DataFrame]):
    """
    Cleans the marks table and adds a Percentage column.

    Rows with missing, non-numeric, negative, or impossible values
    (marks > max marks, max marks <= 0) are REMOVED and reported in
    the warnings list. Nothing is silently corrected.

    Returns: (clean_dataframe, list_of_warning_strings)
    """
    empty = pd.DataFrame(
        columns=REQUIRED_COLUMNS + ["Semester", "Percentage"]
    )
    warnings = []

    if df is None or len(df) == 0:
        return empty, ["No subject marks were provided."]

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in work.columns]
    if missing:
        return empty, [f"Missing required column(s): {', '.join(missing)}."]

    if "Semester" not in work.columns:
        work["Semester"] = None

    work["Subject"] = work["Subject"].astype(str).str.strip()
    work["Marks Obtained"] = pd.to_numeric(work["Marks Obtained"], errors="coerce")
    work["Max Marks"] = pd.to_numeric(work["Max Marks"], errors="coerce")

    bad_subject = work["Subject"].eq("") | work["Subject"].str.lower().isin(
        ["nan", "none"]
    )
    bad_numbers = (
        work["Marks Obtained"].isna()
        | work["Max Marks"].isna()
        | (work["Max Marks"] <= 0)
        | (work["Marks Obtained"] < 0)
        | (work["Marks Obtained"] > work["Max Marks"])
    )
    bad = bad_subject | bad_numbers

    if bad.any():
        rows = [str(i + 1) for i in work.index[bad]]
        warnings.append(
            f"{int(bad.sum())} row(s) were ignored because of missing or invalid "
            f"values (row numbers: {', '.join(rows)}). Please check them."
        )

    clean = work.loc[~bad].copy().reset_index(drop=True)
    if clean.empty:
        return empty, warnings + ["No valid subject rows remain after validation."]

    clean["Percentage"] = (clean["Marks Obtained"] / clean["Max Marks"] * 100).round(2)
    return clean, warnings


# ----------------------------------------------------------------------
# 2. Small helpers
# ----------------------------------------------------------------------
def _semester_sort_key(label):
    """Sorts 'Semester 2' before 'Semester 10'; non-numeric labels go last."""
    match = re.search(r"\d+", str(label))
    if match:
        return (0, int(match.group()), str(label))
    return (1, 0, str(label))


def _semester_label(value):
    """3.0 -> 'Semester 3' (the PDF extractor stores semesters as numbers); text is kept as is."""
    try:
        number = float(value)
        if number == int(number):
            return f"Semester {int(number)}"
    except (TypeError, ValueError):
        pass
    return str(value)


def _band(p, pass_pct, weak_pct, good_pct):
    if p < pass_pct:
        return "Fail"
    if p < weak_pct:
        return "Weak"
    if p < good_pct:
        return "Average"
    return "Good"


def _scale_note(value, scale, name, warnings):
    """Describes a GPA/CGPA value relative to its scale. Returns a dict or None."""
    if value is None:
        return None
    try:
        value = float(value)
        scale = float(scale)
    except (TypeError, ValueError):
        warnings.append(f"{name} value is not a number and was ignored.")
        return None
    if scale <= 0 or value < 0 or value > scale:
        warnings.append(
            f"{name} ({value}) is outside the range 0 to {scale} and was ignored."
        )
        return None

    ratio = value / scale
    if ratio >= 0.75:
        level = "strong"
    elif ratio >= 0.60:
        level = "good"
    elif ratio >= 0.50:
        level = "average"
    else:
        level = "low"
    return {"value": value, "scale": scale, "ratio_percent": round(ratio * 100, 1), "level": level}


# ----------------------------------------------------------------------
# 3. Main analysis
# ----------------------------------------------------------------------
def analyze_performance(
    df: Optional[pd.DataFrame],
    pass_percentage: float = 40.0,
    weak_threshold: float = 50.0,
    good_threshold: float = 75.0,
    gpa: Optional[float] = None,
    cgpa: Optional[float] = None,
    gpa_scale: float = 10.0,
    attendance: Optional[float] = None,
    required_attendance: float = 75.0,
) -> dict:
    """
    Analyzes the marks table. Always returns a dictionary.
    result["ok"] is False if there was no usable data (see result["warnings"]).

    Raises ValueError only if the threshold settings themselves are invalid.
    """
    for name, val in [
        ("pass_percentage", pass_percentage),
        ("weak_threshold", weak_threshold),
        ("good_threshold", good_threshold),
        ("required_attendance", required_attendance),
    ]:
        if not (0 <= float(val) <= 100):
            raise ValueError(f"{name} must be between 0 and 100 (got {val}).")

    # Keep thresholds in a sensible order: pass <= weak <= good
    weak_threshold = max(float(weak_threshold), float(pass_percentage))
    good_threshold = max(float(good_threshold), weak_threshold)
    pass_percentage = float(pass_percentage)

    clean, warnings = prepare_marks_dataframe(df)
    result = {"ok": False, "warnings": warnings, "clean_df": clean}
    if clean.empty:
        return result

    clean["Performance Band"] = clean["Percentage"].apply(
        lambda p: _band(p, pass_percentage, weak_threshold, good_threshold)
    )

    failed = clean[clean["Percentage"] < pass_percentage]
    below = clean[clean["Percentage"] < weak_threshold]

    best = clean.loc[clean["Percentage"].idxmax()]
    worst = clean.loc[clean["Percentage"].idxmin()]

    band_counts = (
        clean["Performance Band"].value_counts().reindex(BAND_ORDER, fill_value=0)
    )

    # ---- Semester-wise trend (only if semester labels exist) ----
    semester_summary = None
    trend = "not available"
    trend_change = None
    sem_df = clean[clean["Semester"].notna() & (clean["Semester"].astype(str).str.strip() != "")]
    if not sem_df.empty:
        grouped = (
            sem_df.groupby("Semester")
            .agg(
                Subjects=("Subject", "count"),
                Average_Percentage=("Percentage", "mean"),
                Failed=("Percentage", lambda s: int((s < pass_percentage).sum())),
            )
            .reset_index()
        )
        grouped["Average_Percentage"] = grouped["Average_Percentage"].round(2)
        grouped = grouped.sort_values(
            "Semester", key=lambda col: col.map(_semester_sort_key)
        ).reset_index(drop=True)
        semester_summary = grouped

        if len(grouped) >= 2:
            trend_change = round(
                float(grouped["Average_Percentage"].iloc[-1]
                      - grouped["Average_Percentage"].iloc[-2]), 2
            )
            if trend_change >= 2:
                trend = "improving"
            elif trend_change <= -2:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "only one semester available"

    # ---- GPA / CGPA / attendance ----
    gpa_info = _scale_note(gpa, gpa_scale, "GPA", warnings)
    cgpa_info = _scale_note(cgpa, gpa_scale, "CGPA", warnings)

    attendance_info = None
    if attendance is not None:
        try:
            att = float(attendance)
            if 0 <= att <= 100:
                attendance_info = {
                    "value": att,
                    "required": float(required_attendance),
                    "below_required": att < float(required_attendance),
                }
            else:
                warnings.append("Attendance must be between 0 and 100; it was ignored.")
        except (TypeError, ValueError):
            warnings.append("Attendance is not a number and was ignored.")

    # ---- Plain-language insights ----
    insights = [
        f"Average performance across {len(clean)} subject(s) is "
        f"{clean['Percentage'].mean():.1f}%.",
        f"Strongest subject: {best['Subject']} ({best['Percentage']:.1f}%). "
        f"Weakest subject: {worst['Subject']} ({worst['Percentage']:.1f}%).",
    ]
    if len(failed):
        insights.append(
            f"{len(failed)} subject(s) are below the pass mark of {pass_percentage:g}%: "
            f"{', '.join(failed['Subject'])}."
        )
    else:
        insights.append(f"No subject is below the pass mark of {pass_percentage:g}%.")
    if len(below) > len(failed):
        extra = below[~below.index.isin(failed.index)]
        insights.append(
            f"{len(extra)} more subject(s) passed but scored below {weak_threshold:g}%: "
            f"{', '.join(extra['Subject'])}."
        )
    if trend in ("improving", "declining", "stable"):
        insights.append(
            f"Semester trend is {trend} "
            f"({trend_change:+.1f} percentage points versus the previous semester)."
        )
    if cgpa_info:
        insights.append(
            f"CGPA {cgpa_info['value']:g}/{cgpa_info['scale']:g} is {cgpa_info['level']} "
            f"({cgpa_info['ratio_percent']}% of the scale)."
        )
    if gpa_info:
        insights.append(
            f"GPA {gpa_info['value']:g}/{gpa_info['scale']:g} is {gpa_info['level']} "
            f"({gpa_info['ratio_percent']}% of the scale)."
        )
    if attendance_info:
        state = "is below" if attendance_info["below_required"] else "meets"
        insights.append(
            f"Attendance is {attendance_info['value']:g}%, which {state} the required "
            f"{attendance_info['required']:g}%."
        )

    result.update(
        {
            "ok": True,
            "clean_df": clean,
            "num_subjects": int(len(clean)),
            "average_percentage": round(float(clean["Percentage"].mean()), 2),
            "average_marks": round(float(clean["Marks Obtained"].mean()), 2),
            "highest_marks": float(clean["Marks Obtained"].max()),
            "lowest_marks": float(clean["Marks Obtained"].min()),
            "strongest_subject": {"subject": best["Subject"], "percentage": float(best["Percentage"])},
            "weakest_subject": {"subject": worst["Subject"], "percentage": float(worst["Percentage"])},
            "failed_subjects": failed["Subject"].tolist(),
            "num_failed": int(len(failed)),
            "weak_subjects": below["Subject"].tolist(),
            "num_below_threshold": int(len(below)),
            "band_counts": band_counts.to_dict(),
            "semester_summary": semester_summary,
            "trend": trend,
            "trend_change": trend_change,
            "gpa": gpa_info,
            "cgpa": cgpa_info,
            "attendance": attendance_info,
            "pass_percentage": pass_percentage,
            "weak_threshold": weak_threshold,
            "good_threshold": good_threshold,
            "insights": insights,
        }
    )
    return result


# ----------------------------------------------------------------------
# 4. Charts (Plotly). Each returns a Figure, or None if there is no data.
# ----------------------------------------------------------------------
def _style(fig, title):
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        legend_title_text="",
    )
    return fig


def subject_marks_chart(result: dict):
    """Bar chart of percentage per subject, colored by performance band."""
    if not result.get("ok"):
        return None
    df = result["clean_df"]
    fig = go.Figure()
    for band in BAND_ORDER:
        part = df[df["Performance Band"] == band]
        if part.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=part["Subject"],
                y=part["Percentage"],
                name=band,
                marker_color=BAND_COLORS[band],
                customdata=np.stack([part["Marks Obtained"], part["Max Marks"]], axis=-1),
                hovertemplate="%{x}<br>%{y:.1f}% (%{customdata[0]}/%{customdata[1]})<extra></extra>",
            )
        )
    fig.add_hline(
        y=result["pass_percentage"], line_dash="dash", line_color="#D62728",
        annotation_text=f"Pass mark {result['pass_percentage']:g}%",
    )
    fig.update_yaxes(range=[0, 100], title="Percentage")
    fig.update_xaxes(title="Subject")
    return _style(fig, "Subject-wise Performance")


def performance_distribution_chart(result: dict):
    """Bar chart: how many subjects fall in each performance band."""
    if not result.get("ok"):
        return None
    counts = result["band_counts"]
    fig = go.Figure(
        go.Bar(
            x=BAND_ORDER,
            y=[counts[b] for b in BAND_ORDER],
            marker_color=[BAND_COLORS[b] for b in BAND_ORDER],
            text=[counts[b] for b in BAND_ORDER],
            textposition="outside",
        )
    )
    fig.update_yaxes(title="Number of subjects", dtick=1)
    fig.update_xaxes(title="Performance band")
    return _style(fig, "Performance Distribution")


def weak_subjects_chart(result: dict):
    """Horizontal bars for subjects below the weak threshold. None if none."""
    if not result.get("ok"):
        return None
    df = result["clean_df"]
    weak = df[df["Percentage"] < result["weak_threshold"]].sort_values("Percentage")
    if weak.empty:
        return None
    fig = go.Figure(
        go.Bar(
            x=weak["Percentage"],
            y=weak["Subject"],
            orientation="h",
            marker_color=[BAND_COLORS[b] for b in weak["Performance Band"]],
            text=weak["Percentage"].round(1).astype(str) + "%",
            textposition="outside",
        )
    )
    fig.add_vline(
        x=result["pass_percentage"], line_dash="dash", line_color="#D62728",
        annotation_text="Pass mark",
    )
    fig.update_xaxes(range=[0, 100], title="Percentage")
    return _style(fig, f"Weak Subjects (below {result['weak_threshold']:g}%)")


def semester_trend_chart(result: dict):
    """Line chart of average percentage per semester. None if < 2 semesters."""
    if not result.get("ok"):
        return None
    summary = result.get("semester_summary")
    if summary is None or len(summary) < 2:
        return None
    fig = go.Figure(
        go.Scatter(
            x=summary["Semester"].map(_semester_label),
            y=summary["Average_Percentage"],
            mode="lines+markers+text",
            text=summary["Average_Percentage"].round(1).astype(str) + "%",
            textposition="top center",
            line=dict(color="#1F77B4", width=3),
        )
    )
    fig.update_yaxes(range=[0, 100], title="Average percentage")
    fig.update_xaxes(title="Semester")
    return _style(fig, "Semester Performance Trend")


# ----------------------------------------------------------------------
# 5. Quick self-test:  python app/performance_analyzer.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "Subject": ["Maths", "Physics", "Chemistry", "English", "Programming", "Electronics"],
            "Marks Obtained": [82, 35, 58, 71, 90, 44],
            "Max Marks": [100, 100, 100, 100, 100, 100],
            "Semester": ["Semester 1", "Semester 1", "Semester 1", "Semester 2", "Semester 2", "Semester 2"],
        }
    )
    res = analyze_performance(
        sample, pass_percentage=40, weak_threshold=50,
        cgpa=7.2, gpa_scale=10, attendance=68, required_attendance=75,
    )
    print("OK:", res["ok"])
    for line in res["insights"]:
        print("-", line)
    print("Failed:", res["failed_subjects"])
    print("Charts built:",
          [subject_marks_chart(res) is not None,
           performance_distribution_chart(res) is not None,
           weak_subjects_chart(res) is not None,
           semester_trend_chart(res) is not None])

    # Edge cases: these must not crash
    print("Empty input ok?:", analyze_performance(pd.DataFrame())["ok"])
    bad = pd.DataFrame({"Subject": ["X", ""], "Marks Obtained": ["abc", 50], "Max Marks": [100, 100]})
    print("Invalid rows ok?:", analyze_performance(bad)["ok"])