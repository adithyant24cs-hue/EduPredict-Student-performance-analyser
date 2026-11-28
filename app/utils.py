"""
utils.py
--------
Helper code for the EduPredict Streamlit app (Phase 13). Nothing in this file needs Streamlit,
so it can be tested on its own (tests/test_phase13_utils.py).

  * labels and help text for the model's coded features (the dropdowns on the prediction page)
  * a "form specification" built from the saved model, so the form always matches the model
  * parsing/cleaning of the student details and marks the user edits
  * tables and charts for the Model Performance page, built only from model_metadata.json
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# 1. Meaning of the dataset's codes (UCI "Predict Students' Dropout and Academic Success")
#    Labels are for display only. The allowed codes themselves come from model_metadata.json.
#    Please compare with the dataset page if you quote a label in your report.
# ----------------------------------------------------------------------
CODE_LABELS = {
    "Marital status": {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced",
                       5: "Facto union", 6: "Legally separated"},
    "Application mode": {
        1: "1st phase - general contingent", 2: "Ordinance No. 612/93",
        5: "1st phase - special contingent (Azores)", 7: "Holders of other higher courses",
        10: "Ordinance No. 854-B/99", 15: "International student (bachelor)",
        16: "1st phase - special contingent (Madeira)", 17: "2nd phase - general contingent",
        18: "3rd phase - general contingent",
        26: "Ordinance No. 533-A/99, item b2 (different plan)",
        27: "Ordinance No. 533-A/99, item b3 (other institution)",
        39: "Over 23 years old", 42: "Transfer", 43: "Change of course",
        44: "Technological specialization diploma holders",
        51: "Change of institution/course", 53: "Short cycle diploma holders",
        57: "Change of institution/course (international)"},
    "Application order": {0: "First choice", 9: "Last choice"},
    "Course": {
        33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
        8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
        9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
        9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
        9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication", 9853: "Basic Education",
        9991: "Management (evening)"},
    "Previous qualification": {
        1: "Secondary education", 2: "Higher education - bachelor's degree",
        3: "Higher education - degree", 4: "Higher education - master's",
        5: "Higher education - doctorate", 6: "Frequency of higher education",
        9: "12th year - not completed", 10: "11th year - not completed",
        12: "Other - 11th year", 15: "10th year - not completed",
        19: "Basic education 3rd cycle", 38: "Basic education 2nd cycle",
        39: "Technological specialization course", 40: "Higher education - degree (1st cycle)",
        42: "Professional higher technical course", 43: "Higher education - master (2nd cycle)"},
    "Daytime/evening attendance": {1: "Daytime", 0: "Evening"},
    "Displaced": {1: "Yes", 0: "No"},
    "Debtor": {1: "Yes", 0: "No"},
    "Tuition fees up to date": {1: "Yes", 0: "No"},
    "Gender": {1: "Male", 0: "Female"},
    "Scholarship holder": {1: "Yes", 0: "No"},
}
YES_NO_FEATURES = {"Displaced", "Debtor", "Tuition fees up to date", "Scholarship holder",
                   "Daytime/evening attendance", "Gender"}

FEATURE_HELP = {
    "Marital status": "Marital status at enrollment.",
    "Application mode": "How the student applied to the institution.",
    "Application order": "Order of the student's choice for this course (0 = first, 9 = last).",
    "Course": "Course codes are those of the Portuguese institution in the training dataset.",
    "Daytime/evening attendance": "Whether classes are attended in the daytime or evening.",
    "Previous qualification": "Highest qualification before enrollment.",
    "Previous qualification (grade)": "Grade of the previous qualification (training range 95-190).",
    "Admission grade": "Admission grade (training range 95-190).",
    "Displaced": "Is the student living away from the family home?",
    "Debtor": "Does the student have unpaid dues?",
    "Tuition fees up to date": "Are tuition fees paid up to date?",
    "Gender": "As recorded in the dataset (1 = male, 0 = female).",
    "Scholarship holder": "Does the student hold a scholarship?",
    "Age at enrollment": "Age in years when the student enrolled.",
    "Curricular units 1st sem (credited)": "Units credited (recognised from earlier studies) in semester 1.",
    "Curricular units 1st sem (enrolled)": "Units the student enrolled in during semester 1.",
    "Curricular units 1st sem (evaluations)": "Number of evaluations (exams/tests) taken in semester 1.",
    "Curricular units 1st sem (approved)": "Units passed in semester 1.",
    "Curricular units 1st sem (grade)": "Average grade of semester 1 units (dataset scale 0-20; training range 0-18).",
    "Curricular units 1st sem (without evaluations)": "Units with no evaluation taken in semester 1.",
}

FEATURE_GROUPS = [
    ("Student background", ["Marital status", "Gender", "Age at enrollment", "Displaced"]),
    ("Admission", ["Application mode", "Application order", "Course", "Daytime/evening attendance",
                   "Previous qualification", "Previous qualification (grade)", "Admission grade"]),
    ("Financial situation", ["Debtor", "Tuition fees up to date", "Scholarship holder"]),
    ("First-semester academic record", [
        "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)"]),
]
# Count-type numeric features are whole numbers; the grades are decimals.
INTEGER_FEATURES = {
    "Age at enrollment", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (without evaluations)",
}


def code_label(feature: str, code) -> str:
    """Text shown in a dropdown: 'Yes' / 'Daytime' for flags, '9670 - Advertising ...' for codes."""
    labels = CODE_LABELS.get(feature, {})
    label = labels.get(code)
    if feature in YES_NO_FEATURES:
        return label if label else str(code)
    return f"{code} - {label}" if label else str(code)


def build_form_spec(required_features: list, metadata: dict) -> list:
    """
    Describes the prediction form from the SAVED MODEL: one entry per required feature, in groups.
    Dropdown options and number limits come from model_metadata.json, so the form can never offer
    a value the model was not trained on.

    Returns [{"title": str, "fields": [ {name, kind("select"|"number"), options, min, max, step,
                                          is_int, help} ]}]
    """
    allowed = metadata.get("allowed_values", {})
    ranges = metadata.get("numeric_ranges", {})

    def make_field(name):
        field = {"name": name, "help": FEATURE_HELP.get(name, "")}
        if name in allowed:
            field.update({"kind": "select", "options": list(allowed[name])})
        else:
            low, high = ranges.get(name, [0.0, 1000.0])
            is_int = name in INTEGER_FEATURES
            field.update({
                "kind": "number", "is_int": is_int,
                "min": int(low) if is_int else float(low),
                "max": int(high) if is_int else float(high),
                "step": 1 if is_int else 0.1,
            })
        return field

    remaining = list(required_features)
    groups = []
    for title, names in FEATURE_GROUPS:
        fields = [make_field(n) for n in names if n in remaining]
        for n in names:
            if n in remaining:
                remaining.remove(n)
        if fields:
            groups.append({"title": title, "fields": fields})
    if remaining:                                   # a feature this file does not know about
        groups.append({"title": "Other details", "fields": [make_field(n) for n in remaining]})
    return groups


# ----------------------------------------------------------------------
# 2. Student details and marks entered / edited by the user
# ----------------------------------------------------------------------
TEXT_INFO_FIELDS = ["student_name", "student_id", "course", "academic_year"]
FLOAT_INFO_FIELDS = ["sgpa", "cgpa", "percentage", "attendance"]
INT_INFO_FIELDS = ["semester", "backlogs"]
INFO_FIELD_ORDER = ["student_name", "student_id", "course", "semester", "academic_year",
                    "sgpa", "cgpa", "percentage", "attendance", "backlogs"]


def info_to_text(info: Optional[dict]) -> dict:
    """Student-details dictionary -> {field: text} for the text boxes. Missing -> ''."""
    info = info or {}
    out = {}
    for key in INFO_FIELD_ORDER:
        value = info.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            out[key] = ""
        elif isinstance(value, float) and value == int(value) and key in INT_INFO_FIELDS:
            out[key] = str(int(value))
        else:
            out[key] = f"{value:g}" if isinstance(value, float) else str(value)
    return out


def parse_info_fields(raw: dict):
    """
    Text boxes -> student-details dictionary.
    Empty text becomes None (never a made-up value). Returns (info, error_messages).
    """
    info, errors = {}, []
    for key in TEXT_INFO_FIELDS:
        text = str(raw.get(key) or "").strip()
        info[key] = text or None
    for key in FLOAT_INFO_FIELDS + INT_INFO_FIELDS:
        text = str(raw.get(key) or "").strip().replace(",", ".")
        if not text:
            info[key] = None
            continue
        try:
            number = float(text)
            if key in INT_INFO_FIELDS:
                if number != int(number):
                    raise ValueError
                number = int(number)
            info[key] = number
        except ValueError:
            info[key] = None
            kind = "a whole number" if key in INT_INFO_FIELDS else "a number"
            errors.append(f"'{key.replace('_', ' ')}' must be {kind} (you typed '{raw.get(key)}').")
    return info, errors


def drop_blank_rows(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Removes rows with no subject name AND no marks (empty rows added by accident)."""
    if df is None:
        return pd.DataFrame()
    subject_empty = df["Subject"].isna() | (df["Subject"].astype(str).str.strip() == "")
    marks_empty = df["Marks Obtained"].isna() & df["Max Marks"].isna()
    return df.loc[~(subject_empty & marks_empty)].reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. Model Performance page (numbers come ONLY from model_metadata.json)
# ----------------------------------------------------------------------
def metadata_summary(metadata: dict) -> dict:
    test = metadata.get("test_metrics", {})
    alert = metadata.get("alert_threshold", 0.4)
    at_alert = test.get(f"threshold_{float(alert):.2f}", {})
    training = metadata.get("training", {})
    return {
        "model_name": metadata.get("model_name", "Unknown"),
        "feature_set": metadata.get("feature_set"),
        "n_features": len(metadata.get("features", [])),
        "roc_auc": test.get("roc_auc"),
        "alert_threshold": alert,
        "recall": at_alert.get("recall"),
        "precision": at_alert.get("precision"),
        "n_train": training.get("n_train"),
        "n_test": training.get("n_test"),
        "dropout_rate_train": training.get("dropout_rate_train"),
    }


def comparison_table(metadata: dict) -> pd.DataFrame:
    rows = metadata.get("model_comparison", [])
    columns = ["feature_set", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


FEATURE_SET_NAMES = {
    "enrollment": "Enrollment data only",
    "sem1": "Enrollment + 1st semester",
    "full": "Enrollment + 1st and 2nd semester",
}


def comparison_chart(table: pd.DataFrame, metric: str) -> Optional[go.Figure]:
    if table.empty or metric not in table.columns:
        return None
    fig = go.Figure()
    palette = {"enrollment": "#9AA5B1", "sem1": "#1F3A5F", "full": "#2CA02C"}
    for feature_set, part in table.groupby("feature_set", sort=False):
        fig.add_trace(go.Bar(
            x=part["model"], y=part[metric], name=FEATURE_SET_NAMES.get(feature_set, feature_set),
            marker_color=palette.get(feature_set, "#1F77B4"),
            text=part[metric].round(3), textposition="outside"))
    fig.update_layout(barmode="group", template="plotly_white", title=f"{metric} by model",
                      yaxis=dict(range=[0, 1.05], title=metric), legend_title_text="Data used",
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def top_features_chart(features: dict, title: str) -> Optional[go.Figure]:
    if not features:
        return None
    ordered = sorted(features.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in ordered], y=[k for k, _ in ordered], orientation="h",
                           marker_color="#1F3A5F"))
    fig.update_layout(template="plotly_white", title=title, xaxis_title="Importance",
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


DEFAULT_SETTINGS = {
    "pass_pct": 40.0, "weak_pct": 50.0, "good_pct": 75.0,
    "required_attendance": 75.0, "gpa_scale": 10.0, "study_hours": 15,
}