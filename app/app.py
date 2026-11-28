"""
EduPredict - Student Dropout Prediction and Academic Performance Analysis
Streamlit web application (Phase 13).

Run from the EduPredict folder:   streamlit run app/app.py

How the pieces connect
  pdf_extractor.py        PDF -> student details + marks table (Phase 8)
  performance_analyzer.py marks table -> statistics and charts (Phase 9)
  predictor.py            20 form values -> dropout risk, via the saved model (Phase 10)
  recommendations.py      analysis + prediction -> advice (Phase 11)
  report_generator.py     everything -> downloadable PDF (Phase 12)
  utils.py                form layout, labels, parsing, Model Performance tables/charts

All data lives only in this browser session (st.session_state). Nothing is saved to disk.
"""

import copy
import hashlib
import inspect

import pandas as pd
import streamlit as st

st.set_page_config(page_title="EduPredict", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

import predictor                                    # noqa: E402
import recommendations as rec                       # noqa: E402
import utils                                        # noqa: E402
from performance_analyzer import (analyze_performance, performance_distribution_chart,  # noqa: E402
                                  semester_trend_chart, subject_marks_chart, weak_subjects_chart)
from report_generator import generate_report, report_filename  # noqa: E402

try:
    import pdf_extractor as pdfx
except Exception as import_error:                   # missing PyMuPDF / pdfplumber, etc.
    st.error("The PDF tools could not be loaded: "
             f"{type(import_error).__name__}: {import_error}\n\n"
             "Install them with:  pip install pymupdf pdfplumber pytesseract pillow")
    st.stop()

PAGES = [
    "Home Dashboard",
    "Student Dropout Prediction",
    "PDF Student Report Analyzer",
    "Academic Performance Analysis",
    "Recommendations",
    "Model Performance and Evaluation",
]

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {background:#F4F7FB; border:1px solid #D9E2EE;
        padding:12px 16px; border-radius:10px;}
    h1, h2, h3 {color:#1F3A5F;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
_PLOTLY_HAS_WIDTH = False
try:
    _PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
except (TypeError, ValueError):
    pass


def show_figure(fig):
    """Draws a Plotly chart at full width on any Streamlit version. Ignores None."""
    if fig is None:
        return
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


@st.cache_resource(show_spinner=False)
def load_resources():
    """Loads the saved model and its metadata once and keeps them in memory."""
    return predictor.load_model(), predictor.load_metadata()


def init_state():
    defaults = {
        "student_info": None, "marks_df": None, "data_source": None,
        "form_inputs": {}, "prediction": None,
        "settings": dict(utils.DEFAULT_SETTINGS),
        "draft_version": 0, "draft_info": None, "draft_marks": None,
        "draft_source": None, "draft_notes": [],
        "pdf_sig": None, "pdf_errors": [],
        "report_bytes": None, "report_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)


def current_analysis():
    """Runs the Phase 9 analysis on the saved marks with the saved settings (None if no marks)."""
    marks = st.session_state["marks_df"]
    if marks is None or len(marks) == 0:
        return None
    s = st.session_state["settings"]
    info = st.session_state["student_info"] or {}
    try:
        return analyze_performance(
            marks, pass_percentage=s["pass_pct"], weak_threshold=s["weak_pct"],
            good_threshold=s["good_pct"], gpa=info.get("sgpa"), cgpa=info.get("cgpa"),
            gpa_scale=s["gpa_scale"], attendance=info.get("attendance"),
            required_attendance=s["required_attendance"])
    except ValueError as exc:
        st.error(f"The settings are not valid: {exc}")
        return None


def percent(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


# ----------------------------------------------------------------------
# Page 1: Home Dashboard
# ----------------------------------------------------------------------
def page_home(meta):
    st.title("🎓 EduPredict")
    st.markdown("**Student dropout prediction and academic performance analysis**, with "
                "automatic reading of PDF student reports.")

    summary = utils.metadata_summary(meta)
    st.subheader("The prediction model")
    cols = st.columns(4)
    cols[0].metric("Model", summary["model_name"])
    cols[1].metric("ROC-AUC (test)", "n/a" if summary["roc_auc"] is None else f"{summary['roc_auc']:.3f}")
    cols[2].metric(f"Dropouts found (alert at {summary['alert_threshold'] * 100:.0f}%)",
                   percent(summary["recall"]), help="Recall on held-out test students.")
    cols[3].metric("Flagged who did drop out", percent(summary["precision"]),
                   help="Precision on held-out test students.")

    st.subheader("Your current session")
    marks = st.session_state["marks_df"]
    info = st.session_state["student_info"] or {}
    prediction = st.session_state["prediction"]
    cols = st.columns(3)
    cols[0].metric("Student", info.get("student_name") or "Not set")
    cols[1].metric("Subjects loaded", 0 if marks is None else len(marks))
    if prediction and prediction.get("ok"):
        cols[2].metric("Dropout risk", prediction["risk_level"])
    else:
        cols[2].metric("Dropout risk", "Not predicted")

    st.subheader("How to use EduPredict")
    st.markdown(
        "1. **PDF Student Report Analyzer** - upload a PDF marksheet, or type the marks in. "
        "Check and correct the data, then save it.\n"
        "2. **Academic Performance Analysis** - averages, weak and failed subjects, trends, charts.\n"
        "3. **Student Dropout Prediction** - fill in the student's details to get a risk estimate.\n"
        "4. **Recommendations** - personalised advice, and the downloadable PDF report.\n"
        "5. **Model Performance and Evaluation** - how well the model works, and its limits."
    )
    with st.expander("Important limits"):
        st.markdown(
            "- Marks read from a PDF can contain mistakes. **Always check the extracted table.**\n"
            "- The dropout model needs 20 details (admission, finances, first-semester record) that a "
            "marksheet does not contain, so it is filled in on its own page. "
            "**It never predicts from incomplete data.**\n"
            "- The model was trained on students of one Portuguese institution. Treat its output as "
            "a screening aid for a teacher or counsellor, not as a verdict.\n"
            "- Your data stays in this browser session and is not stored by the app."
        )


# ----------------------------------------------------------------------
# Page 2: Student Dropout Prediction
# ----------------------------------------------------------------------
def render_field(field, saved_value):
    name, key = field["name"], f"in_{field['name']}"
    help_text = field["help"] or None
    if field["kind"] == "select":
        options = field["options"]
        index = options.index(saved_value) if saved_value in options else None
        return st.selectbox(name, options, index=index, key=key, help=help_text,
                            format_func=lambda code, n=name: utils.code_label(n, code),
                            placeholder="Select an option")
    default = None
    if saved_value is not None:
        default = int(saved_value) if field["is_int"] else float(saved_value)
    return st.number_input(name, min_value=field["min"], max_value=field["max"], value=default,
                           step=field["step"], key=key, help=help_text)


def show_prediction(result):
    if not result:
        return
    st.divider()
    st.subheader("Result")
    if not result["ok"]:
        st.error(result["error"])
        if result["missing_features"]:
            st.write("**Still to fill in:** " + ", ".join(result["missing_features"]))
        for feature, reason in result["invalid_features"].items():
            st.write(f"**{feature}:** {reason}")
        return

    level, probability = result["risk_level"], result["dropout_probability"]
    cols = st.columns(3)
    cols[0].metric("Risk level", level)
    cols[1].metric("Model dropout score", f"{probability * 100:.0f}%")
    cols[2].metric("Alert threshold", f"{result['alert_threshold'] * 100:.0f}%")
    st.progress(min(max(float(probability), 0.0), 1.0))
    banner = {"High": st.error, "Medium": st.warning, "Low": st.success}[level]
    banner(f"{level} risk. This is an estimate from a statistical model, not a certain outcome.")
    for line in result["explanation"]:
        st.markdown("- " + line)
    for warning in result["warnings"]:
        st.warning(warning)
    st.caption(result["disclaimer"])


def page_prediction(pipe, meta):
    st.title("Student Dropout Prediction")
    st.info(
        "This model uses the student's **admission details, finances and first-semester record** "
        "(20 details). A marksheet does not contain most of them, so they are entered here. "
        "Fields start empty on purpose: **the model will not run on guessed values.** "
        "Course codes are those of the Portuguese institution in the training dataset."
    )
    spec = utils.build_form_spec(predictor.get_required_features(pipe), meta)
    saved = st.session_state["form_inputs"]
    values = {}
    with st.form("prediction_form"):
        for group in spec:
            st.subheader(group["title"])
            columns = st.columns(2)
            for position, field in enumerate(group["fields"]):
                with columns[position % 2]:
                    values[field["name"]] = render_field(field, saved.get(field["name"]))
        submitted = st.form_submit_button("Predict dropout risk", type="primary")

    if submitted:
        st.session_state["form_inputs"] = dict(values)
        st.session_state["prediction"] = predictor.predict_student(pipe, values, meta)
        st.session_state["report_bytes"] = None
    show_prediction(st.session_state["prediction"])


# ----------------------------------------------------------------------
# Page 3: PDF Student Report Analyzer (and manual marks entry)
# ----------------------------------------------------------------------
def load_draft(info, marks, source, notes):
    st.session_state["draft_info"] = dict(info or {})
    st.session_state["draft_marks"] = pdfx.clean_subjects_df(marks)
    st.session_state["draft_source"] = source
    st.session_state["draft_notes"] = list(notes)
    st.session_state["draft_version"] += 1          # new widget keys -> editors start fresh


def page_pdf():
    st.title("PDF Student Report Analyzer")
    st.caption("Upload a PDF marksheet or type the marks in. Extraction is never perfect, so you "
               "review and correct everything before it is used.")

    if st.session_state["marks_df"] is not None:
        saved_info = st.session_state["student_info"] or {}
        st.success(f"Saved data in use: {len(st.session_state['marks_df'])} subject(s)"
                   + (f" for {saved_info['student_name']}" if saved_info.get("student_name") else "")
                   + f". Source: {st.session_state['data_source']}.")

    tab_pdf, tab_manual = st.tabs(["Upload a PDF report", "Enter marks manually"])

    with tab_pdf:
        ocr_ok = pdfx.ocr_available()
        use_ocr = st.checkbox("Use OCR for scanned PDFs", value=ocr_ok, disabled=not ocr_ok)
        if not ocr_ok:
            st.caption("OCR (Tesseract) is not installed on this computer, so scanned PDFs cannot be "
                       "read. Text-based PDFs work normally.")
        uploaded = st.file_uploader(f"Choose a PDF file (up to {pdfx.MAX_FILE_MB} MB)", type=["pdf"],
                                    key="pdf_upload")
        if uploaded is None:
            st.session_state["pdf_sig"] = None
            st.session_state["pdf_errors"] = []
        else:
            data = uploaded.getvalue()
            signature = hashlib.sha256(data).hexdigest() + f"|{use_ocr}"
            if signature != st.session_state["pdf_sig"]:
                with st.spinner("Reading the PDF..."):
                    result = pdfx.extract_from_pdf(data, uploaded.name, use_ocr=use_ocr)
                st.session_state["pdf_sig"] = signature
                st.session_state["pdf_errors"] = list(result.errors)
                if result.ok:
                    notes = list(result.warnings)
                    if result.semester_gpas:
                        notes.append("Semester-wise SGPA found: " + ", ".join(
                            f"Sem {g['semester']}: {g['sgpa']:g}" for g in result.semester_gpas))
                    load_draft(result.info, result.subjects,
                               f"PDF file '{uploaded.name}' ({result.method} extraction, "
                               f"{result.pages} page(s))", notes)
            for message in st.session_state["pdf_errors"]:
                st.error(message)

    with tab_manual:
        st.write("Start with an empty table and type in the student's details and subject marks.")
        if st.button("Start a blank form"):
            load_draft({}, pdfx.empty_subjects_df(), "Manual entry", [])
        if st.session_state["marks_df"] is not None and st.button("Edit the saved data again"):
            load_draft(st.session_state["student_info"], st.session_state["marks_df"],
                       st.session_state["data_source"], [])

    draft_marks = st.session_state["draft_marks"]
    if draft_marks is None:
        st.info("Upload a PDF or start a blank form to begin.")
        return

    version = st.session_state["draft_version"]
    st.divider()
    st.subheader("Review and correct")
    st.caption(f"Source: {st.session_state['draft_source']}. Press **Check and save** before leaving "
               "this page: unsaved edits are lost when you switch pages.")
    for note in st.session_state["draft_notes"]:
        st.warning(note)

    st.markdown("**Student details** (leave blank if unknown)")
    texts = utils.info_to_text(st.session_state["draft_info"])
    raw = {}
    columns = st.columns(3)
    for position, key in enumerate(utils.INFO_FIELD_ORDER):
        with columns[position % 3]:
            raw[key] = st.text_input(pdfx.INFO_LABELS[key], value=texts[key], key=f"info_{key}_{version}")

    st.markdown("**Subject marks** - edit cells, add rows at the bottom, or select a row and press Delete.")
    edited = st.data_editor(
        draft_marks, num_rows="dynamic", hide_index=True, key=f"marks_editor_{version}",
        column_config={
            "Semester": st.column_config.NumberColumn("Semester", min_value=1, max_value=12, step=1),
            "Marks Obtained": st.column_config.NumberColumn("Marks Obtained", min_value=0),
            "Max Marks": st.column_config.NumberColumn("Max Marks", min_value=0),
            "Check": st.column_config.TextColumn("Check", disabled=True,
                                                 help="Notes from the PDF reader. Fix the row, then ignore this note."),
        })

    if st.button("Check and save this data", type="primary"):
        info, parse_errors = utils.parse_info_fields(raw)
        marks = utils.drop_blank_rows(pdfx.clean_subjects_df(edited))
        issues = [] if parse_errors else pdfx.validate_extraction(info, marks)
        errors = parse_errors + [i["message"] for i in issues if i["level"] == "error"]
        warnings = [i["message"] for i in issues if i["level"] == "warning"]
        if errors:
            st.error("Please fix these before saving:")
            for message in errors:
                st.write("- " + message)
        else:
            st.session_state["student_info"] = info
            st.session_state["marks_df"] = marks
            st.session_state["data_source"] = st.session_state["draft_source"]
            st.session_state["report_bytes"] = None
            st.success(f"Saved {len(marks)} subject(s). Open **Academic Performance Analysis** next.")
            for message in warnings:
                st.warning(message)


# ----------------------------------------------------------------------
# Page 4: Academic Performance Analysis
# ----------------------------------------------------------------------
def settings_input(label, setting, minimum=0.0, maximum=100.0):
    """A number box whose value is remembered between pages (widgets are forgotten when a page is
    left, so the value is kept in st.session_state['settings'] and copied back in)."""
    key = f"set_{setting}"
    if key not in st.session_state:
        st.session_state[key] = float(st.session_state["settings"][setting])
    value = st.number_input(label, min_value=minimum, max_value=maximum, step=1.0, key=key)
    st.session_state["settings"][setting] = value


def page_analysis():
    st.title("Academic Performance Analysis")
    if st.session_state["marks_df"] is None or len(st.session_state["marks_df"]) == 0:
        st.info("No marks saved yet. Go to **PDF Student Report Analyzer**, add the marks and press "
                "**Check and save**.")
        return

    s = st.session_state["settings"]
    with st.expander("Settings: pass mark and thresholds (institutions differ)", expanded=False):
        cols = st.columns(5)
        with cols[0]:
            settings_input("Pass mark (%)", "pass_pct")
        with cols[1]:
            settings_input("Weak-subject mark (%)", "weak_pct")
        with cols[2]:
            settings_input("Good mark (%)", "good_pct")
        with cols[3]:
            settings_input("Required attendance (%)", "required_attendance")
        with cols[4]:
            settings_input("GPA scale (max)", "gpa_scale", minimum=1.0)
        if s["weak_pct"] < s["pass_pct"] or s["good_pct"] < s["weak_pct"]:
            st.warning("The marks should go pass <= weak <= good. They are adjusted automatically.")

    analysis = current_analysis()
    if analysis is None:
        return
    if not analysis["ok"]:
        for message in analysis["warnings"]:
            st.error(message)
        st.info("Fix the marks table on the PDF Student Report Analyzer page and save again.")
        return

    cols = st.columns(6)
    cols[0].metric("Average", f"{analysis['average_percentage']:.1f}%")
    cols[1].metric("Subjects", analysis["num_subjects"])
    cols[2].metric("Highest marks", f"{analysis['highest_marks']:g}")
    cols[3].metric("Lowest marks", f"{analysis['lowest_marks']:g}")
    cols[4].metric("Below pass mark", analysis["num_failed"])
    cols[5].metric(f"Below {analysis['weak_threshold']:g}%", analysis["num_below_threshold"])

    st.subheader("Key findings")
    for line in analysis["insights"]:
        st.markdown("- " + line)
    for message in analysis["warnings"]:
        st.warning(message)

    st.subheader("Charts")
    show_figure(subject_marks_chart(analysis))
    left, right = st.columns(2)
    with left:
        show_figure(performance_distribution_chart(analysis))
    with right:
        weak_chart = weak_subjects_chart(analysis)
        if weak_chart is None:
            st.success("No subject is below the weak-subject mark.")
        show_figure(weak_chart)
    trend_chart = semester_trend_chart(analysis)
    if trend_chart is None:
        st.caption("A semester trend chart needs marks from at least two semesters.")
    show_figure(trend_chart)

    st.subheader("Marks used in the analysis")
    columns = [c for c in ["Subject", "Semester", "Marks Obtained", "Max Marks", "Percentage",
                           "Performance Band"] if c in analysis["clean_df"].columns]
    st.dataframe(analysis["clean_df"][columns], hide_index=True)


# ----------------------------------------------------------------------
# Page 5: Recommendations and report download
# ----------------------------------------------------------------------
def report_section(analysis, prediction, recs, plan):
    st.divider()
    st.subheader("Download the report")
    if st.button("Prepare PDF report"):
        try:
            with st.spinner("Building the report..."):
                st.session_state["report_bytes"] = generate_report(
                    st.session_state["student_info"], analysis, prediction, recs, plan,
                    data_source=st.session_state["data_source"])
                st.session_state["report_name"] = report_filename(st.session_state["student_info"])
        except Exception as exc:
            st.session_state["report_bytes"] = None
            st.error(f"The report could not be built ({type(exc).__name__}). "
                     "Check the saved data and try again.")
    if st.session_state["report_bytes"]:
        st.download_button("Download PDF report", data=st.session_state["report_bytes"],
                           file_name=st.session_state["report_name"], mime="application/pdf")
        st.caption("The report reflects the data at the moment you pressed Prepare. "
                   "Prepare it again after changing anything. It contains student information: "
                   "store and share it carefully.")


def page_recommendations():
    st.title("Recommendations")
    analysis = current_analysis()
    prediction = st.session_state["prediction"]
    has_analysis = bool(analysis) and analysis["ok"]
    has_prediction = bool(prediction) and prediction["ok"]

    cols = st.columns(2)
    cols[0].metric("Academic analysis", "Available" if has_analysis else "Not available")
    cols[1].metric("Dropout prediction", "Available" if has_prediction else "Not available")
    if not has_analysis and not has_prediction:
        st.info("Recommendations need data. Save marks on the **PDF Student Report Analyzer** page and/or "
                "run a prediction on the **Student Dropout Prediction** page.")
        return

    recs = rec.generate_recommendations(analysis, prediction, st.session_state["form_inputs"])
    st.subheader(recs["summary"])
    for note in recs["notes"]:
        st.info(note)
    colors = {"High": "red", "Medium": "orange", "Low": "blue"}
    for number, item in enumerate(recs["items"], start=1):
        st.markdown(f"**{number}.** :{colors[item['priority']]}[**{item['priority']}**] "
                    f"**{item['title']}**")
        st.write(item["action"])
        st.caption("Why: " + item["reason"])

    plan = pd.DataFrame()
    if has_analysis:
        st.subheader("Weekly study plan")
        s = st.session_state["settings"]
        if "set_study_hours" not in st.session_state:
            st.session_state["set_study_hours"] = int(s["study_hours"])
        s["study_hours"] = st.slider("Study hours available per week", 5, 40, key="set_study_hours")
        plan = rec.weekly_study_plan(analysis, hours_per_week=s["study_hours"])
        if plan.empty:
            st.success("No weak subjects, so no extra study plan is needed.")
        else:
            st.dataframe(plan, hide_index=True)
            st.caption("Hours are split in proportion to how far each subject is below the "
                       "weak-subject mark. Adjust them to the real timetable.")

    report_section(analysis, prediction, recs, plan)


# ----------------------------------------------------------------------
# Page 6: Model Performance and Evaluation
# ----------------------------------------------------------------------
def page_model(pipe, meta):
    st.title("Model Performance and Evaluation")
    summary = utils.metadata_summary(meta)
    st.markdown(f"**Chosen model:** {summary['model_name']} using the "
                f"**{utils.FEATURE_SET_NAMES.get(summary['feature_set'], summary['feature_set'])}** "
                f"data ({summary['n_features']} features). All numbers below were measured on "
                f"{summary['n_test']} held-out test students and come from `model_metadata.json`.")

    cols = st.columns(4)
    cols[0].metric("ROC-AUC", "n/a" if summary["roc_auc"] is None else f"{summary['roc_auc']:.3f}")
    cols[1].metric("Dropout recall", percent(summary["recall"]),
                   help=f"At the {summary['alert_threshold'] * 100:.0f}% alert threshold.")
    cols[2].metric("Precision", percent(summary["precision"]),
                   help=f"At the {summary['alert_threshold'] * 100:.0f}% alert threshold.")
    cols[3].metric("Dropout rate in training data", percent(summary["dropout_rate_train"]))

    st.subheader("Why recall matters more than accuracy here")
    st.markdown(
        "Only about a third of students drop out, so a model that always says 'will not drop out' "
        "would still look fairly accurate while helping nobody. What matters is **recall** - the share "
        "of real dropouts the model catches - balanced against **precision**, the share of flagged "
        "students who really drop out. The alert threshold sets that balance."
    )
    thresholds = [{"Alert threshold": key.split("_", 1)[1], "Recall": value.get("recall"),
                   "Precision": value.get("precision")}
                  for key, value in meta.get("test_metrics", {}).items()
                  if key.startswith("threshold_") and isinstance(value, dict)]
    if thresholds:
        st.dataframe(pd.DataFrame(thresholds), hide_index=True)
        st.caption("A lower threshold catches more dropouts but also flags more students who would "
                   "have been fine.")

    st.subheader("Model comparison")
    table = utils.comparison_table(meta)
    if table.empty:
        st.info("No model comparison found in the metadata file.")
    else:
        shown = table.copy()
        shown["feature_set"] = shown["feature_set"].map(lambda k: utils.FEATURE_SET_NAMES.get(k, k))
        st.dataframe(shown, hide_index=True)
        metric = st.selectbox("Metric to chart", ["recall", "precision", "f1", "roc_auc", "accuracy"])
        show_figure(utils.comparison_chart(table, metric))
        st.caption("'Enrollment data only' models can screen students before any results exist; "
                   "adding first-semester results clearly improves recall.")

    st.subheader("What the Random Forest found most important")
    show_figure(utils.top_features_chart(meta.get("random_forest_top_features", {}),
                                         "Random Forest feature importance"))

    st.subheader("Training details")
    training = meta.get("training", {})
    details = {"Training students": training.get("n_train"), "Test students": training.get("n_test"),
               "Random seed": training.get("random_state")}
    details.update({f"{name} version": version for name, version in meta.get("versions", {}).items()})
    st.dataframe(pd.DataFrame({"Item": list(details), "Value": [str(v) for v in details.values()]}),
                 hide_index=True)

    st.subheader("Limits to keep in mind")
    st.markdown(
        "- The data come from one Portuguese higher-education institution, so results may not carry "
        "over to other countries or colleges.\n"
        "- Probabilities are not calibrated (the model was trained with balanced class weights); "
        "treat them as a ranking and a screening signal.\n"
        "- The model finds patterns associated with dropout; it does not explain causes.\n"
        "- Final decisions about support should always involve people who know the student."
    )


# ----------------------------------------------------------------------
# Main: sidebar navigation and page routing
# ----------------------------------------------------------------------
def main():
    init_state()
    try:
        pipe, meta = load_resources()
    except predictor.ModelLoadError as error:
        st.error(f"The prediction model could not be loaded: {error}")
        st.stop()

    st.sidebar.title("🎓 EduPredict")
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Your data stays in this browser session and is not saved by the app.")
    if st.sidebar.button("Clear all my data"):
        st.session_state.clear()
        st.rerun()

    if page == PAGES[0]:
        page_home(meta)
    elif page == PAGES[1]:
        page_prediction(pipe, meta)
    elif page == PAGES[2]:
        page_pdf()
    elif page == PAGES[3]:
        page_analysis()
    elif page == PAGES[4]:
        page_recommendations()
    else:
        page_model(pipe, meta)


main()