"""
report_generator.py
-------------------
Builds the downloadable PDF student analysis report for EduPredict (Phase 12).

    pdf_bytes = generate_report(student_info, analysis, prediction, recommendations, study_plan)

Everything is built IN MEMORY and returned as bytes: no file is written to disk, which keeps
student information private. The Streamlit app (Phase 13) hands these bytes straight to
st.download_button.

Every input is optional. A section with no data says so plainly instead of being filled with
made-up values. The academic analysis and the machine-learning dropout estimate are shown in
separate sections, because they are different things.

Charts are drawn with Matplotlib (no extra browser engine needed, unlike saving Plotly charts).
"""

import io
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

import matplotlib
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, KeepTogether, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

# Same colours as performance_analyzer.py, repeated here so this file has no Plotly dependency.
BAND_ORDER = ["Fail", "Weak", "Average", "Good"]
BAND_COLORS = {"Fail": "#D62728", "Weak": "#FF7F0E", "Average": "#1F77B4", "Good": "#2CA02C"}
RISK_COLORS = {"Low": "#2CA02C", "Medium": "#FF7F0E", "High": "#D62728"}
PRIORITY_COLORS = {"High": "#D62728", "Medium": "#FF7F0E", "Low": "#1F77B4"}
NAVY = colors.HexColor("#1F3A5F")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#EEF2F7")

INFO_LABELS = [
    ("student_name", "Student name"), ("student_id", "Student ID / USN"),
    ("course", "Course / Programme"), ("semester", "Semester"),
    ("academic_year", "Academic year"), ("sgpa", "SGPA / GPA"),
    ("cgpa", "CGPA"), ("percentage", "Overall percentage"),
    ("attendance", "Attendance (%)"), ("backlogs", "Backlogs"),
]
PAGE_WIDTH_CM = 17.0      # usable width on A4 with 2 cm margins


# ----------------------------------------------------------------------
# Fonts and styles
# ----------------------------------------------------------------------
def _register_fonts():
    """Use the DejaVu Sans font that ships with Matplotlib (handles accents, Greek, Cyrillic).
    Falls back to Helvetica. Indic scripts are not covered by either font."""
    try:
        font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        regular, bold = font_dir / "DejaVuSans.ttf", font_dir / "DejaVuSans-Bold.ttf"
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("EduSans", str(regular)))
            pdfmetrics.registerFont(TTFont("EduSans-Bold", str(bold)))
            pdfmetrics.registerFontFamily("EduSans", normal="EduSans", bold="EduSans-Bold",
                                          italic="EduSans", boldItalic="EduSans-Bold")
            return "EduSans", "EduSans-Bold"
    except Exception:
        pass
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _register_fonts()


def _styles() -> dict:
    def style(name, **kw):
        kw.setdefault("fontName", FONT)
        return ParagraphStyle(name, **kw)

    return {
        "title": style("title", fontName=FONT_BOLD, fontSize=21, leading=25, textColor=NAVY),
        "subtitle": style("subtitle", fontSize=9.5, leading=13, textColor=GREY),
        "h1": style("h1", fontName=FONT_BOLD, fontSize=13.5, leading=17, textColor=NAVY,
                    spaceBefore=14, spaceAfter=2, keepWithNext=1),
        "body": style("body", fontSize=9.5, leading=13.5),
        "small": style("small", fontSize=8, leading=11, textColor=GREY),
        "cell": style("cell", fontSize=8.5, leading=11),
        "cellb": style("cellb", fontName=FONT_BOLD, fontSize=8.5, leading=11),
        "bullet": style("bullet", fontSize=9.5, leading=13.5, leftIndent=12, bulletIndent=2),
        "metric": style("metric", fontName=FONT_BOLD, fontSize=16, leading=19, textColor=NAVY,
                        alignment=1),
        "metric_label": style("metric_label", fontSize=7.5, leading=10, textColor=GREY,
                              alignment=1),
        "rec_title": style("rec_title", fontSize=10, leading=14, spaceBefore=6),
        "rec_body": style("rec_body", fontSize=9, leading=12.5, leftIndent=12),
        "rec_why": style("rec_why", fontSize=8.5, leading=12, leftIndent=12, textColor=GREY),
    }


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _is_empty(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _fmt(value, suffix="") -> str:
    """Readable text for any value; 'Not provided' when missing. 3.0 -> '3', 7.25 -> '7.25'."""
    if _is_empty(value) or (isinstance(value, str) and not value.strip()):
        return "Not provided"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}{suffix}"
    return f"{value}{suffix}"


def _short(text, limit=28) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _semester_label(value) -> str:
    try:
        number = float(value)
        if number == int(number):
            return f"Semester {int(number)}"
    except (TypeError, ValueError):
        pass
    return str(value)


def _p(text, style) -> Paragraph:
    """Paragraph from PLAIN text. Characters such as & and < in names are escaped so they
    cannot break the layout."""
    return Paragraph(escape(str(text)), style)


def report_filename(student_info: Optional[dict] = None, when: Optional[datetime] = None) -> str:
    when = when or datetime.now()
    name = ""
    if student_info and not _is_empty(student_info.get("student_name")):
        name = re.sub(r"[^A-Za-z0-9]+", "_", str(student_info["student_name"])).strip("_")[:30]
    return f"EduPredict_Report_{name + '_' if name else ''}{when:%Y%m%d}.pdf"


# ----------------------------------------------------------------------
# Charts (Matplotlib Figure objects: no global state, safe inside Streamlit)
# ----------------------------------------------------------------------
def _figure_to_image(fig, width_cm=PAGE_WIDTH_CM, max_height_cm=21.0) -> Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    pixel_w, pixel_h = ImageReader(buffer).getSize()
    width = width_cm * cm
    height = width * pixel_h / pixel_w
    if height > max_height_cm * cm:
        width, height = width * (max_height_cm * cm / height), max_height_cm * cm
    buffer.seek(0)
    return Image(buffer, width=width, height=height)


def _chart_subjects(analysis: dict) -> Image:
    data = analysis["clean_df"]
    count = len(data)
    labels = [_short(s) for s in data["Subject"]]
    bar_colors = [BAND_COLORS[b] for b in data["Performance Band"]]
    pass_mark = analysis["pass_percentage"]
    horizontal = count > 8

    fig = Figure(figsize=(7.5, min(8.5, 0.32 * count + 1.6)) if horizontal else (7.5, 3.1))
    ax = fig.add_subplot(111)
    if horizontal:
        bars = ax.barh(range(count), data["Percentage"], color=bar_colors)
        ax.set_yticks(range(count))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(pass_mark, linestyle="--", color="#D62728", linewidth=1)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Percentage")
    else:
        bars = ax.bar(range(count), data["Percentage"], color=bar_colors)
        ax.set_xticks(range(count))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.axhline(pass_mark, linestyle="--", color="#D62728", linewidth=1)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percentage")
    ax.bar_label(bars, fmt="%.0f%%", fontsize=7.5, padding=2)
    present = [b for b in BAND_ORDER if b in set(data["Performance Band"])]
    ax.legend(handles=[Patch(color=BAND_COLORS[b], label=b) for b in present]
              + [Patch(facecolor="none", edgecolor="#D62728", linestyle="--",
                       label=f"Pass mark {pass_mark:g}%")],
              fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18 if not horizontal else -0.08),
              ncol=5, frameon=False)
    ax.set_title("Subject-wise performance", fontsize=10.5, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _figure_to_image(fig)


def _chart_distribution_and_trend(analysis: dict) -> Image:
    summary = analysis.get("semester_summary")
    has_trend = summary is not None and len(summary) >= 2
    fig = Figure(figsize=(7.5, 2.7))
    ax1 = fig.add_subplot(1, 2, 1) if has_trend else fig.add_subplot(1, 1, 1)

    counts = analysis["band_counts"]
    bars = ax1.bar(BAND_ORDER, [counts[b] for b in BAND_ORDER],
                   color=[BAND_COLORS[b] for b in BAND_ORDER])
    ax1.bar_label(bars, fontsize=8, padding=2)
    ax1.set_title("Performance distribution", fontsize=10.5, loc="left")
    ax1.set_ylabel("Number of subjects")
    ax1.set_ylim(0, max(counts.values()) * 1.25 + 1)
    ax1.yaxis.get_major_locator().set_params(integer=True)

    if has_trend:
        ax2 = fig.add_subplot(1, 2, 2)
        labels = [_semester_label(s) for s in summary["Semester"]]
        values = list(summary["Average_Percentage"])
        ax2.plot(labels, values, marker="o", color="#1F77B4", linewidth=2)
        for x, y in zip(labels, values):
            ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=8)
        ax2.set_ylim(0, 105)
        ax2.set_title("Semester average trend", fontsize=10.5, loc="left")
        ax2.set_ylabel("Average percentage")
        ax2.tick_params(axis="x", labelsize=8)
    for axis in fig.axes:
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    return _figure_to_image(fig)


# ----------------------------------------------------------------------
# Report sections
# ----------------------------------------------------------------------
def _heading(text, st) -> list:
    rule = HRFlowable(width="100%", thickness=0.8, color=NAVY, spaceAfter=6)
    rule.keepWithNext = 1          # heading + rule + next item are never split across pages
    return [Paragraph(escape(text), st["h1"]), rule]


def _student_section(student_info, data_source, st) -> list:
    story = _heading("1. Student information", st)
    info = student_info or {}
    cells = []
    for key, label in INFO_LABELS:
        cells.append([_p(label, st["cellb"]), _p(_fmt(info.get(key)), st["cell"])])
    rows = []
    for i in range(0, len(cells), 2):
        pair = cells[i:i + 2]
        row = pair[0] + (pair[1] if len(pair) > 1 else ["", ""])
        rows.append(row)
    table = Table(rows, colWidths=[3.3 * cm, 5.2 * cm, 3.3 * cm, 5.2 * cm])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), LIGHT), ("BACKGROUND", (2, 0), (2, -1), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D1DC")),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    if data_source:
        story += [Spacer(1, 4), _p("Data source: " + data_source, st["small"])]
    return story


def _summary_section(analysis, st) -> list:
    story = _heading("2. Academic summary", st)
    if not analysis or not analysis.get("ok"):
        story.append(_p("No subject marks were analysed, so there is no academic summary.", st["body"]))
        for warning in (analysis or {}).get("warnings", []):
            story.append(_p(warning, st["small"]))
        return story

    def card(value, label):
        return [_p(value, st["metric"]), _p(label, st["metric_label"])]

    cards = [
        card(f"{analysis['average_percentage']:.1f}%", "Average"),
        card(str(analysis["num_subjects"]), "Subjects"),
        card(f"{analysis['highest_marks']:g}", "Highest marks"),
        card(f"{analysis['lowest_marks']:g}", "Lowest marks"),
        card(str(analysis["num_failed"]), f"Below pass ({analysis['pass_percentage']:g}%)"),
        card(str(analysis["num_below_threshold"]), f"Below {analysis['weak_threshold']:g}%"),
    ]
    table = Table([[c[0] for c in cards], [c[1] for c in cards]],
                  colWidths=[PAGE_WIDTH_CM / 6 * cm] * 6)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("LINEAFTER", (0, 0), (-2, -1), 0.6, colors.white),
        ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [table, Spacer(1, 8)]
    for line in analysis.get("insights", []):
        story.append(Paragraph(escape(line), st["bullet"], bulletText="•"))
    for warning in analysis.get("warnings", []):
        story.append(Paragraph(escape(warning), st["small"]))
    story.append(Spacer(1, 3))
    story.append(_p(
        f"Settings used: pass mark {analysis['pass_percentage']:g}%, weak-subject mark "
        f"{analysis['weak_threshold']:g}%, good mark {analysis['good_threshold']:g}%. "
        "Pass marks differ between institutions.", st["small"]))
    return story


def _marks_section(analysis, st) -> list:
    story = _heading("3. Subject-wise marks", st)
    if not analysis or not analysis.get("ok"):
        story.append(_p("No subject marks available.", st["body"]))
        return story
    data = analysis["clean_df"]
    show_sem = "Semester" in data.columns and data["Semester"].notna().any()

    header = ["Subject"] + (["Sem"] if show_sem else []) + ["Marks", "Max", "%", "Band"]
    widths = ([6.4, 2.0, 2.0, 2.0, 2.0, 2.6] if show_sem else [8.4, 2.2, 2.2, 2.0, 2.2])
    rows = [[_p(h, st["cellb"]) for h in header]]
    band_cells = []
    for index, row in data.iterrows():
        cells = [_p(row["Subject"], st["cell"])]
        if show_sem:
            cells.append(_p("" if _is_empty(row["Semester"]) else _semester_label(row["Semester"]).replace("Semester ", "Sem "), st["cell"]))
        cells += [_p(f"{row['Marks Obtained']:g}", st["cell"]), _p(f"{row['Max Marks']:g}", st["cell"]),
                  _p(f"{row['Percentage']:.1f}", st["cell"])]
        band = row["Performance Band"]
        cells.append(Paragraph(f'<font color="{BAND_COLORS[band]}"><b>{band}</b></font>', st["cell"]))
        rows.append(cells)
        band_cells.append(band)

    table = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D1DC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(table)
    return story


def _charts_section(analysis, st) -> list:
    story = _heading("4. Performance charts", st)
    if not analysis or not analysis.get("ok"):
        story.append(_p("No charts: there are no analysed subject marks.", st["body"]))
        return story
    for builder in (_chart_subjects, _chart_distribution_and_trend):
        try:
            story += [builder(analysis), Spacer(1, 8)]
        except Exception:
            story.append(_p("A chart could not be drawn for this data.", st["small"]))
    return story


def _risk_section(prediction, st) -> list:
    story = _heading("5. Dropout risk estimate (machine-learning model)", st)
    story.append(_p("This section is separate from the academic analysis above. It comes from a "
                    "Logistic Regression model that needs the full set of student details, not "
                    "only marks.", st["small"]))
    story.append(Spacer(1, 4))
    if not prediction:
        story.append(_p("No dropout prediction was run for this report.", st["body"]))
        return story
    if not prediction.get("ok"):
        story.append(_p("No dropout prediction was made.", st["body"]))
        if prediction.get("error"):
            story.append(_p(prediction["error"], st["small"]))
        missing = prediction.get("missing_features") or []
        if missing:
            story.append(_p("Missing: " + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""),
                            st["small"]))
        for feature, reason in list((prediction.get("invalid_features") or {}).items())[:10]:
            story.append(_p(f"Invalid {feature}: {reason}", st["small"]))
        return story

    level = prediction["risk_level"]
    color = RISK_COLORS.get(level, "#333333")
    cards = [
        [Paragraph(f'<font color="{color}">{escape(level)}</font>', st["metric"]),
         Paragraph(f"{prediction['dropout_probability'] * 100:.0f}%", st["metric"]),
         Paragraph(f"{prediction['alert_threshold'] * 100:.0f}%", st["metric"])],
        [_p("Risk level", st["metric_label"]), _p("Model dropout score", st["metric_label"]),
         _p("Alert threshold", st["metric_label"])],
    ]
    table = Table(cards, colWidths=[PAGE_WIDTH_CM / 3 * cm] * 3)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("LINEAFTER", (0, 0), (-2, -1), 0.6, colors.white),
        ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
    ]))
    story += [table, Spacer(1, 8)]
    block = [Paragraph(escape(line), st["bullet"], bulletText="•")
             for line in prediction.get("explanation", [])]
    block += [Paragraph("Caution: " + escape(w), st["small"]) for w in prediction.get("warnings", [])]
    block += [Spacer(1, 4), _p(prediction.get("disclaimer", ""), st["small"])]
    story.append(KeepTogether(block))
    return story


def _recommendations_section(recommendations, study_plan, st) -> list:
    story = _heading("6. Personalised recommendations", st)
    items = (recommendations or {}).get("items") or []
    if not items:
        story.append(_p("No recommendations were generated.", st["body"]))
    for number, item in enumerate(items, start=1):
        color = PRIORITY_COLORS.get(item["priority"], "#333333")
        story.append(KeepTogether([
            Paragraph(f'{number}. <font color="{color}"><b>[{escape(item["priority"])}]</b></font> '
                      f'<b>{escape(item["title"])}</b>', st["rec_title"]),
            _p(item["action"], st["rec_body"]),
            _p("Why: " + item["reason"], st["rec_why"]),
        ]))
    for note in (recommendations or {}).get("notes", []):
        story += [Spacer(1, 3), _p("Note: " + note, st["small"])]

    if study_plan is not None and len(study_plan):
        story += [Spacer(1, 10), Paragraph("<b>Suggested weekly study plan</b>", st["body"]), Spacer(1, 3)]
        rows = [[_p(h, st["cellb"]) for h in ("Subject", "Current %", "Suggested hours/week")]]
        for _, row in study_plan.iterrows():
            rows.append([_p(row["Subject"], st["cell"]), _p(f"{row['Current %']:.1f}", st["cell"]),
                         _p(f"{row['Suggested hours/week']:g}", st["cell"])])
        table = Table(rows, colWidths=[8 * cm, 3.5 * cm, 5 * cm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D1DC")),
        ]))
        story += [table, _p("Hours are split in proportion to how far each subject is below the "
                            "weak-subject mark. Adjust them to the student's real timetable.", st["small"])]
    return story


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def generate_report(student_info: Optional[dict] = None,
                    analysis: Optional[dict] = None,
                    prediction: Optional[dict] = None,
                    recommendations: Optional[dict] = None,
                    study_plan: Optional[pd.DataFrame] = None,
                    data_source: Optional[str] = None,
                    generated_on: Optional[datetime] = None) -> bytes:
    """Returns the finished PDF as bytes. Every argument is optional."""
    st = _styles()
    generated_on = generated_on or datetime.now()

    story = [
        Paragraph("EduPredict", st["title"]),
        Paragraph("Student Analysis Report", ParagraphStyle("sub", parent=st["subtitle"], fontSize=12,
                                                            leading=16, textColor=NAVY)),
        Paragraph(f"Date of analysis: {generated_on:%d %B %Y, %H:%M}", st["subtitle"]),
    ]
    story += _student_section(student_info, data_source, st)
    story += _summary_section(analysis, st)
    story += _marks_section(analysis, st)
    story += _charts_section(analysis, st)
    story += _risk_section(prediction, st)
    story += _recommendations_section(recommendations, study_plan, st)
    story += [Spacer(1, 12), HRFlowable(width="100%", thickness=0.4, color=GREY), Spacer(1, 4),
              _p("Limits of this report: marks read from a PDF can contain extraction errors unless "
                 "they were reviewed by a person. The dropout estimate is a statistical guess from a "
                 "public dataset, not a certain outcome. Recommendations are rule-based suggestions "
                 "and do not replace advice from teachers or counsellors.", st["small"])]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(2 * cm, 1.1 * cm, "EduPredict - confidential: contains student information")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                                 topMargin=1.8 * cm, bottomMargin=2 * cm,
                                 title="EduPredict Student Analysis Report", author="EduPredict")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


# ----------------------------------------------------------------------
# Self-test:   python app/report_generator.py     (writes reports/sample_report.pdf)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import predictor
    import recommendations as rec
    from performance_analyzer import analyze_performance

    marks = pd.DataFrame({
        "Subject": ["Maths", "Physics", "Chemistry", "English", "Programming", "Electronics"],
        "Marks Obtained": [82, 35, 58, 71, 90, 44], "Max Marks": [100] * 6,
        "Semester": [1, 1, 1, 2, 2, 2],
    })
    analysis = analyze_performance(marks, attendance=68, required_attendance=75, cgpa=7.2)
    pipe, meta = predictor.load_model(), predictor.load_metadata()
    form = dict(meta["smoke_test"]["row"])
    form.update({"Debtor": 1, "Tuition fees up to date": 0, "Scholarship holder": 0,
                 "Curricular units 1st sem (approved)": 2})
    prediction = predictor.predict_student(pipe, form, meta)
    recs = rec.generate_recommendations(analysis, prediction, form)
    info = {"student_name": "Sample Student (fake data)", "student_id": "TEST001",
            "course": "B.Tech CSE", "semester": 2, "attendance": 68, "cgpa": 7.2}

    pdf = generate_report(info, analysis, prediction, recs, rec.weekly_study_plan(analysis),
                          data_source="Sample data for testing")
    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "sample_report.pdf"
    out_file.write_bytes(pdf)
    print(f"Report written: {out_file}  ({len(pdf) / 1024:.0f} KB)")
    