"""
Create sample academic report PDFs for testing the EduPredict PDF extractor.

Run from the project root:
    python scripts/make_sample_pdfs.py

Files are written to tests/sample_pdfs/ . All students and marks are FAKE.
"""
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent.parent / "tests" / "sample_pdfs"
OUT.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()


def grid_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ]))
    return t


def sample_1_standard_table():
    """Header details + bordered table: Subject | Max Marks | Marks Obtained | Grade."""
    doc = SimpleDocTemplate(str(OUT / "sample_1_standard_table.pdf"), pagesize=A4)
    story = [
        Paragraph("Sunrise Institute of Technology", styles["Title"]),
        Paragraph("Semester Grade Report", styles["Heading2"]),
        Paragraph("Student Name: Asha Rao", styles["Normal"]),
        Paragraph("USN: 1SI23CS045", styles["Normal"]),
        Paragraph("Course: B.E. Computer Science and Engineering", styles["Normal"]),
        Paragraph("Semester: 3", styles["Normal"]),
        Paragraph("Academic Year: 2023-24", styles["Normal"]),
        Spacer(1, 12),
        grid_table([
            ["Subject", "Max Marks", "Marks Obtained", "Grade"],
            ["Data Structures", "100", "78", "A"],
            ["Digital Electronics", "100", "62", "B"],
            ["Discrete Mathematics", "100", "34", "F"],
            ["Computer Organization", "100", "85", "A+"],
            ["Object Oriented Programming", "100", "71", "B+"],
        ]),
        Spacer(1, 12),
        Paragraph("SGPA: 7.20", styles["Normal"]),
        Paragraph("CGPA: 7.05", styles["Normal"]),
        Paragraph("Attendance: 82%", styles["Normal"]),
        Paragraph("Backlogs: 1", styles["Normal"]),
    ]
    doc.build(story)


def sample_2_internal_external():
    """Indian-university style: code, name, internal, external, total, result (no max marks)."""
    doc = SimpleDocTemplate(str(OUT / "sample_2_internal_external.pdf"), pagesize=A4)
    story = [
        Paragraph("Riverdale University - Statement of Marks", styles["Title"]),
        Paragraph("Name of the Student: Karthik S", styles["Normal"]),
        Paragraph("USN: 1RV22EC101", styles["Normal"]),
        Paragraph("Branch: Electronics and Communication", styles["Normal"]),
        Paragraph("Semester: V", styles["Normal"]),
        Spacer(1, 12),
        grid_table([
            ["Subject Code", "Subject Name", "Internal Marks", "External Marks", "Total Marks", "Result"],
            ["21EC51", "Digital Signal Processing", "38", "52", "90", "P"],
            ["21EC52", "Microwave Engineering", "30", "41", "71", "P"],
            ["21EC53", "VLSI Design", "25", "18", "43", "F"],
            ["21EC54", "Control Systems", "35", "AB", "35", "F"],
        ]),
        Spacer(1, 12),
        Paragraph("SGPA: 6.4", styles["Normal"]),
        Paragraph("Attendance Percentage: 74.5%", styles["Normal"]),
    ]
    doc.build(story)


def sample_3_text_only():
    """No table borders: plain lines of text."""
    c = canvas.Canvas(str(OUT / "sample_3_text_only.pdf"), pagesize=A4)
    y = 800
    lines = [
        "Greenfield College of Engineering",
        "PROGRESS REPORT",
        "Name: Meera Nair      Roll No: GF2021-114",
        "Programme: B.Tech Information Technology",
        "Semester: 2          Academic Year: 2022-2023",
        "",
        "Mathematics II 64 100",
        "Physics 58 100",
        "Programming in C 91 100",
        "Engineering Graphics 45 50",
        "",
        "Percentage: 72.4%",
        "CGPA: 7.6",
        "Attendance: 91%",
    ]
    for line in lines:
        c.drawString(60, y, line)
        y -= 20
    c.save()


def sample_4_multi_semester():
    """Two semesters on two pages, each with its own SGPA."""
    doc = SimpleDocTemplate(str(OUT / "sample_4_multi_semester.pdf"), pagesize=A4)
    from reportlab.platypus import PageBreak
    story = [
        Paragraph("Lakeside Institute - Consolidated Report", styles["Title"]),
        Paragraph("Student Name: Rohan Verma", styles["Normal"]),
        Paragraph("Student ID: LI2022-0457", styles["Normal"]),
        Paragraph("Course: B.Sc Computer Science", styles["Normal"]),
        Spacer(1, 10),
        Paragraph("Semester: 1", styles["Heading3"]),
        grid_table([
            ["Subject", "Max Marks", "Marks Obtained"],
            ["Calculus", "100", "81"],
            ["Intro to Programming", "100", "88"],
            ["Physics", "100", "69"],
        ]),
        Paragraph("Semester 1 SGPA: 8.4", styles["Normal"]),
        PageBreak(),
        Paragraph("Semester: 2", styles["Heading3"]),
        grid_table([
            ["Subject", "Max Marks", "Marks Obtained"],
            ["Linear Algebra", "100", "66"],
            ["Data Structures", "100", "59"],
            ["Statistics", "100", "47"],
        ]),
        Paragraph("Semester 2 SGPA: 6.9", styles["Normal"]),
        Paragraph("CGPA: 7.65", styles["Normal"]),
    ]
    doc.build(story)


def sample_5_scanned():
    """An image-only PDF (no text layer) to test OCR."""
    img = Image.new("RGB", (1240, 1000), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 34)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 34)
        except OSError:
            font = ImageFont.load_default()
    lines = [
        "Name: Divya Menon",
        "USN: 1MS21IS033",
        "Semester: 4",
        "",
        "Operating Systems 72 100",
        "Computer Networks 55 100",
        "Database Systems 80 100",
        "",
        "SGPA: 7.1",
        "Attendance: 88%",
    ]
    y = 60
    for line in lines:
        d.text((60, y), line, fill="black", font=font)
        y += 60
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    from reportlab.lib.utils import ImageReader
    c = canvas.Canvas(str(OUT / "sample_5_scanned.pdf"), pagesize=A4)
    c.drawImage(ImageReader(buf), 20, 300, width=555, height=447)
    c.save()


def sample_6_empty_page():
    c = canvas.Canvas(str(OUT / "sample_6_empty.pdf"), pagesize=A4)
    c.showPage()
    c.save()


def sample_7_not_a_pdf():
    (OUT / "sample_7_not_a_pdf.pdf").write_text("This is just a text file renamed to .pdf", encoding="utf-8")


if __name__ == "__main__":
    for fn in [sample_1_standard_table, sample_2_internal_external, sample_3_text_only,
               sample_4_multi_semester, sample_5_scanned, sample_6_empty_page, sample_7_not_a_pdf]:
        fn()
        print("created", fn.__name__)
    print("Done. Files are in", OUT)