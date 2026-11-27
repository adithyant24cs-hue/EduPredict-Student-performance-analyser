import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.pdf_extractor import (  # noqa: E402
    MAX_FILE_MB, SUBJECT_COLUMNS, extract_from_pdf, ocr_available, to_number,
    validate_extraction, validate_pdf_file)

SAMPLES = ROOT / "tests" / "sample_pdfs"


@pytest.fixture(scope="session", autouse=True)
def make_samples():
    if not (SAMPLES / "sample_1_standard_table.pdf").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_pdfs.py")], check=True)


def run(name):
    path = SAMPLES / name
    return extract_from_pdf(path.read_bytes(), name)


def test_standard_table():
    r = run("sample_1_standard_table.pdf")
    assert r.ok and r.method == "text"
    assert r.subjects["Subject"].tolist() == ["Data Structures", "Digital Electronics", "Discrete Mathematics",
                                              "Computer Organization", "Object Oriented Programming"]
    assert r.subjects["Marks Obtained"].tolist() == [78, 62, 34, 85, 71]
    assert (r.subjects["Max Marks"] == 100).all()
    assert r.info["student_name"] == "Asha Rao"
    assert r.info["student_id"] == "1SI23CS045"
    assert r.info["semester"] == 3 and r.info["academic_year"] == "2023-24"
    assert r.info["cgpa"] == 7.05 and r.info["attendance"] == 82.0 and r.info["backlogs"] == 1


def test_internal_external_has_no_invented_max_marks():
    r = run("sample_2_internal_external.pdf")
    assert r.subjects["Marks Obtained"].tolist() == [90, 71, 43, 35]
    assert r.subjects["Max Marks"].isna().all()                 # never assumed
    assert any("Maximum marks were not found" in w for w in r.warnings)
    assert r.info["semester"] == 5                              # roman numeral V
    assert r.info["attendance"] == 74.5 and r.info["percentage"] is None


def test_text_only_lines_are_flagged_for_review():
    r = run("sample_3_text_only.pdf")
    assert len(r.subjects) == 4
    assert r.subjects["Check"].str.contains("verify").all()
    assert r.subjects.set_index("Subject").loc["Engineering Graphics", "Max Marks"] == 50
    assert r.info["student_id"] == "GF2021-114" and r.info["cgpa"] == 7.6


def test_multi_semester_pdf():
    r = run("sample_4_multi_semester.pdf")
    assert r.subjects["Semester"].tolist() == [1, 1, 1, 2, 2, 2]
    assert [g["semester"] for g in r.semester_gpas] == [1, 2]
    assert r.info["sgpa"] is None and r.info["semester"] is None


@pytest.mark.skipif(not ocr_available(), reason="Tesseract OCR is not installed")
def test_scanned_pdf_uses_ocr():
    r = run("sample_5_scanned.pdf")
    assert r.method == "ocr"
    assert "Operating Systems" in r.subjects["Subject"].tolist()
    assert any("OCR" in w for w in r.warnings)


def test_empty_pdf_does_not_crash():
    r = run("sample_6_empty.pdf")
    assert not r.ok and r.errors
    assert list(r.subjects.columns) == SUBJECT_COLUMNS


def test_not_a_pdf():
    r = run("sample_7_not_a_pdf.pdf")
    assert not r.ok and "real PDF" in r.errors[0]


def test_empty_bytes_and_random_bytes_do_not_crash():
    assert extract_from_pdf(b"").errors
    assert extract_from_pdf(os.urandom(2000)).errors
    assert extract_from_pdf(b"%PDF-1.4 this is garbage").errors


def test_file_too_large():
    big = b"%PDF-1.4\n" + b"0" * int((MAX_FILE_MB + 1) * 1024 * 1024)
    assert any("limit" in e for e in validate_pdf_file(big))


def test_password_protected_pdf(tmp_path):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Name: Secret Student " * 5)
    path = tmp_path / "locked.pdf"
    doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="abc123", owner_pw="owner456")
    doc.close()
    r = extract_from_pdf(path.read_bytes())
    assert not r.ok and any("password" in e.lower() for e in r.errors)


def test_validate_extraction_finds_mistakes():
    subjects = pd.DataFrame({
        "Semester": [1, 1], "Code": [None, None], "Subject": ["Maths", "Physics"],
        "Marks Obtained": [120, -5], "Max Marks": [100, 100], "Grade": [None, None],
        "Result": [None, None], "Check": ["", ""]})
    info = {"student_name": None, "attendance": 150.0, "cgpa": 11.2}
    messages = [i["message"] for i in validate_extraction(info, subjects)]
    text = " | ".join(messages)
    assert "higher than maximum" in text
    assert "cannot be negative" in text
    assert "Attendance" in text and "CGPA" in text and "name is empty" in text


def test_validate_extraction_accepts_good_data():
    r = run("sample_1_standard_table.pdf")
    errors = [i for i in validate_extraction(r.info, r.subjects) if i["level"] == "error"]
    assert errors == []


def test_to_number():
    assert to_number("78") == 78 and to_number(" 78.5 ") == 78.5 and to_number("1,000") == 1000
    assert to_number("AB") is None and to_number("--") is None and to_number(None) is None