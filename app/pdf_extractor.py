"""
EduPredict - PDF academic report extractor.

What it does
------------
1. Validates an uploaded PDF (size, real PDF, not password protected, page limit).
2. Reads the text with PyMuPDF. If a page has no text (scanned), it tries OCR (Tesseract).
3. Finds student details (name, ID, course, semester, GPA, CGPA, attendance, ...) using regular expressions.
4. Finds the subject marks table with pdfplumber. If there is no usable table, it falls back to
   reading plain text lines such as "Physics 58 100".
5. Returns everything in an ExtractionResult. Nothing is guessed silently:
   - fields that are not found stay empty (None),
   - uncertain rows are marked in the "Check" column,
   - the user can edit everything in the app before it is used.

Privacy: files are processed in memory only. Nothing is written to disk and nothing is logged.
"""
from __future__ import annotations

import io
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

try:  # new package name
    import pymupdf as fitz
except ImportError:  # older PyMuPDF versions
    import fitz  # type: ignore

import pdfplumber

try:
    import pytesseract
    from PIL import Image
except ImportError:  # OCR is optional
    pytesseract = None  # type: ignore
    Image = None  # type: ignore

# ----------------------------------------------------------------------------
# Settings and constants
# ----------------------------------------------------------------------------
MAX_FILE_MB = 10
MAX_PAGES = 40
MIN_TEXT_CHARS_PER_PAGE = 30   # fewer characters than this per page = probably a scanned page

SUBJECT_COLUMNS = ["Semester", "Code", "Subject", "Marks Obtained", "Max Marks", "Grade", "Result", "Check"]
INFO_FIELDS = ["student_name", "student_id", "course", "semester", "academic_year",
               "sgpa", "cgpa", "percentage", "attendance", "backlogs"]

INFO_LABELS = {
    "student_name": "Student name",
    "student_id": "Student ID / USN",
    "course": "Course / Programme",
    "semester": "Semester",
    "academic_year": "Academic year",
    "sgpa": "SGPA / GPA",
    "cgpa": "CGPA",
    "percentage": "Overall percentage",
    "attendance": "Attendance (%)",
    "backlogs": "Backlogs",
}

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
ABSENT_WORDS = {"AB", "ABS", "ABSENT"}
LABEL_WORDS = (r"usn|roll|reg|register|registration|enrol|enrolment|enrollment|id|course|branch|programme|"
               r"program|semester|sem|year|dob|date|section|class|sgpa|cgpa|gpa|attendance|name")
STOP_SUBJECT = re.compile(
    r"^(grand\s+)?total\b|^sub\s*total\b|sgpa|cgpa|\bgpa\b|percentage|aggregate|attendance|backlog|arrear|"
    r"^result\b|^semester\b|academic\s+year|^name\b|^usn\b|^roll\b|^class\b|^grade\b|^remarks?\b|^credits?\b|"
    r"^date\b|^page\b|^signature|^principal|^controller",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    ok: bool = False
    method: str = "none"                       # "text", "ocr" or "none"
    pages: int = 0
    info: dict = field(default_factory=lambda: {k: None for k in INFO_FIELDS})
    subjects: pd.DataFrame = field(default_factory=lambda: empty_subjects_df())
    semester_gpas: list = field(default_factory=list)   # [{"semester": 1, "sgpa": 8.4}, ...]
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    raw_text: str = ""


def empty_subjects_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Semester": pd.Series(dtype="float64"),
        "Code": pd.Series(dtype="object"),
        "Subject": pd.Series(dtype="object"),
        "Marks Obtained": pd.Series(dtype="float64"),
        "Max Marks": pd.Series(dtype="float64"),
        "Grade": pd.Series(dtype="object"),
        "Result": pd.Series(dtype="object"),
        "Check": pd.Series(dtype="object"),
    })[SUBJECT_COLUMNS]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def to_number(value) -> Optional[float]:
    """Convert '78', '78.5', ' 78 ', '78*' to a float. Anything else (AB, --, NA) gives None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)
    text = str(value).strip().replace(",", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*\*?", text)
    return float(match.group(1)) if match else None


def semester_to_int(text) -> Optional[int]:
    """'3' -> 3, 'V' -> 5, 'III' -> 3. Returns None if it cannot be understood."""
    if text is None:
        return None
    s = str(text).strip().upper()
    if s.isdigit():
        n = int(s)
        return n if 1 <= n <= 12 else None
    return ROMAN.get(s)


def _clean_cell(cell) -> str:
    return re.sub(r"\s+", " ", str(cell).replace("\n", " ")).strip() if cell is not None else ""


def _cut_at_labels(value: str) -> str:
    """'Asha Rao   USN: 123' -> 'Asha Rao' (stops where the next label starts)."""
    value = re.split(r"\s{2,}|\t", value.strip())[0]
    value = re.split(rf"\s+(?=(?:{LABEL_WORDS})\b\s*[:\-.])", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return value.strip(" .,:;-")


def ocr_available() -> bool:
    """True if pytesseract and the Tesseract program are both installed."""
    if pytesseract is None or Image is None:
        return False
    if shutil.which("tesseract") is None:
        windows_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if windows_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(windows_path)
        else:
            return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# 1. File validation
# ----------------------------------------------------------------------------
def validate_pdf_file(file_bytes: bytes, filename: str = "upload.pdf") -> list:
    """Return a list of error messages. An empty list means the file can be processed."""
    errors = []
    if not file_bytes:
        return ["The file is empty."]
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        errors.append(f"The file is {size_mb:.1f} MB. The limit is {MAX_FILE_MB} MB.")
    if b"%PDF" not in file_bytes[:1024]:
        errors.append("This does not look like a real PDF file (wrong file header). "
                      "Make sure you uploaded a PDF, not a renamed image or document.")
        return errors
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        errors.append("The PDF is damaged and could not be opened.")
        return errors
    try:
        if doc.needs_pass or doc.is_encrypted:
            errors.append("The PDF is password protected. Remove the password and upload it again.")
        elif doc.page_count == 0:
            errors.append("The PDF has no pages.")
        elif doc.page_count > MAX_PAGES:
            errors.append(f"The PDF has {doc.page_count} pages. The limit is {MAX_PAGES} pages.")
    finally:
        doc.close()
    return errors


# ----------------------------------------------------------------------------
# 2. Text extraction (normal and OCR)
# ----------------------------------------------------------------------------
def _read_text_pages(file_bytes: bytes) -> list:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        return [page.get_text("text") for page in doc]
    finally:
        doc.close()


def _ocr_pages(file_bytes: bytes) -> list:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texts = []
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))      # about 216 dpi for good OCR
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            texts.append(pytesseract.image_to_string(image, config="--psm 6"))
    finally:
        doc.close()
    return texts


# ----------------------------------------------------------------------------
# 3. Student information
# ----------------------------------------------------------------------------
_NUM = r"(\d{1,3}(?:\.\d{1,3})?)"


def _search(pattern: str, text: str, flags=re.IGNORECASE):
    return re.search(pattern, text, flags)


def extract_info(text: str) -> dict:
    """Find student details in free text. Missing fields stay None."""
    info = {k: None for k in INFO_FIELDS}

    m = _search(r"(?:student\s*name|name\s*of\s*the\s*student|candidate\s*name|name\s*of\s*student|\bname)"
                r"\s*[:\-]\s*([^\n]{2,80})", text)
    if m:
        name = _cut_at_labels(m.group(1))
        if re.search(r"[A-Za-z]{2,}", name) and not re.search(r"\d", name):
            info["student_name"] = name.title() if name.isupper() else name

    m = _search(r"(?:\busn|roll\s*(?:no|number|\.)?|reg(?:ister|istration)?\.?\s*(?:no|number)?|"
                r"student\s*id|enrol(?:l)?ment\s*(?:no|number)?|id\s*no)\s*[:\-.]?\s*([A-Za-z0-9][A-Za-z0-9/\-]{3,24})",
                text)
    if m and any(ch.isdigit() for ch in m.group(1)):
        info["student_id"] = m.group(1).upper()

    m = _search(r"(?:course|programme|program|branch|degree|department)\s*[:\-]\s*([^\n]{2,80})", text)
    if m:
        course = _cut_at_labels(m.group(1))
        if len(course) >= 2:
            info["course"] = course

    sems = _find_semesters(text)
    if sems:
        info["semester"] = sems[0]

    m = _search(r"(?:academic\s*year|a\.?\s?y\.?|session)\s*[:\-]?\s*(20\d{2}\s*[-–/]\s*(?:20)?\d{2})", text) \
        or _search(r"\b(20\d{2}\s*[-–/]\s*(?:20)?\d{2})\b", text)
    if m:
        info["academic_year"] = re.sub(r"\s+", "", m.group(1)).replace("–", "-").replace("/", "-")

    m = _search(rf"\bSGPA\b\s*[:\-=]?\s*{_NUM}", text) or _search(rf"(?<![A-Za-z])GPA\b\s*[:\-=]?\s*{_NUM}", text)
    if m:
        info["sgpa"] = float(m.group(1))

    m = _search(rf"\bCGPA\b\s*[:\-=]?\s*{_NUM}", text) or _search(rf"cumulative\s*gpa\s*[:\-=]?\s*{_NUM}", text)
    if m:
        info["cgpa"] = float(m.group(1))

    text_without_attendance = re.sub(r"attendance[^\n]*", "", text, flags=re.IGNORECASE)
    m = _search(rf"(?:overall\s*)?(?:percentage|aggregate)(?:\s*marks)?\s*[:\-=]?\s*{_NUM}\s*%?",
                text_without_attendance)
    if m:
        info["percentage"] = float(m.group(1))

    m = _search(rf"attendance(?:\s*percentage)?(?:\s*\(?%\)?)?\s*[:\-=]?\s*{_NUM}\s*%?", text)
    if m:
        info["attendance"] = float(m.group(1))

    m = _search(r"(?:backlogs?|arrears?|failed\s*subjects?|no\.?\s*of\s*(?:backlogs?|arrears?))\s*[:\-=]?\s*(\d{1,2})\b",
                text)
    if m:
        info["backlogs"] = int(m.group(1))

    return info


def _find_semesters(text: str) -> list:
    """All semester numbers mentioned in the text, in order of appearance."""
    found = []
    for m in re.finditer(r"\bsem(?:ester)?\s*[:\-]?\s*(\d{1,2}|[IVX]{1,4})\b", text, re.IGNORECASE):
        n = semester_to_int(m.group(1))
        if n and n not in found:
            found.append(n)
    for m in re.finditer(r"\b(\d)(?:st|nd|rd|th)\s*sem(?:ester)?\b", text, re.IGNORECASE):
        n = semester_to_int(m.group(1))
        if n and n not in found:
            found.append(n)
    return found


def extract_semester_gpas(text: str) -> list:
    results, seen = [], set()
    pattern = (r"\bsem(?:ester)?\s*[-:]?\s*(\d{1,2}|[IVX]{1,4})\s*(?:SGPA|GPA)\s*[:\-=]?\s*"
               r"(\d{1,2}(?:\.\d{1,3})?)")
    for m in re.finditer(pattern, text, re.IGNORECASE):
        sem = semester_to_int(m.group(1))
        if sem and sem not in seen:
            seen.add(sem)
            results.append({"semester": sem, "sgpa": float(m.group(2))})
    return results


# ----------------------------------------------------------------------------
# 4a. Subject marks from tables (pdfplumber)
# ----------------------------------------------------------------------------
def _header_roles(cells: list) -> dict:
    """Work out which column is which from header names. Returns {role: column_index}."""
    roles = {}
    for idx, cell in enumerate(cells):
        n = re.sub(r"[^a-z]", "", _clean_cell(cell).lower())
        if not n:
            continue
        if any(k in n for k in ("max", "outof", "fullmarks")):
            role = "max"
        elif "code" in n:
            role = "code"
        elif "internal" in n or n in ("ia", "cie", "sessional"):
            role = "internal"
        elif "external" in n or n in ("see", "ee"):
            role = "external"
        elif "total" in n or "obtained" in n or "secured" in n or n in ("marks", "score"):
            role = "obtained"
        elif "grade" in n and "point" not in n:
            role = "grade"
        elif "result" in n or "status" in n or "remark" in n:
            role = "result"
        elif n in ("sem", "semester"):
            role = "semester"
        elif any(k in n for k in ("subject", "course", "paper", "title", "description", "name")):
            role = "subject"
        else:
            continue
        roles.setdefault(role, idx)
    return roles


def _is_valid_header(roles: dict) -> bool:
    has_marks = "obtained" in roles or ("internal" in roles and "external" in roles)
    return "subject" in roles and has_marks


def _rows_from_table(table: list, default_semester: Optional[int]) -> list:
    rows_out = []
    if not table or len(table) < 2:
        return rows_out
    header_index, roles = None, {}
    for i, row in enumerate(table[:3]):              # header is within the first 3 rows
        r = _header_roles(row)
        if _is_valid_header(r):
            header_index, roles = i, r
            break
    if header_index is None:
        return rows_out

    def cell(row, role):
        idx = roles.get(role)
        return _clean_cell(row[idx]) if idx is not None and idx < len(row) else ""

    for row in table[header_index + 1:]:
        subject = cell(row, "subject")
        if not subject or STOP_SUBJECT.search(subject) or re.fullmatch(r"[\d\W]+", subject):
            continue
        notes = []
        obtained_text = cell(row, "obtained")
        obtained = to_number(obtained_text)
        internal, external = to_number(cell(row, "internal")), to_number(cell(row, "external"))
        result = cell(row, "result") or None
        if obtained is None and "obtained" not in roles and internal is not None and external is not None:
            obtained = internal + external
            notes.append("Total calculated as internal + external")
        if obtained is None:
            if obtained_text.upper() in ABSENT_WORDS or "AB" in {cell(row, "external").upper(), cell(row, "internal").upper()}:
                notes.append("Marked absent in the PDF")
                result = result or "Absent"
            notes.append("Marks not read - please enter")
        max_marks = to_number(cell(row, "max"))
        sem = semester_to_int(cell(row, "semester")) or default_semester
        rows_out.append({
            "Semester": float(sem) if sem else None,
            "Code": cell(row, "code") or None,
            "Subject": subject,
            "Marks Obtained": obtained,
            "Max Marks": max_marks,
            "Grade": cell(row, "grade") or None,
            "Result": result,
            "Check": "; ".join(notes) if notes else "",
        })
    return rows_out


def _table_rows_for_page(plumber_page, default_semester: Optional[int]) -> list:
    """Try ruled-table detection first, then whitespace-based detection."""
    strategies = [
        {},                                                                  # lines (default)
        {"vertical_strategy": "text", "horizontal_strategy": "text"},        # aligned text columns
    ]
    for settings in strategies:
        try:
            tables = plumber_page.extract_tables(settings) if settings else plumber_page.extract_tables()
        except Exception:
            continue
        rows = []
        for table in tables or []:
            rows.extend(_rows_from_table(table, default_semester))
        if rows:
            return rows
    return []


# ----------------------------------------------------------------------------
# 4b. Subject marks from plain text lines (fallback, also used after OCR)
# ----------------------------------------------------------------------------
_CODE = r"(?:[A-Z]{2,5}\d{2,4}[A-Z]?|\d{2}[A-Z]{2,5}\d{1,3})"
_LINE = re.compile(
    rf"^(?:(?P<code>{_CODE})\s+)?"
    r"(?P<name>[A-Za-z][A-Za-z0-9 &/\-\(\)\.,']*?[A-Za-z\)])\s+"
    r"(?P<n1>\d{1,3}(?:\.\d+)?)"
    r"(?:\s*(?:/|out\s+of|of)?\s*(?P<n2>\d{1,3}(?:\.\d+)?))?"
    r"\s*(?P<tail>[A-Za-z][+\-]?)?\s*$"
)


def _rows_from_text(text: str, default_semester: Optional[int], source_note: str) -> list:
    rows_out = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if len(line) < 5 or STOP_SUBJECT.search(line):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        if STOP_SUBJECT.search(name) or len(re.findall(r"[A-Za-z]", name)) < 3:
            continue
        n1 = float(m.group("n1"))
        n2 = float(m.group("n2")) if m.group("n2") else None
        notes = [source_note]
        if n2 is not None:
            if n2 <= 0 or n2 < n1:
                continue                      # cannot be "obtained, maximum"; skip rather than guess
            obtained, max_marks = n1, n2
        else:
            obtained, max_marks = n1, None
            notes.append("Only one number found; treated as marks obtained")
        tail = (m.group("tail") or "").upper()
        rows_out.append({
            "Semester": float(default_semester) if default_semester else None,
            "Code": m.group("code"),
            "Subject": name,
            "Marks Obtained": obtained,
            "Max Marks": max_marks,
            "Grade": tail if tail and tail not in ("P", "F") else None,
            "Result": {"P": "Pass", "F": "Fail"}.get(tail),
            "Check": "; ".join(notes),
        })
    return rows_out


def _page_semester(page_text: str) -> Optional[int]:
    """The semester of a page, but only if the page mentions exactly one semester."""
    sems = _find_semesters(page_text)
    return sems[0] if len(sems) == 1 else None


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def extract_from_pdf(file_bytes: bytes, filename: str = "upload.pdf", use_ocr: bool = True) -> ExtractionResult:
    """Read a PDF academic report. Never raises: problems are returned in result.errors / result.warnings."""
    result = ExtractionResult()
    result.errors = validate_pdf_file(file_bytes, filename)
    if result.errors:
        return result

    try:
        page_texts = _read_text_pages(file_bytes)
        result.pages = len(page_texts)
        total_chars = sum(len(t.strip()) for t in page_texts)
        scanned = total_chars < MIN_TEXT_CHARS_PER_PAGE * max(1, result.pages)
        result.method = "text"

        if scanned:
            if not use_ocr:
                result.warnings.append("This PDF looks scanned (no readable text) and OCR is switched off.")
            elif not ocr_available():
                result.errors.append(
                    "This PDF looks scanned (it is an image) and OCR is not available on this computer. "
                    "Install Tesseract OCR (see the project README), or enter the marks manually."
                )
                return result
            else:
                page_texts = _ocr_pages(file_bytes)
                result.method = "ocr"
        result.raw_text = "\n\n".join(page_texts)
        if len(result.raw_text.strip()) < MIN_TEXT_CHARS_PER_PAGE:
            result.errors.append("No readable text was found in this PDF (it may be blank).")
            return result
        if result.method == "ocr":
            result.warnings.append("Text was read with OCR from a scanned PDF. OCR can misread digits, "
                                   "so check every value carefully.")

        # --- student details and semester GPAs ---
        result.info = extract_info(result.raw_text)
        result.semester_gpas = extract_semester_gpas(result.raw_text)
        if len(result.semester_gpas) >= 2:
            result.info["sgpa"] = None        # several semesters: one SGPA value would be misleading
        several_semesters = _find_semesters(result.raw_text)
        if len(several_semesters) >= 2:
            result.info["semester"] = None    # the semester is stored per subject row instead
            result.warnings.append("Several semesters were found (" + ", ".join(map(str, several_semesters)) +
                                   "). The semester is set for each subject row - check them.")

        # --- subject marks, page by page ---
        rows = []
        plumber = None
        if result.method == "text":
            try:
                plumber = pdfplumber.open(io.BytesIO(file_bytes))
            except Exception:
                result.warnings.append("The table reader could not open this PDF; using plain text only.")
        try:
            for i, page_text in enumerate(page_texts):
                sem = _page_semester(page_text) or result.info.get("semester")
                page_rows = []
                if plumber is not None and i < len(plumber.pages):
                    page_rows = _table_rows_for_page(plumber.pages[i], sem)
                if not page_rows:
                    note = "Read from OCR text - verify" if result.method == "ocr" else "Read from plain text - verify"
                    page_rows = _rows_from_text(page_text, sem, note)
                rows.extend(page_rows)
        finally:
            if plumber is not None:
                plumber.close()

        subjects = pd.DataFrame(rows, columns=SUBJECT_COLUMNS) if rows else empty_subjects_df()
        subjects = clean_subjects_df(subjects).drop_duplicates(
            subset=["Semester", "Subject", "Marks Obtained"], keep="first").reset_index(drop=True)
        result.subjects = subjects

        # --- warnings the user should see ---
        if subjects.empty:
            result.warnings.append("No subject marks table was recognised. You can add subjects manually in the table.")
        else:
            missing_max = int(subjects["Max Marks"].isna().sum())
            if missing_max:
                result.warnings.append(f"Maximum marks were not found for {missing_max} subject(s). "
                                       "Enter them in the table - they are not assumed.")
        skip_fields = {"sgpa"} | ({"semester"} if len(several_semesters) >= 2 else set())
        not_found = [INFO_LABELS[k] for k in INFO_FIELDS if result.info.get(k) is None and k not in skip_fields]
        if not_found:
            result.warnings.append("Not found in the PDF: " + ", ".join(not_found) + ". Fill them in if you know them.")
        result.ok = True
    except Exception as exc:                         # last safety net - never crash the app
        result.errors.append(f"Unexpected problem while reading the PDF ({type(exc).__name__}). "
                             "Try another file or enter the data manually.")
        result.ok = False
    return result


# ----------------------------------------------------------------------------
# Cleaning and validation (also used after the user edits the table)
# ----------------------------------------------------------------------------
def clean_subjects_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure the table has the right columns and number types. Bad numbers become empty (NaN)."""
    out = df.copy() if df is not None else empty_subjects_df()
    for col in SUBJECT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[SUBJECT_COLUMNS]
    for col in ("Semester", "Marks Obtained", "Max Marks"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("Code", "Subject", "Grade", "Result", "Check"):
        out[col] = out[col].astype("object").where(out[col].notna(), None)
    return out


def validate_extraction(info: dict, subjects: pd.DataFrame) -> list:
    """Check values for mistakes. Returns [{"level": "error"|"warning", "message": str}, ...]."""
    issues = []

    def add(level, message):
        issues.append({"level": level, "message": message})

    def number_in_range(key, label, low, high, level="error", hint=""):
        value = info.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            add("error", f"{label} '{value}' is not a number.")
            return
        if not (low <= v <= high):
            add(level, f"{label} {v:g} is outside the expected range {low:g}-{high:g}. {hint}".strip())

    number_in_range("sgpa", "SGPA", 0, 10, hint="Check the value or your institution's grading scale.")
    number_in_range("cgpa", "CGPA", 0, 10, hint="Check the value or your institution's grading scale.")
    number_in_range("percentage", "Percentage", 0, 100)
    number_in_range("attendance", "Attendance", 0, 100)
    number_in_range("semester", "Semester", 1, 12)
    number_in_range("backlogs", "Backlogs", 0, 50)

    if not info.get("student_name"):
        add("warning", "Student name is empty.")

    if subjects is None or subjects.empty:
        add("warning", "There are no subjects in the table.")
        return issues

    df = clean_subjects_df(subjects)
    for i, row in df.iterrows():
        label = f"Row {i + 1} ({row['Subject'] or 'no subject name'})"
        if not row["Subject"]:
            add("error", f"Row {i + 1}: subject name is missing.")
        obtained, maximum = row["Marks Obtained"], row["Max Marks"]
        if pd.isna(obtained):
            add("warning", f"{label}: marks obtained are missing.")
        elif obtained < 0:
            add("error", f"{label}: marks obtained cannot be negative.")
        if not pd.isna(maximum) and maximum <= 0:
            add("error", f"{label}: maximum marks must be greater than 0.")
        if not pd.isna(obtained) and not pd.isna(maximum) and maximum > 0 and obtained > maximum:
            add("error", f"{label}: marks obtained ({obtained:g}) are higher than maximum marks ({maximum:g}).")
    duplicated = df[df.duplicated(subset=["Semester", "Subject"], keep=False) & df["Subject"].notna()]
    if not duplicated.empty:
        add("warning", "Some subjects appear twice in the same semester: "
                       + ", ".join(sorted(set(duplicated["Subject"].astype(str)))) + ".")
    if df["Max Marks"].isna().any():
        add("warning", "Maximum marks are missing for some subjects, so percentages cannot be calculated for them.")
    return issues