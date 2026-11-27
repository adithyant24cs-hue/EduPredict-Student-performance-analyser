"""
Small Streamlit page to try the PDF extractor (Phase 8 check). The real app comes in Phase 13.

Run from the project root:
    python -m streamlit run scripts/pdf_demo.py
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pdf_extractor import (INFO_FIELDS, INFO_LABELS, SUBJECT_COLUMNS, clean_subjects_df,  # noqa: E402
                               extract_from_pdf, ocr_available, to_number, validate_extraction)

st.set_page_config(page_title="EduPredict - PDF extractor test", page_icon="📄", layout="wide")
st.title("📄 PDF report extractor (test page)")
st.caption("Files are processed in memory only and are not saved. Always check the extracted values.")

use_ocr = st.checkbox("Use OCR for scanned PDFs", value=True, disabled=not ocr_available(),
                      help="Needs Tesseract OCR installed on this computer.")
if not ocr_available():
    st.info("OCR is not available here, so scanned (image) PDFs cannot be read. Normal text PDFs work fine.")

uploaded = st.file_uploader("Upload a student academic report (PDF)", type=["pdf"])

if uploaded is None:
    st.stop()

file_bytes = uploaded.getvalue()
file_key = hashlib.md5(file_bytes).hexdigest() + str(use_ocr)

if st.session_state.get("file_key") != file_key:          # only process a new file once
    with st.spinner("Reading the PDF..."):
        st.session_state["result"] = extract_from_pdf(file_bytes, uploaded.name, use_ocr=use_ocr)
    st.session_state["file_key"] = file_key
    st.session_state["subjects"] = st.session_state["result"].subjects.copy()

result = st.session_state["result"]

if result.errors:
    for message in result.errors:
        st.error(message)
    st.stop()

st.success(f"Read {result.pages} page(s) using {result.method.upper()} extraction.")
for message in result.warnings:
    st.warning(message)

# ---------------------------- student details (editable) ----------------------------
st.subheader("1. Student details (edit anything that is wrong)")
edited_info = {}
columns = st.columns(2)
for i, key in enumerate(INFO_FIELDS):
    value = result.info.get(key)
    shown = "" if value is None else str(value)
    edited_info[key] = columns[i % 2].text_input(INFO_LABELS[key], value=shown, key=f"info_{file_key}_{key}")

numeric_keys = {"semester", "sgpa", "cgpa", "percentage", "attendance", "backlogs"}
info_for_check = {}
for key, text in edited_info.items():
    text = text.strip()
    if not text:
        info_for_check[key] = None
    elif key in numeric_keys:
        number = to_number(text)
        info_for_check[key] = number if number is not None else text      # text -> reported as an error
    else:
        info_for_check[key] = text

# ---------------------------- subject marks (editable) ----------------------------
st.subheader("2. Subject marks (you can edit cells, add rows or delete rows)")
fill_col1, fill_col2 = st.columns([1, 3])
fill_value = fill_col1.number_input("Maximum marks", min_value=0.0, value=0.0, step=10.0,
                                    help="Optional. Fills ONLY the empty 'Max Marks' cells when you click the button.")
if fill_col2.button("Fill empty Max Marks cells with this value") and fill_value > 0:
    df = clean_subjects_df(st.session_state["subjects"])
    df["Max Marks"] = df["Max Marks"].fillna(fill_value)
    st.session_state["subjects"] = df

edited = st.data_editor(
    clean_subjects_df(st.session_state["subjects"]),
    num_rows="dynamic",
    width="stretch",
    key=f"editor_{file_key}",
    column_config={
        "Semester": st.column_config.NumberColumn("Semester", min_value=1, max_value=12, step=1),
        "Marks Obtained": st.column_config.NumberColumn("Marks Obtained", min_value=0.0),
        "Max Marks": st.column_config.NumberColumn("Max Marks", min_value=0.0),
        "Check": st.column_config.TextColumn("Check", disabled=True),
    },
)

# ---------------------------- validation ----------------------------
st.subheader("3. Validation")
issues = validate_extraction(info_for_check, edited)
if not issues:
    st.success("No problems found.")
for issue in issues:
    (st.error if issue["level"] == "error" else st.warning)(issue["message"])

with st.expander("Show the raw text that was read from the PDF"):
    st.text(result.raw_text[:6000])