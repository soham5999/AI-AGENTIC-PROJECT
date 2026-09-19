"""Safe organization dataset upload and validation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from utils.preprocessing import FEATURES

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".docx"}

@dataclass
class UploadReport:
    frame: pd.DataFrame | None
    file_name: str
    size_bytes: int
    extension: str
    errors: list[str]
    warnings: list[str]
    source: str = "organization upload"

    @property
    def valid(self) -> bool:
        return self.frame is not None and not self.errors


def _docx_to_frame(raw: bytes) -> pd.DataFrame:
    from docx import Document
    document = Document(BytesIO(raw))
    tables = document.tables
    if not tables:
        raise ValueError("The DOCX file does not contain a table. Upload a tabular document.")
    table = tables[0]
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    if len(rows) < 2:
        raise ValueError("The DOCX table must contain a header row and at least one data row.")
    return pd.DataFrame(rows[1:], columns=rows[0])


def read_uploaded_file(uploaded_file: Any) -> UploadReport:
    """Read CSV, Excel, or the first table in a DOCX without crashing the UI."""
    name = getattr(uploaded_file, "name", "uploaded_file")
    raw = uploaded_file.getvalue()
    extension = Path(name).suffix.lower()
    errors: list[str] = []
    warnings: list[str] = []
    frame: pd.DataFrame | None = None

    if extension not in ALLOWED_EXTENSIONS:
        errors.append("Unsupported file type. Upload CSV, XLSX, XLS, or a DOCX table.")
        return UploadReport(None, name, len(raw), extension, errors, warnings)
    if not raw:
        errors.append("The uploaded file is empty.")
        return UploadReport(None, name, 0, extension, errors, warnings)

    try:
        if extension == ".csv":
            frame = pd.read_csv(BytesIO(raw))
        elif extension == ".xlsx":
            frame = pd.read_excel(BytesIO(raw), engine="openpyxl")
        elif extension == ".xls":
            frame = pd.read_excel(BytesIO(raw), engine="xlrd")
        else:
            frame = _docx_to_frame(raw)
    except Exception as exc:
        errors.append(f"Could not read this file. Check that it is not corrupted: {type(exc).__name__}.")
        return UploadReport(None, name, len(raw), extension, errors, warnings)

    if frame is None or frame.empty:
        errors.append("The dataset contains no rows.")
        return UploadReport(frame, name, len(raw), extension, errors, warnings)

    frame.columns = [str(column).strip() for column in frame.columns]
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        errors.append(f"Duplicate column names found: {duplicate_columns}.")

    missing_features = [feature for feature in FEATURES if feature not in frame.columns]
    if missing_features:
        warnings.append(
            "Prediction requires these sample-schema fields: " + ", ".join(missing_features)
        )
    null_columns = frame.columns[frame.isna().any()].tolist()
    if null_columns:
        warnings.append("Missing values detected in: " + ", ".join(map(str, null_columns)))

    return UploadReport(frame, name, len(raw), extension, errors, warnings)


def prepare_for_prediction(frame: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """Return numeric model features and row-level validation errors."""
    if frame is None:
        return None, ["No dataset is loaded."]
    missing = [feature for feature in FEATURES if feature not in frame.columns]
    if missing:
        return None, ["Cannot predict because required columns are missing: " + ", ".join(missing)]
    converted = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    invalid = int(converted.isna().any(axis=1).sum())
    if invalid:
        return None, [f"{invalid} row(s) contain missing or non-numeric model values."]
    return converted, []


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"
