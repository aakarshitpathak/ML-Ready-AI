"""
modules/data_loader.py — MODULE 3
Reads the uploaded file into a Pandas DataFrame.
Supports: CSV, Excel (.xlsx/.xls), JSON, XML, HTML
"""

import pandas as pd

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'xml', 'html', 'htm'}

# Real-world CSVs are frequently NOT UTF-8 (Excel on Windows exports as
# cp1252/latin-1 by default, and plenty of public datasets - like the
# classic UCI SMS Spam dataset - ship this way). Try UTF-8 first since
# it's the modern default, then fall back through the encodings that
# cover the vast majority of "real" files. latin-1 is listed last
# because it can decode ANY byte sequence without raising - so it's a
# safe final fallback rather than something we want to reach for first.
CSV_ENCODING_FALLBACKS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded filename has an allowed extension.
    Returns True if allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _read_csv_with_encoding_fallback(filepath: str) -> pd.DataFrame:
    """
    Try reading a CSV with a sequence of common encodings, since a
    'UnicodeDecodeError' almost always just means the file isn't UTF-8
    - not that the file is corrupt or unreadable.
    """
    last_error = None

    for encoding in CSV_ENCODING_FALLBACKS:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    # Every fallback failed (very unlikely, since latin-1 never raises
    # UnicodeDecodeError) - re-raise the original error.
    raise last_error


def load_dataframe(filepath: str) -> pd.DataFrame:
    """
    Read a file from 'filepath' and return a Pandas DataFrame.

    Supported formats:
      .csv          → pd.read_csv()      (with automatic encoding fallback)
      .xlsx / .xls  → pd.read_excel()   (needs openpyxl / xlrd)
      .json         → pd.read_json()
      .xml          → pd.read_xml()      (needs lxml)
      .html / .htm  → pd.read_html()[0]  (returns a list; we take first table)

    Raises:
      ValueError   — if the extension is not supported
      Exception    — passes through Pandas parse errors
    """
    ext = filepath.rsplit('.', 1)[1].lower()

    if ext == 'csv':
        df = _read_csv_with_encoding_fallback(filepath)

    elif ext in ('xlsx', 'xls'):
        df = pd.read_excel(filepath)

    elif ext == 'json':
        df = pd.read_json(filepath)

    elif ext == 'xml':
        df = pd.read_xml(filepath)

    elif ext in ('html', 'htm'):
        # read_html returns a list of tables; use the first one
        tables = pd.read_html(filepath)
        if not tables:
            raise ValueError("No HTML table found in the uploaded file.")
        df = tables[0]

    else:
        raise ValueError(f"Unsupported file format: .{ext}")

    return df

