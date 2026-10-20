"""The upload template (one Excel workbook) and the reader for coded tables researchers upload.

    build()            -> bytes of construct_audit_template.xlsx: Instructions, Variables and Correlations sheets
    read(files)        -> (variables, correlations, notes) from the filled workbook or from two CSV files;
                          raises UploadError with a plain-language message when something required is missing
"""
import io, re
from pathlib import Path
import pandas as pd

from . import schema

# (sheet, column, required?, what to put, allowed values)
COLUMNS = [
    ("Variables", "article", "required", "An ID for the article, e.g. 'Smith 2019'. Use the same ID in both sheets.", ""),
    ("Variables", "study", "required", "1, unless the article reports separate samples with separate correlation tables (then 1, 2, ...).", ""),
    ("Variables", "var", "required", "The variable's number in the article's correlation table.", ""),
    ("Variables", "name", "required", "The variable's name as the authors wrote it.", ""),
    ("Variables", "description", "recommended", "How it was measured, in a sentence. The model uses it to name constructs and measures.", ""),
    ("Variables", "data_type", "recommended", "Who or what produced the score.", " | ".join(schema.DATA_TYPES)),
    ("Variables", "reliability", "optional", "Cronbach's alpha or similar, between 0 and 1.", ""),
    ("Variables", "construct", "optional", "Your construct for this variable. Leave empty to let the model draft a codebook.", ""),
    ("Variables", "measure", "optional", "Your name for the measure, written the same way in every article. Leave empty to let the model name it.", ""),
    ("Variables", "measure_form", "optional", "The form of the measure.", " | ".join(schema.FORMS)),
    ("Variables", "instrument", "optional", "The published scale or named measure the article cites, e.g. 'Allen & Meyer 1990'.", ""),
    ("Correlations", "article", "required", "The article ID, as in the Variables sheet.", ""),
    ("Correlations", "study", "required", "The study number, as in the Variables sheet.", ""),
    ("Correlations", "var_a", "required", "The number of the first variable (its 'var' in the Variables sheet).", ""),
    ("Correlations", "var_b", "required", "The number of the second variable.", ""),
    ("Correlations", "r", "required", "The correlation, between -1 and 1.", ""),
    ("Correlations", "n", "optional", "The study's sample size.", ""),
    ("Codebook", "variable name", "optional sheet", "A variable name as authors write it. Every variable with this name gets the construct on the right.", ""),
    ("Codebook", "construct", "optional sheet", "The construct it belongs to. Use this sheet or the 'construct' column in Variables, not both. "
                                               "Leave the sheet empty to let the model draft a codebook.", ""),
]
REQUIRED = {sheet: [c for s, c, req, *_ in COLUMNS if s == sheet and req == "required"] for sheet in ("Variables", "Correlations")}
EX_CODEBOOK = [("Affective commitment", "Organisational commitment"), ("Organizational commitment", "Organisational commitment"),
               ("Task performance", "Job performance"), ("Tenure", "Tenure"), ("ROA", "Firm performance"),
               ("Tobin's q", "Firm performance"), ("Board independence", "Board independence")]

# Example rows from two different fields, so the template does not suggest the tool is for one literature
EX_VARS = [
    ("Smith 2019", 1, 1, "Affective commitment", "Six items, 7-point agreement scale", "survey", 0.86, "", "", "multi-item scale", "Allen & Meyer 1990"),
    ("Smith 2019", 1, 2, "Task performance", "Supervisor rated each employee on four items", "other-report", 0.91, "", "", "multi-item scale", ""),
    ("Smith 2019", 1, 3, "Tenure", "Years with the organisation, from HR records", "archival", "", "", "", "time or age", ""),
    ("Lee 2021", 1, 1, "ROA", "Net income divided by total assets", "archival", "", "", "return on assets", "ratio or share", ""),
    ("Lee 2021", 1, 2, "Tobin's q", "Market value of the firm over the replacement cost of its assets", "archival", "", "", "Tobin's q", "ratio or share", "Tobin's q"),
    ("Lee 2021", 1, 3, "Board independence", "Share of outside directors on the board", "archival", "", "", "", "ratio or share", ""),
]
EX_CORS = [("Smith 2019", 1, 1, 2, 0.31, 214), ("Smith 2019", 1, 1, 3, 0.12, 214), ("Smith 2019", 1, 2, 3, 0.05, 214),
           ("Lee 2021", 1, 1, 2, 0.42, 1250), ("Lee 2021", 1, 1, 3, -0.08, 1250), ("Lee 2021", 1, 2, 3, 0.03, 1250)]


class UploadError(ValueError):
    pass


def build():
    var_cols = [c for s, c, *_ in COLUMNS if s == "Variables"]
    cor_cols = [c for s, c, *_ in COLUMNS if s == "Correlations"]
    guide = pd.DataFrame([dict(sheet=s, column=c, needed=req, what_to_put=w, allowed_values=a) for s, c, req, w, a in COLUMNS])
    intro = pd.DataFrame({"How to use this template": [
        "Fill in one row per variable (Variables sheet) and one row per correlation (Correlations sheet).",
        "Replace the example rows (Smith 2019, Lee 2021) with your own.",
        "Optional: the Codebook sheet says which construct each variable name belongs to. Delete its rows if you want the model to draft one.",
        "Keep the column names in row 1 exactly as they are. Optional columns can stay empty.",
        "Each article's variables are numbered as in its correlation table; the correlations refer to those numbers.",
        "Upload the filled-in file in the app (step 1). Nothing is stored after your session ends.", ""]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        intro.to_excel(w, sheet_name="Instructions", index=False)
        guide.to_excel(w, sheet_name="Instructions", index=False, startrow=len(intro) + 2)
        pd.DataFrame(EX_VARS, columns=var_cols).to_excel(w, sheet_name="Variables", index=False)
        pd.DataFrame(EX_CORS, columns=cor_cols).to_excel(w, sheet_name="Correlations", index=False)
        pd.DataFrame(EX_CODEBOOK, columns=["variable name", "construct"]).to_excel(w, sheet_name="Codebook", index=False)
        for ws in w.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(70, max(10, *(len(str(c.value or "")) + 2 for c in col)))
    return buf.getvalue()


def _clean(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df.dropna(how="all").fillna("").astype(str).apply(lambda s: s.str.strip())


def _check(df, sheet):
    missing = [c for c in REQUIRED[sheet] if c not in df.columns]
    if missing:
        raise UploadError(f"The {sheet} table has no column {', '.join(repr(c) for c in missing)}. "
                          f"Required columns: {', '.join(REQUIRED[sheet])}. The template shows the layout.")


def _whole(s):
    """Excel turns 1 into '1.0'; IDs must match across sheets."""
    return s.str.replace(r"^(\d+)\.0$", r"\1", regex=True)


def read_codebook(files):
    """The Codebook sheet of an uploaded workbook, cleaned (columns label, construct), or None if absent or empty."""
    from . import codebook
    xl = [f for f in (files or []) if str(getattr(f, "name", "")).lower().endswith((".xlsx", ".xls"))]
    if not xl:
        return None
    if hasattr(xl[0], "seek"):
        xl[0].seek(0)
    sheets = {k.strip().lower(): v for k, v in pd.read_excel(xl[0], sheet_name=None, dtype=str).items()}
    if "codebook" not in sheets or sheets["codebook"].dropna(how="all").empty:
        return None
    try:
        cb = codebook.load(sheets["codebook"])
    except ValueError as e:
        raise UploadError(f"Codebook sheet: {e}.") from None
    example = codebook.load(pd.DataFrame(EX_CODEBOOK, columns=["variable name", "construct"]))
    if set(zip(cb.label, cb.construct)) <= set(zip(example.label, example.construct)):
        return None                    # only the template's example rows: treat the sheet as empty, so the model drafts a codebook
    return cb if len(cb) else None


def _csv(f):
    """A CSV as Excel writes it anywhere: comma or semicolon (French/German Excel), UTF-8 or Windows encoding."""
    raw = f.getvalue() if hasattr(f, "getvalue") else Path(f).read_bytes()
    if not raw.strip():
        raise UploadError(f"The file '{getattr(f, 'name', f)}' is empty.")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    first = text.splitlines()[0]
    sep = ";" if first.count(";") > first.count(",") else ("\t" if first.count("\t") > first.count(",") else ",")
    try:
        return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    except Exception as e:
        raise UploadError(f"The file '{getattr(f, 'name', f)}' could not be read as a table ({str(e)[:120]}).") from None


EXAMPLE_ARTICLES = {a for a, *_ in EX_VARS}


def _number(sr):
    """'−0,31**' -> '-0.31': Unicode minus first, then decimal comma, then significance stars and spaces."""
    sr = sr.str.replace("−", "-", regex=False).str.replace("–", "-", regex=False).str.replace(r"[*†‡\s]", "", regex=True)
    return sr.str.replace(r"^(-?\d*),(\d+)$", r"\1.\2", regex=True)


def read(files):
    """files: the uploaded workbook, or two CSV files (in any order). Returns (variables, correlations, notes)."""
    files = list(files or [])
    xl = [f for f in files if str(getattr(f, "name", "")).lower().endswith((".xlsx", ".xls"))]
    if xl:
        if hasattr(xl[0], "seek"):
            xl[0].seek(0)
        try:
            sheets = pd.read_excel(xl[0], sheet_name=None, dtype=str)
        except Exception:
            raise UploadError("The Excel file could not be opened. Save it again from Excel as .xlsx and upload it again.") from None
        by = {k.strip().lower(): v for k, v in sheets.items()}
        if "variables" not in by or "correlations" not in by:
            raise UploadError("The workbook needs a sheet called 'Variables' and one called 'Correlations'. The template shows the layout.")
        v, c = _clean(by["variables"]), _clean(by["correlations"])
    else:
        if len(files) != 2:
            raise UploadError("Upload the filled-in template, or two CSV files: the variables table and the correlations table.")
        a, b = (_clean(_csv(f)) for f in files)
        v, c = (b, a) if "var_a" in a.columns else (a, b)
    _check(v, "Variables"); _check(c, "Correlations")
    if v.empty or c.empty:
        raise UploadError(f"The {'Variables' if v.empty else 'Correlations'} table has no rows. Fill in the template with your data.")
    for col in ("study", "var"):
        v[col] = _whole(v[col])
    for col in ("study", "var_a", "var_b"):
        c[col] = _whole(c[col])
    notes = []
    left = EXAMPLE_ARTICLES & set(v.article)
    if left:
        notes.append(f"The template's example rows ({', '.join(sorted(left))}) are still in the file. Delete them, or they will be audited with your data.")
    c["r"] = _number(c["r"])
    rel = _number(v.get("reliability", pd.Series("", index=v.index)))
    num = pd.to_numeric(rel, errors="coerce")
    pct = (num > 1) & (num <= 100)
    if pct.any():
        notes.append(f"{int(pct.sum())} reliabilities above 1 were read as percentages (85 -> 0.85).")
        rel = rel.where(~pct, (num / 100).astype(str))
    low = pd.to_numeric(rel, errors="coerce") <= 0.3
    if low.any():
        notes.append(f"{int(low.sum())} reliabilities of 0.30 or less were left out as implausible.")
    v["reliability"] = rel
    bad_type = ~v.get("data_type", pd.Series("", index=v.index)).isin(list(schema.DATA_TYPES) + [""])
    if bad_type.any():
        notes.append(f"{int(bad_type.sum())} variables have a data_type outside {', '.join(schema.DATA_TYPES)}; they were set to 'unknown'.")
    try:
        V = schema.load_variables(v)
    except ValueError as e:
        raise UploadError(f"Variables: {e}. Each (article, study, var) must appear once.") from None
    n0 = len(c)
    C = schema.load_correlations(c)
    if len(C) < n0:
        notes.append(f"{n0 - len(C)} correlations were left out: not a number, outside -1 to 1, or a variable paired with itself.")
    # the same pair twice (1-2 and 2-1, or repeated) would be counted twice
    key = pd.DataFrame({"a": C.article, "s": C.study, "x": C[["var_a", "var_b"]].min(axis=1), "y": C[["var_a", "var_b"]].max(axis=1)})
    dup = key.duplicated()
    if dup.any():
        notes.append(f"{int(dup.sum())} correlations appear twice for the same pair of variables; only the first was kept.")
        C = C[~dup.to_numpy()].reset_index(drop=True)
    known = set(zip(V.article, V.study, V["var"]))
    orphan = sum(1 for r in C.itertuples() if (r.article, r.study, r.var_a) not in known or (r.article, r.study, r.var_b) not in known)
    if orphan:
        notes.append(f"{orphan} correlations refer to a variable that is not in the Variables table; they cannot be used.")
    if C.empty:
        raise UploadError("None of the correlations could be used. Check the r column (numbers between -1 and 1) and the variable numbers.")
    if (V.measure.fillna("") == "").all():
        notes.append("The tables do not name the measures. Add a model key in step 2 so the model can name them from the descriptions; "
                     "without one, the check of measures within a construct (jingle) is skipped.")
    return V, C, notes
