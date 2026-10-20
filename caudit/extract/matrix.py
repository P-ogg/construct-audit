"""Transcribed matrix rows -> one cell per pair. Ported from pipeline3/review.py.

The model transcribes rows and says where each row starts; which two variables a cell belongs to is arithmetic,
done here. Asked to pair variables itself while reading a 120-cell rotated matrix, the model attached real values
to the wrong pairs, an error nothing downstream could catch.
"""
from .numbers import canon_number


def _int(v, default=None):
    s = str(v if v is not None else "").strip()
    return int(s) if s.lstrip("-").isdigit() else default


def expand_rows(study):
    """[(var_row, var_col, value_as_printed)]; blank strings keep a cell's place; diagonal cells are kept for the self-pair check."""
    out = []
    for row in study.get("matrix_rows") or []:
        var = _int(row.get("variable"))
        start = _int(row.get("first_column"), 1)
        if var is None:
            continue
        for i, v in enumerate(row.get("values") or []):
            if canon_number(v):
                out.append((var, start + i, v))
    return out


def cells(study):
    """{(a, b): value} with a < b; self-pairs and conflicting duplicates reported separately."""
    got, self_pairs, conflicts = {}, [], []
    for a, b, v in expand_rows(study):
        if a == b:
            self_pairs.append((a, v)); continue
        key = (min(a, b), max(a, b))
        if key in got and canon_number(got[key]) != canon_number(v):
            conflicts.append((key, got[key], v)); continue
        got[key] = v
    return got, self_pairs, conflicts
