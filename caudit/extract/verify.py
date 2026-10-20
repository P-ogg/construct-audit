"""Mechanical checks on what the coders report. Ported from pipeline3/review.py.

Quotes: consume the quote left to right, taking the longest in-order run at or after the previous one. Two-column
pages interleave sentences in extracted text, so a contiguity test failed 12 of 107 genuine quotes. Measured:
genuine quotes cover 81-100%, invented ones 0-12%. Numbers in a quote are checked separately.

Correlations: checked against the numbers on the page that holds the matrix. Across a whole article 26-90% of all
two-decimal values appear somewhere, so a whole-article check catches almost nothing; on the located page about 11%.
Flags are advice for a human ("check the page"), never automatic rejection.
"""
import re
from .numbers import canon_number, numbers_of
from .pdf import PAGE_MARKER, text_pages

QUOTE_MIN_RUN, QUOTE_MIN_COVERAGE = 15, 0.70
MATRIX_PAGE_SPAN, MIN_VALUES_TO_LOCATE = 2, 4


def squash(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def haystack(text):
    return squash(PAGE_MARKER.sub(" ", text))


def quote_runs(q, hay, min_run=QUOTE_MIN_RUN):
    i = cursor = covered = 0
    while i < len(q):
        lo, hi, best, best_at = 1, len(q) - i, 0, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            at = hay.find(q[i:i + mid], cursor)
            if at >= 0:
                best, best_at, lo = mid, at, mid + 1
            else:
                hi = mid - 1
        if best == 0:
            i += 1; continue
        if best >= min_run:
            covered += best; cursor = best_at + best
        i += best
    return covered / len(q) if q else 0.0


def digits_present(raw, hay):
    for g in re.findall(r"\d{2,}", re.sub(r"[,\s]", "", raw or "")):
        if g not in hay:
            return g
    return None


def _check_once(raw, hay):
    q = squash(raw)
    if not q:
        return "fail", 0.0, None
    if q in hay:
        return "ok", 1.0, None
    missing = digits_present(raw, hay); cov = quote_runs(q, hay)
    if missing is not None:
        return "fail", cov, missing
    return ("loose" if cov >= QUOTE_MIN_COVERAGE else "fail"), cov, None


def check_quote(raw, hay):
    """(verdict ok|loose|fail, coverage, missing digits). Some publishers map '=' to a glyph extracted as '5'."""
    first = _check_once(raw, hay)
    if first[0] == "ok" or "=" not in (raw or ""):
        return first
    second = _check_once(raw.replace("=", "5"), hay)
    rank = {"fail": 0, "loose": 1, "ok": 2}
    return max(first, second, key=lambda r: (rank[r[0]], r[1]))


def locate_matrix(values, text):
    """(signed, unsigned, page_label) of the page(s) whose numbers best cover the reported values."""
    by_page = [(n, *numbers_of(body)) for n, body in text_pages(text)]
    whole = numbers_of(PAGE_MARKER.sub(" ", text))
    cov = lambda s, u: sum(1 for v in values if v in s or v.lstrip("-") in u)
    if len(values) < MIN_VALUES_TO_LOCATE or not by_page:
        return whole[0], whole[1], "whole article"
    pages, sig, uns = [], set(), set()
    for n, s, u in sorted(by_page, key=lambda p: -cov(p[1], p[2]))[:MATRIX_PAGE_SPAN]:
        if pages and cov(sig | s, uns | u) - cov(sig, uns) < 2:
            break
        pages.append(n); sig |= s; uns |= u
    return sig, uns, "PDF sheet " + ", ".join(map(str, sorted(pages)))


def number_verdict(value, signed, unsigned):
    """'ok' | 'sign' (magnitude printed, sign differs: often a lost minus glyph) | 'absent'."""
    c = canon_number(value)
    if not c:
        return "absent"
    if c in signed:
        return "ok"
    return "sign" if c.lstrip("-") in unsigned else "absent"
