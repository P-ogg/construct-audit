"""Printed numbers in comparable form. Ported from pipeline3/rules/schema.py and review.py.

Comma decimals: continental tables print "0,54". Before this was handled, every correlation of one article was
misread (0,54 -> 54) and effect-size accuracy was 74%; with it, 100%.
"""
import re

# ",\d{3}" with a non-zero integer part -> thousands ("1,059", "85,706"); anything else -> decimal ("0,54", "-0,01", ",540").
_THOUSANDS = re.compile(r"^-?[1-9]\d{0,2}(,\d{3})+$")


def _fold_comma(s):
    if "," not in s:
        return s
    core = re.sub(r"[^\d,.\-]", "", s)
    if "." in core and "," in core:          # mixed notation: the rightmost separator is the decimal point
        return s.replace(".", "").replace(",", ".") if core.rfind(",") > core.rfind(".") else s.replace(",", "")
    if _THOUSANDS.match(core):
        return s.replace(",", "")
    return s.replace(",", ".")


def canon_number(v):
    """'−.480' -> '-0.48'; stars, parentheses and unicode minus handled. '' if no number."""
    s = str(v if v is not None else "").strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = _fold_comma(s)
    m = re.search(r"-?\s*\d*\.?\d+", s)
    if not m:
        return ""
    t = re.sub(r"\s+", "", m.group(0))
    neg = t.startswith("-")
    t = t.lstrip("-")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
        whole, _, frac = t.partition(".")
        whole = whole.lstrip("0") or "0"
        t = f"{whole}.{frac}" if frac else whole
    else:
        t = t.lstrip("0") or "0"
    return ("-" if neg and t not in ("0", "") else "") + t


def as_r(v):
    """A correlation as a float in (-1, 1), or None."""
    c = canon_number(v)
    try:
        f = float(c)
    except ValueError:
        return None
    return f if -1 < f < 1 else None


# Numbers as printed in the PDF text. A comma decimal may not touch a decimal point: a rotated table's text comes out
# reversed, so "p < 0.05" reads ",50.0<p", and ",50" must not be taken for 0.5.
_NUMBER = re.compile(
    r"-?\s*(?<![\d,.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])"
    r"|-?\s*\d*\.\d+"
    r"|-?\s*(?<![\d,.])\d{0,3},\d{1,3}(?![\d,.])"
    r"|-?\s*\d+")


def numbers_of(text):
    """(signed, unsigned) sets of canonical numbers printed in a text."""
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    signed = {canon_number(m.group(0)) for m in _NUMBER.finditer(text)}
    signed.discard("")
    return signed, {v.lstrip("-") for v in signed}
