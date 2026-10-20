"""The reviewer: a third reader for what the two coders disagreed on. Ported from pipeline3/adjudicator.py.

    coder A  ─┐
              ├─→ disagreements ─→ REVIEWER ─→ settled, or sent to a human ("rows to check")
    coder B  ─┘

Design kept from pipeline3:
  * Independence. The reviewer sees ONLY items the two coders already disagreed on, so it cannot talk them out of an
    agreement, and coder agreement is measured before it runs. A reviewer shown one value and a real quote tends to
    confirm it; two independent readings do not share that bias.
  * Evidence first. For each item the reviewer sees both values with the quote each coder relied on, reads the
    article itself, and returns its own verbatim quote, a reason and a confidence BEFORE the value.
  * Python decides, not the model. needs_human() applies fixed rules to the answer: no answer, low confidence, a
    high-impact field without high confidence, a value outside the allowed range, or a quote that is not in the
    article all send the item to a human, whatever the model claimed.

What is reviewed: correlation cells, and per variable the data source, the form of the measure and the reliability, and per study the
sample size. Free text (names, descriptions) is not: wording differs without a coding decision. A name mismatch is
flagged separately because it usually means the two coders numbered the variables differently.
"""
from . import verify
from .numbers import canon_number, as_r
from ..schema import DATA_TYPES, FORMS

MAX_ITEMS_PER_REQUEST = 35
CONFIDENCE_LEVELS = ("low", "medium", "high")
# Fields that change the audit's results need a high-confidence answer; the others a medium one.
HIGH_IMPACT = {"r", "data_type"}
FIELD_TEXT = {"r": "correlation", "data_type": "who or what produced the score (" + ", ".join(DATA_TYPES) + ")",
              "measure_form": "the form of the measure (" + ", ".join(FORMS) + ")",
              "reliability": "reliability (e.g. Cronbach's alpha)", "n": "sample size of the study"}

S = {"type": "string"}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["decisions"], "properties": {"decisions": {"type": "array", "items": {
    "type": "object", "additionalProperties": False,
    "required": ["ref", "quote", "page", "reason", "value", "chose", "confidence", "escalate"],
    "properties": {"ref": S, "quote": S, "page": S, "reason": S, "value": S,
                   "chose": {"type": "string", "enum": ["A", "B", "other"]},
                   "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
                   "escalate": {"type": "boolean"}}}}}}

SYSTEM = "You are the reviewer on a research coding team. You decide from the article, never from the coders. You answer with JSON only."

HEADER = """Two coders read this article INDEPENDENTLY and extracted its variables and its correlation table.
They agreed on almost everything. Listed below are ONLY the items where they disagree. Decide each one by reading the article yourself.

How to decide:
* Decide from the ARTICLE. Do not decide from which coder sounds more certain, and do not split the difference. A real sentence can still be the wrong evidence.
* You are not picking a coder. If the article supports a value neither gave, give that value and set "chose" to "other".
* For a correlation, give the value exactly as printed, sign included, or NOT_FOUND if no value is printed for that pair.
* Where allowed values are listed, answer with one of them exactly.
* Fill the fields in order: first find the evidence ("quote", "page"), then say what it shows ("reason"), and only then give "value", "chose" and "confidence".
* Confidence is checked. Use "high" only when you can quote the sentence or table that settles it. Use "low" whenever the article does not resolve the disagreement or both readings are defensible. Low-confidence items go to a human, and that is the right outcome: a wrong value given confidently costs more than an item someone has to look at.
* Set "escalate" to true for anything a human must see regardless of confidence: the coders read two different studies in the paper, the table is damaged, anything odd.

"quote" must be verbatim from the article; it is checked character by character. For a correlation, quote the table's title or note. If you cannot find a sentence to quote, your confidence is not "high".

Return one decision per item, with "ref" copied exactly. Do not omit an item: if you cannot decide it, return it with confidence "low".
"""


def item(ref, field, where, a, b, qa="", qb=""):
    """One disagreement. a/b: the two coders' values; qa/qb: the quote each relied on."""
    return dict(ref=ref, field=field, where=where, a=a, b=b, qa=qa, qb=qb)


def block(it):
    lines = [f'--- ref: "{it["ref"]}"', f"    {it['where']}", f"    field: {FIELD_TEXT.get(it['field'], it['field'])}"
             + ("   [HIGH IMPACT]" if it["field"] in HIGH_IMPACT else "")]
    if it["field"] in ("data_type", "measure_form"):
        lines.append("    allowed values: " + " | ".join(DATA_TYPES if it["field"] == "data_type" else FORMS))
    for who, val, quo in (("A", it["a"], it["qa"]), ("B", it["b"], it["qb"])):
        lines.append(f"    coder {who} said: {val if str(val).strip() else '(nothing)'}")
        if quo and quo != "NOT_FOUND":
            lines.append(f'        quoting: "{str(quo)[:600]}"')
        elif it["field"] != "r":
            lines.append("        (no supporting quote: treat this side as unevidenced)")
    return "\n".join(lines)


def prompt(items):
    return HEADER + f"\nItems to settle: {len(items)}.\n\n" + "\n\n".join(block(i) for i in items)


def chunks(items, size=MAX_ITEMS_PER_REQUEST):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def parse(reply, items):
    """Model reply -> {ref: decision}. Every item asked about appears, so a dropped item escalates instead of vanishing."""
    got = {str(d.get("ref", "")).strip(): d for d in (reply or {}).get("decisions", []) if isinstance(d, dict)}
    out = {}
    for it in items:
        d = got.get(it["ref"])
        out[it["ref"]] = ({"answered": False, "reason": "the reviewer did not return this item"} if d is None else
                          {"answered": True, **{k: d.get(k, "") for k in ("value", "chose", "reason", "quote", "page")},
                           "confidence": str(d.get("confidence", "")).strip().lower(), "escalate": bool(d.get("escalate"))})
    return out


def valid_value(field, value):
    """'' if the value is allowed for the field, otherwise why not."""
    v = str(value).strip()
    if field == "r":
        return "" if v == "NOT_FOUND" or as_r(v) is not None else f"{v!r} is not a correlation"
    if field == "data_type":
        return "" if v in DATA_TYPES else f"{v!r} is not one of {', '.join(DATA_TYPES)}"
    if field == "measure_form":
        return "" if v in FORMS else f"{v!r} is not one of {', '.join(FORMS)}"
    if field == "reliability":
        c = canon_number(v)
        return "" if v in ("", "NOT_FOUND") or (c and 0 < float(c) <= 1) else f"{v!r} is not a reliability between 0 and 1"
    if field == "n":
        c = canon_number(v)
        return "" if v in ("", "NOT_FOUND") or (c and float(c) >= 2) else f"{v!r} is not a sample size"
    return ""


def needs_human(it, d, hay):
    """'' when the reviewer's answer stands, otherwise why a human must look. Applied after the fact: a model that says
    "high" about an impossible value is overruled; a model that declines to be confident is believed."""
    if not d.get("answered"):
        return d.get("reason") or "the reviewer gave no usable answer"
    conf = d.get("confidence", "")
    if conf not in CONFIDENCE_LEVELS:
        return f"the reviewer's confidence was unreadable ({conf!r})"
    if d.get("escalate"):
        return d.get("reason") or "the reviewer asked for a human"
    if conf == "low":
        return "the reviewer was not confident; the article does not settle it"
    if it["field"] in HIGH_IMPACT and conf != "high":
        return f"this changes the results and the reviewer was only {conf}ly confident"
    bad = valid_value(it["field"], d.get("value", ""))
    if bad:
        return f"the reviewer's answer is not allowed: {bad}"
    q = d.get("quote", "")
    # Table text is often garbled in PDFs, so a correlation's quote is not held to the check; its number is checked
    # against the table page later like every other cell.
    if it["field"] != "r" and conf == "high" and (not q or q == "NOT_FOUND" or verify.check_quote(q, hay)[0] == "fail"):
        return "the reviewer's supporting quote is not in the article"
    return ""


def settle(items, ask, hay):
    """Send the disagreements to the reviewer in chunks. ask(prompt, part) -> reply dict (cached by the caller).
    Returns {ref: (decision, why_human)}; why_human is '' when the reviewer's answer stands."""
    out = {}
    for n, group in enumerate(chunks(items), 1):
        try:
            dec = parse(ask(prompt(group), n), group)
        except Exception as e:                         # a failed review must not lose the article
            dec = {it["ref"]: {"answered": False, "reason": f"the reviewer failed ({str(e)[:150]})"} for it in group}
        for it in group:
            out[it["ref"]] = (dec[it["ref"]], needs_human(it, dec[it["ref"]], hay))
    return out
