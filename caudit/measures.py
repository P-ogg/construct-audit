"""Name each variable's MEASURE consistently across articles, separately from its construct.

A construct is what the authors say they measured (job satisfaction, firm performance). A measure is how they observed
it (a five-item self-report scale, return on assets, Tobin's q). The jingle check compares the measures of one
construct, so measures must be named the same way in every article.

Three routes, in order of preference:
  draft(variables, client)   the model reads each variable's description, form and cited instrument and names the
                             measure, keeping one list of names across the literature; the researcher edits the list
  fallback(variables)        basic labels, no model call: the form of the measure plus the instrument cited, if any.
                             Used only for tables coded without a model and for variables the model left unnamed
                             (e.g. "multi-item scale: Allen & Meyer 1990", "multi-item scale (rated by others)", "ratio or share")
  firm_level_rules(...)      hand-written keyword rules for firm-level innovation research, for tables
                             whose tables were coded before forms were recorded. They do not transfer to other fields.
"""
import math, re
import pandas as pd

from .schema import FORMS

# Measures too broad to say anything when two constructs share them (the "one measure, two constructs" check skips them)
GENERIC = set(FORMS) | {f"{f} (rated by others)" for f in FORMS} | {f"{f} (observed)" for f in FORMS} | {"", "unknown", "multi-item / perceptual scale", "other archival", "survey scale"}


# ------------------------------------------------------------------ basic labels
def _instrument(s):
    s = re.sub(r"\s+", " ", str(s or "")).strip(" .;,")
    return "" if s.lower() in ("", "none", "not_found", "n/a", "na") else s


def fallback(variables):
    """Basic label: form (+ rater) (+ cited scale). Coarser than draft(): archival measures of one form share a name."""
    v = variables.copy()
    for c in ("measure_form", "instrument"):
        if c not in v.columns:
            v[c] = ""
    empty = v["measure"].fillna("") == ""
    forms = v.loc[empty, "measure_form"].fillna("").replace("", "unknown")
    inst = v.loc[empty, "instrument"].map(_instrument)
    # a self-rated and a supervisor-rated scale are different measures even when the form is the same
    who = v.loc[empty, "data_type"].map({"other-report": " (rated by others)", "observed": " (observed)"}).fillna("")
    v.loc[empty, "measure"] = [f"{f}{w}: {i}" if i else f"{f}{w}" for f, w, i in zip(forms, who, inst)]
    return v


apply = fallback      # the default when no model is used


# ------------------------------------------------------------------ model draft
SYSTEM = """You name the MEASURES used in a research literature, so that the same measure gets the same name in every article.
A measure is how a variable was observed, not the concept it stands for: name "return on assets", "Tobin's q",
"patent counts", "OCQ (Mowday et al. 1979)", "supervisor-rated task performance", never "firm performance" or "commitment".
Rules:
- Same operation, same name: "ROA", "return on assets" and "net income / total assets" are one measure.
- A published scale is named by its source (e.g. "UWES (Schaufeli et al. 2002)"); its short or adapted versions keep the source name
  unless the article says it built a new scale.
- Different operations get different names even when the construct is the same (patent counts vs new-product sales share).
- Self-made survey scales without a cited source: "own multi-item scale" or "own single-item rating", plus the source of the rating
  if not self-report (e.g. "own scale, supervisor-rated").
- Reuse a name from the current list whenever it fits. Add a new name only for a genuinely different measure.
- Keep names short (2-6 words)."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["measures"],
          "properties": {"measures": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["id", "measure"],
              "properties": {"id": {"type": "integer"}, "measure": {"type": "string"}}}}}}


def inventory(variables, chars=180):
    """Distinct (form, instrument, description, data source) combinations, so identical descriptions are named once."""
    v = variables.copy()
    for c in ("measure_form", "instrument"):
        if c not in v.columns:
            v[c] = ""
    v["_key"] = (v.measure_form.fillna("") + "|" + v.instrument.map(_instrument) + "|" + v.data_type.fillna("") + "|"
                 + v.description.fillna("").str.strip().str.lower().str[:chars])
    g = v.groupby("_key").agg(name=("name", "first"), description=("description", "first"), form=("measure_form", "first"),
                              instrument=("instrument", "first"), data_type=("data_type", "first"), variables=("var", "size"))
    return v, g.sort_values(["form", "instrument", "description"]).reset_index()


def draft(variables, client, batch=120, progress=print):
    """The model names each distinct measure, carrying the growing list of names from batch to batch.
    Returns (variables with 'measure' filled, the list of names with counts)."""
    v, inv = inventory(variables)
    names, out = [], {}
    n = math.ceil(len(inv) / batch) if len(inv) else 0
    for b in range(n):
        chunk = inv.iloc[b * batch:(b + 1) * batch]
        progress(f"naming measures: batch {b + 1} of {n}")
        lines = "\n".join(f"{i}. [{r.form or 'form unknown'}; {r.data_type}] {r.name}: {r.description[:200]}"
                          + (f"  (cites: {_instrument(r.instrument)})" if _instrument(r.instrument) else "")
                          for i, r in enumerate(chunk.itertuples()))
        current = "\n".join(f"- {m}" for m in names) or "(none yet)"
        res = client.json(SYSTEM, f"Current list of measure names:\n{current}\n\nName the measure of each numbered variable "
                                  f"(return its number as id):\n\n{lines}", SCHEMA, label=f"measures {b + 1}")
        got = {m["id"]: m["measure"].strip() for m in res["measures"]}
        for i, key in enumerate(chunk["_key"]):
            m = got.get(i, "")
            out[key] = m
            if m and m not in names:
                names.append(m)
    v["measure"] = v["_key"].map(out).fillna("")
    v = fallback(v.drop(columns="_key"))           # anything the model left blank gets the basic label
    counts = v.groupby("measure").agg(variables=("var", "size"), articles=("article", "nunique")).reset_index()
    return v, counts.sort_values("articles", ascending=False).reset_index(drop=True)


def rename(variables, mapping):
    """Apply the researcher's edits: {old measure name: new name}. Giving two names the same new name merges them."""
    v = variables.copy()
    m = {k: str(x).strip() for k, x in mapping.items() if str(x).strip()}
    v["measure"] = v["measure"].map(lambda s: m.get(s, s))
    return v


# ------------------------------------------------------------------ firm-level innovation research only
R = lambda p: re.compile(p, re.I)
PAT = R(r"patent")
RATIO = R(r"(share|percent|proportion|ratio|%|fraction|divided|per (employee|sales)|/\s*(sales|revenue|turnover|employees|assets)|\bto (total|the total|all)\b|\bover\b)")
RD = R(r"(r ?& ?d|research and development|\brd\b)")
SPEND = R(r"(expenditure|spending|spend|invest|budget|expense|intensity|outlay|cost)")
SALES = R(r"(sales|turnover|revenue)")
NEWPROD = R(r"new[- ]?(to[- ]?(the[- ]?)?(market|firm)[- ]?)?products?|innovative products|product innovations?|innovat")
COUNT = R(r"(number of|count|how many|total of|introduc|launch)")
EMP = R(r"(employee|workforce|staff|headcount|personnel|employment)")
ASSETS = R(r"total assets|\bassets\b")
AGE = R(r"(\bage\b|years since|founded|founding|establish|incorporat)")
BINARY = R(r"(dummy|binary|dichotom|whether|yes/no|1 if|takes? (the )?value|=\s*1|equals one|0/1)")
SCALE_WORDS = R(r"(likert|scale|extent|items?\b|agree|rated|rating)")


def firm_level_rules(name, description, data_type):
    """Keyword rules for firm-level innovation research. Not for other literatures; the app does not use them."""
    t = f"{name} | {description}"
    if PAT.search(t):
        return "patent counts"
    if SALES.search(t) and NEWPROD.search(t) and RATIO.search(t):
        return "innovation yes/no" if BINARY.search(t) else "new-product sales share"
    if COUNT.search(t) and NEWPROD.search(t):
        if BINARY.search(t):                       # "whether the firm introduced a new product (1 = yes)" is not a count
            return "innovation yes/no"
        if data_type == "survey" and SCALE_WORDS.search(t):
            return "multi-item / perceptual scale"
        return "new-product counts"
    if RD.search(t) and SPEND.search(t):
        return "R&D ratio" if (RATIO.search(t) or R(r"intensity").search(t)) else "R&D spending (absolute)"
    if EMP.search(t) and not RATIO.search(t):
        return "employees"
    if ASSETS.search(t) and not R(r"return on assets|\broa\b").search(t) and not RATIO.search(t):
        return "assets"
    if SALES.search(t) and not RATIO.search(t) and not R(r"(growth|herfindahl|index|diversi|entropy)").search(t):
        return "sales"
    if AGE.search(t) and data_type != "survey":
        return "age (years)"
    if data_type == "survey":
        return "multi-item / perceptual scale"
    return "other archival" if data_type == "archival" else ""


def apply_firm_level_rules(variables):
    v = variables.copy()
    empty = v["measure"].fillna("") == ""
    v.loc[empty, "measure"] = [firm_level_rules(n, d, t) for n, d, t in zip(v.name[empty], v.description[empty], v.data_type[empty])]
    return v
