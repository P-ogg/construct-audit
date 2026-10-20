"""Normalise authors' variable names into labels, so 'Job satisfaction (T1)', 'job satisfaction' and 'Job Satisfaction'
group together.

Deliberately conservative and field-independent: it only removes formatting (case, brackets, wave and year markers,
possessives, punctuation) and plural endings. It never merges concepts or strips words like "organizational" or "firm",
which change the meaning in some fields (organizational commitment is not team commitment). Grouping concepts is the
codebook's job. A literature-specific synonym list can be passed in.
"""
import re

PLURAL = re.compile(r"(?<=[a-z]{3})(?<!s)s\b")      # "partners" -> "partner"; leaves "business", "analysis", "process"


def label(name, synonyms=()):
    s = re.sub(r"\(.*?\)|\[.*?\]", "", str(name).lower())
    s = re.sub(r"\b(t|wave|time)\s?\d\b|\b\d{4}\b", "", s)
    s = re.sub(r"['’]s\b", "", s); s = re.sub(r"[^a-z0-9& ]", " ", s)
    s = re.sub(r"^the\s+", "", s.strip())
    s = " ".join(w if re.search(r"(ss|is|us|ics|ness)$", w) else PLURAL.sub("", w) for w in s.split())
    for a, b in synonyms:
        s = re.sub(a, b, s)
    return re.sub(r"\s+", " ", s).strip()


def apply(variables, synonyms=()):
    v = variables.copy()
    empty = v["label"].fillna("") == ""
    v.loc[empty, "label"] = v.loc[empty, "name"].map(lambda n: label(n, synonyms))
    return v
