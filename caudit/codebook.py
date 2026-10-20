"""Group variables into constructs.

Two routes:
  load(path)    a researcher's codebook: CSV with columns label,construct (label = normalised variable label)
  draft(...)    an LLM drafts the codebook from the variable inventory; the researcher reviews and edits it

The LLM never reads the articles here. It reads the inventory: each distinct label with the number of articles
using it and two example descriptions. Two passes:
  1. propose constructs from the frequent labels
  2. assign every label to one construct (or 'not a construct'), in batches
Constructs are concepts (firm size, job satisfaction), never measures (number of employees, a five-item scale):
measures are named separately (measures.py).
"""
import json, math
import pandas as pd

NOT_CONSTRUCT = "not a construct"

PROPOSE_SYSTEM = """You build codebooks for research syntheses in organisational science and the wider social sciences.
A construct is a concept a study claims to measure (e.g. job satisfaction, team cohesion, firm performance, absorptive capacity).
A measure is how it was observed (a five-item scale, supervisor ratings, return on assets, patent counts). Name constructs, never
measures: variables labelled "number of employees" or "ROA" belong to the construct they measure (often firm size, firm performance).
Keep distinct concepts apart even if they usually correlate (do not merge job satisfaction and organisational commitment,
or exploration and exploitation, unless the literature uses the labels interchangeably). Keep the level of analysis in the name
when it changes the concept (individual vs team vs organisational learning). Do not create umbrella bins that lump different
concepts together. Dummy and bookkeeping variables (industry, country, year dummies, survey wave) are not constructs."""

ASSIGN_SYSTEM = PROPOSE_SYSTEM + """
You assign variable labels to the codebook. Use the label and the example descriptions. If a label fits no construct,
propose a new construct name (prefix it with NEW:). Use "not a construct" for dummies and bookkeeping variables.
If a label is a measure name, assign the construct it measures, judging from the descriptions."""

PROPOSE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["constructs"],
                  "properties": {"constructs": {"type": "array", "items": {
                      "type": "object", "additionalProperties": False, "required": ["name", "definition", "example_labels"],
                      "properties": {"name": {"type": "string"}, "definition": {"type": "string"},
                                     "example_labels": {"type": "array", "items": {"type": "string"}}}}}}}

ASSIGN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["assignments"],
                 "properties": {"assignments": {"type": "array", "items": {
                     "type": "object", "additionalProperties": False, "required": ["id", "construct", "confidence"],
                     "properties": {"id": {"type": "integer"}, "construct": {"type": "string"},
                                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}}}}}


def inventory(variables, n_examples=2, chars=140):
    v = variables[variables.label != ""]
    g = v.groupby("label").agg(articles=("article", "nunique"), variables=("var", "size"),
                               survey=("data_type", lambda s: int((s == "survey").sum())),
                               examples=("description", lambda s: [d[:chars] for d in s[s != ""].drop_duplicates().head(n_examples)]))
    return g.sort_values(["articles", "variables"], ascending=False).reset_index()


def _line(i, row):
    ex = " | ".join(row.examples) if row.examples else "(no description)"
    return f"{i}. {row.label}  [{row.articles} articles; {row.survey} survey]  e.g. {ex}"


def draft(variables, client, min_articles=2, max_propose=700, batch=150, progress=print):
    inv = inventory(variables)
    top = inv[inv.articles >= min_articles].head(max_propose)
    lines = "\n".join(_line(i, r) for i, r in enumerate(top.itertuples()))
    progress(f"proposing constructs from {len(top)} frequent labels")
    proposal = client.json(PROPOSE_SYSTEM,
                           "Here are the variable labels used in a literature, most frequent first. Propose the codebook: "
                           "every construct needed to classify them, with a one-sentence definition.\n\n" + lines,
                           PROPOSE_SCHEMA, label="propose")
    constructs = proposal["constructs"]
    book = "\n".join(f"- {c['name']}: {c['definition']}" for c in constructs)
    rows = []
    n = math.ceil(len(inv) / batch)
    for b in range(n):
        chunk = inv.iloc[b * batch:(b + 1) * batch]
        progress(f"assigning labels: batch {b + 1} of {n}")
        lines = "\n".join(_line(i, r) for i, r in enumerate(chunk.itertuples()))
        out = client.json(ASSIGN_SYSTEM, f"Codebook:\n{book}\n\nAssign each numbered label (return its number as id):\n\n{lines}",
                          ASSIGN_SCHEMA, label=f"assign {b + 1}")
        got = {a["id"]: a for a in out["assignments"]}
        for i, r in enumerate(chunk.itertuples()):
            a = got.get(i, {"construct": "", "confidence": "low"})
            rows.append(dict(label=r.label, construct=a["construct"].strip(), confidence=a["confidence"], articles=r.articles))
    return pd.DataFrame(constructs), pd.DataFrame(rows)


def load(path_or_df):
    """A researcher's codebook: columns 'variable name' (as authors wrote it; 'name' or 'label' also accepted) and 'construct'.
    Names are cleaned the same way as the variables' names, so 'Job Satisfaction (T1)' matches 'job satisfaction'."""
    from .labels import label as clean
    df = path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    src = next((c for c in ("variable name", "name", "label") if c in df.columns), None)
    if src is None or "construct" not in df.columns:
        raise ValueError("the codebook needs two columns: 'variable name' and 'construct'")
    df = df.fillna("").astype(str)
    out = pd.DataFrame({"label": df[src].map(clean), "construct": df["construct"].str.strip()})
    return out[(out.label != "") & (out.construct != "")].drop_duplicates("label")


def apply(variables, assignments, column="construct", keep_existing=False):
    """Set each variable's construct from the codebook. keep_existing: a construct already in the table wins."""
    m = dict(zip(assignments.label, assignments.construct.str.replace(r"^NEW:\s*", "", regex=True)))
    v = variables.copy()
    new = v.label.map(m).fillna("")
    v[column] = v[column].fillna("").where((v[column].fillna("") != "") & keep_existing, new) if column in v else new
    v.loc[v[column].str.lower() == NOT_CONSTRUCT, column] = ""
    return v


CONSOLIDATE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["mapping"],
                      "properties": {"mapping": {"type": "array", "items": {
                          "type": "object", "additionalProperties": False, "required": ["id", "construct"],
                          "properties": {"id": {"type": "integer"}, "construct": {"type": "string"}}}}}}


def consolidate(constructs, assignments, client, batch=250, progress=print):
    """Map the construct names invented during assignment (NEW: ...) onto the codebook, or onto one shared new name.
    Without this pass the draft fragments: the same concept ends up under several new names."""
    new = assignments[assignments.construct.str.startswith("NEW:")]
    if new.empty:
        return constructs, assignments
    names = (new.assign(name=new.construct.str.replace(r"^NEW:\s*", "", regex=True))
             .groupby("name").articles.sum().sort_values(ascending=False))
    book = "\n".join(f"- {c}" for c in constructs.name)
    mapping, n = {}, math.ceil(len(names) / batch)
    for b in range(n):
        chunk = names.iloc[b * batch:(b + 1) * batch]
        progress(f"consolidating new construct names: batch {b + 1} of {n}")
        lines = "\n".join(f"{i}. {nm} [{a} articles]" for i, (nm, a) in enumerate(chunk.items()))
        out = client.json(PROPOSE_SYSTEM,
                          f"Codebook:\n{book}\n\nThese construct names were proposed while coding. For each, return the codebook construct "
                          "it is the same concept as, exactly as written above; if none fits, return a short new construct name, and give "
                          "synonyms the same new name. Use 'not a construct' for dummies or bookkeeping variables.\n\n" + lines,
                          CONSOLIDATE_SCHEMA, label=f"consolidate {b + 1}")
        got = {m["id"]: m["construct"].strip() for m in out["mapping"]}
        for i, nm in enumerate(chunk.index):
            mapping[nm] = got.get(i, nm)
    a = assignments.copy()
    is_new = a.construct.str.startswith("NEW:")
    a.loc[is_new, "construct"] = a.loc[is_new, "construct"].str.replace(r"^NEW:\s*", "", regex=True).map(mapping)
    known = set(constructs.name)
    extra = sorted({c for c in mapping.values() if c not in known and c.lower() != NOT_CONSTRUCT})
    cons = pd.concat([constructs, pd.DataFrame({"name": extra, "definition": "(added during coding)", "example_labels": [[]] * len(extra)})],
                     ignore_index=True)
    return cons, a
