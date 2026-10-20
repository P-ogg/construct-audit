"""The two tables every stage reads and writes. Field-independent: nothing here knows about any literature.

variables.csv     one row per variable in a study
  article, study, var        identifiers (strings); (article, study, var) is unique
  name                       the label the authors used
  description                how the authors describe / measure it
  data_type                  who or what produced the score: 'survey' (self-report) | 'other-report' (supervisor, peer, expert)
                             | 'archival' (records, databases) | 'observed' (coded behaviour, experiment) | 'unknown'
  reliability                e.g. Cronbach's alpha, or empty
  construct                  the construct it belongs to (from a codebook); may be empty before grouping
  label                      optional finer grouping (normalised author label)
  measure                    the measure, named consistently across articles (e.g. 'patent counts', 'OCQ (Mowday et al.)');
                             may be empty before classification
  measure_form, instrument   optional: the form of the measure (FORMS) and the published scale it cites, if any

correlations.csv  one row per reported correlation
  article, study, var_a, var_b, r, n
"""
import pandas as pd

DATA_TYPES = ("survey", "other-report", "archival", "observed", "unknown")
FORMS = ("multi-item scale", "single-item rating", "count", "ratio or share", "amount", "yes/no indicator", "time or age", "index",
         "category", "other")
VAR_COLS = ["article", "study", "var", "name", "description", "data_type", "reliability", "construct", "label", "measure"]
COR_COLS = ["article", "study", "var_a", "var_b", "r", "n"]
KEY = ["article", "study", "var"]


def load_variables(path_or_df):
    df = pd.read_csv(path_or_df, dtype=str, keep_default_na=False) if not isinstance(path_or_df, pd.DataFrame) else path_or_df.copy()
    for c in VAR_COLS:
        if c not in df.columns:
            df[c] = ""
    for c in ["article", "study", "var"]:
        df[c] = df[c].astype(str)
    df["reliability"] = pd.to_numeric(df["reliability"], errors="coerce")
    df.loc[(df.reliability <= 0.3) | (df.reliability >= 1), "reliability"] = float("nan")
    df["data_type"] = df["data_type"].where(df["data_type"].isin(DATA_TYPES), "unknown")
    for c in ("measure_form", "instrument"):
        if c not in df.columns:
            df[c] = ""
    if df.duplicated(KEY).any():
        raise ValueError(f"{df.duplicated(KEY).sum()} duplicate (article, study, var) rows in variables")
    return df[VAR_COLS + [c for c in df.columns if c not in VAR_COLS]]


def load_correlations(path_or_df):
    df = pd.read_csv(path_or_df, dtype=str, keep_default_na=False) if not isinstance(path_or_df, pd.DataFrame) else path_or_df.copy()
    missing = [c for c in COR_COLS if c not in df.columns and c != "n"]
    if missing:
        raise ValueError(f"correlations table lacks columns {missing}")
    if "n" not in df.columns:
        df["n"] = ""
    for c in ["article", "study", "var_a", "var_b"]:
        df[c] = df[c].astype(str)
    df["r"] = pd.to_numeric(df["r"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df = df[df.r.between(-0.999, 0.999) & (df.var_a != df.var_b)]
    return df[COR_COLS].reset_index(drop=True)


def pairs(variables, correlations):
    """Correlations with both ends' variable attributes attached (columns prefixed a_ and b_)."""
    v = variables.set_index(KEY)
    a = v.add_prefix("a_"); b = v.add_prefix("b_")
    p = correlations.join(a, on=["article", "study", "var_a"]).join(b, on=["article", "study", "var_b"])
    return p.dropna(subset=["a_name", "b_name"]).reset_index(drop=True)
