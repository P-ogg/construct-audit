"""The audit: four checks on a coded literature. No model calls; works for any field.

Terms
  construct  what the authors say they measured (from the codebook)
  measure    what was actually counted or asked (kind of measure)
  profile    a measure's pooled correlations with a set of common "anchor" constructs

Checks
  jingle         one construct, several kinds of measure: do they behave alike?
                 (profile comparison against random splits + direct correlations within studies)
  jangle         two constructs: are they the same thing under two names?
                 strict test: corrected correlation near 1 AND near-identical profile (Le et al. 2010)
  contamination  a kind of measure that tracks an anchor (e.g. firm size) unlike the construct's other measures
  method         survey measures correlating more with survey than with archival measures in the same study

Pooling: correlations are Fisher-z averaged within each article, then across articles (each article counts once).
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .schema import pairs as _pairs
from .measures import GENERIC as GENERIC_MEASURES


@dataclass
class Config:
    n_anchors: int = 8                 # how many common constructs serve as comparison points
    min_anchor_articles: int = 20      # an anchor must appear in at least this many articles
    min_measure_articles: int = 10     # a kind of measure needs this many articles to be profiled
    min_cell_articles: int = 3         # articles needed for one profile entry (measure x anchor)
    min_jangle_articles: int = 3       # articles in which two constructs must be measured together
    min_corrected_articles: int = 5    # articles reporting both reliabilities before the corrected r is trusted for a duplicate verdict
    min_shared_articles: int = 2       # articles before "one measure, two constructs" is flagged
    alpha: float = 0.05                # false discovery rate for the jingle tests (Benjamini-Hochberg across all tests)
    n_perm: int = 2000
    seed: int = 1
    duplicate_r: float = 0.85          # construct-level correlation that may signal redundancy (Bowling et al. 2026; cf. Le et al. 2010)
    profile_pct: float = 90.0          # profile similarity percentile treated as "behaves alike"
    contamination_gap: float = 0.15    # difference from sibling measures that counts as tracking an anchor
    jangle_level: str = "label"        # 'label' (fine) or 'construct'
    exclude_constructs: tuple = ()     # e.g. ('Disregard',) for codebook bins that are not constructs
    anchors: tuple = ()                # fix the anchor constructs instead of choosing the most common ones


def z(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


def pool(df, col="z"):
    """Article-level pooling: mean z per article, then mean over articles. Returns (r, k)."""
    if df.empty:
        return (np.nan, 0)
    per_article = df.groupby("article")[col].mean()
    return (float(np.tanh(per_article.mean())), int(per_article.size))


def icc_de(x, y):
    """Double-entry intraclass correlation (Lawson & Robins 2021): equal level AND shape."""
    a = np.r_[x, y]; b = np.r_[y, x]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else np.nan


def congruence(x, y):
    """Tucker's congruence coefficient (Lorenzo-Seva & ten Berge 2006), as proposed for nomological networks by Franke, Sarstedt & Danks (2021): proportional profiles."""
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d else np.nan


def directed(p):
    """Each correlation twice, as x->y and y->x, for 'how does x relate to others' questions."""
    keep = ["article", "study", "r", "z"]
    cols = [c[2:] for c in p.columns if c.startswith("a_")]
    a = p[keep + ["a_" + c for c in cols] + ["b_" + c for c in cols]].copy()
    a.columns = keep + ["x_" + c for c in cols] + ["y_" + c for c in cols]
    b = p[keep + ["b_" + c for c in cols] + ["a_" + c for c in cols]].copy()
    b.columns = a.columns
    return pd.concat([a, b], ignore_index=True)


def choose_anchors(D, cfg):
    if cfg.anchors:
        return list(cfg.anchors)
    d = D[(D.y_construct != "") & ~D.y_construct.isin(cfg.exclude_constructs)]
    counts = d.groupby("y_construct")["article"].nunique().sort_values(ascending=False)
    return list(counts[counts >= cfg.min_anchor_articles].index[: cfg.n_anchors])


# ---------------------------------------------------------------- jingle
def jingle(D, P, construct, anchors, cfg, rng):
    others = [a for a in anchors if a != construct]
    d = D[(D.x_construct == construct) & (D.x_measure != "") & D.y_construct.isin(others)]
    if d.empty:
        return None
    # blocks = (article, measure); block x anchor = mean z
    blk = d.groupby(["article", "x_measure", "y_construct"])["z"].mean().unstack("y_construct")
    blk = blk.reindex(columns=others)
    kinds = blk.reset_index().groupby("x_measure")["article"].nunique()
    kinds = sorted(kinds[kinds >= cfg.min_measure_articles].index)
    if len(kinds) < 2:
        return None
    blk = blk[blk.index.get_level_values("x_measure").isin(kinds)]
    labels = blk.index.get_level_values("x_measure").to_numpy()
    M = blk.to_numpy()                      # blocks x anchors (NaN where missing)
    present = ~np.isnan(M)
    Mz = np.where(present, M, 0.0)

    def profiles(lab):
        out = {}
        for k in kinds:
            m = lab == k
            n = present[m].sum(0)
            s = Mz[m].sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[k] = np.where(n >= cfg.min_cell_articles, np.tanh(s / n), np.nan)
        return out

    def stats(pr, k1, k2):
        ok = ~np.isnan(pr[k1]) & ~np.isnan(pr[k2])
        if ok.sum() < 3:
            return None
        x, y = pr[k1][ok], pr[k2][ok]
        return dict(anchors=int(ok.sum()), icc=icc_de(x, y), congruence=congruence(x, y), mad=float(np.abs(x - y).mean()))

    obs = profiles(labels)
    comps = {}
    for i, k1 in enumerate(kinds):
        for k2 in kinds[i + 1:]:
            s = stats(obs, k1, k2)
            if s:
                comps[(k1, k2)] = dict(s, null_icc=[], null_cong=[], null_mad=[])
    for _ in range(cfg.n_perm):
        pr = profiles(rng.permutation(labels))
        for (k1, k2), c in comps.items():
            s = stats(pr, k1, k2)
            if s:
                c["null_icc"].append(s["icc"]); c["null_cong"].append(s["congruence"]); c["null_mad"].append(s["mad"])
    comparisons = []
    for (k1, k2), c in comps.items():
        ni, nc, nm = map(np.asarray, (c.pop("null_icc"), c.pop("null_cong"), c.pop("null_mad")))
        comparisons.append(dict(measure_1=k1, measure_2=k2, **c,
                                null_icc_median=float(np.median(ni)) if ni.size else np.nan,
                                # permutation p-values with the +1 correction, so p is never exactly 0
                                p_icc=float(((ni <= c["icc"]).sum() + 1) / (ni.size + 1)) if ni.size else np.nan,
                                p_congruence=float(((nc <= c["congruence"]).sum() + 1) / (nc.size + 1)) if nc.size else np.nan,
                                p_mad=float(((nm >= c["mad"]).sum() + 1) / (nm.size + 1)) if nm.size else np.nan))
    # direct correlations between kinds of measure of this construct, within the same study
    same = P[(P.a_construct == construct) & (P.b_construct == construct) & (P.a_measure != P.b_measure)
             & P.a_measure.isin(kinds) & P.b_measure.isin(kinds)]
    direct = []
    if not same.empty:
        k1 = np.minimum(same.a_measure, same.b_measure); k2 = np.maximum(same.a_measure, same.b_measure)
        for (m1, m2), g in same.assign(m1=k1, m2=k2).groupby(["m1", "m2"]):
            r, k = pool(g)
            direct.append(dict(measure_1=m1, measure_2=m2, r=r, articles=k))
    # profiles table + article counts
    arts = blk.reset_index().groupby("x_measure")["article"].nunique()
    cell_k = {k: dict(zip(others, present[labels == k].sum(0).tolist())) for k in kinds}
    prof = {k: {a: (None if np.isnan(v) else float(v)) for a, v in zip(others, obs[k])} for k in kinds}
    # contamination: a measure that tracks an anchor unlike the construct's other measures
    contamination = []
    for a_i, a in enumerate(others):
        vals = {k: obs[k][a_i] for k in kinds if not np.isnan(obs[k][a_i])}
        if len(vals) < 2:
            continue
        for k, v in vals.items():
            rest = [w for kk, w in vals.items() if kk != k]
            gap = v - float(np.mean(rest))
            if abs(gap) >= cfg.contamination_gap:
                contamination.append(dict(measure=k, anchor=a, r=v, others_mean=float(np.mean(rest)), gap=gap))
    return dict(construct=construct, measures=kinds, articles={k: int(arts[k]) for k in kinds},
                profiles=prof, cell_articles=cell_k, comparisons=comparisons, direct=direct, contamination=contamination)


# ---------------------------------------------------------------- jangle
def jangle(D, P, anchors, cfg):
    lvl = cfg.jangle_level if cfg.jangle_level in ("label", "construct") else "construct"
    a, b = "a_" + lvl, "b_" + lvl
    q = P[(P[a] != "") & (P[b] != "") & (P[a] != P[b])
          & ~P.a_construct.isin(cfg.exclude_constructs) & ~P.b_construct.isin(cfg.exclude_constructs)].copy()
    if q.empty:
        return dict(level=lvl, pairs=[], baseline={})
    q["u1"] = np.minimum(q[a], q[b]); q["u2"] = np.maximum(q[a], q[b])
    rel = q.a_reliability * q.b_reliability
    q["zc"] = z(q.r / np.sqrt(rel))
    # profiles of each unit over the anchor constructs (excluding its own construct)
    x, y = "x_" + lvl, "y_" + lvl
    d = D[(D[x] != "") & D.y_construct.isin(anchors) & (D.y_construct != D.x_construct) & (D[x] != D[y])]
    prof = d.groupby([x, "y_construct", "article"])["z"].mean().reset_index()
    cnt = prof.groupby([x, "y_construct"])["article"].nunique()
    mean = np.tanh(prof.groupby([x, "y_construct"])["z"].mean())
    profile = mean[cnt >= 2].unstack("y_construct").reindex(columns=anchors)

    def sim(u, w):
        if u not in profile.index or w not in profile.index:
            return None
        pu, pw = profile.loc[u].to_numpy(), profile.loc[w].to_numpy()
        ok = ~np.isnan(pu) & ~np.isnan(pw)
        if ok.sum() < 4:
            return None
        return dict(anchors=int(ok.sum()), icc=icc_de(pu[ok], pw[ok]), congruence=congruence(pu[ok], pw[ok]))

    units = list(profile.index)
    base = [s for i, u in enumerate(units) for w in units[i + 1:] for s in [sim(u, w)] if s]
    base_icc = np.array([s["icc"] for s in base]); base_c = np.array([s["congruence"] for s in base])
    out = []
    for (u1, u2), g in q.groupby(["u1", "u2"]):
        r, k = pool(g)
        if k < cfg.min_jangle_articles:
            continue
        gc = g.dropna(subset=["zc"])
        rc, kc = pool(gc, "zc") if not gc.empty else (np.nan, 0)
        rs, _ = pool(gc) if not gc.empty else (np.nan, 0)        # raw r in the SAME articles, so the correction is not confused with a change of sample
        s = sim(u1, u2)
        row = dict(unit_1=u1, unit_2=u2, r=r, articles=k, r_same_articles=rs, r_corrected=rc, articles_corrected=kc)
        if s:
            row.update(profile_anchors=s["anchors"], profile_icc=s["icc"], profile_congruence=s["congruence"],
                       icc_percentile=float((base_icc <= s["icc"]).mean() * 100) if base_icc.size else np.nan)
        strength = rc if kc >= cfg.min_corrected_articles else r
        alike = row.get("icc_percentile", 0) >= cfg.profile_pct
        row["note"] = ("" if not (rc == rc and rc >= cfg.duplicate_r) or (kc >= cfg.min_corrected_articles and alike) else
                       f"corrected r rests on {kc} article(s); {cfg.min_corrected_articles} needed" if kc < cfg.min_corrected_articles else
                       "correlate near 1 but relate to other constructs differently")
        row["verdict"] = ("possible duplicate" if strength >= cfg.duplicate_r and alike else
                          "siblings" if r >= 0.4 and alike else
                          "related" if r >= 0.4 else "distinct")
        out.append(row)
    out.sort(key=lambda t: -t["r"])
    return dict(level=lvl, pairs=out, baseline=dict(pairs=len(base),
                median_icc=float(np.median(base_icc)) if base_icc.size else np.nan,
                median_congruence=float(np.median(base_c)) if base_c.size else np.nan))


# ---------------------------------------------------------------- shared measures


def shared_measures(P, cfg):
    """Jangle through the measure: one kind of measure filed under two different constructs.
    If two constructs are measured the same way in the same study, their 'relationship' is partly true by construction
    (e.g. technological capability = patents up to 2004, innovation performance = patents from 2005: r = .98)."""
    q = P[(P.a_construct != "") & (P.b_construct != "") & (P.a_construct != P.b_construct)
          & (P.a_measure == P.b_measure) & ~P.a_measure.isin(GENERIC_MEASURES)
          & ~P.a_construct.isin(cfg.exclude_constructs) & ~P.b_construct.isin(cfg.exclude_constructs)].copy()
    out = []
    if not q.empty:
        q["c1"] = np.minimum(q.a_construct, q.b_construct); q["c2"] = np.maximum(q.a_construct, q.b_construct)
        for (m, c1, c2), g in q.groupby(["a_measure", "c1", "c2"]):
            r, k = pool(g)
            out.append(dict(measure=m, construct_1=c1, construct_2=c2, r=r, articles=k,
                            flag=bool(r >= 0.7 and k >= cfg.min_shared_articles), examples=g.apply(lambda x: f"{x.article}: '{x.a_name}' ~ '{x.b_name}' r={x.r:+.2f}", axis=1).head(3).tolist()))
    out.sort(key=lambda t: (-t["flag"], -t["r"]))
    # how many constructs each kind of measure serves
    v = pd.concat([P[["article", "a_construct", "a_measure"]].set_axis(["article", "construct", "measure"], axis=1),
                   P[["article", "b_construct", "b_measure"]].set_axis(["article", "construct", "measure"], axis=1)])
    v = v[(v.construct != "") & ~v.measure.isin(GENERIC_MEASURES) & ~v.construct.isin(cfg.exclude_constructs)].drop_duplicates()
    spread = (v.groupby(["measure", "construct"]).article.nunique().reset_index()
              .sort_values(["measure", "article"], ascending=[True, False]))
    uses = {m: dict(zip(g.construct, g.article.astype(int))) for m, g in spread.groupby("measure") if len(g) > 1}
    return dict(pairs=out, uses=uses)


# ---------------------------------------------------------------- method
def method_check(D):
    """Common-source check, within articles: mean |r| of self-report (survey) measures with other self-report measures
    vs with measures from any other source (other-report, archival, observed)."""
    d = D[(D.x_data_type == "survey") & D.y_data_type.isin(["survey", "other-report", "archival", "observed"])].copy()
    d["az"] = z(d.r.abs())
    d["src"] = np.where(d.y_data_type == "survey", "same", "other")
    m = d.groupby(["article", "src"])["az"].mean().unstack()
    if not {"same", "other"} <= set(m.columns):
        return None
    m = m.dropna()
    if m.empty:
        return None
    by = d[d.src == "other"].groupby("y_data_type").article.nunique().to_dict()
    return dict(articles=int(len(m)), survey_with_survey=float(np.tanh(m["same"]).mean()),
                survey_with_other=float(np.tanh(m["other"]).mean()), other_sources=by)


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float); n = p.size
    if not n:
        return p
    order = np.argsort(p); ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(q, 1)
    return out


# ---------------------------------------------------------------- run
def run(variables, correlations, cfg=None, constructs=None):
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.seed)
    P = _pairs(variables, correlations)
    P["z"] = z(P.r)
    D = directed(P)
    anchors = choose_anchors(D, cfg)
    targets = constructs or [c for c in D.x_construct.unique() if c and c not in cfg.exclude_constructs]
    named = ~variables.measure.fillna("").isin(["", "unknown"])
    jingle_note = "" if named.any() else "The tables do not name the measures, so this check was skipped."
    jingles = [j for c in sorted(targets) for j in [jingle(D, P, c, anchors, cfg, rng)] if j] if named.any() else []
    # one decision rule for the whole report: a pair of measures differs if either test survives a false discovery rate of alpha
    comps = [c for j in jingles for c in j["comparisons"]]
    if comps:
        q = bh([c["p_icc"] for c in comps] + [c["p_mad"] for c in comps])
        for i, c in enumerate(comps):
            c["q_icc"], c["q_mad"] = float(q[i]), float(q[len(comps) + i])
            c["differs"] = bool(c["q_icc"] < cfg.alpha or c["q_mad"] < cfg.alpha)
        for j in jingles:
            j["differs"] = any(c["differs"] for c in j["comparisons"])
    return dict(summary=dict(articles=int(variables.article.nunique()), variables=int(len(variables)),
                             correlations=int(len(correlations)), constructs=int((variables.construct != "").sum() and variables.construct.nunique()),
                             anchors=anchors),
                jingle=jingles, jingle_note=jingle_note, jangle=jangle(D, P, anchors, cfg), shared=shared_measures(P, cfg), method=method_check(D), config=cfg.__dict__)
