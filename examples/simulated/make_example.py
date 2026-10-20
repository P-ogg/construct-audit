"""A simulated literature with planted problems, and its audit report: the public worked example.

    python examples/simulated/make_example.py      (writes this folder's report and the website's copy)

Every number here is invented. 160 simulated studies of employee attitudes and performance, each measuring a random
subset of constructs with one of the measures real studies use. Planted problems (the report should find them):
  jingle   "job performance": self-rated performance partly reflects job satisfaction and a shared self-report bias;
           supervisor-rated performance does not.
  jangle   "work engagement" and "job involvement" are built from the same latent variable (two names, one thing).
  shared   some studies measure organisational identification with the OCQ, the same commitment scale others use for
           organisational commitment.
  method   all self-report measures share a common-source factor.
Nothing here says anything about the real constructs.
"""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from caudit import audit, report, schema

# construct -> list of (measure name, form, data type, instrument, reliability, builder(latent) -> score)
def scale(*w, noise=0.45, method=0.3):
    return lambda L, e: sum(c * L[k] for k, c in w) + method * L["M"] + e(noise)


MEASURES = {
    "job autonomy": [("WDQ autonomy scale", "multi-item scale", "survey", "Morgeson & Humphrey 2006", .87, scale(("A", 1)))],
    "supervisor support": [("perceived supervisor support", "multi-item scale", "survey", "Kottke & Sharafinski 1988", .90, scale(("S", 1)))],
    "tenure": [("years in the organisation (HR records)", "time or age", "archival", "", None, lambda L, e: L["T"] + e(.1))],
    "age": [("age in years", "time or age", "archival", "", None, lambda L, e: L["G"] + e(.1))],
    "job satisfaction": [("JDI (Smith et al.)", "multi-item scale", "survey", "Smith, Kendall & Hulin 1969", .88, scale(("SAT", 1))),
                         ("MSQ short form", "multi-item scale", "survey", "Weiss et al. 1967", .86, scale(("SAT", 1)))],
    "organisational commitment": [("ACS (Allen & Meyer)", "multi-item scale", "survey", "Allen & Meyer 1990", .85, scale(("COM", 1))),
                                  ("OCQ (Mowday et al.)", "multi-item scale", "survey", "Mowday, Steers & Porter 1979", .88, scale(("COM", 1)))],
    "organisational identification": [("Mael & Ashforth scale", "multi-item scale", "survey", "Mael & Ashforth 1992", .87, scale(("ID", 1))),
                                      ("OCQ (Mowday et al.)", "multi-item scale", "survey", "Mowday, Steers & Porter 1979", .88, scale(("COM", 1)))],
    "work engagement": [("UWES-9", "multi-item scale", "survey", "Schaufeli et al. 2006", .90, scale(("ENG", 1)))],
    "job involvement": [("JIQ (Kanungo)", "multi-item scale", "survey", "Kanungo 1982", .87, scale(("ENG", 1)))],
    "turnover intention": [("intention to quit, 3 items", "multi-item scale", "survey", "", .85, scale(("TI", 1)))],
    "job performance": [("supervisor-rated task performance", "multi-item scale", "other-report", "Williams & Anderson 1991", .90,
                         lambda L, e: L["PERF"] + e(.45)),
                        ("self-rated performance", "multi-item scale", "survey", "", .84,
                         lambda L, e: .5 * L["PERF"] + .6 * L["SAT"] + .5 * L["M"] + e(.45))],
}
P_INCLUDE = {"job autonomy": .55, "supervisor support": .55, "tenure": .7, "age": .7, "job satisfaction": .65,
             "organisational commitment": .5, "organisational identification": .25, "work engagement": .4, "job involvement": .3,
             "turnover intention": .45, "job performance": .6}


def latent(n, rng):
    z = lambda: rng.standard_normal(n)
    A, S, T, G, M = z(), z(), z(), z(), z()
    T = .5 * G + .85 * T
    SAT = .45 * A + .4 * S + .78 * z()
    ENG = .5 * A + .3 * S + .1 * T + .8 * z()
    COM = .5 * SAT + .3 * S + .15 * T + .78 * z()
    ID = .6 * COM + .8 * z()
    PERF = .35 * ENG + .2 * T + .9 * z()
    TI = -.5 * SAT - .3 * COM + .8 * z()
    return dict(A=A, S=S, T=T, G=G, M=M, SAT=SAT, ENG=ENG, COM=COM, ID=ID, PERF=PERF, TI=TI)


def make(n_articles=160, seed=7):
    rng = np.random.default_rng(seed)
    V, C = [], []
    for a in range(1, n_articles + 1):
        art = f"S{a:03d}"
        n = int(rng.integers(150, 600)); L = latent(n, rng); e = lambda s: s * rng.standard_normal(n)
        chosen = [c for c in MEASURES if rng.random() < P_INCLUDE[c]]
        if len(chosen) < 3:
            chosen = list(rng.choice(list(MEASURES), 3, replace=False))
        cols, rows = [], []
        for c in chosen:
            opts = MEASURES[c]
            name, form, dt, inst, rel, build = opts[int(rng.integers(len(opts)))]
            cols.append(build(L, e))
            rows.append(dict(article=art, study="1", var=str(len(cols)), name=c.capitalize(), description=f"{name}", data_type=dt,
                             reliability=("" if rel is None else round(rel + rng.normal(0, .02), 2)), construct=c, label=c,
                             measure=name, measure_form=form, instrument=inst))
        R = np.corrcoef(np.vstack(cols))
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                C.append(dict(article=art, study="1", var_a=str(i + 1), var_b=str(j + 1), r=round(float(R[i, j]), 2), n=n))
        V += rows
    return schema.load_variables(pd.DataFrame(V)), schema.load_correlations(pd.DataFrame(C))


NOTE = ("Simulated data: 160 invented studies of employee attitudes and performance with planted problems, to show what a "
        "report looks like. Planted: self-rated performance partly reflects job satisfaction (jingle); work engagement and job "
        "involvement are one variable under two names (jangle); some studies measure identification with a commitment scale "
        "(one measure, two constructs), which also makes commitment and identification look like duplicates; all self-reports share a "
        "common-source factor (method). Nothing here describes the real constructs.")

if __name__ == "__main__":
    V, C = make()
    res = audit.run(V, C, audit.Config(n_perm=2000, jangle_level="construct"))
    html = report.render(res, "Construct audit: a simulated literature", note=NOTE)
    for path in (os.path.join(HERE, "audit_report.html"), os.path.join(HERE, "..", "..", "site", "examples", "audit_report.html")):
        os.makedirs(os.path.dirname(path), exist_ok=True); open(path, "w").write(html)
    print(f"{V.article.nunique()} studies, {len(V)} variables, {len(C)} correlations")
    for j in res["jingle"]:
        print("jingle", j["construct"], j.get("differs"), [(c["measure_1"], c["measure_2"], round(c["icc"], 2), round(min(c["q_icc"], c["q_mad"]), 4)) for c in j["comparisons"]])
        if j["construct"] == "job performance":
            print("   profiles:", {m: {k: round(v, 2) for k, v in p.items() if v == v and v is not None} for m, p in j["profiles"].items()})
    for p in res["jangle"]["pairs"]:
        if p["verdict"] in ("possible duplicate", "siblings"):
            print("jangle", p["unit_1"], p["unit_2"], p["verdict"], round(p["r"], 2), p.get("r_corrected"), p.get("articles_corrected"), p.get("note"))
    print("shared flagged:", [(p["measure"], p["construct_1"], p["construct_2"], round(p["r"], 2), p["articles"]) for p in res["shared"]["pairs"] if p["flag"]])
    print("method:", res["method"])
