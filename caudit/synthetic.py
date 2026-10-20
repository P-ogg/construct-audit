"""A synthetic literature with known problems, for testing the audit.

Latent model per article (n firms):
  size, age                        common controls
  innovation (IP) measured two ways:
     'patent counts'  = 0.6*size + 0.6*innovation + noise      (tracks size: contamination + jingle)
     'sales share'    = 0.8*innovation + noise                  (does not track size)
  firm size measured two ways ('employees', 'sales')  = size + small noise   (no jingle)
  X and Y: two names for one latent trait (jangle, corr ~ .95 after correction)
  S and T: siblings, latent corr .5
"""
import numpy as np
import pandas as pd


def make(n_articles=60, n=250, seed=0):
    rng = np.random.default_rng(seed)
    V, C = [], []
    for a in range(n_articles):
        art = f"A{a:03d}"
        N = int(rng.integers(120, n * 2))
        size, age, inn, trait, s0, t0, perf = rng.standard_normal((7, N))
        age = 0.3 * size + np.sqrt(1 - .09) * age
        inn = 0.3 * age + np.sqrt(1 - .09) * inn
        t0 = 0.5 * s0 + np.sqrt(.75) * t0
        perf = 0.3 * inn + 0.3 * trait + 0.2 * s0 + 0.8 * perf
        e = lambda w: w * rng.standard_normal(N)
        cols = {}
        # each article measures innovation with ONE kind (sometimes both), size with one kind
        kind = rng.choice(["patent counts", "sales share", "both"], p=[.45, .4, .15])
        if kind in ("patent counts", "both"):
            cols[("innovation performance", "patent counts", "archival", np.nan, "Innovation performance")] = 0.6 * size + 0.6 * inn + e(.5)
        if kind in ("sales share", "both"):
            cols[("innovation performance", "sales share", "archival", np.nan, "Innovative sales")] = 0.8 * inn + e(.6)
        if rng.random() < .5:
            cols[("firm size", "employees", "archival", np.nan, "Firm size")] = size + e(.2)
        else:
            cols[("firm size", "sales", "archival", np.nan, "Size")] = size + e(.25)
        if rng.random() < .3:  # some studies report both size measures
            cols[("firm size", "sales", "archival", np.nan, "Sales")] = size + e(.25)
        cols[("firm age", "years", "archival", np.nan, "Firm age")] = age + e(.1)
        cols[("performance", "survey scale", "survey", .85, "Performance")] = perf + e(.4)
        if rng.random() < .5:  # jangle pair: two names for one trait
            cols[("trait X", "survey scale", "survey", .85, "Trait X")] = trait + e(.4)
            cols[("trait Y", "survey scale", "survey", .85, "Trait Y")] = trait + e(.4)
        if rng.random() < .5:  # siblings
            cols[("sibling S", "survey scale", "survey", .85, "Sibling S")] = s0 + e(.4)
            cols[("sibling T", "survey scale", "survey", .85, "Sibling T")] = t0 + e(.4)
        keys = list(cols); X = np.column_stack([cols[k] for k in keys]); R = np.corrcoef(X, rowvar=False)
        for i, (con, meas, dt, rel, name) in enumerate(keys):
            V.append(dict(article=art, study="1", var=str(i + 1), name=name, description=f"{name} measured as {meas}",
                          data_type=dt, reliability=rel, construct=con, label=con, measure=meas))
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                C.append(dict(article=art, study="1", var_a=str(i + 1), var_b=str(j + 1), r=round(float(R[i, j]), 3), n=N))
    return pd.DataFrame(V), pd.DataFrame(C)
