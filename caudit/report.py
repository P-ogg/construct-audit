"""Render an audit (the dict returned by audit.run) as a self-contained HTML report.

Plain language first: every finding is a sentence, the numbers sit underneath it.
"""
import html
import math

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--warn:#a4410e;--ok:#2f6b3a;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#171715;--fg:#ecebe6;--muted:#a19f97;--line:#34332f;--warn:#f0a070;--ok:#8fd19a;--card:#1f1f1c}}
body{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:0 16px}
main{max-width:980px;margin:32px auto}
h1{font-size:1.7rem;margin:.2em 0}h2{font-size:1.25rem;margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.3em}
h3{font-size:1.05rem;margin-top:1.6em}.muted{color:var(--muted)}.warn{color:var(--warn);font-weight:600}.ok{color:var(--ok);font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:8px 0}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}
details summary{cursor:pointer;margin:8px 0}.kpi{display:flex;gap:24px;flex-wrap:wrap}.kpi div{min-width:120px}.kpi b{display:block;font-size:1.4rem}
"""


def f(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def e(s):
    return html.escape(str(s))


def _differs(c):
    return c["differs"] if "differs" in c else (c["p_mad"] < 0.05 or c["p_icc"] < 0.05)


def _q(c):
    return min(c.get("q_icc", c["p_icc"]), c.get("q_mad", c["p_mad"]))


def _jingle(j):
    out = [f"<h3>{e(j['construct'])}</h3>"]
    diff = [c for c in j["comparisons"] if _differs(c)]
    if diff:
        names = "; ".join(f"{e(c['measure_1'])} vs {e(c['measure_2'])}" for c in diff)
        out.append(f"<p class='warn'>Measures of this construct behave differently: {names}.</p>")
    else:
        out.append("<p class='ok'>No pair of its measures differs at the report's threshold.</p>")
    for c in j["contamination"]:
        if c["gap"] > 0:
            out.append(f"<p>“{e(c['measure'])}” tracks <b>{e(c['anchor'])}</b> ({f(c['r'])}) more than the construct’s other measures do ({f(c['others_mean'])}).</p>")
    anchors = list(next(iter(j["profiles"].values())).keys())
    rows = "".join(f"<tr><td>{e(m)}</td><td>{j['articles'][m]}</td>" + "".join(f"<td>{f(j['profiles'][m][a])}</td>" for a in anchors) + "</tr>"
                   for m in j["measures"])
    out.append("<div class='scroll'><table><tr><th>Measure</th><th>Articles</th>" + "".join(f"<th>{e(a)}</th>" for a in anchors)
               + f"</tr>{rows}</table></div><p class='muted'>Each row is a profile: the measure’s pooled correlation with each comparison construct.</p>")
    if j["comparisons"]:
        rows = "".join(f"<tr><td>{e(c['measure_1'])} vs {e(c['measure_2'])}</td><td>{f(c['icc'])}</td><td>{f(c['null_icc_median'])}</td>"
                       f"<td>{f(_q(c),3,False)}</td><td class='{'warn' if _differs(c) else ''}'>{'differ' if _differs(c) else 'alike'}</td></tr>"
                       for c in j["comparisons"])
        out.append("<div class='scroll'><table><tr><th>Comparison</th><th>Profile correlation</th><th>If interchangeable</th><th>q</th>"
                   f"<th>Verdict</th></tr>{rows}</table></div>")
    if j["direct"]:
        rows = "".join(f"<tr><td>{e(d['measure_1'])} ~ {e(d['measure_2'])}</td><td>{f(d['r'])}</td><td>{d['articles']}</td></tr>" for d in j["direct"])
        out.append(f"<p>Where one study uses two of these measures, they correlate:</p><div class='scroll'><table><tr><th>Measures</th><th>r</th><th>Articles</th></tr>{rows}</table></div>")
    return "<div class='card'>" + "".join(out) + "</div>"


def _guide(cfg):
    a = cfg.get("alpha", 0.05)
    return f"""<details class='card'><summary><b>How to read this report</b></summary>
<p><b>Profile.</b> For each measure, its pooled correlations with a fixed set of common constructs (the comparison constructs).
Two measures of one construct should have similar profiles.</p>
<p><b>Profile correlation.</b> How alike two profiles are: +1 identical pattern, 0 unrelated, −1 opposite.
<b>If interchangeable</b> is the median value when the same studies are randomly re-split between the two measures:
what to expect if the measures were the same thing.</p>
<p><b>Decision rule.</b> A pair of measures <i>differs</i> if its profile correlation is lower than in the random splits, or the average
gap between the two profiles is larger, by a permutation test ({cfg.get('n_perm', '')} random splits). The p-values of all tests in the
report are corrected for multiple testing (Benjamini–Hochberg); a pair differs when q &lt; {a}.</p>
<p><b>Jangle verdicts.</b> <i>Possible duplicate</i>: the two constructs correlate at least {cfg.get('duplicate_r', .85)} once measurement
error is removed (from at least {cfg.get('min_corrected_articles', 5)} articles reporting both reliabilities) <i>and</i> their profiles
are among the {100 - cfg.get('profile_pct', 90):.0f}% most similar of all pairs. <i>Siblings</i>: correlate at least .40 and similar
profiles. <i>Related</i>: correlate at least .40. <i>Distinct</i>: the rest.</p>
<p><b>One measure, two constructs.</b> The same measure filed under two constructs that correlate .70 or more in at least
{cfg.get('min_shared_articles', 2)} articles: their relationship may be true by construction.</p>
<p><b>Minimum evidence.</b> Comparison constructs appear in at least {cfg.get('min_anchor_articles', '')} articles; a measure needs
{cfg.get('min_measure_articles', '')} articles to get a profile; each profile entry needs {cfg.get('min_cell_articles', '')} articles.</p>
</details>"""


def render(res, title="Construct audit", note=""):
    s = res["summary"]; cfg = res.get("config") or {}
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{e(title)}</title><style>{CSS}</style></head><body><main>",
             f"<h1>{e(title)}</h1><div class='kpi'><div><b>{s['articles']}</b>articles</div><div><b>{s['variables']}</b>variables</div>"
             f"<div><b>{s['correlations']}</b>correlations</div></div>"]
    if note:
        parts.append(f"<div class='card'><p class='warn'>{e(note)}</p></div>")
    parts.append(f"<p class='muted'>Comparison constructs: {e(', '.join(s['anchors']))}.</p>" if s["anchors"] else
                 f"<p class='warn'>Too few articles to choose comparison constructs (each needs at least {cfg.get('min_anchor_articles', '')} "
                 "articles), so most checks could not run.</p>")
    parts.append(_guide(cfg))
    parts.append("<h2>Jingle: one name, measures that behave differently</h2>"
                 "<p>For each construct, each measure gets a profile. Profiles are compared with each other and with random splits of the same studies.</p>")
    parts += [_jingle(j) for j in res["jingle"]] or [f"<p class='warn'>{e(res.get('jingle_note') or 'No construct has two measures with enough articles.')}</p>"]
    jg = res["jangle"]
    level = "author labels (cleaned)" if jg.get("level") == "label" else "constructs"
    parts.append(f"<h2>Jangle: two names, one thing?</h2><p>Compared at the level of {level}. A pair counts as a possible duplicate only if the "
                 "two correlate near 1 once measurement error is removed <i>and</i> relate to other constructs almost identically.</p>")
    dup = [p for p in jg["pairs"] if p["verdict"] == "possible duplicate"]
    sib = [p for p in jg["pairs"] if p["verdict"] == "siblings"]
    parts.append(f"<p class='{'warn' if dup else 'ok'}'>{len(dup)} possible duplicate pair(s); {len(sib)} sibling pair(s) (related, similar profiles, not the same).</p>")
    show = (dup + sib + [p for p in jg["pairs"] if p["verdict"] == "related"])[:30]
    if show:
        rows = "".join(f"<tr><td>{e(p['unit_1'])} ~ {e(p['unit_2'])}</td><td>{e(p['verdict'])}</td><td>{f(p['r'])}</td><td>{p['articles']}</td>"
                       f"<td>{f(p.get('r_corrected'))}</td><td>{f(p.get('profile_icc'))}</td><td>{f(p.get('icc_percentile'),0,False)}</td>"
                       f"<td>{e(p.get('note', ''))}</td></tr>" for p in show)
        parts.append("<div class='scroll'><table><tr><th>Pair</th><th>Verdict</th><th>r</th><th>Articles</th><th>Corrected r</th>"
                     f"<th>Profile correlation</th><th>Percentile</th><th>Note</th></tr>{rows}</table></div>"
                     "<p class='muted'>Percentile: how similar the two profiles are, compared with all pairs (100 = most similar).</p>")
    sh = res.get("shared") or {}
    parts.append("<h2>One measure, two constructs</h2><p>When studies file the same measure under different constructs, "
                 "a relationship between those constructs can be true by construction.</p>")
    flagged = [p for p in sh.get("pairs", []) if p["flag"]]
    parts.append(f"<p class='{'warn' if flagged else 'ok'}'>{len(flagged)} pair(s) of constructs measured the same way and correlating at .70 or more "
                 f"in at least {cfg.get('min_shared_articles', 2)} articles.</p>")
    ex = any(p.get("examples") for p in sh.get("pairs", []))
    row = lambda p: (f"<tr><td>{e(p['measure'])}</td><td>{e(p['construct_1'])} ~ {e(p['construct_2'])}</td><td>{f(p['r'])}</td><td>{p['articles']}</td>"
                     + (f"<td>{e('; '.join(p.get('examples', [])[:2]))}</td>" if ex else "") + "</tr>")
    head = ("<div class='scroll'><table><tr><th>Measure</th><th>Constructs</th><th>r</th><th>Articles</th>"
            + ("<th>Examples</th>" if ex else "") + "</tr>")
    if flagged:
        parts.append(head + "".join(row(p) for p in flagged) + "</table></div>")
    rest = [p for p in sh.get("pairs", []) if not p["flag"]]
    if rest:
        parts.append(f"<details><summary>Other pairs measured the same way ({len(rest)})</summary>{head}{''.join(row(p) for p in rest[:40])}</table></div></details>")
    if sh.get("uses"):
        rows = "".join(f"<tr><td>{e(m)}</td><td>{len(u)}</td><td>{e('; '.join(f'{c} ({n})' for c, n in list(u.items())[:6]))}</td></tr>"
                       for m, u in sorted(sh["uses"].items(), key=lambda t: -len(t[1])))
        parts.append("<details><summary>Measures used for more than one construct</summary><div class='scroll'><table><tr><th>Measure</th>"
                     f"<th>Constructs</th><th>Used for (articles)</th></tr>{rows}</table></div></details>")
    m = res.get("method")
    if m:
        other = m.get("survey_with_other", m.get("survey_with_archival"))
        srcs = ", ".join(f"{k} ({n} articles)" for k, n in (m.get("other_sources") or {}).items())
        parts.append(f"<h2>Method</h2><p>Within the same study, self-report measures correlate on average {f(m['survey_with_survey'],2,False)} with other "
                     f"self-report measures and {f(other,2,False)} with measures from other sources ({m['articles']} articles"
                     + (f"; other sources: {e(srcs)}" if srcs else "") + "). A large gap is consistent with common-source variance.</p>")
    parts.append("<p class='muted'>Findings are evidence for human researchers to judge, not verdicts.</p></main></body></html>")
    return "".join(parts)
