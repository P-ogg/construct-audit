"""Extract one article, or a folder of articles, into the variables and correlations tables.

Per article
  1. variables  two coders list the study's variables (different models or framings, read independently)
  2. roster     one list is chosen as the numbering both matrix coders key to (the measured keying failures came
                from mismatched numbering); by default coder B's
  3. matrix     two coders transcribe the correlation matrix row by row against that list; Python pairs the cells
  4. review     values the coders agree on are accepted; every disagreement (correlation cells, survey/archival type,
                reliability, sample size) goes to a third reader (review.py); Python decides whether its answer stands
                or a human must look
  5. verify     quotes and numbers are checked against the article's text; flags are advice, never rejection
Every model reply is saved before anything else and never overwritten, so a run resumes where it stopped.
Every decision is written to a log: who decided each value and why.
"""
import base64, hashlib, io, json, os, re, zipfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as _wait
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import matrix as mx, pdf as pdfx, review as rvw, verify
from .numbers import canon_number, as_r
from ..schema import DATA_TYPES, FORMS

HERE = Path(__file__).parent
PROMPT_VARS = (HERE / "prompts" / "variables.txt").read_text()
PROMPT_MATRIX = (HERE / "prompts" / "matrix.txt").read_text()
SYSTEM = "You extract data from research articles. You copy, you never invent. You answer with JSON only."

S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
obj = lambda props: {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}
arr = lambda item: {"type": "array", "items": item}
VAR_SCHEMA = obj({"page_basis": S, "studies": arr(obj({
    "study": I, "matrix_location": S, "n_quote": S, "n": S,
    "variables": arr(obj({"variable": I, "name": S, "description_quote": S, "description": S, "instrument": S,
                          "measure_form": {"type": "string", "enum": list(FORMS)}, "source_quote": S,
                          "data_type": {"type": "string", "enum": list(DATA_TYPES)},
                          "reliability_quote": S, "reliability": S, "page": S}))}))})
MATRIX_SCHEMA = obj({"page_basis": S, "studies": arr(obj({
    "study": I, "correlation_matrix_present": B, "matrix_location": S, "matrix_quote": S, "matrix_page": S,
    "unlisted_variables": arr(S), "missing_pairs": arr(S),
    "matrix_rows": arr(obj({"variable": I, "label": S, "first_column": I, "values": arr(S)})),
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}))})


@dataclass
class Coders:
    """Who reads what. Any client from caudit.llm.make_client works for any role."""
    var_a: object
    var_b: object
    mat_a: object
    mat_b: object
    reviewer: object

    @classmethod
    def claude(cls, api_key=None):
        from ..llm import make_client
        sonnet = make_client("anthropic", model="claude-sonnet-5-5", api_key=api_key, effort="medium")
        opus = make_client("anthropic", model="claude-opus-5-5", api_key=api_key, effort="medium")
        # Opus reads the matrices: on a rotated matrix Sonnet got 16/27 cells, Opus 27/27; on keying, 651/651 vs 342/651.
        return cls(var_a=sonnet, var_b=opus, mat_a=opus, mat_b=opus, reviewer=opus)

    @classmethod
    def roles(cls, coder_a, coder_b, reviewer):
        """A different model in each role (e.g. two model families for the coders and a third for the reviewer)."""
        return cls(var_a=coder_a, var_b=coder_b, mat_a=coder_a, mat_b=coder_b, reviewer=reviewer)

    @classmethod
    def one_model(cls, client):
        """Same model in every role; independence comes from the framing (document first vs instructions first)."""
        return cls(client, client, client, client, client)

    def usage(self):
        seen, out = set(), []
        for c in (self.var_a, self.var_b, self.mat_a, self.mat_b, self.reviewer):
            if id(c) not in seen:
                seen.add(id(c)); out.append((getattr(c, "model", "?"), c.usage))
        return out


def _cached(path, fn):
    if path.exists():
        return json.loads(path.read_text())
    out = fn()
    path.write_text(json.dumps(out, indent=1))
    return out


def _doc(client, pdf_path, text):
    if getattr(client, "native_pdf", False):
        return {"pdf_b64": base64.standard_b64encode(Path(pdf_path).read_bytes()).decode()}
    return {"text": text}


def _roster_text(study):
    return "\n".join(f"Study {study['study']}, variable {v['variable']}: {v['name']}" for v in study["variables"])


def _same_name(a, b):
    a, b = verify.squash(a), verify.squash(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def _tag(client):
    """Saved replies are named after the model, so a rerun with another model never reuses them."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(getattr(client, "model", "model")))[:60] or "model"


def article_dir(pdf_path, workdir, article_id=None):
    """One folder per article; without an explicit ID it carries a fingerprint of the file, so a different PDF under the
    same name never reuses another article's replies."""
    pdf_path = Path(pdf_path)
    if article_id:
        return Path(workdir) / article_id
    return Path(workdir) / f"{pdf_path.stem}-{hashlib.md5(pdf_path.read_bytes()).hexdigest()[:8]}"


def forget_text(pdf_path, workdir, article_id=None):
    """Delete the article's extracted text (kept only while it is being read); the model replies stay for resuming."""
    for f in article_dir(pdf_path, workdir, article_id).glob("*.txt"):
        f.unlink(missing_ok=True)


def _both(f, g):
    """Run the two coders' calls at the same time: they are independent by design."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(f), pool.submit(g)
        return a.result(), b.result()


def extract_article(pdf_path, coders, workdir, article_id=None, progress=None):
    """progress(text) is called at each step, from this article's thread."""
    say = progress or (lambda s: None)
    pdf_path = Path(pdf_path); art = article_id or pdf_path.stem
    wd = article_dir(pdf_path, workdir, article_id); wd.mkdir(parents=True, exist_ok=True)
    say("extracting the text")
    text = pdfx.pdf_text(pdf_path, cache_dir=wd)
    hay = verify.haystack(text)
    log, flags = [], []

    def ask(client, prompt, schema, name, doc_first, max_tokens=64000, system=SYSTEM):
        native = getattr(client, "native_pdf", False)
        # Claude reads the PDF itself, so it gets only sideways tables turned upright; text-only models also get the
        # best upright table pages, the only way they see a table's layout and its minus signs.
        imgs = lambda: pdfx.table_images(pdf_path, include_upright_tables=not native) if getattr(client, "vision", native) else []
        return _cached(wd / f"{name}.{_tag(client)}.json", lambda: client.json(system, prompt, schema, max_tokens=max_tokens, label=f"{art} {name}",
                                                              doc=_doc(client, pdf_path, text), images=imgs(), doc_first=doc_first))

    # 1-2. variables and roster
    say("listing the variables (both coders)")
    va, vb = _both(lambda: ask(coders.var_a, PROMPT_VARS, VAR_SCHEMA, "variables_a", doc_first=True),
                   lambda: ask(coders.var_b, PROMPT_VARS, VAR_SCHEMA, "variables_b", doc_first=False))
    roster = vb
    items = []                                   # disagreements for the reviewer
    sa = {s["study"]: s for s in va["studies"]}
    for s in vb["studies"]:
        other = sa.get(s["study"])
        if not other:
            flags.append(dict(article=art, study=s["study"], check="study_count", detail="coder A did not report this study")); continue
        if len(other["variables"]) != len(s["variables"]):
            flags.append(dict(article=art, study=s["study"], check="variable_count",
                              detail=f"coder A {len(other['variables'])} variables, coder B {len(s['variables'])}"))
        if canon_number(other.get("n", "")) != canon_number(s.get("n", "")):
            items.append(rvw.item(f"n:{s['study']}", "n", f"study {s['study']}", other.get("n", ""), s.get("n", ""),
                                  other.get("n_quote", ""), s.get("n_quote", "")))
        oa = {v["variable"]: v for v in other["variables"]}
        for v in s["variables"]:
            w = oa.get(v["variable"])
            if not w:
                continue
            if not _same_name(v["name"], w["name"]):
                # different variables under one number: the coders' numbering differs, which a reviewer cannot settle field by field
                flags.append(dict(article=art, study=s["study"], check="numbering", detail=f"variable {v['variable']}: '{v['name']}' vs '{w['name']}'"))
                continue
            where = f"study {s['study']}, variable {v['variable']} ('{v['name']}')"
            if w["data_type"] != v["data_type"]:
                items.append(rvw.item(f"data_type:{s['study']}:{v['variable']}", "data_type", where, w["data_type"], v["data_type"],
                                      w.get("source_quote", ""), v.get("source_quote", "")))
            if w.get("measure_form", "") != v.get("measure_form", ""):
                items.append(rvw.item(f"measure_form:{s['study']}:{v['variable']}", "measure_form", where, w.get("measure_form", ""),
                                      v.get("measure_form", ""), w.get("description_quote", ""), v.get("description_quote", "")))
            if canon_number(w.get("reliability", "")) != canon_number(v.get("reliability", "")):
                items.append(rvw.item(f"reliability:{s['study']}:{v['variable']}", "reliability", where, w.get("reliability", ""),
                                      v.get("reliability", ""), w.get("reliability_quote", ""), v.get("reliability_quote", "")))
    roster_list = "\n".join(_roster_text(s) for s in roster["studies"])

    # 3. matrix, both coders keyed to the roster
    prompt = PROMPT_MATRIX.replace("{VARIABLE_LIST}", roster_list)
    say("copying the correlation table (both coders)")
    ma, mb = _both(lambda: ask(coders.mat_a, prompt, MATRIX_SCHEMA, "matrix_a", doc_first=True),
                   lambda: ask(coders.mat_b, prompt, MATRIX_SCHEMA, "matrix_b", doc_first=False))

    # compare cells: agreements are kept, disagreements go to the reviewer with the variable disagreements
    def cellmap(m):
        out = {}
        for s in m["studies"]:
            c, selfp, conf = mx.cells(s)
            for (a, b), v in c.items():
                out[(s["study"], a, b)] = v
            for v, val in selfp:
                flags.append(dict(article=art, study=s["study"], check="self_pair", detail=f"variable {v} paired with itself ({val})"))
            for key, v1, v2 in conf:
                flags.append(dict(article=art, study=s["study"], check="conflicting_pair", detail=f"{key}: {v1} vs {v2}"))
        return out
    ca, cb = cellmap(ma), cellmap(mb)
    final, disputed = {}, []
    for k in sorted(set(ca) | set(cb)):
        x, y = canon_number(ca.get(k, "")), canon_number(cb.get(k, ""))
        if x and x == y:
            final[k] = (ca[k], "both coders")
        else:
            disputed.append((k, ca.get(k, ""), cb.get(k, "")))
    names = {(s["study"], v["variable"]): v["name"] for s in roster["studies"] for v in s["variables"]}
    for k, a, b in disputed:
        items.append(rvw.item(f"r:{k[0]}:{k[1]}:{k[2]}", "r", f"study {k[0]}: '{names.get((k[0], k[1]), k[1])}' x "
                              f"'{names.get((k[0], k[2]), k[2])}' (variables {k[1]} and {k[2]})", a, b))

    # 4. one reviewer pass for all disagreements of the article; Python decides whether each answer stands
    if items:
        say(f"settling {len(items)} disagreement(s) (reviewer)")
    decided = rvw.settle(items, lambda p, n: ask(coders.reviewer, p, rvw.SCHEMA, f"review_{n}", doc_first=False, system=rvw.SYSTEM), hay) if items else {}
    vars_by = {(s["study"], v["variable"]): v for s in roster["studies"] for v in s["variables"]}
    study_by = {s["study"]: s for s in roster["studies"]}
    for it in items:
        d, why = decided[it["ref"]]
        kind, *key = it["ref"].split(":"); key = [int(x) for x in key]
        val = str(d.get("value", "")).strip() if d.get("answered") else ""
        usable = d.get("answered") and not rvw.valid_value(it["field"], val)
        if kind == "r":
            # a correlation is never taken from one coder: without a usable reviewer answer the cell stays out
            who = ("reviewer" if not why else "reviewer (check)") if usable and canon_number(val) else "unresolved"
            if who != "unresolved":
                final[tuple(key)] = (val, who)
        else:
            if not usable or val == "NOT_FOUND":
                val, who = it["b"], "coder B (unreviewed)"
            else:
                who = "reviewer" if not why else "reviewer (check)"
            target = study_by[key[0]] if kind == "n" else vars_by[tuple(key)]
            target[kind] = val
        log.append(dict(article=art, study=key[0], var=":".join(map(str, key[1:])), field=it["field"], a=it["a"], b=it["b"],
                        decided_by=who, value=val, confidence=d.get("confidence", ""), reason=d.get("reason", ""), quote=d.get("quote", "")))
        if why:
            flags.append(dict(article=art, study=key[0], check="needs_human",
                              detail=f"{it['where']}, {it['field']}: A {it['a'] or '-'}, B {it['b'] or '-'}, used {val or '-'} ({why})"))

    # 5. verify
    say("checking quotes and numbers")
    for s in roster["studies"]:
        if s.get("n_quote") and s["n_quote"] != "NOT_FOUND" and verify.check_quote(s["n_quote"], hay)[0] == "fail":
            flags.append(dict(article=art, study=s["study"], check="quote_not_found", detail=f"sample size quote: {s['n_quote'][:80]}"))
        for v in s["variables"]:
            q = v.get("description_quote", "")
            if q and q != "NOT_FOUND" and verify.check_quote(q, hay)[0] == "fail":
                flags.append(dict(article=art, study=s["study"], check="quote_not_found", detail=f"variable {v['variable']}: {q[:80]}"))
    for st in {k[0] for k in final}:
        cells_st = [(k, v) for k, (v, _) in final.items() if k[0] == st]
        vals = [canon_number(v) for _, v in cells_st]
        sig, uns, where = verify.locate_matrix(vals, text)
        verdicts = [(k, v, verify.number_verdict(v, sig, uns)) for k, v in cells_st]
        absent = [x for x in verdicts if x[2] == "absent"]; wrong_sign = [x for x in verdicts if x[2] == "sign"]
        minus_on_page = any(x.startswith("-") for x in sig)
        # The text layer of many PDFs garbles tables (rotated pages, unmapped minus glyphs). When most cells cannot be
        # found the page text is unusable, not the values: one flag asking for a look at the page, not one per cell.
        if len(absent) > max(2, 0.3 * len(verdicts)):
            flags.append(dict(article=art, study=st, check="matrix_text_unreadable",
                              detail=f"{len(absent)} of {len(verdicts)} values not found in the text of {where}: check the table by eye"))
        else:
            for k, v, _ in absent:
                flags.append(dict(article=art, study=st, check="number_absent", detail=f"{k[1]}x{k[2]} = {v} not on {where}"))
        negatives = sum(1 for x in vals if x.startswith("-"))
        if len(wrong_sign) > max(2, 0.3 * max(negatives, 1)):
            minus_on_page = False        # most negatives lost their sign in the text layer: the glyph is unmapped, not the values wrong
        if wrong_sign and minus_on_page:
            for k, v, _ in wrong_sign:
                flags.append(dict(article=art, study=st, check="number_sign", detail=f"{k[1]}x{k[2]} = {v}: printed with the other sign on {where}"))
        elif wrong_sign:
            flags.append(dict(article=art, study=st, check="signs_unverifiable",
                              detail=f"the text of {where} carries no minus signs, so {len(wrong_sign)} negative values could not be checked"))
        if len(vals) >= 10 and not any(x.startswith("-") for x in vals):
            flags.append(dict(article=art, study=st, check="no_negatives", detail=f"{len(vals)} cells, none negative: check the signs"))

    # tables
    vrows = []
    for s in roster["studies"]:
        for v in s["variables"]:
            vrows.append(dict(article=art, study=str(s["study"]), var=str(v["variable"]), name=v["name"], description=v["description"],
                              data_type=v["data_type"], reliability=canon_number(v.get("reliability", "")), construct="", label="", measure="",
                              measure_form=v.get("measure_form", ""), instrument=v.get("instrument", ""),
                              description_quote=v.get("description_quote", "")))
    nmap = {s["study"]: canon_number(s.get("n", "")) for s in roster["studies"]}
    crows = [dict(article=art, study=str(k[0]), var_a=str(k[1]), var_b=str(k[2]), r=canon_number(v), n=nmap.get(k[0], ""), decided_by=who)
             for k, (v, who) in sorted(final.items()) if as_r(v) is not None]
    agreed = sum(1 for _, w in final.values() if w == "both coders")
    summary = dict(article=art, studies=len(roster["studies"]), variables=len(vrows), correlations=len(crows),
                   agreed=agreed, reviewed=len(items), to_check=sum(1 for f in flags if f['check'] == 'needs_human'), flags=len(flags))
    return dict(summary=summary, variables=vrows, correlations=crows, log=log, flags=flags)


# What each check means, in plain words (the codes stay in the downloadable tables)
CHECK_TEXT = {
    "failed": "The article could not be read",
    "needs_human": "The coders disagreed and the reviewer could not settle it with confidence",
    "numbering": "The two coders numbered the variables differently",
    "variable_count": "The two coders found a different number of variables",
    "study_count": "Only one coder found this study",
    "quote_not_found": "A quote the coder gave is not in the article",
    "number_absent": "A correlation was not found on the table's page",
    "number_sign": "A correlation is printed with the other sign",
    "signs_unverifiable": "Minus signs could not be checked in the PDF's text",
    "matrix_text_unreadable": "The table's text could not be read from the PDF: check the table by eye",
    "no_negatives": "No negative correlations at all: check the signs",
    "self_pair": "A variable was paired with itself",
    "conflicting_pair": "Two different values were given for the same pair",
}


def describe_flags(flags):
    """The rows-to-check table with a plain-language column first."""
    if flags is None or flags.empty:
        return flags
    out = flags.copy()
    out.insert(0, "what to check", out["check"].map(CHECK_TEXT).fillna(out["check"]))
    return out


def extract_many(pdfs, coders, workdir, workers=5, on_done=None, on_tick=None, tick=2.0):
    """Read several articles at once (each article's own steps stay in order). Returns {pdf path: result or exception}.
    on_done(path, result_or_exception, n_done) is called as each article finishes. on_tick(status) is called every
    `tick` seconds from the calling thread with {pdf path: (current step, seconds on it)}, for a live progress display.
    Saved replies make a rerun resume."""
    out, status = {}, {}
    def track(p):
        return lambda s: status.__setitem__(p, (s, time.time()))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(extract_article, p, coders, workdir, None, track(p)): p for p in pdfs}
        pending = set(futures)
        while pending:
            done, pending = _wait(pending, timeout=tick, return_when=FIRST_COMPLETED)
            for f in done:
                p = futures[f]
                try:
                    out[p] = f.result()
                except Exception as e:               # one bad article must not stop a literature
                    out[p] = e
                status[p] = ("done" if not isinstance(out[p], Exception) else "could not be read", time.time())
                if on_done:
                    on_done(p, out[p], len(out))
            if on_tick:
                now = time.time()
                on_tick({p: (s, now - t) for p, (s, t) in status.items()})
    return out


def collect(results):
    """Merge per-article results (or exceptions) into the five tables."""
    V, C, L, F, Sm = [], [], [], [], []
    for p, r in sorted(results.items(), key=lambda t: str(t[0])):
        if isinstance(r, Exception):
            F.append(dict(article=Path(p).stem, study="", check="failed", detail=str(r)[:300])); continue
        V += r["variables"]; C += r["correlations"]; L += r["log"]; F += r["flags"]; Sm.append(r["summary"])
    return V, C, L, F, Sm


def save_progress(workdir):
    """The saved model replies of every article (JSON only: no PDFs, no article text), as a zip to resume from."""
    buf = io.BytesIO(); root = Path(workdir)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.glob("*/*.json")):
            z.write(f, f"{f.parent.name}/{f.name}")
    return buf.getvalue()


_SAFE = re.compile(r"^[^/\\]{1,200}/[A-Za-z0-9_.-]{1,160}\.json$")


def restore_progress(data, workdir):
    """Unpack a progress zip into workdir; only '<article>/<name>.json' entries are accepted. Returns the article count."""
    root = Path(workdir); root.mkdir(parents=True, exist_ok=True); arts = set()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            art, _, fname = name.partition("/")
            if not _SAFE.match(name) or art in ("", ".", "..") or art.startswith(".") or ".." in fname:
                continue
            (root / art).mkdir(exist_ok=True)
            (root / art / fname).write_bytes(z.read(name)); arts.add(art)
    return len(arts)


def extract_folder(folder, coders, workdir, progress=print, workers=5):
    pdfs = sorted(Path(folder).glob("*.pdf"))
    res = extract_many(pdfs, coders, workdir, workers, on_done=lambda p, r, n: progress(f"[{n}/{len(pdfs)}] {p.name}"))
    V, C, L, F, Sm = collect(res)
    out = Path(workdir)
    pd.DataFrame(V).to_csv(out / "variables.csv", index=False); pd.DataFrame(C).to_csv(out / "correlations.csv", index=False)
    pd.DataFrame(L).to_csv(out / "decision_log.csv", index=False); pd.DataFrame(F).to_csv(out / "flags.csv", index=False)
    pd.DataFrame(Sm).to_csv(out / "summary.csv", index=False)
    return pd.DataFrame(Sm)
