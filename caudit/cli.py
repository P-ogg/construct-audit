"""Command line.

  caudit run PDF_DIR -o WORK            articles in, audit report out (extract -> codebook -> measures -> audit -> report)
  caudit extract PDF_DIR -o WORK        only extraction: WORK/variables.csv, correlations.csv, decision_log.csv, flags.csv
  caudit codebook WORK [--codebook F]   draft a codebook with the model (or apply the researcher's F: columns label,construct),
                                        then name the measures with the model (--measures basic: form + cited scale only, for runs without a model)
  caudit audit VARIABLES CORRELATIONS   audit coded tables and write the HTML report

Models: --provider anthropic (default, ANTHROPIC_API_KEY) or an open-weights model through any OpenAI-compatible server:
  --provider openrouter --model qwen/qwen3-235b-a22b      (OPENROUTER_API_KEY)
  --provider lmstudio --model qwen3.5-2b                  (local, no key)

A different model in each role (PROVIDER:MODEL; the default model is used for any role left out):
  caudit extract PDFS -o WORK --coder-a openrouter:qwen/qwen3-235b-a22b --coder-b openrouter:meta-llama/llama-4-maverick \
                              --reviewer anthropic:claude-opus-5-5
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
from . import audit, schema, measures, report, labels, codebook, llm


def _client(a):
    if a.provider == "anthropic":
        return None
    return llm.make_client(a.provider, model=a.model, base_url=a.base_url, vision=a.vision)


def _dump(res, path):
    json.dump(res, open(path, "w"), indent=1, default=lambda o: None if o != o else (float(o) if hasattr(o, "__float__") else str(o)))


def _role(spec, a):
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        return llm.make_client("anthropic", model=model or None, effort="medium")
    return llm.make_client(provider, model=model, base_url=None, vision=a.vision)


def do_extract(a):
    from .extract import run
    c = _client(a)
    if a.coder_a or a.coder_b or a.reviewer:
        base = c or llm.make_client("anthropic", effort="medium")
        pick = lambda spec: _role(spec, a) if spec else base
        coders = run.Coders.roles(pick(a.coder_a), pick(a.coder_b), pick(a.reviewer))
    else:
        coders = run.Coders.claude() if c is None else run.Coders.one_model(c)
    coders.framing = a.framing
    summ = run.extract_folder(a.pdfs, coders, a.out, workers=a.workers)
    cost = [(m, u.cost(m)) for m, u in coders.usage()]
    print(summ.to_string(index=False)); print("model cost:", ", ".join(f"{m} ${x:.2f}" if x is not None else f"{m} (price unknown)" for m, x in cost))


def do_codebook(a):
    work = Path(a.work)
    V = labels.apply(schema.load_variables(work / "variables.csv"))
    if a.codebook:
        assign = codebook.load(a.codebook)
    else:
        c = _client(a) or llm.make_client("anthropic")
        cons, assign = codebook.draft(V, c)
        cons, assign = codebook.consolidate(cons, assign, c)
        cons.to_csv(work / "codebook_constructs.csv", index=False)
    assign.to_csv(work / "codebook.csv", index=False)
    V = codebook.apply(V, assign)
    if a.measures == "model":
        V, names = measures.draft(V.assign(measure=""), _client(a) or llm.make_client("anthropic"))
    else:
        V = measures.fallback(V.assign(measure=""))
        names = V.groupby("measure").agg(articles=("article", "nunique"), variables=("var", "size")).reset_index()
    names.to_csv(work / "measures.csv", index=False)
    V.to_csv(work / "variables.csv", index=False)
    print(f"codebook written to {work / 'codebook.csv'}, measure names to {work / 'measures.csv'}: review them; "
          "edit construct names in the codebook and rerun with --codebook")


def do_audit(a):
    V = measures.apply(labels.apply(schema.load_variables(a.variables))); C = schema.load_correlations(a.correlations)
    cfg = audit.Config(anchors=tuple(a.anchors), exclude_constructs=tuple(a.exclude), n_perm=a.perm,
                       jangle_level="construct" if (V.construct != "").any() else "label")
    res = audit.run(V, C, cfg, constructs=a.constructs)
    open(a.out, "w").write(report.render(res, a.title))
    if a.json:
        _dump(res, a.json)
    print(f"wrote {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="caudit", description="Audit the constructs and measures of a literature.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    def models(p):
        p.add_argument("--provider", default="anthropic"); p.add_argument("--model"); p.add_argument("--base-url")
        p.add_argument("--vision", action="store_true", help="the open-weights model can read images (send table pages)")
        p.add_argument("--workers", type=int, default=5, help="articles read at once (default 5)")
        p.add_argument("--framing", choices=["mixed", "article-first"], default="mixed",
                       help="article-first: both coders read the article first, so Claude's prompt cache is shared (cheaper)")
        for role in ("--coder-a", "--coder-b", "--reviewer"):
            p.add_argument(role, metavar="PROVIDER:MODEL", help="a different model for this role")
    e = sub.add_parser("extract"); e.add_argument("pdfs"); e.add_argument("-o", "--out", required=True); models(e)
    c = sub.add_parser("codebook"); c.add_argument("work"); c.add_argument("--codebook"); models(c)
    c.add_argument("--measures", choices=["model", "basic"], default="model")
    au = sub.add_parser("audit"); au.add_argument("variables"); au.add_argument("correlations")
    au.add_argument("-o", "--out", default="audit_report.html"); au.add_argument("--json")
    au.add_argument("--anchors", nargs="*", default=()); au.add_argument("--exclude", nargs="*", default=())
    au.add_argument("--constructs", nargs="*"); au.add_argument("--perm", type=int, default=2000); au.add_argument("--title", default="Construct audit")
    r = sub.add_parser("run"); r.add_argument("pdfs"); r.add_argument("-o", "--out", required=True); r.add_argument("--codebook"); models(r)
    r.add_argument("--measures", choices=["model", "basic"], default="model")
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        do_extract(a)
    elif a.cmd == "codebook":
        do_codebook(a)
    elif a.cmd == "audit":
        do_audit(a)
    elif a.cmd == "run":
        do_extract(a)
        a.work = a.out; do_codebook(a)
        w = Path(a.out)
        do_audit(argparse.Namespace(variables=w / "variables.csv", correlations=w / "correlations.csv", out=str(w / "audit_report.html"),
                                    json=str(w / "audit.json"), anchors=(), exclude=(), constructs=None, perm=2000, title="Construct audit"))


if __name__ == "__main__":
    sys.exit(main())
