"""Web app: articles in, construct audit out.

Run locally:  streamlit run app/streamlit_app.py
The model key a visitor pastes stays in their browser session (st.session_state) and is passed straight to the
model client. It is never written to disk or to os.environ, which a hosted server shares between visitors, and the
app never falls back on a key the server itself might have (use_env=False).
Uploaded PDFs and their extracted text are deleted as soon as each article has been read; only the model's replies
are kept for the session (for resuming). Working folders left by closed sessions are deleted after a day.
"""
import io, shutil, sys, tempfile, time, zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import hashlib, importlib  # noqa: E402


def _fresh_caudit():
    """Streamlit Cloud reloads this script after a push but keeps the library modules it already imported, so new app
    code can meet old library code. Reload the library whenever its source changed since it was imported."""
    import caudit
    digest = hashlib.md5(b"".join(f.read_bytes() for f in sorted((ROOT / "caudit").rglob("*.py")))).hexdigest()
    if getattr(caudit, "_source_digest", None) != digest:
        names = sorted((n for n in sys.modules if n == "caudit" or n.startswith("caudit.")), key=lambda n: -n.count("."))
        for n in names:                      # deepest first, so packages see their reloaded submodules
            importlib.reload(sys.modules[n])
        caudit._source_digest = digest


_fresh_caudit()
from caudit import audit, codebook, labels, llm, measures, report, schema, template  # noqa: E402
from caudit.extract import run  # noqa: E402

MAX_PDFS = 100
ROUND = 10         # articles per round; the page refreshes between rounds so progress can be saved
CLAUDE_PER_ARTICLE = 1.0     # US$, rough
HOSTED = str(ROOT).startswith("/mount/src")      # Streamlit Community Cloud: "this computer" is the server, not the visitor's
WORKERS = 2 if HOSTED else 5   # articles read at once; the free hosting has about 1 GB of memory
SERVICES = {  # label shown -> (provider id, server address, where to get a key)
    "OpenRouter: hundreds of open models": ("openrouter", "https://openrouter.ai/api/v1", "https://openrouter.ai/keys"),
    "Groq: fast open models": ("groq", "https://api.groq.com/openai/v1", "https://console.groq.com/keys"),
    **({} if HOSTED else {"LM Studio on this computer": ("lmstudio", "http://localhost:1234/v1", ""),
                          "Ollama on this computer": ("ollama", "http://localhost:11434/v1", "")}),
    "Another OpenAI-compatible service": ("openai-compat", "", ""),
}
SERVICE_NAME = {"anthropic": "Anthropic", "openrouter": "OpenRouter", "groq": "Groq", "openai-compat": "Other service"}
PLACEHOLDER = {"anthropic": llm.ANTHROPIC_MODEL, "openrouter": "e.g. qwen/qwen3.8-27b", "groq": "e.g. qwen/qwen3.6-27b",
               "lmstudio": "the model loaded in LM Studio", "ollama": "e.g. qwen3:8b", "openai-compat": "the service's model name"}
ROLE_SERVICES = {"Claude (Anthropic)": ("anthropic", "", "https://console.anthropic.com/settings/keys"), **SERVICES}
SUMMARY_COLUMNS = {"article": "Article", "studies": "Studies", "variables": "Variables", "correlations": "Correlations",
                   "agreed": "Values both coders agreed on", "reviewed": "Disagreements sent to the reviewer",
                   "to_check": "Left for you to check", "flags": "Things to check"}

st.set_page_config(page_title="Construct audit", page_icon="🔎", layout="wide")
ss = st.session_state
if "work" not in ss:
    # delete working folders of sessions that ended more than a day ago
    for d in Path(tempfile.gettempdir()).glob("caudit_*"):
        try:
            if time.time() - d.stat().st_mtime > 86400:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass
    ss.work = Path(tempfile.mkdtemp(prefix="caudit_"))
ss.setdefault("version", 0)

st.title("Construct audit")
tab_run, tab_about = st.tabs(["Audit your articles", "How it works"])

# ------------------------------------------------------------------ run
with tab_run:
    st.subheader("1. Articles")
    pdfs = st.file_uploader(f"Upload the articles (PDF, up to {MAX_PDFS})", type="pdf", accept_multiple_files=True,
                            help="Each article should report a correlation table. Each article takes a few minutes, so 100 articles take a few hours: "
                                 "keep the tab open and save your progress now and then. PDFs are deleted as soon as they have been read.")
    names = [f.name for f in pdfs or []]
    if len(set(names)) < len(names):
        st.warning("Some PDFs have the same file name. Rename them so each article has its own name; otherwise one replaces the other.")
    st.caption("Already have coded tables? Upload them instead of the PDFs. No model key is needed to audit coded tables.")
    st.download_button("Download the template (Excel)", template.build(), file_name="construct_audit_template.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       help="One sheet for variables, one for correlations, an optional codebook sheet, and instructions with an example.")
    ups = st.file_uploader("Coded tables: the filled-in template, or two CSV files", type=["xlsx", "csv"], accept_multiple_files=True,
                           help="Required: article, study, var, name (variables) and article, study, var_a, var_b, r (correlations). "
                                "Optional: description, data_type, reliability, construct, measure, measure_form, instrument, n.")
    if ups:
        try:
            ss.V, ss.C, notes = template.read(ups)
            ss.codebook_sheet = template.read_codebook(ups)
            st.success(f"Loaded {ss.V.article.nunique()} articles, {len(ss.V)} variables, {len(ss.C)} correlations.")
            if ss.codebook_sheet is not None:
                st.info(f"Codebook sheet found: {ss.codebook_sheet.construct.nunique()} constructs for {len(ss.codebook_sheet)} variable names.")
            for n in notes:
                st.warning(n)
        except template.UploadError as e:
            ss.V = ss.C = None; st.error(str(e))
        except Exception as e:
            ss.V = ss.C = None; st.error(f"The file could not be read ({str(e)[:150]}). The template shows the expected layout.")
    cb = st.file_uploader("Your codebook (optional, CSV with columns 'variable name' and 'construct')", type="csv",
                          help="Which construct each variable name belongs to. Without one, the model drafts a codebook for you to check.")
    ss.codebook_upload = None
    if cb:
        try:
            ss.codebook_upload = codebook.load(template._csv(cb))
            st.success(f"Codebook loaded: {ss.codebook_upload.construct.nunique()} constructs for {len(ss.codebook_upload)} variable names.")
        except Exception as e:
            st.error(f"The codebook could not be read: {str(e)[:150]}.")

    st.subheader("2. Model")
    st.caption("An API key is like a password that lets the app use a language model on your account. "
               "You pay the model provider directly; the app itself is free.")
    kind = st.radio("Which model reads the articles?", ["Claude (Anthropic)", "An open model", "A different model for each role"],
                    horizontal=True, help="Open models are cheaper and anyone can rerun them. "
                                          "Different models for the two coders and the reviewer make their errors less alike.")
    roles, keys, priced = None, {}, True
    if kind.startswith("Claude"):
        key = st.text_input("Anthropic API key", type="password", help="Used only in this session. Never stored.")
        st.caption("Get a key at [console.anthropic.com](https://console.anthropic.com/settings/keys). About US$1 per article.")
        provider, model, base, vision = "anthropic", None, None, True
    elif kind.startswith("An open"):
        svc_label = st.selectbox("Service", list(SERVICES))
        provider, base, keypage = SERVICES[svc_label]
        if provider == "openai-compat" or not HOSTED and provider in ("lmstudio", "ollama"):
            base = st.text_input("Server address", value=base, help="The service's OpenAI-compatible address, ending in /v1.")
        model = st.text_input("Model name", placeholder=PLACEHOLDER.get(provider, "model name"),
                              help="The exact name from the provider's list of models.")
        local = provider in ("lmstudio", "ollama")
        key = "" if local else st.text_input("API key", type="password", help="Used only in this session. Never stored.")
        if keypage:
            st.caption(f"Get a key at [{keypage.split('/')[2]}]({keypage}). The provider charges your account; the app cannot show the cost.")
        vision = st.checkbox("This model can read images", help="Then it also sees pictures of the table pages, which helps with signs and layout.")
        priced = False
    else:
        st.caption("Two coders read each article separately. The reviewer settles only what they disagree on. "
                   "Pick any service for each role; a key box appears below for each service you pick.")
        roles, cols = {}, st.columns(3)
        for col, (role, default) in zip(cols, (("Coder A", 0), ("Coder B", 1), ("Reviewer", 0))):
            with col:
                lab = st.selectbox(role, list(ROLE_SERVICES), index=default, key=f"svc_{role}")
                svc = ROLE_SERVICES[lab][0]
                # keyed by service, so switching service does not keep another service's model name
                mdl = st.text_input("Model", key=f"mdl_{role}_{svc}", value=llm.ANTHROPIC_MODEL if svc == "anthropic" else "",
                                    placeholder=PLACEHOLDER.get(svc, "model name"))
                roles[role] = (svc, mdl, ROLE_SERVICES[lab][1])
        for svc in dict.fromkeys(v[0] for v in roles.values()):
            if svc not in ("lmstudio", "ollama"):
                keys[svc] = st.text_input(f"{SERVICE_NAME.get(svc, svc)} API key", type="password", key=f"key_{svc}",
                                          help="Used only in this session. Never stored.")
        if any(v[0] == "openai-compat" for v in roles.values()):
            keys["_base"] = st.text_input("Server address for the other service", key="base_other")
        vision = st.checkbox("The open models you picked can read images", help="Then they also see pictures of the table pages.")
        provider, model, base = "roles", None, None
        priced = all(v[0] == "anthropic" for v in roles.values())
        key = all(v for k, v in keys.items()) and all(m for _, m, _ in roles.values())

    def role_client(svc, mdl, url):
        if svc == "anthropic":
            return llm.make_client("anthropic", api_key=keys.get("anthropic") or None, model=mdl or llm.ANTHROPIC_MODEL, effort="medium", use_env=False)
        return llm.make_client(svc, model=mdl, base_url=(keys.get("_base") if svc == "openai-compat" else url) or None,
                               api_key=keys.get(svc) or None, vision=vision, use_env=False)

    def client(role_model=None):
        if provider == "roles":
            return role_client(*roles["Reviewer"])
        if provider == "anthropic":
            return llm.make_client("anthropic", api_key=key or None, model=role_model or llm.ANTHROPIC_MODEL, effort="medium", use_env=False)
        return llm.make_client(provider, model=model, base_url=base or None, api_key=key or None, vision=vision, use_env=False)

    has_model = bool(key if provider in ("anthropic", "roles") else model)

    def cost_text(clients):
        if not priced:
            return "The provider charges your account; see its website for the cost."
        return f"Model cost about US${sum((u.cost(m) or 0) for m, u in clients):.2f}."

    def prepare():
        """After reading (or uploading): group variables into constructs and name the measures, unless the researcher gave them."""
        V = labels.apply(ss.V)
        ss.version += 1
        try:
            c = client() if has_model else None
            given = ss.get("codebook_upload") if ss.get("codebook_upload") is not None else ss.get("codebook_sheet")
            if given is not None:
                ss.codebook = given
            elif not (V.construct.fillna("") != "").any() and c is not None:
                with st.spinner("Grouping the variables into constructs..."):
                    cons, assign = codebook.draft(V, c, progress=lambda m: None)
                    cons, assign = codebook.consolidate(cons, assign, c, progress=lambda m: None)
                    ss.codebook = assign[["label", "construct"]].copy()
            if (V.measure.fillna("") != "").all() or c is None:
                ss.V_measures = V
            else:
                with st.spinner("Naming the measures..."):
                    ss.V_measures, _ = measures.draft(V.assign(measure=""), c, progress=lambda m: None)
        except Exception as e:
            ss.prepare_error = llm.explain(e); ss.V_measures = V; ss.prepared = None
            return
        ss.V, ss.prepare_error = V, None
        ss.prepared = (f"Variables grouped into constructs and measures named. {cost_text([(getattr(c, 'model', None), c.usage)])}"
                       if c is not None else "Your tables are ready. Run the audit in step 5.")

    st.subheader("3. Read and prepare")
    st.caption("The tool reads the articles, groups the variables into constructs and names the measures. You can review the result in step 4.")
    arts = ss.work / "articles"
    with st.expander("Resume an earlier run"):
        st.caption("If a run stopped, upload its progress file and the same PDFs, then press Start. Finished articles are not read again.")
        prog = st.file_uploader("Progress file (.zip)", type="zip", key="progress_upload")
        if prog is not None and ss.get("restored") != prog.file_id:
            ss.restored = prog.file_id
            try:
                n = run.restore_progress(prog.getvalue(), arts)
            except zipfile.BadZipFile:
                n = 0
            (st.success(f"Progress restored for {n} articles.") if n else
             st.error("This is not a progress file from this app. Use the file from the 'Save progress' button."))
    if pdfs and len(pdfs) > MAX_PDFS:
        st.error(f"Please upload at most {MAX_PDFS} articles here. Larger sets run with the command line.")
    from_tables = not pdfs and ss.get("V") is not None
    if pdfs and not has_model:
        st.caption("Add a model and key in step 2 to read the articles.")
    if from_tables and not has_model:
        st.caption("Without a model key, your tables are used as they are. If they don't name constructs or measures, "
                   "add a key in step 2, or add 'construct' and 'measure' columns. Without measure names, the jingle check is skipped.")
    agreed = True
    if pdfs and has_model and (provider == "anthropic" or provider == "roles" and any(v[0] == "anthropic" for v in roles.values())):
        est = CLAUDE_PER_ARTICLE * len(pdfs) + 1
        agreed = st.checkbox(f"I understand this will cost about US${est:.0f} on my Anthropic account.")
    can_start = ((bool(pdfs) and has_model and len(pdfs) <= MAX_PDFS and agreed) or from_tables)
    if st.button("Start", disabled=not can_start, type="primary"):
        for k in ("summary", "report", "codebook", "V_measures", "read_message", "read_error", "prepare_error", "flags"):
            ss.pop(k, None)
        ss.prepared = None
        if pdfs:
            folder = ss.work / "pdfs"; folder.mkdir(exist_ok=True)
            for f in pdfs:
                (folder / f.name).write_bytes(f.getbuffer())
            try:
                if provider == "roles":
                    ss.coders = run.Coders.roles(*(role_client(*roles[r]) for r in ("Coder A", "Coder B", "Reviewer")))
                elif provider == "anthropic":
                    ss.coders = run.Coders(var_a=client("claude-sonnet-5-5"), var_b=client(), mat_a=client(), mat_b=client(), reviewer=client())
                else:
                    ss.coders = run.Coders.one_model(client())
                ss.queue = sorted(folder.glob("*.pdf")); ss.done = {}; ss.total = len(ss.queue)
                ss.workers, ss.started = WORKERS, time.time()
            except Exception as e:
                ss.read_error = llm.explain(e)
        else:
            prepare()

    if ss.get("queue"):
        # one round per page refresh, so the progress file below is always current
        st.progress(len(ss.done) / ss.total, text=f"Read {len(ss.done)} of {ss.total} articles")
        st.caption("Keep this tab open while the articles are read. Save your progress now and then; if the page closes, you can resume.")
        if any(arts.glob("*/*.json")):
            st.download_button("Save progress", run.save_progress(arts), file_name="construct_audit_progress.zip",
                               mime="application/zip", on_click="ignore",
                               help="If the page closes, upload this file under 'Resume an earlier run' to continue where you stopped.")
        workers = ss.get("workers", WORKERS)
        batch, ss.queue = ss.queue[:ROUND], ss.queue[ROUND:]
        st.caption(f"Each article takes a few minutes: two coders list its variables and copy its table, then a reviewer settles "
                   f"disagreements. {workers} article(s) are read at a time.")
        live, limit_note = st.empty(), st.empty()
        waits_before = sum(getattr(u, 'waits', 0) for _, u in ss.coders.usage())

        def show(status):
            rows = [{"article": Path(p).stem, "step": status[p][0] if p in status else "waiting for its turn",
                     "time on this step": ("starting soon" if p not in status else "done" if status[p][0] == "done"
                                           else "stopped" if status[p][0] == "could not be read" else f"{status[p][1]:.0f} s")}
                    for p in batch]
            live.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if sum(getattr(u, 'waits', 0) for _, u in ss.coders.usage()) > waits_before:
                limit_note.warning("Your model account's speed limit is slowing this down. The tool waits and continues by itself. "
                                   "New Anthropic accounts get higher limits after some use (console.anthropic.com → Settings → Limits).")

        show({})
        got = run.extract_many(batch, ss.coders, arts, workers=workers, on_tick=show)
        if sum(getattr(u, 'waits', 0) for _, u in ss.coders.usage()) > waits_before:
            ss.workers = max(1, workers // 2)             # the account is being limited: send fewer articles at once next round
        ss.done.update(got)
        for p in batch:                                   # the PDF and its text are no longer needed
            run.forget_text(p, arts); Path(p).unlink(missing_ok=True)
        failed = [r for r in got.values() if isinstance(r, Exception)]
        if failed and len(failed) == len(got) and len(ss.done) == len(got):
            ss.queue = []                                 # every article of the first round failed: almost always the key or model
        if ss.queue:
            st.rerun()
        V, C, L, F, S = run.collect(ss.done)
        errors = [r for r in ss.done.values() if isinstance(r, Exception)]
        ss.V = schema.load_variables(pd.DataFrame(V)) if V else None
        ss.C = schema.load_correlations(pd.DataFrame(C)) if C else None
        ss.log, ss.flags, ss.summary = pd.DataFrame(L), pd.DataFrame(F), pd.DataFrame(S)
        ss.queue = []
        if errors:
            ss.read_error = (f"{len(errors)} of {len(ss.done)} articles could not be read. " if S else "No article could be read. ") + llm.explain(errors[0])
        if S:
            waited = sum(getattr(u, 'waited', 0.0) for _, u in ss.coders.usage())
            ss.read_message = (f"Read {len(S)} articles: {len(C)} correlations. {cost_text(ss.coders.usage())}"
                               + (f" Your account's speed limit added about {waited / 60:.0f} minutes of waiting." if waited >= 60 else ""))
        if ss.V is not None and ss.C is not None:
            prepare()
    if ss.get("read_error"):
        st.error(ss.read_error)
    if ss.get("read_message"):
        st.success(ss.read_message)
        st.download_button("Save progress", run.save_progress(arts), file_name="construct_audit_progress.zip", mime="application/zip",
                           on_click="ignore", key="save_done", help="The saved model replies, to rerun later without paying again.")
    if ss.get("prepare_error"):
        st.error(f"Grouping and naming did not finish. {ss.prepare_error}")
    if ss.get("prepared"):
        st.success(ss.prepared)
    if "summary" in ss and not ss.summary.empty:
        st.dataframe(ss.summary.rename(columns=SUMMARY_COLUMNS), hide_index=True, width="stretch")
    if isinstance(ss.get("flags"), pd.DataFrame) and not ss.flags.empty:
        with st.expander(f"Rows to check ({len(ss.flags)})"):
            st.caption("The tool found something it could not confirm. Look at the article before trusting the value.")
            st.dataframe(run.describe_flags(ss.flags), hide_index=True, width="stretch")

    st.subheader("4. Review constructs and measures (optional)")
    edited_cb, edited_mc = None, None
    if ss.get("V_measures") is None and ss.get("codebook") is None:
        st.caption("This appears after step 3.")
    else:
        if ss.get("codebook") is not None:
            st.markdown("**Constructs.** Each variable name and the construct it was put in. To move a variable, type a different construct.")
            inv = codebook.inventory(ss.V)[["label", "articles"]]
            table = inv.merge(ss.codebook[["label", "construct"]], on="label", how="left").fillna({"construct": ""})
            edited_cb = st.data_editor(table, hide_index=True, width="stretch", height=320, key=f"cb_edit_{ss.version}",
                                       column_config={"articles": st.column_config.NumberColumn(disabled=True),
                                                      "label": st.column_config.TextColumn("variable name", disabled=True)})
        if ss.get("V_measures") is not None and (ss.V_measures.measure.fillna("") != "").any():
            st.markdown("**Measures.** Articles describe the same measure in different words, so each measure got one short name. "
                        "To correct a name, type a new one under 'rename to'. If two rows are the same measure, give them the same new name.")
            mc = (ss.V_measures.groupby("measure").agg(articles=("article", "nunique"), variables=("var", "size"))
                  .reset_index().sort_values("articles", ascending=False))
            mc["rename to"] = ""
            edited_mc = st.data_editor(mc, hide_index=True, width="stretch", height=300, key=f"mc_edit_{ss.version}",
                                       column_config={"measure": st.column_config.TextColumn(disabled=True),
                                                      "articles": st.column_config.NumberColumn(disabled=True),
                                                      "variables": st.column_config.NumberColumn(disabled=True)})

    st.subheader("5. Report")
    ready = ss.get("V") is not None and ss.get("C") is not None and not ss.get("queue")
    if not ready:
        st.caption("This appears after step 3.")
    elif st.button("Run the audit", type="primary"):
        try:
            V = (ss.V_measures if ss.get("V_measures") is not None else ss.V).copy()
            if edited_cb is not None:
                V = codebook.apply(V, edited_cb, keep_existing=True)
            if edited_mc is not None:
                V = measures.rename(V, dict(zip(edited_mc["measure"], edited_mc["rename to"])))
            V = measures.fallback(V)
            n = V.article.nunique()
            cfg = audit.Config(jangle_level="construct" if (V.construct != "").any() else "label", n_perm=1000,
                               min_anchor_articles=min(20, max(3, n // 4)), min_measure_articles=min(10, max(3, n // 6)))
            with st.spinner("Running the checks..."):
                res = audit.run(V, ss.C, cfg)
            ss.report = report.render(res, "Construct audit")
            ss.V_final, ss.codebook_final = V, edited_cb
            ss.pop("audit_error", None)
        except Exception as e:
            ss.audit_error = f"The audit could not run ({str(e)[:150]}). With very few articles, most checks cannot run; try more articles."
    if ss.get("audit_error"):
        st.error(ss.audit_error)
    if ss.get("report"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("audit_report.html", ss.report)
            z.writestr("variables.csv", ss.V_final.to_csv(index=False)); z.writestr("correlations.csv", ss.C.to_csv(index=False))
            if isinstance(ss.get("log"), pd.DataFrame) and not ss.log.empty:
                z.writestr("decision_log.csv", ss.log.to_csv(index=False))
            if isinstance(ss.get("flags"), pd.DataFrame) and not ss.flags.empty:
                z.writestr("rows_to_check.csv", run.describe_flags(ss.flags).to_csv(index=False))
            if ss.get("codebook_final") is not None:
                z.writestr("codebook.csv", ss.codebook_final.to_csv(index=False))
        st.download_button("Download the report and tables", buf.getvalue(), file_name="construct_audit.zip", mime="application/zip",
                           help="The report opens full-screen in any browser.")
        st.iframe(ss.report, height=1400)

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**Reading.** Two AI coders read each article separately and copy its correlation table row by row.
Where they agree, the value is kept. Where they differ, a third AI reviewer reads the article, quotes it and decides.
If the reviewer is unsure, or its quote is not in the article, the item goes to your list of rows to check.
The coders and the reviewer can be different models. Every quote and every number is checked against the article's own text.

**Grouping.** Variables are grouped into constructs, using your codebook or one the model drafts. Each measure gets one name,
so the same measure is recognised across articles. You can review and correct both before the audit runs.
How each variable was measured (a count from a database, a survey scale...) is kept separate from what it is called.

**Checks.**
- *Jingle*: measures that share a construct name but relate to other variables differently.
- *Jangle*: different construct names that correlate near 1 and relate to everything else the same way.
- *One measure, two constructs*: the same measure filed under different constructs.
- *Method*: self-report measures that correlate more with each other than with measures from other sources.

The report explains its decision rule and every term in a "How to read this report" box.

**Your files.** PDFs and their text are deleted as soon as each article has been read. Your API key is used only in this
session and never stored. The model's replies stay in this session so you can save and resume; working folders of
closed sessions are deleted within a day.

Findings are evidence for human researchers to judge, not verdicts.
""")
