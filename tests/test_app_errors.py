"""What a user sees when things go wrong: plain messages, never a traceback or a false 'success'. Stub models, no cost."""
import io, shutil, tempfile, zipfile
from pathlib import Path
from streamlit.testing.v1 import AppTest
from caudit import llm, schema, synthetic
from caudit.llm import Usage
from tests.make_pdfs import make


class Refused:
    native_pdf, vision, model = False, False, "stub"
    def __init__(self):
        self.usage, self.calls = Usage(), 0
    def json(self, *a, **k):
        self.calls += 1
        raise RuntimeError("Error code: 401 - {'type': 'authentication_error', 'message': 'invalid x-api-key'}")


def pick_open_model(at):
    next(r for r in at.radio if r.label == "Which model reads the articles?").set_value("An open model").run()
    next(t for t in at.text_input if t.label == "Model name").set_value("some/model").run()


def test_bad_key_with_pdfs_says_so(monkeypatch):
    stub = Refused(); monkeypatch.setattr(llm, "make_client", lambda *a, **k: stub)
    d = Path(tempfile.mkdtemp()); make(d / "src")
    files = []
    for i in range(23):
        shutil.copy(d / "src" / "A001.pdf", d / f"P{i:02d}.pdf"); files.append(d / f"P{i:02d}.pdf")
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120).run()
    pick_open_model(at)
    at.session_state["coders"] = __import__("caudit.extract.run", fromlist=["run"]).Coders.one_model(stub)
    at.session_state["queue"] = files; at.session_state["done"] = {}; at.session_state["total"] = len(files)
    at.run()
    assert not at.exception, at.exception
    errs = " ".join(e.value for e in at.error)
    assert "No article could be read" in errs and "API key was refused" in errs
    assert not any("Read 0 articles" in s.value for s in at.success)          # no false success
    assert len(at.session_state["done"]) == 10                                  # stopped after the first round, not 23 failures
    assert not any(p.exists() for p in files[:10])                              # read PDFs are deleted
    shutil.rmtree(d)


def test_tables_with_bad_key_no_traceback(monkeypatch):
    stub = Refused(); monkeypatch.setattr(llm, "make_client", lambda *a, **k: stub)
    V, C = synthetic.make(n_articles=20); V["construct"] = ""; V["label"] = ""; V["measure"] = ""
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120).run()
    at.session_state["V"] = schema.load_variables(V); at.session_state["C"] = schema.load_correlations(C)
    pick_open_model(at)
    next(b for b in at.button if b.label == "Start").click().run()
    assert not at.exception, at.exception
    assert any("did not finish" in e.value and "refused" in e.value for e in at.error)


def test_bogus_progress_file():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60).run()
    # AppTest cannot upload files; call the same reader the app uses
    from caudit.extract import run
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "hello")
    assert run.restore_progress(buf.getvalue(), Path(tempfile.mkdtemp())) == 0
    try:
        run.restore_progress(b"not a zip", Path(tempfile.mkdtemp())); assert False
    except zipfile.BadZipFile:
        pass
    assert not at.exception
