"""Reading many articles: in parallel, in rounds in the app, and resumable from a progress file. Stub model, no cost."""
import shutil, tempfile
from pathlib import Path
from streamlit.testing.v1 import AppTest
from caudit.extract import run
from tests.make_pdfs import make
from tests.test_extract import Stub


def pdfs(n):
    d = Path(tempfile.mkdtemp()); make(d / "src")
    for i in range(n):
        shutil.copy(d / "src" / "A001.pdf", d / f"P{i:02d}.pdf")
    return d, sorted(d.glob("P*.pdf"))


def calls(*stubs):
    return sum(len(s.seen) for s in stubs)


def test_parallel_and_resume():
    d, files = pdfs(7)
    A, B = Stub("A"), Stub("B"); coders = run.Coders(var_a=A, var_b=B, mat_a=A, mat_b=B, reviewer=B)
    res = run.extract_many(files, coders, d / "work", workers=4)
    assert len(res) == 7 and not any(isinstance(r, Exception) for r in res.values())
    V, C, L, F, S = run.collect(res)
    assert len(S) == 7 and len({c["article"] for c in C}) == 7
    zipped = run.save_progress(d / "work")
    assert b".txt" not in zipped                                       # no article text in the progress file
    # a new session: restore the progress file, read again, no model calls
    A2, B2 = Stub("A"), Stub("B"); coders2 = run.Coders(var_a=A2, var_b=B2, mat_a=A2, mat_b=B2, reviewer=B2)
    assert run.restore_progress(zipped, d / "work2") == 7
    run.extract_many(files, coders2, d / "work2", workers=4)
    assert calls(A2, B2) == 0
    shutil.rmtree(d)


def test_restore_ignores_unsafe_paths():
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil/x.json", "{}"); z.writestr("A1/../../y.json", "{}"); z.writestr("A1/ok_1.json", "{}")
    d = Path(tempfile.mkdtemp())
    assert run.restore_progress(buf.getvalue(), d / "w") == 1 and (d / "w" / "A1" / "ok_1.json").exists()
    assert not (d / "evil").exists()


def test_app_reads_in_rounds():
    d, files = pdfs(23)
    A, B = Stub("A"), Stub("B")
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=300)
    at.run()
    at.session_state["coders"] = run.Coders(var_a=A, var_b=B, mat_a=A, mat_b=B, reviewer=B)
    at.session_state["queue"] = files; at.session_state["done"] = {}; at.session_state["total"] = len(files)
    at.run()
    assert not at.exception, at.exception
    assert len(at.session_state["done"]) == 23 and not at.session_state["queue"]
    assert at.session_state["summary"].shape[0] == 23
    assert any("Read 23 articles" in s.value for s in at.success)
    shutil.rmtree(d)


class Slow(Stub):
    """A stub that takes 0.3 s per call and can pretend the provider's rate limit made it wait."""
    def __init__(self, coder, limited=False):
        super().__init__(coder); self.limited = limited
    def json(self, *a, **k):
        import time
        time.sleep(0.3)
        if self.limited:
            self.usage.note_wait(1.0)
        return super().json(*a, **k)


def test_coders_run_together_and_report_steps():
    import time
    d, files = pdfs(1)
    A, B = Slow("A"), Slow("B"); coders = run.Coders(var_a=A, var_b=B, mat_a=A, mat_b=B, reviewer=B)
    steps = []
    t = time.time()
    run.extract_article(files[0], coders, d / "work", progress=steps.append)
    took = time.time() - t
    assert took < 1.5, took                              # 4 coder calls + 1 review at 0.3 s: ~0.9 s together, 1.5 s one after another
    assert steps[:3] == ["extracting the text", "listing the variables (both coders)", "copying the correlation table (both coders)"]
    seen = []
    run.extract_many(files, run.Coders(var_a=Slow("A"), var_b=Slow("B"), mat_a=Slow("A"), mat_b=Slow("B"), reviewer=Slow("B")),
                     d / "work2", workers=1, on_tick=lambda s: seen.append(dict(s)), tick=0.1)
    assert any(v[0].startswith("listing") for s in seen for v in s.values()) and list(seen[-1].values())[0][0] == "done"
    shutil.rmtree(d)


def test_app_warns_about_rate_limits_and_slows_down():
    d, files = pdfs(12)
    A, B = Slow("A", limited=True), Slow("B")
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=300).run()
    at.session_state["coders"] = run.Coders(var_a=A, var_b=B, mat_a=A, mat_b=B, reviewer=B)
    at.session_state["queue"] = files; at.session_state["done"] = {}; at.session_state["total"] = len(files)
    at.session_state["workers"] = 5
    at.run()
    assert not at.exception, at.exception
    assert len(at.session_state["done"]) == 12
    assert at.session_state["workers"] < 5                 # halved after a limited round
    shutil.rmtree(d)
