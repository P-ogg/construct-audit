"""Drive the web app headlessly: coded tables in, report out. No model calls."""
from streamlit.testing.v1 import AppTest
from caudit import synthetic, schema


def check_app():
    V, C = synthetic.make(n_articles=40)
    V["construct"] = ""; V["label"] = ""; V["measure"] = ""      # as if freshly extracted: no codebook yet
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    at.session_state["V"] = schema.load_variables(V); at.session_state["C"] = schema.load_correlations(C)
    at.run()
    run_btn = next(b for b in at.button if b.label == "Run the audit")
    run_btn.click().run()
    assert not at.exception, at.exception
    rep = at.session_state["report"]
    assert "Jingle" in rep and "Jangle" in rep
    return rep


def test_app():
    check_app()


if __name__ == "__main__":
    r = check_app(); print("PASSED", len(r), "chars of report")
