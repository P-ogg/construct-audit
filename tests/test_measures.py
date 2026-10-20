"""Naming measures in any literature: the basic labels, the model draft (stub model, no cost) and the app's rename step."""
import re
import pandas as pd
from streamlit.testing.v1 import AppTest
from caudit import measures, schema, synthetic
from caudit.llm import Usage

ROWS = [  # an organisational-behaviour literature: nothing here is about firms or innovation
    ("A1", "Organizational commitment", "Eight items of affective commitment from Allen & Meyer, 7-point scale", "survey", "multi-item scale", "Allen & Meyer 1990"),
    ("A2", "Affective commitment", "Allen and Meyer's affective commitment scale, 6 items", "survey", "multi-item scale", "Allen and Meyer (1990)"),
    ("A3", "Commitment", "OCQ, 15 items", "survey", "multi-item scale", "Mowday et al. 1979"),
    ("A4", "Job performance", "Supervisor rated each employee's task performance on four items", "other-report", "multi-item scale", ""),
    ("A5", "Job performance", "Employees rated their own performance on four items", "survey", "multi-item scale", ""),
    ("A6", "Absenteeism", "Number of days absent in the following year, from HR records", "archival", "count", ""),
]


def variables():
    return schema.load_variables(pd.DataFrame([dict(article=a, study="1", var="1", name=n, description=d, data_type=t, reliability="",
                                                    construct="", label="", measure="", measure_form=f, instrument=i)
                                               for a, n, d, t, f, i in ROWS]))


def test_basic_labels_are_field_independent():
    m = dict(zip(variables().article, measures.fallback(variables()).measure))
    assert m["A1"] == "multi-item scale: Allen & Meyer 1990" and m["A3"] == "multi-item scale: Mowday et al. 1979"
    assert m["A4"] == "multi-item scale (rated by others)" and m["A5"] == "multi-item scale"     # self vs supervisor rating differ
    assert m["A6"] == "count"


class Stub:
    native_pdf, vision, model = False, False, "stub"

    def __init__(self):
        self.usage, self.prompts = Usage(), []

    def json(self, system, user, schema, label="", **kw):
        self.prompts.append(user)
        out = []
        for i, line in re.findall(r"^(\d+)\. (.*)$", user, flags=re.M):
            name = ("affective commitment scale (Allen & Meyer 1990)" if "Allen" in line else "OCQ (Mowday et al. 1979)" if "OCQ" in line
                    else "supervisor-rated task performance" if "Supervisor" in line else "self-rated performance" if "own performance" in line
                    else "days absent (records)")
            out.append({"id": int(i), "measure": name})
        return {"measures": out}


def test_draft_names_measures_consistently():
    c = Stub()
    v, counts = measures.draft(variables(), c, batch=4, progress=lambda m: None)
    m = dict(zip(v.article, v.measure))
    assert m["A1"] == m["A2"] == "affective commitment scale (Allen & Meyer 1990)"      # same scale, two wordings: one name
    assert m["A3"] != m["A1"] and m["A4"] != m["A5"]
    assert "affective commitment scale (Allen & Meyer 1990)" in c.prompts[1]             # the list of names carries over batches
    assert counts.set_index("measure").loc["affective commitment scale (Allen & Meyer 1990)", "articles"] == 2
    v2 = measures.rename(v, {"supervisor-rated task performance": "rated performance", "self-rated performance": "rated performance"})
    assert (v2.measure == "rated performance").sum() == 2


FORM = {"patent counts": "count", "sales share": "ratio or share", "employees": "count", "sales": "amount", "years": "time or age",
        "survey scale": "multi-item scale"}


def test_app_tables_without_model():
    V, C = synthetic.make(n_articles=40)
    V["measure_form"] = V.measure.map(FORM).fillna("other"); V["instrument"] = ""
    V["construct"] = ""; V["label"] = ""; V["measure"] = ""
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120).run()
    at.session_state["V"] = schema.load_variables(V); at.session_state["C"] = schema.load_correlations(C)
    at.run()
    assert any("used as they are" in c.value for c in at.caption)               # no key: the app says what happens
    next(b for b in at.button if b.label == "Start").click().run()
    next(b for b in at.button if b.label == "Run the audit").click().run()
    assert not at.exception, at.exception
    assert "ratio or share" in set(at.session_state["V_final"].measure)          # basic labels fill the gap


class AnyStub:
    """Answers the codebook and measure-naming calls, whatever the schema."""
    native_pdf, vision, model = False, False, "stub"

    def __init__(self):
        self.usage, self.labels = Usage(), []

    def json(self, system, user, schema, label="", **kw):
        self.labels.append(label)
        ids = [int(i) for i in re.findall(r"^(\d+)\. ", user, flags=re.M)]
        props = schema["properties"]
        if "constructs" in props:
            return {"constructs": [{"name": "Size", "definition": "how big", "example_labels": []}]}
        if "assignments" in props:
            return {"assignments": [{"id": i, "construct": "Size", "confidence": "high"} for i in ids]}
        if "mapping" in props:
            return {"mapping": [{"id": i, "construct": "Size"} for i in ids]}
        if "measures" in props:
            return {"measures": [{"id": i, "measure": f"named measure {i % 3}"} for i in ids]}
        raise AssertionError(label)


def test_app_prepares_automatically(monkeypatch):
    from caudit import llm
    stub = AnyStub()
    monkeypatch.setattr(llm, "make_client", lambda *a, **k: stub)
    V, C = synthetic.make(n_articles=30)
    V["construct"] = ""; V["label"] = ""; V["measure"] = ""
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120).run()
    at.session_state["V"] = schema.load_variables(V); at.session_state["C"] = schema.load_correlations(C)
    next(r for r in at.radio if r.label == "Which model reads the articles?").set_value("An open model").run()
    next(t for t in at.text_input if t.label == "Model name").set_value("some/model").run()
    next(b for b in at.button if b.label == "Start").click().run()
    assert not at.exception, at.exception
    assert any(l.startswith("propose") for l in stub.labels) and any(l.startswith("measures") for l in stub.labels)  # no extra clicks
    assert any("measures named" in s.value for s in at.success)
    assert set(at.session_state["V_measures"].measure) <= {"named measure 0", "named measure 1", "named measure 2"}
