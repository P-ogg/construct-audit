"""Whole extraction run on synthetic PDFs with a stub model: no API calls, no cost."""
import copy, re, shutil, tempfile
from pathlib import Path
from caudit.extract import run, pdf as pdfx
from caudit.llm import Usage
from tests.make_pdfs import make

VARS = {"page_basis": "pdf_sheet_number", "studies": [{"study": 1, "matrix_location": "Table 1", "n_quote": "the final sample comprises N = 312 firms", "n": "312",
        "variables": [
            {"variable": 1, "name": "Firm size", "description_quote": "Firm size was measured as the logarithm of the number of employees",
             "description": "log of number of employees", "source_quote": "annual accounts", "data_type": "archival", "reliability_quote": "NOT_FOUND", "reliability": "", "page": "1"},
            {"variable": 2, "name": "Innovation performance", "description_quote": "Innovation performance was measured as the number of patent applications filed at the EPO",
             "description": "number of EPO patent applications", "source_quote": "EPO", "data_type": "archival", "reliability_quote": "NOT_FOUND", "reliability": "", "page": "1"},
            {"variable": 3, "name": "Openness", "description_quote": "Openness was measured with a five-item scale adapted from prior work",
             "description": "five-item scale", "source_quote": "We surveyed", "data_type": "survey", "reliability_quote": "Cronbach's alpha = 0,84", "reliability": "0,84", "page": "1"},
            {"variable": 4, "name": "Collaboration breadth", "description_quote": "This sentence was invented by the model and is not in the article at all",
             "description": "count of partner types", "source_quote": "NOT_FOUND", "data_type": "survey", "reliability_quote": "NOT_FOUND", "reliability": "", "page": "1"}]}]}


def matrix(wrong=False, invented=False):
    rows = [{"variable": 2, "label": "Innovation performance", "first_column": 1, "values": ["0,31"]},
            {"variable": 3, "label": "Openness", "first_column": 1, "values": ["0,05", "0,22"]},
            {"variable": 4, "label": "Collaboration breadth", "first_column": 1, "values": ["0,12" if wrong else "-0,12", "0,18", "0,63" if invented else "0,41"]}]
    return {"page_basis": "pdf_sheet_number", "studies": [{"study": 1, "correlation_matrix_present": True, "matrix_location": "Table 1",
            "matrix_quote": "Correlations among the study variables", "matrix_page": "2", "unlisted_variables": [], "missing_pairs": [],
            "matrix_rows": rows, "confidence": "high"}]}


def vars_a():
    """Coder A disagrees on three variable fields: data type of 3, reliability of 3, data type of 1 (where A is right)."""
    v = copy.deepcopy(VARS); vs = v["studies"][0]["variables"]
    vs[2]["data_type"] = "archival"; vs[2]["reliability"] = "0,48"; vs[0]["data_type"] = "unknown"
    return v


REVIEW = {   # what the stub reviewer answers per ref
    "r:1:1:4": dict(quote="Table 1. Correlations among the study variables", page="2", reason="Row 4 prints -0,12.", value="-0,12",
                    chose="B", confidence="high", escalate=False),
    "data_type:1:3": dict(quote="Openness was measured with a five-item scale adapted from prior work", page="1",
                          reason="A rating scale answered by respondents.", value="survey", chose="B", confidence="high", escalate=False),
    "reliability:1:3": dict(quote="", page="", reason="Cannot tell which alpha applies.", value="0,84", chose="B", confidence="low", escalate=False),
    "data_type:1:1": dict(quote="an invented sentence that the article never printed", page="1", reason="Annual accounts.",
                          value="archival", chose="B", confidence="high", escalate=False),
}


class Stub:
    native_pdf, vision, model = False, False, "stub"

    def __init__(self, coder):
        self.coder, self.usage, self.seen = coder, Usage(), []

    def json(self, system, user, schema, max_tokens=0, label="", doc=None, images=None, doc_first=False):
        self.seen.append((label, doc_first, bool(doc and doc.get("text"))))
        if "variables" in label:
            return vars_a() if self.coder == "A" else VARS
        if "matrix" in label:      # coder A misreads one sign; both coders "read" an invented value for 3x4
            return matrix(wrong=(self.coder == "A"), invented=True)
        if "review" in label:
            self.review_prompt = user
            return {"decisions": [dict(ref=r, **REVIEW[r]) for r in re.findall(r'ref: "([^"]+)"', user)]}
        raise AssertionError(label)


def check_extract():
    d = Path(tempfile.mkdtemp()); make(d / "pdfs")
    A, B = Stub("A"), Stub("B")
    coders = run.Coders(var_a=A, var_b=B, mat_a=A, mat_b=B, reviewer=B)
    r = run.extract_article(d / "pdfs" / "A001.pdf", coders, d / "work")
    cor = {(c["var_a"], c["var_b"]): c for c in r["correlations"]}
    assert cor[("1", "4")]["r"] == "-0.12" and cor[("1", "4")]["decided_by"] == "reviewer"      # disagreement settled
    assert cor[("1", "2")]["r"] == "0.31" and cor[("1", "2")]["decided_by"] == "both coders"   # comma decimal read
    assert cor[("1", "2")]["n"] == "312"
    checks = {(f["check"], f["detail"][:4]) for f in r["flags"]}
    assert ("number_absent", "3x4 ") in checks                       # invented 0,63 is not on the page
    assert not any(f["check"].startswith("number") and not f["detail"].startswith("3x4") for f in r["flags"])
    assert any(f["check"] == "quote_not_found" and "variable 4" in f["detail"] for f in r["flags"])
    assert not any(f["check"] == "quote_not_found" and "variable 1" in f["detail"] for f in r["flags"])
    v3 = next(v for v in r["variables"] if v["var"] == "3"); v1 = next(v for v in r["variables"] if v["var"] == "1")
    assert v3["reliability"] == "0.84"
    # the reviewer saw only the four disagreements, never an agreed value
    assert sorted(re.findall(r'ref: "([^"]+)"', B.review_prompt)) == sorted(REVIEW)
    log = {(l["field"], l["var"]): l for l in r["log"]}
    assert v3["data_type"] == "survey" and log[("data_type", "3")]["decided_by"] == "reviewer"              # settled, quote checked
    assert log[("reliability", "3")]["decided_by"] == "reviewer (check)"                                     # low confidence: human
    assert v1["data_type"] == "archival" and log[("data_type", "1")]["decided_by"] == "reviewer (check)"     # invented quote: human
    human = [f["detail"] for f in r["flags"] if f["check"] == "needs_human"]
    assert len(human) == 2 and any("not confident" in h for h in human) and any("quote is not in the article" in h for h in human)
    assert r["summary"]["reviewed"] == 4 and r["summary"]["to_check"] == 2
    assert A.seen[0][1] is True and B.seen[0][1] is False             # framing differs between coders
    # resumable: a second run reuses saved replies, no new calls
    n = len(A.seen) + len(B.seen); run.extract_article(d / "pdfs" / "A001.pdf", coders, d / "work")
    assert len(A.seen) + len(B.seen) == n
    assert pdfx.sideways_pages(d / "pdfs" / "A002.pdf") and not pdfx.sideways_pages(d / "pdfs" / "A001.pdf")
    shutil.rmtree(d)
    return r


def test_extract():
    check_extract()


if __name__ == "__main__":
    r = check_extract(); print("PASSED", r["summary"]); [print(" ", f) for f in r["flags"]]


class Nulls(Stub):
    """A model that answers null for empty parts, e.g. an article without a correlation table (a review, an appendix)."""
    def json(self, system, user, schema, max_tokens=0, label="", doc=None, images=None, doc_first=False):
        if "matrix" in label:
            return {"page_basis": None, "studies": [{"study": 1, "correlation_matrix_present": False, "matrix_location": None,
                    "matrix_quote": None, "matrix_page": None, "unlisted_variables": None, "missing_pairs": None,
                    "matrix_rows": None, "confidence": "low"}]}
        if "variables" in label and self.coder == "A":
            return {"page_basis": None, "studies": None}
        return super().json(system, user, schema, max_tokens, label, doc, images, doc_first)


def test_null_parts_do_not_crash():
    d = Path(tempfile.mkdtemp()); make(d / "pdfs")
    A, B = Nulls("A"), Nulls("B")
    r = run.extract_article(d / "pdfs" / "A001.pdf", run.Coders(var_a=A, var_b=B, mat_a=A, mat_b=B, reviewer=B), d / "work")
    assert r["correlations"] == [] and r["summary"]["variables"] == 4       # no table: no correlations, but no crash
    shutil.rmtree(d)
