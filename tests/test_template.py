"""The upload template: it round-trips through the reader, and mistakes get plain messages."""
import io
import pandas as pd
import pytest
from caudit import template, audit, report


class Upload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data); self.name = name


def test_template_round_trip():
    V, C, notes = template.read([Upload(template.build(), "construct_audit_template.xlsx")])
    assert V.article.nunique() == 2 and len(V) == 6 and len(C) == 6
    assert set(V.data_type) == {"survey", "other-report", "archival"} and V.study.iloc[0] == "1"
    assert not any("refer to a variable" in n for n in notes)              # IDs match across sheets (no '1.0' vs '1')
    sheets = pd.read_excel(io.BytesIO(template.build()), sheet_name=None)
    assert list(sheets) == ["Instructions", "Variables", "Correlations", "Codebook"]


def csv(df, name):
    return Upload(df.to_csv(index=False).encode(), name)


def test_two_csvs_any_order_and_plain_errors():
    v = pd.DataFrame(dict(article=["A"] * 3, study=["1"] * 3, var=["1", "2", "3"], name=["x", "y", "z"]))
    c = pd.DataFrame(dict(article=["A"] * 3, study=["1"] * 3, var_a=["1", "1", "2"], var_b=["2", "3", "4"], r=["0,31", "abc", ".2"]))
    V, C, notes = template.read([csv(c, "cors.csv"), csv(v, "vars.csv")])
    assert C.r.round(2).tolist() == [0.31, 0.2]                             # decimal comma read, text dropped
    assert any("1 correlations were left out" in n for n in notes)
    assert any("refer to a variable that is not" in n for n in notes)      # var 4 does not exist
    assert any("do not name the measures" in n for n in notes)
    with pytest.raises(template.UploadError, match="no column 'name'"):
        template.read([csv(v.drop(columns="name"), "v.csv"), csv(c, "c.csv")])
    with pytest.raises(template.UploadError, match="two CSV files"):
        template.read([csv(v, "v.csv")])


def test_report_says_when_jingle_is_skipped():
    from caudit import synthetic, schema
    V, C = synthetic.make(n_articles=30); V["measure"] = ""
    res = audit.run(schema.load_variables(V), schema.load_correlations(C), audit.Config(n_perm=50))
    assert res["jingle"] == [] and "do not name the measures" in report.render(res)


def test_codebook_sheet():
    from caudit import codebook, labels
    # the untouched template: example rows are flagged, and its example codebook is ignored so the model drafts one
    up = Upload(template.build(), "construct_audit_template.xlsx")
    V, C, notes = template.read([up])
    assert any("example rows" in n for n in notes) and template.read_codebook([up]) is None
    # a filled-in codebook sheet is used
    sheets = pd.read_excel(io.BytesIO(template.build()), sheet_name=None, dtype=str)
    sheets["Codebook"] = pd.concat([sheets["Codebook"], pd.DataFrame({"variable name": ["Board size"], "construct": ["Board structure"]})])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for k, d in sheets.items():
            d.to_excel(w, sheet_name=k, index=False)
    up = Upload(buf.getvalue(), "mine.xlsx")
    V, C, notes = template.read([up])
    cb = template.read_codebook([up])
    assert set(cb.columns) == {"label", "construct"} and "affective commitment" in set(cb.label)
    V = codebook.apply(labels.apply(V), cb, keep_existing=True)
    got = dict(zip(V.name, V.construct))
    assert got["Affective commitment"] == "Organisational commitment" and got["ROA"] == got["Tobin's q"] == "Firm performance"
    # a construct already in the Variables table wins over the codebook
    V2 = V.assign(construct=lambda d: d.construct.where(d.name != "ROA", "Profitability"))
    assert dict(zip(V2.name, codebook.apply(V2, cb, keep_existing=True).construct))["ROA"] == "Profitability"
    # a CSV codebook with the authors' spelling is matched after cleaning
    csv_cb = codebook.load(pd.DataFrame({"Variable name": ["Task Performance (T1)"], "Construct": ["Job performance"]}))
    assert csv_cb.label.tolist() == ["task performance"]
    with pytest.raises(ValueError, match="variable name"):
        codebook.load(pd.DataFrame({"x": ["a"], "construct": ["b"]}))


def test_messy_files():
    v = pd.DataFrame(dict(article=["A"] * 3, study=["1"] * 3, var=["1", "2", "3"], name=["Zufriedenheit é", "y", "z"], reliability=["85", "0,2", ""]))
    c = pd.DataFrame(dict(article=["A"] * 4, study=["1"] * 4, var_a=["1", "1", "2", "2"], var_b=["2", "3", "3", "1"],
                          r=["\u20130,31", ".31**", "0.2", "-0.31"]))     # en dash: what Windows Excel writes
    semi = lambda df, n: Upload(df.to_csv(index=False, sep=";").encode("cp1252"), n)
    V, C, notes = template.read([semi(v, "v.csv"), semi(c, "c.csv")])       # French Excel: semicolons, Windows encoding
    assert C.r.round(2).tolist() == [-0.31, 0.31, 0.2]                       # unicode minus + comma, stars; 2-1 duplicate dropped
    assert any("appear twice" in n for n in notes) and any("percentages" in n for n in notes) and any("0.30 or less" in n for n in notes)
    assert V.reliability.iloc[0] == 0.85
    V2, C2, _ = template.read([csv(v, "v.csv"), csv(c.assign(r=["\u22120,31", "0.1", "0.2", "0.3"]), "c.csv")])
    assert C2.r.iloc[0] == -0.31                                               # unicode minus, UTF-8
    with pytest.raises(template.UploadError, match="empty"):
        template.read([Upload(b"", "v.csv"), csv(c, "c.csv")])
    with pytest.raises(template.UploadError, match="could not be opened"):
        template.read([Upload(b"not an excel file", "x.xlsx")])
    with pytest.raises(template.UploadError, match="no rows"):
        template.read([csv(v.iloc[:0], "v.csv"), csv(c, "c.csv")])
