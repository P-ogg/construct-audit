"""The audit must find the problems planted in the synthetic literature, and only those."""
from caudit import audit, schema, synthetic


def run():
    V, C = synthetic.make()
    V = schema.load_variables(V); C = schema.load_correlations(C)
    cfg = audit.Config(n_perm=500, min_measure_articles=8, min_anchor_articles=10)
    return audit.run(V, C, cfg)


def check_all():
    res = run()
    j = {x["construct"]: x for x in res["jingle"]}
    ip = j["innovation performance"]
    comp = ip["comparisons"][0]
    assert comp["p_mad"] < 0.05, comp                       # planted jingle is found
    assert any(c["measure"] == "patent counts" and c["anchor"] == "firm size" for c in ip["contamination"])
    size = j.get("firm size")
    if size and size["comparisons"]:
        assert size["comparisons"][0]["p_mad"] > 0.05      # two size measures behave alike: no jingle
    verdict = {(p["unit_1"], p["unit_2"]): p["verdict"] for p in res["jangle"]["pairs"]}
    assert verdict[("trait X", "trait Y")] == "possible duplicate", verdict
    assert verdict.get(("sibling S", "sibling T")) in ("siblings", "related"), verdict
    assert not any(v == "possible duplicate" for k, v in verdict.items() if k != ("trait X", "trait Y"))
    return res


def test_all():
    check_all()


if __name__ == "__main__":
    import json
    r = check_all()
    print("PASSED")
    print(json.dumps({k: r[k] for k in ("summary", "method")}, indent=1, default=str))
    for p in r["jangle"]["pairs"][:6]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()})
