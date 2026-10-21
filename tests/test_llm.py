"""Request shape and cost arithmetic for prompt caching (no API calls)."""
from caudit import llm


def test_cache_breakpoint_only_when_article_first():
    c = llm.AnthropicClient.__new__(llm.AnthropicClient)          # no network, no key needed for _content
    doc = {"pdf_b64": "JVBERi0="}
    first = c._content("instructions", doc, [("aW1n", "image/jpeg")], doc_first=True)
    assert first[0]["type"] == "document" and "cache_control" not in first[0]
    assert first[1]["cache_control"] == {"type": "ephemeral"} and first[-1]["text"] == "instructions"
    later = c._content("instructions", doc, [], doc_first=False)
    assert later[0]["text"] == "instructions" and not any("cache_control" in p for p in later)


def test_cost_counts_cache_at_its_price():
    u = llm.Usage(price=(4.0, 20.0))
    u.add("a", 1_000, 1_000, "end_turn", cache_write=100_000)       # first call writes the article to the cache
    u.add("b", 1_000, 1_000, "end_turn", cache_read=100_000)        # later calls read it
    expect = (2_000 * 4 + 100_000 * 4 * 1.25 + 100_000 * 4 * 0.1 + 2_000 * 20) / 1e6
    assert abs(u.cost() - expect) < 1e-9
