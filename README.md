# construct-audit

Give it the articles of a literature. It shows where measures that share a construct's name behave differently
(*jingle*), where different names may be one thing (*jangle*), and where one kind of measure is filed under
several constructs.

**Try it:** https://construct-audit.streamlit.app/ · **Website:** https://p-ogg.github.io/construct-audit/

Articles in, audit report out:

1. **Read** – two AI coders list each article's variables and copy its correlation table row by row; a third AI
   reviewer reads the article and settles only what they disagree on (correlations, survey vs archival, reliability,
   sample size), with a quote and a confidence level. Fixed rules decide whether its answer stands: if it is unsure, its
   quote is not in the article, or its value is impossible, the item goes to a list of rows for a human to check.
   Every quote and number is checked against the article's own text.
2. **Group** – variables are grouped into constructs with your codebook, or with one the model drafts for you to edit.
   Separately, each variable's measure is named the same way across articles (e.g. "OCQ (Mowday et al. 1979)",
   "supervisor-rated performance", "return on assets"): the model drafts the names from the coders' notes on the form of
   the measure, its source and the scale it cites, and you edit them. Tables coded without a model get basic labels
   (form + rater + cited scale), which cannot tell apart archival measures of the same form.
3. **Check** – each measure's correlations with common constructs form a profile; profiles are compared with each other
   and with random groupings of the same measures. Duplicates must correlate near 1 *and* relate to everything else alike.
4. **Report** – a self-contained HTML page; findings are evidence for researchers to judge, not verdicts.

Works with Claude or with open-weights models through any OpenAI-compatible server (OpenRouter, Groq, LM Studio, Ollama, vLLM).

**Scope.** Any literature whose studies report correlation matrices: most survey and archival research in
organisational science. Experiments that report only means, and qualitative studies, are out of scope.
The checks need enough studies per construct; the app scales its thresholds to the size of the literature.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # or use an open model, see below
python -m caudit run my_articles/ -o results/
```

Open model instead of Claude:

```bash
export OPENROUTER_API_KEY=...
python -m caudit run my_articles/ -o results/ --provider openrouter --model qwen/qwen3-235b-a22b --vision
```

A different model in each role, so the coders' errors are less alike:

```bash
python -m caudit run my_articles/ -o results/ --coder-a openrouter:qwen/qwen3-235b-a22b \
    --coder-b openrouter:meta-llama/llama-4-maverick --reviewer anthropic:claude-opus-5-5
```

Already have coded tables? Fill in [the template](templates/construct_audit_template.xlsx) (one sheet for variables, one for
correlations, an optional codebook sheet, instructions included) and upload it in the app. On the command line, audit two CSV files (`variables.csv`,
`correlations.csv`, see `caudit/schema.py`) directly. Without a model, measures are compared only if the tables name them:

```bash
python -m caudit audit variables.csv correlations.csv -o report.html
```

Web app: `streamlit run app/streamlit_app.py`.

## Tests (no API calls)

```bash
pip install -r requirements-dev.txt
pytest -q
```

They use a synthetic literature with planted problems, synthetic PDFs (one with a sideways table and comma decimals),
and a stub model.

## Example

`examples/simulated/` builds a simulated literature (160 invented studies of employee attitudes and performance) with
planted problems, and audits it: `python examples/simulated/make_example.py`. The report states what was planted, so you
can check that the audit finds it. The website shows the same report.

## Licence

MIT.
