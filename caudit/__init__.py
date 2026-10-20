"""construct-audit: audit the constructs and measures of a research literature.

Articles in, audit report out:
  extract  -> one row per variable, one row per correlation (LLM, double-coded)
  codebook -> group variables into constructs (researcher's codebook or an LLM draft the researcher edits)
  measures -> classify what each variable actually measured (kind of measure)
  audit    -> jingle, jangle, contamination and method checks
  report   -> HTML report
"""
__version__ = "0.1.0"
