"""Synthetic article PDFs for tests (no copyrighted articles needed).
A001: upright correlation table with comma decimals and negatives. A002: the same table printed sideways."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

METHODS = [
    "Data and measures. We surveyed manufacturing firms; the final sample comprises N = 312 firms.",
    "Firm size was measured as the logarithm of the number of employees reported in annual accounts.",
    "Innovation performance was measured as the number of patent applications filed at the EPO.",
    "Openness was measured with a five-item scale adapted from prior work (Cronbach's alpha = 0,84).",
    "Collaboration breadth counts the types of external partners the firm cooperated with.",
]
NAMES = ["1 Firm size", "2 Innovation performance", "3 Openness", "4 Collaboration breadth"]
ROWS = [[], ["0,31"], ["0,05", "0,22"], ["-0,12", "0,18", "0,41"]]


def _table(c, x0, y0):
    c.setFont("Helvetica-Bold", 10); c.drawString(x0, y0, "Table 1. Correlations among the study variables (N = 312)")
    c.setFont("Helvetica", 10)
    for i, (name, vals) in enumerate(zip(NAMES, ROWS)):
        y = y0 - 22 * (i + 1)
        c.drawString(x0, y, name)
        for j, v in enumerate(vals):
            c.drawString(x0 + 200 + 60 * j, y, v)
    c.drawString(x0, y0 - 22 * 6, "Note: correlations above 0,11 in absolute value are significant at p < 0.05.")


def make(outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    for name, rotated in (("A001", False), ("A002", True)):
        c = canvas.Canvas(str(out / f"{name}.pdf"), pagesize=A4)
        c.setFont("Helvetica", 11)
        for i, line in enumerate(METHODS):
            c.drawString(50, 780 - 18 * i, line)
        c.showPage()
        if rotated:
            c.saveState(); c.translate(595, 0); c.rotate(90)
            _table(c, 60, 520)
            for k in range(12):   # enough sideways words to count as a rotated page
                c.drawString(60, 260 - 12 * k, "variables correlations table descriptive statistics sample firms")
            c.restoreState()
        else:
            _table(c, 50, 760)
        c.showPage(); c.save()
    return out


if __name__ == "__main__":
    print(make("tests/pdfs"))
