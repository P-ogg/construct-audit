"""PDF text, page splitting, sideways-table detection and upright page images. Ported from pipeline3/core.py.

Text is used (a) to verify quotes and numbers and (b) as the document input for models that cannot read PDFs
(most open-weights models). Wide correlation matrices are often printed sideways; on one article the model read
every value of a rotated 120-cell matrix and still attached them to the wrong pairs. Re-rendering those pages
upright removed the problem.
"""
import threading
import base64, hashlib, io, re
from pathlib import Path

PAGE_MARK = "\n--- PDF PAGE {n} ---\n"
PAGE_MARKER = re.compile(r"---\s*PDF PAGE \d+\s*---")
SIDEWAYS_MIN_WORDS, SIDEWAYS_SHARE, SIDEWAYS_MAX_PAGES = 40, 0.7, 3
RENDER_SCALE, JPEG_QUALITY = 2.0, 92        # below scale 2 table digits blur; below quality ~85 thin rules artifact


def pdf_text(path, cache_dir=None):
    path = Path(path)
    if cache_dir:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{path.stem}.{hashlib.md5(path.read_bytes()).hexdigest()[:16]}.txt"
        if cached.exists():
            return cached.read_text(encoding="utf-8")
    import pdfplumber
    chunks = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            chunks.append(PAGE_MARK.format(n=i))
            t = page.extract_text() or ""
            chunks.append(t)
            try:                                  # rotated-text salvage: regroup words by x-band, reverse each token
                words = page.extract_words()
                if words and t.count(" ") < len(words) / 3:
                    bands = {}
                    for w in words:
                        bands.setdefault(round(w["x0"] / 6), []).append(w)
                    for k in sorted(bands):
                        chunks.append("\n" + " ".join(w["text"][::-1] for w in sorted(bands[k], key=lambda w: -w["top"])))
            except Exception:
                pass
    text = "".join(chunks)
    if cache_dir:
        cached.write_text(text, encoding="utf-8")
    return text


def text_pages(text):
    parts = re.split(r"\n--- PDF PAGE (\d+) ---\n", text)
    return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)] or [(1, text)]


def sideways_pages(path):
    """[(sheet_no, rotation_degrees)] for pages whose text runs sideways."""
    import pdfplumber
    out = []
    with pdfplumber.open(path) as pdf:
        for i, pg in enumerate(pdf.pages, 1):
            try:
                words = pg.extract_words(extra_attrs=["upright"])   # 'direction' comes by default; asking for it raises
            except Exception:
                continue
            if len(words) < SIDEWAYS_MIN_WORDS:
                continue
            side = [w for w in words if not w.get("upright", True)]
            if len(side) < len(words) * SIDEWAYS_SHARE:
                continue
            ttb = sum(1 for w in side if w.get("direction") == "ttb")
            out.append((i, -90 if ttb >= len(side) / 2 else 90))
    return out


def rank_table_pages(path, top_n=3):
    """Pages most likely to hold the correlation matrix, from text signals: [(page, score, rotated)]."""
    import pdfplumber
    scored = []
    with pdfplumber.open(path) as pdf:
        for i, pg in enumerate(pdf.pages, 1):
            t = pg.extract_text() or ""; low = t.lower()
            try:
                nw = len(pg.extract_words())
            except Exception:
                nw = 0
            rotated = nw > 40 and t.count(" ") < nw / 3
            s = (12 * len(re.findall(r"correlation", low)) + 6 * len(re.findall(r"\bmean\b|\bs\.?d\.?\b|std\.? dev", low))
                 + 4 * len(re.findall(r"\btable\s*\d", low)) + 1.2 * len(re.findall(r"(?<![\d.])[01]?\.\d{2}(?![\d])", t))
                 + 0.8 * len(re.findall(r"\*{1,3}", t)) + (40 if rotated else 0))
            scored.append((i, round(s, 1), rotated))
    return sorted(scored, key=lambda x: -x[1])[:top_n]


_PDFIUM = threading.Lock()      # PDFium is not thread-safe; articles are read in parallel


def render(path, pages_with_turn, scale=RENDER_SCALE):
    """[(base64_jpeg, 'image/jpeg')] for (page, rotation) pairs."""
    import pypdfium2 as pdfium
    out = []
    with _PDFIUM:
        doc = pdfium.PdfDocument(str(path))
        for sheet, turn in pages_with_turn:
            if 1 <= sheet <= len(doc):
                pil = doc[sheet - 1].render(scale=scale).to_pil().convert("RGB").rotate(turn, expand=True)
                buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=JPEG_QUALITY)
                out.append((base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"))
        doc.close()
    return out


def table_images(path, include_upright_tables=False):
    """Sideways pages turned upright (always worth sending); optionally the best upright table pages too
    (for models that read only text, the image is the only way to see a table's layout and signs)."""
    side = sideways_pages(path)
    rank = {p: s for p, s, _ in rank_table_pages(path, top_n=99)}
    side = sorted(side, key=lambda t: -rank.get(t[0], 0))[:SIDEWAYS_MAX_PAGES]
    pages = list(side)
    if include_upright_tables:
        for p, _s, _rot in rank_table_pages(path, top_n=SIDEWAYS_MAX_PAGES):
            if p not in {q for q, _ in pages} and len(pages) < SIDEWAYS_MAX_PAGES:
                pages.append((p, 0))
    return render(path, pages)
