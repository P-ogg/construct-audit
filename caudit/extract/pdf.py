"""PDF text, page splitting, sideways-table detection and upright page images. Ported from pipeline3/core.py.

Text is used (a) to verify quotes and numbers and (b) as the document input for models that cannot read PDFs
(most open-weights models). Wide correlation matrices are often printed sideways; on one article the model read
every value of a rotated 120-cell matrix and still attached them to the wrong pairs. Re-rendering those pages
upright removed the problem.
"""
import threading
import base64, hashlib, io, json, re
from pathlib import Path

PAGE_MARK = "\n--- PDF PAGE {n} ---\n"
PAGE_MARKER = re.compile(r"---\s*PDF PAGE \d+\s*---")
SIDEWAYS_MIN_WORDS, SIDEWAYS_SHARE, SIDEWAYS_MAX_PAGES = 40, 0.7, 3
RENDER_SCALE, JPEG_QUALITY = 2.0, 92        # below scale 2 table digits blur; below quality ~85 thin rules artifact


_PARSE = threading.Lock()       # one PDF parsed at a time: parsing is memory-hungry and the hosted app has little memory


def _salvage(page, t):
    """Rotated-text salvage: regroup words by x-band and reverse each token."""
    try:
        words = page.extract_words()
        if words and t.count(" ") < len(words) / 3:
            bands = {}
            for w in words:
                bands.setdefault(round(w["x0"] / 6), []).append(w)
            return "".join("\n" + " ".join(w["text"][::-1] for w in sorted(bands[k], key=lambda w: -w["top"])) for k in sorted(bands))
    except Exception:
        pass
    return ""


def _score(t, nw):
    low = t.lower(); rotated = nw > 40 and t.count(" ") < nw / 3
    s = (12 * len(re.findall(r"correlation", low)) + 6 * len(re.findall(r"\bmean\b|\bs\.?d\.?\b|std\.? dev", low))
         + 4 * len(re.findall(r"\btable\s*\d", low)) + 1.2 * len(re.findall(r"(?<![\d.])[01]?\.\d{2}(?![\d])", t))
         + 0.8 * len(re.findall(r"\*{1,3}", t)) + (40 if rotated else 0))
    return round(s, 1), rotated


def analyse(path, cache_dir=None):
    """One pass over the PDF: its text (with page markers), sideways pages and table-page scores.
    Each page's memory is released as soon as it has been read; results are cached in cache_dir."""
    path = Path(path)
    if cache_dir:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{path.stem}.{hashlib.md5(path.read_bytes()).hexdigest()[:16]}"
        txt, meta = cache_dir / f"{tag}.txt", cache_dir / f"{tag}.pages.json"
        if txt.exists() and meta.exists():
            m = json.loads(meta.read_text())
            return dict(text=txt.read_text(encoding="utf-8"), sideways=[tuple(x) for x in m["sideways"]], ranks=[tuple(x) for x in m["ranks"]])
    import pdfplumber
    chunks, sideways, ranks = [], [], []
    with _PARSE, pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            try:
                t = page.extract_text() or ""
                chunks += [PAGE_MARK.format(n=i), t, _salvage(page, t)]
                try:
                    words = page.extract_words(extra_attrs=["upright"])   # 'direction' comes by default; asking for it raises
                except Exception:
                    words = []
                if len(words) >= SIDEWAYS_MIN_WORDS:
                    side = [w for w in words if not w.get("upright", True)]
                    if len(side) >= len(words) * SIDEWAYS_SHARE:
                        ttb = sum(1 for w in side if w.get("direction") == "ttb")
                        sideways.append((i, -90 if ttb >= len(side) / 2 else 90))
                sc, rot = _score(t, len(words))
                ranks.append((i, sc, rot))
            finally:
                try:
                    page.close()                  # frees the page's parsed objects
                except Exception:
                    pass
    out = dict(text="".join(chunks), sideways=sideways, ranks=sorted(ranks, key=lambda x: -x[1]))
    if cache_dir:
        txt.write_text(out["text"], encoding="utf-8")
        meta.write_text(json.dumps({"sideways": sideways, "ranks": out["ranks"]}))
    return out


def pdf_text(path, cache_dir=None):
    return analyse(path, cache_dir)["text"]


def text_pages(text):
    parts = re.split(r"\n--- PDF PAGE (\d+) ---\n", text)
    return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)] or [(1, text)]


def sideways_pages(path):
    """[(sheet_no, rotation_degrees)] for pages whose text runs sideways."""
    return analyse(path)["sideways"]


def rank_table_pages(path, top_n=3):
    """Pages most likely to hold the correlation matrix, from text signals: [(page, score, rotated)]."""
    return analyse(path)["ranks"][:top_n]


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


def table_images(path, include_upright_tables=False, info=None):
    """Sideways pages turned upright (always worth sending); optionally the best upright table pages too
    (for models that read only text, the image is the only way to see a table's layout and signs).
    info: the result of analyse(), so the PDF is not parsed again."""
    info = info or analyse(path)
    rank = {p: sc for p, sc, _ in info["ranks"]}
    side = sorted(info["sideways"], key=lambda t: -rank.get(t[0], 0))[:SIDEWAYS_MAX_PAGES]
    pages = list(side)
    if include_upright_tables:
        for p, _s, _rot in info["ranks"][:SIDEWAYS_MAX_PAGES]:
            if p not in {q for q, _ in pages} and len(pages) < SIDEWAYS_MAX_PAGES:
                pages.append((p, 0))
    return render(path, pages)
