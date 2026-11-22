"""
docs.py — read the user's files (PDF, Word, text) and find the parts that answer a question.

Everything stays on the device: text is pulled out of the file here, cut into overlapping
pieces, and searched with SQLite full-text search (see db.py). No embeddings model needed,
so it works the same on a weak laptop and a gaming PC.

Arabic search needs a little help: we drop diacritics and tatweel, unify alef / ya / ta
marbuta forms, and strip the article "ال" (with و/ب/ف/ك/ل in front of it), so «المكتبة»
and «مكتبه» find each other.
"""

import io
import re
import unicodedata
import zipfile

MAX_BYTES = 25 * 1024 * 1024        # biggest file we accept
MAX_CHARS = 2_000_000               # text we keep from one file
CHUNK = 900                         # characters per piece
OVERLAP = 150

KINDS = {".pdf": "pdf", ".docx": "docx", ".txt": "text", ".md": "text", ".csv": "text", ".json": "text",
         ".html": "text", ".htm": "text", ".log": "text"}


class FileError(ValueError):
    """Shown to the user as is: say what went wrong and what to do."""


def kind_of(name):
    m = re.search(r"\.[A-Za-z0-9]+$", name or "")
    return KINDS.get(m.group(0).lower()) if m else None


# ------------------------------------------------------------------ extraction

def _decode(data):
    for enc in ("utf-8-sig", "utf-16", "cp1256", "latin-1"):
        try:
            text = data.decode(enc)
            if enc == "utf-16" and not data[:2] in (b"\xff\xfe", b"\xfe\xff"):
                continue
            return text
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def _docx(data):
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    except (zipfile.BadZipFile, KeyError):
        raise FileError("bad_docx")
    paras = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        p = re.sub(r"<w:tab/>", "\t", p)
        p = re.sub(r"<w:br/>", "\n", p)
        runs = re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", p, re.S)
        if runs:
            paras.append("".join(runs))
    text = "\n".join(paras)
    for a, b in (("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'"), ("&amp;", "&")):
        text = text.replace(a, b)
    return text


def _pdf(data):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise FileError("no_pdf_lib")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise FileError("pdf_locked")
        pages = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            if t.strip():
                pages.append(t)
            if sum(map(len, pages)) > MAX_CHARS:
                break
    except FileError:
        raise
    except Exception:
        raise FileError("bad_pdf")
    text = "\n\n".join(pages)
    if len(text.strip()) < 20:
        raise FileError("pdf_no_text")        # a scan: pictures of pages, no text inside
    return text


def extract(name, data):
    if len(data) > MAX_BYTES:
        raise FileError("too_big")
    kind = kind_of(name)
    if not kind:
        raise FileError("bad_type")
    if kind == "pdf":
        text = _pdf(data)
    elif kind == "docx":
        text = _docx(data)
    else:
        text = _decode(data)
        if name.lower().endswith((".html", ".htm")):
            text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
            text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text)        # Arabic presentation forms from PDFs -> letters
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()
    if not text:
        raise FileError("empty")
    return kind, text[:MAX_CHARS]


# ------------------------------------------------------------------ pieces

def chunks(text, size=CHUNK, overlap=OVERLAP):
    """Cut on paragraph / sentence ends near the size limit, with a little overlap."""
    out, i, n = [], 0, len(text)
    while i < n:
        end = min(n, i + size)
        if end < n:
            window = text[i + size // 2:end]
            cut = max(window.rfind("\n"), window.rfind(". "), window.rfind("۔"), window.rfind("؟"), window.rfind("? "))
            if cut != -1:
                end = i + size // 2 + cut + 1
        piece = text[i:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return out


# ------------------------------------------------------------------ search text

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ARTICLE = re.compile(r"^(?:[وفبكل]?ال|لل)(?=\w{2,})")
WORD = re.compile(r"\w+", re.UNICODE)

STOP = set("""
من في على عن الى إلى الي إلي اللي الذي التي هذا هذه هاد هاي هيك ذلك تلك هو هي هم انا أنا انت أنت احنا نحن
ما ماذا شو ايش إيش وش كيف ليش لماذا متى وين أين هل لا لم لن مع او أو ثم بس كان كانت يكون فيه فيها عند
كل بعض اي أي قد لقد انه أنه ان أن إن يا بدي بدك عندي عندك منو منها الملف ملف لخص لخصلي احكيلي قلي
the a an of to in on at for and or is are was were be been it this that these those what which who how
why when where do does did can could would should will i you he she we they me my your about from with as by
""".split())


def norm_word(w):
    w = _DIACRITICS.sub("", w.lower())
    w = w.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    w = w.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    if len(w) > 4:
        w = _ARTICLE.sub("", w)
    return w


def norm_text(text):
    return " ".join(norm_word(w) for w in WORD.findall(text))


def query_words(text, limit=24):
    out = []
    for w in WORD.findall(text):
        n = norm_word(w)
        if len(n) >= 2 and n not in STOP and w.lower() not in STOP and n not in out:
            out.append(n)
    return out[:limit]
