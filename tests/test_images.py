"""Images: OCR reading (text becomes a searchable file), the image is stored and served,
and when a vision backend is active the image is sent to the model as a content part.

    python tests/test_images.py
"""
import io
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from pcore import docs as docslib
from pcore.db import DB
from pcore.server import MockBackend, create_app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def an_image(text_lines):
    im = Image.new("RGB", (760, 90 + 70 * len(text_lines)), "white")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except OSError:
        f = None
    for i, line in enumerate(text_lines):
        d.text((30, 40 + 70 * i), line, font=f, fill="black")
    buf = io.BytesIO(); im.save(buf, "PNG"); return buf.getvalue()


# ---- OCR extraction, directly
if docslib._ocr_engine() is None:
    print("SKIP: OCR engine (rapidocr-onnxruntime) not installed")
    sys.exit(0)
kind, text = docslib.extract("note.png", an_image(["Launch code", "MANGO 2026"]))
assert kind == "image", kind
assert "MANGO" in text.upper() and "2026" in text, repr(text)


class Recorder(MockBackend):
    vision = False
    def stream(self, messages, max_tokens, temperature):
        self.last = messages
        yield "[neutral] ok"


tmp = tempfile.mkdtemp()
try:
    db = DB(os.path.join(tmp, "data"))
    be = Recorder()
    c = TestClient(create_app(be, db, os.path.join(ROOT, "personas")))

    # upload an image: OCR runs, the image is stored and served
    img = an_image(["Invoice total", "USD 4240"])
    r = c.post("/api/files", params={"name": "bill.png"}, content=img)
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["kind"] == "image" and doc["has_image"] and "image" not in doc, doc
    assert c.get(f"/api/files/{doc['id']}/image").headers["content-type"].startswith("image/")

    def ask(q, docs=None):
        body = {"messages": [{"role": "user", "content": q}]}
        if docs is not None:
            body["docs"] = docs
        j = c.post("/v1/chat/completions", json=body).json()
        return j["pcore"]["sources"], be.last

    # no vision model: the image's OCR text answers, as a normal source
    src, msgs = ask("what is the invoice total?", docs=[doc["id"]])
    assert [s["name"] for s in src] == ["bill.png"], src
    sys_text = msgs[0]["content"]
    assert "4240" in sys_text, "OCR text should be in the context"
    assert isinstance(msgs[-1]["content"], str), "no vision model: last message stays plain text"

    # vision model on: the image is sent to the model, its OCR text is NOT dumped into the context
    be.vision = True
    src, msgs = ask("what is in this image?", docs=[doc["id"]])
    assert [s["name"] for s in src] == ["bill.png"], src
    parts = msgs[-1]["content"]
    assert isinstance(parts, list), "vision: last message should be content parts"
    kinds = [p["type"] for p in parts]
    assert kinds.count("image_url") == 1 and "text" in kinds, kinds
    uri = [p for p in parts if p["type"] == "image_url"][0]["image_url"]["url"]
    assert uri.startswith("data:image/") and ";base64," in uri
    assert "4240" not in msgs[0]["content"], "vision: OCR text should not also be dumped in the context"

    # delete removes the image file
    path = db.doc_image_path(doc["id"])
    assert path and os.path.exists(path)
    c.delete(f"/api/files/{doc['id']}")
    assert not os.path.exists(path)
    print("IMAGES OK")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
