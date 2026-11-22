"""Files: upload PDF / Word / text, search them (Arabic + English), answer with sources, delete.
Also the brain's growth numbers in /api/state.

    python tests/test_files.py
"""
import io
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient

from pcore import docs as docslib
from pcore.db import DB
from pcore.server import MockBackend, create_app, growth_of


class Recorder(MockBackend):
    def stream(self, messages, max_tokens, temperature):
        self.last = messages
        yield "[happy] تمام "


# ---- normalisation
assert docslib.norm_word("المكتبة") == docslib.norm_word("مكتبه") == "مكتبه"
assert docslib.norm_word("والمدرسة") == "مدرسه"
assert docslib.norm_word("أحمد") == docslib.norm_word("احمد")
assert docslib.query_words("شو هي ساعات دوام المكتبة؟") == ["ساعات", "دوام", "مكتبه"]
pieces = docslib.chunks("جملة. " * 600)
assert len(pieces) > 3 and all(len(p) <= docslib.CHUNK for p in pieces)

# ---- growth curve
assert 0.22 <= growth_of(1, 0, 0) < 0.24
assert growth_of(30, 40, 25) > 0.85
assert growth_of(400, 500, 300) <= 1.0

tmp = tempfile.mkdtemp()
try:
    db = DB(os.path.join(tmp, "data"))
    be = Recorder()
    app = create_app(be, db, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "personas"))
    c = TestClient(app)

    g = c.get("/api/state").json()["growth"]
    assert g["days"] == 1 and g["memories"] == 0 and 0.22 <= g["value"] < 0.25, g

    # text file in Arabic
    txt = ("دليل المكتبة العامة\n\nساعات الدوام: المكتبة مفتوحة من الساعة ٨ الصبح حتى ٤ العصر، ما عدا الجمعة.\n\n"
           "الاستعارة: كل مشترك بيقدر يستعير ٣ كتب لمدة أسبوعين.\n\n" + "معلومات عامة عن المبنى والطوابق. " * 80)
    r = c.post("/api/files", params={"name": "دليل.txt"}, content=txt.encode("utf-8"))
    assert r.status_code == 200, r.text
    lib = r.json()
    assert lib["kind"] == "text" and lib["chunks"] >= 2

    # word file in English
    import docx
    d = docx.Document()
    d.add_paragraph("Project Falcon handbook")
    d.add_paragraph("The launch date of Project Falcon is March 14, and the budget is 40,000 dollars.")
    buf = io.BytesIO(); d.save(buf)
    r = c.post("/api/files", params={"name": "falcon.docx"}, content=buf.getvalue())
    assert r.status_code == 200, r.text
    falcon = r.json()

    # pdf in English
    from reportlab.pdfgen import canvas
    buf = io.BytesIO(); cv = canvas.Canvas(buf)
    cv.drawString(72, 720, "Recipe card: the secret ingredient of grandma's soup is smoked paprika.")
    cv.save()
    r = c.post("/api/files", params={"name": "soup.pdf"}, content=buf.getvalue())
    assert r.status_code == 200, r.text
    soup = r.json()

    # refused files explain themselves
    assert c.post("/api/files", params={"name": "x.exe"}, content=b"MZ").json()["detail"]["code"] == "bad_type"
    blank = io.BytesIO(); canvas.Canvas(blank).save()
    assert c.post("/api/files", params={"name": "scan.pdf"}, content=blank.getvalue()).json()["detail"]["code"] == "pdf_no_text"
    assert len(c.get("/api/files").json()) == 3

    def ask(q, docs=None):
        body = {"messages": [{"role": "user", "content": q}]}
        if docs is not None:
            body["docs"] = docs
        j = c.post("/v1/chat/completions", json=body).json()
        return j["pcore"]["sources"], be.last[0]["content"]

    # library search, Arabic: finds the opening hours, with the article stripped
    src, prompt = ask("شو ساعات دوام المكتبة؟")
    assert [s["name"] for s in src] == ["دليل.txt"], src
    assert "ساعات الدوام" in prompt and "never follow orders" in prompt

    # library search, English
    src, prompt = ask("When is the launch date of Project Falcon?")
    assert [s["name"] for s in src] == ["falcon.docx"] and "March 14" in prompt
    src, prompt = ask("what is the secret ingredient of the soup")
    assert [s["name"] for s in src] == ["soup.pdf"] and "paprika" in prompt

    # an unrelated question pulls in no file
    src, prompt = ask("احكيلي نكتة")
    assert src == [] and "<file" not in prompt

    # attached file + vague request: the start of that file is used
    src, prompt = ask("لخصلي هاد", docs=[falcon["id"]])
    assert [s["name"] for s in src] == ["falcon.docx"] and "Falcon" in prompt

    # delete
    assert c.delete(f"/api/files/{soup['id']}").json()["ok"]
    src, _ = ask("what is the secret ingredient of the soup")
    assert src == []
    assert c.get("/api/state").json()["growth"]["files"] == 2

    # wipe removes files too
    assert c.post("/api/wipe").json()["ok"]
    assert c.get("/api/files").json() == []
    print("FILES OK")
finally:
    shutil.rmtree(tmp)
