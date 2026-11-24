"""
server.py — local, OpenAI-compatible API + the app.

    python -m pcore.server --store store/ --data data/        (real model)
    python -m pcore.server --mock                             (no model: for UI work)

Then open http://127.0.0.1:8000 . Any OpenAI client also works with
base_url="http://127.0.0.1:8000/v1". Nothing leaves this machine: the server
binds to 127.0.0.1 only.
"""

import argparse
import json
import os
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import docs as docslib
from . import persona as personalib
from . import runtime as runtimelib
from .db import DB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSONA_FIELDS = ("id", "name", "name_en", "tagline", "tagline_en", "art", "face", "rights", "avatar", "personality", "created_by")


# ---------------------------------------------------------------- backends

class RealBackend:
    name = "real"

    def __init__(self, store_path, db, device, dtype, cache_experts):
        from . import engine, night
        self.night = night
        self.model, self.tok, self.store = engine.load(store_path, device=device, dtype=dtype,
                                                       core=db.get("core"), cache_experts=cache_experts)
        night.load_current_adapter(self.model, db)
        self.model_id = self.store.meta["source"]

    def _prompt_ids(self, messages):
        tok = self.tok
        if tok.chat_template:
            try:
                return tok.apply_chat_template(messages, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)["input_ids"]
            except Exception:          # template without a system role: fold it into the first user turn
                sys = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
                rest = [dict(m) for m in messages if m["role"] != "system"]
                if rest and rest[0]["role"] == "user" and sys:
                    rest[0]["content"] = sys + "\n\n" + rest[0]["content"]
                return tok.apply_chat_template(rest, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)["input_ids"]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return tok(text, return_tensors="pt")["input_ids"]

    def stream(self, messages, max_tokens, temperature):
        import torch
        from transformers import TextIteratorStreamer
        ids = self._prompt_ids(messages)[:, -3072:].to(next(self.model.parameters()).device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(input_ids=ids, max_new_tokens=max_tokens, streamer=streamer,
                      do_sample=temperature > 0, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        if temperature > 0:
            kwargs["temperature"] = temperature
        error = []

        def work():
            try:
                with torch.no_grad():
                    self.model.generate(**kwargs)
            except Exception as e:      # surface generation errors to the request
                error.append(e)
                streamer.end()

        t = threading.Thread(target=work, daemon=True)
        t.start()
        yield from streamer
        t.join()
        if error:
            raise error[0]

    def brain(self):
        return self.store.summary()

    def flush_usage(self, db):
        db.add_usage(self.store.drain_usage())

    def run_night(self, db, cfg_overrides):
        cfg = self.night.NightConfig(**cfg_overrides)
        return self.night.run_night(self.model, self.tok, self.store, db, cfg,
                                    os.path.join(ROOT, "exams", "general_exam.jsonl"),
                                    os.path.join(ROOT, "exams", "general_replay.jsonl"))

    def rollback(self, db):
        return self.night.rollback(self.model, db)


class MockBackend:
    """No model at all — canned replies, so the app can be built and demoed anywhere."""
    name, model_id = "mock", "mock"
    REPLIES = [
        "[happy] Hi! I'm running here on your computer, offline. This is mock mode: the replies are canned, not from a real model.",
        "[thinking] Let me think about it... There is no model in mock mode, but this is how I look while thinking and talking.",
        "[surprised] Good question! Run me with a real model and I'll answer it for real.",
        "[confused] Not sure I understood. Start the server with --store to talk to the real model.",
    ]

    def __init__(self):
        self.i = 0

    def stream(self, messages, max_tokens, temperature):
        reply = self.REPLIES[self.i % len(self.REPLIES)]
        self.i += 1
        time.sleep(0.9)
        for w in reply.split(" "):
            time.sleep(0.05)
            yield w + " "

    def brain(self):
        return {"hot_experts": 192, "cached_experts": 40, "total_experts": 768,
                "hot_ram_bytes": int(0.62e9), "cache_ram_bytes": int(0.13e9), "all_experts_bytes": int(2.48e9),
                "skeleton_bytes": int(0.21e9), "hit_rate_hot": 0.87, "hit_rate_ram": 0.94,
                "ssd_loads": 311, "ssd_seconds": 1.9, "activations": 48210}

    def flush_usage(self, db):
        pass

    def run_night(self, db, cfg_overrides):
        time.sleep(4)
        night = time.strftime("%Y-%m-%d_%H%M%S")
        report = {"night": night, "mock": True,
                  "core": {"experts_per_layer": 8, "of": 32, "swapped": 5, "coverage_before": 0.84,
                           "coverage_after": 0.87, "coverage_random": 0.25},
                  "ram": {"hot_core_bytes": int(0.62e9), "skeleton_bytes": int(0.21e9),
                          "all_experts_bytes": int(2.48e9)},
                  "learning": {"accepted": True, "reasons": [], "new_examples": 9, "corrections": 3, "replayed": 9,
                               "before": {"general": 1.92, "personal": 2.41},
                               "after": {"general": 1.93, "personal": 2.07},
                               "samples": ["How do I set up the GDT in protected mode?",
                                           "Write Verilog for a 4-bit counter"]}}
        db.add_report(night, report)
        return report

    def rollback(self, db):
        return "mock"


# ---------------------------------------------------------------- app

class ChatRequest(BaseModel):
    messages: list
    model: str | None = None
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    user: str | None = None            # used as the conversation id
    docs: list[int] | None = None      # files attached to this chat: search these first


class Feedback(BaseModel):
    message_id: int
    action: str
    correction: str | None = None


class Text(BaseModel):
    text: str


class PersonaChoice(BaseModel):
    id: str


class PersonaDesign(BaseModel):
    name: str
    tagline: str = ""
    personality: str = ""
    avatar: dict = {}


def growth_of(days, memories, nights, files=0):
    """How grown the brain looks (0..1). A new user starts with a sparse outline (~0.22);
    memories count most, then nights of learning, then days and files."""
    import math
    score = days + 3 * memories + 2 * nights + 2 * files
    return round(0.22 + 0.78 * (1 - math.exp(-score / 90)), 4)


DOC_CONTEXT_CHARS = 3200
COMPLETE_AT = 12          # saved memories at which Eve "knows the basics about you": the brain fills and turns gold


def create_app(backend, db, personas_dir, default_max_tokens=512, night_overrides=None):
    app = FastAPI(title="Eve")
    gen_lock = threading.Lock()
    state = {"sleeping": False, "night_error": None}
    night_overrides = night_overrides or {}

    def personas():
        return personalib.load_all(personas_dir)

    def current_persona():
        cards, _ = personas()
        if not cards:
            raise HTTPException(500, "no usable persona cards (check the rights block)")
        return cards.get(db.get("persona")) or cards.get("eve") or next(iter(cards.values()))

    def image_parts(attached, limit=3):
        """Attached images as data URIs for a vision model, newest-first, a few at most."""
        import base64, mimetypes
        out = []
        for d in (attached or []):
            doc = db.doc(int(d))
            if not doc or doc["kind"] != "image":
                continue
            path = db.doc_image_path(int(d))
            if not path:
                continue
            mime = mimetypes.guess_type(path)[0] or "image/png"
            with open(path, "rb") as f:
                out.append({"doc_id": int(d), "name": doc["name"],
                            "data_uri": f"data:{mime};base64," + base64.b64encode(f.read()).decode()})
            if len(out) >= limit:
                break
        return out

    def doc_context(question, attached, skip=()):
        """Pieces of the user's files that help answer. Attached files are searched first; when the
        question matches nothing in them ("summarise this"), the start of the file is used. Without
        attachments the whole library is searched, but only clear matches are used."""
        attached = [int(d) for d in (attached or []) if db.doc(int(d)) and int(d) not in skip][:5]
        if attached:
            found = db.search_docs(question, attached, k=3, min_hits=1)
            if not found:
                found = db.doc_head(attached[0], 3)
        else:
            found = db.search_docs(question, None, k=3, min_hits=2)
        out, used = [], 0
        for p in found:
            if used + len(p["text"]) > DOC_CONTEXT_CHARS and out:
                break
            out.append(p); used += len(p["text"])
        return out

    def content_of(m):
        c = m.get("content", "")
        if isinstance(c, list):        # OpenAI "content parts"
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        return str(c)

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        if state["sleeping"]:
            raise HTTPException(503, "The assistant is asleep (nightly learning in progress).")
        turns = [{"role": m.get("role", "user"), "content": content_of(m)} for m in req.messages
                 if m.get("role") in ("user", "assistant")]
        if not turns or turns[-1]["role"] != "user":
            raise HTTPException(400, "the last message must be from the user")
        question = turns[-1]["content"]
        conv = req.user or "default"

        # "remember: ..." / "تذكّر: ..." stores an explicit memory
        for prefix in ("remember:", "تذكر:", "تذكّر:"):
            if question.strip().lower().startswith(prefix):
                db.remember(question.strip()[len(prefix):], source="chat")
        card = current_persona()
        notes = [m["text"] for m in db.recall(question)]
        vision = bool(getattr(backend, "vision", False))
        images = image_parts(req.docs) if vision else []
        # with a vision model the image goes to the model directly, so skip its OCR text in the context
        passages = doc_context(question, req.docs, skip={im["doc_id"] for im in images})
        sources = list({p["doc_id"]: {"id": p["doc_id"], "name": p["name"]} for p in passages}.values())
        sources += [{"id": im["doc_id"], "name": im["name"]} for im in images
                    if im["doc_id"] not in {s["id"] for s in sources}]
        messages = [{"role": "system", "content": personalib.system_prompt(card, notes, passages)}] + turns[-20:]
        if images:
            last = dict(messages[-1])
            last["content"] = [{"type": "text", "text": last["content"]}] + \
                              [{"type": "image_url", "image_url": {"url": im["data_uri"]}} for im in images]
            messages[-1] = last

        db.add_message(conv, "user", question)
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        model_name = req.model or backend.model_id
        max_tokens = req.max_tokens or default_max_tokens
        temperature = 0.7 if req.temperature is None else req.temperature

        def pieces():
            """Yields visible text; returns via StopIteration nothing. Fills `result`."""
            filt, think, scrub = personalib.EmotionFilter(), personalib.ThinkFilter(), personalib.TagScrubber()
            with gen_lock:
                for piece in backend.stream(messages, max_tokens, temperature):
                    out = scrub.feed(filt.feed(think.feed(piece)))
                    if out:
                        result["text"] += out
                        yield out
                tail = scrub.feed(filt.feed(think.flush()) + filt.flush()) + scrub.flush()
                if tail:
                    result["text"] += tail
                    yield tail
                if not result["text"].strip():
                    # small models sometimes stop right after the mood tag: let them go on from it
                    again = messages + [{"role": "assistant", "content": f"[{filt.emotion}] "}]
                    think, scrub = personalib.ThinkFilter(), personalib.TagScrubber()
                    for piece in backend.stream(again, max_tokens, temperature):
                        out = scrub.feed(think.feed(piece))
                        if out:
                            result["text"] += out
                            yield out
                    out = scrub.feed(think.flush()) + scrub.flush()
                    if out:
                        result["text"] += out
                        yield out
                backend.flush_usage(db)
            result["emotion"] = filt.emotion
            result["id"] = db.add_message(conv, "assistant", result["text"].strip(), filt.emotion)

        result = {"text": "", "emotion": "neutral", "id": None}

        if not req.stream:
            for _ in pieces():
                pass
            return {"id": rid, "object": "chat.completion", "created": created, "model": model_name,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": result["text"].strip()}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "pcore": {"emotion": result["emotion"], "message_id": result["id"], "memories_used": len(notes),
                              "sources": sources}}

        def sse():
            def chunk(delta, finish=None, extra=None):
                body = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if extra:
                    body["pcore"] = extra
                return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"
            yield chunk({"role": "assistant", "content": ""})
            try:
                for piece in pieces():
                    yield chunk({"content": piece})
            except Exception as e:
                yield "data: " + json.dumps({"error": {"message": str(e)}}) + "\n\n"
            yield chunk({}, "stop", {"emotion": result["emotion"], "message_id": result["id"],
                                     "memories_used": len(notes), "sources": sources})
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": backend.model_id, "object": "model", "owned_by": "local"}]}

    # ---------------------------------------------------------- app api
    @app.post("/api/feedback")
    def feedback(fb: Feedback):
        if fb.action not in ("approve", "correct"):
            raise HTTPException(400, "action must be approve or correct")
        try:
            ok = db.feedback(fb.message_id, fb.action, fb.correction)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not ok:
            raise HTTPException(404, "no such assistant message")
        return {"ok": True}

    def grow():
        born = db.get("born")
        if born is None:
            born = time.time(); db.set("born", born)
        days = int((time.time() - born) // 86400) + 1
        mem, nights, files = len(db.memories()), len(db.reports(10000)), len(db.docs())
        return {"value": growth_of(days, mem, nights, files), "days": days, "memories": mem,
                "nights": nights, "files": files, "complete": mem >= COMPLETE_AT, "complete_at": COMPLETE_AT}

    @app.get("/api/state")
    def get_state():
        cards, refused = personas()
        card = current_persona()
        reports = db.reports(1)
        latest = reports[0] if reports else None
        return {
            "backend": backend.name, "model": backend.model_id, "sleeping": state["sleeping"],
            "night_error": state["night_error"],
            "persona": {k: card.get(k) for k in PERSONA_FIELDS},
            "personas": [{k: c.get(k) for k in PERSONA_FIELDS} for c in cards.values()],
            "refused_personas": refused,
            "brain": backend.brain(),
            "counts": {"memories": len(db.memories()), "examples_waiting": len(db.examples(new_only=True)),
                       "examples_learned": len(db.examples(new_only=False)),
                       "adapters": len(db.get("adapters", []))},
            "growth": grow(),
            "morning": latest if latest and db.get("morning_seen") != latest["night"] else None,
            "last_report": latest,
        }

    # ---------------------------------------------------------- files
    ocr_state = {"installing": False}

    def ensure_ocr_async():
        home = os.path.dirname(db.dir)
        ok = runtimelib.ensure_ocr(home)
        ocr_state["installing"] = False
        return ok

    @app.post("/api/files")
    async def add_file(request: Request, name: str):
        data = await request.body()
        name = os.path.basename(name.replace("\\", "/"))[:120] or "file"
        if docslib.kind_of(name) == "image" and not docslib.ocr_available() and not getattr(backend, "vision", False):
            if not ocr_state["installing"]:
                ocr_state["installing"] = True
                threading.Thread(target=ensure_ocr_async, daemon=True).start()
            raise HTTPException(400, {"code": "ocr_setup"})
        try:
            kind, text = docslib.extract(name, data)
        except docslib.FileError as e:
            raise HTTPException(400, {"code": str(e)})
        if kind == "image":
            ext = os.path.splitext(name)[1].lower() or ".png"
            doc_id = db.add_doc(name, kind, text, image_bytes=data, ext=ext)
        else:
            doc_id = db.add_doc(name, kind, text)
        d = db.doc(doc_id)
        d["has_image"] = bool(d.pop("image", None))
        return d

    @app.get("/api/files")
    def list_files():
        return db.docs()

    @app.get("/api/files/{doc_id}/image")
    def file_image(doc_id: int):
        path = db.doc_image_path(doc_id)
        if not path:
            raise HTTPException(404, "no image")
        return FileResponse(path)

    @app.delete("/api/files/{doc_id}")
    def delete_file(doc_id: int):
        if not db.delete_doc(doc_id):
            raise HTTPException(404, "no such file")
        return {"ok": True}

    @app.post("/api/morning/seen")
    def morning_seen():
        reports = db.reports(1)
        if reports:
            db.set("morning_seen", reports[0]["night"])
        return {"ok": True}

    @app.get("/api/history")
    def history(conv: str = "default"):
        return db.history(conv)

    @app.get("/api/memories")
    def memories():
        return db.memories()

    @app.post("/api/memories")
    def add_memory(t: Text):
        if not t.text.strip():
            raise HTTPException(400, "empty")
        return {"id": db.remember(t.text, source="user")}

    @app.delete("/api/memories/{memory_id}")
    def delete_memory(memory_id: int):
        if not db.forget(memory_id):
            raise HTTPException(404, "no such memory")
        return {"ok": True}

    @app.post("/api/persona")
    def set_persona(choice: PersonaChoice):
        cards, refused = personas()
        if choice.id not in cards:
            raise HTTPException(400, "unknown persona, or its rights block is missing")
        db.set("persona", choice.id)
        return {"ok": True}

    @app.post("/api/personas")
    def create_persona(d: PersonaDesign):
        try:
            card = personalib.user_card("u" + uuid.uuid4().hex[:8], d.name, d.tagline, d.personality, d.avatar)
        except ValueError as e:
            raise HTTPException(400, str(e))
        personalib.save_card(personas_dir, card)
        db.set("persona", card["id"])
        return {"ok": True, "id": card["id"]}

    def user_persona(persona_id):
        cards, _ = personas()
        card = cards.get(persona_id)
        if not card:
            raise HTTPException(404, "no such persona")
        if card.get("created_by") != "user":
            raise HTTPException(403, "only characters designed in the app can be changed here")
        return card

    @app.put("/api/personas/{persona_id}")
    def update_persona(persona_id: str, d: PersonaDesign):
        user_persona(persona_id)
        try:
            card = personalib.user_card(persona_id, d.name, d.tagline, d.personality, d.avatar)
        except ValueError as e:
            raise HTTPException(400, str(e))
        personalib.save_card(personas_dir, card)
        return {"ok": True, "id": persona_id}

    @app.delete("/api/personas/{persona_id}")
    def delete_persona(persona_id: str):
        user_persona(persona_id)
        os.remove(os.path.join(personas_dir, f"{persona_id}.json"))
        if db.get("persona") == persona_id:
            db.set("persona", None)
        return {"ok": True}

    def night_job():
        try:
            with gen_lock:
                backend.run_night(db, night_overrides)
            state["night_error"] = None
        except Exception as e:
            state["night_error"] = str(e)
        finally:
            state["sleeping"] = False

    @app.post("/api/night")
    def night_now():
        if state["sleeping"]:
            raise HTTPException(409, "already asleep")
        state["sleeping"] = True
        threading.Thread(target=night_job, daemon=True).start()
        return {"ok": True}

    @app.post("/api/rollback")
    def rollback():
        if state["sleeping"]:
            raise HTTPException(409, "asleep")
        with gen_lock:
            dropped = backend.rollback(db)
        if dropped is None:
            raise HTTPException(404, "nothing to roll back")
        return {"ok": True, "dropped": dropped}

    @app.post("/api/wipe")
    def wipe():
        if state["sleeping"]:
            raise HTTPException(409, "asleep")
        with gen_lock:
            while backend.rollback(db):
                if backend.name == "mock":
                    break
            db.wipe()
            adir = os.path.join(db.dir, "adapters")
            if os.path.isdir(adir):
                for f in os.listdir(adir):
                    os.remove(os.path.join(adir, f))
        return {"ok": True}

    @app.get("/")
    def index():
        return FileResponse(os.path.join(ROOT, "web", "index.html"))

    app.mount("/web", StaticFiles(directory=os.path.join(ROOT, "web")), name="web")
    app.state.night_job = night_job
    app.state.flags = state
    return app


def schedule_night(app, at):
    """Run the night job once a day at HH:MM (only on mains power, if that can be detected)."""
    def on_battery():
        try:
            import psutil
            b = psutil.sensors_battery()
            return b is not None and not b.power_plugged
        except Exception:
            return False

    def loop():
        last = None
        while True:
            now = time.strftime("%H:%M")
            day = time.strftime("%Y-%m-%d")
            if now == at and last != day and not app.state.flags["sleeping"] and not on_battery():
                last = day
                app.state.flags["sleeping"] = True
                app.state.night_job()
            time.sleep(20)
    threading.Thread(target=loop, daemon=True).start()


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", help="folder made by: python -m pcore.engine split")
    ap.add_argument("--mock", action="store_true", help="run without a model (canned replies)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--personas", default=os.path.join(ROOT, "personas"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--cache-experts", type=int, default=64, help="extra experts kept in RAM (LRU) besides the core")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--night-at", default="", help="HH:MM — learn automatically every night at this time")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if not args.mock and not args.store:
        ap.error("give --store, or --mock")
    db = DB(args.data)
    backend = MockBackend() if args.mock else RealBackend(args.store, db, args.device, args.dtype, args.cache_experts)
    app = create_app(backend, db, args.personas, args.max_tokens)
    if args.night_at:
        schedule_night(app, args.night_at)
    print(f"\n  open http://127.0.0.1:{args.port}\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
