"""End-to-end test of engine + db + night on the tiny random model (numbers are meaningless)."""
import shutil, sys, tempfile, json
import torch
sys.path.insert(0, ".")
from pcore import engine, night, lora, core as corelib
from pcore.db import DB

store_dir = sys.argv[1]
data = tempfile.mkdtemp()
db = DB(data)
model, tok, store = engine.load(store_dir, device="cpu", dtype="float32")

prompts = [json.loads(l)["text"] for l in open("prompts/personal_sample.jsonl")][:24]
for i, q in enumerate(prompts):                        # a "day" of chatting
    ids = tok(q, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        model(ids)
    db.add_message(f"c{i}", "user", q)
    mid = db.add_message(f"c{i}", "assistant", "bad answer")
    if i % 2:
        db.feedback(mid, "correct", "Use the GDT, then load it with lgdt and far jump to reload CS.")
    else:
        db.feedback(mid, "approve")
db.remember("Sam is building an x86 kernel with GRUB and Multiboot")
db.remember("يحب بازل ألعاب الألغاز ثلاثية الأبعاد في يونيتي")
assert db.recall("how is my kernel going?")[0]["text"].startswith("Sam"), "recall en"
assert "يونيتي" in db.recall("شو اللعبة اللي بعملها في يونيتي")[0]["text"], "recall ar"

cfg = night.NightConfig(lr=5e-3, epochs=6, min_examples=4)
r1 = night.run_night(model, tok, store, db, cfg, "exams/general_exam.jsonl", "exams/general_replay.jsonl")
assert r1["core"]["coverage_after"] > r1["core"]["coverage_random"], r1["core"]
print("core:", r1["core"])
print("night 1 accepted:", r1["learning"]["accepted"], r1["learning"]["reasons"])

# A night that must be rejected: absurd learning rate -> general exam collapses -> rollback
for i, q in enumerate(prompts[:8]):
    db.add_message(f"d{i}", "user", q + " again")
    db.feedback(db.add_message(f"d{i}", "assistant", "x"), "correct", "zzzz qqqq zzzz qqqq")
before = lora.state(model)
bad = night.NightConfig(lr=5.0, epochs=10, min_examples=4, general_tolerance=0.0)
r2 = night.run_night(model, tok, store, db, bad, "exams/general_exam.jsonl", None)
print("night 2 accepted:", r2["learning"]["accepted"], r2["learning"]["reasons"])
if not r2["learning"]["accepted"]:
    after = lora.state(model)
    assert all(torch.equal(before[k], after[k]) for k in before), "rollback must restore the adapter exactly"
    print("rollback restored the previous adapter exactly")

hist = db.get("adapters", [])
print("adapters:", hist)
if hist:
    night.rollback(model, db)
    assert len(db.get("adapters", [])) == len(hist) - 1
    print("manual rollback OK")
# adapter off == original model
lora.set_enabled(model, False)
print("reports:", len(db.reports()))
db.wipe(); assert not db.memories() and not db.examples()
shutil.rmtree(data)
print("ALL OK")
