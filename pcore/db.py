"""
db.py — everything the user owns, in one local SQLite file.

  messages   every chat turn; assistant turns can be approved or corrected
  memories   long-term notes, searched with FTS5 (works offline, Arabic + English)
  usage      which experts fired, per day  -> feeds the nightly core rebuild
  reports    one row per night ("what I learned")
  docs       files the user gave the assistant to read, cut into searchable chunks
  kv         small state (current adapter, current persona, ...)

The data is independent of the model: when a better base model comes out,
retrain the adapter and rebuild the core from this same file.
"""

import json
import os
import re
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY, conv TEXT, role TEXT, content TEXT, emotion TEXT,
  status TEXT, correction TEXT, trained_night TEXT, created REAL);
CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, text TEXT, source TEXT, created REAL);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, content='memories', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, text) VALUES('delete', old.id, old.text); END;
CREATE TABLE IF NOT EXISTS usage(day TEXT, layer INTEGER, expert INTEGER, count INTEGER,
  PRIMARY KEY(day, layer, expert));
CREATE TABLE IF NOT EXISTS reports(night TEXT PRIMARY KEY, body TEXT, created REAL);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, name TEXT, kind TEXT, chars INTEGER, chunks INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc_id INTEGER, ord INTEGER, text TEXT);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id, ord);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(norm);
"""

WORD = re.compile(r"\w{2,}", re.UNICODE)


def today():
    return time.strftime("%Y-%m-%d")


class DB:
    def __init__(self, data_dir):
        os.makedirs(data_dir, exist_ok=True)
        self.dir = data_dir
        self.path = os.path.join(data_dir, "pcore.db")
        self.lock = threading.RLock()
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)
        self.con.commit()

    def _x(self, sql, args=()):
        with self.lock:
            cur = self.con.execute(sql, args)
            self.con.commit()
            return cur

    def _q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.con.execute(sql, args).fetchall()]

    # ---------------------------------------------------------- messages
    def add_message(self, conv, role, content, emotion=None):
        return self._x("INSERT INTO messages(conv, role, content, emotion, created) VALUES(?,?,?,?,?)",
                       (conv, role, content, emotion, time.time())).lastrowid

    def feedback(self, message_id, action, correction=None):
        """action: 'approve' or 'correct' (with the text the user wanted instead)."""
        if action == "correct" and not (correction or "").strip():
            raise ValueError("a correction needs text")
        status = {"approve": "approved", "correct": "corrected"}[action]
        cur = self._x("UPDATE messages SET status=?, correction=?, trained_night=NULL WHERE id=? AND role='assistant'",
                      (status, correction if action == "correct" else None, message_id))
        return cur.rowcount == 1

    def history(self, conv, limit=50):
        return self._q("SELECT * FROM (SELECT * FROM messages WHERE conv=? ORDER BY id DESC LIMIT ?) ORDER BY id",
                       (conv, limit))

    def user_messages_since(self, ts):
        return self._q("SELECT id, conv, content, created FROM messages WHERE role='user' AND created>? ORDER BY id", (ts,))

    def examples(self, new_only=None):
        """Training pairs (question -> answer the user approved or wrote).

        Only human-vetted answers are used: the model never trains on its own
        unreviewed output (that is what causes model collapse).
        """
        where = {None: "", True: "AND a.trained_night IS NULL", False: "AND a.trained_night IS NOT NULL"}[new_only]
        rows = self._q(f"""
            SELECT a.id, a.status, a.content, a.correction,
                   (SELECT u.content FROM messages u WHERE u.conv=a.conv AND u.role='user' AND u.id<a.id
                    ORDER BY u.id DESC LIMIT 1) AS question
            FROM messages a WHERE a.role='assistant' AND a.status IN ('approved','corrected') {where}
            ORDER BY a.id""")
        return [{"id": r["id"], "q": r["question"],
                 "a": r["correction"] if r["status"] == "corrected" else r["content"],
                 "corrected": r["status"] == "corrected"} for r in rows if r["question"]]

    def mark_trained(self, ids, night):
        with self.lock:
            self.con.executemany("UPDATE messages SET trained_night=? WHERE id=?", [(night, i) for i in ids])
            self.con.commit()

    # ---------------------------------------------------------- memory
    def remember(self, text, source="user"):
        text = text.strip()
        if not text:
            return None
        dup = self._q("SELECT id FROM memories WHERE text=?", (text,))
        if dup:
            return dup[0]["id"]
        return self._x("INSERT INTO memories(text, source, created) VALUES(?,?,?)",
                       (text, source, time.time())).lastrowid

    def recall(self, query, k=4):
        words = list(dict.fromkeys(WORD.findall(query.lower())))[:24]
        if not words:
            return []
        match = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
        try:
            return self._q("""SELECT m.id, m.text, m.source FROM memories_fts f JOIN memories m ON m.id=f.rowid
                              WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT ?""", (match, k))
        except sqlite3.OperationalError:
            return []

    def memories(self):
        return self._q("SELECT id, text, source, created FROM memories ORDER BY id DESC")

    def forget(self, memory_id):
        return self._x("DELETE FROM memories WHERE id=?", (memory_id,)).rowcount == 1

    # ---------------------------------------------------------- files
    def add_doc(self, name, kind, text):
        from . import docs as docslib
        pieces = docslib.chunks(text)
        with self.lock:
            cur = self.con.execute("INSERT INTO docs(name, kind, chars, chunks, created) VALUES(?,?,?,?,?)",
                                   (name, kind, len(text), len(pieces), time.time()))
            doc_id = cur.lastrowid
            for i, piece in enumerate(pieces):
                cid = self.con.execute("INSERT INTO chunks(doc_id, ord, text) VALUES(?,?,?)", (doc_id, i, piece)).lastrowid
                self.con.execute("INSERT INTO chunks_fts(rowid, norm) VALUES(?,?)", (cid, docslib.norm_text(piece)))
            self.con.commit()
        return doc_id

    def docs(self):
        return self._q("SELECT id, name, kind, chars, chunks, created FROM docs ORDER BY id DESC")

    def doc(self, doc_id):
        rows = self._q("SELECT id, name, kind, chars, chunks, created FROM docs WHERE id=?", (doc_id,))
        return rows[0] if rows else None

    def delete_doc(self, doc_id):
        with self.lock:
            ids = [r[0] for r in self.con.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
            self.con.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
            self.con.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            n = self.con.execute("DELETE FROM docs WHERE id=?", (doc_id,)).rowcount
            self.con.commit()
        return n == 1

    def search_docs(self, question, doc_ids=None, k=3, min_hits=2):
        """Best chunks for a question. A chunk must contain at least `min_hits` of the question's
        words (fewer if the question is short), so a vague match does not drag a file into the chat."""
        from . import docs as docslib
        words = docslib.query_words(question)
        if not words:
            return []
        match = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
        where, args = "", [match]
        if doc_ids:
            where = " AND c.doc_id IN (%s)" % ",".join("?" * len(doc_ids))
            args += [int(i) for i in doc_ids]
        try:
            rows = self._q(f"""SELECT c.id, c.doc_id, c.ord, c.text, d.name, f.norm FROM chunks_fts f
                               JOIN chunks c ON c.id=f.rowid JOIN docs d ON d.id=c.doc_id
                               WHERE chunks_fts MATCH ?{where} ORDER BY bm25(chunks_fts) LIMIT 30""", args)
        except sqlite3.OperationalError:
            return []
        need = min(min_hits, len(words))
        out = []
        for r in rows:
            have = set(r.pop("norm").split())
            hits = sum(1 for w in words if w in have)
            if hits >= need:
                r["hits"] = hits
                out.append(r)
        out.sort(key=lambda r: -r["hits"])              # stable: keeps bm25 order among equals
        return out[:k]

    def doc_head(self, doc_id, n=2):
        return self._q("""SELECT c.id, c.doc_id, c.ord, c.text, d.name FROM chunks c JOIN docs d ON d.id=c.doc_id
                          WHERE c.doc_id=? ORDER BY c.ord LIMIT ?""", (doc_id, n))

    # ---------------------------------------------------------- expert usage
    def add_usage(self, counts, day=None):
        import numpy as np
        day = day or today()
        rows = [(day, int(l), int(e), int(counts[l, e])) for l, e in zip(*np.nonzero(counts))]
        with self.lock:
            self.con.executemany("""INSERT INTO usage(day, layer, expert, count) VALUES(?,?,?,?)
                ON CONFLICT(day, layer, expert) DO UPDATE SET count = count + excluded.count""", rows)
            self.con.commit()

    def usage(self, shape, decay=0.9, day=None):
        """Decayed sum over days: recent days count more, old habits fade slowly."""
        import numpy as np
        days = sorted({r["day"] for r in self._q("SELECT DISTINCT day FROM usage")}, reverse=True)
        if day is not None:
            days = [d for d in days if d == day]
        out = np.zeros(shape, dtype=np.float64)
        for age, d in enumerate(days):
            for r in self._q("SELECT layer, expert, count FROM usage WHERE day=?", (d,)):
                if r["layer"] < shape[0] and r["expert"] < shape[1]:
                    out[r["layer"], r["expert"]] += r["count"] * (decay ** age if day is None else 1.0)
        return out

    # ---------------------------------------------------------- state
    def get(self, key, default=None):
        rows = self._q("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def set(self, key, value):
        self._x("INSERT INTO kv(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value, ensure_ascii=False)))

    def add_report(self, night, body):
        self._x("INSERT OR REPLACE INTO reports(night, body, created) VALUES(?,?,?)",
                (night, json.dumps(body, ensure_ascii=False), time.time()))

    def reports(self, limit=14):
        return [json.loads(r["body"]) for r in
                self._q("SELECT body FROM reports ORDER BY created DESC LIMIT ?", (limit,))]

    def wipe(self):
        """Delete everything the system knows about the user."""
        with self.lock:
            for t in ("messages", "memories", "usage", "reports", "kv", "docs", "chunks", "chunks_fts"):
                self.con.execute(f"DELETE FROM {t}")
            self.con.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self.con.commit()
            self.con.execute("VACUUM")
