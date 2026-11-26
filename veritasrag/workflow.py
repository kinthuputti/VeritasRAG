"""Upload approval: an admin or uploader submits a document and chooses who may read it; it is NOT searchable until a
DIFFERENT admin approves it (four-eyes rule). The submitter's access choice always wins over any `access:` line inside the file."""
import os, re, threading, time
from .ingest import _front_matter

ACCESS = re.compile(r"[a-z0-9_-]{1,30}")
NAME = re.compile(r"[\w.\- ]{1,100}\.(md|txt)")

class Workflow:
    def __init__(self, engine, docs_dir=None, require_approval=True):
        self.engine, self.docs_dir, self.require_approval = engine, docs_dir, require_approval
        self.items, self._n, self._lock = {}, 0, threading.Lock()

    def submit(self, name, text, access, user, max_bytes=500_000):
        access = "all" if access in ("everyone", "all") else access.lower()
        if not NAME.fullmatch(name) or ".." in name: raise ValueError("name must be like policy.md or notes.txt")
        if not ACCESS.fullmatch(access): raise ValueError("access must be 'everyone' or a role name such as hr")
        if len(text) > max_bytes: raise ValueError("document too large")
        with self._lock:
            self._n += 1
            it = dict(id=self._n, name=name, access=access, text=_front_matter(text)[1], submitted_by=user, status="pending",
                      decided_by=None, reason="", time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            self.items[it["id"]] = it
        if not self.require_approval: self._publish(it, user)
        return self.view(it)

    def _publish(self, it, approver):
        raw = f"---\ntitle: {it['name']}\naccess: {it['access']}\n---\n{it['text']}"
        with self._lock:
            chunks = self.engine.add_document(it["name"], raw)
            it.update(status="approved", decided_by=approver)
            if self.docs_dir:
                with open(os.path.join(self.docs_dir, it["name"]), "w", encoding="utf-8") as fh: fh.write(raw)
        return chunks

    def approve(self, item_id, approver):
        it = self.items.get(item_id)
        if it is None: raise KeyError("no such submission")
        if it["status"] != "pending": raise ValueError(f"already {it['status']}")
        if self.require_approval and approver == it["submitted_by"]: raise PermissionError("a different admin must approve your upload")
        self._publish(it, approver); return self.view(it)

    def reject(self, item_id, user, reason=""):
        it = self.items.get(item_id)
        if it is None: raise KeyError("no such submission")
        if it["status"] != "pending": raise ValueError(f"already {it['status']}")
        it.update(status="rejected", decided_by=user, reason=reason[:300]); return self.view(it)

    def pending(self): return [self.view(i) for i in self.items.values() if i["status"] == "pending"]
    @staticmethod
    def view(it): return {k: v for k, v in it.items() if k != "text"} | dict(preview=it["text"][:200])
