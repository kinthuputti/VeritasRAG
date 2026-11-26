"""Tamper-evident audit log: every line stores the hash of the line before it, so editing, deleting or re-ordering any
earlier line breaks the chain and `verify()` reports where."""
import hashlib, json, os, threading, time

GENESIS = "0" * 64

def _digest(entry):
    core = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()

class AuditLog:
    def __init__(self, path):
        self.path, self._lock = path, threading.Lock()

    def entries(self):
        if not os.path.exists(self.path): return []
        with open(self.path, encoding="utf-8") as fh: return [json.loads(l) for l in fh if l.strip()]

    def append(self, user, roles, kind, query="", **extra):
        with self._lock:
            rows = self.entries()
            entry = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), user=user, roles=list(roles), kind=kind, query=query, **extra,
                         prev=rows[-1]["hash"] if rows else GENESIS)
            entry["hash"] = _digest(entry)
            with open(self.path, "a", encoding="utf-8") as fh: fh.write(json.dumps(entry) + "\n")
            return entry

    def verify(self):
        """Returns (True, number_of_lines) if the chain is intact, else (False, first_bad_line_number)."""
        prev, rows = GENESIS, self.entries()
        for i, e in enumerate(rows, 1):
            if e.get("prev") != prev or e.get("hash") != _digest(e): return False, i
            prev = e["hash"]
        return True, len(rows)
