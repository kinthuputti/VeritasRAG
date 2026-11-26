"""Keep the search index in step with a source of documents and their permissions.

A connector lists documents as {name: (text, access)}. `Sync.run` adds new ones, updates changed ones (including a change of
permission) and removes ones that disappeared from the source, so access rights always follow the source of truth.
`FolderConnector` is implemented (permissions come from an `acl.json` file next to the documents, e.g. {"salary.md": "hr"}).
SharePoint, Google Drive or Confluence connectors would implement the same one-method interface; they are not built."""
import hashlib, json, os
from .ingest import _front_matter

class FolderConnector:
    def __init__(self, folder): self.folder = folder
    def list(self):
        acl_path = os.path.join(self.folder, "acl.json")
        acl = {}
        if os.path.exists(acl_path):
            with open(acl_path, encoding="utf-8") as fh: acl = json.load(fh)
        out = {}
        for name in sorted(os.listdir(self.folder)):
            if not name.lower().endswith((".md", ".txt")): continue
            with open(os.path.join(self.folder, name), encoding="utf-8") as fh: raw = fh.read()
            meta, body = _front_matter(raw)
            access = acl.get(name) or meta.get("access", "all")     # the ACL file wins over anything written in the file
            out[name] = (body, "all" if access in ("everyone", "all") else access.lower())
        return out

class Sync:
    def __init__(self, engine): self.engine, self.state = engine, {}      # name -> hash of (text, access)
    def run(self, connector):
        src, res = connector.list(), dict(added=[], updated=[], removed=[], unchanged=[])
        for name, (text, access) in src.items():
            h = hashlib.sha256((access + "\0" + text).encode()).hexdigest()
            if self.state.get(name) == h: res["unchanged"].append(name); continue
            self.engine.add_document(name, f"---\ntitle: {name}\naccess: {access}\n---\n{text}")
            res["updated" if name in self.state else "added"].append(name); self.state[name] = h
        for name in [n for n in self.state if n not in src]:
            self.engine.remove_document(name); self.state.pop(name); res["removed"].append(name)
        return res
