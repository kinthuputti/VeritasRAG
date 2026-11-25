"""FastAPI service: API keys with roles, /search, /ask, an audit log, and a small web page.

    VERITASRAG_TOKENS="token-at-least-12-chars:alice:employee,another-token-123:bob:employee|hr"
    uvicorn veritasrag.api:app
"""
import collections, hashlib, json, os, re, time
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .engine import Engine, NO_ANSWER
from .ingest import load_chunks

class Query(BaseModel):
    query: str
    top_k: int = 5

def hash_token(token):
    return hashlib.sha256(token.encode()).hexdigest()

def load_users(raw):
    """Entries are `token:user:role1|role2[:expiry-date]`. The token may be plain (min 12 chars) or `sha256$<hex>`
    so no plain secret has to sit in the configuration. Only the hash is kept in memory."""
    users = {}
    for part in raw.split(","):
        if not part.strip(): continue
        f = [x.strip() for x in part.split(":")]
        if len(f) not in (3, 4): raise ValueError("each key must look like token:user:role1|role2[:YYYY-MM-DD]")
        token, user, roles = f[:3]
        if token.startswith("sha256$"): key = token[7:].lower()
        else:
            if len(token) < 12: raise ValueError("tokens must be at least 12 characters")
            key = hash_token(token)
        expires = f[3] if len(f) == 4 else None
        if expires: time.strptime(expires, "%Y-%m-%d")      # fail early on a bad date
        users[key] = dict(user=user, roles=[r for r in roles.split("|") if r], expires=expires)
    return users

class Limiter:
    """Sliding-window counters: requests per user per minute, and failed sign-ins per client."""
    def __init__(self, per_minute=60, max_failures=5, lock_seconds=300, clock=time.time):
        self.per_minute, self.max_failures, self.lock_seconds, self.clock = per_minute, max_failures, lock_seconds, clock
        self.hits, self.fails = collections.defaultdict(list), collections.defaultdict(list)
    def locked(self, client):
        now = self.clock(); self.fails[client] = [t for t in self.fails[client] if now - t < self.lock_seconds]
        return len(self.fails[client]) >= self.max_failures
    def fail(self, client): self.fails[client].append(self.clock())
    def allow(self, user):
        now = self.clock(); self.hits[user] = [t for t in self.hits[user] if now - t < 60]
        if len(self.hits[user]) >= self.per_minute: return False
        self.hits[user].append(now); return True

def create_app(engine, users, audit_path="audit_log.jsonl", docs_dir=None, limiter=None):
    app = FastAPI(title="VeritasRAG")
    app.state.engine, app.state.users = engine, users
    limiter = limiter or Limiter(int(os.environ.get("VERITASRAG_RATE", 60)))

    def current_user(request: Request, x_api_key: str = Header(default="")):
        client = request.client.host if request.client else "unknown"
        if limiter.locked(client):
            raise HTTPException(429, "too many failed sign-ins; try again in a few minutes", headers={"Retry-After": str(limiter.lock_seconds)})
        u = users.get(hash_token(x_api_key)) if x_api_key else None
        if u is not None and u.get("expires") and time.strftime("%Y-%m-%d") > u["expires"]: u = None
        if u is None:
            limiter.fail(client)
            raise HTTPException(401, "missing, invalid or expired API key (send it in the X-API-Key header)")
        if not limiter.allow(u["user"]):
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "60"})
        return u

    def audit(u, kind, query, **extra):
        with open(audit_path, "a") as fh:
            fh.write(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), user=u["user"], roles=u["roles"], kind=kind, query=query, **extra)) + "\n")

    @app.get("/health")
    def health(): return {"status": "ok", "chunks": len(engine.chunks)}

    @app.post("/search")
    def search(q: Query, u=Depends(current_user)):
        hits = engine.search(q.query, u["roles"], top_k=q.top_k)
        audit(u, "search", q.query, sources=[h.chunk.doc for h in hits])
        return [dict(doc=h.chunk.doc, title=h.chunk.title, section=h.chunk.section, score=round(h.score, 3), text=h.chunk.text) for h in hits]

    @app.post("/ask")
    def ask(q: Query, u=Depends(current_user)):
        r = engine.ask(q.query, u["roles"])
        audit(u, "ask", q.query, answered=not r["abstained"], reason=r["reason"], sources=[c["doc"] for c in r["citations"]])
        if r["abstained"]: r["message"] = NO_ANSWER
        return r

    @app.get("/audit")
    def read_audit(u=Depends(current_user)):
        if "admin" not in u["roles"]: raise HTTPException(403, "this needs the admin role")
        if not os.path.exists(audit_path): return []
        with open(audit_path) as fh: return [json.loads(l) for l in fh]

    def admin(u=Depends(current_user)):
        if "admin" not in u["roles"]: raise HTTPException(403, "this needs the admin role")
        return u

    class Doc(BaseModel):
        name: str
        text: str

    @app.get("/documents")
    def list_documents(u=Depends(admin)): return engine.documents()

    @app.post("/documents")
    def add_document(d: Doc, u=Depends(admin)):
        """Upload or replace a document (markdown/text with an optional `access:` line). It is searchable immediately."""
        if not re.fullmatch(r"[\w.\- ]{1,100}\.(md|txt)", d.name) or ".." in d.name: raise HTTPException(400, "name must be like policy.md or notes.txt")
        if len(d.text) > 500_000: raise HTTPException(413, "document too large (500 KB limit)")
        n = engine.add_document(d.name, d.text)
        if docs_dir:
            with open(os.path.join(docs_dir, d.name), "w", encoding="utf-8") as fh: fh.write(d.text)
        audit(u, "upload", d.name, chunks=n)
        return dict(doc=d.name, chunks=n)

    @app.delete("/documents/{name}")
    def delete_document(name: str, u=Depends(admin)):
        n = engine.remove_document(name)
        if n == 0: raise HTTPException(404, "no such document")
        if docs_dir and os.path.exists(os.path.join(docs_dir, os.path.basename(name))): os.remove(os.path.join(docs_dir, os.path.basename(name)))
        audit(u, "delete", name, chunks=n)
        return dict(doc=name, removed_chunks=n)

    web = os.path.join(os.path.dirname(__file__), "web")
    if os.path.isdir(web): app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app

def _default_app():
    """Used by `uvicorn veritasrag.api:app`. Set VERITASRAG_BACKEND=mock to run without model downloads."""
    from .models import MockBackend, RealBackend
    docs = os.environ.get("VERITASRAG_DOCS", "docs")
    backend = MockBackend() if os.environ.get("VERITASRAG_BACKEND") == "mock" else RealBackend(generator=os.environ.get("VERITASRAG_GENERATOR", "extractive"))
    raw = os.environ.get("VERITASRAG_TOKENS") or "demo-employee-0001:alice:employee,demo-hr-user-0001:bob:employee|hr,demo-admin-user-1:root:employee|hr|finance|admin"
    return create_app(Engine(backend, load_chunks(docs)), load_users(raw), docs_dir=docs)

app = _default_app() if os.environ.get("VERITASRAG_AUTOSTART", "1") == "1" else None
