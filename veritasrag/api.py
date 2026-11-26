"""FastAPI service: sign-in by API key or identity-provider token, roles, /search, /ask, approval workflow for uploads,
document sync, a tamper-evident audit log, and a small web page.

    VERITASRAG_TOKENS="token-at-least-12-chars:alice:employee,another-token-123:bob:employee|hr"
    uvicorn veritasrag.api:app
"""
import collections, hashlib, json, os, re, time
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .engine import Engine, NO_ANSWER
from .ingest import load_chunks
from .audit import AuditLog
from .workflow import Workflow
from .connectors import Sync

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

def create_app(engine, users, audit_path="audit_log.jsonl", docs_dir=None, limiter=None, verifier=None,
               require_approval=True, source=None):
    app = FastAPI(title="VeritasRAG")
    app.state.engine, app.state.users = engine, users
    limiter = limiter or Limiter(int(os.environ.get("VERITASRAG_RATE", 60)))
    log, flow, sync = AuditLog(audit_path), Workflow(engine, docs_dir, require_approval), Sync(engine)
    app.state.audit, app.state.workflow = log, flow

    def current_user(request: Request, x_api_key: str = Header(default=""), authorization: str = Header(default="")):
        client = request.client.host if request.client else "unknown"
        if limiter.locked(client):
            raise HTTPException(429, "too many failed sign-ins; try again in a few minutes", headers={"Retry-After": str(limiter.lock_seconds)})
        u = None
        if authorization.lower().startswith("bearer ") and verifier is not None:
            try: u = verifier.verify(authorization[7:].strip())
            except ValueError: u = None
        elif x_api_key:
            u = users.get(hash_token(x_api_key))
            if u is not None and u.get("expires") and time.strftime("%Y-%m-%d") > u["expires"]: u = None
        if u is None:
            limiter.fail(client)
            raise HTTPException(401, "missing, invalid or expired credentials (X-API-Key header, or Authorization: Bearer <token>)")
        if not limiter.allow(u["user"]):
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "60"})
        return u

    def audit(u, kind, query, **extra): log.append(u["user"], u["roles"], kind, query, **extra)

    def uploader(u=Depends(current_user)):
        if not ({"uploader", "admin"} & set(u["roles"])): raise HTTPException(403, "this needs the uploader or admin role")
        return u

    def admin(u=Depends(current_user)):
        if "admin" not in u["roles"]: raise HTTPException(403, "this needs the admin role")
        return u

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
    def read_audit(u=Depends(admin)): return log.entries()

    class Doc(BaseModel):
        name: str
        text: str
        access: str = "everyone"          # who may read it: "everyone" or a role name such as "hr"

    class Reason(BaseModel):
        reason: str = ""

    @app.get("/documents")
    def list_documents(u=Depends(admin)): return engine.documents()

    @app.post("/documents")
    def submit_document(d: Doc, u=Depends(uploader)):
        """Submit a document and choose who may read it. With approval on (default) it waits until ANOTHER admin approves."""
        try: item = flow.submit(d.name, d.text, d.access, u["user"])
        except ValueError as e: raise HTTPException(400 if "too large" not in str(e) else 413, str(e))
        audit(u, "submit", d.name, access=item["access"], submission=item["id"], status=item["status"])
        return item

    @app.get("/documents/pending")
    def pending(u=Depends(admin)): return flow.pending()

    @app.post("/documents/pending/{item_id}/approve")
    def approve(item_id: int, u=Depends(admin)):
        try: item = flow.approve(item_id, u["user"])
        except KeyError: raise HTTPException(404, "no such submission")
        except PermissionError as e: raise HTTPException(403, str(e))
        except ValueError as e: raise HTTPException(409, str(e))
        audit(u, "approve", item["name"], submission=item_id, submitted_by=item["submitted_by"], access=item["access"])
        return item

    @app.post("/documents/pending/{item_id}/reject")
    def reject(item_id: int, r: Reason, u=Depends(admin)):
        try: item = flow.reject(item_id, u["user"], r.reason)
        except KeyError: raise HTTPException(404, "no such submission")
        except ValueError as e: raise HTTPException(409, str(e))
        audit(u, "reject", item["name"], submission=item_id, reason=item["reason"])
        return item

    @app.delete("/documents/{name}")
    def delete_document(name: str, u=Depends(admin)):
        n = engine.remove_document(name)
        if n == 0: raise HTTPException(404, "no such document")
        if docs_dir and os.path.exists(os.path.join(docs_dir, os.path.basename(name))): os.remove(os.path.join(docs_dir, os.path.basename(name)))
        audit(u, "delete", name, chunks=n)
        return dict(doc=name, removed_chunks=n)

    @app.post("/sync")
    def run_sync(u=Depends(admin)):
        """Make the index match the configured source folder and its acl.json permissions."""
        if source is None: raise HTTPException(404, "no source configured (set VERITASRAG_SOURCE)")
        res = sync.run(source)
        audit(u, "sync", "", **{k: v for k, v in res.items()})
        return res

    @app.get("/audit/verify")
    def verify_audit(u=Depends(admin)):
        ok, n = log.verify()
        return dict(intact=ok, lines=n) if ok else dict(intact=False, first_bad_line=n)

    web = os.path.join(os.path.dirname(__file__), "web")
    if os.path.isdir(web): app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app

def _default_app():
    """Used by `uvicorn veritasrag.api:app`. VERITASRAG_BACKEND=real (default, PyTorch), onnx (small, no PyTorch) or mock (tests)."""
    from .models import MockBackend, OnnxBackend, RealBackend
    from .auth import JwtVerifier, parse_group_map
    from .connectors import FolderConnector
    docs = os.environ.get("VERITASRAG_DOCS", "docs")
    kind = os.environ.get("VERITASRAG_BACKEND", "real")
    backend = (MockBackend() if kind == "mock" else OnnxBackend() if kind == "onnx"
               else RealBackend(generator=os.environ.get("VERITASRAG_GENERATOR", "extractive")))
    raw = os.environ.get("VERITASRAG_TOKENS") or ("demo-employee-0001:alice:employee,demo-hr-user-0001:bob:employee|hr,"
                                                  "demo-admin-user-1:root:employee|hr|finance|admin,demo-admin-user-2:root2:employee|hr|finance|admin")
    verifier = None
    secret, pub, jwks = (os.environ.get(k) for k in ("VERITASRAG_JWT_SECRET", "VERITASRAG_JWT_PUBLIC_KEY", "VERITASRAG_JWKS_URL"))
    if secret or pub or jwks:
        verifier = JwtVerifier(secret=secret, public_key=pub, jwks_url=jwks, audience=os.environ.get("VERITASRAG_JWT_AUDIENCE"),
                               issuer=os.environ.get("VERITASRAG_JWT_ISSUER"), group_map=parse_group_map(os.environ.get("VERITASRAG_GROUP_MAP")))
    src = os.environ.get("VERITASRAG_SOURCE")
    return create_app(Engine(backend, load_chunks(docs)), load_users(raw), docs_dir=docs, verifier=verifier,
                      require_approval=os.environ.get("VERITASRAG_REQUIRE_APPROVAL", "1") != "0", source=FolderConnector(src) if src else None)

app = _default_app() if os.environ.get("VERITASRAG_AUTOSTART", "1") == "1" else None
