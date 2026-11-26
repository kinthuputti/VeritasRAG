"""Run:  python -m unittest discover -s tests -v   (uses stand-in models, no downloads)"""
import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["VERITASRAG_AUTOSTART"] = "0"
from veritasrag import Engine, load_chunks
from veritasrag.engine import rrf, NO_ANSWER
from veritasrag.models import MockBackend
from veritasrag.text import norm, f1, grounded
from veritasrag.api import create_app, load_users

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
QS = [json.loads(l) for l in open(os.path.join(os.path.dirname(DOCS), "eval", "enterprise_questions.jsonl"))]

def make(**kw): return Engine(MockBackend(), load_chunks(DOCS), **kw)

class Ingest(unittest.TestCase):
    def test_access_labels(self):
        acc = {c.doc: c.access for c in load_chunks(DOCS)}
        self.assertEqual(acc["salary_bands.md"], "hr"); self.assertEqual(acc["finance_close_calendar.md"], "finance")
        self.assertEqual(acc["leave_policy.md"], "all")
    def test_gold_answers_really_in_documents(self):
        by_role = {"employee": "all", "hr": "hr", "finance": "finance"}
        for q in QS:
            if q["type"] == "unanswerable": continue
            text = norm(" ".join(c.text for c in load_chunks(DOCS)))
            for g in q["gold"]: self.assertIn(norm(g), text, f"{q['id']}: gold '{g}' not in documents")
    def test_restricted_golds_only_in_restricted_docs(self):
        open_text = norm(" ".join(c.text for c in load_chunks(DOCS) if c.access == "all"))
        for q in QS:
            if q["type"] == "restricted":
                for g in q["gold"]: self.assertNotIn(norm(g), open_text, f"{q['id']} answer leaks into an open document")

class AccessControl(unittest.TestCase):
    def test_employee_never_sees_restricted_chunks(self):
        e = make()
        for q in QS:
            for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
                for h in e.search(q["question"], ["employee"], mode, top_k=10):
                    self.assertEqual(h.chunk.access, "all", f"{q['id']} {mode} returned {h.chunk.doc}")
    def test_hr_can_read_hr_documents(self):
        r = make().ask("What is the salary range for band L4?", ["employee", "hr"], guards=False)
        self.assertIn("salary_bands.md", [c["doc"] for c in r["citations"]])
    def test_employee_asking_restricted_question_gets_no_leak(self):
        e = make()
        for q in [x for x in QS if x["type"] == "restricted"]:
            r = e.ask(q["question"], ["employee"], guards=False)
            if not r["abstained"]:
                self.assertLess(f1(r["answer"], q["gold"]), 0.5, f"{q['id']} leaked: {r['answer']}")
            self.assertNotIn("salary_bands.md", [c["doc"] for c in r["citations"]])

class Search(unittest.TestCase):
    def test_rrf_prefers_items_high_in_both_lists(self):
        self.assertEqual(rrf([[1, 2, 3], [3, 1, 2]])[0], 1)
    def test_hybrid_finds_leave_document(self):
        h = make().search("How many days of annual leave per year?", ["employee"], "hybrid_rerank", top_k=3)
        self.assertEqual(h[0].chunk.doc, "leave_policy.md")
    def test_answer_has_citations(self):
        r = make().ask("How many days of annual leave does a full-time employee get per year?", ["employee"])
        self.assertFalse(r["abstained"]); self.assertTrue(r["citations"])

class Guards(unittest.TestCase):
    def test_threshold_guard_abstains(self):
        r = make(threshold=999).ask("How many days of annual leave per year?", ["employee"])
        self.assertTrue(r["abstained"]); self.assertEqual(r["reason"], "low_confidence")
    def test_threshold_guard_off_when_guards_off(self):
        r = make(threshold=999).ask("How many days of annual leave per year?", ["employee"], guards=False)
        self.assertEqual(r["reason"], "answered")
    def test_grounding_guard_catches_invented_answer(self):
        class Liar(MockBackend):
            def generate(self, q, p): return "about 5000 days"
        e = Engine(Liar(), load_chunks(DOCS))
        self.assertEqual(e.ask("How many days of leave?", ["employee"], guards=True)["reason"], "not_grounded")
        self.assertFalse(e.ask("How many days of leave?", ["employee"], guards=False)["abstained"])
    def test_grounded_helper(self):
        self.assertTrue(grounded("24 days", ["You receive 24 days of leave."])); self.assertFalse(grounded("30 days", ["24 days"]))

class Api(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.tmp = tempfile.mkdtemp()
        users = load_users("emp-token-0001:alice:employee,hr-token-00001:bob:employee|hr,adm-token-0001:root:employee|hr|finance|admin")
        self.c = TestClient(create_app(make(), users, audit_path=os.path.join(self.tmp, "audit.jsonl")))
    def test_requires_key(self):
        self.assertEqual(self.c.post("/ask", json={"query": "x"}).status_code, 401)
        self.assertEqual(self.c.post("/ask", json={"query": "x"}, headers={"X-API-Key": "wrong-wrong-wrong"}).status_code, 401)
    def test_ask_and_audit(self):
        h = {"X-API-Key": "emp-token-0001"}
        r = self.c.post("/ask", json={"query": "How many days of annual leave does a full-time employee get per year?"}, headers=h).json()
        self.assertFalse(r["abstained"])
        self.assertEqual(self.c.get("/audit", headers=h).status_code, 403)
        log = self.c.get("/audit", headers={"X-API-Key": "adm-token-0001"}).json()
        self.assertEqual(log[0]["user"], "alice"); self.assertEqual(log[0]["kind"], "ask")
    def test_roles_over_api(self):
        q = {"query": "What is the salary range for band L4?"}
        emp = self.c.post("/search", json=q, headers={"X-API-Key": "emp-token-0001"}).json()
        hr = self.c.post("/search", json=q, headers={"X-API-Key": "hr-token-00001"}).json()
        self.assertNotIn("salary_bands.md", [x["doc"] for x in emp]); self.assertIn("salary_bands.md", [x["doc"] for x in hr])
    def test_short_token_rejected(self):
        with self.assertRaises(ValueError): load_users("short:a:employee")

class Upload(unittest.TestCase):
    NEW = "---\ntitle: Pet Policy\naccess: all\n---\n## Pets\nOne dog per floor is allowed on Fridays."
    def test_add_search_remove(self):
        e = make(); n0 = len(e.chunks)
        self.assertEqual(e.add_document("pets.md", self.NEW), 1)
        top = e.search("Are dogs allowed on Fridays?", ["employee"], "hybrid", 1)[0]
        self.assertEqual(top.chunk.doc, "pets.md")
        e.add_document("pets.md", self.NEW)                       # replacing does not duplicate
        self.assertEqual(len(e.chunks), n0 + 1)
        self.assertEqual(e.remove_document("pets.md"), 1); self.assertEqual(len(e.chunks), n0)
        self.assertNotIn("pets.md", [h.chunk.doc for h in e.search("dogs Fridays", ["employee"], "hybrid", 5)])
    def test_uploaded_restricted_doc_stays_restricted(self):
        e = make(); e.add_document("secret.md", "---\ntitle: S\naccess: hr\n---\n## X\nThe zebra budget is 77 dollars.")
        self.assertNotIn("secret.md", [h.chunk.doc for h in e.search("zebra budget", ["employee"], "hybrid", 10)])
        self.assertIn("secret.md", [h.chunk.doc for h in e.search("zebra budget", ["employee", "hr"], "hybrid", 10)])
    def test_embeddings_cached_for_real_style_backend(self):
        class Counting(MockBackend):
            stateless = True
            calls = 0
            def embed(self, texts, query=False):
                if not query: Counting.calls += len(texts)
                return super().embed(texts, query)
        e = Engine(Counting(), load_chunks(DOCS)); e.search("leave", ["employee"]); first = Counting.calls
        e.add_document("pets.md", self.NEW); e.search("leave", ["employee"])
        self.assertEqual(Counting.calls - first, 1)                # only the new chunk was embedded

class Security(unittest.TestCase):
    def client(self, raw, **kw):
        from fastapi.testclient import TestClient
        from veritasrag.api import Limiter
        self.tmp = tempfile.mkdtemp(); self.docs = tempfile.mkdtemp()
        return TestClient(create_app(make(), load_users(raw), audit_path=os.path.join(self.tmp, "a.jsonl"), docs_dir=self.docs, limiter=Limiter(**kw)))
    def test_hashed_token(self):
        from veritasrag.api import hash_token
        c = self.client(f"sha256${hash_token('a-long-secret-key')}:amy:employee")
        self.assertEqual(c.post("/search", json={"query": "leave"}, headers={"X-API-Key": "a-long-secret-key"}).status_code, 200)
        self.assertEqual(c.post("/search", json={"query": "leave"}, headers={"X-API-Key": hash_token("a-long-secret-key")}).status_code, 401)
    def test_expired_key(self):
        c = self.client("old-token-00001:amy:employee:2020-01-01,new-token-00001:bo:employee:2099-01-01")
        self.assertEqual(c.post("/search", json={"query": "x"}, headers={"X-API-Key": "old-token-00001"}).status_code, 401)
        self.assertEqual(c.post("/search", json={"query": "x"}, headers={"X-API-Key": "new-token-00001"}).status_code, 200)
    def test_rate_limit(self):
        c = self.client("emp-token-0001:amy:employee", per_minute=3); h = {"X-API-Key": "emp-token-0001"}
        codes = [c.post("/search", json={"query": "x"}, headers=h).status_code for _ in range(5)]
        self.assertEqual(codes, [200, 200, 200, 429, 429])
        self.assertIn("retry-after", c.post("/search", json={"query": "x"}, headers=h).headers)
    def test_lockout_after_failures(self):
        c = self.client("emp-token-0001:amy:employee", max_failures=3)
        for _ in range(3): self.assertEqual(c.post("/search", json={"query": "x"}, headers={"X-API-Key": "bad-bad-bad-bad"}).status_code, 401)
        self.assertEqual(c.post("/search", json={"query": "x"}, headers={"X-API-Key": "emp-token-0001"}).status_code, 429)  # even the right key is locked out
    def test_upload_needs_admin_and_validates(self):
        c = self.client("emp-token-0001:amy:employee,adm-token-0001:root:employee|admin,adm-token-0002:two:employee|admin", )
        doc = {"name": "pets.md", "text": Upload.NEW}
        self.assertEqual(c.post("/documents", json=doc, headers={"X-API-Key": "emp-token-0001"}).status_code, 403)
        for bad in ("../evil.md", "a.exe", "x/y.md"):
            self.assertEqual(c.post("/documents", json={"name": bad, "text": "x"}, headers={"X-API-Key": "adm-token-0001"}).status_code, 400)
        self.assertEqual(c.post("/documents", json={**doc, "access": "HR TEAM!"}, headers={"X-API-Key": "adm-token-0001"}).status_code, 400)
        a1, a2 = {"X-API-Key": "adm-token-0001"}, {"X-API-Key": "adm-token-0002"}
        sub = c.post("/documents", json=doc, headers=a1).json(); self.assertEqual(sub["status"], "pending")
        ask = lambda: c.post("/ask", json={"query": "Are dogs allowed on Fridays?"}, headers={"X-API-Key": "emp-token-0001"}).json()
        self.assertTrue(ask()["abstained"] or all(x["doc"] != "pets.md" for x in ask()["citations"]))   # not searchable while pending
        self.assertEqual(c.post(f"/documents/pending/{sub['id']}/approve", headers=a1).status_code, 403)  # cannot approve your own upload
        self.assertEqual(c.post(f"/documents/pending/{sub['id']}/approve", headers=a2).status_code, 200)
        self.assertTrue(os.path.exists(os.path.join(self.docs, "pets.md")))
        self.assertIn("pets.md", [x["doc"] for x in ask()["citations"]])
        self.assertEqual(c.post(f"/documents/pending/{sub['id']}/approve", headers=a2).status_code, 409)
        self.assertIn("pets.md", [d["doc"] for d in c.get("/documents", headers=a1).json()])
        self.assertEqual(c.delete("/documents/pets.md", headers=a1).status_code, 200)
        self.assertFalse(os.path.exists(os.path.join(self.docs, "pets.md")))
        self.assertEqual(c.delete("/documents/pets.md", headers=a1).status_code, 404)
    def test_uploader_role_and_reject(self):
        c = self.client("up-token-00001:ulla:employee|uploader,adm-token-0001:root:employee|admin")
        up, ad = {"X-API-Key": "up-token-00001"}, {"X-API-Key": "adm-token-0001"}
        sub = c.post("/documents", json={"name": "a.md", "text": "x"}, headers=up).json()
        self.assertEqual(c.get("/documents/pending", headers=up).status_code, 403)       # uploaders cannot see or decide the queue
        self.assertEqual(c.post(f"/documents/pending/{sub['id']}/reject", json={"reason": "not accurate"}, headers=ad).json()["status"], "rejected")
        self.assertEqual(c.get("/documents/pending", headers=ad).json(), [])
    def test_submitters_access_choice_beats_text_inside_file(self):
        c = self.client("adm-token-0001:root:employee|admin,adm-token-0002:two:employee|admin,emp-token-0001:amy:employee")
        a1, a2 = {"X-API-Key": "adm-token-0001"}, {"X-API-Key": "adm-token-0002"}
        sneaky = "---\naccess: all\n---\n## Bonus\nThe quokka bonus is 99 dollars."
        sub = c.post("/documents", json={"name": "q.md", "text": sneaky, "access": "hr"}, headers=a1).json()
        c.post(f"/documents/pending/{sub['id']}/approve", headers=a2)
        r = c.post("/search", json={"query": "quokka bonus"}, headers={"X-API-Key": "emp-token-0001"}).json()
        self.assertNotIn("q.md", [x["doc"] for x in r])
    def test_audit_chain_detects_tampering(self):
        from veritasrag.audit import AuditLog
        path = os.path.join(tempfile.mkdtemp(), "a.jsonl"); lg = AuditLog(path)
        for i in range(4): lg.append("amy", ["employee"], "ask", f"q{i}")
        self.assertEqual(lg.verify(), (True, 4))
        rows = open(path).read().splitlines(); rows[1] = rows[1].replace("q1", "q9")
        open(path, "w").write("\n".join(rows) + "\n"); self.assertEqual(lg.verify(), (False, 2))
        open(path, "w").write("\n".join([rows[0], rows[2], rows[3]]) + "\n"); self.assertFalse(lg.verify()[0])   # deleting a line
    def test_audit_verify_endpoint_and_actor(self):
        c = self.client("adm-token-0001:root:employee|admin")
        c.post("/search", json={"query": "leave"}, headers={"X-API-Key": "adm-token-0001"})
        self.assertEqual(c.get("/audit/verify", headers={"X-API-Key": "adm-token-0001"}).json(), dict(intact=True, lines=1))

class Identity(unittest.TestCase):
    def test_password_hash(self):
        from veritasrag.auth import hash_password, check_password
        h = hash_password("correct horse"); self.assertTrue(check_password("correct horse", h)); self.assertFalse(check_password("wrong", h))
        self.assertNotEqual(h, hash_password("correct horse"))                 # salted
    def _client(self, verifier):
        from fastapi.testclient import TestClient
        d = tempfile.mkdtemp()
        return TestClient(create_app(make(), {}, audit_path=os.path.join(d, "a.jsonl"), verifier=verifier))
    def test_hs256_token_groups_become_roles(self):
        import jwt, time as t
        from veritasrag.auth import JwtVerifier, parse_group_map
        v = JwtVerifier(secret="s" * 32, audience="veritasrag", group_map=parse_group_map("hr-team=hr;it=admin|uploader"))
        c = self._client(v); mk = lambda **k: jwt.encode(dict(email="pat@corp.example", aud="veritasrag", exp=t.time() + 600, **k), "s" * 32, algorithm="HS256")
        q = {"query": "What is the salary range for band L4?"}
        emp = c.post("/search", json=q, headers={"Authorization": "Bearer " + mk(groups=[])}).json()
        hr = c.post("/search", json=q, headers={"Authorization": "Bearer " + mk(groups=["hr-team"])}).json()
        self.assertNotIn("salary_bands.md", [x["doc"] for x in emp]); self.assertIn("salary_bands.md", [x["doc"] for x in hr])
        self.assertEqual(c.get("/documents", headers={"Authorization": "Bearer " + mk(groups=["hr-team"])}).status_code, 403)
        self.assertEqual(c.get("/documents", headers={"Authorization": "Bearer " + mk(groups=["it"])}).status_code, 200)
    def test_bad_tokens_rejected(self):
        import jwt, time as t
        from veritasrag.auth import JwtVerifier
        v = JwtVerifier(secret="s" * 32, audience="veritasrag"); c = self._client(v)
        good = dict(email="a@b.c", aud="veritasrag", exp=t.time() + 600)
        bad = [jwt.encode({**good, "exp": t.time() - 60}, "s" * 32, algorithm="HS256"),          # expired
               jwt.encode(good, "x" * 32, algorithm="HS256"),                                    # wrong key
               jwt.encode({**good, "aud": "other"}, "s" * 32, algorithm="HS256"),                # wrong audience
               jwt.encode({k: x for k, x in good.items() if k != "exp"}, "s" * 32, algorithm="HS256"),  # no expiry
               "not.a.token", jwt.encode(good, None, algorithm="none")]                          # unsigned
        for tok_ in bad:      # a fresh client each time, so the failed-sign-in lockout does not interfere
            self.assertEqual(self._client(v).post("/search", json={"query": "x"}, headers={"Authorization": "Bearer " + tok_}).status_code, 401, tok_[:20])
        c2 = self._client(v)
        for tok_ in bad[:5]: c2.post("/search", json={"query": "x"}, headers={"Authorization": "Bearer " + tok_})
        good_tok = jwt.encode(good, "s" * 32, algorithm="HS256")
        self.assertEqual(c2.post("/search", json={"query": "x"}, headers={"Authorization": "Bearer " + good_tok}).status_code, 429)   # locked out
    def test_rs256_token(self):
        import jwt, time as t
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization as ser
        from veritasrag.auth import JwtVerifier
        k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pub = k.public_key().public_bytes(ser.Encoding.PEM, ser.PublicFormat.SubjectPublicKeyInfo).decode()
        c = self._client(JwtVerifier(public_key=pub, issuer="https://idp.example"))
        tok_ = jwt.encode(dict(email="a@b.c", iss="https://idp.example", exp=t.time() + 600), k, algorithm="RS256")
        self.assertEqual(c.post("/search", json={"query": "leave"}, headers={"Authorization": "Bearer " + tok_}).status_code, 200)
        self.assertEqual(c.post("/search", json={"query": "leave"}, headers={"Authorization": "Bearer " + jwt.encode(dict(email="a", iss="evil", exp=t.time() + 600), k, algorithm="RS256")}).status_code, 401)

class SourceSync(unittest.TestCase):
    def test_sync_follows_source_and_permissions(self):
        from veritasrag.connectors import FolderConnector, Sync
        d = tempfile.mkdtemp(); w = lambda n, t: open(os.path.join(d, n), "w").write(t)
        w("pets.md", "## Pets\nOne dog per floor is allowed on Fridays."); w("acl.json", "{}")
        e = make(); sy = Sync(e); conn = FolderConnector(d)
        self.assertEqual(sy.run(conn)["added"], ["pets.md"])
        vis = lambda roles: "pets.md" in [h.chunk.doc for h in e.search("dogs Fridays", roles, "hybrid", 5)]
        self.assertTrue(vis(["employee"]))
        self.assertEqual(sy.run(conn)["unchanged"], ["pets.md"])
        w("acl.json", '{"pets.md": "hr"}')                       # permission changed at the source
        self.assertEqual(sy.run(conn)["updated"], ["pets.md"]); self.assertFalse(vis(["employee"])); self.assertTrue(vis(["employee", "hr"]))
        os.remove(os.path.join(d, "pets.md"))                    # document removed at the source
        self.assertEqual(sy.run(conn)["removed"], ["pets.md"]); self.assertFalse(vis(["employee", "hr"]))
        self.assertGreaterEqual(len({c.doc for c in e.chunks}), 16)   # the original documents are untouched
    def test_sync_endpoint_admin_only(self):
        from fastapi.testclient import TestClient
        from veritasrag.connectors import FolderConnector
        d = tempfile.mkdtemp(); open(os.path.join(d, "x.md"), "w").write("## X\nThe quokka bonus is 99 dollars.")
        c = TestClient(create_app(make(), load_users("emp-token-0001:amy:employee,adm-token-0001:root:employee|admin"),
                                  audit_path=os.path.join(d, "..", "au.jsonl"), source=FolderConnector(d)))
        self.assertEqual(c.post("/sync", headers={"X-API-Key": "emp-token-0001"}).status_code, 403)
        self.assertEqual(c.post("/sync", headers={"X-API-Key": "adm-token-0001"}).json()["added"], ["x.md"])

class Questions(unittest.TestCase):
    def test_size_split_and_balance(self):
        self.assertGreaterEqual(len(QS), 80)
        self.assertEqual({q["split"] for q in QS}, {"dev", "test"})
        for t in ("answerable", "unanswerable", "restricted"):
            self.assertTrue(any(q["type"] == t and q["split"] == s for q in QS for s in ("dev", "test")))
        self.assertEqual(len({q["id"] for q in QS}), len(QS))
        self.assertGreaterEqual(len({c.doc for c in load_chunks(DOCS)}), 16)

if __name__ == "__main__": unittest.main()
