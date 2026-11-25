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
        c = self.client("emp-token-0001:amy:employee,adm-token-0001:root:employee|admin")
        doc = {"name": "pets.md", "text": Upload.NEW}
        self.assertEqual(c.post("/documents", json=doc, headers={"X-API-Key": "emp-token-0001"}).status_code, 403)
        for bad in ("../evil.md", "a.exe", "x/y.md"):
            self.assertEqual(c.post("/documents", json={"name": bad, "text": "x"}, headers={"X-API-Key": "adm-token-0001"}).status_code, 400)
        a = {"X-API-Key": "adm-token-0001"}
        self.assertEqual(c.post("/documents", json=doc, headers=a).status_code, 200)
        self.assertTrue(os.path.exists(os.path.join(self.docs, "pets.md")))
        r = c.post("/ask", json={"query": "Are dogs allowed on Fridays?"}, headers={"X-API-Key": "emp-token-0001"}).json()
        self.assertFalse(r["abstained"])
        self.assertIn("pets.md", [d["doc"] for d in c.get("/documents", headers=a).json()])
        self.assertEqual(c.delete("/documents/pets.md", headers=a).status_code, 200)
        self.assertFalse(os.path.exists(os.path.join(self.docs, "pets.md")))
        self.assertEqual(c.delete("/documents/pets.md", headers=a).status_code, 404)

class Questions(unittest.TestCase):
    def test_size_split_and_balance(self):
        self.assertGreaterEqual(len(QS), 80)
        self.assertEqual({q["split"] for q in QS}, {"dev", "test"})
        for t in ("answerable", "unanswerable", "restricted"):
            self.assertTrue(any(q["type"] == t and q["split"] == s for q in QS for s in ("dev", "test")))
        self.assertEqual(len({q["id"] for q in QS}), len(QS))
        self.assertGreaterEqual(len({c.doc for c in load_chunks(DOCS)}), 16)

if __name__ == "__main__": unittest.main()
