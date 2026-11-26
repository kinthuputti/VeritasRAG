"""VeritasRAG live demo (Streamlit): role-restricted search over made-up company documents.

Visitors can ask questions as one of three demo personas. Staff accounts (set in Streamlit secrets, passwords stored as salted
hashes) can upload documents and choose who may read them; a DIFFERENT admin must approve before a document becomes searchable.
Every sign-in, question, upload and decision goes into a tamper-evident audit log."""
import os, sys, tempfile, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import streamlit as st
from veritasrag import Engine, load_chunks
from veritasrag.audit import AuditLog
from veritasrag.auth import check_password
from veritasrag.extract import extract_text
from veritasrag.workflow import Workflow

st.set_page_config(page_title="VeritasRAG", page_icon="🔎", layout="centered")
PERSONAS = {"Alice (employee)": ["employee"], "Bob (employee + HR)": ["employee", "hr"], "Carol (employee + Finance)": ["employee", "finance"]}
STAFF = "Staff sign-in (username + password)"
MAX_FAILS, LOCK_SECONDS = 5, 300

@st.cache_resource(show_spinner="Loading the search models (first start takes a minute)...")
def load_state():
    from veritasrag.models import MockBackend, OnnxBackend
    backend = MockBackend() if os.environ.get("VERITASRAG_BACKEND") == "mock" else OnnxBackend()
    eng = Engine(backend, load_chunks(os.path.join(ROOT, "docs")), threshold=-0.469)
    return dict(engine=eng, flow=Workflow(eng, None, require_approval=True), audit=AuditLog(os.path.join(tempfile.gettempdir(), "veritasrag_audit.jsonl")),
                fails={}, lock=threading.Lock(), docs={})

def accounts():
    try: return {k: dict(v) for k, v in st.secrets.get("accounts", {}).items()}
    except Exception: return {}

S = load_state(); eng, flow, audit = S["engine"], S["flow"], S["audit"]

st.title("🔎 VeritasRAG")
st.write("Ask a question about a **made-up company's** documents. You only get answers from documents your role may read, "
         "every answer shows its source, and if nothing supports an answer it says so instead of guessing.")
who = st.selectbox("Sign in as", list(PERSONAS) + [STAFF])
user, roles = who, PERSONAS.get(who, ["employee"])
if who == STAFF:
    uname = st.text_input("Username").strip(); pw = st.text_input("Password", type="password")
    st.caption(f"{len(accounts())} staff account(s) configured on this deployment.")
    user, roles = None, ["employee"]
    if uname and pw:
        acc, now = accounts().get(uname), time.time()
        recent = [t for t in S["fails"].get(uname, []) if now - t < LOCK_SECONDS]
        attempt = (uname, pw)
        if len(recent) >= MAX_FAILS: st.error("Too many failed attempts. Try again in a few minutes.")
        elif acc and check_password(pw, acc.get("password", "").strip()):
            user, roles = uname, ["employee"] + [r for r in acc.get("roles", "").split("|") if r and r != "employee"]
        else:
            if st.session_state.get("last_attempt") != attempt:      # count each new attempt once, not every page refresh
                S["fails"][uname] = recent + [now]; audit.append(uname, [], "login_failed")
            st.error("Wrong username or password." if accounts() else "Staff sign-in is not set up on this deployment.")
        st.session_state["last_attempt"] = attempt
can_submit = bool({"uploader", "admin"} & set(roles)) and user is not None and who == STAFF
is_admin = "admin" in roles and user is not None and who == STAFF
if user and who == STAFF: st.success(f"Signed in as **{user}** (roles: {', '.join(roles)})")

if can_submit:
    with st.expander("Add a document", expanded=True):
        up = st.file_uploader("Upload a Word, PDF, .txt or .md file", type=["docx", "pdf", "txt", "md"])
        level = st.selectbox("Who may read it?", ["everyone", "hr", "finance"], help="Only people with this role will ever see answers or sources from it.")
        if up is not None and st.button("Submit for approval"):
            try:
                it = flow.submit(up.name, extract_text(up.name, up.getvalue()), level, user, max_bytes=500_000)
                audit.append(user, roles, "submit", up.name, access=it["access"], submission=it["id"])
                st.success(f"Submitted. It becomes searchable after a different admin approves it (submission #{it['id']}).")
            except ValueError as e: st.error(str(e))
if is_admin:
    with st.expander("Approvals and documents (admin)", expanded=True):
        for it in flow.pending():
            st.write(f"**#{it['id']} {it['name']}**: readable by {it['access']} · submitted by {it['submitted_by']}")
            st.caption(it["preview"])
            c1, c2, _ = st.columns([1, 1, 3])
            if c1.button("Approve", key=f"a{it['id']}"):
                try:
                    r = flow.approve(it["id"], user); audit.append(user, roles, "approve", r["name"], submission=r["id"], submitted_by=r["submitted_by"], access=r["access"]); st.rerun()
                except PermissionError as e: st.error(str(e))
            if c2.button("Reject", key=f"r{it['id']}"):
                flow.reject(it["id"], user); audit.append(user, roles, "reject", it["name"], submission=it["id"]); st.rerun()
        if not flow.pending(): st.caption("Nothing waiting for approval.")
        published = [i for i in flow.items.values() if i["status"] == "approved"]
        for it in published:
            c1, c2 = st.columns([4, 1]); c1.write(f"✅ **{it['name']}**: {it['access']} · approved by {it['decided_by']}")
            if c2.button("Delete", key=f"d{it['id']}"):
                eng.remove_document(it["name"]); it["status"] = "deleted"; audit.append(user, roles, "delete", it["name"]); st.rerun()
        ok, n = audit.verify()
        st.caption(("Audit log intact" if ok else f"⚠️ Audit log was modified (first bad line: {n})") + (f" · {n} entries" if ok else ""))
        with st.expander("Recent audit entries"):
            st.dataframe([{k: e.get(k) for k in ("time", "user", "kind", "query", "status")} for e in audit.entries()[-15:][::-1]])
        st.caption("Uploads are kept in memory and are lost when the app restarts.")
elif who != STAFF:
    st.caption("Try: *How many days of annual leave do I get?* Then *What is the salary range for band L4?* as Alice (no answer) and as Bob (answered from the HR-only file).")

q = st.text_input("Your question", placeholder="How many days of annual leave does a full-time employee get?")
if q:
    r = eng.ask(q, roles)
    audit.append(user or who, roles, "ask", q, answered=not r["abstained"], reason=r["reason"], sources=[c["doc"] for c in r["citations"]])
    if r["abstained"]:
        st.warning("No answer found in the documents you can access."); st.caption(f"Reason: {r['reason']}")
    else:
        st.success(r["answer"])
        for c in r["citations"][:3]:
            with st.expander(f"Source: {c['title']} › {c['section']}  (score {c['score']})"): st.write(c["text"])
st.divider()
st.caption("Pipeline: BM25 + dense search (bge-small) → rank fusion → cross-encoder re-rank (MiniLM) → guards. "
           "Demo personas are a drop-down because this is a made-up company; staff accounts have real passwords (salted hashes). "
           "A production deployment would use single sign-on with two-factor login from the company's identity provider; the API in this repo already accepts those tokens. "
           "The live demo writes answers by picking the best-matching sentence; the README numbers (SQuAD 0.81 → 0.46 hallucination, 0 restricted leaks) used a 1.5B generator. "
           "[Code and results on GitHub](https://github.com/kinthuputti/VeritasRAG)")
