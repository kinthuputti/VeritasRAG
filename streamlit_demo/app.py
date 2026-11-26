"""VeritasRAG live demo (Streamlit). Same engine as the API: hybrid search, cross-encoder re-ranking, answer guards and
role-restricted search over made-up company documents. Answers are extractive (the best-matching sentence, chosen by meaning).

Visitors can only ask questions. Only the admin (password in Streamlit secrets as ADMIN_PASSWORD) can upload or delete
documents and decide who may read each one."""
import hmac, os, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import streamlit as st
from veritasrag import Engine, load_chunks
from veritasrag.ingest import _front_matter

st.set_page_config(page_title="VeritasRAG", page_icon="🔎", layout="centered")

PERSONAS = {   # demo sign-ins for the made-up company: there are no real accounts here
    "Alice (employee)": ["employee"],
    "Bob (employee + HR)": ["employee", "hr"],
    "Carol (employee + Finance)": ["employee", "finance"],
}
ADMIN = "Admin (password)"
MAX_BYTES, MAX_UPLOADS = 200_000, 20

@st.cache_resource(show_spinner="Loading the search models (first start takes a minute)...")
def load_engine():
    from veritasrag.models import MockBackend, OnnxBackend
    backend = MockBackend() if os.environ.get("VERITASRAG_BACKEND") == "mock" else OnnxBackend()
    eng = Engine(backend, load_chunks(os.path.join(ROOT, "docs")), threshold=-0.469)
    return dict(engine=eng, lock=threading.Lock(), uploaded={}, log=[])   # shared by everyone; uploads last until the app restarts

def admin_password():
    try: return st.secrets.get("ADMIN_PASSWORD", "")
    except Exception: return os.environ.get("ADMIN_PASSWORD", "")

st.title("🔎 VeritasRAG")
st.write("Ask a question about a **made-up company's** documents. You only get answers from documents your role may read, "
         "every answer shows its source, and if nothing supports an answer it says so instead of guessing.")
who = st.selectbox("Sign in as", list(PERSONAS) + [ADMIN])
is_admin = False
if who == ADMIN:
    pw = st.text_input("Admin password", type="password")
    real = admin_password()
    is_admin = bool(real) and hmac.compare_digest(pw.encode(), real.encode())
    if pw and not is_admin: st.error("Wrong password." if real else "Admin sign-in is not set up on this deployment.")
roles = ["employee", "hr", "finance"] if is_admin else PERSONAS.get(who, ["employee"])
state = load_engine(); eng = state["engine"]

if is_admin:
    st.success("Signed in as admin: you can add and remove documents.")
    with st.expander("Documents (admin)", expanded=True):
        up = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])
        level = st.selectbox("Who may read it?", ["everyone", "hr", "finance"],
                             help="Only people with this role will ever see answers or sources from the file.")
        if up is not None and st.button("Add to search"):
            raw = up.getvalue().decode("utf-8", errors="ignore")
            if len(raw) > MAX_BYTES: st.error("File too large (200 KB limit).")
            elif len(state["uploaded"]) >= MAX_UPLOADS and up.name not in state["uploaded"]: st.error(f"Limit of {MAX_UPLOADS} uploads reached.")
            else:
                acc = "all" if level == "everyone" else level
                with state["lock"]:
                    n = eng.add_document(up.name, f"---\ntitle: {up.name}\naccess: {acc}\n---\n{_front_matter(raw)[1]}")
                    state["uploaded"][up.name] = acc
                    state["log"].append(f"{time.strftime('%H:%M')} uploaded {up.name} (readable by {level})")
                st.success(f"Added {up.name}: {n} passages, readable by {level}.")
        for name, acc in list(state["uploaded"].items()):
            c1, c2 = st.columns([4, 1]); c1.write(f"**{name}**: {'everyone' if acc == 'all' else acc}")
            if c2.button("Delete", key="del-" + name):
                with state["lock"]:
                    eng.remove_document(name); state["uploaded"].pop(name, None); state["log"].append(f"{time.strftime('%H:%M')} deleted {name}")
                st.rerun()
        if state["log"]: st.caption("Activity: " + " · ".join(state["log"][-5:]))
        st.caption("Uploads are kept in memory and are lost when the app restarts.")
else:
    st.caption("Try: *How many days of annual leave do I get?* Then *What is the salary range for band L4?* as Alice (no answer) and as Bob (answered from the HR-only file).")

q = st.text_input("Your question", placeholder="How many days of annual leave does a full-time employee get?")
if q:
    r = eng.ask(q, roles)
    if r["abstained"]:
        st.warning("No answer found in the documents you can access.")
        st.caption(f"Reason: {r['reason']}")
    else:
        st.success(r["answer"])
        for c in r["citations"][:3]:
            with st.expander(f"Source: {c['title']} › {c['section']}  (score {c['score']})"):
                st.write(c["text"])
st.divider()
st.caption("Pipeline: BM25 + dense search (bge-small) → rank fusion → cross-encoder re-rank (MiniLM) → guards. "
           "Demo personas are a drop-down because this is a made-up company with no real accounts. "
           "The live demo writes answers by picking the best-matching sentence; the evaluation numbers in the README (SQuAD 0.81 → 0.46 hallucination, 0 restricted leaks) "
           "were measured with a 1.5B generator. [Code and results on GitHub](https://github.com/kinthuputti/VeritasRAG)")
