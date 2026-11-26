"""VeritasRAG live demo (Streamlit). Same engine as the API: hybrid search, cross-encoder re-ranking, answer guards and
role-restricted search over made-up company documents. Answers are extractive (the best-matching sentence)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import streamlit as st
from veritasrag import Engine, load_chunks

st.set_page_config(page_title="VeritasRAG", page_icon="🔎", layout="centered")

KEYS = {   # public demo keys for made-up documents. These two roles cannot upload or delete anything.
    "Alice (employee)": ["employee"],
    "Bob (employee + HR)": ["employee", "hr"],
    "Carol (employee + Finance)": ["employee", "finance"],
}

@st.cache_resource(show_spinner="Loading the search models (first start takes a minute)...")
def load_engine():
    from veritasrag.models import MockBackend, OnnxBackend
    backend = MockBackend() if os.environ.get("VERITASRAG_BACKEND") == "mock" else OnnxBackend()
    return Engine(backend, load_chunks(os.path.join(ROOT, "docs")), threshold=-0.469)

st.title("🔎 VeritasRAG")
st.write("Ask a question about a **made-up company's** documents. You only get answers from documents your role may read, "
         "every answer shows its source, and if nothing supports an answer it says so instead of guessing.")
who = st.selectbox("Sign in as", list(KEYS))
roles = KEYS[who]
st.caption("Try: *How many days of annual leave do I get?* → then *What is the salary range for band L4?* as Alice (no answer) and as Bob (answered from the HR-only file).")
q = st.text_input("Your question", placeholder="How many days of annual leave does a full-time employee get?")
if q:
    r = load_engine().ask(q, roles)
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
           "The live demo writes answers by extracting the best-matching sentence; the evaluation numbers in the README (SQuAD 0.81 → 0.46 hallucination, 0 restricted leaks) "
           "were measured with a 1.5B generator. [Code and results on GitHub](https://github.com/kinthuputti/VeritasRAG)")
