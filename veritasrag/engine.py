"""The search engine.

Access control happens BEFORE anything is scored: each role set gets its own index built only from chunks that role may read,
so restricted text can never be ranked, quoted, or handed to the language model for the wrong user.
"""
import collections, os, re
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
from .text import tok, is_abstain, grounded, sentences

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
_DOCWORD = r"(document|doc|file|pdf|policy|handbook|guide|manual|report)"
ASKS_TITLE = re.compile(rf"\b(title|called|named|name)\b.*\b{_DOCWORD}\b|\b{_DOCWORD}\b.*\b(title|called|named)\b", re.I)
ASKS_ABOUT = re.compile(rf"\b(summar(y|ise|ize)|overview)\b|\b(what|which)\b.*\b{_DOCWORD}\b.*\babout\b|\babout\b.*\b{_DOCWORD}\b", re.I)
THIS_DOC = re.compile(r"\b(this|the attached|the uploaded|the selected)\s+" + _DOCWORD + r"\b", re.I)
_GENERIC = set("what is the a an of this that name title called named which document doc file pdf about summary summarise summarize overview does it".split())
NO_ANSWER = "No answer found in the documents you can access."

def rrf(lists, k=60):
    """Reciprocal rank fusion: documents ranked high in several lists rise to the top."""
    sc = collections.defaultdict(float)
    for lst in lists:
        for rank, i in enumerate(lst): sc[i] += 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])]

@dataclass
class Hit:
    chunk: object
    score: float

class _Index:
    def __init__(self, chunks, backend, cache):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tok(c.text) for c in chunks]) if chunks else None
        self.emb = None
        if chunks:
            missing = [c.text for c in chunks if c.text not in cache]
            if missing:
                for t, v in zip(missing, backend.embed(missing)): cache[t] = v
            self.emb = np.stack([cache[c.text] for c in chunks])

class Engine:
    def __init__(self, backend, chunks, threshold=None, candidates=20, context_k=3):
        self.backend, self.chunks = backend, chunks
        self.threshold = float(os.environ.get("VERITASRAG_THRESHOLD", 0.0)) if threshold is None else threshold
        self.candidates, self.context_k = candidates, context_k
        backend.fit([c.text for c in chunks])
        self._indexes, self._cache = {}, {}

    # ---- adding and removing documents while running (only the new text is embedded) ----
    def _changed(self):
        self._indexes = {}                                   # role indexes are rebuilt lazily on the next query
        if not getattr(self.backend, "stateless", False):    # the mock embedder is fitted on the corpus, so refit it
            self.backend.fit([c.text for c in self.chunks]); self._cache = {}

    def documents(self):
        out = {}
        for c in self.chunks: out.setdefault(c.doc, dict(doc=c.doc, title=c.title, access=c.access, chunks=0))["chunks"] += 1
        return sorted(out.values(), key=lambda d: d["doc"])

    def add_document(self, name, raw):
        from .ingest import chunks_from_text
        self.remove_document(name, _quiet=True)
        new = chunks_from_text(name, raw, max((c.id for c in self.chunks), default=-1) + 1)
        self.chunks = self.chunks + new; self._changed()
        return len(new)

    def remove_document(self, name, _quiet=False):
        keep = [c for c in self.chunks if c.doc != name]
        n = len(self.chunks) - len(keep)
        if n or not _quiet: self.chunks = keep; self._changed()
        return n

    def _index(self, roles, doc=None):
        allowed = frozenset({"all"} | {r.lower() for r in roles})
        if (allowed, doc) not in self._indexes:      # a single-document index only ever holds chunks this role may read
            self._indexes[(allowed, doc)] = _Index([c for c in self.chunks if c.access in allowed and (doc is None or c.doc == doc)], self.backend, self._cache)
        return self._indexes[(allowed, doc)]

    def readable_documents(self, roles):
        allowed = {"all"} | {r.lower() for r in roles}
        return [d for d in self.documents() if d["access"] in allowed]

    def search(self, query, roles=("employee",), mode="hybrid_rerank", top_k=5, doc=None):
        ix = self._index(roles, doc)
        if not ix.chunks: return []
        n = len(ix.chunks)
        def bm(): return list(np.argsort(-ix.bm25.get_scores(tok(query)))[:min(50, n)])
        def de():
            q = self.backend.embed([query], query=True)[0]
            return list(np.argsort(-(ix.emb @ q))[:min(50, n)])
        if mode == "bm25": order = bm()
        elif mode == "dense": order = de()
        else: order = rrf([bm(), de()])
        if mode == "hybrid_rerank":
            cand = order[:self.candidates]
            scores = self.backend.rerank(query, [ix.chunks[i].text for i in cand])
            ranked = sorted(zip(cand, scores), key=lambda x: -x[1])
            return [Hit(ix.chunks[i], float(s)) for i, s in ranked[:top_k]]
        return [Hit(ix.chunks[i], float(len(order) - r)) for r, i in enumerate(order[:top_k])]

    def _about_document(self, question, roles, doc, mode):
        """Questions about the file itself (its title, what it covers) are answered from the real title and opening text."""
        ix = self._index(roles, doc)
        if doc is None and THIS_DOC.search(question) : return "needs_document", None
        if doc is None:
            ask_words = set(tok(question)) - _GENERIC
            best, best_n = None, 0
            for c in ix.chunks:                                  # compare the question with each readable document's title and file name
                if best is not None and c.doc == best.doc: continue
                n = len(ask_words & (set(tok(c.title)) | set(tok(c.doc.rsplit(".", 1)[0]))))
                if n > best_n: best, best_n = c, n
            if best is None:                                     # no title words in the question: fall back to the best-matching passage
                hits = self.search(question, roles, mode, top_k=1)
                if not hits: return "no_documents", None
                best = hits[0].chunk
            first = next(c for c in ix.chunks if c.doc == best.doc)
        else:
            if not ix.chunks: return "no_documents", None
            first = ix.chunks[0]
        ans = f'The document is titled "{first.title}" ({first.doc}).'
        if ASKS_ABOUT.search(question) and not ASKS_TITLE.search(question):
            ans = f'"{first.title}" begins: ' + " ".join(sentences(first.text)[:2])
        return "ok", (ans, first)

    def ask(self, question, roles=("employee",), mode="hybrid_rerank", guards=True, doc=None):
        if ASKS_TITLE.search(question) or ASKS_ABOUT.search(question):
            status, res = self._about_document(question, roles, doc, mode)
            if status != "ok": return dict(answer=None, abstained=True, reason=status, mode=mode, guards=guards, citations=[])
            ans, c = res
            return dict(answer=ans, abstained=False, reason="answered", mode=mode, guards=guards,
                        citations=[dict(doc=c.doc, title=c.title, section=c.section, score=1.0, text=c.text)])
        hits = self.search(question, roles, mode, top_k=self.context_k, doc=doc)
        cites = [dict(doc=h.chunk.doc, title=h.chunk.title, section=h.chunk.section, score=round(h.score, 3), text=h.chunk.text) for h in hits]
        def out(answer, reason):
            return dict(answer=answer, abstained=answer is None, reason=reason, mode=mode, guards=guards,
                        citations=cites if answer is not None else [])
        if not hits: return out(None, "no_documents")
        passages = [h.chunk.text for h in hits]
        if guards and mode == "hybrid_rerank" and hits[0].score < self.threshold:
            return out(None, "low_confidence")                       # guard 1: best passage scores too low
        pred = self.backend.generate(question, passages)
        if is_abstain(pred): return out(None, "model_abstained")
        if guards and not grounded(pred, passages):
            return out(None, "not_grounded")                         # guard 2: answer text is not in the passages
        return out(pred, "answered")
