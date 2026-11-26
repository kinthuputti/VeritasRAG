"""Model backends. `RealBackend` uses the same models as the evaluation notebook. `MockBackend` needs no downloads and is used by the tests."""
import numpy as np
from .text import tok, sentences

SYSTEM = ("Answer the question using only the context. Reply with a short answer copied from the context. "
          "If the context does not contain the answer, reply exactly: UNANSWERABLE")

def _extract(question, passages, min_overlap=2):
    """Lightweight fallback: return the context sentence that best matches the question."""
    qs, best, bs = set(tok(question)), None, 0
    for p in passages:
        for s in sentences(p):
            o = len(qs & set(tok(s)))
            if o > bs: best, bs = s, o
    return best if best and bs >= min_overlap else "UNANSWERABLE"

def _best_sentence(rerank, question, passages):
    """Pick the sentence that best answers the question by MEANING (cross-encoder score), not by shared words,
    so a reworded question still finds its answer. Whether to answer at all is decided earlier by the engine's guards."""
    sents = [x for p in passages for x in sentences(p)]
    if not sents: return "UNANSWERABLE"
    scores = rerank(question, sents)
    return sents[max(range(len(sents)), key=lambda i: scores[i])]

class OnnxBackend:
    """Same two models as RealBackend (bge-small-en-v1.5 and ms-marco-MiniLM-L-6-v2) run through ONNX, with no PyTorch.
    Small enough for a 512 MB free host. Answers are extractive (best-matching sentence)."""
    name = "onnx"; stateless = True
    def __init__(self, embedder="BAAI/bge-small-en-v1.5", reranker="Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed import TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.embedder, self.cross = TextEmbedding(embedder), TextCrossEncoder(reranker)
    def fit(self, texts): pass
    def embed(self, texts, query=False):
        f = self.embedder.query_embed if query else self.embedder.embed
        return np.array(list(f(list(texts))), dtype=np.float32)
    def rerank(self, question, passages):
        return [float(x) for x in self.cross.rerank(question, list(passages))]
    def generate(self, question, passages):
        return _best_sentence(self.rerank, question, passages)

class MockBackend:
    name = "mock"
    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tf = TfidfVectorizer().fit(texts)
    def embed(self, texts, query=False):
        return self.tf.transform(texts).toarray()
    def rerank(self, question, passages):
        qs = set(tok(question))
        return [len(qs & set(tok(p))) - 2.5 for p in passages]
    def generate(self, question, passages):
        return _extract(question, passages)

class RealBackend:
    name = "real"; stateless = True
    QPREFIX = "Represent this sentence for searching relevant passages: "
    def __init__(self, embedder="BAAI/bge-small-en-v1.5", reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 generator="Qwen/Qwen2.5-1.5B-Instruct", device=None):
        import torch
        from sentence_transformers import SentenceTransformer, CrossEncoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(embedder, device=self.device)
        self.cross = CrossEncoder(reranker, device=self.device)
        self.generator_name = generator
        self.gtok = self.gmod = None
        if generator and generator != "extractive":
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.gtok = AutoTokenizer.from_pretrained(generator)
            self.gmod = AutoModelForCausalLM.from_pretrained(
                generator, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device)
    def fit(self, texts): pass
    def embed(self, texts, query=False):
        if query: texts = [self.QPREFIX + t for t in texts]
        return self.embedder.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    def rerank(self, question, passages):
        return [float(x) for x in self.cross.predict([(question, p) for p in passages], batch_size=32)]
    def generate(self, question, passages):
        if self.gmod is None: return _best_sentence(self.rerank, question, passages)
        import torch
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context:\n" + "\n\n".join(passages) + f"\n\nQuestion: {question}"}]
        text = self.gtok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.gtok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.gmod.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=self.gtok.eos_token_id)
        return self.gtok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
