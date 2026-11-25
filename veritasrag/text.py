"""Text helpers shared by the engine and the evaluation (same rules as the official SQuAD scoring)."""
import collections, re, string

def tok(s):
    return re.findall(r"\w+", s.lower())

def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def f1(pred, golds):
    best, p = 0.0, norm(pred).split()
    for g in golds:
        g = norm(g).split()
        n = sum((collections.Counter(p) & collections.Counter(g)).values())
        if n:
            pr, rc = n / len(p), n / len(g)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best

ABSTAIN_WORDS = ("unanswerable", "i don't know", "i do not know", "cannot be answered", "not in the context", "no answer")

def is_abstain(text):
    t = text.strip().lower()
    return t == "" or any(w in t for w in ABSTAIN_WORDS)

def grounded(pred, passages):
    """True if the answer text appears in the passages it was written from."""
    return norm(pred) != "" and norm(pred) in norm(" ".join(passages))

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
