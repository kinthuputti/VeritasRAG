"""Choose the re-ranker confidence threshold on the DEV questions only, so the TEST questions stay untouched.

    python eval/tune_threshold.py --backend real --generator Qwen/Qwen2.5-1.5B-Instruct
Writes eval/results/threshold.json, which run_enterprise_eval.py reads automatically.
Score for a threshold = share of dev questions handled correctly (answerable: right answer; unanswerable/restricted: abstained).
Ties go to the lower hallucination rate, then to the lower threshold.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from veritasrag import Engine, load_chunks
from veritasrag.models import MockBackend, RealBackend
from veritasrag.text import f1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLES = {"employee": ["employee"], "hr": ["employee", "hr"], "finance": ["employee", "finance"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["real", "mock"], default="real")
    ap.add_argument("--generator", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "results"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    backend = MockBackend() if a.backend == "mock" else RealBackend(generator=a.generator)
    eng = Engine(backend, load_chunks(os.path.join(ROOT, "docs")), threshold=float("-inf"))
    dev = [q for q in (json.loads(l) for l in open(os.path.join(ROOT, "eval", "enterprise_questions.jsonl"))) if q["split"] == "dev"]
    rows = []
    for q in dev:                                            # run each dev question once with the threshold guard switched off
        roles = ROLES[q["role"]]
        top = eng.search(q["question"], roles, "hybrid_rerank", top_k=1)
        r = eng.ask(q["question"], roles, "hybrid_rerank", guards=True)
        rows.append((q, top[0].score if top else float("-inf"), r))
    def evaluate(t):
        ok, hall = [], []
        for q, s, r in rows:
            answered = s >= t and not r["abstained"]
            if q["type"] == "answerable": ok.append(answered and f1(r["answer"], q["gold"]) >= 0.5)
            else:
                ok.append(not answered)
                if q["type"] == "unanswerable": hall.append(answered)
        return float(np.mean(ok)), float(np.mean(hall)) if hall else 0.0
    scores = sorted({round(s, 3) for _, s, _ in rows if s > float("-inf")})
    cands = [min(scores) - 1] + [(x + y) / 2 for x, y in zip(scores, scores[1:])] + [max(scores) + 1]
    best = max(((evaluate(t), t) for t in cands), key=lambda z: (z[0][0], -z[0][1], -z[1]))
    (acc, hall), t = best
    base = evaluate(float("-inf"))
    out = dict(threshold=round(t, 3), dev_questions=len(rows), dev_accuracy=round(acc, 3), dev_hallucination=round(hall, 3),
               dev_accuracy_without_threshold=round(base[0], 3), dev_hallucination_without_threshold=round(base[1], 3), backend=a.backend)
    json.dump(out, open(os.path.join(a.out, "threshold.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
