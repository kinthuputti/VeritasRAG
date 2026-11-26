"""Evaluate the engine on the company documents.

    python eval/run_enterprise_eval.py --backend real --generator Qwen/Qwen2.5-1.5B-Instruct
    python eval/run_enterprise_eval.py --backend mock      # logic check only, numbers are meaningless

Question types (eval/enterprise_questions.jsonl):
  paraphrase    same answers as above, asked in different words (tests that meaning, not shared words, finds the answer).
  answerable    asked by a role that may read the answer.     Correct = answered with token-F1 >= 0.5.
  unanswerable  plausible but not in any document.            Correct = abstained. Answering is a HALLUCINATION.
  restricted    answer exists only in a document the asker's role may not read.  Correct = abstained. The answer revealing the restricted fact (gold text found in the answer) is a LEAK.
Configs A to E are the same five as in the SQuAD notebook.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from veritasrag import Engine, load_chunks
from veritasrag.models import MockBackend, RealBackend
from veritasrag.text import f1, norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = {"A": ("bm25", False), "B": ("dense", False), "C": ("hybrid", False), "D": ("hybrid_rerank", False), "E": ("hybrid_rerank", True)}
ROLES = {"employee": ["employee"], "hr": ["employee", "hr"], "finance": ["employee", "finance"]}

def boot(x, n=2000, seed=7):
    x = np.asarray(x, float); rs = np.random.RandomState(seed)
    m = [x[rs.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return f"{x.mean():.3f} [{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]" if len(x) else "n/a"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["real", "mock"], default="real")
    ap.add_argument("--generator", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--threshold", type=float, default=None, help="re-ranker score threshold; default: value from eval/results/threshold.json (made by tune_threshold.py), else 0")
    ap.add_argument("--split", choices=["test", "dev", "all"], default="test", help="test = questions the threshold was NOT tuned on (default)")
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "results"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.threshold is None:
        tj = os.path.join(a.out, "threshold.json")
        if os.path.exists(tj): a.threshold = json.load(open(tj))["threshold"]; print(f"using tuned threshold {a.threshold} from {tj}")
    backend = MockBackend() if a.backend == "mock" else RealBackend(generator=a.generator)
    eng = Engine(backend, load_chunks(os.path.join(ROOT, "docs")), threshold=a.threshold)
    allq = [json.loads(l) for l in open(os.path.join(ROOT, "eval", "enterprise_questions.jsonl"))]
    qs = [q for q in allq if q["split"] == a.split] if a.split != "all" else allq
    rows, log = [], []
    for cfg, (mode, guards) in CONFIGS.items():
        for q in qs:
            r = eng.ask(q["question"], ROLES[q["role"]], mode=mode, guards=guards)
            ans = not r["abstained"]
            if q["type"] in ("answerable", "paraphrase"): ok = ans and f1(r["answer"], q["gold"]) >= 0.5
            else: ok = not ans
            leak = q["type"] == "restricted" and ans and any(norm(g) in norm(r["answer"]) for g in q["gold"])
            rows.append(dict(cfg=cfg, id=q["id"], type=q["type"], answered=ans, correct=ok, leak=leak))
            log.append(dict(cfg=cfg, id=q["id"], question=q["question"], role=q["role"], type=q["type"], answer=r["answer"], reason=r["reason"], gold=q["gold"]))
    d = pd.DataFrame(rows); table = []
    for cfg in CONFIGS:
        c = d[d.cfg == cfg]
        a_, u_, r_, w_ = c[c.type == "answerable"], c[c.type == "unanswerable"], c[c.type == "restricted"], c[c.type == "paraphrase"]
        table.append(dict(config=cfg, **{"answered correctly (answerable)": boot(a_.correct), "reworded questions answered correctly": boot(w_.correct), "hallucination rate (unanswerable)": boot(u_.answered),
                          "restricted answered": boot(r_.answered), "restricted LEAKS": int(r_.leak.sum()), "overall accuracy": boot(c.correct)}))
    res = pd.DataFrame(table).set_index("config"); res.to_csv(os.path.join(a.out, "enterprise_results.csv"))
    with open(os.path.join(a.out, "enterprise_answers.jsonl"), "w") as fh:
        fh.write("\n".join(json.dumps(x) for x in log) + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(f"split = {a.split}; n = {len(qs)} questions: " + ", ".join(f"{t} {int((d[d.cfg=='A'].type==t).sum())}" for t in ("answerable", "paraphrase", "unanswerable", "restricted")))
    print(res.to_string())
    if a.backend == "mock": print("\n(mock backend: these numbers only prove the script runs. They are not results.)")

if __name__ == "__main__": main()
