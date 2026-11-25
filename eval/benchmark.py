"""Speed and scale check. Builds a corpus of N chunks (the real documents plus generated filler policies), then times
indexing and queries.

    python eval/benchmark.py --backend real --chunks 5000
Writes eval/results/benchmark.json. Filler text is generated, so this measures speed, NOT answer quality.
"""
import argparse, json, os, random, resource, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from veritasrag import Engine, load_chunks
from veritasrag.ingest import Chunk
from veritasrag.models import MockBackend, RealBackend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPICS = "leave travel expense security vendor facilities onboarding payroll training safety audit remote equipment compliance".split()
UNITS = "days weeks dollars hours percent months".split()

def filler(n, seed=1):
    rs = random.Random(seed); out = []
    for i in range(n):
        t = rs.choice(TOPICS)
        text = " ".join(f"The {t} rule {rs.randint(1, 999)} says employees in team {rs.randint(1, 80)} need {rs.randint(1, 60)} {rs.choice(UNITS)} "
                        f"for {rs.choice(TOPICS)} requests and must notify {rs.choice(['managers', 'finance', 'the helpdesk', 'security'])}." for _ in range(4))
        out.append(Chunk(0, f"filler_{i // 5}.md", f"Filler policy {i // 5}", f"Section {i % 5}", text, "all"))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["real", "mock"], default="real")
    ap.add_argument("--generator", default="extractive")
    ap.add_argument("--chunks", type=int, default=5000)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "results"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    backend = MockBackend() if a.backend == "mock" else RealBackend(generator=a.generator)
    real = load_chunks(os.path.join(ROOT, "docs"))
    chunks = real + filler(max(a.chunks - len(real), 0))
    for i, c in enumerate(chunks): c.id = i
    eng = Engine(backend, chunks)
    t = time.time(); eng._index(["employee"]); build = time.time() - t
    qs = [json.loads(l)["question"] for l in open(os.path.join(ROOT, "eval", "enterprise_questions.jsonl"))]
    qs = (qs * 3)[:a.queries]
    res = dict(backend=a.backend, chunks=len(chunks), index_build_seconds=round(build, 2))
    for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        lat = []
        for q in qs:
            t = time.time(); eng.search(q, ["employee"], mode, top_k=5); lat.append((time.time() - t) * 1000)
        res[mode + "_ms"] = dict(p50=round(float(np.percentile(lat, 50)), 1), p95=round(float(np.percentile(lat, 95)), 1))
    t = time.time(); eng.add_document("bench_new.md", "---\ntitle: New\naccess: all\n---\n## A\nThe pet policy allows one dog per floor."); eng._index(["employee"])
    res["add_one_document_and_reindex_seconds"] = round(time.time() - t, 2)
    res["peak_memory_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    json.dump(res, open(os.path.join(a.out, "benchmark.json"), "w"), indent=2); print(json.dumps(res, indent=2))

if __name__ == "__main__": main()
