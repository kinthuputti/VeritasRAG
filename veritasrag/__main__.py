"""Command line:  python -m veritasrag "How many days of annual leave?" --role employee"""
import argparse, json
from .engine import Engine
from .ingest import load_chunks
from .models import MockBackend, RealBackend

p = argparse.ArgumentParser()
p.add_argument("question"); p.add_argument("--docs", default="docs"); p.add_argument("--role", action="append", default=None)
p.add_argument("--backend", choices=["real", "mock"], default="real"); p.add_argument("--generator", default="extractive")
a = p.parse_args()
backend = MockBackend() if a.backend == "mock" else RealBackend(generator=a.generator)
r = Engine(backend, load_chunks(a.docs)).ask(a.question, a.role or ["employee"])
print(json.dumps(r, indent=2))
