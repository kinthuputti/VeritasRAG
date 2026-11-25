"""Read documents from a folder and cut them into chunks that carry their source and access level."""
import os, re
from dataclasses import dataclass

@dataclass
class Chunk:
    id: int
    doc: str        # file name
    title: str
    section: str
    text: str
    access: str     # "all" or a role name such as "hr" or "finance"

def _front_matter(raw):
    meta, body = {}, raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1); meta[k.strip()] = v.strip()
            body = parts[2]
    return meta, body.strip()

def _read(path):
    if path.endswith(".pdf"):
        from pypdf import PdfReader          # optional: pip install pypdf
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    with open(path, encoding="utf-8") as fh:
        return fh.read()

def chunks_from_text(name, raw, start_id=0, max_words=120):
    """Cut one document's text into chunks. Access comes from the `access:` line at the top (default: all)."""
    meta, body = _front_matter(raw)
    title, access = meta.get("title", name), meta.get("access", "all").lower()
    chunks = []
    for sec in re.split(r"\n(?=## )", body):
        lines = sec.strip().splitlines()
        if not lines: continue
        heading = lines[0].lstrip("# ").strip() if lines[0].startswith("#") else ""
        text = " ".join(l.strip() for l in (lines[1:] if heading else lines) if l.strip())
        words = text.split()
        for start in range(0, max(len(words), 1), max_words):
            piece = " ".join(words[start:start + max_words]).strip()
            if piece:
                chunks.append(Chunk(start_id + len(chunks), name, title, heading, piece, access))
    return chunks

def load_chunks(folder, max_words=120):
    """Every file in the folder becomes chunks."""
    chunks = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".md", ".txt", ".pdf")): continue
        chunks += chunks_from_text(name, _read(os.path.join(folder, name)), len(chunks), max_words)
    return chunks
