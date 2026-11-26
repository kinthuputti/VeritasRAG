"""Turn an uploaded file into plain text. .txt/.md are read as they are; .docx keeps its headings as sections;
.pdf uses the text layer (a scanned PDF with no text layer cannot be read without OCR, which is not built)."""
import io

MAX_FILE_BYTES = 5_000_000
KINDS = ("txt", "md", "docx", "pdf")

def extract_text(name, data):
    if len(data) > MAX_FILE_BYTES: raise ValueError("file too large (5 MB limit)")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext in ("txt", "md"): return data.decode("utf-8", errors="ignore")
    if ext == "docx":
        from docx import Document
        try: doc = Document(io.BytesIO(data))
        except Exception: raise ValueError("could not read this Word file")
        out = []
        for p in doc.paragraphs:
            t = p.text.strip()
            if not t: continue
            out.append(("## " + t) if (p.style.name or "").lower().startswith(("heading", "title")) else t)
        for table in doc.tables:                       # tables become one line per row
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells: out.append(" | ".join(cells))
        text = "\n".join(out)
    elif ext == "pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted: raise ValueError("this PDF is password-protected")
            text = "\n\n".join((pg.extract_text() or "") for pg in reader.pages[:200])
        except ValueError: raise
        except Exception: raise ValueError("could not read this PDF")
    else:
        raise ValueError("supported files: .txt, .md, .docx, .pdf")
    if len(text.split()) < 5: raise ValueError("no readable text found (a scanned PDF or image needs OCR, which is not supported)")
    return text
