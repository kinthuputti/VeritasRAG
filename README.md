# VeritasRAG

VeritasRAG is a search engine for company documents. You ask a question, it answers from the documents you are allowed to read, shows where the answer came from, and says "no answer found" when it isn't sure. I built it to test one idea: that hybrid search, a cross-encoder re-ranker and an automated evaluation harness can bring down hallucinations, and to measure whether each part really helps.

## What is in it

| Part | What it does | Where |
|---|---|---|
| Ingestion | Reads `.md`, `.txt` and `.pdf` files, splits them by section into chunks, reads each file's `access:` level | `veritasrag/ingest.py` |
| Role-based access control | Each role gets its own index built only from chunks it may read, so restricted text is never scored, cited, or shown to the language model | `veritasrag/engine.py` |
| Hybrid retrieval | BM25 and dense search (`bge-small-en-v1.5`), combined with reciprocal rank fusion | `veritasrag/engine.py` |
| Cross-encoder re-ranking | Re-scores the top 20 candidates (`ms-marco-MiniLM-L-6-v2`) | `veritasrag/engine.py` |
| Answer guards | (1) abstain if the best re-ranker score is below a threshold, (2) abstain if the answer text is not found in the retrieved passages | `veritasrag/engine.py` |
| Citations | Every answer lists the document, section and passage it came from | `/ask` response, web page |
| API and web page | `/search`, `/ask`, `/audit`, admin document upload/delete, hashed API keys with roles, expiry, rate limit and lockout, a small browser page | `veritasrag/api.py`, `veritasrag/web/` |
| Audit log | Every search and question is logged with user, role, sources and whether it was answered | `audit_log.jsonl` |
| Evaluation harness | Compares five configurations with bootstrap confidence intervals, on SQuAD 2.0 and on the company documents | notebook + `eval/` |
| Tests | 36 tests, including a check that restricted text never reaches a user without the role | `tests/` |

The sample documents in `docs/` are made up (a fictional company). Four of the 16 documents are restricted: `salary_bands.md` and `performance_calibration.md` (HR), `finance_close_calendar.md` and `treasury_limits.md` (Finance).

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v               # no model downloads needed
uvicorn veritasrag.api:app                            # open http://127.0.0.1:8000
```

The first start downloads two small models (about 250 MB). Sign in with a demo key below, or set your own with `VERITASRAG_TOKENS="token-12-chars-min:alice:employee,another-token-1:bob:employee|hr"`.

| Demo key | User | Roles |
|---|---|---|
| `demo-employee-0001` | alice | employee |
| `demo-hr-user-0001` | bob | employee, hr |
| `demo-admin-user-1` | root | employee, hr, finance, admin |

Try: ask *"How many days of annual leave do I get?"* as alice (answered, cited), then *"What is the salary range for band L4?"* as alice (no answer found) and as bob (answered from `salary_bands.md`).

By default the answer is the best-matching sentence from the documents. To get short written answers instead, set `VERITASRAG_GENERATOR=Qwen/Qwen2.5-1.5B-Instruct` (a GPU helps). The guard threshold is set with `VERITASRAG_THRESHOLD`.

## Evaluation 1: SQuAD 2.0 (`VeritasRAG_SQuAD_evaluation.ipynb`)

SQuAD 2.0 mixes questions that the text answers with questions it doesn't, so answering one of the second kind counts as a hallucination. No judge model is needed. I used 300 test questions, half of them unanswerable. The guard threshold was tuned on 100 separate dev questions and then left alone. Generator: `Qwen2.5-1.5B-Instruct`, greedy. Intervals are 95% bootstrap over questions.

| Config | Retrieval | Re-rank | Guards | Recall@3 | Answered correctly (answerable) | Hallucination rate (unanswerable) | Overall accuracy |
|---|---|---|---|---|---|---|---|
| A | BM25 | no | no | 0.880 [0.827, 0.933] | 0.693 [0.620, 0.767] | 0.787 [0.720, 0.853] | 0.453 [0.397, 0.510] |
| B | dense | no | no | 0.840 [0.780, 0.893] | 0.660 [0.580, 0.733] | 0.680 [0.607, 0.753] | 0.490 [0.433, 0.550] |
| C | hybrid | no | no | 0.913 [0.867, 0.953] | 0.713 [0.640, 0.787] | 0.753 [0.687, 0.820] | 0.480 [0.427, 0.537] |
| D | hybrid | yes | no | 0.960 [0.927, 0.987] | 0.760 [0.687, 0.827] | 0.807 [0.740, 0.867] | 0.477 [0.423, 0.540] |
| E | hybrid | yes | yes | 0.960 [0.927, 0.987] | 0.600 [0.527, 0.673] | 0.460 [0.380, 0.533] | 0.570 [0.513, 0.623] |

**Findings**
- Hybrid search and re-ranking made retrieval better: recall@3 went from 0.84–0.88 with a single retriever to 0.96 with the re-ranker.
- Better retrieval on its own did **not** cut hallucination. It stayed at 0.68–0.81 from A to D, because the model kept answering questions it should have refused.
- The guards did. The rate dropped from 0.81 (D) to 0.46 (E) and the intervals don't overlap. The price is coverage: correct answers on answerable questions fell from 0.76 to 0.60.
- E shows zero unsupported answers, but that is built in (the grounding guard removes them), so I don't count it as a result.

## Evaluation 2: the company documents (`eval/`)

16 made-up documents (12 open, 2 HR-only, 2 Finance-only) and 89 hand-written questions: 52 answerable (45 asked by employees, 7 by HR/Finance users), 24 unanswerable (plausible, but in no document) and 13 restricted (the answer exists only in a document the asker's role may not read). Each question is marked `dev` (37) or `test` (52). The re-ranker threshold is tuned **on dev only**; the table below is **test only**. The script reports the same metrics as the SQuAD notebook, plus **restricted leaks**: cases where a restricted fact actually showed up in an answer.

```bash
python eval/tune_threshold.py --backend real --generator Qwen/Qwen2.5-1.5B-Instruct   # writes eval/results/threshold.json
python eval/run_enterprise_eval.py --backend real --generator Qwen/Qwen2.5-1.5B-Instruct   # reads it automatically, reports the test split
python eval/benchmark.py --backend real --chunks 5000                                  # speed at ~5000 chunks
```

### Results

I tuned the threshold on the 37 dev questions: **-0.469** (dev accuracy 1.00; 0.946 without the threshold). Test split, 52 questions (31 answerable, 14 unanswerable, 7 restricted), 95% bootstrap intervals, Qwen2.5-1.5B-Instruct generator:

| Config | Answered correctly (answerable) | Hallucination rate (unanswerable) | Restricted answered | Restricted LEAKS | Overall accuracy |
|---|---|---|---|---|---|
| A: BM25 | 0.968 [0.903, 1.000] | 0.000 [0.000, 0.000] | 0.143 [0.000, 0.429] | 0 | 0.962 [0.904, 1.000] |
| B: dense | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.286 [0.000, 0.571] | 0 | 0.962 [0.904, 1.000] |
| C: hybrid | 0.968 [0.903, 1.000] | 0.071 [0.000, 0.214] | 0.286 [0.000, 0.571] | 0 | 0.923 [0.846, 0.981] |
| D: hybrid + re-rank | 0.968 [0.903, 1.000] | 0.000 [0.000, 0.000] | 0.286 [0.000, 0.571] | 0 | 0.942 [0.865, 1.000] |
| E: hybrid + re-rank + guards | 0.968 [0.903, 1.000] | 0.000 [0.000, 0.000] | 0.143 [0.000, 0.429] | 0 | 0.962 [0.904, 1.000] |

- **No restricted fact was revealed in any configuration** (0 leaks). "Restricted answered" counts questions where some answer was returned; those answers came from open documents and did not contain the restricted fact.
- On these questions every configuration is already close to zero hallucination (only C produced one, 1 of 14), so this run can't show the guards helping. The evidence for that is the SQuAD run above (0.81 to 0.46). With only 52 test questions, the intervals are wide.

### Speed at 5,000 chunks (GPU, `eval/benchmark.py`)

The filler text is generated, so this only measures speed, not answer quality.

| Step | Median | 95th percentile |
|---|---|---|
| BM25 search | 8.4 ms | 12.9 ms |
| Dense search | 15.1 ms | 18.3 ms |
| Hybrid search | 28.8 ms | 38.9 ms |
| Hybrid + re-rank | 107.0 ms | 144.5 ms |

Building the first index took 9.0 s; adding one document and re-indexing took 0.3 s; peak memory was 1,566 MB.

## Access control and operations

How a company would run this, and which parts are built here:

| Practice | Status | Where |
|---|---|---|
| Role-restricted search, enforced before ranking | Built, tested (0 leaks) | `veritasrag/engine.py` |
| Sign-in with an identity provider's token (OIDC/JWT: signature, expiry, issuer and audience checked), groups mapped to roles | Built, tested with locally signed tokens | `veritasrag/auth.py` |
| API keys (stored as hashes, with expiry), per-user rate limit, lockout after failed sign-ins | Built, tested | `veritasrag/api.py` |
| Named staff accounts with salted password hashes (demo app) | Built | `streamlit_demo/app.py` |
| Upload by an admin or uploader, who chooses who may read the file; it stays unsearchable until a **different** admin approves (four-eyes rule); the uploader's access choice overrides any `access:` line inside the file | Built, tested | `veritasrag/workflow.py` |
| Tamper-evident audit log (each line holds the hash of the previous one; `GET /audit/verify`) | Built, tested | `veritasrag/audit.py` |
| Keep the index in step with a source of documents and their permissions (add, change, permission change, delete) | Built for a folder with an `acl.json` file, tested; `POST /sync` | `veritasrag/connectors.py` |
| Two-factor login | Provided by the identity provider (Google, Entra ID, Okta); this app only accepts the token | not built here |
| SharePoint, Google Drive or Confluence connectors | Not built; they would implement the same one-method connector interface | n/a |
| Encryption at rest, independent security audit | Not done | n/a |

Configuration is by environment variables: `VERITASRAG_TOKENS` (`token:user:roles[:expiry]`, token may be `sha256$<hex>`), `VERITASRAG_JWT_SECRET` / `VERITASRAG_JWT_PUBLIC_KEY` / `VERITASRAG_JWKS_URL` with `VERITASRAG_JWT_AUDIENCE`, `VERITASRAG_JWT_ISSUER` and `VERITASRAG_GROUP_MAP` (`hr-team=hr;it-admins=admin|uploader`), `VERITASRAG_REQUIRE_APPROVAL` (default on), `VERITASRAG_SOURCE` (folder to sync), `VERITASRAG_RATE` (requests per minute, default 60).

Document endpoints: `POST /documents` (uploader or admin; body `name`, `text`, `access`), `GET /documents/pending`, `POST /documents/pending/{id}/approve` or `/reject` (admin, not the submitter), `GET /documents`, `DELETE /documents/{name}`, `POST /sync`, `GET /audit`, `GET /audit/verify`.

## Limitations
- With guards, the hallucination rate on SQuAD 2.0 (Wikipedia text, 1.5B generator) is still 0.46. The guards cut it from 0.81 but don't remove it.
- Accuracy comes from an automatic score (token-F1 ≥ 0.5), not from human judges.
- The sign-in parts are tested with locally generated tokens, not against a live identity provider. The demo's Alice/Bob/Carol drop-down is not real authentication.
- Nobody outside the project has audited the access control or the API for security, and files are not encrypted at rest.
- Documents uploaded to the live demo are kept in memory and are lost when it restarts.
