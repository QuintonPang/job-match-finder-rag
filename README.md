# Job Match Finder — A RAG-Based Resume-to-Job Matching System

A free, self-hosted Retrieval-Augmented Generation (RAG) system that matches a candidate's resume against real, live job postings — using semantic search rather than keyword matching, with LLM-generated explanations for why each match makes sense.

**[Demo video](https://youtu.be/Bah2bBFhyag)** — upload a resume, see ranked matches, get an AI-generated explanation for the top result.

---

## What it does

1. **Ingests** real, live job postings from the RemoteOK public API
2. **Cleans** noisy job-posting text (stripping marketing fluff like "rockstar," "ninja," "fast-paced dynamic environment") and extracts structured fields using a small, free, locally-run LLM
3. **Embeds** job postings into a vector space using `BAAI/bge-small-en-v1.5`, stored in a self-hosted Qdrant vector database
4. **Parses** an uploaded PDF resume into plain text
5. **Retrieves** the most semantically similar jobs to the resume (vector search)
6. **Reranks** that shortlist using a cross-encoder for higher precision
7. **Explains** the top matches on demand, using an LLM grounded in the actual retrieved resume + job data — not hallucinated

Everything runs on a 100% free stack: no paid APIs, self-hosted vector database, open-source models from Hugging Face.

---

## Architecture

```
RemoteOK API
     │
     ▼
Raw job postings (JSON)
     │
     ▼
LLM cleaning (Phi-3-mini) ──► role_summary, seniority_level
     │
     ▼
Embedding (bge-small-en-v1.5) + RemoteOK's own "tags" field
     │
     ▼
Qdrant vector database  ◄──── Resume PDF → extracted text
     │                              │
     ▼                              │
Stage 1: Vector search (top 20) ◄───┘
     │
     ▼
Stage 2: Cross-encoder reranking (top 5)
     │
     ▼
Stage 3 (on-demand): LLM-generated fit explanation + deterministic skill-gap matching
     │
     ▼
Streamlit UI
```

---

## Tech stack

| Component | Choice |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector database | Qdrant (self-hosted, Docker) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation (cleaning + explanations) | `microsoft/Phi-3-mini-4k-instruct` |
| Resume parsing | `pdfplumber` |
| Job data source | RemoteOK public API |
| UI | Streamlit |

---

## Real engineering challenges (and what I learned)

This project's value isn't just "it works" — it's the debugging and design decisions along the way:

- **Small-model structured-output reliability.** Getting Phi-3-mini to consistently return clean JSON or categorized lists (required vs. nice-to-have skills) proved unreliable across hundreds of real, messy job postings — repeated attempts at prompt engineering (few-shot examples, output priming, stopping criteria, repetition penalties) each fixed one failure mode while revealing another. **Resolution:** simplified to single-field extraction tasks and, for skills specifically, switched to using RemoteOK's own structured `tags` field instead of forcing an unreliable LLM extraction — recognizing when better-suited existing data beats more prompting.
- **GPU memory constraints.** Hit `CUDA out of memory` errors on a free Colab T4 GPU at float32 precision; resolved by switching to float16, understanding the real tradeoff (precision vs. memory) rather than guessing.
- **The "tag soup" problem.** Vector search alone occasionally ranked irrelevant postings (e.g. a Help Desk role) above genuinely relevant ones, because postings with very long, generic tag lists diluted the embedding signal. **Resolution:** added a cross-encoder reranking stage, which correctly demoted these cases by reading the query and job together rather than comparing pre-computed vectors.
- **Local vs. cloud compute split.** Worked around free-tier GPU quota limits by decoupling retrieval (fast, runs locally on CPU) from generation (slow, optionally offloaded to Colab), saving intermediate results to disk so no work is lost between environments.

## Known limitations

- `seniority_level` extraction undercounts junior/senior roles when not explicit in the description body (a title-keyword fallback fix was identified but not re-applied to the full dataset)
- Postings with unusually long, generic tags can still mildly affect ranking even after reranking
- LLM-based skills extraction (`cleaned.skills`) was abandoned in favor of RemoteOK's `tags` field after measuring a 96% empty-result rate — documented here as a deliberate, evidence-based scope reduction, not an oversight

## Resolved during testing

- **Skill-gap detection initially missed real gaps.** The first version asked the LLM to compare a job's tags against the resume and report missing skills in prose — testing surfaced a case where it said "No major gaps found" despite the resume never mentioning Golang, a tag clearly listed on the job. **Fix:** replaced the LLM call for this specific check with deterministic string matching (does each non-generic tag appear in the resume text?), reserving the LLM only for the fit explanation, which genuinely requires judgment rather than exact lookup. This mirrors the earlier `tags`-vs-LLM-extraction decision: use plain code for precise lookups, and the model only for tasks that actually need interpretation.

## Possible future work

- HR-mode reverse search (same retrieval engine, queried from the employer side: "which candidates fit this role")
- Hybrid search (combining BM25 keyword search with vector search)
- Live deployment (would require moving Qdrant to a hosted free tier, e.g. Qdrant Cloud, since self-hosted Docker isn't easily reachable from most free hosting platforms)

---

## Running it locally

### Prerequisites
- Python 3.10+
- Docker Desktop (for Qdrant)

### Setup

```bash
pip install -r requirements.txt
docker run -p 6333:6333 qdrant/qdrant
```

### Build the job index (run once, or whenever you want fresh job data)

```bash
python src/ingestion/fetch_remoteok.py
# clean_postings.py is run via Colab/notebook due to local hardware constraints -- see src/ingestion/clean_postings.py
python src/retrieval/build_embeddings.py
```

### Launch the app

```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` and upload a resume PDF.

---

## Project structure

```
job-rag-system/
├── data/
│   ├── raw/              # raw scraped job postings
│   └── processed/        # cleaned postings, embeddings inputs
├── src/
│   ├── ingestion/        # RemoteOK fetching + LLM-based cleaning
│   ├── parsing/          # PDF resume → plain text
│   ├── retrieval/         # embeddings, Qdrant, search, reranking
│   └── generation/       # LLM fit explanations + skill gaps
├── app/
│   └── streamlit_app.py  # web UI
└── requirements.txt
```