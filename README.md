# HR Resume Evaluation Agent

An autonomous agent that evaluates candidate resumes against job descriptions using LLM-based scoring, semantic search, and natural language query filtering. Built for recruiters who need to shortlist candidates quickly without manually reviewing every resume.

---

## Overview

The agent ingests a batch of PDF resumes, evaluates each one against a given job description using an LLM, ranks candidates by a weighted score, and lets recruiters query results in plain English (e.g., "Top 3 candidates with Python and ML").

It operates as a state machine with four stages: `INIT → JOB_SETUP → EVALUATION → QUERY`.

---

## Features

- **LLM-based resume scoring** across three dimensions: skills, project quality, and resume clarity
- **Weighted final score** (configurable — default: skills 50%, projects 30%, clarity 20%)
- **Semantic resume search** using FAISS and sentence-transformer embeddings
- **Natural language query interface** powered by Gemini 2.5 Flash
- **Job description profiling** to extract required skills, experience, education, and tools
- **Batch mode** to evaluate the same resume pool against multiple job openings
- **Graceful error handling** — LLM failures fall back to default scores; invalid resumes are skipped

---

## Project Structure

```
hr-resume-evaluation-agent/
├── agent/
│   ├── agent_loop.py       # Core state machine and orchestration logic
│   ├── memory.py           # Data structures: Resume, EvaluationScores, CandidateEvaluation
│   ├── state.py            # AgentState enum and StateContext
│   └── init.py             # Module exports
├── jd_profiling/
│   ├── jd_profiler.py      # Parses job descriptions into structured profiles via Gemini
│   ├── query_interpreter.py # Translates recruiter queries into filter objects via Gemini
│   ├── filter_engine.py    # Applies parsed filters to ranked candidate list
│   ├── prompts.py          # Prompt templates for JD profiling and query interpretation
│   └── schemas.py          # Default schemas for JD and query intent structures
├── src/retrieval/
│   ├── resume_parser.py    # PDF text extraction using pdfplumber
│   ├── embeddings.py       # Sentence-transformer embeddings (all-MiniLM-L6-v2)
│   ├── faiss_index.py      # FAISS vector index with metadata mapping
│   └── search.py           # End-to-end ingest and query pipeline
└── test_query_mode.py      # Mock data test for query interpretation and filtering
```

---

## How It Works

### 1. Resume Ingestion (INIT)
PDFs are parsed with `pdfplumber`, cleaned, and embedded using `sentence-transformers/all-MiniLM-L6-v2`. Vectors are stored in a FAISS flat L2 index alongside metadata.

### 2. Job Setup (JOB_SETUP)
The job description is stored in `JobMemory`. The title is extracted from the first line, or defaults to "Untitled Position".

### 3. Evaluation (EVALUATION)
Each resume is evaluated by an LLM (via `scoring/llm_scoring.py`) against the job description. Three sub-scores are returned:

| Score | Default Weight |
|---|---|
| Skill match | 50% |
| Project quality | 30% |
| Resume clarity | 20% |

The final score is a weighted average on a 0–10 scale. Candidates are then ranked highest to lowest.

### 4. Query (QUERY)
Recruiters query the results in plain English. Gemini interprets the query into a structured filter object (intent, top-K, include/exclude skills, min score, experience level), which `filter_engine.py` applies. A skill-match boost (+0.5 per matched skill) promotes more relevant candidates within the filtered set.

---

## Installation

```bash
git clone https://github.com/aakankshakota12/hr-resume-evaluation-agent.git
cd hr-resume-evaluation-agent
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Usage

### Single job evaluation

```python
from agent import Agent

agent = Agent()
result = agent.run(
    job_description="Software Engineer\nRequired: Python, REST APIs, SQL...",
    query="Top 3 candidates with Python"
)
print(result)
```

### Batch mode (multiple jobs, same resume pool)

```python
jobs = [
    {"id": "job_1", "title": "Backend Engineer", "description": "..."},
    {"id": "job_2", "title": "ML Engineer", "description": "..."},
]

result = agent.run(batch_jobs=jobs)
print(result["results"])
```

### Semantic resume search only

```python
from src.retrieval.search import ingest_resumes, query_resumes

ingest_resumes("path/to/resume/folder")
matches = query_resumes("machine learning engineer with Python", top_k=5)

for match in matches:
    print(match["resume"]["id"], match["score"])
```

### Test query filtering

```bash
python test_query_mode.py
```

---

## Query Examples

The query engine supports natural language. A few examples:

| Query | Behavior |
|---|---|
| `"Top 3 candidates with Python"` | Returns top 3 with Python skill |
| `"Best ML candidate"` | Returns single highest-scoring ML candidate |
| `"Top candidates with Python and FastAPI"` | Filters for both skills, returns top 10 |
| `"Top 2 with Python, exclude Java"` | Filters in Python, filters out Java |

---

## Scoring Weights

Weights are loaded from `scoring/rubric.py`. The defaults are:

```python
WEIGHTS = {
    "skill": 0.5,
    "project": 0.3,
    "clarity": 0.2
}
```

Modify this file to adjust scoring priorities per role type.

---

## Dependencies

- `google-generativeai` — Gemini 2.5 Flash for JD profiling and query interpretation
- `sentence-transformers` — Resume and query embeddings (`all-MiniLM-L6-v2`)
- `faiss-cpu` — Vector similarity search
- `pdfplumber` — PDF text extraction
- `python-dotenv` — Environment variable management

---

## Notes

- Resumes must be in PDF format and placed in a designated folder before ingestion.
- The `scoring/` and `tools/` modules (LLM scoring, resume source loader, email delivery) are referenced by the agent but not included in this repository snapshot.
- If the LLM evaluation fails for a candidate, default scores of 5.0 are used rather than skipping the candidate entirely.
- The FAISS index can be persisted to disk via `vector_db.save_index()` in `search.py` to avoid re-parsing on subsequent runs.
