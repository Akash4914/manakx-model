# MANAKX Model — AI Indian Standards Assistant

The AI/ML core of MANAKX (SIH26108): given a product description, a
detailed spec, or a full tender, it extracts structured attributes,
retrieves and ranks applicable Indian Standards by semantic similarity,
explains why each one matches, surfaces allied standards, flags
outdated standard numbers, and recommends the applicable BIS
certification scheme. This is the exact architecture described in the
"How the solution works" spec — implemented stage for stage.

## Architecture

```
Input (description / spec / tender PDF)
        |
        v
ingestion/text_extractor.py       Text extraction & cleaning, language detection
        |
        v
extraction/attribute_extractor.py  NLP attribute extraction (product, application,
        |                          environment, protection, power, material,
        |                          performance, electrical/mechanical, safety, testing)
        v
embeddings/backend.py              Embedding model (text -> vector)
        |
        v
retrieval/vector_index.py          Vector database search (FAISS)
retrieval/semantic_search.py       -> Top 10 relevant standards
        |
        v
ranking/llm_ranker.py              LLM/RAG layer: ranks + explains
        |                          ("1. IS XXXX - Highly Relevant. Reason: ...")
        v
graph/knowledge_graph.py           Allied standards (testing/safety/terminology/
        |                          installation/material) for the main match
        v
revision/revision_checker.py       Outdated-standard check + current version
        |
        v
certification/certification_rules.py  BIS certification requirement (rules, not LLM)
        |
        v
pipeline.py                        Orchestrates everything into one JSON result

tender/tender_analyzer.py          Second mode: audits an existing tender
                                    (referenced standards, outdated flags, gaps)
```

Every stage is its own module behind a narrow interface, so any one
piece can be swapped without touching the rest — e.g. upgrading the
embedding model only means changing `embeddings/backend.py`.

## Quick start

```bash
pip install -r requirements.txt
python example_usage.py
```

This runs all three modes using the spec document's own examples.

## Mode 1 — Analyze a product description / spec

```python
from manakx_model import ManakxPipeline

pipeline = ManakxPipeline()  # build once, reuse across requests
result = pipeline.analyze("Outdoor LED street light, 120W, IP66, highway installation")
```

Reproduces the spec's exact example:

```json
"detected_attributes": {
  "product": "LED Luminaire",
  "application": "Street/Highway Lighting",
  "environment": "Outdoor",
  "protection": "IP66",
  "power": "120W"
},
"recommended_standards": [
  {"id": "IS-10322", "relevance": "Highly Relevant",
   "reason": "Matches on: street light.", "match_confidence_pct": 39.7, ...}
],
"allied_standards": [
  {"relation": "testing", "id": "IS-10322-P4", ...},
  {"relation": "safety", "id": "IS-3043", ...},
  {"relation": "installation", "id": "IS-1944:2023", ...}
],
"certification": {"applicable_standard": "IS-10322", "bis_certification": "Mandatory",
                   "scheme": "Compulsory Registration Scheme (CRS)"},
"latest_revision": {"is_current": true, "message": "IS-10322 (2019) is the current version."}
```

Also accepts a tender PDF directly: `pipeline.analyze(pdf_path="tender.pdf")`.

## Mode 2 — Tender document audit

```python
audit = pipeline.analyze_tender(
    "Tender for supply of Centrifugal Water Pump for municipal water supply. "
    "The pump shall conform to IS 9137:2010."
)
```

Reproduces the spec's exact example:

```json
{
  "detected_product": "Centrifugal Water Pump",
  "referenced_standards": [
    {"matched_standard_id": "IS-9137", "is_current": false,
     "message": "IS-9137 (2010) is outdated. Superseded by IS-9137:2025."}
  ],
  "issues": ["IS-9137 (2010) is outdated. Superseded by IS-9137:2025."],
  "missing_standards": [
    {"id": "IS-9137-P3", "why": "Allied testing standard not referenced.", ...},
    {"id": "IS-9922", "why": "Allied safety standard not referenced.", ...}
  ],
  "recommended_current_standards": ["IS-9137:2025"]
}
```

## Mode 3 — Direct revision check

```python
pipeline.check_revision("IS 9137:2010")
# {"is_current": false, "message": "IS-9137 (2010) is outdated. Superseded by IS-9137:2025.", ...}
```

## Running it as a REST API

```bash
uvicorn manakx_model.api.server:app --reload --port 8000
```

| Endpoint | Purpose |
|---|---|
| `POST /analyze` | `{"text": "..."}` → Mode 1 |
| `POST /analyze-tender` | `{"text": "..."}` → Mode 2 |
| `GET /revision/{reference}` | e.g. `/revision/IS 9137:2010` → Mode 3 |
| `GET /health` | status + which backends are active |

## Multilingual input

Input language is auto-detected (`langdetect`). If an LLM is
configured (see below), non-English input is translated to English
before running the rest of the pipeline, matching the spec's
Hindi/English/regional-language flow. Without an LLM configured, the
language is still detected and reported, but translation is skipped
(flagged via `input.translation_applied: false`) — set up the LLM
ranker to enable it.

## Upgrading from the offline defaults to the "real" spec'd stack

Everything below runs out of the box with no setup (`pip install -r
requirements.txt` only) so the whole pipeline is demoable offline with
zero API cost. Each upgrade below is a **drop-in swap** — nothing else
in the pipeline needs to change.

| Stage | Default (offline, always works) | Spec'd upgrade | How to enable |
|---|---|---|---|
| Embeddings | TF-IDF (`embeddings/backend.py: TfidfBackend`) | Sentence-Transformers embeddings | `pip install sentence-transformers` — auto-detected and preferred if importable |
| Vector DB | NumPy brute-force cosine search | FAISS | `pip install faiss-cpu` — **already installed and used by default** in this build |
| Ranking/explanation | Keyword-overlap rule-based ranker | Claude LLM ranks + explains | `pip install anthropic` and set `ANTHROPIC_API_KEY` — auto-detected and preferred |
| Translation | Skipped (flagged) | LLM translation | Same `ANTHROPIC_API_KEY` as above enables this too |

The TF-IDF fallback is lexical, not semantic — it will still miss a
case like "protective headgear for construction workers" matching
"industrial safety helmets" if those exact words don't overlap (the
bundled dataset seeds such synonyms into `keywords` by hand to cover
this in the demo). Installing `sentence-transformers` removes that
limitation, per the spec's own warning against a "simple keyword-search
system" — it just needs internet access to download model weights the
first time it runs.

## Swapping in the real standards dataset

Replace `manakx_model/data/standards_dataset.json` (same schema) or
point `ManakxPipeline(standards_dataset_path="path/to/real.json")` at
a different file. Schema per record:

```json
{
  "id": "IS-10322", "title": "...", "category": "...", "scope": "...",
  "keywords": ["...", "..."],
  "year": 2019, "status": "current", "superseded_by": null,
  "relations": {"testing": ["IS-10322-P4"], "safety": ["IS-3043"],
                "terminology": [], "installation": ["IS-1944:2023"], "material": []},
  "certification": {"bis_mandatory": true, "scheme": "Compulsory Registration Scheme (CRS)"}
}
```

The bundled dataset (18 records) covers LED luminaires, water storage
tanks, safety helmets, centrifugal pumps (including a deliberately
outdated 2010 edition + its 2025 revision, for demoing revision
checking) and school furniture — enough to exercise every feature end
to end.

## Design choices worth knowing (for your report/PPT)

- **Certification is rules-based, not LLM-generated** — read straight
  from the dataset's `certification` field, per the spec's explicit
  caution against letting an LLM invent a compliance requirement.
- **The knowledge graph auto-resolves superseded allied standards** to
  their current version, so a relation recorded against an old edition
  still surfaces the right one without manual dataset upkeep.
- **Every offline fallback is a deliberate, documented design
  decision**, not a missing feature: the pipeline is fully runnable
  and demoable with zero setup, and every fallback is a one-line swap
  to the "real" spec'd component (see the upgrade table above).
