# Tasks and Fixtures

The registry is `llm_modelbench/tasks.py`. Each `Task` declares an ID,
category, family, scorer, prompt, level, difficulty, and scorer metadata.

## Families, categories, and levels

Families are `text`, `vision`, `embedding`, `tools`, and `insert`. Vision-capable completion
models remain text-eligible; embed-only models are scheduled only for the
embedding lane. Live Ollama capabilities override name heuristics when they
are available; heuristics are the offline/old-server fallback.

Categories cover coding, text operations, knowledge-base work, git, file
operations, agentic actions, technical writing, long context, OCR, PDF, and
retrieval. Levels are cumulative: `smoke`, `short`, then `full`.

`difficulty` weights a task within its category. A difficulty of `0.0` is an
explicit smoke gate: the task remains useful for checking basic behavior but
does not provide free composite-quality points.

## Adding tasks

Add a `Task(...)` entry with an honest level and difficulty. Deterministic
tasks need a scorer-compatible `meta` dictionary. Subjective tasks set
`judge=True` and provide a rubric. Add deterministic regression tests for
every scorer or fixture change.

## Text, coding, and agentic tasks

Text and coding tasks use deterministic contracts wherever possible. Agentic tasks include deterministic JSON decisions and a native structured tool-call task; the benchmark validates but never invokes the proposed tool. The `insert` family uses suffix-conditioned FIM generation and is routed only to models with declared, configured, or functionally probed support.

## Vision, OCR, and PDF

OCR/PDF tasks are scheduled only to vision-capable models. Existing synthetic
tasks render their reference text. A task with `meta["image_path"]` instead
loads a local PNG, JPEG, or WebP path relative to the repository root.

Exact code-style OCR fixtures use a punctuation-preserving normalizer: spacing
around `-`, `/`, and `.` may vary, but prose, substrings, and punctuation-free
answers do not pass. This preserves the output-only task contract.

## Retrieval and embeddings

Retrieval tasks define `meta["docs"]` and labelled `(query, target_doc_id)`
pairs. The model under test is the embedding model; every real retrieval row
records `embed_model=<model under test>`. The aggregate retrieval score remains
recall@1 plus MRR reporting.

Instrumented retrieval rows include `retrieval_cases` without document or full
query text. Each case records:

- `query_index`, `target_doc_id`, `top1_doc_id`, and `top3_doc_ids`
- `target_rank` and `target_similarity`
- `nearest_distractor_doc_id` and `nearest_distractor_similarity`
- `margin`, `pass_at_1`, and `embed_model`

Report rebuilding writes `retrieval_diagnostics.json` and `.md`. Older runs
without persisted case ranks are labelled partial rather than reconstructed.

## Privacy-safe fixtures

Public fixtures must be synthetic and independent of any real person's
circumstances: invented entities and scenarios, not real-world-derived
content, even anonymized. Retrieval fixtures should prefer fictional internal
company policies and services. See [PRIVACY_FIXTURES.md](PRIVACY_FIXTURES.md)
for the full policy.
